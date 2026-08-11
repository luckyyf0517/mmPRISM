from __future__ import annotations

import json
import os
import zipfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from mmprism.data.csl_news_annotation import (
    CslNewsAnnotationConfig,
    discover_complete_archives,
    sha256_file,
    validate_annotation_output,
)

STATUS_SCHEMA_VERSION = "mmprism.csl_news_pose_annotation_status.v1"


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _count_archive_videos(archive_path: Path) -> int:
    with zipfile.ZipFile(archive_path, "r") as archive:
        return sum(
            not member.is_dir()
            and PurePosixPath(member.filename.replace("\\", "/")).suffix.lower()
            == ".mp4"
            for member in archive.infolist()
        )


def _latest_run(output_root: Path) -> tuple[Path | None, datetime | None]:
    candidates = list((output_root / "runs").glob("run_*.json"))
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    payload = _load_json(latest)
    return latest, _parse_timestamp(payload.get("started_at")) if payload else None


def _validate_sample(sidecar_path: Path, expected_fingerprint: str) -> dict[str, Any]:
    payload = _load_json(sidecar_path)
    npz_path = sidecar_path.with_suffix(".npz")
    result: dict[str, Any] = {
        "sidecar": str(sidecar_path),
        "artifact": str(npz_path),
        "passed": False,
        "failures": [],
    }
    failures: list[str] = result["failures"]
    if payload is None:
        failures.append("invalid sidecar JSON")
        return result
    if payload.get("status") != "completed":
        failures.append("sidecar status is not completed")
    if payload.get("config_fingerprint") != expected_fingerprint:
        failures.append("config fingerprint mismatch")
    annotation = payload.get("annotation")
    if not isinstance(annotation, Mapping) or not isinstance(annotation.get("text"), str):
        failures.append("missing annotation text")
    elif not annotation["text"].strip():
        failures.append("empty annotation text")
    if not validate_annotation_output(npz_path):
        failures.append("artifact contract validation failed")
    artifact = payload.get("artifact")
    expected_sha256 = artifact.get("sha256") if isinstance(artifact, Mapping) else None
    if not isinstance(expected_sha256, str) or not npz_path.is_file():
        failures.append("missing artifact checksum")
    elif sha256_file(npz_path) != expected_sha256:
        failures.append("artifact checksum mismatch")
    result.update(
        {
            "sample_id": payload.get("sample_id"),
            "generated_at": payload.get("generated_at"),
            "passed": not failures,
        }
    )
    return result


def build_csl_news_annotation_status(
    config: CslNewsAnnotationConfig,
    *,
    sample_validate_count: int = 3,
    recent_window: int = 200,
) -> dict[str, Any]:
    """Build a read-only operational report for a running annotation build."""

    if sample_validate_count < 0:
        raise ValueError("sample_validate_count must be non-negative")
    if recent_window < 2:
        raise ValueError("recent_window must be at least 2")

    output_root = config.runtime.output_root
    scratch_root = config.runtime.scratch_root
    archives = discover_complete_archives(config)
    archive_video_counts: dict[str, int] = {}
    archive_errors: dict[str, str] = {}
    for archive_path in archives:
        try:
            archive_video_counts[archive_path.name] = _count_archive_videos(archive_path)
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            archive_errors[archive_path.name] = str(error)

    sample_root = output_root / "samples"
    npz_paths = [
        path
        for path in sample_root.glob("archive_*/*.npz")
        if not path.name.startswith(".")
    ]
    sidecar_paths = [
        path
        for path in sample_root.glob("archive_*/*.json")
        if not path.name.startswith(".")
    ]
    npz_identities = {path.relative_to(sample_root).with_suffix("") for path in npz_paths}
    sidecar_identities = {
        path.relative_to(sample_root).with_suffix("") for path in sidecar_paths
    }
    missing_sidecars = sorted(str(path) for path in npz_identities - sidecar_identities)
    missing_artifacts = sorted(str(path) for path in sidecar_identities - npz_identities)

    latest_run_path, latest_run_started_at = _latest_run(output_root)
    latest_sidecars = sorted(
        sidecar_paths, key=lambda path: path.stat().st_mtime_ns, reverse=True
    )
    recent_payloads: list[dict[str, Any]] = []
    for sidecar_path in latest_sidecars:
        payload = _load_json(sidecar_path)
        if payload is None:
            continue
        generated_at = _parse_timestamp(payload.get("generated_at"))
        if (
            latest_run_started_at is not None
            and generated_at is not None
            and generated_at < latest_run_started_at
        ):
            continue
        recent_payloads.append(payload)
        if len(recent_payloads) >= recent_window:
            break

    recent_timestamps = [
        timestamp
        for payload in recent_payloads
        if (timestamp := _parse_timestamp(payload.get("generated_at"))) is not None
    ]
    wall_seconds = (
        (max(recent_timestamps) - min(recent_timestamps)).total_seconds()
        if len(recent_timestamps) >= 2
        else 0.0
    )
    samples_per_hour = (
        (len(recent_timestamps) - 1) * 3600.0 / wall_seconds if wall_seconds > 0 else None
    )
    recent_frames = 0
    recent_elapsed_seconds = 0.0
    for payload in recent_payloads:
        video = payload.get("video")
        elapsed = payload.get("elapsed_seconds")
        if isinstance(video, Mapping):
            frames = video.get("decoded_frame_count")
            if isinstance(frames, int):
                recent_frames += frames
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            recent_elapsed_seconds += float(elapsed)
    frames_per_second = (
        recent_frames / recent_elapsed_seconds if recent_elapsed_seconds > 0 else None
    )

    failure_paths = list((output_root / "failures").glob("archive_*/*/attempt_*.json"))
    failures_since_latest_run = [
        path
        for path in failure_paths
        if latest_run_started_at is not None
        and datetime.fromtimestamp(path.stat().st_mtime, UTC) >= latest_run_started_at
    ]
    recent_failure_types: Counter[str] = Counter()
    for failure_path in sorted(
        failure_paths, key=lambda path: path.stat().st_mtime_ns, reverse=True
    )[:100]:
        payload = _load_json(failure_path)
        if payload is not None and isinstance(payload.get("error_type"), str):
            recent_failure_types[payload["error_type"]] += 1

    marker_payloads = [
        payload
        for path in (output_root / "archives").glob("archive_*.json")
        if (payload := _load_json(path)) is not None
        and payload.get("config_fingerprint") == config.fingerprint
    ]
    marker_statuses = Counter(
        str(payload.get("status", "unknown")) for payload in marker_payloads
    )
    available_video_count = sum(archive_video_counts.values())
    completed_sample_count = len(npz_identities & sidecar_identities)
    remaining_available_samples = max(0, available_video_count - completed_sample_count)
    eta_hours = (
        remaining_available_samples / samples_per_hour
        if samples_per_hour is not None and samples_per_hour > 0
        else None
    )

    validation = [
        _validate_sample(path, config.fingerprint)
        for path in latest_sidecars[:sample_validate_count]
    ]
    validations_passed = all(item["passed"] for item in validation)
    healthy = bool(
        not archive_errors
        and not missing_sidecars
        and not missing_artifacts
        and not failures_since_latest_run
        and validations_passed
        and (completed_sample_count > 0 or available_video_count == 0)
    )

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "healthy" if healthy else "attention_required",
        "config_fingerprint": config.fingerprint,
        "worker": {
            "worker_index": config.runtime.worker_index,
            "worker_count": config.runtime.worker_count,
        },
        "source": {
            "source_id": config.source.source_id,
            "source_revision": config.source.source_revision,
            "expected_archive_count": config.source.expected_archive_count,
            "complete_archive_count": len(archives),
            "readable_archive_count": len(archive_video_counts),
            "available_video_count": available_video_count,
            "archive_video_counts": archive_video_counts,
            "archive_errors": archive_errors,
        },
        "annotation": {
            "completed_sample_count": completed_sample_count,
            "npz_count": len(npz_paths),
            "sidecar_count": len(sidecar_paths),
            "missing_sidecar_count": len(missing_sidecars),
            "missing_sidecar_examples": missing_sidecars[:20],
            "missing_artifact_count": len(missing_artifacts),
            "missing_artifact_examples": missing_artifacts[:20],
            "remaining_available_sample_count": remaining_available_samples,
            "available_completion_ratio": (
                completed_sample_count / available_video_count
                if available_video_count
                else None
            ),
            "archive_marker_statuses": dict(sorted(marker_statuses.items())),
        },
        "failures": {
            "attempt_count": len(failure_paths),
            "unique_sample_count": len({path.parent.name for path in failure_paths}),
            "since_latest_run_count": len(failures_since_latest_run),
            "recent_error_types": dict(sorted(recent_failure_types.items())),
        },
        "throughput": {
            "recent_sample_count": len(recent_payloads),
            "recent_wall_seconds": wall_seconds,
            "samples_per_hour": samples_per_hour,
            "recent_frame_count": recent_frames,
            "recent_elapsed_seconds": recent_elapsed_seconds,
            "frames_per_second": frames_per_second,
            "available_eta_hours": eta_hours,
            "latest_success_at": (
                max(recent_timestamps).isoformat() if recent_timestamps else None
            ),
        },
        "runtime": {
            "latest_run": str(latest_run_path) if latest_run_path else None,
            "latest_run_started_at": (
                latest_run_started_at.isoformat() if latest_run_started_at else None
            ),
            "output_root": str(output_root),
            "scratch_root": str(scratch_root),
            "output_free_bytes": (
                os.statvfs(output_root).f_bavail * os.statvfs(output_root).f_frsize
                if output_root.exists()
                else None
            ),
        },
        "sample_validation": {
            "requested": sample_validate_count,
            "checked": len(validation),
            "passed": sum(item["passed"] for item in validation),
            "all_passed": validations_passed,
            "samples": validation,
        },
    }


def write_csl_news_annotation_status(
    report: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Write an annotation status report atomically."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
