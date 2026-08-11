from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mmprism.data.csl_news_annotation import (
    OUTPUT_SCHEMA_VERSION,
    CslNewsAnnotationConfig,
    annotation_artifact_stem_matches_sample_id,
)

IDENTITY_AUDIT_SCHEMA_VERSION = (
    "mmprism.csl_news_pose_annotation_identity_audit.v1"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _is_stable(before: Mapping[str, int], after: Mapping[str, int]) -> bool:
    return all(before.get(key) == after.get(key) for key in before)


def _stream_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            bytes_read += len(chunk)
    return digest.hexdigest(), bytes_read


def _frozen_sidecar_digest(relative_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        encoded = relative_path.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _audit_annotation_pair(
    output_root: Path,
    sidecar_relative_path: Path,
    *,
    expected_config_fingerprint: str,
) -> tuple[dict[str, Any] | None, int]:
    sidecar_path = output_root / sidecar_relative_path
    artifact_path = sidecar_path.with_suffix(".npz")
    artifact_relative_path = artifact_path.relative_to(output_root)
    failures: list[str] = []
    payload: Mapping[str, Any] | None = None
    sidecar_sha256: str | None = None
    sidecar_before: dict[str, int] | None = None
    sidecar_after: dict[str, int] | None = None
    artifact_before: dict[str, int] | None = None
    artifact_after: dict[str, int] | None = None
    observed_sha256: str | None = None
    bytes_hashed = 0

    try:
        sidecar_before = _stat_identity(sidecar_path)
        if not stat.S_ISREG(sidecar_before["mode"]):
            failures.append("sidecar_not_regular_file")
        sidecar_bytes = sidecar_path.read_bytes()
        sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
        sidecar_after = _stat_identity(sidecar_path)
        if not _is_stable(sidecar_before, sidecar_after):
            failures.append("sidecar_changed_during_audit")
        try:
            decoded: object = json.loads(sidecar_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            failures.append("invalid_sidecar_json")
        else:
            if not isinstance(decoded, Mapping):
                failures.append("sidecar_not_object")
            else:
                payload = decoded
    except OSError:
        failures.append("sidecar_unreadable_or_missing")

    declared_size: int | None = None
    declared_sha256: str | None = None
    declared_path: str | None = None
    sample_id: str | None = None
    if payload is not None:
        raw_sample_id = payload.get("sample_id")
        sample_id = raw_sample_id if isinstance(raw_sample_id, str) else None
        if payload.get("schema_version") != OUTPUT_SCHEMA_VERSION:
            failures.append("sidecar_schema_mismatch")
        if payload.get("status") != "completed":
            failures.append("sidecar_status_not_completed")
        if payload.get("config_fingerprint") != expected_config_fingerprint:
            failures.append("config_fingerprint_mismatch")
        if sample_id is None or not annotation_artifact_stem_matches_sample_id(
            sidecar_path.stem, sample_id
        ):
            failures.append("sample_id_path_mismatch")

        source = payload.get("source")
        expected_archive = f"{sidecar_path.parent.name}.zip"
        if not isinstance(source, Mapping) or source.get("archive") != expected_archive:
            failures.append("source_archive_path_mismatch")

        artifact = payload.get("artifact")
        if not isinstance(artifact, Mapping):
            failures.append("missing_artifact_identity")
        else:
            raw_size = artifact.get("size_bytes")
            if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size >= 0:
                declared_size = raw_size
            else:
                failures.append("invalid_declared_artifact_size")
            raw_sha256 = artifact.get("sha256")
            if isinstance(raw_sha256, str) and _SHA256_PATTERN.fullmatch(raw_sha256):
                declared_sha256 = raw_sha256
            else:
                failures.append("invalid_declared_artifact_sha256")
            raw_path = artifact.get("path")
            if isinstance(raw_path, str) and raw_path:
                declared_path = raw_path
                if Path(raw_path).expanduser().resolve() != artifact_path.resolve():
                    failures.append("declared_artifact_path_mismatch")
            else:
                failures.append("invalid_declared_artifact_path")

    try:
        artifact_before = _stat_identity(artifact_path)
        if not stat.S_ISREG(artifact_before["mode"]):
            failures.append("artifact_not_regular_file")
        observed_sha256, bytes_hashed = _stream_sha256(artifact_path)
        artifact_after = _stat_identity(artifact_path)
        if not _is_stable(artifact_before, artifact_after):
            failures.append("artifact_changed_during_audit")
        if bytes_hashed != artifact_before["size_bytes"]:
            failures.append("artifact_stream_size_mismatch")
        if declared_size is not None and artifact_before["size_bytes"] != declared_size:
            failures.append("artifact_size_mismatch")
        if declared_sha256 is not None and observed_sha256 != declared_sha256:
            failures.append("artifact_sha256_mismatch")
    except OSError:
        failures.append("artifact_unreadable_or_missing")

    if not failures:
        return None, bytes_hashed

    report: dict[str, Any] = {
        "sidecar": sidecar_relative_path.as_posix(),
        "artifact": artifact_relative_path.as_posix(),
        "sample_id": sample_id,
        "failures": sorted(set(failures)),
        "declared_artifact": {
            "path": declared_path,
            "size_bytes": declared_size,
            "sha256": declared_sha256,
        },
        "observed_artifact": {
            "size_bytes": (
                artifact_before["size_bytes"] if artifact_before is not None else None
            ),
            "sha256": observed_sha256,
            "stat_before": artifact_before,
            "stat_after": artifact_after,
        },
        "sidecar_identity": {
            "sha256": sidecar_sha256,
            "stat_before": sidecar_before,
            "stat_after": sidecar_after,
        },
    }
    return report, bytes_hashed


def build_csl_news_annotation_identity_audit(
    config: CslNewsAnnotationConfig,
    *,
    runtime_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit every pose artifact published by a sidecar visible at audit start."""

    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    output_root = config.runtime.output_root
    sample_root = output_root / "samples"
    sidecar_paths = sorted(
        path
        for path in sample_root.glob("archive_*/*.json")
        if not path.name.startswith(".")
    )
    frozen_relative_paths = [path.relative_to(output_root) for path in sidecar_paths]
    frozen_digest = _frozen_sidecar_digest(frozen_relative_paths)

    invalid_pairs: list[dict[str, Any]] = []
    bytes_hashed = 0
    for sidecar_relative_path in frozen_relative_paths:
        pair_report, pair_bytes_hashed = _audit_annotation_pair(
            output_root,
            sidecar_relative_path,
            expected_config_fingerprint=config.fingerprint,
        )
        bytes_hashed += pair_bytes_hashed
        if pair_report is not None:
            invalid_pairs.append(pair_report)

    failure_reason_counts: Counter[str] = Counter()
    for pair_report in invalid_pairs:
        failure_reason_counts.update(pair_report["failures"])
    audit_failures = [] if frozen_relative_paths else ["no_visible_sidecars"]
    git = runtime_report.get("git")
    if not isinstance(git, Mapping):
        audit_failures.append("missing_git_provenance")
    else:
        commit = git.get("commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            audit_failures.append("invalid_git_commit")
        if git.get("dirty") is not False:
            audit_failures.append("dirty_git_worktree")
    status = "passed" if not invalid_pairs and not audit_failures else "failed"
    completed_at = datetime.now(UTC)
    return {
        "schema_version": IDENTITY_AUDIT_SCHEMA_VERSION,
        "status": status,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "runtime": dict(runtime_report),
        "scope": {
            "definition": "all_non_hidden_sidecars_visible_at_audit_start",
            "output_root": str(output_root),
            "config_fingerprint": config.fingerprint,
            "frozen_sidecar_count": len(frozen_relative_paths),
            "frozen_sidecar_list_sha256": frozen_digest,
            "first_sidecar": (
                frozen_relative_paths[0].as_posix() if frozen_relative_paths else None
            ),
            "last_sidecar": (
                frozen_relative_paths[-1].as_posix() if frozen_relative_paths else None
            ),
        },
        "summary": {
            "audited_pair_count": len(frozen_relative_paths),
            "passed_pair_count": len(frozen_relative_paths) - len(invalid_pairs),
            "failed_pair_count": len(invalid_pairs),
            "artifact_bytes_hashed": bytes_hashed,
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        },
        "audit_failures": audit_failures,
        "invalid_pairs": invalid_pairs,
    }


def write_csl_news_annotation_identity_audit(
    report: Mapping[str, Any], output_path: str | Path
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path
