"""Durable, pauseable sequence scheduler for the CSL-Daily annotation v2 build."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mmprism.data.csl_daily_pose_annotation import (
    ANNOTATION_V2_SCHEMA_VERSION,
    CslDailyPoseAnnotationConfig,
    MMPoseRtmw3dFrameEstimator,
    _finished_sidecar_status,
    _pose_paths,
    _rewrite_manifests,
    discover_sequences,
    run_csl_daily_pose_annotation,
)

SCHEDULER_SCHEMA_VERSION = "mmprism.csl_daily_annotation_scheduler.v1"
DEFAULT_LEASE_SECONDS = 1800


class CslDailySchedulerError(RuntimeError):
    """Raised when the durable CSL-Daily queue cannot operate safely."""


@dataclass(frozen=True)
class ScheduledSequenceLease:
    sequence_id: str
    token: str
    worker_id: str
    lease_path: Path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslDailySchedulerError(f"unable to read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CslDailySchedulerError(f"{description} must be a JSON object: {path}")
    return payload


def scheduler_root(config: CslDailyPoseAnnotationConfig) -> Path:
    return config.runtime.output_root / "scheduler"


def _control_path(config: CslDailyPoseAnnotationConfig) -> Path:
    return scheduler_root(config) / "control.json"


def _lock_path(config: CslDailyPoseAnnotationConfig) -> Path:
    return scheduler_root(config) / "claim.lock"


def _leases_root(config: CslDailyPoseAnnotationConfig) -> Path:
    return scheduler_root(config) / "leases"


def _quarantine_path(config: CslDailyPoseAnnotationConfig, sequence_id: str) -> Path:
    return config.runtime.output_root / "quarantine" / f"{sequence_id}.json"


def _identity(config: CslDailyPoseAnnotationConfig) -> dict[str, str | None]:
    return {
        "annotation_schema": config.schema_version,
        "annotation_config_fingerprint": config.fingerprint,
        "source_id": config.source.source_id,
        "source_receipt_sha256": config.source.receipt_sha256,
    }


def _require_v2(config: CslDailyPoseAnnotationConfig) -> None:
    if config.schema_version != ANNOTATION_V2_SCHEMA_VERSION:
        raise CslDailySchedulerError("CSL-Daily scheduler only accepts annotation v2")


@contextlib.contextmanager
def _claim_lock(config: CslDailyPoseAnnotationConfig) -> Iterator[None]:
    root = scheduler_root(config)
    root.mkdir(parents=True, exist_ok=True)
    with _lock_path(config).open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_control(control: Mapping[str, Any], config: CslDailyPoseAnnotationConfig) -> None:
    if control.get("schema_version") != SCHEDULER_SCHEMA_VERSION:
        raise CslDailySchedulerError("scheduler control schema_version mismatch")
    if control.get("identity") != _identity(config):
        raise CslDailySchedulerError("scheduler control does not match v2 source/config identity")
    if control.get("state") not in {"running", "paused"}:
        raise CslDailySchedulerError("scheduler control state must be running or paused")
    lease_seconds = control.get("lease_seconds")
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or lease_seconds < 60:
        raise CslDailySchedulerError("scheduler lease_seconds must be an integer >= 60")


def initialize_csl_daily_scheduler(
    config: CslDailyPoseAnnotationConfig, *, lease_seconds: int = DEFAULT_LEASE_SECONDS
) -> dict[str, Any]:
    """Initialize the source-bound v2 queue in a deliberately paused state."""

    _require_v2(config)
    if lease_seconds < 60:
        raise CslDailySchedulerError("lease_seconds must be at least 60")
    with _claim_lock(config):
        path = _control_path(config)
        if path.exists():
            control = _load_json(path, "scheduler control")
            _validate_control(control, config)
            return {"status": "already_initialized", "path": str(path), "control": control}
        control = {
            "schema_version": SCHEDULER_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "identity": _identity(config),
            "state": "paused",
            "lease_seconds": lease_seconds,
            "last_transition": {"state": "paused", "reason": "initialization"},
        }
        _atomic_write_json(control, path)
    return {"status": "initialized", "path": str(path), "control": control}


def set_csl_daily_scheduler_state(
    config: CslDailyPoseAnnotationConfig, *, state: str, reason: str | None = None
) -> dict[str, Any]:
    """Atomically request a cooperative queue pause or resume."""

    _require_v2(config)
    if state not in {"running", "paused"}:
        raise CslDailySchedulerError("state must be running or paused")
    with _claim_lock(config):
        path = _control_path(config)
        control = _load_json(path, "scheduler control")
        _validate_control(control, config)
        updated = dict(control)
        updated["state"] = state
        updated["updated_at"] = _utc_now()
        updated["last_transition"] = {
            "state": state,
            "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None,
            "at": updated["updated_at"],
        }
        _atomic_write_json(updated, path)
    return {"status": state, "path": str(path), "control": updated}


def _read_control(config: CslDailyPoseAnnotationConfig) -> dict[str, Any]:
    control = _load_json(_control_path(config), "scheduler control")
    _validate_control(control, config)
    return control


def _lease_path(config: CslDailyPoseAnnotationConfig, sequence_id: str) -> Path:
    return _leases_root(config) / f"{sequence_id}.json"


def _lease_is_stale(lease: Mapping[str, Any], lease_seconds: int, now: float) -> bool:
    heartbeat = lease.get("heartbeat_unix_seconds")
    return (
        isinstance(heartbeat, bool)
        or not isinstance(heartbeat, (int, float))
        or now - heartbeat > lease_seconds
    )


def _recover_stale_leases(
    config: CslDailyPoseAnnotationConfig, *, now: float, lease_seconds: int
) -> int:
    recovered = 0
    for path in sorted(_leases_root(config).glob("*.json")):
        lease = _load_json(path, "scheduler lease")
        if _lease_is_stale(lease, lease_seconds, now):
            target = _leases_root(config) / "expired" / f"{path.stem}--{uuid.uuid4().hex}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            recovered += 1
    return recovered


def _is_finished(config: CslDailyPoseAnnotationConfig, sequence_id: str) -> bool:
    npy_path, sidecar_path = _pose_paths(config, sequence_id)
    return _finished_sidecar_status(npy_path, sidecar_path, config) is not None


def _is_quarantined(config: CslDailyPoseAnnotationConfig, sequence_id: str) -> bool:
    path = _quarantine_path(config, sequence_id)
    if not path.is_file():
        return False
    payload = _load_json(path, "annotation quarantine record")
    return (
        payload.get("schema_version") == "mmprism.csl_daily_annotation_quarantine.v2"
        and payload.get("sequence_id") == sequence_id
        and payload.get("config_fingerprint") == config.fingerprint
    )


def _write_quarantine(
    config: CslDailyPoseAnnotationConfig,
    *,
    sequence_id: str,
    error: BaseException,
) -> Path:
    path = _quarantine_path(config, sequence_id)
    if path.exists():
        if _is_quarantined(config, sequence_id):
            return path
        raise CslDailySchedulerError(f"quarantine path conflicts with another identity: {path}")
    _atomic_write_json(
        {
            "schema_version": "mmprism.csl_daily_annotation_quarantine.v2",
            "generated_at": _utc_now(),
            "sequence_id": sequence_id,
            "config_fingerprint": config.fingerprint,
            "source_receipt_sha256": config.source.receipt_sha256,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        path,
    )
    return path


def claim_csl_daily_annotation_sequence(
    config: CslDailyPoseAnnotationConfig, *, worker_id: str
) -> tuple[ScheduledSequenceLease | None, dict[str, Any]]:
    """Claim one incomplete sequence under a durable exclusive lease."""

    _require_v2(config)
    if not worker_id.strip():
        raise CslDailySchedulerError("worker_id must not be empty")
    with _claim_lock(config):
        control = _read_control(config)
        now = time.time()
        recovered = _recover_stale_leases(
            config, now=now, lease_seconds=control["lease_seconds"]
        )
        if control["state"] == "paused":
            return None, {"state": "paused", "stale_leases_recovered": recovered}
        for sequence in discover_sequences(config.source.sequence_root):
            if _is_finished(config, sequence.name) or _is_quarantined(config, sequence.name):
                continue
            lease_path = _lease_path(config, sequence.name)
            if lease_path.exists():
                continue
            token = uuid.uuid4().hex
            _atomic_write_json(
                {
                    "schema_version": SCHEDULER_SCHEMA_VERSION,
                    "sequence_id": sequence.name,
                    "worker_id": worker_id,
                    "token": token,
                    "claimed_at": _utc_now(),
                    "heartbeat_at": _utc_now(),
                    "heartbeat_unix_seconds": now,
                },
                lease_path,
            )
            return (
                ScheduledSequenceLease(sequence.name, token, worker_id, lease_path),
                {"state": "claimed", "stale_leases_recovered": recovered},
            )
    return None, {"state": "idle", "stale_leases_recovered": recovered}


def release_csl_daily_annotation_lease(
    config: CslDailyPoseAnnotationConfig,
    lease: ScheduledSequenceLease,
    *,
    result: Mapping[str, Any],
) -> None:
    """Retain immutable completion history before returning sequence capacity."""

    with _claim_lock(config):
        if not lease.lease_path.is_file():
            return
        payload = _load_json(lease.lease_path, "scheduler lease")
        if payload.get("token") != lease.token or payload.get("worker_id") != lease.worker_id:
            raise CslDailySchedulerError("scheduler lease ownership changed before release")
        history = dict(payload)
        history["released_at"] = _utc_now()
        history["result"] = dict(result)
        _atomic_write_json(
            history,
            _leases_root(config) / "history" / f"{lease.sequence_id}--{lease.token}.json",
        )
        lease.lease_path.unlink()


def _renew(config: CslDailyPoseAnnotationConfig, lease: ScheduledSequenceLease) -> bool:
    with _claim_lock(config):
        control = _read_control(config)
        payload = _load_json(lease.lease_path, "scheduler lease")
        if payload.get("token") != lease.token or payload.get("worker_id") != lease.worker_id:
            raise CslDailySchedulerError("scheduler lease ownership changed")
        updated = dict(payload)
        updated["heartbeat_at"] = _utc_now()
        updated["heartbeat_unix_seconds"] = time.time()
        _atomic_write_json(updated, lease.lease_path)
        return control.get("state") == "running"


def default_scheduler_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def run_csl_daily_annotation_scheduled_worker(
    config: CslDailyPoseAnnotationConfig,
    *,
    worker_id: str | None = None,
    once: bool = False,
    poll_seconds: int = 15,
) -> dict[str, Any]:
    """Run an elastic v2 worker; manifests are finalized by a separate command."""

    _require_v2(config)
    if poll_seconds < 1:
        raise CslDailySchedulerError("poll_seconds must be positive")
    effective_worker = default_scheduler_worker_id() if worker_id is None else worker_id
    estimator: MMPoseRtmw3dFrameEstimator | None = None
    totals = {"completed": 0, "skipped_qc": 0, "failed": 0, "sequences": 0}
    while True:
        lease, claim = claim_csl_daily_annotation_sequence(config, worker_id=effective_worker)
        if lease is None:
            if claim["state"] == "paused":
                return {"status": "paused", "worker_id": effective_worker, **totals}
            if once:
                return {"status": "idle", "worker_id": effective_worker, **totals}
            time.sleep(poll_seconds)
            continue
        if estimator is None:
            try:
                estimator = MMPoseRtmw3dFrameEstimator(config)
            except Exception as error:
                release_csl_daily_annotation_lease(
                    config,
                    lease,
                    result={
                        "status": "worker_initialization_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise
        try:
            result = run_csl_daily_pose_annotation(
                config,
                estimator=estimator,
                sequence_ids=[lease.sequence_id],
                rewrite_manifests=False,
            )
            if int(result.get("failed", 0)):
                run_record = result.get("run_record")
                quarantine = _write_quarantine(
                    config,
                    sequence_id=lease.sequence_id,
                    error=CslDailySchedulerError(
                        f"per-sequence annotation failed; run_record={run_record}"
                    ),
                )
                result = {
                    **result,
                    "status": "quarantined_failure",
                    "quarantine_path": str(quarantine),
                }
            elif not _renew(config, lease):
                result = {**result, "status": "paused_after_sequence"}
        except Exception as error:
            quarantine = _write_quarantine(
                config, sequence_id=lease.sequence_id, error=error
            )
            result = {
                "status": "quarantined_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "quarantine_path": str(quarantine),
            }
            release_csl_daily_annotation_lease(config, lease, result=result)
        release_csl_daily_annotation_lease(config, lease, result=result)
        totals["completed"] += max(
            0, int(result.get("processed", 0)) - int(result.get("skipped_qc", 0))
        )
        totals["skipped_qc"] += int(result.get("skipped_qc", 0))
        totals["failed"] += int(result.get("failed", 0))
        totals["sequences"] += 1
        if result.get("status") == "paused_after_sequence":
            return {"status": "paused", "worker_id": effective_worker, **totals}
        if once:
            return {"status": "one_sequence_completed", "worker_id": effective_worker, **totals}


def build_csl_daily_scheduler_status(config: CslDailyPoseAnnotationConfig) -> dict[str, Any]:
    """Report queue state and derived coverage without mutating the scheduler."""

    _require_v2(config)
    with _claim_lock(config):
        control = _read_control(config)
        now = time.time()
        leases = []
        for path in sorted(_leases_root(config).glob("*.json")):
            payload = _load_json(path, "scheduler lease")
            leases.append(
                {
                    "sequence_id": payload.get("sequence_id"),
                    "worker_id": payload.get("worker_id"),
                    "stale": _lease_is_stale(payload, control["lease_seconds"], now),
                }
            )
        sequences = discover_sequences(config.source.sequence_root)
        complete = sum(_is_finished(config, path.name) for path in sequences)
    return {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "control": control,
        "leases": {"active": leases, "stale_count": sum(item["stale"] for item in leases)},
        "coverage": {"source_sequences": len(sequences), "finished_sequences": complete},
    }


def finalize_csl_daily_annotation_v2(config: CslDailyPoseAnnotationConfig) -> dict[str, Any]:
    """Generate manifests only while paused and lease-free; never claim work."""

    _require_v2(config)
    with _claim_lock(config):
        control = _read_control(config)
        active = list(_leases_root(config).glob("*.json"))
        if control["state"] != "paused" or active:
            raise CslDailySchedulerError("finalize requires paused scheduler with no active leases")
        sequences = discover_sequences(config.source.sequence_root)
        outcomes = []
        for sequence in sequences:
            npy_path, sidecar_path = _pose_paths(config, sequence.name)
            status = _finished_sidecar_status(npy_path, sidecar_path, config)
            if status is not None:
                outcomes.append(status)
            elif _is_quarantined(config, sequence.name):
                outcomes.append("quarantined")
            else:
                outcomes.append("unfinished")
        source_count = len(sequences)
        coverage = {
            "source_sequences": source_count,
            "completed_eligible": outcomes.count("completed"),
            "skipped_qc": outcomes.count("skipped"),
            "quarantined": outcomes.count("quarantined"),
            "unfinished_sequences": outcomes.count("unfinished"),
        }
        coverage_path = config.runtime.output_root / "coverage.json"
        _atomic_write_json(
            {
                "schema_version": "mmprism.csl_daily_annotation_coverage.v2",
                "generated_at": _utc_now(),
                "config_fingerprint": config.fingerprint,
                "source_receipt_sha256": config.source.receipt_sha256,
                "coverage": coverage,
                "pose_manifest": None,
                "pose_qc": None,
                "quarantine_root": str(config.runtime.output_root / "quarantine"),
            },
            coverage_path,
        )
        if coverage["unfinished_sequences"]:
            raise CslDailySchedulerError(
                "finalize refuses partial annotation: "
                f"{coverage['unfinished_sequences']} source sequences have no terminal record"
            )
        manifest, qc, rows = _rewrite_manifests(config)
        final_payload = _load_json(coverage_path, "annotation coverage")
        final_payload["pose_manifest"] = {"path": str(manifest), "completed_rows": rows}
        final_payload["pose_qc"] = {"path": str(qc)}
        _atomic_write_json(final_payload, coverage_path)
    return {"status": "finalized", **coverage, "pose_manifest": str(manifest), "pose_qc": str(qc)}


__all__ = [
    "CslDailySchedulerError",
    "ScheduledSequenceLease",
    "build_csl_daily_scheduler_status",
    "claim_csl_daily_annotation_sequence",
    "default_scheduler_worker_id",
    "finalize_csl_daily_annotation_v2",
    "initialize_csl_daily_scheduler",
    "release_csl_daily_annotation_lease",
    "run_csl_daily_annotation_scheduled_worker",
    "set_csl_daily_scheduler_state",
]
