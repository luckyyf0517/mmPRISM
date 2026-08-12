from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mmprism.data.csl_news_annotation import (
    CslNewsAnnotationConfig,
    is_completed_annotation_archive,
    run_csl_news_annotation,
)
from mmprism.data.csl_news_integrity import (
    CslNewsIntegrityArchive,
    CslNewsIntegrityError,
    load_csl_news_integrity_registry_snapshot,
    passed_csl_news_integrity_archives,
)

SCHEDULER_SCHEMA_VERSION = "mmprism.csl_news_annotation_scheduler.v1"
CONTROL_FILENAME = "control.json"
DEFAULT_LEASE_SECONDS = 900


class CslNewsSchedulerError(RuntimeError):
    """Raised when an annotation scheduler control-plane operation is unsafe."""


@dataclass(frozen=True)
class ScheduledArchiveLease:
    archive_id: int
    archive_name: str
    archive_path: Path
    token: str
    worker_id: str
    lease_path: Path
    registry_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslNewsSchedulerError(f"Unable to read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CslNewsSchedulerError(f"{description} must be a JSON object: {path}")
    return payload


def scheduler_root(config: CslNewsAnnotationConfig) -> Path:
    return config.runtime.output_root / "scheduler"


def _control_path(config: CslNewsAnnotationConfig) -> Path:
    return scheduler_root(config) / CONTROL_FILENAME


def _lock_path(config: CslNewsAnnotationConfig) -> Path:
    return scheduler_root(config) / "claim.lock"


def _leases_root(config: CslNewsAnnotationConfig) -> Path:
    return scheduler_root(config) / "leases"


def _identity(config: CslNewsAnnotationConfig) -> dict[str, str]:
    return {
        "source_id": config.source.source_id,
        "source_revision": config.source.source_revision,
        "annotation_config_fingerprint": config.fingerprint,
    }


def _validate_control(control: Mapping[str, Any], config: CslNewsAnnotationConfig) -> None:
    if control.get("schema_version") != SCHEDULER_SCHEMA_VERSION:
        raise CslNewsSchedulerError("scheduler control schema_version mismatch")
    if control.get("identity") != _identity(config):
        raise CslNewsSchedulerError(
            "scheduler control does not match annotation source/config identity"
        )
    if control.get("state") not in {"running", "paused"}:
        raise CslNewsSchedulerError("scheduler control state must be running or paused")
    lease_seconds = control.get("lease_seconds")
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or lease_seconds < 60:
        raise CslNewsSchedulerError("scheduler control lease_seconds must be an integer >= 60")


@contextlib.contextmanager
def _claim_lock(config: CslNewsAnnotationConfig) -> Iterator[None]:
    root = scheduler_root(config)
    root.mkdir(parents=True, exist_ok=True)
    with _lock_path(config).open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def initialize_csl_news_scheduler(
    config: CslNewsAnnotationConfig,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> dict[str, Any]:
    """Create the scheduler control plane in the deliberately safe paused state."""

    if lease_seconds < 60:
        raise CslNewsSchedulerError("lease_seconds must be at least 60")
    with _claim_lock(config):
        control_path = _control_path(config)
        if control_path.exists():
            control = _load_json(control_path, "scheduler control")
            _validate_control(control, config)
            return {"status": "already_initialized", "control": control, "path": str(control_path)}
        control = {
            "schema_version": SCHEDULER_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "identity": _identity(config),
            "state": "paused",
            "lease_seconds": lease_seconds,
            "last_transition": {"state": "paused", "reason": "initialization"},
        }
        _atomic_write_json(control, control_path)
    return {"status": "initialized", "control": control, "path": str(control_path)}


def set_csl_news_scheduler_state(
    config: CslNewsAnnotationConfig, *, state: str, reason: str | None = None
) -> dict[str, Any]:
    """Atomically request cooperative pause or resume for all elastic workers."""

    if state not in {"running", "paused"}:
        raise CslNewsSchedulerError("state must be running or paused")
    with _claim_lock(config):
        control_path = _control_path(config)
        if not control_path.is_file():
            raise CslNewsSchedulerError("scheduler is not initialized")
        control = _load_json(control_path, "scheduler control")
        _validate_control(control, config)
        changed_at = _utc_now()
        updated = dict(control)
        updated["state"] = state
        updated["updated_at"] = changed_at
        updated["last_transition"] = {
            "state": state,
            "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None,
            "at": changed_at,
        }
        _atomic_write_json(updated, control_path)
    return {"status": state, "control": updated, "path": str(control_path)}


def _read_control(config: CslNewsAnnotationConfig) -> dict[str, Any]:
    control_path = _control_path(config)
    if not control_path.is_file():
        raise CslNewsSchedulerError("scheduler is not initialized")
    control = _load_json(control_path, "scheduler control")
    _validate_control(control, config)
    return control


def _archive_integrity(
    entry: CslNewsIntegrityArchive, labels_sha256: str
) -> dict[str, str]:
    return {"archive_sha256": entry.sha256, "labels_sha256": labels_sha256}


def _archive_marker_path(
    config: CslNewsAnnotationConfig,
    entry: CslNewsIntegrityArchive,
    integrity: Mapping[str, str],
) -> Path:
    canonical = config.runtime.output_root / "archives" / f"archive_{entry.archive_id:03d}.json"
    if canonical.exists() and not is_completed_annotation_archive(
        canonical,
        config.fingerprint,
        entry.size_bytes,
        integrity,
    ):
        return (
            config.runtime.output_root
            / "archives"
            / f"archive_{entry.archive_id:03d}--source_{entry.sha256}.json"
        )
    return canonical


def _eligible_archives(
    config: CslNewsAnnotationConfig, registry_path: Path
) -> tuple[list[tuple[CslNewsIntegrityArchive, Path]], str]:
    try:
        registry, registry_sha256 = load_csl_news_integrity_registry_snapshot(
            registry_path,
            source_id=config.source.source_id,
            source_revision=config.source.source_revision,
        )
        passed = passed_csl_news_integrity_archives(registry)
    except CslNewsIntegrityError as error:
        raise CslNewsSchedulerError(str(error)) from error
    source = registry.get("source")
    labels_sha256 = source.get("labels_sha256") if isinstance(source, Mapping) else None
    if not isinstance(labels_sha256, str) or len(labels_sha256) != 64:
        raise CslNewsSchedulerError("integrity registry has no valid labels_sha256")

    eligible: list[tuple[CslNewsIntegrityArchive, Path]] = []
    for entry in passed.values():
        archive_path = (config.source.archive_root / entry.archive_path_relative).resolve()
        try:
            archive_path.relative_to(config.source.archive_root)
        except ValueError as error:
            raise CslNewsSchedulerError(
                f"integrity-passed archive escapes archive root: {entry.archive_name}"
            ) from error
        if not archive_path.is_file():
            continue
        stat = archive_path.stat()
        if stat.st_size != entry.size_bytes or stat.st_mtime_ns != entry.mtime_ns:
            continue
        integrity = _archive_integrity(entry, labels_sha256)
        marker = _archive_marker_path(config, entry, integrity)
        if not is_completed_annotation_archive(
            marker, config.fingerprint, entry.size_bytes, integrity
        ):
            eligible.append((entry, archive_path))
    ordered = sorted(eligible, key=lambda item: (-item[0].video_count, item[0].archive_id))
    return ordered, registry_sha256


def _lease_path(config: CslNewsAnnotationConfig, archive_id: int) -> Path:
    return _leases_root(config) / f"archive_{archive_id:03d}.json"


def _lease_is_stale(lease: Mapping[str, Any], lease_seconds: int, now: float) -> bool:
    heartbeat = lease.get("heartbeat_unix_seconds")
    if isinstance(heartbeat, bool) or not isinstance(heartbeat, (int, float)):
        return True
    return now - heartbeat > lease_seconds


def _recover_stale_leases(config: CslNewsAnnotationConfig, lease_seconds: int, now: float) -> int:
    recovered = 0
    active_root = _leases_root(config)
    expired_root = active_root / "expired"
    if not active_root.exists():
        return 0
    for lease_path in sorted(active_root.glob("archive_*.json")):
        lease = _load_json(lease_path, "scheduler lease")
        if not _lease_is_stale(lease, lease_seconds, now):
            continue
        expired_path = (
            expired_root
            / f"{lease_path.stem}--expired_{int(now)}_{uuid.uuid4().hex}.json"
        )
        expired_path.parent.mkdir(parents=True, exist_ok=True)
        lease_path.replace(expired_path)
        recovered += 1
    return recovered


def claim_csl_news_annotation_archive(
    config: CslNewsAnnotationConfig,
    *,
    integrity_registry_path: str | Path,
    worker_id: str,
) -> tuple[ScheduledArchiveLease | None, dict[str, Any]]:
    """Claim one source-valid, incomplete archive using a durable exclusive lease."""

    registry_path = Path(integrity_registry_path).expanduser().resolve()
    if not worker_id.strip():
        raise CslNewsSchedulerError("worker_id must not be empty")
    with _claim_lock(config):
        control = _read_control(config)
        now = time.time()
        recovered = _recover_stale_leases(config, control["lease_seconds"], now)
        if control["state"] == "paused":
            return None, {"state": "paused", "stale_leases_recovered": recovered}
        candidates, registry_sha256 = _eligible_archives(config, registry_path)
        for entry, archive_path in candidates:
            lease_path = _lease_path(config, entry.archive_id)
            if lease_path.exists():
                continue
            token = uuid.uuid4().hex
            lease = {
                "schema_version": SCHEDULER_SCHEMA_VERSION,
                "archive": {
                    "archive_id": entry.archive_id,
                    "archive_name": entry.archive_name,
                    "archive_path_relative": entry.archive_path_relative.as_posix(),
                    "archive_sha256": entry.sha256,
                },
                "worker_id": worker_id,
                "token": token,
                "claimed_at": _utc_now(),
                "heartbeat_at": _utc_now(),
                "heartbeat_unix_seconds": now,
                "registry_sha256": registry_sha256,
            }
            _atomic_write_json(lease, lease_path)
            return (
                ScheduledArchiveLease(
                    archive_id=entry.archive_id,
                    archive_name=entry.archive_name,
                    archive_path=archive_path,
                    token=token,
                    worker_id=worker_id,
                    lease_path=lease_path,
                    registry_sha256=registry_sha256,
                ),
                {
                    "state": "claimed",
                    "stale_leases_recovered": recovered,
                    "candidate_count": len(candidates),
                },
            )
    return None, {
        "state": "idle",
        "stale_leases_recovered": recovered,
        "candidate_count": len(candidates),
    }


def renew_csl_news_annotation_lease(
    config: CslNewsAnnotationConfig, lease: ScheduledArchiveLease
) -> bool:
    """Renew a lease and report whether the worker may start another sample."""

    with _claim_lock(config):
        control = _read_control(config)
        if not lease.lease_path.is_file():
            raise CslNewsSchedulerError(f"scheduler lease disappeared: {lease.lease_path}")
        payload = _load_json(lease.lease_path, "scheduler lease")
        if payload.get("token") != lease.token or payload.get("worker_id") != lease.worker_id:
            raise CslNewsSchedulerError("scheduler lease ownership changed")
        now = time.time()
        updated = dict(payload)
        updated["heartbeat_at"] = _utc_now()
        updated["heartbeat_unix_seconds"] = now
        _atomic_write_json(updated, lease.lease_path)
        state = control.get("state")
        return isinstance(state, str) and state == "running"


def release_csl_news_annotation_lease(
    config: CslNewsAnnotationConfig,
    lease: ScheduledArchiveLease,
    *,
    result: Mapping[str, Any],
) -> None:
    """Retain a completed lease as immutable execution history before releasing capacity."""

    with _claim_lock(config):
        if not lease.lease_path.is_file():
            return
        payload = _load_json(lease.lease_path, "scheduler lease")
        if payload.get("token") != lease.token or payload.get("worker_id") != lease.worker_id:
            raise CslNewsSchedulerError("scheduler lease ownership changed before release")
        history_path = (
            _leases_root(config)
            / "history"
            / f"{lease.lease_path.stem}--{lease.token}.json"
        )
        history = dict(payload)
        history["released_at"] = _utc_now()
        history["result"] = dict(result)
        _atomic_write_json(history, history_path)
        lease.lease_path.unlink()


def build_csl_news_scheduler_status(
    config: CslNewsAnnotationConfig,
    *,
    integrity_registry_path: str | Path,
) -> dict[str, Any]:
    """Inspect scheduler state without changing control or lease records."""

    registry_path = Path(integrity_registry_path).expanduser().resolve()
    with _claim_lock(config):
        control = _read_control(config)
        now = time.time()
        active: list[dict[str, Any]] = []
        stale: list[str] = []
        for path in sorted(_leases_root(config).glob("archive_*.json")):
            payload = _load_json(path, "scheduler lease")
            active.append(
                {
                    "archive": payload.get("archive"),
                    "worker_id": payload.get("worker_id"),
                    "heartbeat_at": payload.get("heartbeat_at"),
                    "stale": _lease_is_stale(payload, control["lease_seconds"], now),
                }
            )
            if active[-1]["stale"]:
                stale.append(path.name)
        eligible, registry_sha256 = _eligible_archives(config, registry_path)
    return {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "control": control,
        "registry": {"path": str(registry_path), "sha256": registry_sha256},
        "leases": {"active": active, "stale": stale},
        "queue": {
            "eligible_archive_count": len(eligible),
            "eligible_video_count": sum(entry.video_count for entry, _ in eligible),
        },
    }


def default_scheduler_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def run_csl_news_annotation_scheduled_worker(
    config: CslNewsAnnotationConfig,
    *,
    integrity_registry_path: str | Path,
    worker_id: str | None = None,
    once: bool = False,
    poll_seconds: int | None = None,
) -> dict[str, Any]:
    """Consume registry-passed archives with elastic worker capacity and lease recovery."""

    effective_worker_id = default_scheduler_worker_id() if worker_id is None else worker_id
    effective_poll_seconds = config.runtime.poll_seconds if poll_seconds is None else poll_seconds
    if effective_poll_seconds < 1:
        raise CslNewsSchedulerError("poll_seconds must be positive")
    totals = {"processed": 0, "skipped": 0, "failed": 0, "archives": 0}
    while True:
        lease, claim = claim_csl_news_annotation_archive(
            config,
            integrity_registry_path=integrity_registry_path,
            worker_id=effective_worker_id,
        )
        state = claim["state"]
        if lease is None:
            if state == "paused":
                return {"status": "paused", "worker_id": effective_worker_id, **totals}
            if once:
                return {"status": "idle", "worker_id": effective_worker_id, **totals}
            time.sleep(effective_poll_seconds)
            continue

        result: dict[str, Any]
        try:
            result = run_csl_news_annotation(
                config,
                archive_id=lease.archive_id,
                worker_index=0,
                worker_count=1,
                integrity_registry_path=integrity_registry_path,
                continue_requested=scheduler_continue_callback(config, lease),
                orchestration_metadata={
                    "scheduler": SCHEDULER_SCHEMA_VERSION,
                    "worker_id": effective_worker_id,
                    "lease_token": lease.token,
                    "registry_sha256_at_claim": lease.registry_sha256,
                },
            )
        except BaseException as error:
            result = {
                "status": "worker_error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            release_csl_news_annotation_lease(config, lease, result=result)
            raise
        release_csl_news_annotation_lease(config, lease, result=result)
        for key in ("processed", "skipped", "failed"):
            value = result.get(key)
            if isinstance(value, int):
                totals[key] += value
        totals["archives"] += 1
        if result.get("status") == "paused":
            return {"status": "paused", "worker_id": effective_worker_id, **totals}
        if once:
            return {"status": "one_archive_completed", "worker_id": effective_worker_id, **totals}


def scheduler_continue_callback(
    config: CslNewsAnnotationConfig, lease: ScheduledArchiveLease
) -> Callable[[], bool]:
    """Expose a typed callback for tests and alternate worker harnesses."""

    return lambda: renew_csl_news_annotation_lease(config, lease)
