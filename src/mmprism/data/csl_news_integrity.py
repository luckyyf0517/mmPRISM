from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mmprism.config import expand_environment
from mmprism.data.csl_news import (
    audit_csl_news_archive,
    load_csl_news_audit_context,
    write_csl_news_audit,
)

INTEGRITY_CONFIG_SCHEMA = "mmprism.csl_news_source_integrity_config.v2"
INTEGRITY_REGISTRY_SCHEMA_V1 = "mmprism.csl_news_source_integrity_registry.v1"
INTEGRITY_REGISTRY_SCHEMA = "mmprism.csl_news_source_integrity_registry.v2"
SUPPORTED_INTEGRITY_REGISTRY_SCHEMAS = {
    INTEGRITY_REGISTRY_SCHEMA_V1,
    INTEGRITY_REGISTRY_SCHEMA,
}
ARCHIVE_PATTERN = re.compile(r"^archive_(\d{3})\.zip$")


class CslNewsIntegrityError(RuntimeError):
    """Raised when the source-integrity registry cannot be updated safely."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CslNewsIntegrityError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CslNewsIntegrityError(
            f"Unknown keys in {location}: {', '.join(unknown)}"
        )


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CslNewsIntegrityError(
            f"{location}.{key} must be a non-empty string"
        )
    return value.strip()


def _integer(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: int
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CslNewsIntegrityError(
            f"{location}.{key} must be an integer >= {minimum}"
        )
    return value


def _boolean(payload: Mapping[str, Any], key: str, location: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise CslNewsIntegrityError(f"{location}.{key} must be a boolean")
    return value


def _relative_path(payload: Mapping[str, Any], key: str, location: str) -> Path:
    path = Path(_text(payload, key, location))
    if path.is_absolute() or ".." in path.parts:
        raise CslNewsIntegrityError(
            f"{location}.{key} must be relative to source.data_root"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


@dataclass(frozen=True)
class CslNewsIntegrityConfig:
    data_root: Path
    archive_root_relative: Path
    replacement_archives_relative: Mapping[int, Path]
    labels_path_relative: Path
    registry_path_relative: Path
    audit_root_relative: Path
    scratch_root_relative: Path
    source_id: str
    source_revision: str
    expected_archive_count: int
    verify_crc: bool
    decode_sample_count: int

    @property
    def archive_root(self) -> Path:
        return (self.data_root / self.archive_root_relative).resolve()

    @property
    def labels_path(self) -> Path:
        return (self.data_root / self.labels_path_relative).resolve()

    @property
    def registry_path(self) -> Path:
        return (self.data_root / self.registry_path_relative).resolve()

    @property
    def audit_root(self) -> Path:
        return (self.data_root / self.audit_root_relative).resolve()

    @property
    def scratch_root(self) -> Path:
        return (self.data_root / self.scratch_root_relative).resolve()

    def portable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INTEGRITY_CONFIG_SCHEMA,
            "source": {
                "archive_root": self.archive_root_relative.as_posix(),
                "replacement_archives": {
                    f"{archive_id:03d}": path.as_posix()
                    for archive_id, path in sorted(
                        self.replacement_archives_relative.items()
                    )
                },
                "labels_path": self.labels_path_relative.as_posix(),
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "expected_archive_count": self.expected_archive_count,
            },
            "validation": {
                "verify_crc": self.verify_crc,
                "decode_sample_count": self.decode_sample_count,
            },
            "output": {
                "registry_path": self.registry_path_relative.as_posix(),
                "audit_root": self.audit_root_relative.as_posix(),
                "scratch_root": self.scratch_root_relative.as_posix(),
            },
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.portable_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CslNewsIntegrityArchive:
    archive_id: int
    archive_name: str
    archive_path_relative: Path
    source_kind: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    video_count: int


def load_csl_news_integrity_config(path: str | Path) -> CslNewsIntegrityConfig:
    """Load the strict and portable cumulative integrity configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CslNewsIntegrityError(
            f"Unable to load source-integrity config: {error}"
        ) from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise CslNewsIntegrityError(str(error)) from error

    root = _mapping(expanded, "root")
    _reject_unknown(root, {"schema_version", "source", "validation", "output"}, "root")
    if root.get("schema_version") != INTEGRITY_CONFIG_SCHEMA:
        raise CslNewsIntegrityError(
            f"schema_version must be {INTEGRITY_CONFIG_SCHEMA}"
        )

    source = _mapping(root.get("source"), "source")
    _reject_unknown(
        source,
        {
            "data_root",
            "archive_root",
            "replacement_archives",
            "labels_path",
            "source_id",
            "source_revision",
            "expected_archive_count",
        },
        "source",
    )
    validation = _mapping(root.get("validation"), "validation")
    _reject_unknown(
        validation, {"verify_crc", "decode_sample_count"}, "validation"
    )
    output = _mapping(root.get("output"), "output")
    _reject_unknown(
        output, {"registry_path", "audit_root", "scratch_root"}, "output"
    )

    expected_archive_count = _integer(
        source, "expected_archive_count", "source", minimum=1
    )
    replacement_payload = _mapping(
        source.get("replacement_archives"), "source.replacement_archives"
    )
    replacement_archives: dict[int, Path] = {}
    for raw_archive_id, raw_path in replacement_payload.items():
        if not isinstance(raw_archive_id, str) or not re.fullmatch(
            r"\d{3}", raw_archive_id
        ):
            raise CslNewsIntegrityError(
                "source.replacement_archives keys must be three-digit archive IDs"
            )
        archive_id = int(raw_archive_id)
        if not 1 <= archive_id <= expected_archive_count:
            raise CslNewsIntegrityError(
                f"replacement archive ID is outside configured range: {raw_archive_id}"
            )
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise CslNewsIntegrityError(
                f"source.replacement_archives.{raw_archive_id} must be a relative path"
            )
        relative_path = Path(raw_path.strip())
        expected_name = f"archive_{raw_archive_id}.zip"
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.name != expected_name
            or relative_path == Path(expected_name)
        ):
            raise CslNewsIntegrityError(
                f"source.replacement_archives.{raw_archive_id} must be a nested "
                f"path ending in {expected_name}"
            )
        replacement_archives[archive_id] = relative_path

    return CslNewsIntegrityConfig(
        data_root=Path(_text(source, "data_root", "source")).expanduser().resolve(),
        archive_root_relative=_relative_path(source, "archive_root", "source"),
        replacement_archives_relative=replacement_archives,
        labels_path_relative=_relative_path(source, "labels_path", "source"),
        registry_path_relative=_relative_path(output, "registry_path", "output"),
        audit_root_relative=_relative_path(output, "audit_root", "output"),
        scratch_root_relative=_relative_path(output, "scratch_root", "output"),
        source_id=_text(source, "source_id", "source"),
        source_revision=_text(source, "source_revision", "source"),
        expected_archive_count=expected_archive_count,
        verify_crc=_boolean(validation, "verify_crc", "validation"),
        decode_sample_count=_integer(
            validation, "decode_sample_count", "validation", minimum=0
        ),
    )


def _runtime_git_commit(runtime_report: Mapping[str, Any]) -> str:
    git = runtime_report.get("git")
    if not isinstance(git, Mapping):
        raise CslNewsIntegrityError("runtime report has no Git provenance")
    commit = git.get("commit")
    dirty = git.get("dirty")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CslNewsIntegrityError("runtime report has no valid Git commit")
    if dirty is not False:
        raise CslNewsIntegrityError("source-integrity scan requires a clean Git state")
    return commit


def _archive_id(path: Path) -> int:
    match = ARCHIVE_PATTERN.fullmatch(path.name)
    if match is None:
        raise CslNewsIntegrityError(f"Unexpected archive name: {path.name}")
    return int(match.group(1))


def _discover_archives(config: CslNewsIntegrityConfig) -> dict[int, Path]:
    if not config.archive_root.is_dir():
        raise CslNewsIntegrityError(
            f"Archive root does not exist: {config.archive_root}"
        )
    archives: dict[int, Path] = {}
    for path in config.archive_root.glob("archive_*.zip"):
        if path.is_symlink() or not path.is_file():
            raise CslNewsIntegrityError(
                f"Primary archive candidate must be a regular file: {path}"
            )
        archive_id = _archive_id(path)
        if not 1 <= archive_id <= config.expected_archive_count:
            raise CslNewsIntegrityError(
                f"Archive ID is outside configured range: {path.name}"
            )
        if archive_id in archives:
            raise CslNewsIntegrityError(f"Duplicate archive ID: {archive_id:03d}")
        archives[archive_id] = path.resolve()
    for archive_id, relative_path in config.replacement_archives_relative.items():
        candidate = config.archive_root / relative_path
        if not candidate.exists():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise CslNewsIntegrityError(
                f"Replacement archive candidate must be a regular file: {candidate}"
            )
        resolved = candidate.resolve()
        try:
            resolved.relative_to(config.archive_root)
        except ValueError as error:
            raise CslNewsIntegrityError(
                f"Replacement archive escapes archive_root: {candidate}"
            ) from error
        archives[archive_id] = resolved
    return dict(sorted(archives.items()))


def load_csl_news_integrity_registry(
    path: str | Path,
    *,
    source_id: str | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Load and minimally validate a cumulative integrity registry."""

    registry, _ = load_csl_news_integrity_registry_snapshot(
        path, source_id=source_id, source_revision=source_revision
    )
    return registry


def load_csl_news_integrity_registry_snapshot(
    path: str | Path,
    *,
    source_id: str | None = None,
    source_revision: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load one registry byte snapshot and return its validated payload and hash."""

    registry, digest, _ = read_csl_news_integrity_registry_snapshot(
        path, source_id=source_id, source_revision=source_revision
    )
    return registry, digest


def read_csl_news_integrity_registry_snapshot(
    path: str | Path,
    *,
    source_id: str | None = None,
    source_revision: str | None = None,
) -> tuple[dict[str, Any], str, bytes]:
    """Read and validate one exact registry snapshot, retaining its source bytes."""

    registry_path = Path(path).expanduser().resolve()
    try:
        serialized = registry_path.read_bytes()
        payload: object = json.loads(serialized.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslNewsIntegrityError(
            f"Unable to load source-integrity registry {registry_path}: {error}"
        ) from error
    registry = dict(_mapping(payload, "registry"))
    if registry.get("schema_version") not in SUPPORTED_INTEGRITY_REGISTRY_SCHEMAS:
        raise CslNewsIntegrityError(
            "registry schema_version must be one of: "
            + ", ".join(sorted(SUPPORTED_INTEGRITY_REGISTRY_SCHEMAS))
        )
    source = _mapping(registry.get("source"), "registry.source")
    if source_id is not None and source.get("source_id") != source_id:
        raise CslNewsIntegrityError("integrity registry source_id mismatch")
    if source_revision is not None and source.get("source_revision") != source_revision:
        raise CslNewsIntegrityError("integrity registry source_revision mismatch")
    _mapping(registry.get("archives"), "registry.archives")
    return registry, hashlib.sha256(serialized).hexdigest(), serialized


def passed_csl_news_integrity_archives(
    registry: Mapping[str, Any],
) -> dict[int, CslNewsIntegrityArchive]:
    """Return typed entries that passed a full source audit."""

    schema_version = registry.get("schema_version")
    if schema_version not in SUPPORTED_INTEGRITY_REGISTRY_SCHEMAS:
        raise CslNewsIntegrityError("unsupported integrity registry schema")
    archives = _mapping(registry.get("archives"), "registry.archives")
    passed: dict[int, CslNewsIntegrityArchive] = {}
    for key, raw_entry in archives.items():
        if not isinstance(key, str) or not re.fullmatch(r"\d{3}", key):
            raise CslNewsIntegrityError(f"Invalid archive registry key: {key!r}")
        entry = _mapping(raw_entry, f"registry.archives.{key}")
        if entry.get("status") != "passed" or entry.get("source_present") is not True:
            continue
        size_bytes = entry.get("size_bytes")
        mtime_ns = entry.get("mtime_ns")
        video_count = entry.get("video_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (size_bytes, mtime_ns, video_count)
        ):
            raise CslNewsIntegrityError(f"Invalid numeric fields for archive {key}")
        assert isinstance(size_bytes, int)
        assert isinstance(mtime_ns, int)
        assert isinstance(video_count, int)
        archive_name = entry.get("archive_name")
        sha256 = entry.get("sha256")
        expected_name = f"archive_{key}.zip"
        if archive_name != expected_name:
            raise CslNewsIntegrityError(f"Archive name mismatch for registry entry {key}")
        if schema_version == INTEGRITY_REGISTRY_SCHEMA_V1:
            archive_path_relative = Path(expected_name)
            source_kind = "primary"
        else:
            raw_relative_path = entry.get("archive_path_relative")
            raw_source_kind = entry.get("source_kind")
            if not isinstance(raw_relative_path, str) or not raw_relative_path:
                raise CslNewsIntegrityError(
                    f"Missing archive_path_relative for archive {key}"
                )
            archive_path_relative = Path(raw_relative_path)
            if (
                archive_path_relative.is_absolute()
                or ".." in archive_path_relative.parts
                or archive_path_relative.name != expected_name
            ):
                raise CslNewsIntegrityError(
                    f"Invalid archive_path_relative for archive {key}"
                )
            if raw_source_kind not in {"primary", "replacement"}:
                raise CslNewsIntegrityError(f"Invalid source_kind for archive {key}")
            assert isinstance(raw_source_kind, str)
            source_kind = raw_source_kind
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise CslNewsIntegrityError(f"Invalid SHA-256 for archive {key}")
        archive_id = int(key)
        passed[archive_id] = CslNewsIntegrityArchive(
            archive_id=archive_id,
            archive_name=archive_name,
            archive_path_relative=archive_path_relative,
            source_kind=source_kind,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            sha256=sha256,
            video_count=video_count,
        )
    return dict(sorted(passed.items()))


def _load_existing_registry(config: CslNewsIntegrityConfig) -> dict[str, Any] | None:
    if not config.registry_path.is_file():
        return None
    registry = load_csl_news_integrity_registry(
        config.registry_path,
        source_id=config.source_id,
        source_revision=config.source_revision,
    )
    if registry.get("config_fingerprint") != config.fingerprint:
        raise CslNewsIntegrityError("integrity registry config fingerprint mismatch")
    return registry


def _entry_is_reusable(
    entry: Mapping[str, Any],
    archive_path: Path,
    config: CslNewsIntegrityConfig,
    *,
    labels_unchanged: bool,
) -> bool:
    archive_stat = archive_path.stat()
    archive_path_relative = archive_path.relative_to(config.archive_root).as_posix()
    if (
        not labels_unchanged
        or entry.get("source_present") is not True
        or entry.get("status") not in {"passed", "failed"}
        or entry.get("size_bytes") != archive_stat.st_size
        or entry.get("mtime_ns") != archive_stat.st_mtime_ns
        or entry.get("archive_path_relative") != archive_path_relative
    ):
        return False
    audit = entry.get("audit")
    if not isinstance(audit, Mapping):
        return False
    audit_path_value = audit.get("path")
    audit_sha256 = audit.get("sha256")
    if not isinstance(audit_path_value, str) or not isinstance(audit_sha256, str):
        return False
    audit_path = (config.data_root / audit_path_value).resolve()
    try:
        audit_path.relative_to(config.data_root)
    except ValueError:
        return False
    return audit_path.is_file() and _sha256(audit_path) == audit_sha256


def _registry_payload(
    config: CslNewsIntegrityConfig,
    *,
    git_commit: str,
    labels_sha256: str,
    entries: Mapping[str, Mapping[str, Any]],
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    present = [entry for entry in entries.values() if entry.get("source_present") is True]
    passed = [entry for entry in present if entry.get("status") == "passed"]
    failed = [entry for entry in present if entry.get("status") == "failed"]
    pending = [entry for entry in present if entry.get("status") == "pending"]
    missing = [entry for entry in entries.values() if entry.get("source_present") is False]
    passed_ids = sorted(str(entry["archive_id"]) for entry in passed)
    failed_ids = sorted(str(entry["archive_id"]) for entry in failed)
    missing_ids = sorted(str(entry["archive_id"]) for entry in missing)
    complete = len(passed) == config.expected_archive_count and not failed
    if complete:
        status = "complete_passed"
    elif failed or missing:
        status = "partial_with_failures"
    else:
        status = "partial"
    return {
        "schema_version": INTEGRITY_REGISTRY_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "config_fingerprint": config.fingerprint,
        "builder": {"git_commit": git_commit, "git_state": "clean"},
        "source": {
            "source_id": config.source_id,
            "source_revision": config.source_revision,
            "expected_archive_count": config.expected_archive_count,
            "archive_root": config.archive_root_relative.as_posix(),
            "labels_sha256": labels_sha256,
        },
        "validation": {
            "verify_crc": config.verify_crc,
            "decode_sample_count": config.decode_sample_count,
        },
        "summary": {
            "present_final_count": len(present),
            "passed_count": len(passed),
            "failed_count": len(failed),
            "pending_count": len(pending),
            "missing_count": len(missing),
            "passed_archive_ids": passed_ids,
            "failed_archive_ids": failed_ids,
            "missing_archive_ids": missing_ids,
            "passed_video_count": sum(int(entry.get("video_count", 0)) for entry in passed),
            "selected_replacement_archive_ids": sorted(
                str(entry["archive_id"])
                for entry in present
                if entry.get("source_kind") == "replacement"
            ),
        },
        "last_scan": dict(scan),
        "archives": dict(sorted(entries.items())),
    }


def scan_csl_news_source_integrity(
    config: CslNewsIntegrityConfig,
    *,
    runtime_report: Mapping[str, Any],
    max_new_archives: int | None = None,
    archive_id: int | None = None,
) -> dict[str, Any]:
    """Incrementally audit immutable final ZIPs and atomically update the registry."""

    if max_new_archives is not None and max_new_archives < 1:
        raise CslNewsIntegrityError("max_new_archives must be positive")
    if archive_id is not None and not 1 <= archive_id <= config.expected_archive_count:
        raise CslNewsIntegrityError("archive_id is outside the configured source range")
    git_commit = _runtime_git_commit(runtime_report)
    archives = _discover_archives(config)
    if archive_id is not None and archive_id not in archives:
        raise CslNewsIntegrityError(f"archive_{archive_id:03d}.zip is not final")

    config.registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = config.registry_path.with_suffix(f"{config.registry_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CslNewsIntegrityError("another integrity scan is already running") from error

        existing = _load_existing_registry(config)
        context = load_csl_news_audit_context(config.labels_path)
        labels_unchanged = bool(
            existing is not None
            and isinstance(existing.get("source"), Mapping)
            and existing["source"].get("labels_sha256") == context.labels_sha256
        )
        entries: dict[str, dict[str, Any]] = (
            {
                str(key): dict(_mapping(value, f"registry.archives.{key}"))
                for key, value in _mapping(existing.get("archives"), "registry.archives").items()
            }
            if existing is not None
            else {}
        )
        observed_at = datetime.now(UTC).isoformat()
        present_keys = {f"{item:03d}" for item in archives}
        missing_count = 0
        for key, existing_entry in entries.items():
            if key not in present_keys and existing_entry.get("source_present") is not False:
                existing_entry["previous_status"] = existing_entry.get("status")
                existing_entry["status"] = "missing"
                existing_entry["source_present"] = False
                existing_entry["last_observed_at"] = observed_at
                missing_count += 1

        candidate_ids = [archive_id] if archive_id is not None else list(archives)
        selected_ids: list[int] = []
        reused_count = 0
        for current_id in candidate_ids:
            archive_path = archives[current_id]
            key = f"{current_id:03d}"
            current_entry = entries.get(key)
            if current_entry is not None and _entry_is_reusable(
                current_entry,
                archive_path,
                config,
                labels_unchanged=labels_unchanged,
            ):
                current_entry["last_observed_at"] = observed_at
                reused_count += 1
                continue
            if max_new_archives is None or len(selected_ids) < max_new_archives:
                selected_ids.append(current_id)
            else:
                archive_stat = archive_path.stat()
                entries[key] = {
                    "archive_id": key,
                    "archive_name": archive_path.name,
                    "archive_path_relative": archive_path.relative_to(
                        config.archive_root
                    ).as_posix(),
                    "source_kind": (
                        "replacement"
                        if current_id in config.replacement_archives_relative
                        and archive_path.relative_to(config.archive_root)
                        == config.replacement_archives_relative[current_id]
                        else "primary"
                    ),
                    "status": "pending",
                    "source_present": True,
                    "size_bytes": archive_stat.st_size,
                    "mtime_ns": archive_stat.st_mtime_ns,
                    "last_observed_at": observed_at,
                }

        audit_results: list[dict[str, Any]] = []
        for current_id in selected_ids:
            archive_path = archives[current_id]
            before = archive_path.stat()
            report = audit_csl_news_archive(
                archive_path,
                config.labels_path,
                source_id=f"{config.source_id}@{config.source_revision}",
                verify_crc=config.verify_crc,
                decode_sample_count=config.decode_sample_count,
                scratch_dir=config.scratch_root,
                audit_context=context,
            )
            after = archive_path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise CslNewsIntegrityError(
                    f"Source changed during audit: {archive_path.name}"
                )
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            archive_sha256 = str(report["archive"]["sha256"])
            audit_path = (
                config.audit_root
                / f"archive_{current_id:03d}"
                / f"audit_{timestamp}_{archive_sha256[:12]}.json"
            )
            write_csl_news_audit(report, audit_path)
            audit_sha256 = _sha256(audit_path)
            key = f"{current_id:03d}"
            entry = {
                "archive_id": key,
                "archive_name": archive_path.name,
                "archive_path_relative": archive_path.relative_to(
                    config.archive_root
                ).as_posix(),
                "source_kind": (
                    "replacement"
                    if current_id in config.replacement_archives_relative
                    and archive_path.relative_to(config.archive_root)
                    == config.replacement_archives_relative[current_id]
                    else "primary"
                ),
                "status": report["status"],
                "source_present": True,
                "size_bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "sha256": archive_sha256,
                "video_count": int(report["archive"]["video_count"]),
                "failures": list(report["failures"]),
                "audited_at": report["generated_at"],
                "last_observed_at": observed_at,
                "builder_commit": git_commit,
                "audit": {
                    "path": audit_path.relative_to(config.data_root).as_posix(),
                    "sha256": audit_sha256,
                },
            }
            entries[key] = entry
            audit_results.append(
                {
                    "archive_id": key,
                    "status": report["status"],
                    "audit_path": entry["audit"]["path"],
                    "audit_sha256": audit_sha256,
                }
            )
            scan = {
                "started_at": observed_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "audited_count": len(audit_results),
                "reused_count": reused_count,
                "new_missing_count": missing_count,
                "audit_results": audit_results,
            }
            registry = _registry_payload(
                config,
                git_commit=git_commit,
                labels_sha256=context.labels_sha256,
                entries=entries,
                scan=scan,
            )
            _write_json_atomic(registry, config.registry_path)

        scan = {
            "started_at": observed_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "audited_count": len(audit_results),
            "reused_count": reused_count,
            "new_missing_count": missing_count,
            "audit_results": audit_results,
        }
        registry = _registry_payload(
            config,
            git_commit=git_commit,
            labels_sha256=context.labels_sha256,
            entries=entries,
            scan=scan,
        )
        _write_json_atomic(registry, config.registry_path)
        return registry
