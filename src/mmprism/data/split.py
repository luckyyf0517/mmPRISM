from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mmprism.config import expand_environment
from mmprism.contracts import (
    SPLIT_ASSIGNMENT_SCHEMA,
    SampleRecord,
    SplitAssignment,
    validate_manifest,
    validate_split_assignments,
)

SPLIT_CONFIG_SCHEMA = "mmprism.split_config.v1"
SPLIT_SNAPSHOT_SCHEMA = "mmprism.split_snapshot.v1"
SPLIT_ALGORITHM = "sha256_mod_weight_v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
SPLIT_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")


class DataSplitError(RuntimeError):
    """Raised when a canonical split snapshot cannot be built."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataSplitError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise DataSplitError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataSplitError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _integer(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: int
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataSplitError(
            f"{location}.{key} must be an integer >= {minimum}"
        )
    return value


def _relative_path(payload: Mapping[str, Any], key: str, location: str) -> Path:
    value = Path(_text(payload, key, location))
    if value.is_absolute() or ".." in value.parts:
        raise DataSplitError(f"{location}.{key} must be relative to source.data_root")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitAllocation:
    name: str
    weight: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight}


@dataclass(frozen=True)
class DataSplitConfig:
    data_root: Path
    manifest_relative: Path
    expected_manifest_sha256: str
    expected_dataset: str
    expected_record_count: int
    source_scope: str
    group_selector: str
    group_namespace: str
    protocol_id: str
    seed: int
    allocations: tuple[SplitAllocation, ...]
    minimum_groups_per_split: int
    minimum_free_bytes: int
    snapshot_root_relative: Path

    @property
    def manifest_path(self) -> Path:
        return (self.data_root / self.manifest_relative).resolve()

    @property
    def snapshot_root(self) -> Path:
        return (self.data_root / self.snapshot_root_relative).resolve()

    def portable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SPLIT_CONFIG_SCHEMA,
            "source": {
                "manifest": self.manifest_relative.as_posix(),
                "expected_manifest_sha256": self.expected_manifest_sha256,
                "expected_dataset": self.expected_dataset,
                "expected_record_count": self.expected_record_count,
                "scope": self.source_scope,
            },
            "grouping": {
                "selector": self.group_selector,
                "namespace": self.group_namespace,
            },
            "assignment": {
                "algorithm": SPLIT_ALGORITHM,
                "protocol_id": self.protocol_id,
                "seed": self.seed,
                "splits": [allocation.to_dict() for allocation in self.allocations],
            },
            "validation": {
                "minimum_groups_per_split": self.minimum_groups_per_split,
                "minimum_free_bytes": self.minimum_free_bytes,
            },
            "output": {"snapshot_root": self.snapshot_root_relative.as_posix()},
        }

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.portable_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(serialized).hexdigest()


def _load_allocations(value: object) -> tuple[SplitAllocation, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise DataSplitError("assignment.splits must contain at least two entries")
    allocations: list[SplitAllocation] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        location = f"assignment.splits[{index}]"
        payload = _mapping(item, location)
        _reject_unknown(payload, {"name", "weight"}, location)
        name = _text(payload, "name", location)
        if not SPLIT_NAME_PATTERN.fullmatch(name):
            raise DataSplitError(f"{location}.name has an invalid split name")
        if name in names:
            raise DataSplitError(f"duplicate split name: {name}")
        names.add(name)
        allocations.append(
            SplitAllocation(
                name=name,
                weight=_integer(payload, "weight", location, minimum=1),
            )
        )
    return tuple(allocations)


def load_data_split_config(path: str | Path) -> DataSplitConfig:
    """Load a strict, portable deterministic split configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DataSplitError(f"Unable to load split config: {error}") from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise DataSplitError(str(error)) from error
    root = _mapping(expanded, "root")
    _reject_unknown(
        root,
        {"schema_version", "source", "grouping", "assignment", "validation", "output"},
        "root",
    )
    if root.get("schema_version") != SPLIT_CONFIG_SCHEMA:
        raise DataSplitError(f"schema_version must be {SPLIT_CONFIG_SCHEMA}")
    source = _mapping(root.get("source"), "source")
    _reject_unknown(
        source,
        {
            "data_root",
            "manifest",
            "expected_manifest_sha256",
            "expected_dataset",
            "expected_record_count",
            "scope",
        },
        "source",
    )
    grouping = _mapping(root.get("grouping"), "grouping")
    _reject_unknown(grouping, {"selector", "namespace"}, "grouping")
    assignment = _mapping(root.get("assignment"), "assignment")
    _reject_unknown(
        assignment, {"algorithm", "protocol_id", "seed", "splits"}, "assignment"
    )
    validation = _mapping(root.get("validation"), "validation")
    _reject_unknown(
        validation,
        {"minimum_groups_per_split", "minimum_free_bytes"},
        "validation",
    )
    output = _mapping(root.get("output"), "output")
    _reject_unknown(output, {"snapshot_root"}, "output")

    expected_hash = _text(source, "expected_manifest_sha256", "source")
    if not SHA256_PATTERN.fullmatch(expected_hash):
        raise DataSplitError(
            "source.expected_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    scope = _text(source, "scope", "source")
    if scope not in {"partial", "complete"}:
        raise DataSplitError("source.scope must be partial or complete")
    selector = _text(grouping, "selector", "grouping")
    valid_selector = selector in {"sample_id", "sequence_id", "subject_id"} or (
        selector.startswith("group_keys.") and len(selector) > len("group_keys.")
    )
    if not valid_selector:
        raise DataSplitError(
            "grouping.selector must be sample_id, sequence_id, subject_id, "
            "or group_keys.<name>"
        )
    if _text(assignment, "algorithm", "assignment") != SPLIT_ALGORITHM:
        raise DataSplitError(f"assignment.algorithm must be {SPLIT_ALGORITHM}")

    return DataSplitConfig(
        data_root=Path(_text(source, "data_root", "source")).expanduser().resolve(),
        manifest_relative=_relative_path(source, "manifest", "source"),
        expected_manifest_sha256=expected_hash,
        expected_dataset=_text(source, "expected_dataset", "source"),
        expected_record_count=_integer(
            source, "expected_record_count", "source", minimum=1
        ),
        source_scope=scope,
        group_selector=selector,
        group_namespace=_text(grouping, "namespace", "grouping"),
        protocol_id=_text(assignment, "protocol_id", "assignment"),
        seed=_integer(assignment, "seed", "assignment", minimum=0),
        allocations=_load_allocations(assignment.get("splits")),
        minimum_groups_per_split=_integer(
            validation, "minimum_groups_per_split", "validation", minimum=1
        ),
        minimum_free_bytes=_integer(
            validation, "minimum_free_bytes", "validation", minimum=0
        ),
        snapshot_root_relative=_relative_path(output, "snapshot_root", "output"),
    )


def _runtime_git_commit(runtime_report: Mapping[str, Any]) -> str:
    git = runtime_report.get("git")
    if not isinstance(git, Mapping):
        raise DataSplitError("runtime report has no Git provenance")
    commit = git.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DataSplitError("runtime report has no valid Git commit")
    if git.get("dirty") is not False:
        raise DataSplitError("formal split snapshots require a clean Git worktree")
    return commit


def _read_source_records(config: DataSplitConfig) -> list[SampleRecord]:
    path = config.manifest_path
    if not path.is_file():
        raise DataSplitError(f"source manifest does not exist: {path}")
    stat_before = path.stat()
    digest = _sha256_file(path)
    if digest != config.expected_manifest_sha256:
        raise DataSplitError(
            f"source manifest SHA-256 mismatch: expected "
            f"{config.expected_manifest_sha256}, found {digest}"
        )
    manifest_summary = validate_manifest(path)
    if manifest_summary.record_count != config.expected_record_count:
        raise DataSplitError(
            "source manifest record count mismatch: "
            f"expected {config.expected_record_count}, found "
            f"{manifest_summary.record_count}"
        )
    if manifest_summary.datasets != (config.expected_dataset,):
        raise DataSplitError(
            f"source manifest dataset mismatch: {manifest_summary.datasets}"
        )
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            records.append(
                SampleRecord.from_mapping(
                    json.loads(raw_line), f"source line {line_number}"
                )
            )
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise DataSplitError("source manifest changed while building split")
    return records


def _group_value(record: SampleRecord, selector: str) -> str:
    value: str | None
    if selector == "sample_id":
        value = record.sample_id
    elif selector == "sequence_id":
        value = record.sequence_id
    elif selector == "subject_id":
        value = record.subject_id
    else:
        key = selector.removeprefix("group_keys.")
        value = record.group_keys.get(key) if record.group_keys is not None else None
    if not isinstance(value, str) or not value.strip():
        raise DataSplitError(
            f"sample {record.sample_id} has no non-empty grouping value for {selector}"
        )
    return value.strip()


def _stable_group_id(config: DataSplitConfig, record: SampleRecord) -> str:
    value = _group_value(record, config.group_selector)
    identity = "\0".join(
        (
            config.group_namespace,
            config.expected_dataset,
            config.group_selector,
            value,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _assign_split(config: DataSplitConfig, group_id: str) -> str:
    identity = f"{config.protocol_id}\0{config.seed}\0{group_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(identity).digest(), "big") % sum(
        allocation.weight for allocation in config.allocations
    )
    cumulative = 0
    for allocation in config.allocations:
        cumulative += allocation.weight
        if bucket < cumulative:
            return allocation.name
    raise AssertionError("split allocation weights did not cover hash bucket")


def build_data_split_snapshot(
    config: DataSplitConfig,
    *,
    runtime_report: Mapping[str, Any],
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Build an atomic deterministic group-disjoint split snapshot."""

    git_commit = _runtime_git_commit(runtime_report)
    records = _read_source_records(config)
    config.snapshot_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(config.snapshot_root).free
    if free_bytes < config.minimum_free_bytes:
        raise DataSplitError(
            f"split root has {free_bytes} free bytes, below minimum "
            f"{config.minimum_free_bytes}"
        )
    generated_at = datetime.now(UTC)
    identifier = snapshot_id or generated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise DataSplitError("snapshot_id contains unsafe characters")
    destination = config.snapshot_root / f"snapshot_{identifier}"
    temporary = config.snapshot_root / f".snapshot_{identifier}.tmp.{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise DataSplitError(f"split snapshot destination already exists: {destination}")
    temporary.mkdir()
    try:
        assignments = [
            SplitAssignment(
                sample_id=record.sample_id,
                group_id=(group_id := _stable_group_id(config, record)),
                split=_assign_split(config, group_id),
            )
            for record in records
        ]
        assignments.sort(key=lambda item: item.sample_id)
        assignment_path = temporary / "assignments.jsonl"
        with assignment_path.open("w", encoding="utf-8") as stream:
            for assignment in assignments:
                stream.write(
                    json.dumps(
                        assignment.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        allowed_splits = {allocation.name for allocation in config.allocations}
        validation = validate_split_assignments(
            assignment_path,
            expected_sample_ids={record.sample_id for record in records},
            allowed_splits=allowed_splits,
        )
        missing_splits = sorted(allowed_splits - set(validation.splits))
        too_small = {
            name: validation.group_counts.get(name, 0)
            for name in allowed_splits
            if validation.group_counts.get(name, 0)
            < config.minimum_groups_per_split
        }
        if missing_splits or too_small:
            raise DataSplitError(
                "split allocation does not satisfy minimum group coverage: "
                f"missing={missing_splits}, group_counts={too_small}"
            )

        config_path = temporary / "config.json"
        config_path.write_text(
            json.dumps(config.portable_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assignment_sha256 = _sha256_file(assignment_path)
        summary = {
            "schema_version": SPLIT_SNAPSHOT_SCHEMA,
            "generated_at": generated_at.isoformat(),
            "status": config.source_scope,
            "config_fingerprint": config.fingerprint,
            "builder": {
                "git_commit": git_commit,
                "git_state": "clean",
                "runtime": dict(runtime_report),
            },
            "source": {
                "manifest": config.manifest_relative.as_posix(),
                "manifest_sha256": config.expected_manifest_sha256,
                "dataset": config.expected_dataset,
                "record_count": len(records),
                "scope": config.source_scope,
            },
            "assignment": {
                "schema_version": SPLIT_ASSIGNMENT_SCHEMA,
                "path": "assignments.jsonl",
                "sha256": assignment_sha256,
                "algorithm": SPLIT_ALGORITHM,
                "protocol_id": config.protocol_id,
                "seed": config.seed,
                "group_selector": config.group_selector,
                "group_namespace": config.group_namespace,
                "allocations": [
                    allocation.to_dict() for allocation in config.allocations
                ],
                "assignment_count": validation.assignment_count,
                "group_count": validation.group_count,
                "sample_counts": validation.sample_counts,
                "group_counts": validation.group_counts,
            },
            "audit": {
                "sample_coverage_missing_count": 0,
                "sample_coverage_extra_count": 0,
                "duplicate_sample_count": 0,
                "cross_split_group_leakage_count": 0,
                "minimum_groups_per_split": config.minimum_groups_per_split,
            },
            "storage": {"free_bytes_before_snapshot": free_bytes},
        }
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksums = {
            "assignments.jsonl": assignment_sha256,
            "config.json": _sha256_file(config_path),
            "summary.json": _sha256_file(summary_path),
        }
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
            ),
            encoding="ascii",
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema_version": SPLIT_SNAPSHOT_SCHEMA,
        "status": config.source_scope,
        "snapshot_dir": str(destination),
        "assignments_path": str(destination / "assignments.jsonl"),
        "summary_path": str(destination / "summary.json"),
        "assignment_count": validation.assignment_count,
        "group_count": validation.group_count,
        "sample_counts": validation.sample_counts,
        "group_counts": validation.group_counts,
        "assignments_sha256": assignment_sha256,
    }
