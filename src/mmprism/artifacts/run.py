from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from mmprism.config import load_experiment_config
from mmprism.runtime.plan import RunPlan

RUN_SCHEMA_VERSION = "mmprism.run.v1"
RUN_INPUTS_SCHEMA_VERSION = "mmprism.run-inputs.v1"
METRICS_SCHEMA_VERSION = "mmprism.metrics.v1"
_SAFE_INPUT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SAFE_ARTIFACT_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_ARTIFACT_NAMES = {
    "run.json",
    "config.resolved.json",
    "environment.json",
    "inputs.json",
}


class ArtifactError(ValueError):
    """Raised when a formal run artifact violates its contract."""


class RunInputKind(StrEnum):
    MANIFEST = "manifest"
    SPLIT = "split"
    CHECKPOINT = "checkpoint"
    MODEL = "model"
    CONFIG = "config"
    OTHER = "other"


def _utc_text(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ArtifactError("artifact timestamps must be timezone-aware")
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"artifact payload is not strict JSON: {error}") from error
    return (serialized + "\n").encode("utf-8")


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"artifact payload is not strict JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            record_count = 0
            for index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    raise ArtifactError(f"JSONL artifact record {index} must be a mapping")
                try:
                    line = json.dumps(
                        record,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                except (TypeError, ValueError) as error:
                    raise ArtifactError(
                        f"JSONL artifact record {index} is not strict JSON: {error}"
                    ) from error
                handle.write(line + b"\n")
                record_count += 1
            if record_count == 0:
                raise ArtifactError("JSONL artifacts require at least one record")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RunInput:
    name: str
    kind: RunInputKind
    path: Path
    sha256: str
    size_bytes: int

    @classmethod
    def capture(
        cls,
        *,
        name: str,
        kind: RunInputKind | str,
        path: str | Path,
        expected_sha256: str | None = None,
    ) -> RunInput:
        if not _SAFE_INPUT_NAME.fullmatch(name):
            raise ArtifactError("input name must match [a-z0-9][a-z0-9._-]*")
        try:
            input_kind = kind if isinstance(kind, RunInputKind) else RunInputKind(kind)
        except ValueError as error:
            supported = ", ".join(item.value for item in RunInputKind)
            raise ArtifactError(
                f"unsupported input kind {kind!r}; expected: {supported}"
            ) from error

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ArtifactError(f"run input is not a file: {source}")
        before = source.stat()
        digest = sha256_file(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ArtifactError(f"run input changed while hashing: {source}")

        if expected_sha256 is not None:
            normalized = expected_sha256.lower()
            if not _SHA256.fullmatch(normalized):
                raise ArtifactError("expected input SHA-256 must contain 64 hex characters")
            if digest != normalized:
                raise ArtifactError(
                    f"run input SHA-256 mismatch for {source}: expected {normalized}, got {digest}"
                )

        return cls(
            name=name,
            kind=input_kind,
            path=source,
            sha256=digest,
            size_bytes=after.st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RunArtifactWriter:
    run_dir: Path
    run_id: str

    @classmethod
    def initialize(
        cls,
        plan: RunPlan,
        *,
        source_config: str | Path,
        inputs: Sequence[RunInput] = (),
        command: Sequence[str] = (),
    ) -> RunArtifactWriter:
        run_dir = plan.run_dir.resolve()
        if run_dir.name != plan.run_id:
            raise ArtifactError("run directory name must equal run_id")
        if run_dir.exists():
            raise ArtifactError(f"run directory already exists: {run_dir}")
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ArtifactError("formal runs require a non-empty launch command")
        if not inputs:
            raise ArtifactError("formal runs require registered inputs")
        if not any(item.kind is RunInputKind.MANIFEST for item in inputs):
            raise ArtifactError("formal runs require a manifest input")

        config_path = Path(source_config).expanduser().resolve()
        if not config_path.is_file():
            raise ArtifactError(f"source configuration is not a file: {config_path}")
        if _canonical_json_sha256(plan.resolved_config) != plan.config_sha256:
            raise ArtifactError("run plan config hash does not match resolved configuration")

        project_root_value = plan.runtime_report.get("project_root")
        if not isinstance(project_root_value, str) or not project_root_value:
            raise ArtifactError("run plan has no project root provenance")
        source_config_payload = (
            load_experiment_config(config_path).resolved(Path(project_root_value)).to_dict()
        )
        if _canonical_json_sha256(source_config_payload) != plan.config_sha256:
            raise ArtifactError("source configuration does not match the resolved run plan")

        git = plan.runtime_report.get("git")
        if not isinstance(git, Mapping) or not git.get("commit"):
            raise ArtifactError("formal runs require Git commit provenance")

        input_names = [item.name for item in inputs]
        if len(set(input_names)) != len(input_names):
            raise ArtifactError("run input names must be unique")
        input_payload = {
            "schema_version": RUN_INPUTS_SCHEMA_VERSION,
            "inputs": [item.to_dict() for item in sorted(inputs, key=lambda item: item.name)],
        }

        run_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = run_dir.parent / f".{plan.run_id}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o750)
        try:
            _atomic_write_json(temporary / "config.resolved.json", plan.resolved_config)
            _atomic_write_json(temporary / "environment.json", plan.runtime_report)
            _atomic_write_json(temporary / "inputs.json", input_payload)
            files = {
                name: {
                    "sha256": sha256_file(temporary / name),
                    "size_bytes": (temporary / name).stat().st_size,
                }
                for name in ("config.resolved.json", "environment.json", "inputs.json")
            }
            run_payload: dict[str, Any] = {
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": plan.run_id,
                "status": "initialized",
                "created_at": plan.created_at,
                "completed_at": None,
                "experiment": {
                    "name": plan.resolved_config["name"],
                    "task": plan.resolved_config["task"],
                    "seed": plan.resolved_config["runtime"]["seed"],
                },
                "command": list(command),
                "source_config": {
                    "path": str(config_path),
                    "sha256": sha256_file(config_path),
                    "size_bytes": config_path.stat().st_size,
                },
                "config_sha256": plan.config_sha256,
                "git": dict(git),
                "input_count": len(inputs),
                "expected_artifacts": list(plan.expected_artifacts),
                "artifacts": files,
                "failure": None,
            }
            _atomic_write_json(temporary / "run.json", run_payload)
            try:
                os.replace(temporary, run_dir)
            except OSError as error:
                if run_dir.exists():
                    raise ArtifactError(f"run directory already exists: {run_dir}") from error
                raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return cls(run_dir=run_dir, run_id=plan.run_id)

    def write_metrics(
        self,
        *,
        protocol_id: str,
        split: str,
        values: Mapping[str, int | float],
        sample_count: int,
        created_at: datetime | None = None,
    ) -> Path:
        if not protocol_id.strip():
            raise ArtifactError("metrics protocol_id must be non-empty")
        if not split.strip():
            raise ArtifactError("metrics split must be non-empty")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
            raise ArtifactError("metrics sample_count must be a non-negative integer")
        if not values:
            raise ArtifactError("metrics values must not be empty")

        normalized: dict[str, int | float] = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ArtifactError("metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArtifactError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ArtifactError(f"metric {name!r} must be finite")
            normalized[name] = value

        destination = self.run_dir / "metrics.json"
        if destination.exists():
            raise ArtifactError(f"metrics artifact already exists: {destination}")
        payload = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "protocol_id": protocol_id.strip(),
            "split": split.strip(),
            "sample_count": sample_count,
            "created_at": _utc_text(created_at),
            "values": dict(sorted(normalized.items())),
        }
        _atomic_write_json(destination, payload)
        self._register_artifact("metrics.json")
        return destination

    def artifact_path(self, name: str) -> Path:
        """Return one safe top-level artifact path without creating it."""

        if not _SAFE_ARTIFACT_NAME.fullmatch(name):
            raise ArtifactError("artifact name must be one safe top-level filename")
        if name in _RESERVED_ARTIFACT_NAMES:
            raise ArtifactError(f"artifact name is reserved: {name}")
        return self.run_dir / name

    def write_json_artifact(self, name: str, payload: Mapping[str, Any]) -> Path:
        destination = self.artifact_path(name)
        if destination.exists():
            raise ArtifactError(f"artifact already exists: {destination}")
        _atomic_write_json(destination, payload)
        self._register_artifact(name)
        return destination

    def write_jsonl_artifact(self, name: str, records: Iterable[Mapping[str, Any]]) -> Path:
        destination = self.artifact_path(name)
        if destination.exists():
            raise ArtifactError(f"artifact already exists: {destination}")
        _atomic_write_jsonl(destination, records)
        self._register_artifact(name)
        return destination

    def register_artifact(self, name: str) -> Path:
        """Register an atomically completed artifact created by a domain writer."""

        destination = self.artifact_path(name)
        if not destination.is_file():
            raise ArtifactError(f"artifact does not exist: {destination}")
        self._register_artifact(name)
        return destination

    def finalize(
        self,
        *,
        status: str,
        failure: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        if status not in {"completed", "failed", "aborted"}:
            raise ArtifactError("final run status must be completed, failed, or aborted")
        run_path = self.run_dir / "run.json"
        payload = self._read_run()
        if payload.get("status") != "initialized":
            raise ArtifactError(f"run is already finalized: {self.run_id}")
        if status == "completed" and not (self.run_dir / "metrics.json").is_file():
            raise ArtifactError("completed runs require metrics.json")
        if status == "completed" and failure is not None:
            raise ArtifactError("completed runs cannot record a failure")
        if status != "completed" and (failure is None or not failure.strip()):
            raise ArtifactError("failed or aborted runs require a failure reason")

        payload["status"] = status
        payload["completed_at"] = _utc_text(completed_at)
        payload["failure"] = failure.strip() if failure is not None else None
        _atomic_write_json(run_path, payload)

    def _read_run(self) -> dict[str, Any]:
        run_path = self.run_dir / "run.json"
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ArtifactError(f"invalid run metadata: {run_path}") from error
        if not isinstance(payload, dict):
            raise ArtifactError(f"run metadata must be a mapping: {run_path}")
        if payload.get("schema_version") != RUN_SCHEMA_VERSION:
            raise ArtifactError(f"unsupported run metadata schema: {run_path}")
        if payload.get("run_id") != self.run_id:
            raise ArtifactError(f"run metadata ID mismatch: {run_path}")
        return payload

    def _register_artifact(self, name: str) -> None:
        artifact = self.run_dir / name
        payload = self._read_run()
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ArtifactError("run metadata artifacts must be a mapping")
        if name in artifacts:
            raise ArtifactError(f"artifact is already registered: {name}")
        artifacts[name] = {
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        _atomic_write_json(self.run_dir / "run.json", payload)
