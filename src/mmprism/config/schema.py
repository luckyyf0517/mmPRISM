from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a configuration violates the canonical schema."""


class Task(str, Enum):
    POSE_RECONSTRUCTION = "pose_reconstruction"
    SIGN_LANGUAGE_TRANSLATION = "sign_language_translation"
    RADAR_SIMULATION = "radar_simulation"
    EVALUATION = "evaluation"


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _required_text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class PathConfig:
    data_root: Path
    artifact_root: Path
    cache_root: Path

    @classmethod
    def from_mapping(cls, value: Any) -> "PathConfig":
        payload = _mapping(value, "paths")
        _reject_unknown(payload, {"data_root", "artifact_root", "cache_root"}, "paths")
        return cls(
            data_root=Path(_required_text(payload, "data_root", "paths")).expanduser(),
            artifact_root=Path(_required_text(payload, "artifact_root", "paths")).expanduser(),
            cache_root=Path(_required_text(payload, "cache_root", "paths")).expanduser(),
        )

    def resolved(self, project_root: Path) -> "PathConfig":
        def resolve(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()

        return PathConfig(
            data_root=resolve(self.data_root),
            artifact_root=resolve(self.artifact_root),
            cache_root=resolve(self.cache_root),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "data_root": str(self.data_root),
            "artifact_root": str(self.artifact_root),
            "cache_root": str(self.cache_root),
        }


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    accelerator: str = "auto"
    devices: str | tuple[int, ...] = "auto"
    precision: str = "32-true"
    deterministic: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeConfig":
        payload = _mapping(value, "runtime")
        allowed = {"seed", "accelerator", "devices", "precision", "deterministic"}
        _reject_unknown(payload, allowed, "runtime")

        seed = payload.get("seed", 42)
        if not isinstance(seed, int) or seed < 0:
            raise ConfigError("runtime.seed must be a non-negative integer")

        accelerator = payload.get("accelerator", "auto")
        if not isinstance(accelerator, str) or not accelerator:
            raise ConfigError("runtime.accelerator must be a non-empty string")

        devices_value = payload.get("devices", "auto")
        if isinstance(devices_value, str):
            devices: str | tuple[int, ...] = devices_value
        elif isinstance(devices_value, list) and devices_value and all(
            isinstance(device, int) and device >= 0 for device in devices_value
        ):
            devices = tuple(devices_value)
        else:
            raise ConfigError("runtime.devices must be 'auto' or a list of non-negative integers")

        precision = payload.get("precision", "32-true")
        if precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise ConfigError("runtime.precision must be 32-true, 16-mixed, or bf16-mixed")

        deterministic = payload.get("deterministic", True)
        if not isinstance(deterministic, bool):
            raise ConfigError("runtime.deterministic must be a boolean")

        return cls(
            seed=seed,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            deterministic=deterministic,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "accelerator": self.accelerator,
            "devices": list(self.devices) if isinstance(self.devices, tuple) else self.devices,
            "precision": self.precision,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    name: str
    task: Task
    paths: PathConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Any) -> "ExperimentConfig":
        payload = _mapping(value, "configuration root")
        allowed = {"schema_version", "name", "task", "paths", "runtime"}
        _reject_unknown(payload, allowed, "configuration root")

        schema_version = _required_text(payload, "schema_version", "configuration root")
        if schema_version != "mmprism.experiment.v1":
            raise ConfigError(f"Unsupported experiment schema: {schema_version}")

        task_text = _required_text(payload, "task", "configuration root")
        try:
            task = Task(task_text)
        except ValueError as error:
            supported = ", ".join(task.value for task in Task)
            message = f"Unsupported task {task_text!r}; expected one of: {supported}"
            raise ConfigError(message) from error

        return cls(
            schema_version=schema_version,
            name=_required_text(payload, "name", "configuration root"),
            task=task,
            paths=PathConfig.from_mapping(payload.get("paths")),
            runtime=RuntimeConfig.from_mapping(payload.get("runtime", {})),
        )

    def resolved(self, project_root: Path) -> "ExperimentConfig":
        return ExperimentConfig(
            schema_version=self.schema_version,
            name=self.name,
            task=self.task,
            paths=self.paths.resolved(project_root),
            runtime=self.runtime,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "task": self.task.value,
            "paths": self.paths.to_dict(),
            "runtime": self.runtime.to_dict(),
        }
