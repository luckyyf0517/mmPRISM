from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mmprism.config import expand_environment

MT5_SMOKE_CONFIG_SCHEMA = "mmprism.mt5_smoke.v1"
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


class MT5SmokeError(RuntimeError):
    """Raised when the canonical mT5 smoke contract is invalid or fails."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MT5SmokeError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise MT5SmokeError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MT5SmokeError(f"{location}.{key} must be non-empty text")
    return value.strip()


def _integer(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: int = 1
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise MT5SmokeError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: float = 0.0
) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MT5SmokeError(f"{location}.{key} must be a number")
    result = float(value)
    if result < minimum:
        raise MT5SmokeError(f"{location}.{key} must be >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class MT5ModelConfig:
    asset_id: str
    hidden_size: int
    radar_feature_dim: int
    joint_count: int
    coordinate_dim: int
    pose_channels: tuple[int, ...]
    temporal_kernel_size: int
    dropout: float
    label_smoothing: float
    freeze_language_model: bool

    @classmethod
    def from_mapping(cls, value: object) -> MT5ModelConfig:
        location = "model"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {
                "asset_id",
                "hidden_size",
                "radar_feature_dim",
                "joint_count",
                "coordinate_dim",
                "pose_channels",
                "temporal_kernel_size",
                "dropout",
                "label_smoothing",
                "freeze_language_model",
            },
            location,
        )
        asset_id = _text(payload, "asset_id", location)
        if not _ID_PATTERN.fullmatch(asset_id):
            raise MT5SmokeError("model.asset_id must be a stable lowercase ID")
        channels_value = payload.get("pose_channels")
        if (
            not isinstance(channels_value, list)
            or not channels_value
            or any(
                not isinstance(channel, int) or isinstance(channel, bool) or channel < 1
                for channel in channels_value
            )
        ):
            raise MT5SmokeError("model.pose_channels must be a non-empty list of positive integers")
        temporal_kernel_size = _integer(payload, "temporal_kernel_size", location)
        if temporal_kernel_size % 2 == 0:
            raise MT5SmokeError("model.temporal_kernel_size must be odd")
        dropout = _number(payload, "dropout", location)
        if dropout >= 1:
            raise MT5SmokeError("model.dropout must be < 1")
        label_smoothing = _number(payload, "label_smoothing", location)
        if label_smoothing >= 1:
            raise MT5SmokeError("model.label_smoothing must be < 1")
        freeze_language_model = payload.get("freeze_language_model")
        if not isinstance(freeze_language_model, bool):
            raise MT5SmokeError("model.freeze_language_model must be a boolean")
        return cls(
            asset_id=asset_id,
            hidden_size=_integer(payload, "hidden_size", location),
            radar_feature_dim=_integer(payload, "radar_feature_dim", location),
            joint_count=_integer(payload, "joint_count", location),
            coordinate_dim=_integer(payload, "coordinate_dim", location),
            pose_channels=tuple(channels_value),
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
            label_smoothing=label_smoothing,
            freeze_language_model=freeze_language_model,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "hidden_size": self.hidden_size,
            "radar_feature_dim": self.radar_feature_dim,
            "joint_count": self.joint_count,
            "coordinate_dim": self.coordinate_dim,
            "pose_channels": list(self.pose_channels),
            "temporal_kernel_size": self.temporal_kernel_size,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "freeze_language_model": self.freeze_language_model,
        }


@dataclass(frozen=True, slots=True)
class MT5BatchConfig:
    batch_size: int
    frame_count: int
    prompt: str
    targets: tuple[str, ...]
    max_target_length: int

    @classmethod
    def from_mapping(cls, value: object) -> MT5BatchConfig:
        location = "batch"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {"batch_size", "frame_count", "prompt", "targets", "max_target_length"},
            location,
        )
        batch_size = _integer(payload, "batch_size", location)
        targets_value = payload.get("targets")
        if (
            not isinstance(targets_value, list)
            or len(targets_value) != batch_size
            or any(not isinstance(target, str) or not target.strip() for target in targets_value)
        ):
            raise MT5SmokeError("batch.targets must contain one non-empty string per sample")
        return cls(
            batch_size=batch_size,
            frame_count=_integer(payload, "frame_count", location),
            prompt=_text(payload, "prompt", location),
            targets=tuple(target.strip() for target in targets_value),
            max_target_length=_integer(payload, "max_target_length", location),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "frame_count": self.frame_count,
            "prompt": self.prompt,
            "targets": list(self.targets),
            "max_target_length": self.max_target_length,
        }


@dataclass(frozen=True, slots=True)
class MT5OptimizationConfig:
    steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float

    @classmethod
    def from_mapping(cls, value: object) -> MT5OptimizationConfig:
        location = "optimization"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {"steps", "learning_rate", "weight_decay", "beta1", "beta2"},
            location,
        )
        learning_rate = _number(payload, "learning_rate", location)
        if learning_rate <= 0:
            raise MT5SmokeError("optimization.learning_rate must be > 0")
        beta1 = _number(payload, "beta1", location)
        beta2 = _number(payload, "beta2", location)
        if beta1 >= 1 or beta2 >= 1:
            raise MT5SmokeError("optimization beta values must be < 1")
        return cls(
            steps=_integer(payload, "steps", location),
            learning_rate=learning_rate,
            weight_decay=_number(payload, "weight_decay", location),
            beta1=beta1,
            beta2=beta2,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
        }


@dataclass(frozen=True, slots=True)
class MT5GenerationConfig:
    max_new_tokens: int
    num_beams: int

    @classmethod
    def from_mapping(cls, value: object) -> MT5GenerationConfig:
        location = "generation"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"max_new_tokens", "num_beams"}, location)
        return cls(
            max_new_tokens=_integer(payload, "max_new_tokens", location),
            num_beams=_integer(payload, "num_beams", location),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
        }


@dataclass(frozen=True, slots=True)
class MT5SmokeRuntimeConfig:
    seed: int
    dtype: str
    deterministic: bool

    @classmethod
    def from_mapping(cls, value: object) -> MT5SmokeRuntimeConfig:
        location = "runtime"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"seed", "dtype", "deterministic"}, location)
        seed = _integer(payload, "seed", location, minimum=0)
        dtype = _text(payload, "dtype", location)
        if dtype not in {"float32", "bfloat16"}:
            raise MT5SmokeError("runtime.dtype must be float32 or bfloat16")
        deterministic = payload.get("deterministic")
        if not isinstance(deterministic, bool):
            raise MT5SmokeError("runtime.deterministic must be a boolean")
        return cls(seed=seed, dtype=dtype, deterministic=deterministic)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "dtype": self.dtype,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True, slots=True)
class MT5SmokeConfig:
    smoke_id: str
    model: MT5ModelConfig
    batch: MT5BatchConfig
    optimization: MT5OptimizationConfig
    generation: MT5GenerationConfig
    runtime: MT5SmokeRuntimeConfig

    @classmethod
    def from_mapping(cls, value: object) -> MT5SmokeConfig:
        payload = _mapping(value, "mT5 smoke config")
        _reject_unknown(
            payload,
            {
                "schema_version",
                "smoke_id",
                "model",
                "batch",
                "optimization",
                "generation",
                "runtime",
            },
            "mT5 smoke config",
        )
        if payload.get("schema_version") != MT5_SMOKE_CONFIG_SCHEMA:
            raise MT5SmokeError(f"schema_version must be {MT5_SMOKE_CONFIG_SCHEMA}")
        smoke_id = _text(payload, "smoke_id", "mT5 smoke config")
        if not _ID_PATTERN.fullmatch(smoke_id):
            raise MT5SmokeError("smoke_id must be a stable lowercase ID")
        return cls(
            smoke_id=smoke_id,
            model=MT5ModelConfig.from_mapping(payload.get("model")),
            batch=MT5BatchConfig.from_mapping(payload.get("batch")),
            optimization=MT5OptimizationConfig.from_mapping(payload.get("optimization")),
            generation=MT5GenerationConfig.from_mapping(payload.get("generation")),
            runtime=MT5SmokeRuntimeConfig.from_mapping(payload.get("runtime")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MT5_SMOKE_CONFIG_SCHEMA,
            "smoke_id": self.smoke_id,
            "model": self.model.to_dict(),
            "batch": self.batch.to_dict(),
            "optimization": self.optimization.to_dict(),
            "generation": self.generation.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_mt5_smoke_config(path: str | Path) -> MT5SmokeConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise MT5SmokeError(f"Unable to load mT5 smoke config: {error}") from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise MT5SmokeError(str(error)) from error
    return MT5SmokeConfig.from_mapping(expanded)
