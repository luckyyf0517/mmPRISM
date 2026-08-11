from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mmprism.config import expand_environment
from mmprism.training.omnihand_config import OmniHandModelConfig, OmniHandSmokeError

OMNIHAND_RUN_CONFIG_SCHEMA = "mmprism.omnihand_run.v1"


class OmniHandRunError(RuntimeError):
    """Raised when an OmniHand formal-run contract is invalid or execution fails."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OmniHandRunError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise OmniHandRunError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _integer(payload: Mapping[str, Any], key: str, location: str, *, minimum: int) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise OmniHandRunError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(payload: Mapping[str, Any], key: str, location: str, *, minimum: float) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OmniHandRunError(f"{location}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise OmniHandRunError(f"{location}.{key} must be finite and >= {minimum}")
    return result


def _boolean(payload: Mapping[str, Any], key: str, location: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise OmniHandRunError(f"{location}.{key} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class OmniHandDataConfig:
    batch_size: int
    num_workers: int
    verify_checksums: bool
    shuffle: bool

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandDataConfig:
        location = "data"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {"batch_size", "num_workers", "verify_checksums", "shuffle"},
            location,
        )
        return cls(
            batch_size=_integer(payload, "batch_size", location, minimum=1),
            num_workers=_integer(payload, "num_workers", location, minimum=0),
            verify_checksums=_boolean(payload, "verify_checksums", location),
            shuffle=_boolean(payload, "shuffle", location),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "verify_checksums": self.verify_checksums,
            "shuffle": self.shuffle,
        }


@dataclass(frozen=True, slots=True)
class OmniHandRunOptimizationConfig:
    epochs: int
    max_steps: int | None
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip_norm: float

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandRunOptimizationConfig:
        location = "optimization"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {
                "epochs",
                "max_steps",
                "learning_rate",
                "weight_decay",
                "beta1",
                "beta2",
                "gradient_clip_norm",
            },
            location,
        )
        max_steps_value = payload.get("max_steps")
        if max_steps_value is None:
            max_steps = None
        elif (
            not isinstance(max_steps_value, int)
            or isinstance(max_steps_value, bool)
            or max_steps_value < 1
        ):
            raise OmniHandRunError("optimization.max_steps must be null or a positive integer")
        else:
            max_steps = max_steps_value
        learning_rate = _number(payload, "learning_rate", location, minimum=0.0)
        beta1 = _number(payload, "beta1", location, minimum=0.0)
        beta2 = _number(payload, "beta2", location, minimum=0.0)
        gradient_clip_norm = _number(payload, "gradient_clip_norm", location, minimum=0.0)
        if learning_rate <= 0:
            raise OmniHandRunError("optimization.learning_rate must be > 0")
        if beta1 >= 1 or beta2 >= 1:
            raise OmniHandRunError("optimization beta values must be < 1")
        if gradient_clip_norm <= 0:
            raise OmniHandRunError("optimization.gradient_clip_norm must be > 0")
        return cls(
            epochs=_integer(payload, "epochs", location, minimum=1),
            max_steps=max_steps,
            learning_rate=learning_rate,
            weight_decay=_number(payload, "weight_decay", location, minimum=0.0),
            beta1=beta1,
            beta2=beta2,
            gradient_clip_norm=gradient_clip_norm,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "gradient_clip_norm": self.gradient_clip_norm,
        }


@dataclass(frozen=True, slots=True)
class OmniHandEvaluationConfig:
    pck_threshold_mm: float
    save_targets: bool

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandEvaluationConfig:
        location = "evaluation"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"pck_threshold_mm", "save_targets"}, location)
        threshold = _number(payload, "pck_threshold_mm", location, minimum=0.0)
        if threshold <= 0:
            raise OmniHandRunError("evaluation.pck_threshold_mm must be > 0")
        return cls(
            pck_threshold_mm=threshold,
            save_targets=_boolean(payload, "save_targets", location),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pck_threshold_mm": self.pck_threshold_mm,
            "save_targets": self.save_targets,
        }


@dataclass(frozen=True, slots=True)
class OmniHandRunConfig:
    model: OmniHandModelConfig
    data: OmniHandDataConfig
    optimization: OmniHandRunOptimizationConfig
    evaluation: OmniHandEvaluationConfig

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandRunConfig:
        payload = _mapping(value, "OmniHand run config")
        _reject_unknown(
            payload,
            {"schema_version", "model", "data", "optimization", "evaluation"},
            "OmniHand run config",
        )
        if payload.get("schema_version") != OMNIHAND_RUN_CONFIG_SCHEMA:
            raise OmniHandRunError(f"schema_version must be {OMNIHAND_RUN_CONFIG_SCHEMA}")
        try:
            model = OmniHandModelConfig.from_mapping(payload.get("model"))
        except OmniHandSmokeError as error:
            raise OmniHandRunError(str(error)) from error
        return cls(
            model=model,
            data=OmniHandDataConfig.from_mapping(payload.get("data")),
            optimization=OmniHandRunOptimizationConfig.from_mapping(payload.get("optimization")),
            evaluation=OmniHandEvaluationConfig.from_mapping(payload.get("evaluation")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OMNIHAND_RUN_CONFIG_SCHEMA,
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "optimization": self.optimization.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @property
    def model_fingerprint(self) -> str:
        return _fingerprint(self.model.to_dict())


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_omnihand_run_config(path: str | Path) -> OmniHandRunConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise OmniHandRunError(f"Unable to load OmniHand run config: {error}") from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise OmniHandRunError(str(error)) from error
    return OmniHandRunConfig.from_mapping(expanded)
