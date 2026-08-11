from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mmprism.config import expand_environment
from mmprism.training.mt5_config import MT5ModelConfig, MT5SmokeError

WAVELLM_RUN_CONFIG_SCHEMA = "mmprism.wavellm_run.v1"


class WaveLLMRunError(RuntimeError):
    """Raised when a formal WaveLLM run contract is invalid or fails."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WaveLLMRunError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise WaveLLMRunError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WaveLLMRunError(f"{location}.{key} must be non-empty text")
    return value.strip()


def _integer(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: int = 1
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise WaveLLMRunError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: float = 0.0
) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WaveLLMRunError(f"{location}.{key} must be a number")
    result = float(value)
    if result < minimum:
        raise WaveLLMRunError(f"{location}.{key} must be >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class WaveLLMDataConfig:
    batch_size: int
    num_workers: int
    verify_checksums: bool
    shuffle: bool
    max_frames: int
    prompt: str
    max_prompt_length: int
    max_target_length: int

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMDataConfig:
        location = "data"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {
                "batch_size",
                "num_workers",
                "verify_checksums",
                "shuffle",
                "max_frames",
                "prompt",
                "max_prompt_length",
                "max_target_length",
            },
            location,
        )
        num_workers = payload.get("num_workers")
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise WaveLLMRunError("data.num_workers must be a non-negative integer")
        verify_checksums = payload.get("verify_checksums")
        shuffle = payload.get("shuffle")
        if not isinstance(verify_checksums, bool):
            raise WaveLLMRunError("data.verify_checksums must be a boolean")
        if not isinstance(shuffle, bool):
            raise WaveLLMRunError("data.shuffle must be a boolean")
        return cls(
            batch_size=_integer(payload, "batch_size", location),
            num_workers=num_workers,
            verify_checksums=verify_checksums,
            shuffle=shuffle,
            max_frames=_integer(payload, "max_frames", location),
            prompt=_text(payload, "prompt", location),
            max_prompt_length=_integer(payload, "max_prompt_length", location),
            max_target_length=_integer(payload, "max_target_length", location),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "verify_checksums": self.verify_checksums,
            "shuffle": self.shuffle,
            "max_frames": self.max_frames,
            "prompt": self.prompt,
            "max_prompt_length": self.max_prompt_length,
            "max_target_length": self.max_target_length,
        }


@dataclass(frozen=True, slots=True)
class WaveLLMRunOptimizationConfig:
    epochs: int
    max_steps: int | None
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip_norm: float

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMRunOptimizationConfig:
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
        if max_steps_value is not None and (
            not isinstance(max_steps_value, int)
            or isinstance(max_steps_value, bool)
            or max_steps_value < 1
        ):
            raise WaveLLMRunError("optimization.max_steps must be null or a positive integer")
        learning_rate = _number(payload, "learning_rate", location)
        gradient_clip_norm = _number(payload, "gradient_clip_norm", location)
        beta1 = _number(payload, "beta1", location)
        beta2 = _number(payload, "beta2", location)
        if learning_rate <= 0:
            raise WaveLLMRunError("optimization.learning_rate must be > 0")
        if gradient_clip_norm <= 0:
            raise WaveLLMRunError("optimization.gradient_clip_norm must be > 0")
        if beta1 >= 1 or beta2 >= 1:
            raise WaveLLMRunError("optimization beta values must be < 1")
        return cls(
            epochs=_integer(payload, "epochs", location),
            max_steps=max_steps_value,
            learning_rate=learning_rate,
            weight_decay=_number(payload, "weight_decay", location),
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
class WaveLLMGenerationConfig:
    max_new_tokens: int
    num_beams: int

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMGenerationConfig:
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
class WaveLLMEvaluationConfig:
    save_references: bool

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMEvaluationConfig:
        location = "evaluation"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"save_references"}, location)
        save_references = payload.get("save_references")
        if not isinstance(save_references, bool):
            raise WaveLLMRunError("evaluation.save_references must be a boolean")
        return cls(save_references=save_references)

    def to_dict(self) -> dict[str, bool]:
        return {"save_references": self.save_references}


@dataclass(frozen=True, slots=True)
class WaveLLMRunConfig:
    model: MT5ModelConfig
    data: WaveLLMDataConfig
    optimization: WaveLLMRunOptimizationConfig
    generation: WaveLLMGenerationConfig
    evaluation: WaveLLMEvaluationConfig

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMRunConfig:
        payload = _mapping(value, "WaveLLM run config")
        _reject_unknown(
            payload,
            {
                "schema_version",
                "model",
                "data",
                "optimization",
                "generation",
                "evaluation",
            },
            "WaveLLM run config",
        )
        if payload.get("schema_version") != WAVELLM_RUN_CONFIG_SCHEMA:
            raise WaveLLMRunError(
                f"schema_version must be {WAVELLM_RUN_CONFIG_SCHEMA}"
            )
        try:
            model = MT5ModelConfig.from_mapping(payload.get("model"))
        except MT5SmokeError as error:
            raise WaveLLMRunError(str(error)) from error
        return cls(
            model=model,
            data=WaveLLMDataConfig.from_mapping(payload.get("data")),
            optimization=WaveLLMRunOptimizationConfig.from_mapping(
                payload.get("optimization")
            ),
            generation=WaveLLMGenerationConfig.from_mapping(payload.get("generation")),
            evaluation=WaveLLMEvaluationConfig.from_mapping(payload.get("evaluation")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WAVELLM_RUN_CONFIG_SCHEMA,
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "optimization": self.optimization.to_dict(),
            "generation": self.generation.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def model_fingerprint(self) -> str:
        encoded = json.dumps(
            self.model.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_wavellm_run_config(path: str | Path) -> WaveLLMRunConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise WaveLLMRunError(f"Unable to load WaveLLM run config: {error}") from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise WaveLLMRunError(str(error)) from error
    return WaveLLMRunConfig.from_mapping(expanded)
