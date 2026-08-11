from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mmprism.config import expand_environment

OMNIHAND_SMOKE_CONFIG_SCHEMA = "mmprism.omnihand_smoke.v1"
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


class OmniHandSmokeError(RuntimeError):
    """Raised when the canonical OmniHand smoke contract is invalid or fails."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OmniHandSmokeError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise OmniHandSmokeError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OmniHandSmokeError(f"{location}.{key} must be non-empty text")
    return value.strip()


def _integer(payload: Mapping[str, Any], key: str, location: str, *, minimum: int = 1) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise OmniHandSmokeError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(payload: Mapping[str, Any], key: str, location: str, *, minimum: float = 0.0) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OmniHandSmokeError(f"{location}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise OmniHandSmokeError(f"{location}.{key} must be finite and >= {minimum}")
    return result


def _boolean(payload: Mapping[str, Any], key: str, location: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise OmniHandSmokeError(f"{location}.{key} must be a boolean")
    return value


def _integer_tuple(
    payload: Mapping[str, Any], key: str, location: str, *, length: int | None = None
) -> tuple[int, ...]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value)
    ):
        raise OmniHandSmokeError(f"{location}.{key} must be a non-empty list of positive integers")
    if length is not None and len(value) != length:
        raise OmniHandSmokeError(f"{location}.{key} must contain {length} integers")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class OmniHandSpatialConfig:
    in_channels: int
    stem_channels: int
    stage_channels: tuple[int, ...]
    stage_depths: tuple[int, ...]
    channel_attention: bool
    spatial_attention: bool
    se_attention: bool
    use_pafpn: bool
    fpn_channels: int

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandSpatialConfig:
        location = "model.spatial"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {
                "in_channels",
                "stem_channels",
                "stage_channels",
                "stage_depths",
                "channel_attention",
                "spatial_attention",
                "se_attention",
                "use_pafpn",
                "fpn_channels",
            },
            location,
        )
        stage_channels = _integer_tuple(payload, "stage_channels", location)
        stage_depths = _integer_tuple(payload, "stage_depths", location)
        if len(stage_channels) != len(stage_depths):
            raise OmniHandSmokeError(
                "model.spatial.stage_channels and stage_depths must have equal length"
            )
        use_pafpn = _boolean(payload, "use_pafpn", location)
        if use_pafpn and len(stage_channels) < 2:
            raise OmniHandSmokeError("model.spatial.use_pafpn requires at least two stages")
        return cls(
            in_channels=_integer(payload, "in_channels", location),
            stem_channels=_integer(payload, "stem_channels", location),
            stage_channels=stage_channels,
            stage_depths=stage_depths,
            channel_attention=_boolean(payload, "channel_attention", location),
            spatial_attention=_boolean(payload, "spatial_attention", location),
            se_attention=_boolean(payload, "se_attention", location),
            use_pafpn=use_pafpn,
            fpn_channels=_integer(payload, "fpn_channels", location),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "stem_channels": self.stem_channels,
            "stage_channels": list(self.stage_channels),
            "stage_depths": list(self.stage_depths),
            "channel_attention": self.channel_attention,
            "spatial_attention": self.spatial_attention,
            "se_attention": self.se_attention,
            "use_pafpn": self.use_pafpn,
            "fpn_channels": self.fpn_channels,
        }


@dataclass(frozen=True, slots=True)
class OmniHandTemporalConfig:
    max_frames: int
    layers: int
    heads: int
    feedforward_dim: int
    dropout: float

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandTemporalConfig:
        location = "model.temporal"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {"max_frames", "layers", "heads", "feedforward_dim", "dropout"},
            location,
        )
        dropout = _number(payload, "dropout", location)
        if dropout >= 1:
            raise OmniHandSmokeError("model.temporal.dropout must be < 1")
        return cls(
            max_frames=_integer(payload, "max_frames", location),
            layers=_integer(payload, "layers", location),
            heads=_integer(payload, "heads", location),
            feedforward_dim=_integer(payload, "feedforward_dim", location),
            dropout=dropout,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_frames": self.max_frames,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
        }


@dataclass(frozen=True, slots=True)
class OmniHandModelConfig:
    spatial: OmniHandSpatialConfig
    temporal: OmniHandTemporalConfig
    joint_count: int
    coordinate_dim: int

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandModelConfig:
        location = "model"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"spatial", "temporal", "joint_count", "coordinate_dim"}, location)
        spatial = OmniHandSpatialConfig.from_mapping(payload.get("spatial"))
        temporal = OmniHandTemporalConfig.from_mapping(payload.get("temporal"))
        feature_dim = spatial.fpn_channels if spatial.use_pafpn else spatial.stage_channels[-1]
        if feature_dim % temporal.heads != 0:
            raise OmniHandSmokeError(
                "model.temporal.heads must divide the spatial output feature dimension"
            )
        if temporal.feedforward_dim < feature_dim:
            raise OmniHandSmokeError(
                "model.temporal.feedforward_dim must be >= the spatial feature dimension"
            )
        joint_count = _integer(payload, "joint_count", location)
        coordinate_dim = _integer(payload, "coordinate_dim", location)
        if (joint_count, coordinate_dim) != (24, 3):
            raise OmniHandSmokeError("model output must use the canonical [2,24,3] pose")
        return cls(
            spatial=spatial,
            temporal=temporal,
            joint_count=joint_count,
            coordinate_dim=coordinate_dim,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "spatial": self.spatial.to_dict(),
            "temporal": self.temporal.to_dict(),
            "joint_count": self.joint_count,
            "coordinate_dim": self.coordinate_dim,
        }


@dataclass(frozen=True, slots=True)
class OmniHandBatchConfig:
    batch_size: int
    frame_count: int
    spatial_shape: tuple[int, int, int]

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandBatchConfig:
        location = "batch"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"batch_size", "frame_count", "spatial_shape"}, location)
        spatial_shape = _integer_tuple(payload, "spatial_shape", location, length=3)
        return cls(
            batch_size=_integer(payload, "batch_size", location),
            frame_count=_integer(payload, "frame_count", location),
            spatial_shape=(spatial_shape[0], spatial_shape[1], spatial_shape[2]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "frame_count": self.frame_count,
            "spatial_shape": list(self.spatial_shape),
        }


@dataclass(frozen=True, slots=True)
class OmniHandOptimizationConfig:
    steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandOptimizationConfig:
        location = "optimization"
        payload = _mapping(value, location)
        _reject_unknown(
            payload, {"steps", "learning_rate", "weight_decay", "beta1", "beta2"}, location
        )
        learning_rate = _number(payload, "learning_rate", location)
        beta1 = _number(payload, "beta1", location)
        beta2 = _number(payload, "beta2", location)
        if learning_rate <= 0:
            raise OmniHandSmokeError("optimization.learning_rate must be > 0")
        if beta1 >= 1 or beta2 >= 1:
            raise OmniHandSmokeError("optimization beta values must be < 1")
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
class OmniHandMetricConfig:
    pck_threshold_mm: float

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandMetricConfig:
        location = "metrics"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"pck_threshold_mm"}, location)
        threshold = _number(payload, "pck_threshold_mm", location)
        if threshold <= 0:
            raise OmniHandSmokeError("metrics.pck_threshold_mm must be > 0")
        return cls(pck_threshold_mm=threshold)

    def to_dict(self) -> dict[str, float]:
        return {"pck_threshold_mm": self.pck_threshold_mm}


@dataclass(frozen=True, slots=True)
class OmniHandRuntimeConfig:
    seed: int
    dtype: str
    deterministic: bool

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandRuntimeConfig:
        location = "runtime"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"seed", "dtype", "deterministic"}, location)
        dtype = _text(payload, "dtype", location)
        if dtype not in {"float32", "bfloat16"}:
            raise OmniHandSmokeError("runtime.dtype must be float32 or bfloat16")
        return cls(
            seed=_integer(payload, "seed", location, minimum=0),
            dtype=dtype,
            deterministic=_boolean(payload, "deterministic", location),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "dtype": self.dtype,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True, slots=True)
class OmniHandSmokeConfig:
    smoke_id: str
    model: OmniHandModelConfig
    batch: OmniHandBatchConfig
    optimization: OmniHandOptimizationConfig
    metrics: OmniHandMetricConfig
    runtime: OmniHandRuntimeConfig

    @classmethod
    def from_mapping(cls, value: object) -> OmniHandSmokeConfig:
        payload = _mapping(value, "OmniHand smoke config")
        _reject_unknown(
            payload,
            {"schema_version", "smoke_id", "model", "batch", "optimization", "metrics", "runtime"},
            "OmniHand smoke config",
        )
        if payload.get("schema_version") != OMNIHAND_SMOKE_CONFIG_SCHEMA:
            raise OmniHandSmokeError(f"schema_version must be {OMNIHAND_SMOKE_CONFIG_SCHEMA}")
        smoke_id = _text(payload, "smoke_id", "OmniHand smoke config")
        if not _ID_PATTERN.fullmatch(smoke_id):
            raise OmniHandSmokeError("smoke_id must be a stable lowercase ID")
        model = OmniHandModelConfig.from_mapping(payload.get("model"))
        batch = OmniHandBatchConfig.from_mapping(payload.get("batch"))
        if batch.frame_count > model.temporal.max_frames:
            raise OmniHandSmokeError("batch.frame_count exceeds model.temporal.max_frames")
        if batch.batch_size < 2 or batch.frame_count < 2:
            raise OmniHandSmokeError(
                "OmniHand smoke requires batch_size and frame_count >= 2 to exercise masking"
            )
        return cls(
            smoke_id=smoke_id,
            model=model,
            batch=batch,
            optimization=OmniHandOptimizationConfig.from_mapping(payload.get("optimization")),
            metrics=OmniHandMetricConfig.from_mapping(payload.get("metrics")),
            runtime=OmniHandRuntimeConfig.from_mapping(payload.get("runtime")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OMNIHAND_SMOKE_CONFIG_SCHEMA,
            "smoke_id": self.smoke_id,
            "model": self.model.to_dict(),
            "batch": self.batch.to_dict(),
            "optimization": self.optimization.to_dict(),
            "metrics": self.metrics.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_omnihand_smoke_config(path: str | Path) -> OmniHandSmokeConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise OmniHandSmokeError(f"Unable to load OmniHand smoke config: {error}") from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise OmniHandSmokeError(str(error)) from error
    return OmniHandSmokeConfig.from_mapping(expanded)
