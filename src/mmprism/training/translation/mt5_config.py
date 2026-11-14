from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


class ReaderConfigurationError(RuntimeError):
    """Raised when the reader model configuration is invalid."""


READER_GEOMETRY_BRANCH_KINDS: tuple[str, ...] = (
    "local_geometry",
    "local_geometry_plus_motion",
)


def _optional_reader_geometry_branch(
    payload: Mapping[str, Any],
) -> str | None:
    if "reader_geometry_branch" not in payload:
        return None
    value = payload["reader_geometry_branch"]
    if value not in READER_GEOMETRY_BRANCH_KINDS:
        raise ReaderConfigurationError(
            "model.reader_geometry_branch must be one of: "
            + ", ".join(READER_GEOMETRY_BRANCH_KINDS)
        )
    return str(value)


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReaderConfigurationError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ReaderConfigurationError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReaderConfigurationError(f"{location}.{key} must be non-empty text")
    return value.strip()


def _integer(payload: Mapping[str, Any], key: str, location: str, *, minimum: int = 1) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ReaderConfigurationError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(payload: Mapping[str, Any], key: str, location: str, *, minimum: float = 0.0) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReaderConfigurationError(f"{location}.{key} must be a number")
    result = float(value)
    if result < minimum:
        raise ReaderConfigurationError(f"{location}.{key} must be >= {minimum}")
    return result


WAVELLM_LEARNABLE_POSE_OUTPUT_SCALE_V1 = "learnable_global_pose_output_scale_v1"


@dataclass(frozen=True, slots=True)
class WaveLLMPoseOutputScalingConfig:
    """Optional global scale applied after the pose encoder."""

    initial_scale: float
    target_token_l2_norm: float

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMPoseOutputScalingConfig:
        location = "model.pose_output_scaling"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {"kind", "initial_scale", "target_token_l2_norm"},
            location,
        )
        if payload.get("kind") != WAVELLM_LEARNABLE_POSE_OUTPUT_SCALE_V1:
            raise ReaderConfigurationError(
                f"{location}.kind must be {WAVELLM_LEARNABLE_POSE_OUTPUT_SCALE_V1}"
            )
        initial_scale = _number(payload, "initial_scale", location)
        target_token_l2_norm = _number(payload, "target_token_l2_norm", location)
        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ReaderConfigurationError(f"{location}.initial_scale must be finite and > 0")
        if not math.isfinite(target_token_l2_norm) or target_token_l2_norm <= 0.0:
            raise ReaderConfigurationError(
                f"{location}.target_token_l2_norm must be finite and > 0"
            )
        return cls(
            initial_scale=initial_scale,
            target_token_l2_norm=target_token_l2_norm,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": WAVELLM_LEARNABLE_POSE_OUTPUT_SCALE_V1,
            "initial_scale": self.initial_scale,
            "target_token_l2_norm": self.target_token_l2_norm,
        }


@dataclass(frozen=True, slots=True)
class MT5ModelConfig:
    asset_id: str
    hidden_size: int
    radar_feature_dim: int | None
    joint_count: int
    coordinate_dim: int
    pose_channels: tuple[int, ...]
    temporal_kernel_size: int
    dropout: float
    label_smoothing: float
    freeze_language_model: bool
    pose_output_scaling: WaveLLMPoseOutputScalingConfig | None = None
    reader_geometry_branch: str | None = None

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
                "pose_output_scaling",
                "reader_geometry_branch",
            },
            location,
        )
        asset_id = _text(payload, "asset_id", location)
        if not _ID_PATTERN.fullmatch(asset_id):
            raise ReaderConfigurationError("model.asset_id must be a stable lowercase ID")
        channels_value = payload.get("pose_channels")
        if (
            not isinstance(channels_value, list)
            or not channels_value
            or any(
                not isinstance(channel, int) or isinstance(channel, bool) or channel < 1
                for channel in channels_value
            )
        ):
            raise ReaderConfigurationError(
                "model.pose_channels must be a non-empty list of positive integers"
            )
        temporal_kernel_size = _integer(payload, "temporal_kernel_size", location)
        if temporal_kernel_size % 2 == 0:
            raise ReaderConfigurationError("model.temporal_kernel_size must be odd")
        dropout = _number(payload, "dropout", location)
        if dropout >= 1:
            raise ReaderConfigurationError("model.dropout must be < 1")
        label_smoothing = _number(payload, "label_smoothing", location)
        if label_smoothing >= 1:
            raise ReaderConfigurationError("model.label_smoothing must be < 1")
        freeze_language_model = payload.get("freeze_language_model")
        if not isinstance(freeze_language_model, bool):
            raise ReaderConfigurationError("model.freeze_language_model must be a boolean")
        return cls(
            asset_id=asset_id,
            hidden_size=_integer(payload, "hidden_size", location),
            radar_feature_dim=(
                _integer(payload, "radar_feature_dim", location)
                if "radar_feature_dim" in payload
                else None
            ),
            joint_count=_integer(payload, "joint_count", location),
            coordinate_dim=_integer(payload, "coordinate_dim", location),
            pose_channels=tuple(channels_value),
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
            label_smoothing=label_smoothing,
            freeze_language_model=freeze_language_model,
            pose_output_scaling=(
                WaveLLMPoseOutputScalingConfig.from_mapping(payload["pose_output_scaling"])
                if "pose_output_scaling" in payload
                else None
            ),
            reader_geometry_branch=_optional_reader_geometry_branch(payload),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "hidden_size": self.hidden_size,
            **(
                {"radar_feature_dim": self.radar_feature_dim}
                if self.radar_feature_dim is not None
                else {}
            ),
            "joint_count": self.joint_count,
            "coordinate_dim": self.coordinate_dim,
            "pose_channels": list(self.pose_channels),
            "temporal_kernel_size": self.temporal_kernel_size,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "freeze_language_model": self.freeze_language_model,
            **(
                {"pose_output_scaling": self.pose_output_scaling.to_dict()}
                if self.pose_output_scaling is not None
                else {}
            ),
            **(
                {"reader_geometry_branch": self.reader_geometry_branch}
                if self.reader_geometry_branch is not None
                else {}
            ),
        }
