"""Small shared contracts for sign-language translation inputs.

This module deliberately has no NumPy, Torch, or model dependencies so data
and configuration validation can use the same vocabulary on CPU-only paths.
"""

from __future__ import annotations

POSE_ONLY_INPUT_MODE = "pose_only"

RADAR_FEATURE_ONLY_INPUT_MODE = "radar_feature_only"

POINTCLOUD_ONLY_INPUT_MODE = "pointcloud_only"

POSE_PLUS_RADAR_FEATURE_INPUT_MODE = "pose_plus_radar_feature"

TRANSLATION_INPUT_MODES = frozenset(
    {
        POSE_ONLY_INPUT_MODE,
        RADAR_FEATURE_ONLY_INPUT_MODE,
        POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
        POINTCLOUD_ONLY_INPUT_MODE,
    }
)

WAVELLM_INPUT_MODES = frozenset(
    {
        POSE_ONLY_INPUT_MODE,
        RADAR_FEATURE_ONLY_INPUT_MODE,
        POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
        POINTCLOUD_ONLY_INPUT_MODE,
    }
)


def translation_input_mode_modalities(input_mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return required and forbidden data modalities for one input mode."""

    if input_mode == POSE_ONLY_INPUT_MODE:
        return (
            ("pose", "pose_confidence", "frame_mask", "caption"),
            ("radar_feature",),
        )
    if input_mode == RADAR_FEATURE_ONLY_INPUT_MODE:
        return (
            ("radar_feature", "frame_mask", "caption"),
            ("pose", "pose_confidence"),
        )
    if input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
        return (
            ("pose", "pose_confidence", "radar_feature", "frame_mask", "caption"),
            (),
        )
    if input_mode == POINTCLOUD_ONLY_INPUT_MODE:
        return (
            ("pointcloud", "frame_mask", "caption"),
            ("pose", "pose_confidence", "radar_feature"),
        )
    raise ValueError(f"unsupported translation input mode: {input_mode!r}")


def wavellm_input_mode_uses_radar_feature(input_mode: str) -> bool:
    """Return whether one WaveLLM consumer mode requires contextual features."""

    if input_mode not in WAVELLM_INPUT_MODES:
        raise ValueError(f"unsupported WaveLLM input mode: {input_mode!r}")
    return input_mode in {
        RADAR_FEATURE_ONLY_INPUT_MODE,
        POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
    }


def translation_delivery_supports_wavellm_input_mode(
    delivery_input_mode: str,
    consumer_input_mode: str,
) -> bool:
    """Return whether a translation delivery contains a requested consumer subset."""

    if delivery_input_mode not in TRANSLATION_INPUT_MODES:
        raise ValueError(f"unsupported translation delivery mode: {delivery_input_mode!r}")
    if consumer_input_mode not in WAVELLM_INPUT_MODES:
        raise ValueError(f"unsupported WaveLLM input mode: {consumer_input_mode!r}")
    if delivery_input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
        return True
    return consumer_input_mode == delivery_input_mode
