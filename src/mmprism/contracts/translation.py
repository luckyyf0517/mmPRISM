"""Small shared contracts for sign-language translation inputs.

This module deliberately has no NumPy, Torch, or model dependencies so data
and configuration validation can use the same vocabulary on CPU-only paths.
"""

from __future__ import annotations

POSE_ONLY_INPUT_MODE = "pose_only"
POSE_PLUS_RADAR_FEATURE_INPUT_MODE = "pose_plus_radar_feature"
TRANSLATION_INPUT_MODES = frozenset(
    {POSE_ONLY_INPUT_MODE, POSE_PLUS_RADAR_FEATURE_INPUT_MODE}
)


def translation_input_mode_modalities(input_mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return required and forbidden data modalities for one input mode."""

    if input_mode == POSE_ONLY_INPUT_MODE:
        return (
            ("pose", "pose_confidence", "frame_mask", "caption"),
            ("radar_feature",),
        )
    if input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
        return (
            ("pose", "pose_confidence", "radar_feature", "frame_mask", "caption"),
            (),
        )
    raise ValueError(f"unsupported translation input mode: {input_mode!r}")
