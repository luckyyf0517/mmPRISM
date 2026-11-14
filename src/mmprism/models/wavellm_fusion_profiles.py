"""Stable WaveLLM fusion variant identities without training dependencies."""

from __future__ import annotations

WAVELLM_FUSION_COMPACT_CONFIDENCE_V1 = "confidence_aware_compact_v1"

WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1 = "pose_anchored_residual_pf_v1"

WAVELLM_FUSION_VARIANTS = frozenset(
    (
        WAVELLM_FUSION_COMPACT_CONFIDENCE_V1,
        WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1,
    )
)
