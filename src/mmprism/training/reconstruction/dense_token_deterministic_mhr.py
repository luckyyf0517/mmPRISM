"""Body-parameter packing, normalization and optimizer-step scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from mmprism.models.dense_token_deterministic_mhr import (
    MHR_DYNAMIC_DIMENSION,
    MHR_HAND_DIMENSION,
    MHR_MORPHOLOGY_DIMENSION,
    MHR_STANDARD_SCALE_DIMENSION,
    MHR_STATE_DIMENSION,
    MHR_UPPER_DIMENSION,
)

UPPER_BODY_INDICES: Final[tuple[int, ...]] = tuple(range(44)) + (124, 125, 126, 127)


class DenseTokenMhrTrainingError(ValueError):
    """Raised when DMHR target or step scheduling contracts are violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DenseTokenMhrTrainingError(message)


@dataclass(frozen=True)
class CompleteMhrStateParts:
    """Named slices of a complete 191-D state for frozen evaluation decode."""

    upper_articulation: Tensor
    left_hand_articulation: Tensor
    right_hand_articulation: Tensor
    global_rotation: Tensor
    camera_translation: Tensor
    morphology: Tensor
    lambda_to_standard: Tensor


def split_complete_mhr_state(state: Tensor) -> CompleteMhrStateParts:
    """Split ``[...,191]`` using the one MHR state order."""

    _require(state.shape[-1] == MHR_STATE_DIMENSION, "complete MHR state width differs")
    _require(bool(torch.isfinite(state).all()), "complete MHR state is non-finite")
    return CompleteMhrStateParts(
        upper_articulation=state[..., :MHR_UPPER_DIMENSION],
        left_hand_articulation=state[
            ..., MHR_UPPER_DIMENSION : MHR_UPPER_DIMENSION + MHR_HAND_DIMENSION
        ],
        right_hand_articulation=state[
            ...,
            MHR_UPPER_DIMENSION + MHR_HAND_DIMENSION : MHR_UPPER_DIMENSION + 2 * MHR_HAND_DIMENSION,
        ],
        global_rotation=state[
            ..., 2 * MHR_HAND_DIMENSION + MHR_UPPER_DIMENSION : MHR_DYNAMIC_DIMENSION - 3
        ],
        camera_translation=state[..., MHR_DYNAMIC_DIMENSION - 3 : MHR_DYNAMIC_DIMENSION],
        morphology=state[
            ..., MHR_DYNAMIC_DIMENSION : MHR_DYNAMIC_DIMENSION + MHR_MORPHOLOGY_DIMENSION
        ],
        lambda_to_standard=state[..., -MHR_STANDARD_SCALE_DIMENSION:],
    )
