"""Small, testable sequence and rectified-flow utilities.

This is intentionally not a generic diffusion framework.  It owns only the
fixed-clip roster, padded-loss arithmetic, rectified-flow interpolation, and
mask-safe Heun integrator used by the training launcher.
"""

from __future__ import annotations

import torch
from torch import Tensor


class RadarConditionedMotionFlowTrainingError(ValueError):
    """Raised when a fixed training contract is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RadarConditionedMotionFlowTrainingError(message)


def _masked_coordinate_reduce(
    values: Tensor, *, valid_mask: Tensor, capacity_frames: int, squared: bool
) -> Tensor:
    _require(values.ndim >= 3, "masked values must have batch, time, and coordinates")
    batch_size, frames = values.shape[:2]
    _require(
        valid_mask.shape == (batch_size, frames) and valid_mask.dtype == torch.bool,
        "masked loss mask differs",
    )
    _require(capacity_frames > 0, "masked loss capacity must be positive")
    per_frame_dimensions = values[0, 0].numel()
    _require(per_frame_dimensions > 0, "masked loss coordinates are empty")
    reduced = values.square() if squared else values.abs()
    weighted = reduced * valid_mask.reshape(batch_size, frames, *([1] * (values.ndim - 2))).to(
        dtype=values.dtype
    )
    return weighted.sum() / float(batch_size * capacity_frames * per_frame_dimensions)


def masked_coordinate_l1(
    prediction: Tensor, target: Tensor, *, valid_mask: Tensor, capacity_frames: int
) -> Tensor:
    _require(prediction.shape == target.shape, "masked L1 prediction and target differ")
    return _masked_coordinate_reduce(
        prediction - target,
        valid_mask=valid_mask,
        capacity_frames=capacity_frames,
        squared=False,
    )
