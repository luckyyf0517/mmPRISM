"""Coordinate target and loss helpers for the direct-coordinate heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmprism.models.dense_token_direct_xyz import HAND_POINT_COUNT
from mmprism.training.reconstruction.radar_conditioned_motion_flow import masked_coordinate_l1

COORDINATE_SCALE_METRES = 0.1


class DenseTokenDirectXyzTrainingError(ValueError):
    """Raised when direct-coordinate targets or masks violate their contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DenseTokenDirectXyzTrainingError(message)


@dataclass(frozen=True)
class DirectXyzTargets:
    """Direct camera-global coarse and wrist-relative bilateral hand targets."""

    coarse18_m: Tensor
    hand120_wrist_relative_m: Tensor


@dataclass(frozen=True)
class DirectXyzCoordinateLoss:
    """The fixed equal-weight coarse-plus-hand coordinate objective."""

    total: Tensor
    coarse_coordinate_l1: Tensor
    hand_coordinate_l1: Tensor


def direct_xyz_targets(target_global_m: Tensor) -> DirectXyzTargets:
    """Extract the 18-D coarse and 120-D local hand targets from full24."""

    _require(
        target_global_m.ndim == 5 and target_global_m.shape[2:] == (2, 24, 3),
        "DXYZ target must be [B,T,2,24,3]",
    )
    _require(bool(torch.isfinite(target_global_m).all()), "DXYZ target is non-finite")
    coarse = target_global_m[..., :3, :].reshape(*target_global_m.shape[:2], 18)
    hand = (target_global_m[..., 4:, :] - target_global_m[..., 2:3, :]).reshape(
        *target_global_m.shape[:2], 2 * HAND_POINT_COUNT * 3
    )
    _require(coarse.shape[-1] == 18 and hand.shape[-1] == 120, "DXYZ target width differs")
    return DirectXyzTargets(coarse18_m=coarse, hand120_wrist_relative_m=hand)


def masked_direct_xyz_coordinate_l1(
    *,
    predicted_coarse18_m: Tensor,
    predicted_hand120_wrist_relative_m: Tensor,
    target_global_m: Tensor,
    valid_mask: Tensor,
    capacity_frames: int,
    coordinate_scale_metres: float = COORDINATE_SCALE_METRES,
) -> DirectXyzCoordinateLoss:
    """Apply RCMF-C's fixed 0.1-m scaled masked L1 to both coordinate groups."""

    _require(coordinate_scale_metres > 0.0, "DXYZ coordinate scale must be positive")
    targets = direct_xyz_targets(target_global_m)
    _require(
        predicted_coarse18_m.shape == targets.coarse18_m.shape,
        "DXYZ coarse prediction differs from target",
    )
    _require(
        predicted_hand120_wrist_relative_m.shape == targets.hand120_wrist_relative_m.shape,
        "DXYZ hand prediction differs from target",
    )
    _require(
        valid_mask.shape == target_global_m.shape[:2] and valid_mask.dtype == torch.bool,
        "DXYZ loss mask differs",
    )
    _require(
        bool(torch.isfinite(predicted_coarse18_m).all())
        and bool(torch.isfinite(predicted_hand120_wrist_relative_m).all()),
        "DXYZ prediction is non-finite",
    )
    coarse = masked_coordinate_l1(
        predicted_coarse18_m / coordinate_scale_metres,
        targets.coarse18_m / coordinate_scale_metres,
        valid_mask=valid_mask,
        capacity_frames=capacity_frames,
    )
    hand = masked_coordinate_l1(
        predicted_hand120_wrist_relative_m / coordinate_scale_metres,
        targets.hand120_wrist_relative_m / coordinate_scale_metres,
        valid_mask=valid_mask,
        capacity_frames=capacity_frames,
    )
    total = coarse + hand
    _require(bool(torch.isfinite(total)), "DXYZ total coordinate loss is non-finite")
    return DirectXyzCoordinateLoss(
        total=total,
        coarse_coordinate_l1=coarse,
        hand_coordinate_l1=hand,
    )
