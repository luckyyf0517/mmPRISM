from __future__ import annotations

import torch
from torch import Tensor

POSE_METRIC_PROTOCOL = "mmprism.pose_metric.dual_hand_metric_v1"
HAND_WRIST_INDEX = 3
FINGER_JOINT_START_INDEX = 4


def _validate_pose_inputs(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share shape [batch,2,24,3]")
    if prediction.shape[1:] != (2, 24, 3):
        raise ValueError("pose trailing shape must be [2,24,3]")
    if valid_mask.shape != prediction.shape[:-1]:
        raise ValueError("valid_mask must have shape [batch,2,24]")
    valid = valid_mask.to(device=prediction.device, dtype=torch.bool)
    if bool(torch.any(valid.flatten(start_dim=1).sum(dim=1) == 0)):
        raise ValueError("every pose sample requires at least one valid joint")
    if not bool(torch.all(torch.isfinite(prediction))):
        raise ValueError("pose prediction must be finite")
    selected_target = target[valid]
    if not bool(torch.all(torch.isfinite(selected_target))):
        raise ValueError("valid pose targets must be finite")
    return valid


def masked_pose_l1_metres(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Mean coordinate L1 error over valid metric joints."""

    valid = _validate_pose_inputs(prediction, target, valid_mask)
    return torch.mean(torch.abs(prediction[valid] - target[valid]))


def pose_metric_tensors(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    pck_threshold_mm: float = 40.0,
) -> dict[str, Tensor]:
    """Return explicit absolute and wrist-relative metrics for metric poses."""

    if pck_threshold_mm <= 0:
        raise ValueError("pck_threshold_mm must be positive")
    valid = _validate_pose_inputs(prediction, target, valid_mask)
    distances_mm = torch.linalg.vector_norm(prediction - target, dim=-1) * 1000.0
    absolute_mpjpe_mm = distances_mm[valid].mean()

    root_valid = valid[:, :, HAND_WRIST_INDEX]
    finger_valid = valid[:, :, FINGER_JOINT_START_INDEX:] & root_valid.unsqueeze(-1)
    if not bool(torch.any(finger_valid)):
        raise ValueError("wrist-relative PCK requires a valid wrist and finger joint")
    prediction_relative = (
        prediction[:, :, FINGER_JOINT_START_INDEX:]
        - prediction[:, :, HAND_WRIST_INDEX : HAND_WRIST_INDEX + 1]
    )
    target_relative = (
        target[:, :, FINGER_JOINT_START_INDEX:]
        - target[:, :, HAND_WRIST_INDEX : HAND_WRIST_INDEX + 1]
    )
    relative_distances_mm = (
        torch.linalg.vector_norm(prediction_relative - target_relative, dim=-1) * 1000.0
    )
    root_relative_mpjpe_mm = relative_distances_mm[finger_valid].mean()
    root_relative_pck = (relative_distances_mm[finger_valid] <= pck_threshold_mm).float().mean()

    per_sample: list[Tensor] = []
    for sample_index in range(prediction.shape[0]):
        sample_valid = valid[sample_index]
        if bool(torch.any(sample_valid)):
            per_sample.append(distances_mm[sample_index][sample_valid].mean())
        else:
            per_sample.append(torch.full((), float("nan"), device=prediction.device))
    return {
        "absolute_mpjpe_mm": absolute_mpjpe_mm,
        "root_relative_mpjpe_mm": root_relative_mpjpe_mm,
        "root_relative_pck": root_relative_pck,
        "per_sample_absolute_mpjpe_mm": torch.stack(per_sample),
    }
