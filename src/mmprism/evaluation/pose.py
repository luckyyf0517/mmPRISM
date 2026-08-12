from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

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


@dataclass(slots=True)
class PoseMetricAccumulator:
    """Streaming, count-weighted implementation of the pose metric protocol."""

    pck_threshold_mm: float = 40.0
    sample_count: int = field(default=0, init=False)
    _coordinate_l1_sum: float = field(default=0.0, init=False)
    _coordinate_count: int = field(default=0, init=False)
    _absolute_distance_sum_mm: float = field(default=0.0, init=False)
    _absolute_joint_count: int = field(default=0, init=False)
    _relative_distance_sum_mm: float = field(default=0.0, init=False)
    _relative_joint_count: int = field(default=0, init=False)
    _relative_correct_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.pck_threshold_mm <= 0:
            raise ValueError("pck_threshold_mm must be positive")

    def update(self, prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
        """Accumulate one batch and return per-sample absolute MPJPE in millimetres."""

        valid = _validate_pose_inputs(prediction, target, valid_mask)
        root_valid = valid[:, :, HAND_WRIST_INDEX]
        finger_valid = valid[:, :, FINGER_JOINT_START_INDEX:] & root_valid.unsqueeze(-1)
        if not bool(torch.any(finger_valid)):
            raise ValueError("wrist-relative PCK requires a valid wrist and finger joint")

        coordinate_error = torch.abs(prediction - target)
        distances_mm = torch.linalg.vector_norm(prediction - target, dim=-1) * 1000.0
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

        valid_joint_count = int(valid.sum().item())
        relative_joint_count = int(finger_valid.sum().item())
        self._coordinate_l1_sum += float(coordinate_error[valid].sum().item())
        self._coordinate_count += valid_joint_count * prediction.shape[-1]
        self._absolute_distance_sum_mm += float(distances_mm[valid].sum().item())
        self._absolute_joint_count += valid_joint_count
        self._relative_distance_sum_mm += float(relative_distances_mm[finger_valid].sum().item())
        self._relative_joint_count += relative_joint_count
        self._relative_correct_count += int(
            (relative_distances_mm[finger_valid] <= self.pck_threshold_mm).sum().item()
        )
        self.sample_count += prediction.shape[0]

        per_sample = [
            distances_mm[index][valid[index]].mean() for index in range(prediction.shape[0])
        ]
        return torch.stack(per_sample).detach()

    def values(self) -> dict[str, float]:
        if (
            self.sample_count == 0
            or self._coordinate_count == 0
            or self._absolute_joint_count == 0
            or self._relative_joint_count == 0
        ):
            raise ValueError("pose metric accumulator contains no complete samples")
        return {
            "masked_pose_l1_metres": self._coordinate_l1_sum / self._coordinate_count,
            "absolute_mpjpe_mm": self._absolute_distance_sum_mm / self._absolute_joint_count,
            "root_relative_mpjpe_mm": (self._relative_distance_sum_mm / self._relative_joint_count),
            "root_relative_pck": self._relative_correct_count / self._relative_joint_count,
        }

    def state_dict(self) -> dict[str, int | float]:
        return {
            "pck_threshold_mm": self.pck_threshold_mm,
            "sample_count": self.sample_count,
            "coordinate_l1_sum": self._coordinate_l1_sum,
            "coordinate_count": self._coordinate_count,
            "absolute_distance_sum_mm": self._absolute_distance_sum_mm,
            "absolute_joint_count": self._absolute_joint_count,
            "relative_distance_sum_mm": self._relative_distance_sum_mm,
            "relative_joint_count": self._relative_joint_count,
            "relative_correct_count": self._relative_correct_count,
        }

    def merge_state(self, state: Mapping[str, int | float]) -> None:
        expected = {
            "pck_threshold_mm",
            "sample_count",
            "coordinate_l1_sum",
            "coordinate_count",
            "absolute_distance_sum_mm",
            "absolute_joint_count",
            "relative_distance_sum_mm",
            "relative_joint_count",
            "relative_correct_count",
        }
        if set(state) != expected or float(state["pck_threshold_mm"]) != self.pck_threshold_mm:
            raise ValueError("pose metric state does not match the accumulator protocol")
        integer_fields = (
            "sample_count",
            "coordinate_count",
            "absolute_joint_count",
            "relative_joint_count",
            "relative_correct_count",
        )
        if any(
            isinstance(state[name], bool)
            or not isinstance(state[name], int)
            or state[name] < 0
            for name in integer_fields
        ):
            raise ValueError("pose metric state contains an invalid count")
        self.sample_count += int(state["sample_count"])
        self._coordinate_l1_sum += float(state["coordinate_l1_sum"])
        self._coordinate_count += int(state["coordinate_count"])
        self._absolute_distance_sum_mm += float(state["absolute_distance_sum_mm"])
        self._absolute_joint_count += int(state["absolute_joint_count"])
        self._relative_distance_sum_mm += float(state["relative_distance_sum_mm"])
        self._relative_joint_count += int(state["relative_joint_count"])
        self._relative_correct_count += int(state["relative_correct_count"])
