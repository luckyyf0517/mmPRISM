"""Dual coordinate supervision, body-parameter anchoring and a hold-then-cosine schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from mmprism.models.dense_token_deterministic_mhr import (
    MhrSemanticLoss,
    MhrTargetNormalization,
    masked_semantic_mhr_smooth_l1,
)
from mmprism.training.reconstruction.dense_token_direct_xyz import (
    COORDINATE_SCALE_METRES,
    DirectXyzCoordinateLoss,
    masked_direct_xyz_coordinate_l1,
)


class DenseTokenDualSupervisionTrainingError(ValueError):
    """Raised when E2E-DUAL50's fixed loss or schedule is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DenseTokenDualSupervisionTrainingError(message)


@dataclass(frozen=True)
class DualSupervisionLoss:
    """The three unambiguous terms of the fixed dual-supervision objective."""

    total: Tensor
    direct_xyz: DirectXyzCoordinateLoss
    fk_xyz: DirectXyzCoordinateLoss
    mhr_parameter: MhrSemanticLoss


def masked_dual_supervision_loss(
    *,
    direct_coarse18_m: Tensor,
    direct_hand120_wrist_relative_m: Tensor,
    fk_decoded_full24_m: Tensor,
    predicted_mhr191: Tensor,
    target_global_m: Tensor,
    target_mhr191: Tensor,
    valid_mask: Tensor,
    capacity_frames: int,
    normalization: MhrTargetNormalization,
    direct_weight: float = 0.5,
    fk_weight: float = 0.5,
    parameter_weight: float = 0.05,
    mhr_parameter_excluded_groups: tuple[str, ...] = (),
) -> DualSupervisionLoss:
    """Apply ``0.5 direct + 0.5 FK + 0.05 MHR`` without adaptive rescaling.

    ``mhr_parameter_excluded_groups`` is for routes that retain a shared MHR
    head shape while making a named semantic group non-predictive.
    """

    _require(direct_weight == 0.5, "E2E-DUAL50 direct weight must remain 0.5")
    _require(fk_weight == 0.5, "E2E-DUAL50 FK weight must remain 0.5")
    _require(parameter_weight == 0.05, "E2E-DUAL50 MHR weight must remain 0.05")
    _require(
        fk_decoded_full24_m.shape == target_global_m.shape
        and fk_decoded_full24_m.ndim == 5
        and fk_decoded_full24_m.shape[2:] == (2, 24, 3),
        "E2E-DUAL50 FK full24 target shape differs",
    )
    direct = masked_direct_xyz_coordinate_l1(
        predicted_coarse18_m=direct_coarse18_m,
        predicted_hand120_wrist_relative_m=direct_hand120_wrist_relative_m,
        target_global_m=target_global_m,
        valid_mask=valid_mask,
        capacity_frames=capacity_frames,
        coordinate_scale_metres=COORDINATE_SCALE_METRES,
    )
    fk = masked_direct_xyz_coordinate_l1(
        predicted_coarse18_m=fk_decoded_full24_m[..., :3, :].reshape(
            *fk_decoded_full24_m.shape[:2], 18
        ),
        predicted_hand120_wrist_relative_m=(
            fk_decoded_full24_m[..., 4:, :] - fk_decoded_full24_m[..., 2:3, :]
        ).reshape(*fk_decoded_full24_m.shape[:2], 120),
        target_global_m=target_global_m,
        valid_mask=valid_mask,
        capacity_frames=capacity_frames,
        coordinate_scale_metres=COORDINATE_SCALE_METRES,
    )
    parameter = masked_semantic_mhr_smooth_l1(
        predicted_mhr191,
        target_mhr191,
        valid_mask=valid_mask,
        normalization=normalization,
        excluded_semantic_groups=mhr_parameter_excluded_groups,
    )
    total = direct_weight * direct.total + fk_weight * fk.total + parameter_weight * parameter.total
    _require(bool(total.isfinite()), "E2E-DUAL50 total loss is non-finite")
    return DualSupervisionLoss(
        total=total,
        direct_xyz=direct,
        fk_xyz=fk,
        mhr_parameter=parameter,
    )


@dataclass
class HoldCosineSchedule:
    """Optionally hold, then decay continuously to the configured LR floor."""

    completed_optimizer_steps: int = 0
    hold_steps: int = 20_000
    max_optimizer_steps: int = 50_000
    peak_learning_rate: float = 1.0e-4
    final_learning_rate: float = 1.0e-6

    def __post_init__(self) -> None:
        _require(self.hold_steps >= 0, "dual schedule hold steps must be non-negative")
        _require(
            self.max_optimizer_steps > self.hold_steps,
            "dual schedule decay interval must be positive",
        )
        _require(0 <= self.completed_optimizer_steps <= self.max_optimizer_steps, "step range")
        _require(self.peak_learning_rate > self.final_learning_rate > 0.0, "learning rates")

    def next_learning_rate(self) -> float:
        """Return the LR for the update after ``completed_optimizer_steps``."""

        step = self.completed_optimizer_steps
        if step <= self.hold_steps:
            return self.peak_learning_rate
        fraction = (step - self.hold_steps) / (self.max_optimizer_steps - self.hold_steps)
        return (
            self.final_learning_rate
            + (self.peak_learning_rate - self.final_learning_rate)
            * (1.0 + math.cos(math.pi * fraction))
            * 0.5
        )

    def step(self) -> None:
        _require(
            self.completed_optimizer_steps < self.max_optimizer_steps,
            "dual schedule cannot advance after terminal step",
        )
        self.completed_optimizer_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "completed_optimizer_steps": self.completed_optimizer_steps,
            "hold_steps": self.hold_steps,
            "max_optimizer_steps": self.max_optimizer_steps,
            "peak_learning_rate": self.peak_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "next_update_learning_rate": self.next_learning_rate(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> HoldCosineSchedule:
        try:
            schedule = cls(
                completed_optimizer_steps=int(state["completed_optimizer_steps"]),
                hold_steps=int(state["hold_steps"]),
                max_optimizer_steps=int(state["max_optimizer_steps"]),
                peak_learning_rate=float(state["peak_learning_rate"]),
                final_learning_rate=float(state["final_learning_rate"]),
            )
            expected = float(state["next_update_learning_rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise DenseTokenDualSupervisionTrainingError(
                f"dual schedule state is malformed: {error}"
            ) from error
        _require(
            abs(schedule.next_learning_rate() - expected) <= 1.0e-15,
            "dual schedule resume next-learning-rate differs",
        )
        return schedule
