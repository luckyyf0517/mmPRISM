"""Typed direct mesh-surface paths consumed by the FMCW frontend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class FmcwPathBatchError(ValueError):
    """Raised when a geometry backend emits an invalid fixed-shape path batch."""


@dataclass(frozen=True, slots=True)
class FmcwPathBatch:
    """Per-sample, per-receiver path parameters consumed by one FMCW frontend.

    All path tensors use ``[batch, receiver, sample]`` layout.  ``sample_mask``
    is optional diagnostic structure; masked samples must already have zero
    amplitude so the synthesis layer does not need backend-specific masking.
    """

    amplitude: Tensor
    delay_s: Tensor
    radial_mps: Tensor
    sample_mask: Tensor | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.amplitude.shape)
        if self.amplitude.ndim != 3 or any(size < 1 for size in shape):
            raise FmcwPathBatchError(
                f"amplitude must have non-empty shape [batch, receiver, sample], got {shape}"
            )
        if (
            self.delay_s.shape != self.amplitude.shape
            or self.radial_mps.shape != self.amplitude.shape
        ):
            raise FmcwPathBatchError(
                "amplitude, delay_s, and radial_mps must have identical shapes"
            )
        values = (self.amplitude, self.delay_s, self.radial_mps)
        if not all(value.is_floating_point() for value in values):
            raise FmcwPathBatchError("path tensors must be floating point")
        if len({value.device for value in values}) != 1:
            raise FmcwPathBatchError("path tensors must share a device")
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise FmcwPathBatchError("path tensors must be finite")
        if bool((self.amplitude < 0.0).any()):
            raise FmcwPathBatchError("amplitude must be non-negative")
        if bool((self.delay_s < 0.0).any()):
            raise FmcwPathBatchError("delay_s must be non-negative")
        if self.sample_mask is None:
            return
        expected_mask_shape = (self.amplitude.shape[0], self.amplitude.shape[2])
        if self.sample_mask.shape != expected_mask_shape or self.sample_mask.dtype != torch.bool:
            raise FmcwPathBatchError(
                "sample_mask must have boolean shape "
                f"{expected_mask_shape}, got {tuple(self.sample_mask.shape)} "
                f"{self.sample_mask.dtype}"
            )
        if self.sample_mask.device != self.amplitude.device:
            raise FmcwPathBatchError("sample_mask must share the path tensor device")
        if bool(self.amplitude.masked_select(~self.sample_mask[:, None]).any()):
            raise FmcwPathBatchError("masked path samples must have zero amplitude")

    @property
    def batch_size(self) -> int:
        """Number of independently synthesized frames."""

        return int(self.amplitude.shape[0])

    @property
    def receiver_count(self) -> int:
        """Number of virtual receiver elements."""

        return int(self.amplitude.shape[1])

    @property
    def sample_count(self) -> int:
        """Number of fixed path slots per sample."""

        return int(self.amplitude.shape[2])

    def slice(self, index: int) -> FmcwPathBatch:
        """One sample's path view (views, no copy) for per-pair synthesis."""

        if not 0 <= index < self.amplitude.shape[0]:
            raise FmcwPathBatchError(
                f"slice index {index} out of range [0, {self.amplitude.shape[0]})"
            )
        return FmcwPathBatch(
            self.amplitude[index : index + 1],
            self.delay_s[index : index + 1],
            self.radial_mps[index : index + 1],
            None if self.sample_mask is None else self.sample_mask[index : index + 1],
        )
