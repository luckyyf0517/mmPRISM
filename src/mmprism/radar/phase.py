"""Receiver-phase calibration before complex antenna mixing."""

import torch
from torch import Tensor


def apply_channel_phase_compensation(raw_complex: Tensor, phase_rad: Tensor) -> Tensor:
    """Rotate every receiver by the negative of its calibrated phase.

    ``corrected[b, d, c, r] = raw[b, d, c, r] * exp(-1j * phase[c])`` with the
    ``[116]`` phase vector broadcast over batch, Doppler, and range.
    """
    if raw_complex.ndim != 4 or raw_complex.shape[2] != phase_rad.shape[0]:
        raise ValueError("phase compensation expects [batch,doppler,116,range] complex input")
    compensation = torch.exp(-1j * phase_rad.to(dtype=torch.float32)).to(dtype=raw_complex.dtype)
    return raw_complex * compensation.view(1, 1, -1, 1)
