"""Incoherent spectral features added to the radar-token projection."""

from __future__ import annotations

import torch
from torch import Tensor, nn

SLOW_TIME_AXIS = 1

ANTENNA_AXIS = 2

HIDDEN = 384


class IncoherentSpectralBranch(nn.Module):
    """Per-antenna slow-time magnitude, incoherently averaged, zero-init residual."""

    def __init__(self, hidden: int = HIDDEN, pyramid: int = 4) -> None:
        super().__init__()
        self.pyramid = pyramid
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((pyramid, pyramid)),
            nn.Flatten(),
            nn.Linear(128 * pyramid * pyramid, hidden),
            nn.GELU(),
        )
        self.residual = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def range_doppler_map(self, raw_radar: Tensor) -> Tensor:
        """[B, T, 64, 116, 32, 2] -> log-magnitude [B*T, 64, 32].

        Axis 2 of the input is the 116-antenna axis and axis 1 is the
        64-chirp slow-time axis (matching the delivered raw layout).  Each
        antenna is transformed along slow time, the magnitude is taken per
        antenna, and the antennas are averaged incoherently.
        """

        if raw_radar.ndim != 6 or tuple(raw_radar.shape[2:]) != (64, 116, 32, 2):
            raise ValueError(f"raw radar must be [B,T,64,116,32,2], got {tuple(raw_radar.shape)}")
        frames = raw_radar.shape[0] * raw_radar.shape[1]
        flat = raw_radar.reshape(frames, 64, 116, 32, 2)
        complex_raw = torch.complex(flat[..., 0], flat[..., 1])
        per_antenna = torch.fft.fft(complex_raw, dim=SLOW_TIME_AXIS).abs()
        averaged = per_antenna.mean(dim=ANTENNA_AXIS)
        return torch.log1p(averaged)

    def forward(self, raw_radar: Tensor) -> Tensor:
        """[B, T, ...] raw radar -> zero-init residual context [B, T, hidden]."""

        maps = self.range_doppler_map(raw_radar)
        context = self.trunk(maps[:, None, :, :])
        frames = raw_radar.shape[0] * raw_radar.shape[1]
        return (
            self.residual(context).reshape(raw_radar.shape[0], raw_radar.shape[1], self.hidden)
            if context.shape[0] == frames
            else self.residual(context)[:frames].reshape(
                raw_radar.shape[0], raw_radar.shape[1], self.hidden
            )
        )
