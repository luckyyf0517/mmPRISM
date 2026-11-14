"""Background transfer and amplitude calibration for mixed reconstruction training.

The mixed phase combines measured and simulated radar windows. Simulated echoes
receive measured background calibration:

    z' = g * (z_sim - B_sim) + alpha * B_real          (alpha = 0.7)

Here B is a window's complex temporal median and g is the amplitude median
ratio. Background donors come from the measured training pool.

Radar arrays use [frame, 64 chirps, 116 channels, 32 range] complex values,
combining the separate real and imaginary components of the model input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

FRAME_AXIS: Final[int] = 0

BACKGROUND_ALPHA: Final[float] = 0.7

GAIN_BAND: Final[tuple[float, float]] = (0.1, 10.0)

SCALE_BAND: Final[tuple[float, float]] = (1e2, 1e4)

SCALE_RATIO_BAND: Final[tuple[float, float]] = SCALE_BAND


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _median(values: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Median with numpy semantics (mean of the two middle values for an even count).

    ``torch.median`` returns the lower middle value; ``kthvalue`` keeps the cost
    linear, on the CPU and on the GPU alike, and has no input-size limit.
    """

    if dim is None:
        flat = values.reshape(-1)
        count = int(flat.numel())
        upper = torch.kthvalue(flat, count // 2 + 1).values
        if count % 2:
            return upper
        return (torch.kthvalue(flat, count // 2).values + upper) / 2
    count = int(values.shape[dim])
    upper = torch.kthvalue(values, count // 2 + 1, dim=dim).values
    if count % 2:
        return upper
    return (torch.kthvalue(values, count // 2, dim=dim).values + upper) / 2


def _background_t(pooled: torch.Tensor) -> torch.Tensor:
    return torch.complex(_median(pooled.real, FRAME_AXIS), _median(pooled.imag, FRAME_AXIS))


def _scale_t(pooled: torch.Tensor) -> float:
    scale = float(torch.sqrt(torch.abs(pooled).double().square().mean()).item())
    _require(
        math.isfinite(scale) and scale > 0.0,
        "pooled amplitude scale is not positive and finite",
    )
    return scale


@dataclass(frozen=True)
class GainFit:
    """The two forms of the fitted amplitude gain, the scales they bridge, and the guard."""

    gain: float
    normalised_gain: float
    real_scale: float
    sim_scale: float

    @property
    def scale_ratio(self) -> float:
        """Pooled-RMS scale factor between the two sides, always >= 1."""

        return max(self.real_scale, self.sim_scale) / min(self.real_scale, self.sim_scale)

    def as_dict(self) -> dict[str, object]:
        return {
            "gain": float(self.gain),
            "gain_normalised": float(self.normalised_gain),
            "gain_normalised_reported_band": list(GAIN_BAND),
            "gain_normalised_is_a_guard": False,
            "scale_ratio_guard": list(SCALE_RATIO_BAND),
            "scale_ratio": float(self.scale_ratio),
            "sim_over_real_scale": float(self.sim_scale / self.real_scale),
            "real_scale_rms": float(self.real_scale),
            "sim_scale_rms": float(self.sim_scale),
        }


@dataclass(frozen=True)
class DonorStats:
    """Everything the calibration needs from a donor pool, computed once.

    The donor side of the transform only ever uses the complex time-median
    background and two scalars (the median magnitude and the pooled RMS), so a
    worker process can reduce a ~180 MB donor pool to these and ship 2 MB.
    """

    background: np.ndarray
    median_abs: float
    scale: float
    frames: int


def donor_statistics(
    raw_donor: np.ndarray, *, device: torch.device | str | None = None
) -> DonorStats:
    """Donor statistics of a ``[frame, 64, 116, 32, 2]`` float32 I/Q donor pool."""

    _require(raw_donor.shape[-1] == 2, "donor statistics need I/Q")
    if device is None or str(device) == "cpu":
        return _donor_statistics_numpy(raw_donor)
    donor_iq = torch.as_tensor(np.ascontiguousarray(raw_donor), device=device or "cpu")
    donor = torch.complex(donor_iq[..., 0], donor_iq[..., 1])
    return DonorStats(
        background=_background_t(donor).cpu().numpy().astype(np.complex64),
        median_abs=float(_median(torch.abs(donor)).item()),
        scale=_scale_t(donor),
        frames=int(raw_donor.shape[0]),
    )


def _numpy_median_all(values: np.ndarray) -> float:
    """Median of every element, numpy semantics, by partition (no full sort)."""

    flat = values.reshape(-1)
    count = flat.size
    half = count // 2
    if count % 2:
        return float(np.partition(flat, half)[half])
    part = np.partition(flat, [half - 1, half])
    return float((part[half - 1] + part[half]) / 2)


def _donor_statistics_numpy(raw_donor: np.ndarray) -> DonorStats:
    """CPU form of :func:`donor_statistics` for data-loader workers (one thread).

    The same three quantities as the torch path: complex time-median background
    (real and imaginary parts separately), median magnitude, pooled RMS.  Torch's
    CPU ``kthvalue`` takes about 4 s on a 96-frame donor in one thread; this takes
    about 1.8 s.
    """

    iq = np.asarray(raw_donor, dtype=np.float32)
    frames = iq.shape[0]
    real = np.ascontiguousarray(iq[..., 0].reshape(frames, -1).T)
    imag = np.ascontiguousarray(iq[..., 1].reshape(frames, -1).T)
    background = (np.median(real, axis=1) + 1j * np.median(imag, axis=1)).astype(np.complex64)
    magnitude = np.hypot(iq[..., 0], iq[..., 1])
    scale = float(np.sqrt(np.mean(np.square(magnitude, dtype=np.float64))))
    _require(
        math.isfinite(scale) and scale > 0.0,
        "pooled amplitude scale is not positive and finite",
    )
    return DonorStats(
        background=background.reshape(iq.shape[1:-1]),
        median_abs=_numpy_median_all(magnitude),
        scale=scale,
        frames=int(frames),
    )


def _fit_gain_from_stats(*, real_median: float, real_scale: float, sim: torch.Tensor) -> GainFit:
    """The gain fit from the real side's two scalars and the simulated pool: the
    single implementation every entry point goes through."""

    sim_median = float(_median(torch.abs(sim)).item())
    sim_scale = _scale_t(sim)
    scale_ratio = sim_scale / real_scale
    scale_low, scale_high = SCALE_BAND
    _require(
        math.isfinite(scale_ratio) and scale_low <= scale_ratio <= scale_high,
        f"simulated/real RMS ratio {scale_ratio:.4g} is outside the absolute scale band "
        f"[{scale_low:g}, {scale_high:g}]: the two sides are not on the same "
        "preprocessing scale",
    )
    _require(real_median > 0.0 and sim_median > 0.0, "a gain-fit pool has a zero amplitude median")
    normalised = (real_median / real_scale) / (sim_median / sim_scale)
    _require(
        math.isfinite(normalised) and normalised > 0.0,
        f"fitted normalised gain {normalised!r} is not positive and finite",
    )
    gain = normalised * real_scale / sim_scale
    _require(math.isfinite(gain) and gain > 0.0, "calibration gain is not positive and finite")
    return GainFit(
        gain=float(gain),
        normalised_gain=float(normalised),
        real_scale=float(real_scale),
        sim_scale=float(sim_scale),
    )


def calibrate_iq(
    raw_sim: np.ndarray,
    raw_donor: np.ndarray,
    *,
    alpha: float = BACKGROUND_ALPHA,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Calibrate one simulated window against one donor window on ``device``.

    Inputs and output are the ``[frame, 64, 116, 32, 2]`` float32 I/Q layout the
    trainer streams.  Everything between the two copies runs on ``device`` with the
    same functions the numpy entry points use, so the trained system and the
    measured system share one implementation.
    """

    _require(raw_sim.shape[-1] == 2 and raw_donor.shape[-1] == 2, "calibration needs I/Q")
    return calibrate_iq_with_donor_stats(
        raw_sim, donor_statistics(raw_donor, device=device), alpha=alpha, device=device
    )


def calibrate_iq_with_donor_stats(
    raw_sim: np.ndarray,
    donor: DonorStats,
    *,
    alpha: float = BACKGROUND_ALPHA,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Calibrate one simulated window against precomputed donor statistics.

    Identical in value to :func:`calibrate_iq` on the donor pool the statistics
    came from; it exists so the donor reduction can run in a worker process.
    """

    _require(raw_sim.shape[-1] == 2, "calibration needs I/Q")
    _require(0.0 <= alpha <= 2.0, "background alpha is outside the physical band")
    sim_iq = torch.as_tensor(np.ascontiguousarray(raw_sim), device=device or "cpu")
    sim = torch.complex(sim_iq[..., 0], sim_iq[..., 1])
    fit = _fit_gain_from_stats(real_median=donor.median_abs, real_scale=donor.scale, sim=sim)
    donor_background = torch.as_tensor(np.ascontiguousarray(donor.background), device=sim.device)
    calibrated = fit.gain * (sim - _background_t(sim)[None]) + alpha * donor_background[None]
    stacked = torch.stack((calibrated.real, calibrated.imag), dim=-1).float()
    return stacked.cpu().numpy(), {
        "gain": float(fit.gain),
        "gain_normalised": float(fit.normalised_gain),
        "alpha": float(alpha),
        "donor_frames": int(donor.frames),
    }
