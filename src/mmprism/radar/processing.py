from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mmprism.contracts.tensors import (
    LeadingAxis,
    TensorContractError,
    validate_range_doppler,
    validate_raw_radar,
)

WindowName = Literal["rectangular", "hann_periodic"]
_WINDOW_NAMES = {"rectangular", "hann_periodic"}
RANGE_DOPPLER_PROTOCOL_V1 = "mmprism.range_doppler.v1"


class RadarProcessingError(ValueError):
    """Raised when range-Doppler processing configuration is invalid."""


@dataclass(frozen=True, slots=True)
class RangeDopplerConfig:
    """Explicit FFT, window, clutter-removal, and range-selection settings."""

    protocol_version: str = RANGE_DOPPLER_PROTOCOL_V1
    range_fft_size: int | None = None
    doppler_fft_size: int | None = None
    range_bin_start: int = 0
    range_bin_count: int | None = None
    range_window: WindowName = "hann_periodic"
    doppler_window: WindowName = "hann_periodic"
    subtract_slow_time_mean: bool = True

    def __post_init__(self) -> None:
        if self.protocol_version != RANGE_DOPPLER_PROTOCOL_V1:
            raise RadarProcessingError(
                f"unsupported range-Doppler protocol: {self.protocol_version!r}"
            )
        for name, value in (
            ("range_fft_size", self.range_fft_size),
            ("doppler_fft_size", self.doppler_fft_size),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise RadarProcessingError(f"{name} must be a positive integer or None")
        if (
            not isinstance(self.range_bin_start, int)
            or isinstance(self.range_bin_start, bool)
            or self.range_bin_start < 0
        ):
            raise RadarProcessingError("range_bin_start must be a non-negative integer")
        if self.range_bin_count is not None and (
            not isinstance(self.range_bin_count, int)
            or isinstance(self.range_bin_count, bool)
            or self.range_bin_count <= 0
        ):
            raise RadarProcessingError("range_bin_count must be a positive integer or None")
        if self.range_window not in _WINDOW_NAMES:
            raise RadarProcessingError(f"unsupported range_window: {self.range_window!r}")
        if self.doppler_window not in _WINDOW_NAMES:
            raise RadarProcessingError(f"unsupported doppler_window: {self.doppler_window!r}")
        if not isinstance(self.subtract_slow_time_mean, bool):
            raise RadarProcessingError("subtract_slow_time_mean must be a boolean")


def _window(name: WindowName, size: int, dtype: np.dtype[Any]) -> np.ndarray:
    if name == "rectangular":
        return np.ones(size, dtype=dtype)
    # np.hanning(size + 1)[:-1] matches the periodic Hann convention used by FFT APIs.
    return np.hanning(size + 1)[:-1].astype(dtype, copy=False)


def _fft_size(configured: int | None, input_size: int, name: str) -> int:
    size = input_size if configured is None else configured
    if size < input_size:
        raise RadarProcessingError(
            f"{name} ({size}) must not truncate its input dimension ({input_size})"
        )
    return size


def range_doppler_transform(
    raw_radar: np.ndarray,
    config: RangeDopplerConfig,
    *,
    leading_axes: tuple[LeadingAxis, ...] = (),
) -> np.ndarray:
    """Transform ``[..., chirp, antenna, sample]`` ADC data into range-Doppler spectra.

    Range FFT is applied first. Optional static-clutter removal subtracts the complex
    slow-time mean before the Doppler window and FFT. The Doppler axis is centered with
    ``fftshift``. The input is never modified and complex precision is preserved.
    """

    try:
        validate_raw_radar(raw_radar, leading_axes=leading_axes)
    except TensorContractError as error:
        raise RadarProcessingError(str(error)) from error

    chirp_axis = len(leading_axes)
    sample_axis = raw_radar.ndim - 1
    chirp_count = raw_radar.shape[chirp_axis]
    sample_count = raw_radar.shape[sample_axis]
    range_fft_size = _fft_size(config.range_fft_size, sample_count, "range_fft_size")
    doppler_fft_size = _fft_size(
        config.doppler_fft_size, chirp_count, "doppler_fft_size"
    )
    if config.range_bin_start >= range_fft_size:
        raise RadarProcessingError(
            "range_bin_start must be smaller than the resolved range FFT size"
        )
    range_stop = (
        range_fft_size
        if config.range_bin_count is None
        else config.range_bin_start + config.range_bin_count
    )
    if range_stop > range_fft_size:
        raise RadarProcessingError(
            "requested range bins exceed the resolved range FFT size: "
            f"stop={range_stop}, size={range_fft_size}"
        )

    real_dtype = np.dtype(np.float32 if raw_radar.dtype == np.complex64 else np.float64)
    range_window = _window(config.range_window, sample_count, real_dtype)
    range_spectrum = np.fft.fft(
        raw_radar * range_window,
        n=range_fft_size,
        axis=sample_axis,
    ).astype(raw_radar.dtype, copy=False)

    if config.subtract_slow_time_mean:
        range_spectrum = range_spectrum - range_spectrum.mean(
            axis=chirp_axis, keepdims=True
        )
    doppler_window = _window(config.doppler_window, chirp_count, real_dtype)
    window_shape = [1] * range_spectrum.ndim
    window_shape[chirp_axis] = chirp_count
    windowed = range_spectrum * doppler_window.reshape(window_shape)
    doppler_spectrum = np.fft.fft(
        windowed,
        n=doppler_fft_size,
        axis=chirp_axis,
    ).astype(raw_radar.dtype, copy=False)
    centered = np.fft.fftshift(doppler_spectrum, axes=chirp_axis)
    output = np.ascontiguousarray(centered[..., config.range_bin_start : range_stop])

    try:
        validate_range_doppler(output, leading_axes=leading_axes)
    except TensorContractError as error:
        raise RadarProcessingError(str(error)) from error
    return output
