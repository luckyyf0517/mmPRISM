from __future__ import annotations

import numpy as np
import pytest

from mmprism.radar.processing import (
    RadarProcessingError,
    RangeDopplerConfig,
    range_doppler_transform,
)


def _tone(
    *,
    chirps: int = 8,
    antennas: int = 2,
    samples: int = 16,
    doppler_bin: int = 2,
    range_bin: int = 3,
    dtype: type[np.complex64] | type[np.complex128] = np.complex64,
) -> np.ndarray:
    slow_time = np.arange(chirps, dtype=np.float64)[:, None, None]
    fast_time = np.arange(samples, dtype=np.float64)[None, None, :]
    phase = 2.0 * np.pi * (
        doppler_bin * slow_time / chirps + range_bin * fast_time / samples
    )
    signal = np.exp(1j * phase)
    return np.repeat(signal, antennas, axis=1).astype(dtype)


def _rectangular_config(**overrides: object) -> RangeDopplerConfig:
    values: dict[str, object] = {
        "range_window": "rectangular",
        "doppler_window": "rectangular",
        "subtract_slow_time_mean": False,
    }
    values.update(overrides)
    return RangeDopplerConfig(**values)  # type: ignore[arg-type]


def test_analytic_tone_peaks_at_exact_range_and_shifted_doppler_bins() -> None:
    raw = _tone(doppler_bin=2, range_bin=3)

    spectrum = range_doppler_transform(raw, _rectangular_config())

    assert spectrum.shape == (8, 2, 16)
    assert spectrum.dtype == np.complex64
    peak = np.unravel_index(np.abs(spectrum[:, 0, :]).argmax(), (8, 16))
    assert peak == (6, 3)
    assert np.abs(spectrum[6, 0, 3]) == pytest.approx(8 * 16)


def test_static_slow_time_signal_is_removed_before_doppler_fft() -> None:
    raw = _tone(doppler_bin=0, range_bin=4)
    config = RangeDopplerConfig(
        range_window="rectangular",
        doppler_window="rectangular",
        subtract_slow_time_mean=True,
    )

    spectrum = range_doppler_transform(raw, config)

    np.testing.assert_allclose(spectrum, 0.0, atol=1e-5)


def test_transform_supports_explicit_batch_and_time_axes_without_mutation() -> None:
    frame = _tone()
    raw = np.broadcast_to(frame, (2, 3, *frame.shape)).copy()
    original = raw.copy()

    spectrum = range_doppler_transform(
        raw,
        _rectangular_config(range_bin_start=2, range_bin_count=5),
        leading_axes=("batch", "time"),
    )

    assert spectrum.shape == (2, 3, 8, 2, 5)
    assert spectrum.flags.c_contiguous
    np.testing.assert_array_equal(raw, original)


def test_zero_padding_and_periodic_hann_preserve_complex_precision() -> None:
    raw = _tone(dtype=np.complex128)
    config = RangeDopplerConfig(
        range_fft_size=32,
        doppler_fft_size=16,
        range_bin_start=1,
        range_bin_count=20,
    )

    spectrum = range_doppler_transform(raw, config)

    assert spectrum.shape == (16, 2, 20)
    assert spectrum.dtype == np.complex128
    assert np.all(np.isfinite(spectrum))


@pytest.mark.parametrize(
    "config",
    [
        RangeDopplerConfig(range_fft_size=8),
        RangeDopplerConfig(doppler_fft_size=4),
        RangeDopplerConfig(range_bin_start=16),
        RangeDopplerConfig(range_bin_start=12, range_bin_count=5),
    ],
)
def test_transform_rejects_truncation_and_out_of_bounds_range_bins(
    config: RangeDopplerConfig,
) -> None:
    with pytest.raises(RadarProcessingError):
        range_doppler_transform(_tone(), config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"range_fft_size": 0},
        {"doppler_fft_size": True},
        {"range_bin_start": -1},
        {"range_bin_count": 0},
        {"range_window": "blackman"},
        {"doppler_window": "hann"},
        {"subtract_slow_time_mean": 1},
        {"protocol_version": "mmprism.range_doppler.v0"},
    ],
)
def test_config_rejects_implicit_or_invalid_processing_choices(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(RadarProcessingError):
        RangeDopplerConfig(**kwargs)  # type: ignore[arg-type]


def test_transform_wraps_input_contract_failures() -> None:
    with pytest.raises(RadarProcessingError, match="complex dtype"):
        range_doppler_transform(
            np.zeros((8, 2, 16), dtype=np.float32), RangeDopplerConfig()
        )

    raw = _tone()
    raw[0, 0, 0] = complex(np.nan, 0.0)
    with pytest.raises(RadarProcessingError, match="finite"):
        range_doppler_transform(raw, RangeDopplerConfig())
