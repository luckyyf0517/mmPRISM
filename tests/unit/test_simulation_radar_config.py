from __future__ import annotations

import pytest

from mmprism.simulation.radar_config import (
    IWR1843_SIM_V1,
    RADAR_CONFIG_REGISTRY,
    RadarConfig,
    RadarConfigError,
    get_radar_config,
)


def test_iwr1843_sim_v1_matches_legacy_derived_values() -> None:
    config = get_radar_config(IWR1843_SIM_V1)

    assert config.start_freq == 77e9
    assert config.freq_slope == 70e12
    assert config.adc_sample_rate == 5.209e6
    assert config.num_adc_samples == 256
    assert config.num_chirps == 64
    assert config.chirp_period == pytest.approx(355e-6)
    assert config.frame_period == 5e-2

    # Values computed by the legacy config/radar/iwr1843.py module.
    assert config.range_resolution == pytest.approx(0.04285714285714286)
    assert config.doppler_resolution == pytest.approx(0.08574172306566674)
    assert config.max_range == pytest.approx(10.971428571428572)
    assert config.max_doppler == pytest.approx(2.7437351381013357)
    assert config.bandwidth == pytest.approx(3.5e9)
    assert config.sim_sample_rate == config.adc_sample_rate


def test_registry_is_immutable_snapshot_of_known_ids() -> None:
    assert set(RADAR_CONFIG_REGISTRY) == {IWR1843_SIM_V1}
    assert RADAR_CONFIG_REGISTRY[IWR1843_SIM_V1] is get_radar_config(IWR1843_SIM_V1)


def test_unknown_radar_config_id_raises() -> None:
    with pytest.raises(RadarConfigError, match="unknown radar_config_id"):
        get_radar_config("not_a_radar")


def test_config_is_frozen() -> None:
    config = get_radar_config(IWR1843_SIM_V1)
    with pytest.raises(AttributeError):
        config.start_freq = 1.0  # type: ignore[misc]


def test_timing_validation_rejects_infeasible_frontend() -> None:
    with pytest.raises(RadarConfigError, match="ADC sampling time exceeds"):
        RadarConfig(adc_start_time=50e-6)
    with pytest.raises(RadarConfigError, match="Duty cycle exceeds"):
        RadarConfig(frame_period=1e-3)


def test_field_validation_rejects_non_positive_values() -> None:
    with pytest.raises(RadarConfigError):
        RadarConfig(num_adc_samples=0)
    with pytest.raises(RadarConfigError):
        RadarConfig(freq_slope=-1.0)
