"""Radar frontend configuration for the point-reflector FMCW simulator.

This is a typed, validated port of the legacy ``config/radar/iwr1843.py``
module-level constants. Only the physical frontend parameters are stored;
derived quantities (bandwidth, resolutions, duty cycle) are exposed as
properties so they can never drift from the source parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

IWR1843_SIM_V1 = "iwr1843_sim_v1"


class RadarConfigError(ValueError):
    """Raised when a radar configuration is invalid or unknown."""


@dataclass(frozen=True, slots=True)
class RadarConfig:
    """IWR1843 FMCW frontend parameters used by the legacy simulator.

    Attributes use SI units (Hz, s, m/s). ``light_speed`` is the rounded value
    ``3e8`` exactly as in the legacy config; it feeds only the derived
    resolution/range properties below. The simulator core itself uses the
    exact constant ``2.99792458e8`` (see :mod:`mmprism.simulation.simulator`),
    matching the legacy hard-coded behaviour.
    """

    light_speed: float = 3e8
    start_freq: float = 77e9
    freq_slope: float = 70e12
    adc_sample_rate: float = 5.209e6
    num_adc_samples: int = 256
    idle_time: float = 300e-6
    adc_start_time: float = 5e-6
    ramp_end_time: float = 55e-6
    num_chirps: int = 64
    frame_period: float = 5e-2

    def __post_init__(self) -> None:
        for name in ("num_adc_samples", "num_chirps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RadarConfigError(f"{name} must be a positive integer")
        for name in (
            "light_speed",
            "start_freq",
            "freq_slope",
            "adc_sample_rate",
            "idle_time",
            "adc_start_time",
            "ramp_end_time",
            "frame_period",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise RadarConfigError(f"{name} must be a positive number")
        # Legacy timing feasibility asserts (config/radar/iwr1843.py).
        if not self.adc_sample_end_time < self.ramp_end_time:
            raise RadarConfigError("ADC sampling time exceeds chirp duration")
        if not self.chirp_last_period <= self.frame_period:
            raise RadarConfigError(
                f"Duty cycle exceeds frame period: {self.chirp_last_period} >= "
                f"{self.frame_period}"
            )

    @property
    def wave_length(self) -> float:
        """Carrier wavelength in metres (rounded light speed, as legacy)."""
        return self.light_speed / self.start_freq

    @property
    def chirp_period(self) -> float:
        """Chirp repetition interval: idle time plus ramp end time."""
        return self.idle_time + self.ramp_end_time

    @property
    def chirp_last_period(self) -> float:
        """Total slow-time observation span of one frame."""
        return self.chirp_period * self.num_chirps

    @property
    def adc_sample_end_time(self) -> float:
        """Time within the chirp at which the last ADC sample is taken."""
        return self.adc_start_time + self.num_adc_samples / self.adc_sample_rate

    @property
    def sim_sample_rate(self) -> float:
        """Simulation sample rate; legacy aliases it to the ADC rate."""
        return self.adc_sample_rate

    @property
    def bandwidth(self) -> float:
        """Effective swept bandwidth during the ADC sampling window."""
        return self.freq_slope * (self.ramp_end_time - self.adc_start_time)

    @property
    def range_resolution(self) -> float:
        """Range resolution in metres, c / (2 * bandwidth)."""
        return self.light_speed / (2 * self.bandwidth)

    @property
    def doppler_resolution(self) -> float:
        """Doppler (radial velocity) resolution in m/s."""
        return self.wave_length / (2 * self.chirp_last_period)

    @property
    def max_range(self) -> float:
        """Maximum unambiguous range in metres."""
        return self.num_adc_samples * self.range_resolution

    @property
    def max_doppler(self) -> float:
        """Maximum unambiguous radial velocity in m/s."""
        return self.num_chirps / 2 * self.doppler_resolution


RADAR_CONFIG_REGISTRY: dict[str, RadarConfig] = {
    IWR1843_SIM_V1: RadarConfig(),
}


def get_radar_config(radar_config_id: str) -> RadarConfig:
    """Look up a registered radar configuration by its identifier."""
    try:
        return RADAR_CONFIG_REGISTRY[radar_config_id]
    except KeyError:
        known = ", ".join(sorted(RADAR_CONFIG_REGISTRY))
        raise RadarConfigError(
            f"unknown radar_config_id: {radar_config_id!r}; known: {known}"
        ) from None
