"""Radar frontend configurations for maintained mesh-surface FMCW synthesis.

Only physical frontend parameters live here. Derived quantities (sampling
duration, FFT-bin spacing, bandwidth, resolutions, and duty cycle) are
properties so they cannot drift from the declared waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Active real-anchored MeshSim source. The 100 samples at 2.034 MHz retain
#: the collector acquisition duration while exposing its native fast-time
#: layout to the Processor. One 12-Tx TDM loop is active for 1.2 ms, but the
#: same-Tx slow-time PRT is the 100 ms frame periodicity divided across 64
#: loops: 1.5625 ms, including 0.3625 ms of inter-loop idle.
#: The retired transitional ``mmwcas_tdm_v1`` synthesis parameters
#: (256 samples @ 5.209 MHz — an eighth-length beat window that collapses
#: all relative-distance energy into the first bins) were removed from the
#: registry after producing a degenerate ROI in the meshsim sidecar lane.
#: ``mmwcas_tdm_v1`` survives only as a manifest ``radar_config_ref``
#: data-contract string in paired manifests; it can no longer be used for
#: synthesis. Completed historical configs that still name it fail loudly
#: at ``get_radar_config`` if replayed.
MMWCAS_TDM_V2 = "mmwcas_tdm_v2"
ACTIVE_MMWCAS_TDM_CONFIG_ID = MMWCAS_TDM_V2


class RadarConfigError(ValueError):
    """Raised when a radar configuration is invalid or unknown."""


@dataclass(frozen=True, slots=True)
class RadarConfig:
    """MMWCAS FMCW frontend parameters used by mesh-surface synthesis.

    Attributes use SI units (Hz, s, m/s). ``light_speed`` matches the exact
    constant used by :mod:`mmprism.simulation.simulator`, so the reported
    range and Doppler axes match the phase synthesis.
    """

    light_speed: float = 2.99792458e8
    start_freq: float = 77e9
    freq_slope: float = 70e12
    adc_sample_rate: float = 2.034e6
    num_adc_samples: int = 100
    idle_time: float = 1200e-6 - 55e-6
    adc_start_time: float = 5e-6
    ramp_end_time: float = 55e-6
    num_chirps: int = 64
    frame_period: float = 100e-3

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
        # Acquisition timing feasibility.
        if not self.adc_sample_end_time < self.ramp_end_time:
            raise RadarConfigError("ADC sampling time exceeds chirp duration")
        if self.tdm_loop_active_duration > self.chirp_period:
            raise RadarConfigError(
                "Duty cycle exceeds slow-time period: "
                f"{self.tdm_loop_active_duration} > {self.chirp_period}"
            )

    @property
    def wave_length(self) -> float:
        """Carrier wavelength in metres."""
        return self.light_speed / self.start_freq

    @property
    def tdm_loop_active_duration(self) -> float:
        """Active duration of one complete 12-Tx TDM loop."""
        return self.idle_time + self.ramp_end_time

    @property
    def chirp_period(self) -> float:
        """Same-Tx slow-time PRT, including inter-loop idle."""
        return self.frame_period / self.num_chirps

    @property
    def inter_loop_idle_time(self) -> float:
        """Idle interval between active TDM loops."""
        return self.chirp_period - self.tdm_loop_active_duration

    @property
    def chirp_last_period(self) -> float:
        """Period represented by all slow-time samples in one frame."""
        return self.chirp_period * self.num_chirps

    @property
    def adc_sample_end_time(self) -> float:
        """Time within the chirp at which the last ADC sample is taken."""
        return self.adc_start_time + self.num_adc_samples / self.adc_sample_rate

    @property
    def adc_sampling_duration(self) -> float:
        """Duration covered by the declared ADC sample count in seconds."""

        return self.num_adc_samples / self.adc_sample_rate

    @property
    def sim_sample_rate(self) -> float:
        """Simulation sample rate; legacy aliases it to the ADC rate."""
        return self.adc_sample_rate

    @property
    def bandwidth(self) -> float:
        """Effective swept bandwidth covered by the ADC sampling window."""

        return self.freq_slope * self.adc_sampling_duration

    @property
    def range_resolution(self) -> float:
        """Native range-FFT bin spacing in metres, ``c / (2 * bandwidth)``."""
        return self.light_speed / (2 * self.bandwidth)

    @property
    def range_fft_bin_spacing(self) -> float:
        """Explicit alias for the physical spacing of one native FFT bin."""

        return self.range_resolution

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
    MMWCAS_TDM_V2: RadarConfig(
        adc_sample_rate=2.034e6,
        num_adc_samples=100,
        idle_time=1200e-6 - 55e-6,
        frame_period=100e-3,
    ),
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
