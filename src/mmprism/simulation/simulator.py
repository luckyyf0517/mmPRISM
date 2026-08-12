"""Point-reflector FMCW radar simulator (port of legacy ``src/fmcw/simulator.py``).

Physical model (baseband-equivalent, no noise, no antenna pattern): each scene
point is an isotropic reflector. For every RX element the round-trip delay
``tau = (|p - tx| + |rx - p|) / c`` and free-space amplitude ``1 / (d/2)^2``
are computed, and the beat signal of one frame is the sum over paths of
``a * exp(1j * 2*pi * (f_slope * t + f0) * tau_chirp)`` where ``tau_chirp``
adds the Doppler-induced delay drift ``(2 * v_radial / c) * chirp_time``.

Legacy quirks preserved deliberately (do NOT "fix"; see
``docs/authority/20_CONTRACTS/TENSOR_CONTRACTS.md`` evidence conflicts):

1. ``PointReflectorSimulator`` computes paths in float64 (legacy
   ``torch.tensor(numpy_array)`` promotion) and casts results to ``dtype``.
2. ``Simulation.get_raw_radar_frame`` ends with ``.to(dtype)`` on a complex
   echo; with the default ``dtype=torch.float32`` this silently DISCARDS THE
   IMAGINARY PART of the radar echo. The behaviour is preserved for
   bit-level equivalence with legacy evidence and is flagged in the
   docstring of :meth:`Simulation.get_raw_radar_frame`.
3. Legacy ``mmSimulator.init`` calls an undefined ``get_index()``; the only
   executable interpretation (and the one used by the equivalence fixture,
   which monkeypatches it to ``get_index_full``) is the full virtual array.
   This port uses the full virtual array explicitly.
4. Legacy ``Simulation.forward`` called the dual-hand pose densifier on
   already-densified ``[B, N, 3]`` clouds, which cannot execute (it indexes
   a ``[T, 2, 24, 3]`` layout). Densification lives in
   :mod:`mmprism.simulation.point_cloud`; this ``forward`` takes densified
   clouds directly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from mmprism.simulation.radar_config import (
    IWR1843_SIM_V1,
    RadarConfig,
    get_radar_config,
)

# Exact speed of light hard-coded in the legacy simulator core; intentionally
# distinct from the rounded ``RadarConfig.light_speed`` (3e8) used for derived
# resolution metadata.
SIMULATION_LIGHT_SPEED = 2.99792458e8

# Legacy radar mounting position in camera coordinates (metres).
RADAR_POSITION = (0.0, 0.0, -0.80)


def get_index_full() -> tuple[list[list[int]], np.ndarray]:
    """Full virtual-array layout: 116 half-wavelength grid coordinates.

    Each entry is ``[column, row]`` on a grid whose spacing unit is half a
    wavelength; column 43 is the phase centre (subtracted by callers).
    """
    D = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
        [10, 0], [11, 0], [11, 1], [11, 4], [11, 6], [12, 0], [12, 1], [12, 4],
        [12, 6], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 0], [19, 0],
        [20, 0], [21, 0], [22, 0], [22, 1], [22, 4], [22, 6], [23, 0], [23, 1],
        [23, 4], [23, 6], [24, 0], [25, 0], [26, 0], [27, 0], [28, 0], [29, 0],
        [30, 0], [31, 0], [32, 0], [33, 0], [34, 0], [35, 0], [36, 0], [37, 0],
        [38, 0], [39, 0], [40, 0], [41, 0], [42, 0], [43, 0], [44, 0], [45, 0],
        [46, 0], [47, 0], [48, 0], [49, 0], [50, 0], [51, 0], [52, 0], [53, 0],
        [54, 0], [55, 0], [56, 0], [57, 0], [57, 1], [57, 4], [57, 6], [58, 0],
        [58, 1], [58, 4], [58, 6], [59, 0], [59, 1], [59, 4], [59, 6], [60, 0],
        [60, 1], [60, 4], [60, 6], [61, 0], [61, 1], [61, 4], [61, 6], [62, 0],
        [62, 1], [62, 4], [62, 6], [63, 0], [64, 0], [65, 0], [66, 0], [67, 0],
        [68, 0], [69, 0], [70, 0], [71, 0], [72, 0], [73, 0], [74, 0], [75, 0],
        [76, 0], [77, 0], [78, 0], [79, 0], [80, 0], [81, 0], [82, 0], [83, 0],
        [84, 0], [85, 0],
    ]
    return D, np.arange(0, len(D))


def get_index_large() -> tuple[list[list[int]], list[int]]:
    """44-element sub-array of the full layout (indices into ``get_index_full``)."""
    D = [
        [11, 0], [11, 1], [11, 4], [11, 6], [12, 0], [12, 1], [12, 4], [12, 6],
        [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 0], [19, 0], [20, 0],
        [21, 0], [22, 0], [22, 1], [22, 4], [22, 6], [23, 0], [23, 1], [23, 4],
        [23, 6], [24, 0], [25, 0], [26, 0], [27, 0], [28, 0], [29, 0], [30, 0],
        [31, 0], [32, 0], [33, 0], [34, 0], [35, 0], [36, 0], [37, 0], [38, 0],
        [39, 0], [40, 0], [41, 0], [42, 0],
    ]
    D_full, _ = get_index_full()
    index = [D_full.index(item) for item in D]
    return D, index


def get_index_middle() -> tuple[list[list[int]], list[int]]:
    """24-element sub-array of the full layout."""
    D = [
        [11, 0], [11, 1], [11, 4], [12, 0], [12, 1], [12, 4], [13, 0], [14, 0],
        [15, 0], [16, 0], [17, 0], [18, 0], [19, 0], [20, 0], [21, 0], [22, 0],
        [22, 1], [22, 4], [23, 0], [23, 1], [23, 4], [24, 0], [25, 0], [26, 0],
    ]
    D_full, _ = get_index_full()
    index = [D_full.index(item) for item in D]
    return D, index


def get_index_small() -> tuple[list[list[int]], list[int]]:
    """10-element sub-array of the full layout."""
    D = [
        [11, 0], [11, 1], [12, 0], [12, 1], [13, 0], [14, 0], [15, 0], [16, 0],
        [17, 0], [18, 0],
    ]
    D_full, _ = get_index_full()
    index = [D_full.index(item) for item in D]
    return D, index


class PointReflectorSimulator(nn.Module):
    """Geometry core: 3D points/velocities -> per-RX path delays and amplitudes.

    Port of legacy ``mmSimulator``. TX and the 116-element virtual RX array
    sit at :data:`RADAR_POSITION`; RX elements are spaced by half a wavelength
    on the grid returned by :func:`get_index_full`.
    """

    # Buffers (declared for type checkers; populated via register_buffer).
    tx_position: Tensor
    rx_positions: Tensor

    def __init__(
        self,
        radar_config: RadarConfig | None = None,
        dtype: torch.dtype = torch.float32,
        ctype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        config = radar_config if radar_config is not None else get_radar_config(IWR1843_SIM_V1)
        self.frequency = config.start_freq
        self.dtype = dtype
        self.ctype = ctype

        radar_position = np.array(RADAR_POSITION)
        array_grid = np.array(get_index_full()[0]) - np.array([43, 0])
        # Half-wavelength element spacing (exact light speed, as legacy).
        spacing = (SIMULATION_LIGHT_SPEED / self.frequency) / 2
        rx_ref_array = np.column_stack(
            (array_grid[:, 0], array_grid[:, 1], np.zeros(len(array_grid)))
        )
        rx_ref_array = rx_ref_array * spacing

        # Legacy keeps these buffers in float64 (torch.tensor of float64
        # numpy arrays); path computation below therefore runs in float64.
        self.register_buffer("tx_position", torch.tensor(radar_position), persistent=False)
        self.register_buffer(
            "rx_positions",
            torch.tensor(radar_position[None] + rx_ref_array),
            persistent=False,
        )

    def compute_paths_from_points(
        self, points_3d: Tensor, velocities_3d: Tensor
    ) -> dict[str, Tensor]:
        """Compute per-RX path delay, amplitude, and radial velocity.

        Args:
            points_3d: ``[N, 3]`` reflector positions in camera coordinates.
            velocities_3d: ``[N, 3]`` reflector velocities in m/s.

        Returns:
            Dict with ``a`` (amplitude), ``tau`` (round-trip delay in s), and
            ``vel`` (radial velocity in m/s), each ``[num_rx, N]`` in
            ``self.dtype``. The radial velocity is the two-way bistatic
            average ``(v . k_i - v . k_s) / 2`` over the normalised incidence
            and scattering directions.
        """
        k_i = points_3d - self.tx_position  # [N, 3] incidence vectors
        k_s = self.rx_positions[:, None] - points_3d  # [num_rx, N, 3] scatter vectors

        k_i_length = torch.norm(k_i, dim=-1)  # [N]
        k_s_length = torch.norm(k_s, dim=-1)  # [num_rx, N]
        distances = k_i_length[None] + k_s_length  # [num_rx, N] two-way path length

        tau = distances / SIMULATION_LIGHT_SPEED  # [num_rx, N]
        a = torch.ones_like(tau) / (distances / 2) ** 2

        k_i = k_i / k_i_length.unsqueeze(-1)  # [N, 3]
        k_s = k_s / k_s_length.unsqueeze(-1)  # [num_rx, N, 3]
        vel = (
            torch.sum(velocities_3d * k_i, dim=-1)[None]
            - torch.sum(velocities_3d * k_s, dim=-1)
        ) / 2  # [num_rx, N]

        return {
            "a": a.to(self.dtype),
            "tau": tau.to(self.dtype),
            "vel": vel.to(self.dtype),
        }


class Simulation(nn.Module):
    """Frame-level wrapper: densified point clouds -> raw radar frames.

    Port of legacy ``Simulation``. Chirp timing follows the radar config:
    fast-time sample instants ``t = adc_start_time + n / adc_sample_rate``
    within each chirp, slow-time chirp instants ``m * chirp_period``.
    """

    def __init__(
        self,
        radar_config: RadarConfig | None = None,
        dtype: torch.dtype = torch.float32,
        ctype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        config = radar_config if radar_config is not None else get_radar_config(IWR1843_SIM_V1)
        self.radar_config = config
        self.simulator = PointReflectorSimulator(config, dtype=dtype, ctype=ctype)

        self.start_freq = config.start_freq
        self.freq_slope = config.freq_slope
        self.adc_sample_rate = config.adc_sample_rate
        self.adc_start_time = config.adc_start_time
        self.num_adc_samples = config.num_adc_samples
        self.ts = self.adc_start_time + torch.arange(self.num_adc_samples, dtype=dtype) / (
            self.adc_sample_rate
        )
        self.fs = self.ts * self.freq_slope
        self.chirp_period = config.chirp_period
        self.num_chirps = config.num_chirps
        self.time_steps = torch.arange(self.num_chirps, dtype=dtype) * self.chirp_period
        self.ramp_end_time = config.ramp_end_time

        self.dtype = dtype
        self.ctype = ctype

    def simulate_batch(self, points_3d: Tensor, velocities_3d: Tensor) -> Tensor:
        """Simulate one frame from a single densified point cloud.

        Args:
            points_3d: ``[N, 3]`` reflector positions in camera coordinates.
            velocities_3d: ``[N, 3]`` reflector velocities in m/s.

        Returns:
            Raw radar frame ``[num_chirps, num_rx, num_samples]``.
        """
        path_dict = self.simulator.compute_paths_from_points(
            points_3d=points_3d,
            velocities_3d=velocities_3d,
        )
        return self.get_raw_radar_frame(path_dict)

    def forward(self, points_3d: Tensor, velocities_3d: Tensor) -> Tensor:
        """Simulate a batch of frames.

        Args:
            points_3d: ``[B, N, 3]`` densified reflector positions.
            velocities_3d: ``[B, N, 3]`` densified reflector velocities.

        Returns:
            ``[B, num_chirps, num_rx, num_samples]`` raw radar frames. With
            the default ``dtype`` these are REAL tensors: the legacy
            complex-to-real cast quirk documented on
            :meth:`get_raw_radar_frame` applies.
        """
        batch_size = points_3d.shape[0]
        radar_frame_list = []
        for b in range(batch_size):
            radar_frame = self.simulate_batch(points_3d[b], velocities_3d[b])
            radar_frame_list.append(radar_frame)
        return torch.stack(radar_frame_list, dim=0)

    def get_raw_radar_frame(
        self, path_dict: dict[str, Tensor], save_cuda_memory: bool = False
    ) -> Tensor:
        """Synthesise the raw beat-signal frame from path parameters.

        Sums ``a * exp(1j * 2*pi * (f_slope * t + f0) * tau_chirp)`` over
        paths, with ``tau_chirp = tau + (2 * vel / c) * chirp_time`` and the
        phase reduced modulo ``2*pi`` exactly as legacy.

        Args:
            path_dict: ``a``, ``tau``, ``vel`` each ``[num_rx, N]`` from
                :meth:`PointReflectorSimulator.compute_paths_from_points`.
            save_cuda_memory: legacy float16 accumulation switch.

        Returns:
            ``[num_chirps, num_rx, num_samples]``.

        LEGACY QUIRK (preserved, documented in TENSOR_CONTRACTS evidence
        conflicts): the result is cast with ``.to(self.dtype)``; with the
        default ``torch.float32`` this discards the imaginary part of the
        complex echo. Downstream consumers of the default output therefore
        process the real part only.
        """
        a = path_dict["a"]  # [num_rx, N]
        tau = path_dict["tau"]  # [num_rx, N]
        vel = path_dict["vel"]  # [num_rx, N]
        tau_velocity = vel * 2 / SIMULATION_LIGHT_SPEED  # [num_rx, N]
        time_steps = self.time_steps[:, None, None]  # [num_chirps, 1, 1]
        # tau_chirp: [num_chirps, num_rx, 1, N]
        tau_chirp = tau_velocity * time_steps.to(a.device)
        tau_chirp = tau.unsqueeze(0) + tau_chirp
        tau_chirp = tau_chirp[:, :, None, :]
        # frequencies: [1, 1, num_samples, 1]
        frequencies = self.fs[None, None, :, None].to(a.device)
        # ft_phase: [num_chirps, num_rx, num_samples, N]
        ft_phase = 2 * np.pi * (frequencies + self.start_freq) * tau_chirp
        ft_phase %= 2 * np.pi
        # a: [1, num_rx, 1, N]
        a = a[None, :, None, :]
        if save_cuda_memory:
            a = a.to(torch.float16)
            ft_phase = ft_phase.to(torch.float16)
        radar_frame = (a * torch.exp(1j * ft_phase)).sum(dim=-1)
        return radar_frame.to(self.dtype)
