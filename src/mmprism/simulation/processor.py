"""Signal-processing operators for range, Doppler and antenna responses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mmprism.simulation.simulator import (
    get_index_full,
    get_index_large,
    get_index_middle,
    get_index_small,
)

_ARRAY_SIZES = ("small", "middle", "large", "full")


def _phi_2d(azi_theta: Tensor, ele_theta: Tensor, azi_idx: Tensor, ele_idx: Tensor) -> Tensor:
    """Far-field phase: for large range, ``phi ~= pi * k * sin(theta)``."""
    phi = np.pi * azi_idx * torch.sin(azi_theta) + np.pi * ele_idx * torch.cos(
        azi_theta
    ) * torch.sin(ele_theta)
    return phi


def build_steering_vector(
    azi_ele_id: Tensor, azi_theta_grid: Tensor, ele_theta_grid: Tensor
) -> Tensor:
    """2D steering vectors for every grid coordinate and angle pair.

    Args:
        azi_ele_id: ``[num_antenna, 2]`` half-wavelength grid indices.
        azi_theta_grid: ``[W]`` azimuth angles in radians.
        ele_theta_grid: ``[H]`` elevation angles in radians.

    Returns:
        Complex tensor ``[num_antenna, W*H]`` (angle pairs ordered by
        ``torch.cartesian_prod(azi, ele)``, i.e. azimuth-major).
    """
    azi_ele_id = azi_ele_id.unsqueeze(1)
    azi_ele_theta = torch.cartesian_prod(azi_theta_grid, ele_theta_grid).unsqueeze(0)
    steering_vector = torch.exp(
        -1j
        * _phi_2d(
            azi_ele_theta[:, :, 0],
            azi_ele_theta[:, :, 1],
            azi_ele_id[:, :, 0],
            azi_ele_id[:, :, 1],
        )
    )
    return steering_vector


class Processor(nn.Module):
    """Signal-processing module converting raw frames into mmwave power cubes.

    Faithful port of legacy ``src/fmcw/simulator.py::Processor``, including
    its windowing (periodic Hann via ``torch.hann_window``), optional
    slow-time mean removal, ``fftshift``-centred Doppler axis,
    non-conjugated steering einsum, and squared-magnitude power output.

    ``range_bin_start`` and ``range_bin_count`` express the original FFT-bin
    interval. The retained compatibility default is ``[0,32)``; MeshSim v2
    binds ``[8,24)`` explicitly and never relies on this compatibility path.
    """

    def __init__(
        self,
        learnable_weights: bool = False,
        W: int = 32,
        H: int = 32,
        dtype: torch.dtype = torch.float32,
        ctype: torch.dtype = torch.complex64,
        array_size: str = "full",
        process_range: bool = False,
        process_doppler: bool = True,
        remove_static: bool = True,
        range_bin_start: int = 0,
        range_bin_count: int = 32,
        angle_fov_deg: float = 30.0,
        output_phasor: bool = False,
        apply_doppler_window: bool = True,
        output_log: bool = True,
        output_db: bool = False,
        cube_normalize: bool = False,
        output_range: str = "db",
        legacy_mode: bool = False,
    ) -> None:
        super().__init__()
        if legacy_mode:
            # legacy_mode locks the NEW toggles off (Doppler Hann, direct-log,
            # dB output, cube-internal normalisation, output range mapping) but
            # leaves the legacy ``remove_static`` switch alone so callers can
            # still control static-clutter subtraction.
            apply_doppler_window = False
            output_log = False
            output_db = False
            cube_normalize = False
            output_range = "db"
        if output_range not in ("db", "neg1_1", "zero_1"):
            raise ValueError(
                f"output_range must be one of ('db', 'neg1_1', 'zero_1'), got {output_range!r}"
            )
        if output_log:
            # Direct log10 of power: no peak normalisation and no range
            # remapping. It resolves against the normalised-dB flags because
            # both cannot describe the same output tensor.
            output_db = False
            cube_normalize = False
            output_range = "db"
        if cube_normalize and output_db:
            # Both flags request a log10 normalisation; resolve to cube-internal
            # peak normalisation since it is the more explicit choice.
            output_db = False
        if array_size not in _ARRAY_SIZES:
            raise ValueError(f"array_size must be one of {_ARRAY_SIZES}, got {array_size!r}")
        self.array_size = array_size

        antenna_indices: Sequence[int] | np.ndarray
        if array_size == "small":
            D, antenna_indices = get_index_small()
        elif array_size == "middle":
            D, antenna_indices = get_index_middle()
        elif array_size == "large":
            D, antenna_indices = get_index_large()
        else:
            D, antenna_indices = get_index_full()

        # Kept as a numpy array exactly as legacy: numpy indexing works for
        # both CPU and CUDA frames, and ``len()`` drives the selection test.
        self.antenna_indices: np.ndarray = np.asarray(antenna_indices)
        self.D_antennas = len(D)
        if (
            not isinstance(range_bin_start, int)
            or isinstance(range_bin_start, bool)
            or range_bin_start < 0
        ):
            raise ValueError("range_bin_start must be a non-negative integer")
        if (
            not isinstance(range_bin_count, int)
            or isinstance(range_bin_count, bool)
            or range_bin_count < 1
        ):
            raise ValueError("range_bin_count must be a positive integer")
        self.D, self.R, self.W, self.H = 64, range_bin_count, W, H
        self.range_bin_start = range_bin_start
        self.range_bin_count = range_bin_count

        # Beamforming weights: steering vectors over a symmetric azimuth/
        # elevation grid (default +/-30 deg — the legacy delivery caliber;
        # wider grids are new named input arms), referenced to grid column
        # 43 (the phase centre).
        if not 0.0 < float(angle_fov_deg) < 90.0:
            raise ValueError(f"angle_fov_deg must be in (0, 90), got {angle_fov_deg!r}")
        self.angle_fov_deg = float(angle_fov_deg)
        fov_rad = np.deg2rad(self.angle_fov_deg)
        azi_ele_id = torch.tensor(np.array(D) - np.array([43, 0]))
        azi_theta_grid = torch.linspace(-fov_rad, fov_rad, self.W)
        ele_theta_grid = torch.linspace(-fov_rad, fov_rad, self.H)
        bm_weights = build_steering_vector(azi_ele_id, azi_theta_grid, ele_theta_grid)
        self.bm_weights = nn.Parameter(
            torch.view_as_real(bm_weights), requires_grad=learnable_weights
        )

        self.dtype = dtype
        self.ctype = ctype

        self.if_process_range = process_range
        self.if_process_doppler = process_doppler
        self.if_remove_static = remove_static
        self.if_apply_doppler_window = apply_doppler_window
        self.if_output_log = output_log
        self.if_output_db = output_db
        self.if_cube_normalize = cube_normalize
        self.if_output_phasor = output_phasor
        self.output_range = output_range
        self.legacy_mode = legacy_mode

    @property
    def range_bin_stop(self) -> int:
        """Exclusive end of the retained original range-FFT-bin interval."""

        return self.range_bin_start + self.range_bin_count

    def process_range(self, radar_frame: Tensor) -> Tensor:
        """Periodic-Hann range FFT and explicit original-bin crop.

        Args:
            radar_frame: ``[B, num_chirps, num_antenna, num_samples]``.

        Returns:
            ``[B, num_chirps, num_antenna, range_bin_count]``.
        """
        B = radar_frame.shape[0]
        num_samples = radar_frame.shape[-1]
        if self.range_bin_stop > num_samples:
            raise ValueError(
                "range crop exceeds native FFT length: "
                f"[{self.range_bin_start},{self.range_bin_stop}) for {num_samples} samples"
            )
        window = torch.hann_window(num_samples, device=radar_frame.device)
        window = window.view(1, 1, 1, -1).expand(B, -1, -1, -1)

        radar_frame = radar_frame * window
        radar_frame = torch.fft.fft(radar_frame, dim=-1)
        return radar_frame[..., self.range_bin_start : self.range_bin_stop]

    def process_doppler(self, radar_frame: Tensor) -> Tensor:
        """Slow-time mean removal, Hann window, Doppler FFT, and fftshift.

        Args:
            radar_frame: ``[B, num_chirps, num_antenna, R]``.

        Returns:
            ``[B, D, num_antenna, R]`` with a centred Doppler axis.
        """
        B = radar_frame.shape[0]
        num_chirps = radar_frame.shape[1]

        # The synthetic route removes slow-time mean here. Processed measured
        # products may already have that operation applied and must opt out.
        if self.if_remove_static:
            radar_frame = radar_frame - radar_frame.mean(dim=1, keepdim=True)

        if self.if_apply_doppler_window:
            window = torch.hann_window(num_chirps, device=radar_frame.device)
            window = window.view(1, -1, 1, 1).expand(B, -1, -1, -1)
            radar_frame = radar_frame * window

        radar_frame = torch.fft.fftshift(torch.fft.fft(radar_frame, dim=1), dim=1)
        return radar_frame

    def process_beamforming(self, radar_frame: Tensor) -> Tensor:
        """Steering-vector beamforming and squared-magnitude power.

        Args:
            radar_frame: ``[B, D, num_antenna, R]``.

        Returns:
            Non-negative power ``[B, D, R, W*H]``.
        """
        return self._beamform(radar_frame).abs() ** 2

    def _beamform(self, radar_frame: Tensor) -> Tensor:
        """Complex steering-vector beamforming output ``[B, D, R, W*H]``."""

        bm_weights = self.bm_weights[..., 0] + 1j * self.bm_weights[..., 1]
        return torch.einsum("bdar,aw->bdrw", radar_frame, bm_weights)

    def forward(self, raw_radar_frame: Tensor) -> Tensor:
        """Convert a raw radar frame batch into mmwave power cubes.

        Args:
            raw_radar_frame: ``[B, num_chirps, num_antenna, num_samples]``.

        Returns:
            ``[B, D, R, W, H]`` non-negative power cubes; with
            ``output_phasor`` the return is ``[B, 3*D, R, W, H]`` — the
            power block first, then cos/sin of the unit beamformed
            phasor (zero where the bin is empty). The phase channels
            carry the cross-bin interferometric structure that the
            squared magnitude discards; they ride along the Doppler
            axis so the encoder sees them as extra input channels.
        """
        if len(self.antenna_indices) < raw_radar_frame.shape[2]:
            radar_frame = raw_radar_frame[:, :, self.antenna_indices, :].clone()
        else:
            radar_frame = raw_radar_frame.clone()

        if self.if_process_range:
            radar_frame = self.process_range(radar_frame)
        elif radar_frame.shape[-1] != self.R:
            raise ValueError(
                "raw range dimension must equal range_bin_count when process_range=False"
            )
        if self.if_process_doppler:
            radar_frame = self.process_doppler(radar_frame)
        phasor: Tensor | None = None
        if self.if_output_phasor:
            complex_frame = self._beamform(radar_frame)
            magnitude = complex_frame.abs().clamp(min=1e-30)
            phasor = complex_frame / magnitude
            radar_frame = magnitude**2
        else:
            radar_frame = self.process_beamforming(radar_frame)

        if self.if_output_log:
            # Direct log10 of beamforming power; absolute, un-normalised. The
            # clamp only guards exact zeros (log10(0) = -inf).
            radar_frame = torch.log10(radar_frame.clamp(min=1e-30))
        elif self.if_cube_normalize or self.if_output_db:
            peak = radar_frame.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
            radar_frame = 10.0 * torch.log10(radar_frame / peak + 1e-12)

        if self.output_range == "neg1_1":
            radar_frame = (radar_frame.clamp(min=-40.0, max=0.0) + 40.0) / 20.0 - 1.0
        elif self.output_range == "zero_1":
            radar_frame = (radar_frame.clamp(min=-40.0, max=0.0) + 40.0) / 40.0

        B = radar_frame.shape[0]
        power = radar_frame.view(B, self.D, self.R, self.W, self.H)
        if phasor is not None:
            unit = phasor.view(B, self.D, self.R, self.W, self.H)
            return torch.cat([power, unit.real, unit.imag], dim=1)
        return power
