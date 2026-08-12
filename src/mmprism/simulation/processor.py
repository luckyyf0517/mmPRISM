"""Raw-frame -> power-cube signal processor (port of legacy ``Processor``).

Pipeline: periodic-Hann range FFT keeping the first 32 bins, slow-time
mean removal plus periodic-Hann Doppler FFT with ``fftshift`` over 64 chirp
bins, then synthetic steering-vector beamforming onto a 32x32
azimuth/elevation grid, and finally squared magnitude.

Steering vectors come from the legacy ``src/fmcw/beamformer.py`` far-field
approximation ``phi = pi * (i * sin(azi) + j * cos(azi) * sin(ele))`` with
half-wavelength grid indices. The einsum applies the steering matrix without
explicit conjugation, exactly as legacy.
"""

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


def _phi_2d(
    azi_theta: Tensor, ele_theta: Tensor, azi_idx: Tensor, ele_idx: Tensor
) -> Tensor:
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


def build_steering_vector_1d(azi_id: Tensor, azi_theta_grid: Tensor) -> Tensor:
    """1D (azimuth-only) steering vectors, ``[num_antenna, W]``."""
    azi_theta = azi_theta_grid.unsqueeze(0)
    azi_id = azi_id.unsqueeze(1)
    steering_vector = torch.exp(-1j * _phi_2d(azi_theta, azi_theta, azi_id, azi_id))
    return steering_vector


class Processor(nn.Module):
    """Signal-processing module converting raw frames into mmwave power cubes.

    Faithful port of legacy ``src/fmcw/simulator.py::Processor``, including
    its windowing (periodic Hann via ``torch.hann_window``), slow-time mean
    removal, ``fftshift``-centred Doppler axis, non-conjugated steering
    einsum, and squared-magnitude power output.

    Note: the legacy default ``process_range=False`` only produces a valid
    cube when the fast-time dimension already equals ``range_bins``; the
    legacy call sites that actually ran (``src/model/omnihand.py``,
    ``view_mmwave_cube.py``) set it to ``True``. The legacy defaults are kept
    here for fidelity.
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
    ) -> None:
        super().__init__()
        if array_size not in _ARRAY_SIZES:
            raise ValueError(
                f"array_size must be one of {_ARRAY_SIZES}, got {array_size!r}"
            )
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
        self.D, self.R, self.W, self.H = 64, 32, W, H

        # Beamforming weights: steering vectors over a +/-30 deg azimuth/
        # elevation grid, referenced to grid column 43 (the phase centre).
        azi_ele_id = torch.tensor(np.array(D) - np.array([43, 0]))
        azi_theta_grid = torch.linspace(-np.pi / 6, np.pi / 6, self.W)
        ele_theta_grid = torch.linspace(-np.pi / 6, np.pi / 6, self.H)
        bm_weights = build_steering_vector(azi_ele_id, azi_theta_grid, ele_theta_grid)
        self.bm_weights = nn.Parameter(
            torch.view_as_real(bm_weights), requires_grad=learnable_weights
        )

        self.dtype = dtype
        self.ctype = ctype

        self.if_process_range = process_range
        self.if_process_doppler = process_doppler

    def process_range(self, radar_frame: Tensor) -> Tensor:
        """Periodic-Hann windowed range FFT, keeping bins ``[0, R)``.

        Args:
            radar_frame: ``[B, num_chirps, num_antenna, num_samples]``.

        Returns:
            ``[B, num_chirps, num_antenna, R]``.
        """
        B = radar_frame.shape[0]
        num_samples = radar_frame.shape[-1]
        window = torch.hann_window(num_samples, device=radar_frame.device)
        window = window.view(1, 1, 1, -1).expand(B, -1, -1, -1)

        radar_frame = radar_frame * window
        radar_frame = torch.fft.fft(radar_frame, dim=-1)
        return radar_frame[..., : self.R]

    def process_doppler(self, radar_frame: Tensor) -> Tensor:
        """Slow-time mean removal, Hann window, Doppler FFT, and fftshift.

        Args:
            radar_frame: ``[B, num_chirps, num_antenna, R]``.

        Returns:
            ``[B, D, num_antenna, R]`` with a centred Doppler axis.
        """
        B = radar_frame.shape[0]
        num_chirps = radar_frame.shape[1]

        # Remove static clutter by subtracting the complex slow-time mean.
        radar_frame = radar_frame - radar_frame.mean(dim=1, keepdim=True)

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
        bm_weights = self.bm_weights[..., 0] + 1j * self.bm_weights[..., 1]
        radar_frame = torch.einsum("bdar,aw->bdrw", radar_frame, bm_weights)
        return radar_frame.abs() ** 2

    def forward(self, raw_radar_frame: Tensor) -> Tensor:
        """Convert a raw radar frame batch into mmwave power cubes.

        Args:
            raw_radar_frame: ``[B, num_chirps, num_antenna, num_samples]``.

        Returns:
            ``[B, D, R, W, H]`` non-negative power cubes.
        """
        if len(self.antenna_indices) < raw_radar_frame.shape[2]:
            radar_frame = raw_radar_frame[:, :, self.antenna_indices, :].clone()
        else:
            radar_frame = raw_radar_frame.clone()

        if self.if_process_range:
            radar_frame = self.process_range(radar_frame)
        if self.if_process_doppler:
            radar_frame = self.process_doppler(radar_frame)
        radar_frame = self.process_beamforming(radar_frame)

        B = radar_frame.shape[0]
        return radar_frame.view(B, self.D, self.R, self.W, self.H)
