"""Pose densification and temporal preprocessing for the simulator.

Two legacy densifiers are ported:

- :func:`densify_dual_hand_pose` — torch port of
  ``src/fmcw/simulator.py::process_point_cloud`` for ``[T, 2, 24, 3]``
  dual-hand poses (3 arm points + 21 hand points per side). Body-skeleton
  edges are interpolated with 3 interior points each and NaN-invalid points
  are masked out per frame.
- :func:`densify_body_hand_frames` — numpy port of
  ``run_simulation.py::process_point_cloud`` for full-body pose sequences:
  body joints ``5:11``, hands as the last 42 joints, z scaled by 0.6.

:func:`temporal_smooth_decimate` ports the legacy sequence preprocessing
from ``run_simulation.py::process_sequence``: Gaussian smoothing
(``sigma=1`` along time), finite-difference velocities scaled by 10
(30 fps source, so the scale yields m/s), and 30 -> 10 fps decimation by
taking every third frame.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]
from torch import Tensor


def densify_dual_hand_pose(
    data: Tensor, nan_mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Densify ``[T, 2, 24, 3]`` dual-hand poses into reflector clouds.

    Per side, the first 3 joints are arm points (shoulder/elbow/wrist chain)
    and the remaining 21 are hand points. Each arm edge gets 3 linearly
    interpolated interior points; one cross edge connects the two arm roots.
    Output point order is ``[left_arm(3), right_arm(3), interpolated(15),
    left_hand(21), right_hand(21)]`` = 63 candidate points.

    Args:
        data: ``[T, 2, 24, 3]`` pose tensor, may contain NaNs.
        nan_mask: optional precomputed ``[T, 63]`` validity mask; computed
            from ``data`` when omitted.

    Returns:
        ``(padded_points, nan_mask)`` where ``padded_points`` is
        ``[T, max_valid, 3]`` with zero padding so every frame shares the
        second dimension, and ``nan_mask`` is the boolean mask used. If no
        point is valid anywhere, returns ``[T, 1, 3]`` zeros, matching the
        legacy guard.
    """
    left_body = data[:, 0, :3, :]  # [T, 3, 3] left arm points
    left_hand = data[:, 0, 3:, :]  # [T, 21, 3] left hand points
    right_body = data[:, 1, :3, :]  # [T, 3, 3] right arm points
    right_hand = data[:, 1, 3:, :]  # [T, 21, 3] right hand points

    # Skeleton edges: within each arm, plus one cross-connection between roots.
    left_skeleton = torch.tensor([[0, 1], [1, 2]], device=data.device)
    right_skeleton = torch.tensor([[0, 1], [1, 2]], device=data.device)
    cross_skeleton = torch.tensor([[0, 0]], device=data.device)

    def interpolate_points_vectorized(
        p1: Tensor, p2: Tensor, num_points: int = 3
    ) -> Tensor:
        t_values = torch.linspace(0, 1, num_points + 2, device=data.device)[1:-1]
        t_values = t_values.unsqueeze(0).unsqueeze(-1)
        return p1.unsqueeze(1) + (p2 - p1).unsqueeze(1) * t_values

    left_interpolated = [
        interpolate_points_vectorized(left_body[:, i], left_body[:, j])
        for i, j in left_skeleton
    ]
    right_interpolated = [
        interpolate_points_vectorized(right_body[:, i], right_body[:, j])
        for i, j in right_skeleton
    ]
    cross_interpolated = [
        interpolate_points_vectorized(left_body[:, i], right_body[:, j])
        for i, j in cross_skeleton
    ]

    interpolated_points = []
    if left_interpolated:
        interpolated_points.append(torch.cat(left_interpolated, dim=1))
    if right_interpolated:
        interpolated_points.append(torch.cat(right_interpolated, dim=1))
    if cross_interpolated:
        interpolated_points.append(torch.cat(cross_interpolated, dim=1))

    interpolated = torch.cat(interpolated_points, dim=1)

    all_points = torch.cat(
        [left_body, right_body, interpolated, left_hand, right_hand], dim=1
    )

    if nan_mask is None:
        nan_mask = ~torch.isnan(all_points).any(dim=-1)  # [T, 63]

    batch_size = all_points.shape[0]

    # Legacy guard: if every point is filtered out, return a minimal tensor.
    if not nan_mask.any():
        return torch.zeros((batch_size, 1, 3), device=data.device), nan_mask

    num_valid_per_batch = nan_mask.sum(dim=1)  # [T]
    max_valid_points = int(num_valid_per_batch.max().item())

    if max_valid_points == 0:
        return torch.zeros((batch_size, 1, 3), device=data.device), nan_mask

    padded_points = torch.zeros((batch_size, max_valid_points, 3), device=data.device)

    for t in range(batch_size):
        valid_count = int(num_valid_per_batch[t].item())
        if valid_count > 0:
            padded_points[t, :valid_count] = all_points[t, nan_mask[t]]

    return padded_points, nan_mask


def densify_body_hand_frames(data: np.ndarray) -> np.ndarray:
    """Densify full-body pose frames ``[T, N, 3]`` into reflector clouds.

    Legacy ``run_simulation.py`` layout: body joints are ``data[:, 5:11]``
    (two arm chains of three), hand points are the trailing 42 joints
    (``-42:-21`` left, ``-21:`` right). Each of the 5 skeleton edges gets 3
    interpolated interior points. The z coordinate is scaled by 0.6 (legacy
    depth compression), matching the legacy pipeline before smoothing.

    Returns:
        ``[T, 6 + 15 + 42, 3]`` = ``[T, 63, 3]`` reflector cloud per frame.
    """
    body = data[:, 5:11, :]
    handl = data[:, -42:-21, :]
    handr = data[:, -21:, :]

    body_skeleton = np.array([(0, 2), (2, 4), (1, 3), (3, 5), (0, 1)])

    def interpolate_points_vectorized(
        p1: np.ndarray, p2: np.ndarray, num_points: int = 3
    ) -> np.ndarray:
        t_values: np.ndarray = np.linspace(0, 1, num_points + 2)[1:-1]
        t_values = t_values[np.newaxis, :, np.newaxis]
        return np.asarray(p1[:, np.newaxis, :] + (p2 - p1)[:, np.newaxis, :] * t_values)

    interpolated_points_list = [
        interpolate_points_vectorized(body[:, i], body[:, j]) for i, j in body_skeleton
    ]
    interpolated_points = np.concatenate(interpolated_points_list, axis=1)
    # np.concatenate already returns a fresh array; legacy scales z in place.
    all_body_points = np.concatenate((body, interpolated_points, handl, handr), axis=1)
    all_body_points[..., 2] *= 0.6
    return all_body_points


def temporal_smooth_decimate(
    points: np.ndarray,
    *,
    sigma: float = 1.0,
    velocity_scale: float = 10.0,
    decimation: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth, differentiate, and decimate a reflector-cloud sequence.

    Replicates the legacy temporal preprocessing: Gaussian smoothing with
    ``sigma=1`` along time, forward finite-difference velocities scaled by
    10 (the 30 fps source rate, giving m/s), dropping the final frame to
    align lengths, then ``[::3]`` decimation from 30 to 10 fps.

    Args:
        points: ``[T, N, 3]`` reflector clouds (already densified).
        sigma: Gaussian smoothing sigma along the time axis.
        velocity_scale: multiplier for finite-difference velocities.
        decimation: frame stride for downsampling.

    Returns:
        ``(points_ds, velocities_ds)``, both ``[ceil((T-1)/decimation), N, 3]``.
    """
    smoothed: np.ndarray = gaussian_filter1d(points, sigma=sigma, axis=0)
    velocities = (smoothed[1:] - smoothed[:-1]) * velocity_scale
    aligned_points = smoothed[:-1]
    return aligned_points[::decimation], velocities[::decimation]
