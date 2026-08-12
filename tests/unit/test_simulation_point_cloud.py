from __future__ import annotations

import numpy as np
import torch

from mmprism.simulation.point_cloud import (
    densify_body_hand_frames,
    densify_dual_hand_pose,
    temporal_smooth_decimate,
)


def _seeded_poses(frames: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    poses = torch.randn(frames, 2, 24, 3, generator=generator)
    poses[..., 2] = poses[..., 2].abs() + 1.0
    return poses


def test_densify_dual_hand_pose_shape_and_order() -> None:
    poses = _seeded_poses()
    points, mask = densify_dual_hand_pose(poses)

    assert points.shape == (6, 63, 3)
    assert mask.shape == (6, 63)
    assert mask.dtype == torch.bool
    assert bool(mask.all())
    # First three points are the raw left arm joints.
    assert torch.equal(points[:, 0], poses[:, 0, 0])
    # Interpolated point 0 is 1/4 along the (0, 1) left-arm edge.
    expected = poses[:, 0, 0] + (poses[:, 0, 1] - poses[:, 0, 0]) * 0.25
    assert torch.allclose(points[:, 6], expected)


def test_densify_dual_hand_pose_nan_masking_and_padding() -> None:
    poses = _seeded_poses()
    poses[2, 0, 5] = float("nan")  # one left hand joint in frame 2
    poses[4] = float("nan")  # whole frame invalid

    points, mask = densify_dual_hand_pose(poses)

    assert points.shape == (6, 63, 3)
    assert int(mask[2].sum()) == 62
    assert int(mask[4].sum()) == 0
    # Invalid slots are zero-padded at the tail of each frame.
    assert torch.equal(points[2, 62], torch.zeros(3))
    assert torch.equal(points[4], torch.zeros(63, 3))
    assert not torch.isnan(points).any()


def test_densify_dual_hand_pose_all_nan_guard() -> None:
    poses = torch.full((3, 2, 24, 3), float("nan"))
    points, mask = densify_dual_hand_pose(poses)
    assert points.shape == (3, 1, 3)
    assert torch.equal(points, torch.zeros(3, 1, 3))
    assert not bool(mask.any())


def test_densify_dual_hand_pose_reuses_external_mask() -> None:
    poses = _seeded_poses()
    _, mask = densify_dual_hand_pose(poses)
    mask = mask.clone()
    mask[:, 10] = False
    points, reused = densify_dual_hand_pose(poses, mask)
    assert reused is mask
    assert points.shape == (6, 62, 3)


def test_densify_body_hand_frames_layout_and_z_scaling() -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(size=(4, 54, 3))

    densified = densify_body_hand_frames(data)

    assert densified.shape == (4, 63, 3)
    # First 6 points are body joints 5:11 with z scaled by 0.6.
    np.testing.assert_allclose(densified[:, :6, :2], data[:, 5:11, :2])
    np.testing.assert_allclose(densified[:, :6, 2], data[:, 5:11, 2] * 0.6)
    # Input is not mutated by the z scaling.
    assert not np.shares_memory(densified, data)


def test_temporal_smooth_decimate_shapes_and_determinism() -> None:
    rng = np.random.default_rng(9)
    points = rng.normal(size=(13, 63, 3))

    points_ds, velocities_ds = temporal_smooth_decimate(points)
    assert points_ds.shape == (4, 63, 3)
    assert velocities_ds.shape == (4, 63, 3)

    again_points, again_velocities = temporal_smooth_decimate(points)
    np.testing.assert_array_equal(points_ds, again_points)
    np.testing.assert_array_equal(velocities_ds, again_velocities)

    # Smoothing reduces temporal variance relative to the raw sequence.
    assert points_ds.var(axis=0).mean() < points[:-1][::3].var(axis=0).mean()
