"""Train-only, episode-keyed low-frequency rigid arm rotation in Yoke meters."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def sample_rotation(seed: int, epoch: int, index: int) -> dict[str, Any]:
    """Local generator does not consume initialization or sampler randomness."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
    enabled = bool(rng.random() < 0.5)
    baseline = rng.uniform([-5.0, -1.0, -1.0], [20.0, 1.0, 1.0])
    amplitude = rng.uniform(0, [5.0, 1.0, 1.0])
    weight = rng.uniform(0, 1, 3)
    return {
        "enabled": enabled,
        "baseline_deg": baseline.tolist(),
        "amplitudes_deg": np.stack([amplitude * weight, amplitude * (1 - weight)], 1).tolist(),
        "frequencies_hz": rng.uniform(0.1, 0.25, (3, 2)).tolist(),
        "phases_rad": rng.uniform(0, 2 * np.pi, (3, 2)).tolist(),
    }


def rotation_angles(
    parameters: dict[str, Any], timestamps: np.ndarray, *, episode_start: float
) -> np.ndarray:
    """Evaluate the full-episode curve; callers retain its origin when slicing."""
    seconds = np.asarray(timestamps, dtype=np.float64) - episode_start
    if not parameters["enabled"]:
        return np.zeros((len(seconds), 3), dtype=np.float64)
    phase = 2 * np.pi * seconds[:, None, None] * np.asarray(
        parameters["frequencies_hz"]
    ) + np.asarray(parameters["phases_rad"])
    return cast(
        np.ndarray,
        np.asarray(parameters["baseline_deg"])
        + (np.asarray(parameters["amplitudes_deg"]) * np.sin(phase)).sum(-1),
    )


def rotate_arms(root_pose: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rz(roll) Ry(yaw) Rx(pitch), shared across sides, around each shoulder."""
    if not np.any(angles_deg):
        return root_pose.copy()
    shoulder = root_pose[:, :, :1, :]
    p = np.asarray(root_pose, dtype=np.float64) - shoulder
    pitch, yaw, roll = np.deg2rad(angles_deg).T
    # Sequential coordinate rotations implement the declared column convention.
    for axis, angle in ((0, pitch), (1, yaw), (2, roll)):
        c, s = np.cos(angle)[:, None, None], np.sin(angle)[:, None, None]
        i, j = ((1, 2), (2, 0), (0, 1))[axis]
        a, b = p[..., i].copy(), p[..., j].copy()
        p[..., i], p[..., j] = c * a - s * b, s * a + c * b
    return cast(np.ndarray, (p + shoulder).astype(root_pose.dtype))
