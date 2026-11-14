"""Source-gated elbow flexion and shoulder yaw, before shared low-frequency drift."""

from __future__ import annotations

import numpy as np

from mmprism.evaluation.heldout_validity import hanging_hand_mask


def hanging_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def span_weight(times: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Only internal boundaries fade; overlapping 0.3-second windows multiply."""
    t = np.asarray(times[start:stop], dtype=np.float64)
    weight = np.ones(len(t))
    if start > 0:
        weight *= np.sin(0.5 * np.pi * np.minimum((t - t[0]) / 0.3, 1)) ** 2
    if stop < len(times):
        weight *= np.sin(0.5 * np.pi * np.minimum((t[-1] - t) / 0.3, 1)) ** 2
    return weight


def elbow_geometry(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original bend plane; deterministic forward axis for near-straight arms."""
    original = np.asarray(pose, dtype=np.float64)
    upper, fore = original[:, :, 1] - original[:, :, 0], original[:, :, 2] - original[:, :, 1]
    ul, fl = np.linalg.norm(upper, axis=-1), np.linalg.norm(fore, axis=-1)
    valid = (ul > 1e-12) & (fl > 1e-12)
    un = upper / np.maximum(ul[..., None], 1e-12)
    fn = fore / np.maximum(fl[..., None], 1e-12)
    flex = np.rad2deg(np.arccos(np.clip((un * fn).sum(-1), -1, 1)))
    axis = np.cross(un, fn)
    unstable = np.linalg.norm(axis, axis=-1) < 1e-5
    forward = np.cross(un, np.array([0.0, 0.0, 1.0]))
    parallel = np.linalg.norm(forward, axis=-1) < 1e-5
    forward[parallel] = np.cross(un[parallel], np.array([1.0, 0.0, 0.0]))
    axis[unstable] = forward[unstable]
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    return axis, flex, valid


def hanging_controls(
    original: np.ndarray, times: np.ndarray, seed: int, epoch: int, index: int
) -> dict[str, np.ndarray]:
    """One source mask; side/span-keyed random streams never touch the base stream."""
    h = hanging_hand_mask(original)
    weight, requested, yaw = (np.zeros(h.shape) for _ in range(3))
    seconds = np.asarray(times, dtype=np.float64) - float(times[0])
    for side in (0, 1):
        for start, stop in hanging_spans(h[:, side]):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, epoch, index, 0x48414E47, side, start])
            )
            gate = span_weight(times, start, stop)
            weight[start:stop, side] = gate
            for target, limits, max_amplitude in (
                (requested, (15.0, 60.0), 5.0),
                (yaw, (-35.0, 35.0), 4.0),
            ):
                base = rng.uniform(*limits)
                amplitude, frequency, phase = (
                    rng.uniform(0, max_amplitude),
                    rng.uniform(0.1, 0.25),
                    rng.uniform(0, 2 * np.pi),
                )
                target[start:stop, side] = base + amplitude * np.sin(
                    2 * np.pi * frequency * seconds[start:stop] + phase
                )
    _, flex, valid = elbow_geometry(original)
    bend = np.minimum(requested, np.maximum(115 - flex, 0)) * weight * valid
    return {
        "source_hanging": h,
        "weight": weight,
        "requested_bend_deg": requested,
        "bend_deg": bend,
        "yaw_deg": yaw * weight,
    }


def apply_hanging_angles(
    original: np.ndarray, bend_deg: np.ndarray, yaw_deg: np.ndarray
) -> np.ndarray:
    """Rigid forearm/hand elbow rotation, then per-side whole-arm shoulder yaw."""
    result = original.copy()
    axis, _, _ = elbow_geometry(original)
    active = bend_deg != 0
    if active.any():
        elbows = np.asarray(original[:, :, 1][active], dtype=np.float64)
        v = np.asarray(original[:, :, 2:][active], dtype=np.float64) - elbows[:, None]
        a = axis[active][:, None]
        theta = np.deg2rad(bend_deg[active])[:, None, None]
        rotated = (
            v * np.cos(theta)
            + np.cross(a, v) * np.sin(theta)
            + a * (v * a).sum(-1, keepdims=True) * (1 - np.cos(theta))
        )
        result[:, :, 2:][active] = rotated + elbows[:, None]
    active = yaw_deg != 0
    if active.any():
        shoulder = np.asarray(result[:, :, 0][active], dtype=np.float64)
        v = np.asarray(result[:, :, 1:][active], dtype=np.float64) - shoulder[:, None]
        theta = np.deg2rad(yaw_deg[active])[:, None]
        c, s = np.cos(theta), np.sin(theta)
        x, z = v[..., 0].copy(), v[..., 2].copy()
        v[..., 0], v[..., 2] = c * x + s * z, -s * x + c * z
        result[:, :, 1:][active] = v + shoulder[:, None]
    return result
