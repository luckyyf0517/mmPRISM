"""Bounded train-only spatial, parking/amplitude and temporal research packages."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.interpolate import PchipInterpolator

from mmprism.data.daily_hanging_arm import apply_hanging_angles, hanging_controls, hanging_spans
from mmprism.data.daily_pose_rotation import rotate_arms, rotation_angles, sample_rotation
from mmprism.evaluation.heldout_validity import hanging_hand_mask

PACKAGES = ("space", "parking", "time", "joint", "jointnospace", "jointsize", "jointpalm")


def unit(v: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


def axial(v: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    return cast(
        np.ndarray,
        v * c + np.cross(axis, v) * s + axis * (axis * v).sum(-1, keepdims=True) * (1 - c),
    )


def rotate_between(v: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Shortest rotation; deterministic orthogonal fallback for antipodal directions."""
    old, new = unit(old), unit(new)
    axis = np.cross(old, new)
    length = np.linalg.norm(axis, axis=-1)
    fallback = np.cross(old, np.eye(3)[np.argmin(np.abs(old), axis=-1)])
    axis = unit(np.where((length < 1e-10)[..., None], fallback, axis))
    angle = np.arctan2(length, np.clip((old * new).sum(-1), -1, 1))
    return axial(v, axis, angle)


def smooth_gate(mask: np.ndarray, times: np.ndarray, fade: float = 0.4) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=float)
    for side in (0, 1):
        for start, stop in hanging_spans(mask[:, side]):
            t = times[start:stop]
            g = np.ones(len(t))
            if start:
                g *= np.sin(np.pi / 2 * np.minimum((t - t[0]) / fade, 1)) ** 2
            if stop < len(times):
                g *= np.sin(np.pi / 2 * np.minimum((t[-1] - t) / fade, 1)) ** 2
            result[start:stop, side] = g
    return result


def reachable_scale(pose: np.ndarray, delta: np.ndarray) -> float:
    """One scale for both arms and ALL source times: no frame-wise clipping kinks."""
    upper = np.linalg.norm(pose[:, :, 1] - pose[:, :, 0], axis=-1)
    fore = np.linalg.norm(pose[:, :, 2] - pose[:, :, 1], axis=-1)
    v = pose[:, :, 2] - pose[:, :, 0]
    a, b = (delta * delta).sum(-1), 2 * (v * delta).sum(-1)
    distance2 = (v * v).sum(-1)
    # First exit from the annulus along the requested segment, including inward crossings.
    c = np.minimum(distance2 - (upper + fore) ** 2, 0)
    outer = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / np.maximum(2 * a, 1e-20)
    c = np.maximum(distance2 - (upper - fore) ** 2, 0)
    disc = b * b - 4 * a * c
    inner = (-b - np.sqrt(np.maximum(disc, 0))) / np.maximum(2 * a, 1e-20)
    limit = np.where((disc > 0) & (inner >= 0), np.minimum(outer, inner), outer)
    limit = np.where(a > 1e-20, limit, 1)
    return float(np.clip(0.98 * limit.min(), 0, 1))


def wrist_elbow_change(
    pose: np.ndarray, delta: np.ndarray, swivel: np.ndarray
) -> tuple[np.ndarray, float]:
    """Two fixed-length segments; transported source elbow-plane branch; hand translation."""
    p = np.asarray(pose, dtype=float)
    shoulder, elbow, wrist = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    scale = reachable_scale(p, delta)
    target = wrist + scale * delta
    old_d, new_d = wrist - shoulder, target - shoulder
    u, v = unit(old_d), unit(new_d)
    l1, l2 = np.linalg.norm(elbow - shoulder, axis=-1), np.linalg.norm(wrist - elbow, axis=-1)
    d = np.linalg.norm(new_d, axis=-1)
    along = (l1 * l1 - l2 * l2 + d * d) / np.maximum(2 * d, 1e-12)
    radius = np.sqrt(np.maximum(l1 * l1 - along * along, 0))
    radial = elbow - shoulder - u * ((elbow - shoulder) * u).sum(-1, keepdims=True)
    fallback = np.cross(u, np.eye(3)[np.argmin(np.abs(u), axis=-1)])
    radial = unit(np.where((np.linalg.norm(radial, axis=-1) < 1e-9)[..., None], fallback, radial))
    radial = rotate_between(radial, u, v)
    # Swivel alone displaces the elbow by no more than 3 cm.
    cap = 2 * np.arcsin(np.minimum(0.03 / np.maximum(2 * radius, 1e-12), 1))
    radial = axial(radial, v, np.clip(swivel, -cap, cap))
    result = p.copy()
    result[:, :, 1] = shoulder + along[..., None] * v + radius[..., None] * radial
    result[:, :, 2:] += (target - wrist)[:, :, None]
    return result, scale


def space_package(
    pose: np.ndarray,
    common_pose: np.ndarray,
    times: np.ndarray,
    rng: np.random.Generator,
    strength: float,
    *,
    source_pose: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    seconds = times - times[0]
    offset = rng.uniform([-0.02, -0.04, -0.025], [0.02, 0.02, 0.025])
    amplitude, freq, phase = (
        rng.uniform(0, 0.01, 3),
        rng.uniform(0.1, 0.25, 3),
        rng.uniform(0, 2 * np.pi, 3),
    )
    delta = strength * (offset + amplitude * np.sin(2 * np.pi * seconds[:, None] * freq + phase))
    delta = np.broadcast_to(delta[:, None], (len(pose), 2, 3))
    swivel = strength * np.deg2rad(
        rng.uniform(-10, 10, 2)
        + rng.uniform(0, 3, 2)
        * np.sin(
            2 * np.pi * seconds[:, None] * rng.uniform(0.1, 0.25, 2) + rng.uniform(0, 2 * np.pi, 2)
        )
    )
    moved, scale = wrist_elbow_change(pose, delta, swivel)
    # Conservative whole-episode contact guard, measured BEFORE per-side root centering.
    hand_distance = np.linalg.norm(
        common_pose[:, 0, 3:, None] - common_pose[:, 1, None, 3:], axis=-1
    )
    contact = bool(hand_distance.min() < 0.08)
    residual = rng.uniform([-3, -1, -1], [3, 1, 1], (2, 3)) * strength
    active = smooth_gate(~hanging_hand_mask(pose if source_pose is None else source_pose), times)
    if not contact:
        for side in (0, 1):
            moved[:, side : side + 1] = rotate_arms(
                moved[:, side : side + 1], active[:, side, None] * residual[side]
            )
    return moved, {"translation_scale": scale, "contact_residual_disabled": contact}


def angular_amplitude(pose: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Scale spherical angular departures, then rebuild lengths and rigidly carry hands."""
    p = np.asarray(pose, dtype=float)
    bones = np.stack([p[:, :, 1] - p[:, :, 0], p[:, :, 2] - p[:, :, 1]], axis=2)
    lengths = np.linalg.norm(bones, axis=-1, keepdims=True)
    direction = unit(bones)
    reference = unit(direction.mean(0))
    reference = np.where(
        (np.linalg.norm(reference, axis=-1) < 0.5)[..., None], direction[len(p) // 2], reference
    )
    reference = np.broadcast_to(reference, direction.shape)
    axis = np.cross(reference, direction)
    sine = np.linalg.norm(axis, axis=-1)
    fallback = np.cross(reference, np.eye(3)[np.argmin(np.abs(reference), axis=-1)])
    axis = unit(np.where((sine < 1e-10)[..., None], fallback, axis))
    angle = np.arctan2(sine, np.clip((reference * direction).sum(-1), -1, 1))
    new_direction = axial(reference, axis, angle * factor[:, :, None])
    result = p.copy()
    result[:, :, 1] = p[:, :, 0] + new_direction[:, :, 0] * lengths[:, :, 0]
    result[:, :, 2] = result[:, :, 1] + new_direction[:, :, 1] * lengths[:, :, 1]
    local = p[:, :, 3:] - p[:, :, 2:3]
    result[:, :, 3:] = (
        rotate_between(local, direction[:, :, 1, None], new_direction[:, :, 1, None])
        + result[:, :, 2:3]
    )
    return np.where((factor == 1)[:, :, None, None], p, result)


def parking_package(
    pose: np.ndarray,
    times: np.ndarray,
    key: tuple[int, int, int],
    rng: np.random.Generator,
    strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    controls = hanging_controls(pose, times, *key)
    gate = smooth_gate(controls["source_hanging"], times)
    old_gate = controls["weight"]
    ratio = np.divide(gate, old_gate, out=np.zeros_like(gate), where=old_gate > 0)
    parked = apply_hanging_angles(
        pose, strength * controls["bend_deg"] * ratio, strength * controls["yaw_deg"] * ratio
    )
    amplitude = 1 + strength * rng.uniform(-0.1, 0.1)
    active = smooth_gate(~controls["source_hanging"], times)
    dynamic = angular_amplitude(pose, 1 + (amplitude - 1) * active)
    result = np.where(controls["source_hanging"][:, :, None, None], parked, dynamic)
    return result, {"amplitude": amplitude, "fade_seconds": 0.4}


def time_map(times: np.ndarray, rng: np.random.Generator, strength: float) -> dict[str, Any]:
    """Monotone smooth progress map; nearest whole source frame retains exact hand geometry."""
    speed = 1 + strength * (rng.uniform(0.75, 1.15) - 1)
    duration = float(times[-1] - times[0])
    count = max(2, int(np.ceil(duration / speed * 10)) + 1)
    knots = np.linspace(0, 1, 5)
    warped = knots.copy()
    warped[1:-1] += rng.uniform(-0.03, 0.03, 3) * strength
    progress = PchipInterpolator(knots, warped)(np.linspace(0, 1, count))
    source_time = times[0] + progress * duration
    upper = np.clip(np.searchsorted(times, source_time), 0, len(times) - 1)
    lower = np.maximum(upper - 1, 0)
    index = np.where(source_time - times[lower] <= times[upper] - source_time, lower, upper)
    index[0], index[-1] = 0, len(times) - 1
    return {
        "speed": speed,
        "source_index": index,
        "source_time": source_time,
        "output_time": np.linspace(0, duration / speed, count),
        "progress_knots": warped,
    }


def transform(
    common_pose: np.ndarray, times: np.ndarray, key: tuple[int, int, int], package: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return per-side-root metres and source-frame mapping, before frozen affine."""
    if package not in PACKAGES:
        raise ValueError("unknown style package")
    base = sample_rotation(*key)
    pose = common_pose - common_pose[:, :, :1]
    indices = np.arange(len(pose))
    meta: dict[str, Any] = {"enabled": base["enabled"], "package": package}
    if not base["enabled"]:
        return pose, indices, meta
    rng = np.random.default_rng(np.random.SeedSequence([*key, 0x5354594C]))
    mode = (
        package
        if package not in ("joint", "jointnospace", "jointsize", "jointpalm")
        else str(rng.choice(["space", "parking", "time", "combined"]))
    )
    strength = 0.5 if mode == "combined" else 1.0
    meta.update(mode=mode, strength=strength)
    source = pose.copy()
    if mode in ("parking", "combined"):
        pose, info = parking_package(source, times, key, rng, strength)
        meta.update(info)
    if mode in ("space", "combined"):
        # In the joint arm, routing remains tied to source, not the already parked pose.
        moved, info = space_package(pose, common_pose, times, rng, strength, source_pose=source)
        # Consume the identical local draws so later timing controls remain paired.
        # This diagnostic removes geometry only, not the joint recipe's mixture schedule.
        if package != "jointnospace":
            pose = moved
        info["space_geometry_removed"] = package == "jointnospace"
        meta.update(info)
    angles = rotation_angles(base, times, episode_start=float(times[0]))
    pose = rotate_arms(pose, angles)  # Exactly once, including the combined mode.
    if package == "jointpalm":
        # Keep the original hand frame in the shoulder reference, while the wrist
        # follows the joint augmentation. Continuous source motion, no state switch.
        pose[:, :, 3:] = pose[:, :, 2:3] + (source[:, :, 3:] - source[:, :, 2:3])
        meta["hand_frame"] = "source_wrist_relative"
    if package == "jointsize":
        # Constant anatomical scale, not finger-pose noise. Separate stream leaves
        # the joint recipe's temporal draws and all upper-limb coordinates intact.
        size_rng = np.random.default_rng(np.random.SeedSequence([*key, 0x53495A45]))
        scale = float(size_rng.uniform(0.85, 1.15))
        pose[:, :, 3:] = pose[:, :, 2:3] + scale * (pose[:, :, 3:] - pose[:, :, 2:3])
        meta["hand_scale"] = scale
    if mode in ("time", "combined"):
        mapping = time_map(times, rng, strength)
        indices = mapping.pop("source_index")
        meta.update(mapping)
        pose = pose[indices]
    return pose, indices, meta
