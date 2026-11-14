"""SAM 3D Body 70-keypoint mapping to full24 ``[2,24,3]`` in the Yoke frame.

The shoulder-centred frame (token ``yoke``) places each frame's origin at
the bilateral shoulder midpoint and applies a yaw-only rotation that
squares the shoulder line to the image plane (both shoulders at equal
depth, the left->right vector along +x). Shoulder height asymmetry is
preserved — no roll, no pitch. The translation-only form (same origin,
camera axes) is ``yoke-t`` and is recovered exactly from the stored
midpoint; the applied yaw is stored alongside so camera-frame points
round-trip losslessly.

Joint layout per side (frozen full24 lineage, identical to the 2D full24
and the mesh L2 ``[2,24,3]``): slot 0 = that side's shoulder, slot 1 =
elbow, slot 2 = wrist, slots 3..23 = the 21 hand joints wrist-first
(slot 3 duplicates the wrist, mirroring the 2D builder's ``pose[:,3:]``
assignment). Side 0 = left, side 1 = right. Units follow the SAM3D
output (metres). The coordinate transforms use NumPy directly.
"""

from __future__ import annotations

import numpy as np

from mmprism.data.sam3d_hand22_mapping import (
    _FINGER_CHAINS_LEFT,
    _FINGER_CHAINS_RIGHT,
    MHR_LEFT_WRIST,
    MHR_RIGHT_WRIST,
)

FULL24_JOINT_COUNT = 24

POSE_SHAPE = (2, FULL24_JOINT_COUNT, 3)

MHR_LEFT_SHOULDER = 5

MHR_RIGHT_SHOULDER = 6

MHR_LEFT_ELBOW = 7

MHR_RIGHT_ELBOW = 8

_LEFT_SLOTS = (
    MHR_LEFT_SHOULDER,
    MHR_LEFT_ELBOW,
    MHR_LEFT_WRIST,
    MHR_LEFT_WRIST,
    *_FINGER_CHAINS_LEFT["thumb"],
    *_FINGER_CHAINS_LEFT["index"],
    *_FINGER_CHAINS_LEFT["middle"],
    *_FINGER_CHAINS_LEFT["ring"],
    *_FINGER_CHAINS_LEFT["pinky"],
)

_RIGHT_SLOTS = (
    MHR_RIGHT_SHOULDER,
    MHR_RIGHT_ELBOW,
    MHR_RIGHT_WRIST,
    MHR_RIGHT_WRIST,
    *_FINGER_CHAINS_RIGHT["thumb"],
    *_FINGER_CHAINS_RIGHT["index"],
    *_FINGER_CHAINS_RIGHT["middle"],
    *_FINGER_CHAINS_RIGHT["ring"],
    *_FINGER_CHAINS_RIGHT["pinky"],
)

_SIDE_SLOTS = (_LEFT_SLOTS, _RIGHT_SLOTS)

DEGENERATE_SPAN_M = 1e-6


class Sam3dFull24YokeError(RuntimeError):
    """Raised when a 70-point array cannot map/normalize to full24 yoke."""


def _yaw_rotation(applied_yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cos/sin of the applied yaw with the degenerate span forced to 0."""

    safe = np.where(np.isfinite(applied_yaw), applied_yaw, 0.0).astype(np.float64)
    return np.cos(safe), np.sin(safe), safe


def yoke_normalize(
    full24_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize absolute full24 points into the Yoke frame.

    Returns ``(pose_yoke, shoulder_midpoint_m, applied_yaw_rad)``:

    - ``pose_yoke`` ``[..., 2, 24, 3]`` float32 — ``R_y(psi) @ (X - c)`` per
      frame, where ``c`` is the per-frame bilateral shoulder midpoint and
      ``psi = atan2(d_z, d_x)`` is the azimuth of the left->right shoulder
      vector ``d``. After the rotation the shoulders sit at equal depth
      with the right shoulder along ``+x``; height asymmetry is untouched.
    - ``shoulder_midpoint_m`` ``[..., 3]`` float32 — the camera-frame
      origin ``c`` (``yoke-t`` is exactly ``X - c``).
    - ``applied_yaw_rad`` ``[...]`` float32 — the stored rotation angle;
      ``yoke_denormalize`` inverts the transform losslessly.

    Frames whose horizontal shoulder span falls below ``DEGENERATE_SPAN_M``
    keep the identity rotation (translation only) instead of amplifying
    noise.
    """

    array = np.asarray(full24_camera)
    if array.ndim < 3 or array.shape[-3:] != POSE_SHAPE:
        raise Sam3dFull24YokeError(f"expected [..., {POSE_SHAPE}] full24, got {array.shape}")
    left = array[..., 0, 0, :]  # left shoulder  [.., 3]
    right = array[..., 1, 0, :]  # right shoulder [.., 3]
    if not bool(np.isfinite(array).all()):
        raise Sam3dFull24YokeError("full24 points contain non-finite values")
    midpoint = ((left + right) * 0.5).astype(np.float32)
    delta = (right - left).astype(np.float64)
    span = np.hypot(delta[..., 0], delta[..., 2])
    applied_yaw = np.where(span >= DEGENERATE_SPAN_M, np.arctan2(delta[..., 2], delta[..., 0]), 0.0)
    cos_psi, sin_psi, _ = _yaw_rotation(applied_yaw)

    centered = array.astype(np.float64) - midpoint.astype(np.float64)[..., None, None, :]
    x = centered[..., 0]
    z = centered[..., 2]
    pose_yoke = np.empty_like(centered)
    pose_yoke[..., 0] = x * cos_psi[..., None, None] + z * sin_psi[..., None, None]
    pose_yoke[..., 1] = centered[..., 1]
    pose_yoke[..., 2] = -x * sin_psi[..., None, None] + z * cos_psi[..., None, None]
    return (
        pose_yoke.astype(np.float32),
        midpoint,
        applied_yaw.astype(np.float32),
    )
