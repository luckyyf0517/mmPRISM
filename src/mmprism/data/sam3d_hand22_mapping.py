"""SAM 3D Body 70-keypoint to frozen hand22 3D label mapping.

The SAM 3D Body estimator emits a 70-point wholebody keypoint set
(``sam_3d_body/metadata/mhr70.py``). Hand blocks: right hand slots 21-41
with the wrist at 41, left hand slots 42-62 with the wrist at 62; within
each finger the four slots run tip -> first -> second -> third joint.

The frozen Track B 3D label layout is ``[2, 22, 3]`` with side 0 = left,
side 1 = right, slot 0 the wrist global in the camera frame, and slots
1..21 the MANO-21 wrist-relative chain (slot 1 is the wrist itself, i.e. a
zero vector). This module maps camera-space 70-point arrays to that layout
with pure numpy so the mapping is CPU-testable without the provider env.

The per-finger ordering assumption (MHR "first joint" adjacent to the tip)
is observable: with the mapping above, proximal bones should on average be
longer than distal bones within a finger. ``finger_order_report`` measures
that monotonicity so a wrong ordering assumption is caught by a gate or an
eyeball check instead of silently swapping joints.
"""

from __future__ import annotations

MHR_RIGHT_WRIST = 41

MHR_LEFT_WRIST = 62

_FINGER_CHAINS_RIGHT: dict[str, tuple[int, int, int, int]] = {
    "thumb": (24, 23, 22, 21),
    "index": (28, 27, 26, 25),
    "middle": (32, 31, 30, 29),
    "ring": (36, 35, 34, 33),
    "pinky": (40, 39, 38, 37),
}

_FINGER_CHAINS_LEFT: dict[str, tuple[int, int, int, int]] = {
    "thumb": (45, 44, 43, 42),
    "index": (49, 48, 47, 46),
    "middle": (53, 52, 51, 50),
    "ring": (57, 56, 55, 54),
    "pinky": (61, 60, 59, 58),
}
