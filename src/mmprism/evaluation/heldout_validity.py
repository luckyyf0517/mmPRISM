"""Per-hand hanging/rest detection in shoulder-centred coordinates.

A hand is hanging when its wrist lies below its elbow on the image-down
vertical axis. The returned mask identifies each hand independently and is
used by pose augmentation helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

WRIST_SLOT = 2

ELBOW_SLOT = 1

POSE_SHAPE = (2, 24, 3)


def hanging_hand_mask(labels: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Per-hand hanging/rest mask, ``[frames, 2]``.

    ``labels`` is the ``[frames, 2, 24, 3]`` Yoke-frame pose (metres or any
    consistent unit); a hand is hanging iff its wrist sits below its elbow
    (Yoke y is image-down positive, margin 0).
    """

    if labels.ndim != 4 or labels.shape[1:] != POSE_SHAPE:
        raise ValueError(f"labels must be [frames, 2, 24, 3], got {labels.shape}")
    return (labels[:, :, WRIST_SLOT, 1] - labels[:, :, ELBOW_SLOT, 1]) > 0
