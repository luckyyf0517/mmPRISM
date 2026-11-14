"""Torch palm-frame rotations and the two single-target hand auxiliaries.

Torch mirror of :mod:`mmprism.training.reconstruction.hand_palm_frame` for
training losses.  The frame construction is identical to the numpy
diagnostic version (wrist origin, finger-MCP forward, index-to-pinky
lateral, cross-product normal, Gram-Schmidt), so counterfactual diagnostics
and the trainable target share one definition.  Reference branches are
detached: the auxiliary pushes the prediction toward the reference, never
the reverse.

Losses operate on full24 arrays ``[B, T, 2, 24, 3]`` with a ``[B, T]``
frame-validity mask and return scalar means over applied (valid, definable)
entries per side.
"""

from __future__ import annotations

import torch
from torch import Tensor

WRIST_SLOT = 2

PALM_MCP_HAND20_INDICES = (0, 4, 8, 12)

_EPSILON_M = 1e-6


def palm_rotation_torch(points: Tensor) -> tuple[Tensor, Tensor]:
    """Per-frame palm rotations for one hand ``[N, 24, 3]``.

    Returns ``(R, definable)`` with ``R`` ``[N, 3, 3]`` mapping palm-local
    column vectors to camera coordinates and ``definable`` a bool ``[N]``
    mask.  Differentiable in ``points``.
    """

    wrist = points[:, WRIST_SLOT]
    mcp = torch.stack([points[:, 4 + index] for index in PALM_MCP_HAND20_INDICES], dim=1)
    forward0 = mcp.mean(dim=1) - wrist
    lateral0 = mcp[:, 3] - mcp[:, 0]
    normal0 = torch.linalg.cross(forward0, lateral0)
    norms = torch.stack(
        [
            torch.linalg.vector_norm(forward0, dim=-1),
            torch.linalg.vector_norm(lateral0, dim=-1),
            torch.linalg.vector_norm(normal0, dim=-1),
        ],
        dim=-1,
    )
    finite = torch.isfinite(points).all(dim=-1).all(dim=-1)
    definable = finite & (norms > _EPSILON_M).all(dim=-1)
    safe = torch.where(norms > _EPSILON_M, norms, torch.ones_like(norms))
    forward = forward0 / safe[:, 0:1]
    normal = normal0 / safe[:, 2:3]
    lateral = torch.linalg.cross(normal, forward)
    rotation = torch.stack([forward, lateral, normal], dim=-1)
    return rotation, definable


def _local_hand20(points: Tensor, rotation: Tensor) -> Tensor:
    """The twenty hand points (slots 4..24) in the palm frame."""

    wrist = points[:, WRIST_SLOT : WRIST_SLOT + 1]
    local_camera = points[:, WRIST_SLOT + 2 :] - wrist
    return torch.einsum("nij,nkj->nki", rotation.transpose(1, 2), local_camera)


def _side_frames(full24: Tensor) -> tuple[Tensor, Tensor]:
    left, right = full24[:, :, 0], full24[:, :, 1]
    batch, frames = left.shape[0], left.shape[1]
    left = left.reshape(batch * frames, *left.shape[2:])
    right = right.reshape(batch * frames, *right.shape[2:])
    return left, right


def intrinsic_unit_bone_cosine_loss(
    prediction_full24: Tensor, target_full24: Tensor, valid: Tensor
) -> Tensor:
    """Mean cosine distance between corresponding unit bone vectors.

    Bones live in each hand's own palm frame (predicted bones in the
    predicted frame, reference bones in the reference frame), so the loss
    sees intrinsic finger configuration, not whole-hand rotation.  The
    skeleton is the established five-finger chain (wrist + four joints per
    finger, twenty bones).  Reference entries are detached.
    """

    prediction_sides = _side_frames(prediction_full24)
    target_sides = _side_frames(target_full24)
    valid_flat = valid.reshape(-1).to(dtype=torch.bool)
    total = prediction_full24.new_zeros(())
    count = prediction_full24.new_zeros((), dtype=torch.int64)
    for predicted, reference in zip(prediction_sides, target_sides):
        rotation_pred, definable_pred = palm_rotation_torch(predicted)
        with torch.no_grad():
            rotation_ref, definable_ref = palm_rotation_torch(reference)
        applied = valid_flat & definable_pred & definable_ref
        if not bool(applied.any()):
            continue
        local_pred = _local_hand20(predicted, rotation_pred)
        with torch.no_grad():
            local_ref = _local_hand20(reference, rotation_ref)
        pred_bones = _unit_bones(local_pred)
        ref_bones = _unit_bones(local_ref)
        cosine = 1.0 - (pred_bones * ref_bones).sum(dim=-1)
        safe_applied = applied.to(dtype=cosine.dtype)[:, None]
        total = total + (cosine * safe_applied).sum()
        count = count + applied.to(dtype=torch.int64).sum() * cosine.shape[-1]
    if int(count) == 0:
        raise ValueError("intrinsic-shape auxiliary has no applied entries")
    return total / count.to(dtype=total.dtype)


def _unit_bones(local_hand20: Tensor) -> Tensor:
    """Unit bone vectors of the corrected five-finger skeleton.

    ``local_hand20`` is ``[N, 20, 3]`` (slots 4..24).  With the wrist
    prepended as joint 0, finger ``f`` chains ``[0, 1+4f, 2+4f, 3+4f,
    4+4f]`` (plan section 13.2): every finger starts at the wrist, giving
    twenty bones.  Returns ``[N, 20, 3]`` unit vectors.
    """

    points = torch.cat([local_hand20.new_zeros(local_hand20.shape[0], 1, 3), local_hand20], dim=1)
    bones = []
    for finger in range(5):
        chain = torch.tensor(
            [0, 1 + 4 * finger, 2 + 4 * finger, 3 + 4 * finger, 4 + 4 * finger],
            device=points.device,
            dtype=torch.long,
        )
        bones.append(points[:, chain[1:]] - points[:, chain[:-1]])
    stacked = torch.cat(bones, dim=1)
    norm = torch.linalg.vector_norm(stacked, dim=-1, keepdim=True)
    return stacked / torch.where(norm > _EPSILON_M, norm, torch.ones_like(norm))
