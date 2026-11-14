"""Reader-side local hand geometry and fixed-window motion residual branches.

The left/right-shared geometry branch uses wrist-relative hand joints and bone
vectors. Its optional motion companion uses a five-frame mean difference.
Zero-initialized output layers preserve the initial pretrained predictions.
The motion branch uses two future frames within the valid prefix, corresponding
to 200 milliseconds of look-ahead at ten frames per second.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

WRIST_SLOT: int = 2

HAND20_START: int = 4

HAND20_COUNT: int = 20

HAND_FEATURE_DIM: int = 2 * 3 * HAND20_COUNT  # wrist-relative joints + bone vectors

MOTION_CONTEXT: int = 2


def hand_geometry_features(pose: Tensor) -> Tensor:
    """Per-hand wrist-relative joints and bone vectors from a pose tensor.

    ``pose`` is the reader input ``[B,T,2,24,3]`` in its normalized space.
    Returns ``[B,T,2,120]``: slots ``0..59`` hold the twenty wrist-relative
    joint vectors, slots ``60..119`` hold the bone vectors (chain children
    minus parents; finger bases are wrist-relative by construction).
    """

    if pose.ndim != 5 or tuple(pose.shape[2:]) != (2, 24, 3):
        raise ValueError(f"reader pose must be [B,T,2,24,3], got {tuple(pose.shape)}")
    wrist = pose[:, :, :, WRIST_SLOT : WRIST_SLOT + 1, :]
    local = pose[:, :, :, HAND20_START : HAND20_START + HAND20_COUNT, :] - wrist
    previous = torch.cat((torch.zeros_like(local[:, :, :, :1]), local[:, :, :, :-1]), dim=3)
    base = torch.zeros(1, 1, 1, HAND20_COUNT, 1, dtype=pose.dtype, device=pose.device)
    base[..., 0::4, :] = 1.0
    bones = local * base + (local - previous) * (1.0 - base)
    return torch.cat((local, bones), dim=3).reshape(*pose.shape[:2], 2, HAND_FEATURE_DIM)


def motion_features(features: Tensor, frame_mask: Tensor) -> Tensor:
    """Fixed five-frame mean-difference motion of per-hand geometry features.

    ``features`` is ``[B,T,2,F]``; ``frame_mask`` is the bool ``[B,T]`` valid
    prefix.  For each valid center frame the motion is
    ``mean(features[t-2], features[t-1]) - mean(features[t+1], features[t+2])``
    with indices clamped into the sample's valid prefix (edge hold at segment
    boundaries).  Rows beyond a sample's valid length are zeroed.
    """

    if features.ndim != 4 or frame_mask.shape != features.shape[:2]:
        raise ValueError("motion features/mask shape differs")
    if frame_mask.dtype != torch.bool:
        raise ValueError("motion frame mask must be bool")
    if not bool(torch.all(frame_mask[:, 1:] <= frame_mask[:, :-1])):
        raise ValueError("motion branch requires a prefix-contiguous frame mask")
    if not bool(frame_mask.any(dim=1).all()):
        raise ValueError("motion branch requires one valid frame per sample")
    frames = frame_mask.shape[1]
    lengths = frame_mask.sum(dim=1, dtype=torch.long)
    position = torch.arange(frames, device=features.device)[None, :].expand(frame_mask.shape[0], -1)
    early_first = (position - MOTION_CONTEXT).clamp(min=0)
    early_second = (position - MOTION_CONTEXT + 1).clamp(min=0)
    late_limit = (lengths - 1)[:, None]
    late_first = torch.minimum(position + 1, late_limit)
    late_second = torch.minimum(position + 2, late_limit)

    def _gather(index: Tensor) -> Tensor:
        expanded = index[:, :, None, None].expand(-1, -1, *features.shape[2:])
        return features.gather(1, expanded)

    motion = 0.5 * (_gather(early_first) + _gather(early_second)) - 0.5 * (
        _gather(late_first) + _gather(late_second)
    )
    return motion * frame_mask[:, :, None, None].to(dtype=motion.dtype)


class _SharedHandTrunk(nn.Module):
    """Left/right-shared small trunk over one hand's geometry features."""

    def __init__(self, trunk_width: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, trunk_width),
            nn.SiLU(),
            nn.Linear(trunk_width, trunk_width),
            nn.SiLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class LocalHandGeometryBranch(nn.Module):
    """R-G: per-frame local hand geometry residual on the pose embedding."""

    def __init__(self, hidden_size: int = 768, trunk_width: int = 256) -> None:
        super().__init__()
        self.trunk = _SharedHandTrunk(trunk_width)
        self.output = nn.Linear(2 * trunk_width, hidden_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pose: Tensor, frame_mask: Tensor) -> Tensor:
        features = hand_geometry_features(pose)
        per_hand = self.trunk(features)
        fused = torch.cat((per_hand[:, :, 0], per_hand[:, :, 1]), dim=-1)
        return self.output(fused) * frame_mask[:, :, None].to(dtype=fused.dtype)


class LocalHandGeometryMotionBranch(nn.Module):
    """R-M: the R-G residual plus a fixed five-frame motion residual."""

    def __init__(self, hidden_size: int = 768, trunk_width: int = 256) -> None:
        super().__init__()
        self.geometry = LocalHandGeometryBranch(hidden_size, trunk_width)
        self.motion_trunk = _SharedHandTrunk(trunk_width)
        self.motion_output = nn.Linear(2 * trunk_width, hidden_size)
        nn.init.zeros_(self.motion_output.weight)
        nn.init.zeros_(self.motion_output.bias)

    def forward(self, pose: Tensor, frame_mask: Tensor) -> Tensor:
        residual = self.geometry(pose, frame_mask)
        features = hand_geometry_features(pose)
        motion = motion_features(features, frame_mask)
        per_hand = self.motion_trunk(motion)
        fused = torch.cat((per_hand[:, :, 0], per_hand[:, :, 1]), dim=-1)
        return residual + self.motion_output(fused) * frame_mask[:, :, None].to(dtype=fused.dtype)


def build_reader_geometry_branch(kind: str, *, hidden_size: int = 768) -> nn.Module:
    if kind == "local_geometry":
        return LocalHandGeometryBranch(hidden_size)
    if kind == "local_geometry_plus_motion":
        return LocalHandGeometryMotionBranch(hidden_size)
    raise ValueError(f"unknown reader geometry branch kind: {kind!r}")
