"""Pose-preserving fusion for signed contextual WaveLLM features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

POSE_ANCHORED_RESIDUAL_FEATURE_DIM = 1024

POSE_ANCHORED_RESIDUAL_HIDDEN_DIM = 768


class PoseAnchoredResidualFusionError(ValueError):
    """Raised when residual P+F tensors violate their declared contract."""


def _resolve_mask(mask: Tensor, *, batch: int, frames: int, device: torch.device) -> Tensor:
    if mask.ndim != 2 or tuple(mask.shape) != (batch, frames):
        raise PoseAnchoredResidualFusionError(
            f"frame_attention_mask must have shape [{batch},{frames}], got {tuple(mask.shape)}"
        )
    resolved = mask.to(device=device, dtype=torch.bool)
    if bool(torch.any(~torch.any(resolved, dim=1))):
        raise PoseAnchoredResidualFusionError("every sequence must contain one valid frame")
    return resolved


def _masked(value: Tensor, mask: Tensor) -> Tensor:
    return value * mask.to(dtype=value.dtype).unsqueeze(-1)


class PoseAnchoredResidualFeatureProjector(nn.Module):
    """Normalize signed F then map it into the mT5 embedding space."""

    input_dim = POSE_ANCHORED_RESIDUAL_FEATURE_DIM
    output_dim = POSE_ANCHORED_RESIDUAL_HIDDEN_DIM

    def __init__(self) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.input_projection = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.output_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)

    def forward(self, features: Tensor, frame_attention_mask: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise PoseAnchoredResidualFusionError(
                "contextual features must have shape "
                f"[batch,time,{self.input_dim}], got {tuple(features.shape)}"
            )
        batch, frames, _ = features.shape
        mask = _resolve_mask(
            frame_attention_mask,
            batch=batch,
            frames=frames,
            device=features.device,
        )
        normalized = _masked(self.input_normalization(_masked(features, mask)), mask)
        projected = _masked(self.input_projection(normalized), mask)
        return _masked(self.output_projection(functional.gelu(projected)), mask)


class PoseAnchoredResidualFusion(nn.Module):
    """Add a bounded learned F residual without replacing the pose route.

    The residual output map starts at zero.  That gives an exact P-only
    embedding at initialization while still allowing its output weights to
    receive a translation gradient on the first optimizer step.  A zero-start
    multiplicative scale would instead block the entire F path until that
    scalar first moved.
    """

    hidden_dim = POSE_ANCHORED_RESIDUAL_HIDDEN_DIM

    def __init__(self, *, dim: int = POSE_ANCHORED_RESIDUAL_HIDDEN_DIM) -> None:
        super().__init__()
        if dim != self.hidden_dim:
            raise PoseAnchoredResidualFusionError(
                f"pose-anchored residual fusion requires dim={self.hidden_dim}, got {dim}"
            )
        self.dim = dim
        self.gate_normalization = nn.LayerNorm(dim * 2)
        self.feature_gate = nn.Linear(dim * 2, dim)
        self.residual_hidden = nn.Linear(dim, dim, bias=False)
        self.residual_output = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.residual_output.weight)
        self.residual_gain_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        pose_embeddings: Tensor,
        feature_embeddings: Tensor,
        frame_attention_mask: Tensor,
    ) -> Tensor:
        if pose_embeddings.ndim != 3 or feature_embeddings.ndim != 3:
            raise PoseAnchoredResidualFusionError(
                "pose and feature embeddings must be rank-3 tensors"
            )
        if pose_embeddings.shape != feature_embeddings.shape:
            raise PoseAnchoredResidualFusionError(
                "pose and feature embeddings must have identical [batch,time,hidden] shape"
            )
        batch, frames, dim = pose_embeddings.shape
        if dim != self.dim:
            raise PoseAnchoredResidualFusionError(
                f"fusion embeddings must have hidden size {self.dim}, got {dim}"
            )
        mask = _resolve_mask(
            frame_attention_mask,
            batch=batch,
            frames=frames,
            device=pose_embeddings.device,
        )
        pose = _masked(pose_embeddings, mask)
        feature = _masked(feature_embeddings, mask)
        gate_input = self.gate_normalization(torch.cat((pose, feature), dim=-1))
        gate = _masked(torch.sigmoid(self.feature_gate(gate_input)), mask)
        residual = _masked(self.residual_hidden(gate * feature), mask)
        residual = _masked(self.residual_output(functional.gelu(residual)), mask)
        gain = torch.sigmoid(self.residual_gain_logit)
        return _masked(pose + gain * residual, mask)
