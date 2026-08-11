from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn


def dual_hand_adjacency(joint_count: int = 24) -> Tensor:
    if joint_count != 24:
        raise ValueError("the canonical dual-hand graph requires 24 joints per hand")
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (3, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (3, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (3, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (3, 20),
        (20, 21),
        (21, 22),
        (22, 23),
    ]
    adjacency = torch.eye(joint_count, dtype=torch.float32)
    for source, target in edges:
        adjacency[source, target] = 1
        adjacency[target, source] = 1
    degree = adjacency.sum(dim=1).clamp_min(1)
    inverse_sqrt_degree = degree.rsqrt()
    return inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]


class SpatialTemporalGraphBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        temporal_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        self.spatial_projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.temporal_projection = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=(temporal_kernel_size, 1),
            padding=(temporal_kernel_size // 2, 0),
        )
        self.normalization = nn.BatchNorm2d(output_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual: nn.Module
        if input_channels == output_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor, adjacency: Tensor) -> Tensor:
        residual = self.residual(inputs)
        spatial = torch.einsum("bctv,vw->bctw", inputs, adjacency)
        spatial = self.spatial_projection(spatial)
        temporal = self.temporal_projection(spatial)
        activated = self.activation(self.dropout(self.normalization(temporal)) + residual)
        return cast(Tensor, activated)


class HandGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        coordinate_dim: int,
        joint_count: int,
        channels: Sequence[int],
        temporal_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if coordinate_dim < 1:
            raise ValueError("coordinate_dim must be positive")
        if not channels or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain positive dimensions")
        self.coordinate_dim = coordinate_dim
        self.joint_count = joint_count
        self.register_buffer("adjacency", dual_hand_adjacency(joint_count), persistent=True)
        blocks: list[nn.Module] = []
        input_channels = coordinate_dim
        for output_channels in channels:
            blocks.append(
                SpatialTemporalGraphBlock(
                    input_channels,
                    output_channels,
                    temporal_kernel_size=temporal_kernel_size,
                    dropout=dropout,
                )
            )
            input_channels = output_channels
        self.blocks = nn.ModuleList(blocks)

    @property
    def output_dim(self) -> int:
        block = self.blocks[-1]
        if not isinstance(block, SpatialTemporalGraphBlock):
            raise RuntimeError("hand graph encoder has an invalid final block")
        return block.temporal_projection.out_channels

    def forward(self, pose: Tensor) -> Tensor:
        if pose.ndim != 4:
            raise ValueError("hand pose must have shape [batch,time,joint,coordinate]")
        if pose.shape[2:] != (self.joint_count, self.coordinate_dim):
            raise ValueError(
                "hand pose trailing shape must be "
                f"[{self.joint_count},{self.coordinate_dim}], got {list(pose.shape[2:])}"
            )
        features = pose.permute(0, 3, 1, 2)
        for block in self.blocks:
            features = block(features, self.adjacency)
        return features.mean(dim=-1).transpose(1, 2)


class DualHandPoseEncoder(nn.Module):
    def __init__(
        self,
        *,
        coordinate_dim: int = 3,
        joint_count: int = 24,
        channels: Sequence[int] = (64, 128),
        output_dim: int = 768,
        temporal_kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.left_encoder = HandGraphEncoder(
            coordinate_dim=coordinate_dim,
            joint_count=joint_count,
            channels=channels,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.right_encoder = HandGraphEncoder(
            coordinate_dim=coordinate_dim,
            joint_count=joint_count,
            channels=channels,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.left_encoder.output_dim + self.right_encoder.output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.coordinate_dim = coordinate_dim
        self.joint_count = joint_count

    def forward(self, pose: Tensor, confidence: Tensor) -> Tensor:
        expected_pose_shape = (2, self.joint_count, self.coordinate_dim)
        expected_confidence_shape = (2, self.joint_count)
        if pose.ndim != 5 or pose.shape[2:] != expected_pose_shape:
            raise ValueError(
                "pose must have shape [batch,time,2,joint,coordinate] with trailing shape "
                f"{expected_pose_shape}, got {tuple(pose.shape)}"
            )
        if confidence.ndim != 4 or confidence.shape[2:] != expected_confidence_shape:
            raise ValueError(
                "confidence must have shape [batch,time,2,joint] with trailing shape "
                f"{expected_confidence_shape}, got {tuple(confidence.shape)}"
            )
        if pose.shape[:2] != confidence.shape[:2]:
            raise ValueError("pose and confidence batch/time dimensions must match")
        if torch.any((confidence < 0) | (confidence > 1)):
            raise ValueError("pose confidence must be within [0,1]")
        confidence_mask = confidence.unsqueeze(-1).to(dtype=pose.dtype)
        masked_pose = pose * confidence_mask
        left = self.left_encoder(masked_pose[:, :, 0])
        right = self.right_encoder(masked_pose[:, :, 1])
        output = self.output_projection(torch.cat((left, right), dim=-1))
        return cast(Tensor, output)
