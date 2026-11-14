from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor, nn

LegacyGraphLayout = Literal["body", "hand"]

_BODY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
)

_HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
)


def legacy_graph_edges(layout: LegacyGraphLayout) -> tuple[tuple[int, int], ...]:
    if layout == "body":
        return _BODY_EDGES
    if layout == "hand":
        return _HAND_EDGES
    raise ValueError(f"unsupported Legacy graph layout: {layout!r}")


def legacy_distance_adjacency(layout: LegacyGraphLayout) -> Tensor:
    """Build the two max-hop-one distance partitions used by Legacy WaveLLM."""

    node_count = 7 if layout == "body" else 21
    adjacency = torch.eye(node_count, dtype=torch.float32)
    for source, target in legacy_graph_edges(layout):
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    column_degree = adjacency.sum(dim=0)
    normalized = adjacency * column_degree.reciprocal().unsqueeze(0)
    partitions = torch.zeros((2, node_count, node_count), dtype=torch.float32)
    partitions[0] = normalized * torch.eye(node_count, dtype=torch.float32)
    partitions[1] = normalized - partitions[0]
    return partitions


class LegacyGraphConvolution(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        adjacency: Tensor,
    ) -> None:
        super().__init__()
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("adjacency must have shape [partition,node,node]")
        self.partition_count = int(adjacency.shape[0])
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels * self.partition_count,
            kernel_size=(1, 1),
            bias=True,
        )
        self.adjacency = nn.Parameter(adjacency.clone())
        self.normalization = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.convolution(inputs)
        batch, partitioned_channels, frames, nodes = projected.shape
        projected = projected.view(
            batch,
            self.partition_count,
            partitioned_channels // self.partition_count,
            frames,
            nodes,
        )
        aggregated = torch.einsum(
            "bkctv,kvw->bctw",
            projected,
            self.adjacency,
        )
        return cast(Tensor, self.activation(self.normalization(aggregated)))


class LegacySpatialTemporalGraphBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        temporal_kernel_size: int,
        adjacency: Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be within [0,1)")
        self.graph_convolution = LegacyGraphConvolution(
            input_channels,
            output_channels,
            adjacency,
        )
        if temporal_kernel_size > 1:
            self.temporal: nn.Module = nn.Sequential(
                nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=(temporal_kernel_size, 1),
                    stride=(1, 1),
                    padding=((temporal_kernel_size - 1) // 2, 0),
                    bias=True,
                ),
                nn.BatchNorm2d(output_channels),
                nn.Dropout(dropout, inplace=False),
            )
        else:
            self.temporal = nn.Identity()

        if input_channels == output_channels:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=(1, 1),
                    bias=True,
                ),
                nn.BatchNorm2d(output_channels),
            )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.residual(inputs)
        features = self.graph_convolution(inputs)
        return cast(Tensor, self.activation(self.temporal(features) + residual))


class LegacySTGCNChain(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: Sequence[int],
        *,
        temporal_kernel_size: int,
        adjacency: Tensor,
        dropout: float = 0.0,
    ) -> None:
        modules: list[nn.Module] = []
        current_channels = input_channels
        for next_channels in output_channels:
            modules.append(
                LegacySpatialTemporalGraphBlock(
                    current_channels,
                    next_channels,
                    temporal_kernel_size=temporal_kernel_size,
                    adjacency=adjacency,
                    dropout=dropout,
                )
            )
            current_channels = next_channels
        super().__init__(*modules)


class LegacyWaveLLMPoseEncoder(nn.Module):
    """Complete body/hand ST-GCN instantiated by the Legacy WaveLLM trainer."""

    coordinate_dim = 3
    joint_count = 24
    output_dim = 768

    def __init__(
        self,
        *,
        root_centering: bool = False,
        dropout: float = 0.0,
        confidence_branch: bool = False,
        pose_coordinate_dim: int = 3,
    ) -> None:
        super().__init__()
        body_adjacency = legacy_distance_adjacency("body")
        hand_adjacency = legacy_distance_adjacency("hand")

        self.root_centering = root_centering
        self.dropout = dropout
        self.pose_coordinate_dim = pose_coordinate_dim
        self.coordinate_dim = pose_coordinate_dim
        self.coordinate_projection = nn.Linear(pose_coordinate_dim, 64)
        # Graded-confidence branch. The incumbent encoder never reads `confidence` (it only
        # validates it); this opt-in branch turns the error-head value into per-joint feature
        # offsets, and its output layer starts at zero, so a warm-started run is bit-identical.
        self.confidence_branch: ConfidenceBranch | None = (
            ConfidenceBranch(64) if confidence_branch else None
        )
        self.spatial_chains = nn.ModuleDict(
            {
                "body": LegacySTGCNChain(
                    64,
                    (64, 128, 256),
                    temporal_kernel_size=1,
                    adjacency=body_adjacency,
                    dropout=dropout,
                ),
                "hand": LegacySTGCNChain(
                    64,
                    (64, 128, 256),
                    temporal_kernel_size=1,
                    adjacency=hand_adjacency,
                    dropout=dropout,
                ),
            }
        )
        self.temporal_chains = nn.ModuleDict(
            {
                "body": LegacySTGCNChain(
                    256,
                    (256, 256, 256),
                    temporal_kernel_size=5,
                    adjacency=body_adjacency,
                    dropout=dropout,
                ),
                "hand": LegacySTGCNChain(
                    256,
                    (256, 256, 256),
                    temporal_kernel_size=5,
                    adjacency=hand_adjacency,
                    dropout=dropout,
                ),
            }
        )
        self.part_para = nn.Parameter(torch.zeros(768))
        self.output_projection = nn.Linear(768, 768)

    def _validate_pose(self, pose: Tensor) -> None:
        if (
            pose.ndim != 5
            or pose.shape[2:4] != (2, 24)
            or pose.shape[4] != self.pose_coordinate_dim
        ):
            raise ValueError(
                f"pose must have shape [batch,time,2,24,{self.pose_coordinate_dim}], "
                f"got {tuple(pose.shape)}"
            )

    @staticmethod
    def _validate_confidence(pose: Tensor, confidence: Tensor) -> None:
        if confidence.ndim != 4 or confidence.shape[2:] != (2, 24):
            raise ValueError(
                f"confidence must have shape [batch,time,2,24], got {tuple(confidence.shape)}"
            )
        if confidence.shape[:2] != pose.shape[:2]:
            raise ValueError("pose and confidence batch/time dimensions must match")
        if torch.any((confidence < 0) | (confidence > 1)):
            raise ValueError("pose confidence must be within [0,1]")

    def decompose_pose(self, pose: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_pose(pose)
        left = pose[:, :, 0]
        right = pose[:, :, 1]
        body = torch.cat(
            (
                ((left[:, :, 0] + right[:, :, 0]) / 2).unsqueeze(2),
                left[:, :, :3],
                right[:, :, :3],
            ),
            dim=2,
        )
        return body, left[:, :, 3:], right[:, :, 3:]

    def encode_parts(
        self, pose: Tensor, confidence: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        body, left, right = self.decompose_pose(pose)
        projected = {
            "body": self.coordinate_projection(body).permute(0, 3, 1, 2),
            "left": self.coordinate_projection(left).permute(0, 3, 1, 2),
            "right": self.coordinate_projection(right).permute(0, 3, 1, 2),
        }
        if self.confidence_branch is not None and confidence is not None:
            left_conf, right_conf = confidence[:, :, 0], confidence[:, :, 1]
            parts_conf = {
                # body = [mean root, left wrist k0..2, right wrist k0..2], matching decompose_pose
                "body": torch.cat(
                    (
                        ((left_conf[:, :, 0] + right_conf[:, :, 0]) / 2).unsqueeze(2),
                        left_conf[:, :, :3],
                        right_conf[:, :, :3],
                    ),
                    dim=2,
                ),
                "left": left_conf[:, :, 3:],
                "right": right_conf[:, :, 3:],
            }
            for name, part_conf in parts_conf.items():
                offset = self.confidence_branch(part_conf, projected[name].dtype)
                projected[name] = projected[name] + offset.permute(0, 3, 1, 2)
        body_spatial = self.spatial_chains["body"](projected["body"])
        left_spatial = self.spatial_chains["hand"](projected["left"])
        right_spatial = self.spatial_chains["hand"](projected["right"])
        left_spatial = left_spatial + body_spatial[..., [3]].detach()
        right_spatial = right_spatial + body_spatial[..., [6]].detach()

        body_features = self.temporal_chains["body"](body_spatial)
        left_features = self.temporal_chains["hand"](left_spatial)
        right_features = self.temporal_chains["hand"](right_spatial)
        return (
            body_features.mean(dim=-1).transpose(1, 2),
            left_features.mean(dim=-1).transpose(1, 2),
            right_features.mean(dim=-1).transpose(1, 2),
        )

    def encode_pose(self, pose: Tensor, confidence: Tensor | None = None) -> Tensor:
        body, left, right = self.encode_parts(pose, confidence)
        combined = torch.cat((body, left, right), dim=-1) + self.part_para
        return cast(Tensor, self.output_projection(combined))

    def forward(self, pose: Tensor, confidence: Tensor) -> Tensor:
        self._validate_pose(pose)
        self._validate_confidence(pose, confidence)
        if self.root_centering:
            # Legacy dataset semantics: after per-axis normalization the pipeline
            # subtracted the per-frame mean of joint 0 across both hands. Because
            # the subtraction is applied to already normalized coordinates, the
            # corpus mean cancels and the composed legacy transform is exactly
            # (pose - root_mean(pose)) / std.
            geometry, extra = pose[..., :3], pose[..., 3:]
            geometry = geometry - geometry[:, :, :, [0], :].mean(dim=2, keepdim=True)
            pose = torch.cat((geometry, extra), dim=-1) if extra.shape[-1] else geometry
        return self.encode_pose(pose, confidence)


class ConfidenceBranch(nn.Module):
    """Zero-output residual branch that carries graded confidence into the encoder features.

    The branch leaves the coordinates alone and adds a learned offset derived only from the
    confidence value, so confidence reaches the temporal graph as an extra input signal. The
    final projection starts at zero, so a warm-started run reproduces the incumbent predictions
    at step zero.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        final = cast(nn.Linear, self.mlp[2])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, confidence: Tensor, dtype: torch.dtype) -> Tensor:
        return cast(Tensor, self.mlp(confidence.unsqueeze(-1).to(dtype=dtype)))
