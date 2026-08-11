from __future__ import annotations

from typing import NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


def _normalization_groups(channels: int) -> int:
    upper = min(32, max(1, channels // 4))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_normalization_groups(out_channels), out_channels)
        self.activation: nn.Module = nn.SiLU(inplace=True) if activate else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.activation(self.norm(self.conv(inputs))))


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.projection = nn.Sequential(
            nn.Conv3d(channels * 2, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        average = functional.adaptive_avg_pool3d(inputs, 1)
        maximum = functional.adaptive_max_pool3d(inputs, 1)
        weights = cast(Tensor, self.projection(torch.cat((average, maximum), dim=1)))
        return inputs * weights


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("spatial attention kernel_size must be positive and odd")
        self.projection = nn.Conv3d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        average = torch.mean(inputs, dim=1, keepdim=True)
        maximum = torch.amax(inputs, dim=1, keepdim=True)
        weights = torch.sigmoid(self.projection(torch.cat((average, maximum), dim=1)))
        return inputs * weights


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.projection = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weights = cast(
            Tensor,
            self.projection(functional.adaptive_avg_pool3d(inputs, 1)),
        )
        return inputs * weights


class DepthwiseResidualBlock3D(nn.Module):
    """Depthwise-separable residual block with independently ablatable attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        channel_attention: bool,
        spatial_attention: bool,
        se_attention: bool,
    ) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("residual stride must be 1 or 2")
        self.depthwise = ConvNormAct3D(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct3D(in_channels, out_channels, kernel_size=1, activate=False)
        self.channel_attention: nn.Module = (
            ChannelAttention3D(out_channels) if channel_attention else nn.Identity()
        )
        self.spatial_attention: nn.Module = (
            SpatialAttention3D() if spatial_attention else nn.Identity()
        )
        self.se_attention: nn.Module = (
            SqueezeExcitation3D(out_channels) if se_attention else nn.Identity()
        )
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvNormAct3D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activate=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        features = self.pointwise(self.depthwise(inputs))
        features = self.channel_attention(features)
        features = self.spatial_attention(features)
        features = self.se_attention(features)
        return functional.silu(features + residual)


class PathAggregationFPN3D(nn.Module):
    """Shape-safe top-down and bottom-up aggregation over 3D feature levels."""

    def __init__(self, stage_channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        if len(stage_channels) < 2:
            raise ValueError("PAFPN requires at least two feature levels")
        self.lateral = nn.ModuleList(
            ConvNormAct3D(channels, out_channels, kernel_size=1) for channels in stage_channels
        )
        self.top_down_refine = nn.ModuleList(
            DepthwiseResidualBlock3D(
                out_channels,
                out_channels,
                channel_attention=False,
                spatial_attention=False,
                se_attention=False,
            )
            for _ in range(len(stage_channels) - 1)
        )
        self.bottom_up_downsample = nn.ModuleList(
            ConvNormAct3D(out_channels, out_channels, stride=2)
            for _ in range(len(stage_channels) - 1)
        )
        self.bottom_up_refine = nn.ModuleList(
            DepthwiseResidualBlock3D(
                out_channels,
                out_channels,
                channel_attention=False,
                spatial_attention=False,
                se_attention=False,
            )
            for _ in range(len(stage_channels) - 1)
        )

    def forward(self, features: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if len(features) != len(self.lateral):
            raise ValueError(f"expected {len(self.lateral)} feature levels, got {len(features)}")
        lateral = [
            cast(Tensor, projection(level))
            for projection, level in zip(self.lateral, features, strict=True)
        ]
        top_down: list[Tensor] = [lateral[0]] * len(lateral)
        top_down[-1] = lateral[-1]
        for index in range(len(lateral) - 2, -1, -1):
            upsampled = functional.interpolate(
                top_down[index + 1],
                size=lateral[index].shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            top_down[index] = self.top_down_refine[index](lateral[index] + upsampled)

        bottom_up = [top_down[0]]
        for index in range(1, len(top_down)):
            downsampled = self.bottom_up_downsample[index - 1](bottom_up[-1])
            if downsampled.shape[-3:] != top_down[index].shape[-3:]:
                downsampled = functional.interpolate(
                    downsampled,
                    size=top_down[index].shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            bottom_up.append(self.bottom_up_refine[index - 1](top_down[index] + downsampled))
        return tuple(bottom_up)


class CubeNetSpatialEncoder(nn.Module):
    """Frame-wise 3D encoder for ``[B, doppler, range, azimuth, elevation]`` cubes."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        stage_channels: tuple[int, ...],
        stage_depths: tuple[int, ...],
        channel_attention: bool,
        spatial_attention: bool,
        se_attention: bool,
        use_pafpn: bool,
        fpn_channels: int,
    ) -> None:
        super().__init__()
        if in_channels < 1 or stem_channels < 1 or fpn_channels < 1:
            raise ValueError("CubeNet channel counts must be positive")
        if not stage_channels or len(stage_channels) != len(stage_depths):
            raise ValueError("stage_channels and stage_depths must be non-empty and aligned")
        if any(channels < 1 for channels in stage_channels) or any(
            depth < 1 for depth in stage_depths
        ):
            raise ValueError("CubeNet stage channels and depths must be positive")
        self.in_channels = in_channels
        self.attention_configuration = {
            "channel": channel_attention,
            "spatial": spatial_attention,
            "se": se_attention,
        }
        self.stem = ConvNormAct3D(in_channels, stem_channels, stride=2)
        stages: list[nn.Module] = []
        previous_channels = stem_channels
        for stage_index, (channels, depth) in enumerate(
            zip(stage_channels, stage_depths, strict=True)
        ):
            blocks: list[nn.Module] = []
            for block_index in range(depth):
                blocks.append(
                    DepthwiseResidualBlock3D(
                        previous_channels if block_index == 0 else channels,
                        channels,
                        stride=2 if stage_index > 0 and block_index == 0 else 1,
                        channel_attention=channel_attention,
                        spatial_attention=spatial_attention,
                        se_attention=se_attention,
                    )
                )
            stages.append(nn.Sequential(*blocks))
            previous_channels = channels
        self.stages = nn.ModuleList(stages)
        self.neck: PathAggregationFPN3D | None = (
            PathAggregationFPN3D(stage_channels, fpn_channels) if use_pafpn else None
        )
        self.feature_dim = fpn_channels if self.neck is not None else stage_channels[-1]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, radar_cube: Tensor) -> Tensor:
        if radar_cube.ndim != 5:
            raise ValueError(
                "CubeNetSpatialEncoder expects [batch,doppler,range,azimuth,elevation], "
                f"got {tuple(radar_cube.shape)}"
            )
        if radar_cube.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} Doppler channels, got {radar_cube.shape[1]}"
            )
        features = self.stem(radar_cube)
        levels: list[Tensor] = []
        for stage in self.stages:
            features = stage(features)
            levels.append(features)
        if self.neck is not None:
            features = self.neck(tuple(levels))[-1]
        return cast(Tensor, self.pool(features)).flatten(1)


class TemporalTransformerAggregator(nn.Module):
    """Mask-aware sequence aggregation with independently inspectable strategies."""

    def __init__(
        self,
        feature_dim: int,
        *,
        max_frames: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1 or max_frames < 1:
            raise ValueError("temporal layers and max_frames must be positive")
        if heads < 1 or feature_dim % heads != 0:
            raise ValueError("temporal heads must divide the spatial feature dimension")
        if feedforward_dim < feature_dim:
            raise ValueError("temporal feedforward_dim must be >= feature_dim")
        if not 0 <= dropout < 1:
            raise ValueError("temporal dropout must be in [0,1)")
        self.feature_dim = feature_dim
        self.max_frames = max_frames
        self.cls_token = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.position_embedding = nn.Parameter(torch.empty(1, max_frames + 1, feature_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(feature_dim),
            enable_nested_tensor=False,
        )
        self.attention_score = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1, bias=False)
        )
        self.strategy_logits = nn.Parameter(torch.zeros(3))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, frame_features: Tensor, frame_mask: Tensor) -> Tensor:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape [batch,time,feature]")
        batch, frames, feature_dim = frame_features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected temporal feature dimension {self.feature_dim}, got {feature_dim}"
            )
        if frames > self.max_frames:
            raise ValueError(f"received {frames} frames, maximum is {self.max_frames}")
        if frame_mask.shape != (batch, frames):
            raise ValueError("frame_mask must have shape [batch,time]")
        valid = frame_mask.to(device=frame_features.device, dtype=torch.bool)
        if bool(torch.any(valid.sum(dim=1) == 0)):
            raise ValueError("every sample must contain at least one valid frame")

        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat((cls, frame_features * valid.unsqueeze(-1)), dim=1)
        sequence = sequence + self.position_embedding[:, : frames + 1]
        padding_mask = torch.cat(
            (torch.zeros(batch, 1, dtype=torch.bool, device=valid.device), ~valid), dim=1
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        cls_feature = encoded[:, 0]
        temporal = encoded[:, 1:]

        mask_float = valid.unsqueeze(-1).to(dtype=temporal.dtype)
        mean_feature = (temporal * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        scores = self.attention_score(temporal).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention_feature = (temporal * attention).sum(dim=1)
        strategies = torch.stack((cls_feature, mean_feature, attention_feature), dim=1)
        weights = torch.softmax(self.strategy_logits, dim=0).view(1, 3, 1)
        return (strategies * weights).sum(dim=1)


class PoseReconstructionOutput(NamedTuple):
    joints: Tensor
    frame_features: Tensor
    sequence_features: Tensor


class OmniHandCubeNet(nn.Module):
    """Pure CubeNet pose regressor over explicit radar-cube tensors."""

    def __init__(
        self,
        spatial_encoder: CubeNetSpatialEncoder,
        temporal_aggregator: TemporalTransformerAggregator,
        *,
        joint_count: int = 24,
        coordinate_dim: int = 3,
    ) -> None:
        super().__init__()
        if joint_count != 24 or coordinate_dim != 3:
            raise ValueError("canonical OmniHand output must be [2,24,3]")
        if spatial_encoder.feature_dim != temporal_aggregator.feature_dim:
            raise ValueError("spatial and temporal feature dimensions must match")
        self.spatial_encoder = spatial_encoder
        self.temporal_aggregator = temporal_aggregator
        self.pose_head = nn.Sequential(
            nn.LayerNorm(spatial_encoder.feature_dim),
            nn.Linear(spatial_encoder.feature_dim, 2 * joint_count * coordinate_dim),
        )
        self.joint_count = joint_count
        self.coordinate_dim = coordinate_dim

    def _regress(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.pose_head(features)).reshape(
            features.shape[0], 2, self.joint_count, self.coordinate_dim
        )

    def forward_single_frame(self, radar_cube: Tensor) -> PoseReconstructionOutput:
        features = self.spatial_encoder(radar_cube)
        return PoseReconstructionOutput(
            joints=self._regress(features),
            frame_features=features.unsqueeze(1),
            sequence_features=features,
        )

    def forward(self, radar_cube: Tensor, frame_mask: Tensor) -> PoseReconstructionOutput:
        if radar_cube.ndim != 6:
            raise ValueError(
                "OmniHandCubeNet expects [batch,time,doppler,range,azimuth,elevation], "
                f"got {tuple(radar_cube.shape)}"
            )
        batch, frames, channels, range_bins, azimuth_bins, elevation_bins = radar_cube.shape
        flattened = radar_cube.reshape(
            batch * frames, channels, range_bins, azimuth_bins, elevation_bins
        )
        frame_features = self.spatial_encoder(flattened).reshape(batch, frames, -1)
        sequence_features = self.temporal_aggregator(frame_features, frame_mask)
        return PoseReconstructionOutput(
            joints=self._regress(sequence_features),
            frame_features=frame_features,
            sequence_features=sequence_features,
        )
