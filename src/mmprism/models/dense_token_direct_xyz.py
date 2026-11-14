"""Direct-coordinate dense-token decoder.

The module preserves the dense radar-token trunk exactly: four
per-frame double-stream blocks, deterministic 16-region pooling, and eight
cross-time joint-stream blocks.  It changes only the readout and emits camera
global coarse joints plus wrist-relative hand points directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch
from torch import Tensor, nn

from mmprism.models.dense_token_deterministic_mhr import (
    DENSE_TOKEN_CHANNELS,
    DENSE_TOKEN_COUNT,
    PATCH_GRID_SIZE,
    QUERY_COUNT,
    REGIONAL_GRID_SIZE,
    REGIONAL_TOKEN_COUNT,
    _attention_from_qkv,
    _DoubleStreamBlock,
    _SingleStreamBlock,
)

COARSE_JOINT_COUNT: Final[int] = 3

COARSE_COORDINATE_DIMENSION: Final[int] = 2 * COARSE_JOINT_COUNT * 3

HAND_POINT_COUNT: Final[int] = 20

HAND_COORDINATE_DIMENSION: Final[int] = HAND_POINT_COUNT * 3

FULL24_JOINT_COUNT: Final[int] = 24

TEMPORAL_ADAPTER_HIDDEN_CHANNELS: Final[int] = 128

TEMPORAL_ADAPTER_ATTENTION_HEADS: Final[int] = 4

TemporalAdapterPlacement = Literal["none", "prequery", "regional", "dense"]


class DenseTokenDirectXyzError(ValueError):
    """Raised when direct-coordinate tensors violate their fixed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DenseTokenDirectXyzError(message)


@dataclass(frozen=True)
class DirectXyzPrediction:
    """Coordinate heads and their deterministic full24 assembly."""

    coarse18_m: Tensor
    hand120_wrist_relative_m: Tensor
    full24_m: Tensor


class DenseTokenTemporalSlotAdapter(nn.Module):
    """Apply shared mask-safe temporal attention independently to each spatial slot.

    The adapter is intentionally position-agnostic across slots: one set of
    parameters operates on all regional or dense slots, while the caller keeps
    the existing spatial embeddings.  Its zero-initialized final projection
    makes an inserted adapter an exact identity at initialization.
    """

    def __init__(
        self,
        *,
        hidden_channels: int,
        bottleneck_channels: int = TEMPORAL_ADAPTER_HIDDEN_CHANNELS,
        attention_heads: int = TEMPORAL_ADAPTER_ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        _require(hidden_channels > 0, "temporal adapter hidden width must be positive")
        _require(
            bottleneck_channels > 0 and bottleneck_channels % attention_heads == 0,
            "temporal adapter bottleneck must divide attention heads",
        )
        self.hidden_channels = hidden_channels
        self.bottleneck_channels = bottleneck_channels
        self.attention_heads = attention_heads
        self.input_normalization = nn.LayerNorm(hidden_channels)
        self.input_projection = nn.Linear(hidden_channels, bottleneck_channels)
        self.attention_normalization = nn.LayerNorm(bottleneck_channels)
        self.attention_qkv = nn.Linear(bottleneck_channels, bottleneck_channels * 3)
        self.attention_output = nn.Linear(bottleneck_channels, bottleneck_channels)
        self.ffn_normalization = nn.LayerNorm(bottleneck_channels)
        self.ffn_expand = nn.Linear(bottleneck_channels, bottleneck_channels * 2)
        self.ffn_activation = nn.GELU()
        self.ffn_contract = nn.Linear(bottleneck_channels * 2, bottleneck_channels)
        self.output_projection = nn.Linear(bottleneck_channels, hidden_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, values: Tensor, *, valid_mask: Tensor) -> Tensor:
        """Return per-slot temporal residuals without admitting padded frames."""

        _require(
            values.ndim == 4 and values.shape[-1] == self.hidden_channels,
            "temporal adapter values must be [B,T,S,H]",
        )
        batch, frames, slots, _ = values.shape
        _require(
            valid_mask.shape == (batch, frames) and valid_mask.dtype == torch.bool,
            "temporal adapter valid mask differs",
        )
        _require(bool(valid_mask.any(dim=1).all()), "every temporal adapter sample needs a frame")
        frame_weights = valid_mask[:, :, None, None].to(dtype=values.dtype)
        source = values * frame_weights
        flattened = source.transpose(1, 2).reshape(batch * slots, frames, self.hidden_channels)
        slot_mask = valid_mask[:, None, :].expand(-1, slots, -1).reshape(batch * slots, frames)
        positions = torch.arange(frames, device=values.device).expand(batch * slots, -1)
        hidden = self.input_projection(self.input_normalization(flattened))
        attention = _attention_from_qkv(
            self.attention_qkv(self.attention_normalization(hidden)),
            attention_heads=self.attention_heads,
            valid_mask=slot_mask,
            positions=positions,
        )
        hidden = hidden + self.attention_output(attention)
        hidden = hidden + self.ffn_contract(
            self.ffn_activation(self.ffn_expand(self.ffn_normalization(hidden)))
        )
        output: Tensor = (flattened + self.output_projection(hidden)) * slot_mask.unsqueeze(-1).to(
            dtype=values.dtype
        )
        return output.reshape(batch, slots, frames, self.hidden_channels).transpose(1, 2)


class DenseTokenPartDenseRouter(nn.Module):
    """Route each motion query softly over the complete dense radar grid.

    The inherited decoder keeps its regional-mean path.  This module only adds
    a query-conditioned summary back to the corresponding upper/left/right
    query.  Its output projection starts at zero, so an inserted router is an
    exact no-op at initialization rather than an alternate crop or hard ROI.
    """

    def __init__(self, *, hidden_channels: int) -> None:
        super().__init__()
        _require(hidden_channels > 0, "part router hidden width must be positive")
        self.hidden_channels = hidden_channels
        self.query_normalization = nn.LayerNorm(hidden_channels)
        self.token_normalization = nn.LayerNorm(hidden_channels)
        self.query_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.key_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.value_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.output_projection = nn.Linear(hidden_channels, hidden_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, queries: Tensor, radar: Tensor, *, valid_mask: Tensor) -> Tensor:
        _require(
            queries.ndim == 4 and queries.shape[-2:] == (QUERY_COUNT, self.hidden_channels),
            "part router queries must be [B,T,3,H]",
        )
        _require(
            radar.ndim == 4 and radar.shape[-2:] == (DENSE_TOKEN_COUNT, self.hidden_channels),
            "part router radar must be [B,T,256,H]",
        )
        _require(
            queries.shape[:2] == radar.shape[:2] == valid_mask.shape
            and valid_mask.dtype == torch.bool,
            "part router temporal mask differs",
        )
        projected_queries = self.query_projection(self.query_normalization(queries))
        normalized_radar = self.token_normalization(radar)
        keys = self.key_projection(normalized_radar)
        values = self.value_projection(normalized_radar)
        scale = float(self.hidden_channels) ** -0.5
        logits = torch.einsum("btqh,btdh->btqd", projected_queries, keys) * scale
        weights = torch.softmax(logits, dim=-1)
        summary = torch.einsum("btqd,btdh->btqh", weights, values)
        frame_weights = valid_mask[:, :, None, None].to(dtype=queries.dtype)
        routed: Tensor = (queries + self.output_projection(summary)) * frame_weights
        return routed


class DenseTokenAdaptiveRegionalPool(nn.Module):
    """Keep 16 regional means while permitting a zero-tax local reweighting."""

    def __init__(self, *, hidden_channels: int, score_hidden_channels: int = 64) -> None:
        super().__init__()
        _require(hidden_channels > 0, "adaptive pooling hidden width must be positive")
        _require(score_hidden_channels > 0, "adaptive pooling score width must be positive")
        self.hidden_channels = hidden_channels
        self.score_normalization = nn.LayerNorm(hidden_channels)
        self.score_expand = nn.Linear(hidden_channels, score_hidden_channels)
        self.score_activation = nn.GELU()
        self.score_output = nn.Linear(score_hidden_channels, 1)
        nn.init.zeros_(self.score_output.weight)
        nn.init.zeros_(self.score_output.bias)

    def forward(self, radar: Tensor) -> Tensor:
        _require(
            radar.ndim == 4 and radar.shape[-2:] == (DENSE_TOKEN_COUNT, self.hidden_channels),
            "adaptive pooling radar must be [B,T,256,H]",
        )
        batch, frames = radar.shape[:2]
        per_region = PATCH_GRID_SIZE // REGIONAL_GRID_SIZE
        slots = (
            radar.reshape(
                batch,
                frames,
                REGIONAL_GRID_SIZE,
                per_region,
                REGIONAL_GRID_SIZE,
                per_region,
                self.hidden_channels,
            )
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(
                batch, frames, REGIONAL_TOKEN_COUNT, per_region * per_region, self.hidden_channels
            )
        )
        mean = slots.mean(dim=3)
        scores = self.score_output(
            self.score_activation(self.score_expand(self.score_normalization(slots)))
        )
        weights = torch.softmax(scores, dim=3)
        # At the zero-initialized final score layer, delta is exactly zero
        # (1/16 is binary-representable), retaining the inherited mean bitwise.
        uniform = 1.0 / float(per_region * per_region)
        weighted_mean = mean + ((weights - uniform) * slots).sum(dim=3)
        pooled: Tensor = mean + 0.5 * (weighted_mean - mean)
        return pooled


class DenseTokenHandResidualHead(nn.Module):
    """A zero-initialized nonlinear residual readout for one hand query."""

    def __init__(self, *, hidden_channels: int, bottleneck_channels: int = 128) -> None:
        super().__init__()
        _require(hidden_channels > 0 and bottleneck_channels > 0, "hand residual width differs")
        self.expand = nn.Linear(hidden_channels, bottleneck_channels)
        self.activation = nn.GELU()
        self.output = nn.Linear(bottleneck_channels, HAND_COORDINATE_DIMENSION)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: Tensor) -> Tensor:
        _require(values.ndim == 3, "hand residual values must be [B,T,H]")
        residual: Tensor = self.output(self.activation(self.expand(values)))
        return residual


class DenseTokenDirectXyzDecoder(nn.Module):
    """The dense temporal trunk with direct coarse and hand coordinate heads."""

    def __init__(
        self,
        *,
        token_channels: int = DENSE_TOKEN_CHANNELS,
        hidden_channels: int = 384,
        attention_heads: int = 8,
        mlp_ratio: int = 4,
        double_stream_blocks: int = 4,
        fused_single_stream_blocks: int = 8,
        maximum_frames: int = 32,
        temporal_adapter_placement: TemporalAdapterPlacement = "none",
        temporal_adapter_bottleneck_channels: int = TEMPORAL_ADAPTER_HIDDEN_CHANNELS,
        temporal_adapter_attention_heads: int = TEMPORAL_ADAPTER_ATTENTION_HEADS,
        part_dense_routing: bool = False,
        adaptive_regional_pooling: bool = False,
        nonlinear_hand_residual: bool = False,
        regional_grid_size: int = REGIONAL_GRID_SIZE,
        additional_single_stream_blocks: int = 0,
    ) -> None:
        super().__init__()
        _require(token_channels == DENSE_TOKEN_CHANNELS, "DXYZ token channels must be 768")
        _require(
            hidden_channels > 0 and hidden_channels % attention_heads == 0,
            "DXYZ hidden width must divide heads",
        )
        _require(
            (double_stream_blocks, fused_single_stream_blocks) in {(4, 8), (2, 4)},
            "DXYZ requires the declared 4/8 or light 2/4 stream depth",
        )
        _require(
            additional_single_stream_blocks in {0, 2},
            "DXYZ additional joint blocks must be zero or two",
        )
        _require(maximum_frames >= 1, "DXYZ maximum frames must be positive")
        _require(
            regional_grid_size in {REGIONAL_GRID_SIZE, PATCH_GRID_SIZE // 2}
            and PATCH_GRID_SIZE % regional_grid_size == 0,
            "DXYZ regional grid must be 4x4 or 8x8",
        )
        _require(
            not adaptive_regional_pooling or regional_grid_size == REGIONAL_GRID_SIZE,
            "adaptive regional pooling requires the inherited 4x4 grid",
        )
        _require(
            temporal_adapter_placement in {"none", "prequery", "regional", "dense"},
            "DXYZ temporal adapter placement differs",
        )
        self.hidden_channels = hidden_channels
        self.maximum_frames = maximum_frames
        self.regional_grid_size = regional_grid_size
        self.regional_token_count = regional_grid_size * regional_grid_size
        self.additional_single_stream_blocks = additional_single_stream_blocks
        self.temporal_adapter_placement = temporal_adapter_placement
        self.part_dense_routing_enabled = part_dense_routing
        self.adaptive_regional_pooling_enabled = adaptive_regional_pooling
        self.nonlinear_hand_residual_enabled = nonlinear_hand_residual
        self.token_projection = nn.Linear(token_channels, hidden_channels)
        self.dense_row_embedding = nn.Parameter(torch.empty(PATCH_GRID_SIZE, hidden_channels))
        self.dense_column_embedding = nn.Parameter(torch.empty(PATCH_GRID_SIZE, hidden_channels))
        self.regional_row_embedding = nn.Parameter(torch.empty(regional_grid_size, hidden_channels))
        self.regional_column_embedding = nn.Parameter(
            torch.empty(regional_grid_size, hidden_channels)
        )
        self.temporal_embedding = nn.Parameter(torch.empty(maximum_frames, hidden_channels))
        self.radar_type_embedding = nn.Parameter(torch.empty(1, 1, 1, hidden_channels))
        self.query_type_embedding = nn.Parameter(torch.empty(QUERY_COUNT, hidden_channels))
        self.motion_queries = nn.Parameter(torch.empty(QUERY_COUNT, hidden_channels))
        for value in (
            self.dense_row_embedding,
            self.dense_column_embedding,
            self.regional_row_embedding,
            self.regional_column_embedding,
            self.temporal_embedding,
            self.radar_type_embedding,
            self.query_type_embedding,
            self.motion_queries,
        ):
            nn.init.normal_(value, mean=0.0, std=0.02)
        self.double_blocks = nn.ModuleList(
            _DoubleStreamBlock(
                hidden_channels=hidden_channels,
                attention_heads=attention_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(double_stream_blocks)
        )
        self.single_blocks = nn.ModuleList(
            _SingleStreamBlock(
                hidden_channels=hidden_channels,
                attention_heads=attention_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(fused_single_stream_blocks)
        )
        self.extra_single_blocks = nn.ModuleList(
            _SingleStreamBlock(
                hidden_channels=hidden_channels,
                attention_heads=attention_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(additional_single_stream_blocks)
        )
        self.temporal_adapter = (
            None
            if temporal_adapter_placement == "none"
            else DenseTokenTemporalSlotAdapter(
                hidden_channels=hidden_channels,
                bottleneck_channels=temporal_adapter_bottleneck_channels,
                attention_heads=temporal_adapter_attention_heads,
            )
        )
        self.part_dense_router = (
            DenseTokenPartDenseRouter(hidden_channels=hidden_channels)
            if part_dense_routing
            else None
        )
        self.adaptive_regional_pool = (
            DenseTokenAdaptiveRegionalPool(hidden_channels=hidden_channels)
            if adaptive_regional_pooling
            else None
        )
        self.output_normalization = nn.LayerNorm(hidden_channels)
        self.coarse_head = nn.Linear(hidden_channels, COARSE_COORDINATE_DIMENSION)
        self.left_hand_head = nn.Linear(hidden_channels, HAND_COORDINATE_DIMENSION)
        self.right_hand_head = nn.Linear(hidden_channels, HAND_COORDINATE_DIMENSION)
        self.hand_residual_head = (
            DenseTokenHandResidualHead(hidden_channels=hidden_channels)
            if nonlinear_hand_residual
            else None
        )

    def _validate_inputs(self, tokens: Tensor, valid_mask: Tensor) -> tuple[int, int]:
        _require(
            tokens.ndim == 4 and tokens.shape[2:] == (DENSE_TOKEN_COUNT, DENSE_TOKEN_CHANNELS),
            "DXYZ tokens must be [B,T,256,768]",
        )
        batch, frames = tokens.shape[:2]
        _require(frames <= self.maximum_frames, "DXYZ window exceeds temporal positions")
        _require(
            valid_mask.shape == (batch, frames) and valid_mask.dtype == torch.bool,
            "DXYZ valid mask must be bool [B,T]",
        )
        _require(bool(torch.isfinite(tokens).all()), "DXYZ tokens are non-finite")
        _require(bool(valid_mask.any(dim=1).all()), "every DXYZ sample needs one valid frame")
        return batch, frames

    def forward(self, tokens: Tensor, *, valid_mask: Tensor) -> DirectXyzPrediction:
        batch, frames = self._validate_inputs(tokens, valid_mask)
        dense_position = (
            self.dense_row_embedding[:, None, :] + self.dense_column_embedding[None, :, :]
        ).reshape(DENSE_TOKEN_COUNT, self.hidden_channels)
        time_position = self.temporal_embedding[:frames]
        radar = self.token_projection(tokens)
        radar = radar + dense_position[None, None] + time_position[None, :, None]
        radar = radar + self.radar_type_embedding
        queries = (
            self.motion_queries[None, None]
            + self.query_type_embedding[None, None]
            + time_position[None, :, None]
        ).expand(batch, -1, -1, -1)
        mask_value = valid_mask[:, :, None, None].to(dtype=radar.dtype)
        radar = radar * mask_value
        queries = queries * mask_value
        if self.temporal_adapter_placement == "prequery":
            if self.temporal_adapter is None:
                raise DenseTokenDirectXyzError("DXYZ prequery adapter is missing")
            radar = self.temporal_adapter(radar, valid_mask=valid_mask)
        for block in self.double_blocks:
            radar, queries = block(radar, queries, valid_mask=valid_mask)
        if self.part_dense_router is not None:
            queries = self.part_dense_router(queries, radar, valid_mask=valid_mask)
        if self.temporal_adapter_placement == "dense":
            if self.temporal_adapter is None:
                raise DenseTokenDirectXyzError("DXYZ dense adapter is missing")
            radar = self.temporal_adapter(radar, valid_mask=valid_mask)
        if self.adaptive_regional_pool is None:
            per_region = PATCH_GRID_SIZE // self.regional_grid_size
            regional = (
                radar.reshape(
                    batch,
                    frames,
                    self.regional_grid_size,
                    per_region,
                    self.regional_grid_size,
                    per_region,
                    self.hidden_channels,
                )
                .mean(dim=(3, 5))
                .reshape(batch, frames, self.regional_token_count, self.hidden_channels)
            )
        else:
            regional = self.adaptive_regional_pool(radar)
        regional_position = (
            self.regional_row_embedding[:, None, :] + self.regional_column_embedding[None, :, :]
        ).reshape(self.regional_token_count, self.hidden_channels)
        regional = regional + regional_position[None, None]
        if self.temporal_adapter_placement == "regional":
            if self.temporal_adapter is None:
                raise DenseTokenDirectXyzError("DXYZ regional adapter is missing")
            regional = self.temporal_adapter(regional, valid_mask=valid_mask)
        joined = torch.cat((regional, queries), dim=2).reshape(
            batch, frames * (self.regional_token_count + QUERY_COUNT), self.hidden_channels
        )
        joined_mask = valid_mask.repeat_interleave(self.regional_token_count + QUERY_COUNT, dim=1)
        joined_positions = (
            torch.arange(frames, device=tokens.device)
            .repeat_interleave(self.regional_token_count + QUERY_COUNT)
            .expand(batch, -1)
        )
        for block in self.single_blocks:
            joined = block(joined, valid_mask=joined_mask, positions=joined_positions)
        for block in self.extra_single_blocks:
            joined = block(joined, valid_mask=joined_mask, positions=joined_positions)
        normalized = self.output_normalization(
            joined.reshape(
                batch,
                frames,
                self.regional_token_count + QUERY_COUNT,
                self.hidden_channels,
            )
        )
        coarse = self.coarse_head(normalized[:, :, self.regional_token_count]).reshape(
            batch, frames, 2, COARSE_JOINT_COUNT, 3
        )
        left_hand = self.left_hand_head(normalized[:, :, self.regional_token_count + 1]).reshape(
            batch, frames, HAND_POINT_COUNT, 3
        )
        right_hand = self.right_hand_head(normalized[:, :, self.regional_token_count + 2]).reshape(
            batch, frames, HAND_POINT_COUNT, 3
        )
        if self.hand_residual_head is not None:
            left_hand = left_hand + self.hand_residual_head(
                normalized[:, :, self.regional_token_count + 1]
            ).reshape(batch, frames, HAND_POINT_COUNT, 3)
            right_hand = right_hand + self.hand_residual_head(
                normalized[:, :, self.regional_token_count + 2]
            ).reshape(batch, frames, HAND_POINT_COUNT, 3)
        hand = torch.stack((left_hand, right_hand), dim=2)
        wrist = coarse[:, :, :, 2:3, :]
        full24 = torch.cat((coarse, wrist, wrist + hand), dim=3)
        frame_weights = valid_mask[:, :, None, None, None].to(dtype=full24.dtype)
        coarse = coarse * frame_weights
        hand = hand * frame_weights
        full24 = full24 * frame_weights
        _require(
            coarse.shape == (batch, frames, 2, COARSE_JOINT_COUNT, 3),
            "DXYZ coarse output shape differs",
        )
        _require(
            hand.shape == (batch, frames, 2, HAND_POINT_COUNT, 3),
            "DXYZ hand output shape differs",
        )
        _require(
            full24.shape == (batch, frames, 2, FULL24_JOINT_COUNT, 3),
            "DXYZ full24 output shape differs",
        )
        return DirectXyzPrediction(
            coarse18_m=coarse,
            hand120_wrist_relative_m=hand,
            full24_m=full24,
        )
