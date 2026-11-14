"""Dense radar tokens, motion queries and structured body-parameter prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional

DENSE_TOKEN_COUNT: Final[int] = 256

DENSE_TOKEN_CHANNELS: Final[int] = 768

PATCH_GRID_SIZE: Final[int] = 16

REGIONAL_GRID_SIZE: Final[int] = 4

REGIONAL_TOKEN_COUNT: Final[int] = 16

QUERY_COUNT: Final[int] = 3

MHR_STATE_DIMENSION: Final[int] = 191

MHR_DYNAMIC_DIMENSION: Final[int] = 162

MHR_STATIC_DIMENSION: Final[int] = 29

MHR_UPPER_DIMENSION: Final[int] = 48

MHR_HAND_DIMENSION: Final[int] = 54

MHR_GLOBAL_ROTATION_DIMENSION: Final[int] = 3

MHR_CAMERA_TRANSLATION_DIMENSION: Final[int] = 3

MHR_MORPHOLOGY_DIMENSION: Final[int] = 28

MHR_STANDARD_SCALE_DIMENSION: Final[int] = 1


class DenseTokenDeterministicMhrError(ValueError):
    """Raised when a dense-token tensor violates its fixed model contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DenseTokenDeterministicMhrError(message)


def per_head_rms_norm(values: Tensor, *, epsilon: float = 1.0e-6) -> Tensor:
    """RMS-normalize Q or K separately in every attention head."""

    _require(values.ndim == 4, "Q/K values must be [B,H,L,D]")
    _require(epsilon > 0.0, "Q/K RMSNorm epsilon must be positive")
    _require(bool(torch.isfinite(values).all()), "Q/K RMSNorm input is non-finite")
    output = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + epsilon)
    _require(bool(torch.isfinite(output).all()), "Q/K RMSNorm output is non-finite")
    return output


def _masked_mean(values: Tensor, mask: Tensor, *, dimension: int) -> Tensor:
    """Average a rank-three tensor along one sequence dimension safely."""

    _require(values.ndim == 3, "masked mean values must be [B,L,H]")
    _require(
        mask.shape == values.shape[:2] and mask.dtype == torch.bool, "masked mean mask differs"
    )
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    denominator = weights.sum(dim=dimension, keepdim=False).clamp_min(1.0)
    return (values * weights).sum(dim=dimension) / denominator


def _modulate(values: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    _require(values.ndim == 3, "AdaLN values must be [B,L,H]")
    _require(
        shift.shape == scale.shape == (values.shape[0], values.shape[-1]),
        "AdaLN shift or scale differs",
    )
    return values * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _apply_temporal_rope(values: Tensor, positions: Tensor) -> Tensor:
    """Apply rotary temporal positions to Q or K values [B,H,L,D]."""

    _require(values.ndim == 4, "RoPE values must be [B,H,L,D]")
    _require(positions.shape == (values.shape[0], values.shape[2]), "RoPE positions differ")
    channels = values.shape[-1]
    _require(channels % 2 == 0, "RoPE head width must be even")
    half = channels // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=values.dtype, device=values.device)
        / max(half - 1, 1)
    )
    angles = positions.to(dtype=values.dtype)[..., None] * frequency[None, None]
    cosine = angles.cos().unsqueeze(1)
    sine = angles.sin().unsqueeze(1)
    first = values[..., 0::2]
    second = values[..., 1::2]
    return torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine), dim=-1
    ).flatten(start_dim=-2)


def _attention_from_qkv(
    query_key_value: Tensor,
    *,
    attention_heads: int,
    valid_mask: Tensor,
    positions: Tensor,
) -> Tensor:
    """Joint mask-safe attention over an already concatenated stream."""

    _require(query_key_value.ndim == 3, "attention QKV must be [B,L,3H]")
    batch, length, channels = query_key_value.shape
    _require(channels % (3 * attention_heads) == 0, "attention channels do not divide heads")
    head_channels = channels // (3 * attention_heads)
    _require(head_channels > 0, "attention head width must be positive")
    _require(
        valid_mask.shape == (batch, length) and valid_mask.dtype == torch.bool,
        "attention valid mask differs",
    )
    _require(positions.shape == (batch, length), "attention positions differ")
    packed = query_key_value.reshape(batch, length, 3, attention_heads, head_channels)
    query, key, value = packed.permute(2, 0, 3, 1, 4)
    query = per_head_rms_norm(_apply_temporal_rope(query, positions))
    key = per_head_rms_norm(_apply_temporal_rope(key, positions))
    scores = torch.matmul(query, key.transpose(-1, -2)) * (head_channels**-0.5)
    # A fully padded sequence receives finite placeholder logits and is then
    # zeroed at the query side.  This avoids NaNs without admitting padding.
    scores = scores.masked_fill(~valid_mask[:, None, None, :], -10_000.0)
    weights = torch.softmax(scores, dim=-1)
    weights = weights * valid_mask[:, None, :, None].to(dtype=weights.dtype)
    output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, -1)
    return output * valid_mask.unsqueeze(-1).to(dtype=output.dtype)


class _FeedForward(nn.Module):
    def __init__(self, hidden_channels: int, mlp_ratio: int) -> None:
        super().__init__()
        _require(mlp_ratio > 0, "MLP ratio must be positive")
        self.layers = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_channels * mlp_ratio, hidden_channels),
        )

    def forward(self, values: Tensor) -> Tensor:
        output: Tensor = self.layers(values)
        return output


@dataclass(frozen=True)
class _DoubleModulation:
    radar_attention_shift: Tensor
    radar_attention_scale: Tensor
    radar_attention_gate: Tensor
    radar_mlp_shift: Tensor
    radar_mlp_scale: Tensor
    radar_mlp_gate: Tensor
    query_attention_shift: Tensor
    query_attention_scale: Tensor
    query_attention_gate: Tensor
    query_mlp_shift: Tensor
    query_mlp_scale: Tensor
    query_mlp_gate: Tensor


class _DoubleAdaLNZero(nn.Module):
    """Per-stream AdaLN-Zero controls for one double-stream block."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_channels, hidden_channels * 12))
        final = self.projection[-1]
        if not isinstance(final, nn.Linear):
            raise DenseTokenDeterministicMhrError("double AdaLN must end in Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, condition: Tensor) -> _DoubleModulation:
        _require(condition.ndim == 2, "double AdaLN condition must be [B,H]")
        return _DoubleModulation(*self.projection(condition).chunk(12, dim=-1))


@dataclass(frozen=True)
class _SingleModulation:
    attention_shift: Tensor
    attention_scale: Tensor
    attention_gate: Tensor
    mlp_shift: Tensor
    mlp_scale: Tensor
    mlp_gate: Tensor


class _SingleAdaLNZero(nn.Module):
    """AdaLN-Zero controls for one temporally fused joint-stream block."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_channels, hidden_channels * 6))
        final = self.projection[-1]
        if not isinstance(final, nn.Linear):
            raise DenseTokenDeterministicMhrError("single AdaLN must end in Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, condition: Tensor) -> _SingleModulation:
        _require(condition.ndim == 2, "single AdaLN condition must be [B,H]")
        return _SingleModulation(*self.projection(condition).chunk(6, dim=-1))


class _DoubleStreamBlock(nn.Module):
    """Per-frame joint attention that updates radar and query streams."""

    def __init__(self, *, hidden_channels: int, attention_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        _require(
            hidden_channels > 0 and hidden_channels % attention_heads == 0,
            "double-stream width must divide attention heads",
        )
        self.hidden_channels = hidden_channels
        self.attention_heads = attention_heads
        self.modulation = _DoubleAdaLNZero(hidden_channels)
        self.radar_attention_norm = nn.LayerNorm(hidden_channels)
        self.query_attention_norm = nn.LayerNorm(hidden_channels)
        self.radar_qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.query_qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.radar_attention_output = nn.Linear(hidden_channels, hidden_channels)
        self.query_attention_output = nn.Linear(hidden_channels, hidden_channels)
        self.radar_mlp_norm = nn.LayerNorm(hidden_channels)
        self.query_mlp_norm = nn.LayerNorm(hidden_channels)
        self.radar_mlp = _FeedForward(hidden_channels, mlp_ratio)
        self.query_mlp = _FeedForward(hidden_channels, mlp_ratio)

    def forward(
        self,
        radar: Tensor,
        queries: Tensor,
        *,
        valid_mask: Tensor,
        timestep_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        _require(
            radar.ndim == queries.ndim == 4 and radar.shape[:2] == queries.shape[:2],
            "double-stream batch or time dimensions differ",
        )
        _require(
            radar.shape[2:] == (DENSE_TOKEN_COUNT, self.hidden_channels),
            "double-stream radar tokens differ",
        )
        _require(
            queries.shape[2:] == (QUERY_COUNT, self.hidden_channels),
            "double-stream query tokens differ",
        )
        batch, frames = radar.shape[:2]
        _require(
            valid_mask.shape == (batch, frames) and valid_mask.dtype == torch.bool,
            "double-stream mask differs",
        )
        radar_flat = radar.reshape(batch * frames, DENSE_TOKEN_COUNT, self.hidden_channels)
        query_flat = queries.reshape(batch * frames, QUERY_COUNT, self.hidden_channels)
        frame_valid = valid_mask.reshape(batch * frames)
        # The per-frame radar summary is the deterministic AdaLN conditioner.
        # Cached Flow Matching consumers may add a per-window flow-time
        # embedding here.  Leaving it absent preserves the DXYZ/DMHR path
        # bit-for-bit while allowing the same compatible block tensors to be
        # warm-started into a time-conditioned decoder.
        condition = radar_flat.mean(dim=1) * frame_valid[:, None].to(dtype=radar.dtype)
        if timestep_embedding is not None:
            _require(
                timestep_embedding.shape == (batch, self.hidden_channels),
                "double-stream timestep embedding differs",
            )
            condition = condition + timestep_embedding[:, None, :].expand(-1, frames, -1).reshape(
                batch * frames, self.hidden_channels
            )
        modulation = self.modulation(condition)
        radar_qkv = self.radar_qkv(
            _modulate(
                self.radar_attention_norm(radar_flat),
                modulation.radar_attention_shift,
                modulation.radar_attention_scale,
            )
        )
        query_qkv = self.query_qkv(
            _modulate(
                self.query_attention_norm(query_flat),
                modulation.query_attention_shift,
                modulation.query_attention_scale,
            )
        )
        joint_qkv = torch.cat((radar_qkv, query_qkv), dim=1)
        joint_mask = frame_valid[:, None].expand(-1, DENSE_TOKEN_COUNT + QUERY_COUNT)
        frame_positions = torch.arange(frames, device=radar.device).expand(batch, -1).reshape(-1)
        joint_positions = frame_positions[:, None].expand(-1, DENSE_TOKEN_COUNT + QUERY_COUNT)
        joint_update = _attention_from_qkv(
            joint_qkv,
            attention_heads=self.attention_heads,
            valid_mask=joint_mask,
            positions=joint_positions,
        )
        radar_update, query_update = torch.split(
            joint_update, [DENSE_TOKEN_COUNT, QUERY_COUNT], dim=1
        )
        radar_flat = radar_flat + modulation.radar_attention_gate[
            :, None
        ] * self.radar_attention_output(radar_update)
        query_flat = query_flat + modulation.query_attention_gate[
            :, None
        ] * self.query_attention_output(query_update)
        radar_flat = radar_flat + modulation.radar_mlp_gate[:, None] * self.radar_mlp(
            _modulate(
                self.radar_mlp_norm(radar_flat),
                modulation.radar_mlp_shift,
                modulation.radar_mlp_scale,
            )
        )
        query_flat = query_flat + modulation.query_mlp_gate[:, None] * self.query_mlp(
            _modulate(
                self.query_mlp_norm(query_flat),
                modulation.query_mlp_shift,
                modulation.query_mlp_scale,
            )
        )
        mask_value = frame_valid[:, None, None].to(dtype=radar.dtype)
        return (
            (radar_flat * mask_value).reshape(
                batch, frames, DENSE_TOKEN_COUNT, self.hidden_channels
            ),
            (query_flat * mask_value).reshape(batch, frames, QUERY_COUNT, self.hidden_channels),
        )


class _SingleStreamBlock(nn.Module):
    """Cross-time joint attention over regional radar and motion queries."""

    def __init__(self, *, hidden_channels: int, attention_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        _require(
            hidden_channels > 0 and hidden_channels % attention_heads == 0,
            "single-stream width must divide attention heads",
        )
        self.hidden_channels = hidden_channels
        self.attention_heads = attention_heads
        self.modulation = _SingleAdaLNZero(hidden_channels)
        self.attention_norm = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.attention_output = nn.Linear(hidden_channels, hidden_channels)
        self.mlp_norm = nn.LayerNorm(hidden_channels)
        self.mlp = _FeedForward(hidden_channels, mlp_ratio)

    def forward(
        self,
        values: Tensor,
        *,
        valid_mask: Tensor,
        positions: Tensor,
        timestep_embedding: Tensor | None = None,
    ) -> Tensor:
        _require(
            values.ndim == 3 and values.shape[-1] == self.hidden_channels,
            "single-stream values differ",
        )
        _require(
            valid_mask.shape == values.shape[:2] and valid_mask.dtype == torch.bool,
            "single-stream mask differs",
        )
        _require(positions.shape == values.shape[:2], "single-stream positions differ")
        condition = _masked_mean(values, valid_mask, dimension=1)
        if timestep_embedding is not None:
            _require(
                timestep_embedding.shape == (values.shape[0], self.hidden_channels),
                "single-stream timestep embedding differs",
            )
            condition = condition + timestep_embedding
        modulation = self.modulation(condition)
        attention_update = _attention_from_qkv(
            self.qkv(
                _modulate(
                    self.attention_norm(values),
                    modulation.attention_shift,
                    modulation.attention_scale,
                )
            ),
            attention_heads=self.attention_heads,
            valid_mask=valid_mask,
            positions=positions,
        )
        values = values + modulation.attention_gate[:, None] * self.attention_output(
            attention_update
        )
        values = values + modulation.mlp_gate[:, None] * self.mlp(
            _modulate(self.mlp_norm(values), modulation.mlp_shift, modulation.mlp_scale)
        )
        return values * valid_mask.unsqueeze(-1).to(dtype=values.dtype)


@dataclass(frozen=True)
class MhrStatePrediction:
    """Decoded semantic state before frozen MHR/FK evaluation."""

    state: Tensor
    dynamic_state: Tensor
    static_state: Tensor


class DenseTokenDeterministicMhrDecoder(nn.Module):
    """The dense-token decoder: four frame-wise double-stream blocks and eight joint-stream blocks."""

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
    ) -> None:
        super().__init__()
        _require(token_channels == DENSE_TOKEN_CHANNELS, "DMHR token channels must be 768")
        _require(
            hidden_channels > 0 and hidden_channels % attention_heads == 0,
            "DMHR hidden width must divide heads",
        )
        _require(
            (double_stream_blocks, fused_single_stream_blocks) in {(4, 8), (2, 4)},
            "DMHR requires the declared 4/8 or light 2/4 stream depth",
        )
        _require(maximum_frames >= 1, "DMHR maximum frames must be positive")
        self.hidden_channels = hidden_channels
        self.maximum_frames = maximum_frames
        self.token_projection = nn.Linear(token_channels, hidden_channels)
        self.dense_row_embedding = nn.Parameter(torch.empty(PATCH_GRID_SIZE, hidden_channels))
        self.dense_column_embedding = nn.Parameter(torch.empty(PATCH_GRID_SIZE, hidden_channels))
        self.regional_row_embedding = nn.Parameter(torch.empty(REGIONAL_GRID_SIZE, hidden_channels))
        self.regional_column_embedding = nn.Parameter(
            torch.empty(REGIONAL_GRID_SIZE, hidden_channels)
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
        self.output_normalization = nn.LayerNorm(hidden_channels)
        self.upper_global_head = nn.Linear(
            hidden_channels,
            MHR_UPPER_DIMENSION + MHR_GLOBAL_ROTATION_DIMENSION + MHR_CAMERA_TRANSLATION_DIMENSION,
        )
        self.left_hand_head = nn.Linear(hidden_channels, MHR_HAND_DIMENSION)
        self.right_hand_head = nn.Linear(hidden_channels, MHR_HAND_DIMENSION)
        self.static_head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, MHR_STATIC_DIMENSION),
        )

    def _validate_inputs(self, tokens: Tensor, valid_mask: Tensor) -> tuple[int, int]:
        _require(
            tokens.ndim == 4 and tokens.shape[2:] == (DENSE_TOKEN_COUNT, DENSE_TOKEN_CHANNELS),
            "DMHR tokens must be [B,T,256,768]",
        )
        batch, frames = tokens.shape[:2]
        _require(frames <= self.maximum_frames, "DMHR window exceeds configured temporal positions")
        _require(
            valid_mask.shape == (batch, frames) and valid_mask.dtype == torch.bool,
            "DMHR valid mask must be bool [B,T]",
        )
        _require(bool(torch.isfinite(tokens).all()), "DMHR tokens are non-finite")
        _require(bool(valid_mask.any(dim=1).all()), "every DMHR sample needs one valid frame")
        return batch, frames

    def forward(self, tokens: Tensor, *, valid_mask: Tensor) -> MhrStatePrediction:
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
        for block in self.double_blocks:
            radar, queries = block(radar, queries, valid_mask=valid_mask)
        regional = (
            radar.reshape(
                batch,
                frames,
                REGIONAL_GRID_SIZE,
                PATCH_GRID_SIZE // REGIONAL_GRID_SIZE,
                REGIONAL_GRID_SIZE,
                PATCH_GRID_SIZE // REGIONAL_GRID_SIZE,
                self.hidden_channels,
            )
            .mean(dim=(3, 5))
            .reshape(batch, frames, REGIONAL_TOKEN_COUNT, self.hidden_channels)
        )
        regional_position = (
            self.regional_row_embedding[:, None, :] + self.regional_column_embedding[None, :, :]
        ).reshape(REGIONAL_TOKEN_COUNT, self.hidden_channels)
        regional = regional + regional_position[None, None]
        joined = torch.cat((regional, queries), dim=2).reshape(
            batch, frames * (REGIONAL_TOKEN_COUNT + QUERY_COUNT), self.hidden_channels
        )
        joined_mask = valid_mask.repeat_interleave(REGIONAL_TOKEN_COUNT + QUERY_COUNT, dim=1)
        joined_positions = (
            torch.arange(frames, device=tokens.device)
            .repeat_interleave(REGIONAL_TOKEN_COUNT + QUERY_COUNT)
            .expand(batch, -1)
        )
        for block in self.single_blocks:
            joined = block(joined, valid_mask=joined_mask, positions=joined_positions)
        joined = joined.reshape(
            batch, frames, REGIONAL_TOKEN_COUNT + QUERY_COUNT, self.hidden_channels
        )
        normalized = self.output_normalization(joined)
        upper_global = self.upper_global_head(normalized[:, :, REGIONAL_TOKEN_COUNT])
        left_hand = self.left_hand_head(normalized[:, :, REGIONAL_TOKEN_COUNT + 1])
        right_hand = self.right_hand_head(normalized[:, :, REGIONAL_TOKEN_COUNT + 2])
        dynamic = torch.cat(
            (
                upper_global[..., :MHR_UPPER_DIMENSION],
                left_hand,
                right_hand,
                upper_global[
                    ...,
                    MHR_UPPER_DIMENSION : MHR_UPPER_DIMENSION + MHR_GLOBAL_ROTATION_DIMENSION,
                ],
                upper_global[..., -MHR_CAMERA_TRANSLATION_DIMENSION:],
            ),
            dim=-1,
        )
        _require(
            dynamic.shape == (batch, frames, MHR_DYNAMIC_DIMENSION), "DMHR dynamic shape differs"
        )
        regional_final = normalized[:, :, :REGIONAL_TOKEN_COUNT].reshape(
            batch, frames * REGIONAL_TOKEN_COUNT, self.hidden_channels
        )
        regional_mask = valid_mask.repeat_interleave(REGIONAL_TOKEN_COUNT, dim=1)
        static = self.static_head(_masked_mean(regional_final, regional_mask, dimension=1))
        state = torch.cat((dynamic, static[:, None, :].expand(-1, frames, -1)), dim=-1)
        state = state * valid_mask.unsqueeze(-1).to(dtype=state.dtype)
        _require(state.shape == (batch, frames, MHR_STATE_DIMENSION), "DMHR state shape differs")
        return MhrStatePrediction(state=state, dynamic_state=dynamic, static_state=static)


@dataclass(frozen=True)
class DenseTokenEncoderOutput:
    """Online M2 patch grid and global sidecar before the DMHR decoder."""

    tokens: Tensor
    global_features: Tensor


@dataclass(frozen=True)
class MhrTargetNormalization:
    """Train-only per-dimension normalization for the direct 191-D target."""

    mean: Tensor
    scale: Tensor
    active_dimensions: Tensor

    def __post_init__(self) -> None:
        _require(self.mean.shape == (MHR_STATE_DIMENSION,), "MHR normalization mean differs")
        _require(self.scale.shape == (MHR_STATE_DIMENSION,), "MHR normalization scale differs")
        _require(
            self.active_dimensions.shape == (MHR_STATE_DIMENSION,)
            and self.active_dimensions.dtype == torch.bool,
            "MHR normalization active mask differs",
        )
        _require(bool(torch.isfinite(self.mean).all()), "MHR normalization mean is non-finite")
        _require(
            bool(torch.isfinite(self.scale).all()) and bool((self.scale > 0).all()),
            "MHR normalization scale is invalid",
        )

    def normalize(self, values: Tensor) -> Tensor:
        _require(values.shape[-1] == MHR_STATE_DIMENSION, "MHR target width differs")
        return (values - self.mean.to(device=values.device, dtype=values.dtype)) / self.scale.to(
            device=values.device, dtype=values.dtype
        )

    def denormalize(self, values: Tensor) -> Tensor:
        _require(values.shape[-1] == MHR_STATE_DIMENSION, "MHR target width differs")
        return values * self.scale.to(device=values.device, dtype=values.dtype) + self.mean.to(
            device=values.device, dtype=values.dtype
        )


MHR_SEMANTIC_GROUPS: Final[tuple[tuple[str, slice], ...]] = (
    ("upper_articulation", slice(0, 48)),
    ("left_hand_articulation", slice(48, 102)),
    ("right_hand_articulation", slice(102, 156)),
    ("global_rotation_and_translation", slice(156, 162)),
    ("static_morphology", slice(162, 191)),
)


@dataclass(frozen=True)
class MhrSemanticLoss:
    """Total equal-group loss plus individual group values for logging."""

    total: Tensor
    groups: dict[str, Tensor]


def masked_semantic_mhr_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor,
    normalization: MhrTargetNormalization,
    excluded_semantic_groups: tuple[str, ...] = (),
) -> MhrSemanticLoss:
    """Equal-weight semantic-group SmoothL1 in standardized parameter space.

    Consumers can explicitly omit a semantically inapplicable group, such as
    scattering-only static body fields in a fixed-standard-body decode.  The
    default retains the complete-MHR contract used by existing routes.
    """

    _require(
        prediction.shape == target.shape
        and prediction.ndim == 3
        and prediction.shape[-1] == MHR_STATE_DIMENSION,
        "MHR prediction/target shapes differ",
    )
    _require(
        valid_mask.shape == prediction.shape[:2] and valid_mask.dtype == torch.bool,
        "MHR loss mask differs",
    )
    _require(
        bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all()),
        "MHR loss values are non-finite",
    )
    standardized_prediction = normalization.normalize(prediction)
    standardized_target = normalization.normalize(target)
    pointwise = functional.smooth_l1_loss(
        standardized_prediction, standardized_target, reduction="none"
    )
    mask = valid_mask.unsqueeze(-1).to(dtype=pointwise.dtype)
    active = normalization.active_dimensions.to(device=pointwise.device)
    known_groups = {name for name, _interval in MHR_SEMANTIC_GROUPS}
    excluded = set(excluded_semantic_groups)
    _require(
        len(excluded) == len(excluded_semantic_groups) and excluded <= known_groups,
        "MHR excluded semantic groups differ",
    )
    groups: dict[str, Tensor] = {}
    for name, interval in MHR_SEMANTIC_GROUPS:
        if name in excluded:
            continue
        group_active = active[interval]
        if not bool(group_active.any()):
            continue
        weights = mask * group_active.to(dtype=pointwise.dtype).view(1, 1, -1)
        groups[name] = (pointwise[..., interval] * weights).sum() / weights.sum().clamp_min(1.0)
    _require(bool(groups), "MHR loss has no active semantic groups")
    total = torch.stack(tuple(groups.values())).mean()
    return MhrSemanticLoss(total=total, groups=groups)
