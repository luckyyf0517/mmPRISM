"""Joint direct-coordinate and body-parameter reconstruction from dense radar tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, cast

import torch
from torch import Tensor, nn

from mmprism.models.dense_token_deterministic_mhr import (
    DENSE_TOKEN_CHANNELS,
    DENSE_TOKEN_COUNT,
    MHR_CAMERA_TRANSLATION_DIMENSION,
    MHR_DYNAMIC_DIMENSION,
    MHR_GLOBAL_ROTATION_DIMENSION,
    MHR_STANDARD_SCALE_DIMENSION,
    MHR_STATE_DIMENSION,
    MHR_UPPER_DIMENSION,
    PATCH_GRID_SIZE,
    QUERY_COUNT,
    REGIONAL_GRID_SIZE,
    REGIONAL_TOKEN_COUNT,
    DenseTokenDeterministicMhrDecoder,
    DenseTokenEncoderOutput,
    MhrStatePrediction,
)
from mmprism.models.dense_token_direct_xyz import (
    COARSE_COORDINATE_DIMENSION,
    COARSE_JOINT_COUNT,
    FULL24_JOINT_COUNT,
    HAND_COORDINATE_DIMENSION,
    HAND_POINT_COUNT,
    DirectXyzPrediction,
)


class DenseTokenDualSupervisionError(ValueError):
    """Raised when E2E-DUAL50 violates its shared-reader contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DenseTokenDualSupervisionError(message)


def _masked_mean(values: Tensor, mask: Tensor, *, dimension: int) -> Tensor:
    _require(values.ndim == 3, "masked mean values must be [B,L,H]")
    _require(mask.shape == values.shape[:2] and mask.dtype == torch.bool, "masked mean mask")
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=dimension) / weights.sum(dim=dimension).clamp_min(1.0)


_MHR_HEAD_PREFIXES: Final[tuple[str, ...]] = (
    "upper_global_head.",
    "left_hand_head.",
    "right_hand_head.",
    "static_head.",
)

DENSE_QUERY_GATE_INITIAL_BIAS: Final[float] = 1.0e-3

_DIRECT_TO_DUAL_HEAD_NAMES: Final[dict[str, str]] = {
    "coarse_head.weight": "direct_coarse_head.weight",
    "coarse_head.bias": "direct_coarse_head.bias",
    "left_hand_head.weight": "direct_left_hand_head.weight",
    "left_hand_head.bias": "direct_left_hand_head.bias",
    "right_hand_head.weight": "direct_right_hand_head.weight",
    "right_hand_head.bias": "direct_right_hand_head.bias",
}


@dataclass(frozen=True)
class DualSupervisionDecoderPrediction:
    """Both readouts and the one normalized shared representation."""

    direct: DirectXyzPrediction
    mhr_state: MhrStatePrediction
    normalized_reader_tokens: Tensor


@dataclass(frozen=True)
class OnlineDualSupervisionPrediction:
    """Online raw-radar result with post-DINO tokens for gradient diagnostics."""

    direct: DirectXyzPrediction
    mhr_state: MhrStatePrediction
    dense_tokens: Tensor
    normalized_reader_tokens: Tensor


class DenseTokenDualSupervisionDecoder(DenseTokenDeterministicMhrDecoder):
    """One 4+pool16+8 reader with simultaneous Direct-XYZ and MHR heads."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.direct_coarse_head = nn.Linear(self.hidden_channels, COARSE_COORDINATE_DIMENSION)
        self.direct_left_hand_head = nn.Linear(self.hidden_channels, HAND_COORDINATE_DIMENSION)
        self.direct_right_hand_head = nn.Linear(self.hidden_channels, HAND_COORDINATE_DIMENSION)
        projection = self.static_head[-1]
        if not isinstance(projection, nn.Linear):
            raise DenseTokenDualSupervisionError("dual static head must end in Linear")
        _require(
            projection.out_features >= MHR_STANDARD_SCALE_DIMENSION,
            "dual static head lacks lambda",
        )
        with torch.no_grad():
            projection.weight[-1].zero_()
            projection.bias[-1].fill_(1.0)

    def _validate_boundary(self, boundary: Tensor, valid_mask: Tensor) -> tuple[int, int]:
        """Validate one cached or freshly encoded ``[B,T,19,H]`` boundary."""

        _require(
            boundary.ndim == 4
            and boundary.shape[2:] == (REGIONAL_TOKEN_COUNT + QUERY_COUNT, self.hidden_channels),
            "dual boundary tokens must be [B,T,19,H]",
        )
        batch, frames = boundary.shape[:2]
        _require(frames <= self.maximum_frames, "dual boundary exceeds temporal positions")
        _require(
            valid_mask.shape == (batch, frames) and valid_mask.dtype == torch.bool,
            "dual boundary valid mask must be bool [B,T]",
        )
        _require(bool(torch.isfinite(boundary).all()), "dual boundary tokens are non-finite")
        _require(
            bool(valid_mask.any(dim=1).all()), "every dual boundary sample needs one valid frame"
        )
        return batch, frames

    def _spatial_boundary(
        self, tokens: Tensor, *, valid_mask: Tensor, apply_window_time: bool
    ) -> Tensor:
        """Return the frame-local ``[B,T,19,H]`` boundary before any position term.

        The boundary concatenates the pooled regional mean with the three query
        rows, in the declared ``regional16, upper, left, right`` order.  Dense
        spatial embeddings and the radar type term belong to the spatial prefix;
        the window-time position is added only by the legacy window-time path,
        so the position-independent entry point leaves the boundary free of any
        window-local information and of the post-pool regional position.
        """

        batch, frames = self._validate_inputs(tokens, valid_mask)
        dense_position = (
            self.dense_row_embedding[:, None, :] + self.dense_column_embedding[None, :, :]
        ).reshape(DENSE_TOKEN_COUNT, self.hidden_channels)
        radar = self.token_projection(tokens)
        radar = radar + dense_position[None, None]
        queries = self.motion_queries[None, None] + self.query_type_embedding[None, None]
        if apply_window_time:
            time_position = self.temporal_embedding[:frames]
            radar = radar + time_position[None, :, None]
            queries = queries + time_position[None, :, None]
        radar = radar + self.radar_type_embedding
        queries = queries.expand(batch, -1, -1, -1)
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
        return torch.cat((regional, queries), dim=2)

    def _suffix_normalized(
        self,
        boundary: Tensor,
        *,
        valid_mask: Tensor,
        block_count: int,
        temporal_offset: Tensor | None,
    ) -> Tensor:
        """Apply the post-pool positions, the joint blocks and the output norm once.

        ``block_count`` selects how many of the eight joint blocks run.  The
        single-frame readout runs exactly one, which is the same module the
        temporal stage reuses as its first block.  ``temporal_offset`` carries
        the learned window-time position and is ``None`` whenever the caller
        must stay window-position independent.
        """

        batch, frames = self._validate_boundary(boundary, valid_mask)
        _require(1 <= block_count <= len(self.single_blocks), "dual suffix block count differs")
        regional_position = (
            self.regional_row_embedding[:, None, :] + self.regional_column_embedding[None, :, :]
        ).reshape(REGIONAL_TOKEN_COUNT, self.hidden_channels)
        regional = boundary[:, :, :REGIONAL_TOKEN_COUNT] + regional_position[None, None]
        queries = boundary[:, :, REGIONAL_TOKEN_COUNT:]
        if temporal_offset is not None:
            _require(
                temporal_offset.shape == (frames, self.hidden_channels),
                "dual temporal offset differs",
            )
            regional = regional + temporal_offset[None, :, None]
            queries = queries + temporal_offset[None, :, None]
        joined = torch.cat((regional, queries), dim=2).reshape(
            batch, frames * (REGIONAL_TOKEN_COUNT + QUERY_COUNT), self.hidden_channels
        )
        joined_mask = valid_mask.repeat_interleave(REGIONAL_TOKEN_COUNT + QUERY_COUNT, dim=1)
        joined_positions = (
            torch.arange(frames, device=boundary.device)
            .repeat_interleave(REGIONAL_TOKEN_COUNT + QUERY_COUNT)
            .expand(batch, -1)
        )
        for block in self.single_blocks[:block_count]:
            joined = block(joined, valid_mask=joined_mask, positions=joined_positions)
        normalized: Tensor = self.output_normalization(
            joined.reshape(
                batch,
                frames,
                REGIONAL_TOKEN_COUNT + QUERY_COUNT,
                self.hidden_channels,
            )
        )
        return normalized

    def _predict(
        self, normalized: Tensor, *, valid_mask: Tensor
    ) -> DualSupervisionDecoderPrediction:
        """Branch both families from exactly one normalized shared reader output."""

        batch, frames = normalized.shape[:2]
        upper_query = normalized[:, :, REGIONAL_TOKEN_COUNT]
        left_query = normalized[:, :, REGIONAL_TOKEN_COUNT + 1]
        right_query = normalized[:, :, REGIONAL_TOKEN_COUNT + 2]

        direct_coarse = self.direct_coarse_head(upper_query).reshape(
            batch, frames, 2, COARSE_JOINT_COUNT, 3
        )
        direct_left = self.direct_left_hand_head(left_query).reshape(
            batch, frames, HAND_POINT_COUNT, 3
        )
        direct_right = self.direct_right_hand_head(right_query).reshape(
            batch, frames, HAND_POINT_COUNT, 3
        )
        direct_hand = torch.stack((direct_left, direct_right), dim=2)
        direct_wrist = direct_coarse[:, :, :, 2:3, :]
        direct_full24 = torch.cat((direct_coarse, direct_wrist, direct_wrist + direct_hand), dim=3)

        upper_global = self.upper_global_head(upper_query)
        mhr_left = self.left_hand_head(left_query)
        mhr_right = self.right_hand_head(right_query)
        dynamic = torch.cat(
            (
                upper_global[..., :MHR_UPPER_DIMENSION],
                mhr_left,
                mhr_right,
                upper_global[
                    ...,
                    MHR_UPPER_DIMENSION : MHR_UPPER_DIMENSION + MHR_GLOBAL_ROTATION_DIMENSION,
                ],
                upper_global[..., -MHR_CAMERA_TRANSLATION_DIMENSION:],
            ),
            dim=-1,
        )
        _require(
            dynamic.shape == (batch, frames, MHR_DYNAMIC_DIMENSION),
            "dual dynamic MHR state shape differs",
        )
        regional_final = normalized[:, :, :REGIONAL_TOKEN_COUNT].reshape(
            batch, frames * REGIONAL_TOKEN_COUNT, self.hidden_channels
        )
        regional_mask = valid_mask.repeat_interleave(REGIONAL_TOKEN_COUNT, dim=1)
        static = self.static_head(_masked_mean(regional_final, regional_mask, dimension=1))
        state = torch.cat((dynamic, static[:, None, :].expand(-1, frames, -1)), dim=-1)

        frame_weights = valid_mask[:, :, None, None, None].to(dtype=direct_full24.dtype)
        direct_coarse = direct_coarse * frame_weights
        direct_hand = direct_hand * frame_weights
        direct_full24 = direct_full24 * frame_weights
        state = state * valid_mask.unsqueeze(-1).to(dtype=state.dtype)
        _require(
            direct_full24.shape == (batch, frames, 2, FULL24_JOINT_COUNT, 3),
            "dual Direct full24 shape differs",
        )
        _require(
            state.shape == (batch, frames, MHR_STATE_DIMENSION), "dual MHR state shape differs"
        )
        return DualSupervisionDecoderPrediction(
            direct=DirectXyzPrediction(
                coarse18_m=direct_coarse,
                hand120_wrist_relative_m=direct_hand,
                full24_m=direct_full24,
            ),
            mhr_state=MhrStatePrediction(state=state, dynamic_state=dynamic, static_state=static),
            normalized_reader_tokens=normalized,
        )

    def encode_spatial(self, tokens: Tensor, *, valid_mask: Tensor) -> Tensor:
        """Return the position-independent per-frame cache boundary ``[B,T,19,H]``.

        This is the spatial prefix only: Doppler-based route tokens, the
        projection, the dense spatial and type embeddings, the four per-frame
        double-stream blocks and the deterministic regional pool.  It consumes
        no window-local position, no post-pool regional position and no
        temporal position, and it never runs a joint block.
        """

        return self._spatial_boundary(tokens, valid_mask=valid_mask, apply_window_time=False)

    def readout(
        self,
        boundary: Tensor,
        *,
        valid_mask: Tensor,
        capture_activations: bool = False,
    ) -> DualSupervisionDecoderPrediction:
        """Decode the boundary through the single lightweight frame readout.

        The readout is the first joint block, so this is exactly the input and
        module the temporal stage will reuse as its block zero.  The output
        normalization is applied once here and once at the end of the temporal
        stack; it is never applied twice in one forward.
        """

        normalized = self._suffix_normalized(
            boundary, valid_mask=valid_mask, block_count=1, temporal_offset=None
        )
        if capture_activations and normalized.requires_grad:
            normalized.retain_grad()
        return self._predict(normalized, valid_mask=valid_mask)

    def decode_temporal(
        self,
        boundary: Tensor,
        *,
        valid_mask: Tensor,
        capture_activations: bool = False,
    ) -> DualSupervisionDecoderPrediction:
        """Decode a cached boundary sequence through all eight joint blocks.

        The first block is the same readout module trained by the single-frame
        stage; the remaining seven are initialized independently when the
        temporal stage starts.  Only this entry point adds the learned
        window-time position.
        """

        _batch, frames = self._validate_boundary(boundary, valid_mask)
        normalized = self._suffix_normalized(
            boundary,
            valid_mask=valid_mask,
            block_count=len(self.single_blocks),
            temporal_offset=self.temporal_embedding[:frames],
        )
        if capture_activations and normalized.requires_grad:
            normalized.retain_grad()
        return self._predict(normalized, valid_mask=valid_mask)

    def forward(  # type: ignore[override]
        self,
        tokens: Tensor,
        *,
        valid_mask: Tensor,
        capture_activations: bool = False,
    ) -> DualSupervisionDecoderPrediction:
        """Decode both families from exactly one normalized shared reader output.

        This is the legacy window-time path and is intentionally left bitwise
        unchanged: the learned window-time position still enters the four
        double-stream blocks through both streams.
        """

        boundary = self._spatial_boundary(tokens, valid_mask=valid_mask, apply_window_time=True)
        normalized = self._suffix_normalized(
            boundary,
            valid_mask=valid_mask,
            block_count=len(self.single_blocks),
            temporal_offset=None,
        )
        if capture_activations and normalized.requires_grad:
            normalized.retain_grad()
        return self._predict(normalized, valid_mask=valid_mask)


class _DenseRoute(Protocol):
    """The two accepted M2 route calls used by the dual-supervision model."""

    def complex_rd_maps(self, raw_iq: Tensor) -> Tensor: ...

    def encode_rd_maps(self, complex_maps: Tensor) -> tuple[Tensor, Tensor]: ...


class DenseTokenDualSupervisionModel(nn.Module):
    """Online radar route plus one shared-reader dual-supervision decoder."""

    def __init__(self, *, route: nn.Module, decoder: DenseTokenDualSupervisionDecoder) -> None:
        super().__init__()
        _require(hasattr(route, "complex_rd_maps"), "dual route lacks complex RD maps")
        _require(hasattr(route, "encode_rd_maps"), "dual route lacks dense encoding")
        self.route = route
        self.decoder = decoder
        pose_head = getattr(route, "pose_head", None)
        if not isinstance(pose_head, nn.Module):
            raise DenseTokenDualSupervisionError("dual route lacks unused pose head")
        for parameter in pose_head.parameters():
            parameter.requires_grad_(False)

    def encode_dense_tokens(
        self, raw_radar: Tensor, *, valid_mask: Tensor
    ) -> DenseTokenEncoderOutput:
        _require(raw_radar.ndim >= 3, "dual raw radar must begin [B,T,...]")
        batch, frames = raw_radar.shape[:2]
        _require(
            valid_mask.shape == (batch, frames) and valid_mask.dtype == torch.bool,
            "dual raw-radar valid mask differs",
        )
        flattened = raw_radar.reshape(batch * frames, *raw_radar.shape[2:])
        route = cast(_DenseRoute, self.route)
        complex_maps = route.complex_rd_maps(flattened)
        dense_maps, global_features = route.encode_rd_maps(complex_maps)
        _require(
            dense_maps.shape == (batch * frames, DENSE_TOKEN_CHANNELS, 16, 16),
            "dual route dense map differs from [768,16,16]",
        )
        _require(
            global_features.shape == (batch * frames, 512),
            "dual route global feature differs from 512-D",
        )
        tokens = (
            dense_maps.flatten(start_dim=2)
            .transpose(1, 2)
            .reshape(batch, frames, DENSE_TOKEN_COUNT, DENSE_TOKEN_CHANNELS)
        )
        return DenseTokenEncoderOutput(
            tokens=tokens,
            global_features=global_features.reshape(batch, frames, 512),
        )

    def forward(
        self,
        raw_radar: Tensor,
        *,
        valid_mask: Tensor,
        capture_activations: bool = False,
        single_frame: bool = False,
        spatial_temporal: bool = False,
    ) -> OnlineDualSupervisionPrediction:
        """Encode raw radar, then decode through one explicit reader path.

        ``single_frame`` selects the position-independent spatial boundary plus
        the lightweight readout instead of the window-time eight-block suffix.
        It is a keyword on the model's forward rather than a separate method so
        that a distributed wrapper still owns the call and keeps gradient
        bucketing.  ``spatial_temporal`` instead selects the same spatial
        boundary followed by the eight-block temporal suffix, which is the
        online counterpart of compact-cache Stage B.  The default preserves
        the closed legacy window path exactly.
        """

        _require(
            not (single_frame and spatial_temporal),
            "dual online reader cannot select both single-frame and spatial-temporal paths",
        )
        encoded = self.encode_dense_tokens(raw_radar, valid_mask=valid_mask)
        if capture_activations and encoded.tokens.requires_grad:
            encoded.tokens.retain_grad()
        if single_frame:
            boundary = self.decoder.encode_spatial(encoded.tokens, valid_mask=valid_mask)
            decoded = self.decoder.readout(
                boundary, valid_mask=valid_mask, capture_activations=capture_activations
            )
        elif spatial_temporal:
            boundary = self.decoder.encode_spatial(encoded.tokens, valid_mask=valid_mask)
            decoded = self.decoder.decode_temporal(
                boundary, valid_mask=valid_mask, capture_activations=capture_activations
            )
        else:
            decoded = self.decoder(
                encoded.tokens,
                valid_mask=valid_mask,
                capture_activations=capture_activations,
            )
        return OnlineDualSupervisionPrediction(
            direct=decoded.direct,
            mhr_state=decoded.mhr_state,
            dense_tokens=encoded.tokens,
            normalized_reader_tokens=decoded.normalized_reader_tokens,
        )


def _initialize_direct_compatible_query_gates(
    decoder: nn.Module, *, expected_blocks: int = 4
) -> None:
    """Give fresh Direct coordinates a nonzero path back to dense radar tokens.

    The inherited AdaLN-Zero blocks otherwise leave the learned motion queries
    entirely independent of radar at step zero.  A small query-attention gate
    retains the stable near-identity start while satisfying E2E-DUAL50's
    explicit initial encoder-gradient admission condition.
    """

    blocks = getattr(decoder, "double_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != expected_blocks:
        raise DenseTokenDualSupervisionError("dual double blocks")
    with torch.no_grad():
        for block in blocks:
            modulation = getattr(block, "modulation", None)
            projection = getattr(modulation, "projection", None)
            if not isinstance(projection, nn.Sequential):
                raise DenseTokenDualSupervisionError("dual AdaLN projection")
            final = projection[-1]
            if not isinstance(final, nn.Linear) or final.bias is None:
                raise DenseTokenDualSupervisionError("dual AdaLN output projection")
            _require(final.out_features % 12 == 0, "dual AdaLN output layout")
            channels = final.out_features // 12
            # The 12 chunks are radar attn/MLP then query attn/MLP, each as
            # shift, scale, gate.  Only the radar-to-query attention gate is
            # opened; all modulation weights and all other gate biases stay
            # at their canonical zero initialization.
            final.bias[8 * channels : 9 * channels].fill_(DENSE_QUERY_GATE_INITIAL_BIAS)


def build_dual_decoder_from_direct_seed(
    *,
    mhr_head_seed: int,
    token_channels: int = DENSE_TOKEN_CHANNELS,
    hidden_channels: int = 384,
    attention_heads: int = 8,
    mlp_ratio: int = 4,
    double_stream_blocks: int = 4,
    fused_single_stream_blocks: int = 8,
    maximum_frames: int = 32,
) -> tuple[DenseTokenDualSupervisionDecoder, dict[str, int | float]]:
    """Build a direct-compatible reader before independently seeded MHR heads.

    Constructing the ordinary direct decoder first consumes the shared-model
    random stream exactly as the direct-only route would.  The dual decoder is
    then built inside a private RNG stream; its shared tensors and direct heads
    are overwritten from that direct reference while the MHR/FK-only heads stay
    independently seeded.  Adding MHR supervision therefore cannot perturb the
    route, reader, query, or Direct-XYZ initialization.
    """

    from mmprism.models.dense_token_direct_xyz import DenseTokenDirectXyzDecoder

    kwargs = {
        "token_channels": token_channels,
        "hidden_channels": hidden_channels,
        "attention_heads": attention_heads,
        "mlp_ratio": mlp_ratio,
        "double_stream_blocks": double_stream_blocks,
        "fused_single_stream_blocks": fused_single_stream_blocks,
        "maximum_frames": maximum_frames,
    }
    direct = DenseTokenDirectXyzDecoder(
        token_channels=token_channels,
        hidden_channels=hidden_channels,
        attention_heads=attention_heads,
        mlp_ratio=mlp_ratio,
        double_stream_blocks=double_stream_blocks,
        fused_single_stream_blocks=fused_single_stream_blocks,
        maximum_frames=maximum_frames,
    )
    _initialize_direct_compatible_query_gates(direct, expected_blocks=double_stream_blocks)
    direct_state = direct.state_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(mhr_head_seed)
        dual = DenseTokenDualSupervisionDecoder(**kwargs)
    dual_state = dual.state_dict()
    copied: dict[str, Tensor] = {}
    for name, value in direct_state.items():
        if name in _DIRECT_TO_DUAL_HEAD_NAMES:
            target_name = _DIRECT_TO_DUAL_HEAD_NAMES[name]
        else:
            target_name = name
        if target_name not in dual_state or any(
            target_name.startswith(prefix) for prefix in _MHR_HEAD_PREFIXES
        ):
            continue
        _require(
            dual_state[target_name].shape == value.shape,
            f"direct-seeded dual tensor shape differs: {target_name}",
        )
        copied[target_name] = value
    loaded = dual.load_state_dict(copied, strict=False)
    _require(not loaded.unexpected_keys, "dual direct-seed load has unexpected tensors")
    expected_missing = {
        name for name in dual_state if any(name.startswith(prefix) for prefix in _MHR_HEAD_PREFIXES)
    }
    _require(
        set(loaded.missing_keys) == expected_missing,
        "dual direct-seed load did not leave exactly MHR-only heads",
    )
    _require(
        all(name in copied for name in _DIRECT_TO_DUAL_HEAD_NAMES.values()),
        "dual direct heads were not copied from the direct-compatible stream",
    )
    return dual, {
        "shared_and_direct_tensor_count": len(copied),
        "fresh_mhr_head_tensor_count": len(expected_missing),
        "mhr_head_seed": int(mhr_head_seed),
        "initial_query_attention_gate_bias": DENSE_QUERY_GATE_INITIAL_BIAS,
    }
