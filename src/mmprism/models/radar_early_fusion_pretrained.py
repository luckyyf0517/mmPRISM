"""Complex-radar encoding with a pretrained visual backbone."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet34, wide_resnet50_2
from torchvision.models.swin_transformer import (
    PatchMergingV2,
    SwinTransformer,
    SwinTransformerBlockV2,
)
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from mmprism.models.radar_early_fusion import (
    ANTENNA_COUNT,
    POSE_SHAPE,
    RAW_IQ_TAIL,
    ComplexAntennaMixer,
    ComplexConv2d,
    ComplexGroupRMSNorm,
    ComplexModReLU,
    RadarEarlyFusionProcessorConfig,
    RDContextBlock,
)
from mmprism.radar.phase import apply_channel_phase_compensation
from mmprism.simulation.processor import Processor
from mmprism.simulation.simulator import get_index_full

PRETRAINED_EARLY_FUSION_BACKBONES = frozenset(
    {
        "resnet34",
        "wide_resnet50_2",
        "swin_v2_tiny",
        "swin_v2_small",
        "swin_v2_base",
    }
)

PRETRAINED_EARLY_FUSION_ADAPTERS = frozenset({"a0", "a1", "a_lin"})

PRETRAINED_EARLY_FUSION_READOUTS = frozenset({"stage4", "multiscale"})

PRETRAINED_EARLY_FUSION_ARCHITECTURE_VARIANTS = frozenset(
    {
        "n08",
        "a01_a1_deeper",
        "a02_swin_rd_hybrid",
        "a03_rd_conditioned_mixer",
        "a04_coherence_dual_branch",
        "a05_channel_gated_a1",
        "a06_structured_pose_decoder",
        "a07_linear_projector",
    }
)

PRETRAINED_EARLY_FUSION_MIXER_CHANNELS = 256

PRETRAINED_EARLY_FUSION_INPUT_CHANNELS = 2 * PRETRAINED_EARLY_FUSION_MIXER_CHANNELS

PRETRAINED_EARLY_FUSION_INPUT_GROUPS = 32

PRETRAINED_EARLY_FUSION_WINDOW_SIZE = (8, 8)

DINO_V3_VITB_C01_BACKBONE = "dinov3_vit_base"

DINO_V3_VITB_C01_ARCHITECTURE_VARIANT = "c01_dinov3_vitb_patch2"

DINO_V3_VITB_C03_ARCHITECTURE_VARIANT = "c03_dinov3_vitb_patch2_dense4"

DINO_V3_VITB_C04_ARCHITECTURE_VARIANT = "c04_dinov3_vitb_patch2_direct_bridge"

DINO_V3_VITB_C06_ARCHITECTURE_VARIANT = "c06_dinov3_vitb_patch2_cls_global"

DINO_V3_VITB_M01_ARCHITECTURE_VARIANT = "m01_dinov3_vitb_complex_boundary"

DINO_V3_VITB_M02_ARCHITECTURE_VARIANT = "m02_dinov3_vitb_complex_frontend"

DINO_V3_VITB_M03_ARCHITECTURE_VARIANT = "m03_dinov3_vitb_frontend_expansion"

DINO_V3_VITB_M05_ARCHITECTURE_VARIANT = "m05_dinov3_vitb_growth_dilated_c"

DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT = "m05_dinov3_vitb_growth_dilated_c_alpha_gated"

DINO_V3_VITB_D0_ARCHITECTURE_VARIANT = "d0_dinov3_vitb_m2_continuation_control"

DINO_V3_VITB_D1_ARCHITECTURE_VARIANT = "d1_dinov3_vitb_joint_query_residual_decoder"

DINO_V3_VITB_D3_ARCHITECTURE_VARIANT = "d3_dinov3_vitb_position_conditioned_joint_query"

DINO_V3_VITS_C05_BACKBONE = "dinov3_vit_small"

DINO_V3_VITS_C05_ARCHITECTURE_VARIANT = "c05_dinov3_vits_patch2"

DINO_V3_VITL_C07_BACKBONE = "dinov3_vit_large"

DINO_V3_VITL_C07_ARCHITECTURE_VARIANT = "c07_dinov3_vitl_patch2_direct_bridge"

DINO_V3_VITB_HIDDEN_SIZE = 768

DINO_V3_VITB_REGISTER_TOKEN_COUNT = 4

DINO_V3_VITB_PATCH_GRID = (16, 16)

DINO_V3_VITB_DENSE_FUSION_LAYER_INDICES = (8, 9, 10, 11)

_DINO_V3_VITB_ARCHITECTURE_VARIANTS = frozenset(
    {
        DINO_V3_VITB_C01_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_C03_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_C04_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_C06_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M01_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M02_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M03_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D0_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D1_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D3_ARCHITECTURE_VARIANT,
    }
)

_M_COMPLEX_FRONTEND_ARCHITECTURE_VARIANTS = frozenset(
    {
        DINO_V3_VITB_M02_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M03_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D0_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D1_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D3_ARCHITECTURE_VARIANT,
    }
)

_DINO_V3_DIRECT_PATCH_BRIDGE_ARCHITECTURE_VARIANTS = frozenset(
    {
        DINO_V3_VITB_C04_ARCHITECTURE_VARIANT,
        DINO_V3_VITL_C07_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M01_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M02_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M03_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D0_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D1_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D3_ARCHITECTURE_VARIANT,
    }
)

_N08_ARCHITECTURE_TUPLE = ("swin_v2_small", "a1", "stage4", False, False)

_A07_ARCHITECTURE_TUPLE = ("swin_v2_small", "a_lin", "stage4", False, False)

_A04_GEOMETRY_DISPLACEMENTS = (
    (1, 0),
    (8, 0),
    (24, 0),
    (0, 1),
    (0, 4),
    (0, 6),
    (1, 1),
    (4, 4),
    (6, 6),
)

_A04_BASELINES_PER_DISPLACEMENT = 8

_RESNET_CHANNELS = {
    "resnet34": (64, 128, 256, 512),
    "wide_resnet50_2": (256, 512, 1024, 2048),
}

_SWIN_SPECS = {
    "swin_v2_tiny": ((96, 192, 384, 768), (2, 2, 6, 2), (3, 6, 12, 24)),
    "swin_v2_small": ((96, 192, 384, 768), (2, 2, 18, 2), (3, 6, 12, 24)),
    "swin_v2_base": ((128, 256, 512, 1024), (2, 2, 18, 2), (4, 8, 16, 32)),
}

REGISTERED_PRETRAINED_EARLY_FUSION_VARIANTS = frozenset(
    {
        ("resnet34", "a0", "stage4", True, False),
        ("wide_resnet50_2", "a0", "stage4", True, False),
        ("wide_resnet50_2", "a1", "stage4", True, False),
        ("wide_resnet50_2", "a0", "multiscale", True, False),
        ("wide_resnet50_2", "a0", "stage4", True, True),
        ("wide_resnet50_2", "a1", "multiscale", True, False),
        ("swin_v2_small", "a0", "stage4", False, False),
        ("swin_v2_small", "a1", "stage4", False, False),
        ("swin_v2_small", "a0", "multiscale", False, False),
        ("swin_v2_small", "a0", "stage4", True, False),
        ("swin_v2_small", "a1", "stage4", True, False),
        ("swin_v2_small", "a_lin", "stage4", False, False),
        ("swin_v2_tiny", "a0", "stage4", False, False),
        ("swin_v2_base", "a0", "stage4", False, False),
    }
)


class PretrainedEarlyFusionError(RuntimeError):
    """Raised for unsupported or malformed registered EarlyFusion routes."""


class _AdapterResidualBlock(nn.Module):
    """The fixed ReLU residual block prescribed for the A1 adapter."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.normalization1 = nn.GroupNorm(8, channels)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.normalization2 = nn.GroupNorm(8, channels)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.activation(self.normalization1(self.conv1(inputs)))
        outputs = self.normalization2(self.conv2(outputs))
        return cast(Tensor, self.activation(outputs + inputs))


class _A1ChannelGate(nn.Module):
    """Identity-centred per-sample gate for A1's 192 radar latent channels."""

    def __init__(self, channels: int, *, bottleneck_channels: int = 24) -> None:
        super().__init__()
        self.reduce = nn.Linear(channels, bottleneck_channels)
        self.activation = nn.ReLU()
        self.expand = nn.Linear(bottleneck_channels, channels)
        # The issued gate starts as the exact identity, leaving the N08 path
        # available while its final projection receives a useful first-step gradient.
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        pooled = inputs.mean(dim=(2, 3))
        gate = 1.0 + torch.tanh(self.expand(self.activation(self.reduce(pooled))))
        return inputs * gate[:, :, None, None]


class _A1RadarAdapter(nn.Module):
    """Wide fixed A1 adapter whose final interface matches its pretrained body."""

    def __init__(
        self,
        *,
        output_channels: int,
        use_groupnorm_output: bool,
        extra_residual_blocks: int = 0,
        use_channel_gate: bool = False,
    ) -> None:
        super().__init__()
        if extra_residual_blocks < 0:
            raise PretrainedEarlyFusionError("A1 extra residual depth must be non-negative")
        self.input_projection = nn.Conv2d(
            PRETRAINED_EARLY_FUSION_INPUT_CHANNELS,
            192,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.input_normalization = nn.GroupNorm(8, 192)
        self.activation = nn.ReLU()
        self.residual = _AdapterResidualBlock(192)
        self.extra_residuals: nn.Module
        if extra_residual_blocks:
            self.extra_residuals = nn.Sequential(
                *(_AdapterResidualBlock(192) for _ in range(extra_residual_blocks))
            )
        else:
            self.extra_residuals = nn.Identity()
        self.channel_gate: nn.Module = _A1ChannelGate(192) if use_channel_gate else nn.Identity()
        self.output_projection = nn.Conv2d(192, output_channels, kernel_size=1, bias=False)
        self.output_normalization: nn.Module
        self.output_activation: nn.Module
        if use_groupnorm_output:
            self.output_normalization = nn.GroupNorm(8, output_channels)
            self.output_activation = nn.ReLU()
        else:
            self.output_normalization = nn.Identity()
            self.output_activation = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.activation(self.input_normalization(self.input_projection(inputs)))
        outputs = self.residual(outputs)
        outputs = self.extra_residuals(outputs)
        outputs = self.channel_gate(outputs)
        return cast(
            Tensor,
            self.output_activation(self.output_normalization(self.output_projection(outputs))),
        )


class _A1RadarPrePatchAdapter(nn.Module):
    """C04's retained spatial A1 portion before a direct DINO patch bridge."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(
            PRETRAINED_EARLY_FUSION_INPUT_CHANNELS,
            192,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.input_normalization = nn.GroupNorm(8, 192)
        self.activation = nn.ReLU()
        self.residual = _AdapterResidualBlock(192)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.activation(self.input_normalization(self.input_projection(inputs)))
        return self.residual(outputs)


class _LearnedComplexFrontend(nn.Module):
    """Hot-started local complex RD processor before the retained C04 boundary."""

    def __init__(
        self,
        *,
        use_dilated_block_c: bool,
        gate_dilated_block_c: bool = False,
    ) -> None:
        super().__init__()
        if gate_dilated_block_c and not use_dilated_block_c:
            raise PretrainedEarlyFusionError("an alpha gate requires the dilated Block C")
        channels = PRETRAINED_EARLY_FUSION_MIXER_CHANNELS
        self.block_a = ComplexConv2d(
            input_channels=ANTENNA_COUNT,
            output_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.block_a_normalization = ComplexGroupRMSNorm(channels=channels, groups=16)
        self.block_a_activation = ComplexModReLU(channels=channels)
        self.block_b = ComplexConv2d(
            input_channels=channels,
            output_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.block_b_normalization = ComplexGroupRMSNorm(channels=channels, groups=16)
        self.block_b_activation = ComplexModReLU(channels=channels)
        self.block_c: ComplexConv2d | None
        self.block_c_normalization: ComplexGroupRMSNorm | None
        self.block_c_activation: ComplexModReLU | None
        self.block_c_alpha: nn.Parameter | None
        if use_dilated_block_c:
            self.block_c = ComplexConv2d(
                input_channels=channels,
                output_channels=channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            )
            self.block_c_normalization = ComplexGroupRMSNorm(channels=channels, groups=16)
            self.block_c_activation = ComplexModReLU(channels=channels)
            self.block_c_alpha = nn.Parameter(torch.zeros(())) if gate_dilated_block_c else None
        else:
            self.block_c = None
            self.block_c_normalization = None
            self.block_c_activation = None
            self.block_c_alpha = None

    @torch.no_grad()
    def initialize_from_c04_mixer(self, real_weight: Tensor, imaginary_weight: Tensor) -> None:
        """Install the sole C04-derived M2 state, then create its identity Block B."""

        self.block_a.initialize_center_tap(real_weight, imaginary_weight)
        self.block_b.initialize_complex_identity_center()
        self.block_a_activation.bias.zero_()
        self.block_b_activation.bias.zero_()
        if self.block_c is not None:
            self.block_c.initialize_complex_identity_center()
            if self.block_c_activation is None:  # pragma: no cover - constructor invariant
                raise PretrainedEarlyFusionError("M3 Block C activation is unavailable")
            self.block_c_activation.bias.zero_()
            if self.block_c_alpha is not None:
                self.block_c_alpha.zero_()

    def forward(self, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        block_a_real, block_a_imaginary = self.block_a(real, imaginary)
        block_a_real, block_a_imaginary = self.block_a_normalization(
            block_a_real, block_a_imaginary
        )
        block_a_real, block_a_imaginary = self.block_a_activation(block_a_real, block_a_imaginary)
        block_b_real, block_b_imaginary = self.block_b(block_a_real, block_a_imaginary)
        block_b_real = block_a_real + block_b_real
        block_b_imaginary = block_a_imaginary + block_b_imaginary
        block_b_real, block_b_imaginary = self.block_b_normalization(
            block_b_real, block_b_imaginary
        )
        block_b_real, block_b_imaginary = self.block_b_activation(block_b_real, block_b_imaginary)
        if self.block_c is None:
            return block_b_real, block_b_imaginary
        if (
            self.block_c_normalization is None or self.block_c_activation is None
        ):  # pragma: no cover - constructor invariant
            raise PretrainedEarlyFusionError("M3 Block C normalization is unavailable")
        block_c_real, block_c_imaginary = self.block_c(block_b_real, block_b_imaginary)
        block_c_real = block_b_real + block_c_real
        block_c_imaginary = block_b_imaginary + block_c_imaginary
        block_c_real, block_c_imaginary = self.block_c_normalization(
            block_c_real, block_c_imaginary
        )
        block_c_real, block_c_imaginary = self.block_c_activation(block_c_real, block_c_imaginary)
        if self.block_c_alpha is None:
            return block_c_real, block_c_imaginary
        # M5-v2's ReZero-style gate is deliberately outside the preserved M3
        # inner block: at alpha=0 this is exactly the inherited M2 output,
        # while alpha=1 reproduces M3's Block-C transformation.
        return (
            block_b_real + self.block_c_alpha * (block_c_real - block_b_real),
            block_b_imaginary + self.block_c_alpha * (block_c_imaginary - block_b_imaginary),
        )

    def epoch_diagnostics(self) -> dict[str, float]:
        """Return the M2 structural norms that may be logged after each train epoch."""

        diagnostics = {
            "block_a_noncenter_complex_l2": float(self.block_a.noncenter_weight_l2().detach()),
            "block_b_noncenter_complex_l2": float(self.block_b.noncenter_weight_l2().detach()),
        }
        if self.block_c is not None:
            diagnostics["block_c_noncenter_complex_l2"] = float(
                self.block_c.noncenter_weight_l2().detach()
            )
        if self.block_c_alpha is not None:
            diagnostics["block_c_alpha"] = float(self.block_c_alpha.detach())
        return diagnostics

    def terminal_diagnostics(self) -> dict[str, float]:
        """Return terminal-only spatial-kernel and modReLU summaries."""

        payload = self.epoch_diagnostics()
        blocks = [
            ("block_a_modrelu_bias", self.block_a_activation.bias.detach()),
            ("block_b_modrelu_bias", self.block_b_activation.bias.detach()),
        ]
        if self.block_c_activation is not None:
            blocks.append(("block_c_modrelu_bias", self.block_c_activation.bias.detach()))
        for block_name, bias in blocks:
            values = bias.float()
            payload.update(
                {
                    f"{block_name}_min": float(values.min()),
                    f"{block_name}_p05": float(torch.quantile(values, 0.05)),
                    f"{block_name}_median": float(torch.median(values)),
                    f"{block_name}_p95": float(torch.quantile(values, 0.95)),
                    f"{block_name}_max": float(values.max()),
                    f"{block_name}_l2": float(torch.linalg.vector_norm(values)),
                }
            )
        return payload


class _DenseRDResidualBlock(nn.Module):
    """One fresh dense-RD residual block used only after retained Swin Stage-2."""

    def __init__(
        self, input_channels: int, output_channels: int, *, stride: tuple[int, int]
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.normalization1 = nn.GroupNorm(8, output_channels)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.normalization2 = nn.GroupNorm(8, output_channels)
        if input_channels == output_channels and stride == (1, 1):
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(8, output_channels),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.activation(self.normalization1(self.conv1(inputs)))
        outputs = self.normalization2(self.conv2(outputs))
        return cast(Tensor, self.activation(outputs + self.shortcut(inputs)))


class _DenseRDResidualStage(nn.Module):
    """Two local residual blocks while preserving a dense RD feature map."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.downsample = _DenseRDResidualBlock(input_channels, output_channels, stride=(2, 2))
        self.refine = _DenseRDResidualBlock(output_channels, output_channels, stride=(1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.refine(self.downsample(inputs)))


class _RDConditionedComplexResidualMixer(nn.Module):
    """Rank-limited complex residual ``ΔW(d,r)`` beside the global F256 mixer."""

    _INITIAL_RESIDUAL_SCALE = 1.0e-3

    def __init__(self, *, rank: int = 8) -> None:
        super().__init__()
        if rank < 1:
            raise PretrainedEarlyFusionError("conditional complex mixer rank must be positive")
        self.rank = rank
        self.input_real = nn.Parameter(torch.empty(rank, ANTENNA_COUNT))
        self.input_imaginary = nn.Parameter(torch.empty(rank, ANTENNA_COUNT))
        self.output_real = nn.Parameter(torch.empty(PRETRAINED_EARLY_FUSION_MIXER_CHANNELS, rank))
        self.output_imaginary = nn.Parameter(
            torch.empty(PRETRAINED_EARLY_FUSION_MIXER_CHANNELS, rank)
        )
        self.doppler_real = nn.Parameter(torch.empty(rank, 64))
        self.doppler_imaginary = nn.Parameter(torch.empty(rank, 64))
        self.range_real = nn.Parameter(torch.empty(rank, RAW_IQ_TAIL[2]))
        self.range_imaginary = nn.Parameter(torch.empty(rank, RAW_IQ_TAIL[2]))
        # Keep the issued N08 W0 path effectively intact at initialization,
        # while avoiding an exact zero gate that would withhold gradients from
        # the conditional factors on the first optimizer step.
        self.residual_scale = nn.Parameter(torch.full((rank,), self._INITIAL_RESIDUAL_SCALE))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        antenna_standard_deviation = (1.0 / (2.0 * ANTENNA_COUNT)) ** 0.5
        rank_standard_deviation = (1.0 / (2.0 * self.rank)) ** 0.5
        condition_standard_deviation = 0.5**0.5
        for parameter in (self.input_real, self.input_imaginary):
            nn.init.normal_(parameter, mean=0.0, std=antenna_standard_deviation)
        for parameter in (self.output_real, self.output_imaginary):
            nn.init.normal_(parameter, mean=0.0, std=rank_standard_deviation)
        for parameter in (
            self.doppler_real,
            self.doppler_imaginary,
            self.range_real,
            self.range_imaginary,
        ):
            nn.init.normal_(parameter, mean=0.0, std=condition_standard_deviation)

    def _condition_maps(self) -> tuple[Tensor, Tensor]:
        """Return the rank-wise complex Doppler/Range residual coefficients."""

        condition_real = torch.einsum("qd,qr->qdr", self.doppler_real, self.range_real)
        condition_real = condition_real - torch.einsum(
            "qd,qr->qdr", self.doppler_imaginary, self.range_imaginary
        )
        condition_imaginary = torch.einsum(
            "qd,qr->qdr", self.doppler_real, self.range_imaginary
        ) + torch.einsum("qd,qr->qdr", self.doppler_imaginary, self.range_real)
        return (
            condition_real * self.residual_scale[:, None, None],
            condition_imaginary * self.residual_scale[:, None, None],
        )

    @torch.no_grad()
    def diagnostics(self, global_mixer: ComplexAntennaMixer) -> dict[str, float]:
        """Summarize whether the trained residual is active and RD-conditioned."""

        condition_real, condition_imaginary = self._condition_maps()
        condition = torch.complex(condition_real, condition_imaginary)
        epsilon = torch.finfo(condition_real.dtype).eps
        condition_rms = torch.mean(condition.abs().square()).sqrt()
        denominator = condition_rms.clamp_min(epsilon)

        def relative_rms(values: Tensor) -> Tensor:
            return torch.mean(values.abs().square()).sqrt() / denominator

        # A zero reading means a condition that is constant over the complete
        # D×R grid.  The individual readings make the two axes inspectable.
        dr_variation = relative_rms(condition - condition.mean(dim=(1, 2), keepdim=True))
        doppler_variation = relative_rms(condition - condition.mean(dim=1, keepdim=True))
        range_variation = relative_rms(condition - condition.mean(dim=2, keepdim=True))

        output_factor = torch.complex(self.output_real, self.output_imaginary)
        input_factor = torch.complex(self.input_real, self.input_imaginary).conj()
        output_gram = output_factor.conj().transpose(0, 1) @ output_factor
        input_gram = input_factor.conj() @ input_factor.transpose(0, 1)
        delta_norm_squared = torch.einsum(
            "qp,qdn,pdn,qp->dn",
            output_gram,
            condition.conj(),
            condition,
            input_gram,
        ).real.clamp_min(0.0)
        delta_rms = delta_norm_squared.mean().sqrt()
        global_weight = torch.complex(global_mixer.real_weight, global_mixer.imag_weight)
        global_norm = global_weight.abs().square().sum().sqrt().clamp_min(epsilon)

        return {
            "rank": float(self.rank),
            "initial_residual_scale": self._INITIAL_RESIDUAL_SCALE,
            "residual_scale_mean_abs": float(self.residual_scale.detach().abs().mean()),
            "condition_rms": float(condition_rms),
            "condition_dr_nonconstant_rms_fraction": float(dr_variation),
            "condition_doppler_variation_rms_fraction": float(doppler_variation),
            "condition_range_variation_rms_fraction": float(range_variation),
            "effective_delta_to_w0_frobenius_rms_ratio": float(delta_rms / global_norm),
        }

    def forward(self, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        if (
            real.ndim != 4
            or imaginary.shape != real.shape
            or tuple(real.shape[1:]) != (ANTENNA_COUNT, 64, RAW_IQ_TAIL[2])
        ):
            raise PretrainedEarlyFusionError(
                "conditional mixer expects matching [B,116,64,32] RD maps"
            )
        # vᴴx; the sign follows the complex conjugate of the input-side factor.
        projected_real = torch.einsum("qa,badr->bqdr", self.input_real, real)
        projected_real = projected_real + torch.einsum(
            "qa,badr->bqdr", self.input_imaginary, imaginary
        )
        projected_imaginary = torch.einsum(
            "qa,badr->bqdr", self.input_real, imaginary
        ) - torch.einsum("qa,badr->bqdr", self.input_imaginary, real)
        condition_real, condition_imaginary = self._condition_maps()
        latent_real = (
            projected_real * condition_real[None] - projected_imaginary * condition_imaginary[None]
        )
        latent_imaginary = (
            projected_real * condition_imaginary[None] + projected_imaginary * condition_real[None]
        )
        output_real = torch.einsum("kq,bqdr->bkdr", self.output_real, latent_real)
        output_real = output_real - torch.einsum(
            "kq,bqdr->bkdr", self.output_imaginary, latent_imaginary
        )
        output_imaginary = torch.einsum(
            "kq,bqdr->bkdr", self.output_real, latent_imaginary
        ) + torch.einsum("kq,bqdr->bkdr", self.output_imaginary, latent_real)
        return output_real, output_imaginary


def _a04_geometry_baseline_pairs() -> tuple[Tensor, Tensor, Tensor]:
    """Select aperture-balanced raw-antenna pairs from the canonical array grid."""

    coordinates, antenna_indices = get_index_full()
    if len(coordinates) != ANTENNA_COUNT or tuple(antenna_indices) != tuple(range(ANTENNA_COUNT)):
        raise PretrainedEarlyFusionError(
            "full virtual-array geometry does not match the 116-channel input order"
        )
    coordinate_to_channel = {
        tuple(coordinate): index for index, coordinate in enumerate(coordinates)
    }
    if len(coordinate_to_channel) != ANTENNA_COUNT:
        raise PretrainedEarlyFusionError("full virtual-array geometry contains duplicates")

    left_channels: list[int] = []
    right_channels: list[int] = []
    selected_displacements: list[tuple[int, int]] = []
    for delta_x, delta_y in _A04_GEOMETRY_DISPLACEMENTS:
        candidates: list[tuple[int, int]] = []
        for left_channel, (x_coordinate, y_coordinate) in enumerate(coordinates):
            if y_coordinate != 0:
                continue
            right_channel = coordinate_to_channel.get(
                (x_coordinate + delta_x, y_coordinate + delta_y)
            )
            if right_channel is not None:
                candidates.append((left_channel, right_channel))
        if len(candidates) < _A04_BASELINES_PER_DISPLACEMENT:
            raise PretrainedEarlyFusionError(
                f"virtual-array displacement {(delta_x, delta_y)} has too few baselines"
            )
        selected_positions = [
            selection_index * (len(candidates) - 1) // (_A04_BASELINES_PER_DISPLACEMENT - 1)
            for selection_index in range(_A04_BASELINES_PER_DISPLACEMENT)
        ]
        for candidate_index in selected_positions:
            left_channel, right_channel = candidates[candidate_index]
            left_channels.append(left_channel)
            right_channels.append(right_channel)
            selected_displacements.append((delta_x, delta_y))

    return (
        torch.tensor(left_channels, dtype=torch.int64),
        torch.tensor(right_channels, dtype=torch.int64),
        torch.tensor(selected_displacements, dtype=torch.int64),
    )


class _RawAntennaGeometryCoherenceBranch(nn.Module):
    """Encode fixed-baseline coherence directly from the 116 raw antenna maps."""

    _INITIAL_RESIDUAL_SCALE = 1.0e-3

    def __init__(self) -> None:
        super().__init__()
        left_channels, right_channels, displacements = _a04_geometry_baseline_pairs()
        self.register_buffer("left_channels", left_channels, persistent=True)
        self.register_buffer("right_channels", right_channels, persistent=True)
        self.register_buffer("baseline_displacements", displacements, persistent=True)
        coherence_channels = 2 * len(left_channels)
        self.coherence_normalization = nn.GroupNorm(12, coherence_channels)
        self.input_projection = nn.Conv2d(
            coherence_channels, 64, kernel_size=3, padding=1, bias=False
        )
        self.hidden_normalization = nn.GroupNorm(8, 64)
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(
            64, PRETRAINED_EARLY_FUSION_INPUT_CHANNELS, kernel_size=1, bias=False
        )
        self.residual_scale = nn.Parameter(torch.tensor(self._INITIAL_RESIDUAL_SCALE))

    @staticmethod
    def conjugate_products(
        real: Tensor,
        imaginary: Tensor,
        left_channels: Tensor,
        right_channels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if real.shape != imaginary.shape or real.ndim != 4 or real.shape[1] != ANTENNA_COUNT:
            raise PretrainedEarlyFusionError(
                "coherence branch expects matching 116-channel raw complex RD maps"
            )
        if left_channels.shape != right_channels.shape or left_channels.ndim != 1:
            raise PretrainedEarlyFusionError("coherence baseline indices must be matching vectors")
        left_real = real.index_select(1, left_channels)
        right_real = real.index_select(1, right_channels)
        left_imaginary = imaginary.index_select(1, left_channels)
        right_imaginary = imaginary.index_select(1, right_channels)
        product_real = left_real * right_real + left_imaginary * right_imaginary
        product_imaginary = left_imaginary * right_real - left_real * right_imaginary
        return product_real, product_imaginary

    def forward(self, raw_real: Tensor, raw_imaginary: Tensor) -> Tensor:
        coherence_real, coherence_imaginary = self.conjugate_products(
            raw_real,
            raw_imaginary,
            self.left_channels,
            self.right_channels,
        )
        coherence = torch.cat((coherence_real, coherence_imaginary), dim=1)
        hidden = self.activation(
            self.hidden_normalization(
                self.input_projection(self.coherence_normalization(coherence))
            )
        )
        return cast(
            Tensor,
            self.residual_scale * self.output_projection(hidden),
        )


class _StructuredPoseHead(nn.Module):
    """Decode global arm/wrist anchors and wrist-relative fingers into full24."""

    def __init__(self) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.output = nn.Linear(512, 2 * (3 + 20) * 3)

    def forward(self, features: Tensor) -> Tensor:
        encoded = self.output(self.activation(features)).reshape(features.shape[0], 2, 23, 3)
        global_joints = encoded[:, :, :3]
        finger_offsets = encoded[:, :, 3:]
        duplicate_wrist_offset = torch.zeros_like(global_joints[:, :, :1])
        hand_joints = global_joints[:, :, 2:3] + torch.cat(
            (duplicate_wrist_offset, finger_offsets), dim=2
        )
        return torch.cat((global_joints, hand_joints), dim=2)


class JointQueryResidualDecoder(nn.Module):
    """One shallow per-output-slot reader over the final dense DINO patch tokens.

    The final shared coordinate projection starts at zero, so adding this
    module to the incumbent global readout is exactly prediction-preserving at
    initialization while the query, attention, and token-projection tensors
    retain ordinary trainable initializations.
    """

    def __init__(
        self,
        *,
        token_channels: int = DINO_V3_VITB_HIDDEN_SIZE,
        hidden_channels: int = 256,
        query_count: int = 2 * 24,
        attention_heads: int = 8,
        ffn_channels: int = 512,
    ) -> None:
        super().__init__()
        if (
            token_channels < 1
            or hidden_channels < 1
            or query_count != 2 * 24
            or attention_heads < 1
            or hidden_channels % attention_heads != 0
            or ffn_channels < hidden_channels
        ):
            raise PretrainedEarlyFusionError("D1 decoder dimensions are not registered")
        self.token_channels = token_channels
        self.hidden_channels = hidden_channels
        self.query_count = query_count
        self.token_projection = nn.Linear(token_channels, hidden_channels)
        self.query_embeddings = nn.Parameter(torch.empty(query_count, hidden_channels))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)
        self.self_normalization = nn.LayerNorm(hidden_channels)
        self.self_attention = nn.MultiheadAttention(
            hidden_channels,
            attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.cross_normalization = nn.LayerNorm(hidden_channels)
        self.cross_attention = nn.MultiheadAttention(
            hidden_channels,
            attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.ffn_normalization = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, ffn_channels),
            nn.GELU(),
            nn.Linear(ffn_channels, hidden_channels),
        )
        self.output_projection = nn.Linear(hidden_channels, 3)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        """Return zero-initialized residual xyz for the 48 existing output slots."""

        if (
            patch_tokens.ndim != 3
            or patch_tokens.shape[1] != math.prod(DINO_V3_VITB_PATCH_GRID)
            or patch_tokens.shape[2] != self.token_channels
        ):
            raise PretrainedEarlyFusionError(
                "D1 decoder expects final DINO patch tokens [B,256,768]"
            )
        token_features = self.token_projection(patch_tokens)
        queries = self.query_embeddings.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        self_updates = self.self_attention(
            self.self_normalization(queries),
            self.self_normalization(queries),
            self.self_normalization(queries),
            need_weights=False,
        )[0]
        queries = queries + self_updates
        cross_updates = self.cross_attention(
            self.cross_normalization(queries),
            token_features,
            token_features,
            need_weights=False,
        )[0]
        queries = queries + cross_updates
        queries = queries + self.ffn(self.ffn_normalization(queries))
        return cast(Tensor, self.output_projection(queries))


class PositionConditionedJointQueryResidualDecoder(nn.Module):
    """D1's one-block token reader with detached base-pose query conditioning.

    The conditioning MLP ends in a zero-initialized linear projection.  Thus,
    at initialization this module forms exactly the same learned-slot queries
    as the unconditional D1 form while retaining an ordinary trainable path
    after the residual coordinate head receives its first update.
    """

    _POSITION_FREQUENCY_COUNT = 8

    def __init__(
        self,
        *,
        coordinate_mean_m: tuple[float, float, float],
        coordinate_std_m: tuple[float, float, float],
        token_channels: int = DINO_V3_VITB_HIDDEN_SIZE,
        hidden_channels: int = 256,
        query_count: int = 2 * 24,
        attention_heads: int = 8,
        ffn_channels: int = 512,
    ) -> None:
        super().__init__()
        if (
            token_channels < 1
            or hidden_channels < 1
            or query_count != 2 * 24
            or attention_heads < 1
            or hidden_channels % attention_heads != 0
            or ffn_channels < hidden_channels
        ):
            raise PretrainedEarlyFusionError("D3 decoder dimensions are not registered")
        coordinate_mean = torch.tensor(coordinate_mean_m, dtype=torch.float32)
        coordinate_std = torch.tensor(coordinate_std_m, dtype=torch.float32)
        if (
            coordinate_mean.shape != (3,)
            or coordinate_std.shape != (3,)
            or not bool(torch.isfinite(coordinate_mean).all())
            or not bool(torch.isfinite(coordinate_std).all())
            or bool((coordinate_std <= 0.0).any())
        ):
            raise PretrainedEarlyFusionError(
                "D3 coordinate statistics must be finite positive axes"
            )
        self.token_channels = token_channels
        self.hidden_channels = hidden_channels
        self.query_count = query_count
        # Instantiate the fresh position branch first.  D3 therefore has its
        # own fresh-decoder RNG realization, rather than silently reusing D1's
        # static-query initialization, while its zero final projection keeps
        # the model prediction-preserving at start.
        self.position_conditioner = nn.Sequential(
            nn.Linear(3 * 2 * self._POSITION_FREQUENCY_COUNT, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        final_condition_projection = self.position_conditioner[-1]
        if not isinstance(final_condition_projection, nn.Linear):  # pragma: no cover - fixed above
            raise PretrainedEarlyFusionError("D3 position conditioner is malformed")
        nn.init.zeros_(final_condition_projection.weight)
        nn.init.zeros_(final_condition_projection.bias)
        self.register_buffer(
            "coordinate_mean_m", coordinate_mean.reshape(1, 1, 3), persistent=False
        )
        self.register_buffer("coordinate_std_m", coordinate_std.reshape(1, 1, 3), persistent=False)
        # Geometric wavelengths in metres: 1 m through 1 / 128 m (7.8125 mm).
        self.register_buffer(
            "position_wavelengths_m",
            torch.pow(0.5, torch.arange(self._POSITION_FREQUENCY_COUNT, dtype=torch.float32)),
            persistent=False,
        )
        self.token_projection = nn.Linear(token_channels, hidden_channels)
        self.query_embeddings = nn.Parameter(torch.empty(query_count, hidden_channels))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)
        self.self_normalization = nn.LayerNorm(hidden_channels)
        self.self_attention = nn.MultiheadAttention(
            hidden_channels,
            attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.cross_normalization = nn.LayerNorm(hidden_channels)
        self.cross_attention = nn.MultiheadAttention(
            hidden_channels,
            attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.ffn_normalization = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, ffn_channels),
            nn.GELU(),
            nn.Linear(ffn_channels, hidden_channels),
        )
        self.output_projection = nn.Linear(hidden_channels, 3)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def position_encoding(self, base_pose: Tensor) -> Tensor:
        """Encode the detached incumbent full24 pose into 48 features per slot."""

        if base_pose.ndim != 3 or tuple(base_pose.shape[1:]) != (self.query_count, 3):
            raise PretrainedEarlyFusionError("D3 decoder expects base pose [B,48,3]")
        detached = base_pose.detach()
        normalized = (detached - self.coordinate_mean_m) / self.coordinate_std_m
        phases = (
            normalized.unsqueeze(-1)
            * (2.0 * math.pi * self.coordinate_std_m.unsqueeze(-1))
            / self.position_wavelengths_m.reshape(1, 1, 1, -1)
        )
        encoding = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
        return encoding.flatten(start_dim=2)

    def conditioned_queries(self, base_pose: Tensor) -> Tensor:
        """Return learned queries plus their zero-tax position condition."""

        condition = self.position_conditioner(self.position_encoding(base_pose))
        queries = self.query_embeddings.unsqueeze(0).expand(base_pose.shape[0], -1, -1)
        return queries + condition

    def forward(self, patch_tokens: Tensor, base_pose: Tensor) -> Tensor:
        """Return the zero-initialized residual xyz for existing output slots."""

        if (
            patch_tokens.ndim != 3
            or patch_tokens.shape[1] != math.prod(DINO_V3_VITB_PATCH_GRID)
            or patch_tokens.shape[2] != self.token_channels
        ):
            raise PretrainedEarlyFusionError(
                "D3 decoder expects final DINO patch tokens [B,256,768]"
            )
        if patch_tokens.shape[0] != base_pose.shape[0]:
            raise PretrainedEarlyFusionError("D3 patch-token and base-pose batches differ")
        token_features = self.token_projection(patch_tokens)
        queries = self.conditioned_queries(base_pose)
        self_updates = self.self_attention(
            self.self_normalization(queries),
            self.self_normalization(queries),
            self.self_normalization(queries),
            need_weights=False,
        )[0]
        queries = queries + self_updates
        cross_updates = self.cross_attention(
            self.cross_normalization(queries),
            token_features,
            token_features,
            need_weights=False,
        )[0]
        queries = queries + cross_updates
        queries = queries + self.ffn(self.ffn_normalization(queries))
        return cast(Tensor, self.output_projection(queries))


def _spatial_readout(input_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=1, bias=False),
        nn.GroupNorm(8, 64),
        nn.GELU(),
    )


class PretrainedRadarEarlyFusionRoute(nn.Module):
    """One fixed registered pretrained EarlyFusion architecture."""

    route_id = "radar_early_fusion_pretrained_night01_v1"

    def __init__(
        self,
        *,
        processor_config: RadarEarlyFusionProcessorConfig,
        pose_head_variant: str,
        backbone: str,
        adapter: str,
        readout: str,
        use_context: bool,
        range_preserving: bool,
        channel_phase_rad: Tensor | None = None,
        architecture_variant: str = "n08",
        output_shape: tuple[int, int, int] = POSE_SHAPE,
    ) -> None:
        super().__init__()
        if processor_config.range_bin_count != RAW_IQ_TAIL[2]:
            raise PretrainedEarlyFusionError("registered route requires 32 delivered range bins")
        if pose_head_variant != "relu_linear":
            raise PretrainedEarlyFusionError(
                "registered route requires the incumbent ReLU pose head"
            )
        if output_shape != POSE_SHAPE:
            raise PretrainedEarlyFusionError("registered route requires [2,24,3] output")
        if backbone not in PRETRAINED_EARLY_FUSION_BACKBONES:
            raise PretrainedEarlyFusionError("registered pretrained backbone is unsupported")
        if adapter not in PRETRAINED_EARLY_FUSION_ADAPTERS:
            raise PretrainedEarlyFusionError("registered radar adapter is unsupported")
        if readout not in PRETRAINED_EARLY_FUSION_READOUTS:
            raise PretrainedEarlyFusionError("registered readout is unsupported")
        variant = (backbone, adapter, readout, use_context, range_preserving)
        if variant not in REGISTERED_PRETRAINED_EARLY_FUSION_VARIANTS:
            raise PretrainedEarlyFusionError(
                "pretrained EarlyFusion architecture tuple is not registered"
            )
        if architecture_variant not in PRETRAINED_EARLY_FUSION_ARCHITECTURE_VARIANTS:
            raise PretrainedEarlyFusionError(
                "pretrained EarlyFusion architecture variant is unsupported"
            )
        required_tuple = (
            _A07_ARCHITECTURE_TUPLE
            if architecture_variant == "a07_linear_projector"
            else _N08_ARCHITECTURE_TUPLE
        )
        if architecture_variant != "n08" and variant != required_tuple:
            raise PretrainedEarlyFusionError(
                "N08 follow-up architecture variants require the pure N08 tuple"
            )

        self.output_shape = output_shape
        self.backbone = backbone
        self.adapter_mode = adapter
        self.readout_mode = readout
        self.use_context = use_context
        self.range_preserving = range_preserving
        self.architecture_variant = architecture_variant
        self.is_resnet = backbone in _RESNET_CHANNELS
        self.is_swin = backbone in _SWIN_SPECS
        self.channel_phase_rad: Tensor | None
        if channel_phase_rad is not None:
            phase = channel_phase_rad.detach().clone().reshape(-1)
            if phase.shape != (ANTENNA_COUNT,) or not bool(torch.isfinite(phase).all()):
                raise PretrainedEarlyFusionError(
                    "channel phase calibration must be finite [116] radians"
                )
            self.register_buffer("channel_phase_rad", phase.to(dtype=torch.float32))
        else:
            self.register_buffer("channel_phase_rad", None)
        self.doppler_processor = Processor(
            array_size="full",
            process_range=False,
            process_doppler=True,
            remove_static=processor_config.remove_static,
            learnable_weights=False,
            range_bin_start=0,
            range_bin_count=processor_config.range_bin_count,
            apply_doppler_window=processor_config.apply_doppler_window,
            output_log=False,
            output_db=False,
            cube_normalize=False,
            output_range="db",
            legacy_mode=False,
        )
        self.antenna_mixer = ComplexAntennaMixer(
            output_channels=PRETRAINED_EARLY_FUSION_MIXER_CHANNELS
        )
        self.input_normalization = nn.GroupNorm(
            PRETRAINED_EARLY_FUSION_INPUT_GROUPS,
            PRETRAINED_EARLY_FUSION_INPUT_CHANNELS,
        )
        self.input_activation = nn.GELU()
        self.resnet_body = nn.ModuleDict()
        self.swin_body = nn.ModuleList()
        self.swin_input_normalization: nn.Module = nn.Identity()
        self.rd_context: nn.Module = nn.Identity()
        self.multiscale_readout: nn.Module = nn.Identity()
        self.multiscale_projection: nn.Module = nn.Identity()
        self.hybrid_rd_stage3: nn.Module = nn.Identity()
        self.hybrid_rd_stage4: nn.Module = nn.Identity()
        self.conditional_mixer: nn.Module = nn.Identity()
        self.coherence_branch: nn.Module = nn.Identity()
        self.swin_pretrained_feature_indices: tuple[int, ...] = ()

        if self.is_resnet:
            self._build_resnet_body()
        else:
            self._build_swin_body()

        self.spatial_readout = _spatial_readout(self.encoder_feature_channels)
        height, width = self.final_spatial_shape
        self.feature_projection = nn.Linear(64 * height * width, 512)
        self.feature_normalization = nn.LayerNorm(512)
        self.feature_activation = nn.GELU()
        self.pose_head: nn.Module
        if self.architecture_variant == "a06_structured_pose_decoder":
            self.pose_head = _StructuredPoseHead()
        else:
            self.pose_head = nn.Sequential(nn.ReLU(), nn.Linear(512, 2 * 24 * 3))

    def _build_resnet_body(self) -> None:
        if self.backbone == "resnet34":
            reference = resnet34(weights=None, progress=False)
        elif self.backbone == "wide_resnet50_2":
            reference = wide_resnet50_2(weights=None, progress=False)
        else:  # pragma: no cover - guarded before dispatch
            raise PretrainedEarlyFusionError("invalid ResNet family")
        if self.range_preserving:
            first_block = reference.layer4[0]
            if not isinstance(first_block.conv2, nn.Conv2d) or first_block.downsample is None:
                raise PretrainedEarlyFusionError(
                    "Wide-ResNet range-preserving block is unavailable"
                )
            downsample = first_block.downsample[0]
            if not isinstance(downsample, nn.Conv2d):
                raise PretrainedEarlyFusionError("Wide-ResNet shortcut is unavailable")
            first_block.conv2.stride = (2, 1)
            downsample.stride = (2, 1)
            first_block.stride = (2, 1)
        self.resnet_body = nn.ModuleDict(
            {
                "layer1": reference.layer1,
                "layer2": reference.layer2,
                "layer3": reference.layer3,
                "layer4": reference.layer4,
            }
        )
        _, channel2, channel3, channel4 = _RESNET_CHANNELS[self.backbone]
        if self.adapter_mode == "a0":
            self.radar_adapter: nn.Module = nn.Sequential(
                nn.Conv2d(
                    PRETRAINED_EARLY_FUSION_INPUT_CHANNELS,
                    64,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
            )
        else:
            self.radar_adapter = _A1RadarAdapter(
                output_channels=64,
                use_groupnorm_output=True,
                extra_residual_blocks=(1 if self.architecture_variant == "a01_a1_deeper" else 0),
                use_channel_gate=self.architecture_variant == "a05_channel_gated_a1",
            )
        self.rd_context = RDContextBlock(channels=channel2) if self.use_context else nn.Identity()
        self.encoder_feature_channels = channel4
        self.final_spatial_shape = (4, 8) if self.range_preserving else (4, 4)
        if self.readout_mode == "multiscale":
            self.multiscale_readout = _spatial_readout(channel3)
            self.multiscale_projection = nn.Linear(64 * 8 * 8, 512)
        # Make the layer1 input interface explicit for an accidental upstream
        # torchvision change instead of allowing an opaque convolution error.
        first_layer_conv = reference.layer1[0].conv1
        if not isinstance(first_layer_conv, nn.Conv2d) or first_layer_conv.in_channels != 64:
            raise PretrainedEarlyFusionError("registered ResNet layer1 interface changed")

    def _build_swin_body(self) -> None:
        widths, depths, heads = _SWIN_SPECS[self.backbone]
        reference = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=widths[0],
            depths=list(depths),
            num_heads=list(heads),
            window_size=list(PRETRAINED_EARLY_FUSION_WINDOW_SIZE),
            stochastic_depth_prob=0.0,
            block=SwinTransformerBlockV2,
            downsample_layer=PatchMergingV2,
        )
        features = tuple(reference.features.children())
        if len(features) != 8:
            raise PretrainedEarlyFusionError("issued torchvision Swin V2 body is unavailable")
        if self.architecture_variant == "a02_swin_rd_hybrid":
            # Retain ImageNet Stage1 / PatchMerge1 / Stage2 only.  The late
            # RD stages deliberately start fresh and therefore live outside
            # the strict Swin body loader.
            self.swin_body = nn.ModuleList(features[1:4])
            self.swin_pretrained_feature_indices = (1, 2, 3)
            self.hybrid_rd_stage3 = _DenseRDResidualStage(widths[1], widths[2])
            self.hybrid_rd_stage4 = _DenseRDResidualStage(widths[2], widths[3])
        else:
            self.swin_body = nn.ModuleList(features[1:])
            self.swin_pretrained_feature_indices = tuple(range(1, 8))
        if self.adapter_mode == "a0":
            self.radar_adapter = nn.Conv2d(
                PRETRAINED_EARLY_FUSION_INPUT_CHANNELS,
                widths[0],
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                bias=False,
            )
        elif self.adapter_mode == "a_lin":
            self.radar_adapter = nn.Conv2d(
                PRETRAINED_EARLY_FUSION_INPUT_CHANNELS,
                widths[0],
                kernel_size=1,
                stride=(2, 1),
                bias=False,
            )
        else:
            self.radar_adapter = _A1RadarAdapter(
                output_channels=widths[0],
                use_groupnorm_output=False,
                extra_residual_blocks=(1 if self.architecture_variant == "a01_a1_deeper" else 0),
                use_channel_gate=self.architecture_variant == "a05_channel_gated_a1",
            )
        self.swin_input_normalization = nn.LayerNorm(widths[0])
        self.rd_context = RDContextBlock(channels=widths[1]) if self.use_context else nn.Identity()
        self.encoder_feature_channels = widths[-1]
        self.final_spatial_shape = (4, 4)
        if self.readout_mode == "multiscale":
            self.multiscale_readout = _spatial_readout(widths[2])
            self.multiscale_projection = nn.Linear(64 * 8 * 8, 512)
        if self.architecture_variant == "a03_rd_conditioned_mixer":
            self.conditional_mixer = _RDConditionedComplexResidualMixer(rank=8)
        if self.architecture_variant == "a04_coherence_dual_branch":
            self.coherence_branch = _RawAntennaGeometryCoherenceBranch()

    @staticmethod
    def _require_raw_iq(raw_iq: Tensor) -> None:
        if (
            raw_iq.dtype != torch.float32
            or raw_iq.ndim != 5
            or tuple(raw_iq.shape[1:]) != RAW_IQ_TAIL
            or not bool(torch.isfinite(raw_iq).all())
        ):
            raise PretrainedEarlyFusionError("raw input must be finite float32 [B,64,116,32,2]")

    def complex_rd_maps(self, raw_iq: Tensor) -> Tensor:
        """Return the shared Doppler-only complex map as ``[B,116,64,32]``."""

        self._require_raw_iq(raw_iq)
        raw_complex = torch.view_as_complex(raw_iq.contiguous())
        if self.channel_phase_rad is not None:
            raw_complex = apply_channel_phase_compensation(
                raw_complex, self.channel_phase_rad.to(raw_complex.device)
            )
        spectrum = self.doppler_processor.process_doppler(raw_complex)
        if tuple(spectrum.shape[1:]) != (64, ANTENNA_COUNT, RAW_IQ_TAIL[2]):
            raise PretrainedEarlyFusionError("Doppler prefix produced unexpected spectrum")
        return spectrum.permute(0, 2, 1, 3).contiguous()

    def _encode_resnet(self, inputs: Tensor) -> tuple[Tensor, Tensor | None]:
        outputs = self.radar_adapter(inputs)
        outputs = self.resnet_body["layer1"](outputs)
        outputs = self.resnet_body["layer2"](outputs)
        outputs = self.rd_context(outputs)
        stage3 = self.resnet_body["layer3"](outputs)
        return self.resnet_body["layer4"](stage3), stage3

    def _encode_swin(self, inputs: Tensor) -> tuple[Tensor, Tensor | None]:
        tokens = self.swin_input_normalization(self.radar_adapter(inputs).permute(0, 2, 3, 1))
        if self.architecture_variant == "a02_swin_rd_hybrid":
            for module in self.swin_body:
                tokens = module(tokens)
            stage2 = tokens.permute(0, 3, 1, 2).contiguous()
            hybrid_stage3 = self.hybrid_rd_stage3(stage2)
            return self.hybrid_rd_stage4(hybrid_stage3), hybrid_stage3
        stage3: Tensor | None = None
        for index, module in enumerate(self.swin_body):
            tokens = module(tokens)
            if index == 2 and self.use_context:
                context_inputs = tokens.permute(0, 3, 1, 2).contiguous()
                tokens = self.rd_context(context_inputs).permute(0, 2, 3, 1).contiguous()
            if index == 4:
                stage3 = tokens.permute(0, 3, 1, 2).contiguous()
        return tokens.permute(0, 3, 1, 2).contiguous(), stage3

    def encode_rd_maps(self, rd_maps: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a complex RD map into its final map and 512-D pose feature."""

        if (
            rd_maps.ndim != 4
            or tuple(rd_maps.shape[1:]) != (ANTENNA_COUNT, 64, RAW_IQ_TAIL[2])
            or not torch.is_complex(rd_maps)
        ):
            raise PretrainedEarlyFusionError("RD maps must be complex [B,116,64,32]")
        mixed_real, mixed_imaginary = self.antenna_mixer(rd_maps.real, rd_maps.imag)
        if self.architecture_variant == "a03_rd_conditioned_mixer":
            conditional_real, conditional_imaginary = self.conditional_mixer(
                rd_maps.real, rd_maps.imag
            )
            mixed_real = mixed_real + conditional_real
            mixed_imaginary = mixed_imaginary + conditional_imaginary
        outputs = self.input_activation(
            self.input_normalization(torch.cat((mixed_real, mixed_imaginary), dim=1))
        )
        if self.architecture_variant == "a04_coherence_dual_branch":
            outputs = outputs + self.coherence_branch(rd_maps.real, rd_maps.imag)
        if self.is_resnet:
            encoder_features, stage3 = self._encode_resnet(outputs)
        else:
            encoder_features, stage3 = self._encode_swin(outputs)
        expected_shape = (self.encoder_feature_channels, *self.final_spatial_shape)
        if tuple(encoder_features.shape[1:]) != expected_shape:
            raise PretrainedEarlyFusionError(
                "pretrained RD encoder produced an unexpected final feature shape"
            )
        deep_readout = self.spatial_readout(encoder_features).flatten(start_dim=1)
        feature = self.feature_projection(deep_readout)
        if self.readout_mode == "multiscale":
            if stage3 is None or tuple(stage3.shape[2:]) != (8, 8):
                raise PretrainedEarlyFusionError("multiscale readout requires an 8x8 Stage-3 map")
            feature = feature + self.multiscale_projection(
                self.multiscale_readout(stage3).flatten(start_dim=1)
            )
        feature = self.feature_activation(self.feature_normalization(feature))
        return encoder_features, feature

    def parameter_count(self) -> int:
        """Return the number of trainable route parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture_diagnostics(self) -> dict[str, dict[str, float]] | None:
        """Return terminal-only diagnostics for a registered architecture arm."""

        if self.architecture_variant != "a03_rd_conditioned_mixer":
            return None
        if not isinstance(self.conditional_mixer, _RDConditionedComplexResidualMixer):
            raise PretrainedEarlyFusionError("A03 conditional mixer is unavailable")
        return {"rd_conditioned_mixer": self.conditional_mixer.diagnostics(self.antenna_mixer)}

    def forward(self, raw_iq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoder_features, features = self.encode_rd_maps(self.complex_rd_maps(raw_iq))
        joints = self.pose_head(features)
        if tuple(joints.shape[1:]) != self.output_shape:
            joints = joints.reshape(raw_iq.shape[0], *self.output_shape)
        if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(joints).all()):
            raise PretrainedEarlyFusionError(
                "pretrained EarlyFusion route produced non-finite tensors"
            )
        return joints, features, encoder_features


class DinoV3RadarEarlyFusionRoute(nn.Module):
    """A finite registered C-series DINOv3 patch-2 EarlyFusion route.

    This route deliberately remains separate from the issued ImageNet-body
    route.  It keeps the F256 prefix and A1 adapter, then feeds its 96-channel
    32x32 RD map through a fresh 2x2 patch embedding and an externally loaded
    DINOv3 body.  DINOv3 uses 2D RoPE, so there is no absolute position table
    to interpolate for the 16x16 radar token grid.
    """

    route_id = "radar_early_fusion_dinov3_c01_v1"

    def __init__(
        self,
        *,
        processor_config: RadarEarlyFusionProcessorConfig,
        pose_head_variant: str,
        backbone: str,
        adapter: str,
        readout: str,
        use_context: bool,
        range_preserving: bool,
        channel_phase_rad: Tensor | None = None,
        architecture_variant: str = DINO_V3_VITB_C01_ARCHITECTURE_VARIANT,
        joint_query_position_mean_m: tuple[float, float, float] | None = None,
        joint_query_position_std_m: tuple[float, float, float] | None = None,
        output_shape: tuple[int, int, int] = POSE_SHAPE,
    ) -> None:
        super().__init__()
        if processor_config.range_bin_count != RAW_IQ_TAIL[2]:
            raise PretrainedEarlyFusionError("C01 requires 32 delivered range bins")
        if pose_head_variant != "relu_linear":
            raise PretrainedEarlyFusionError("C01 requires the incumbent ReLU pose head")
        if output_shape != POSE_SHAPE:
            raise PretrainedEarlyFusionError("C01 requires [2,24,3] output")
        if architecture_variant in _DINO_V3_VITB_ARCHITECTURE_VARIANTS:
            expected_backbone = DINO_V3_VITB_C01_BACKBONE
            hidden_size, intermediate_size = 768, 3072
            num_attention_heads, num_hidden_layers = 12, 12
        elif architecture_variant == DINO_V3_VITS_C05_ARCHITECTURE_VARIANT:
            expected_backbone = DINO_V3_VITS_C05_BACKBONE
            hidden_size, intermediate_size = 384, 1536
            num_attention_heads, num_hidden_layers = 6, 12
        elif architecture_variant == DINO_V3_VITL_C07_ARCHITECTURE_VARIANT:
            expected_backbone = DINO_V3_VITL_C07_BACKBONE
            hidden_size, intermediate_size = 1024, 4096
            num_attention_heads, num_hidden_layers = 16, 24
        else:
            raise PretrainedEarlyFusionError(
                "C-series DINOv3 architecture variant is not registered"
            )
        if (
            backbone,
            adapter,
            readout,
            use_context,
            range_preserving,
        ) != (
            expected_backbone,
            "a1",
            "stage4",
            False,
            False,
        ):
            raise PretrainedEarlyFusionError("C-series DINOv3 architecture tuple is not registered")
        if channel_phase_rad is not None:
            raise PretrainedEarlyFusionError("C01 does not permit channel phase calibration")
        requires_position_conditioning = (
            architecture_variant == DINO_V3_VITB_D3_ARCHITECTURE_VARIANT
        )
        if requires_position_conditioning and (
            joint_query_position_mean_m is None or joint_query_position_std_m is None
        ):
            raise PretrainedEarlyFusionError(
                "D3 requires both train-only position-statistics constants"
            )
        if not requires_position_conditioning and (
            joint_query_position_mean_m is not None or joint_query_position_std_m is not None
        ):
            raise PretrainedEarlyFusionError(
                "only D3 permits train-only position-statistics constants"
            )

        self.output_shape = output_shape
        self.backbone = backbone
        self.adapter_mode = adapter
        self.readout_mode = readout
        self.use_context = use_context
        self.range_preserving = range_preserving
        self.architecture_variant = architecture_variant
        self.dino_v3_hidden_size = hidden_size
        self.dino_v3_num_hidden_layers = num_hidden_layers
        self.register_buffer("channel_phase_rad", None)
        self.doppler_processor = Processor(
            array_size="full",
            process_range=False,
            process_doppler=True,
            remove_static=processor_config.remove_static,
            learnable_weights=False,
            range_bin_start=0,
            range_bin_count=processor_config.range_bin_count,
            apply_doppler_window=processor_config.apply_doppler_window,
            output_log=False,
            output_db=False,
            cube_normalize=False,
            output_range="db",
            legacy_mode=False,
        )
        self.antenna_mixer: ComplexAntennaMixer | None
        self.complex_frontend: _LearnedComplexFrontend | None
        if architecture_variant in _M_COMPLEX_FRONTEND_ARCHITECTURE_VARIANTS:
            self.antenna_mixer = None
            self.complex_frontend = _LearnedComplexFrontend(
                use_dilated_block_c=architecture_variant
                in {
                    DINO_V3_VITB_M03_ARCHITECTURE_VARIANT,
                    DINO_V3_VITB_M05_ARCHITECTURE_VARIANT,
                    DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT,
                },
                gate_dilated_block_c=(
                    architecture_variant == DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT
                ),
            )
        else:
            self.antenna_mixer = ComplexAntennaMixer(
                output_channels=PRETRAINED_EARLY_FUSION_MIXER_CHANNELS
            )
            self.complex_frontend = None
        # This is deliberately opt-in and parameter-free.  The default M2
        # route remains byte-for-byte state-dict compatible with its accepted
        # pre-task construction state; RLC1 installs its fixed transform only
        # after that state has been restored.
        self._complex_boundary_transform: nn.Module | None = None
        self.input_normalization: nn.Module
        self.input_activation: nn.Module
        self.complex_boundary_normalization: ComplexGroupRMSNorm | None = None
        self.complex_boundary_activation: ComplexModReLU | None = None
        if architecture_variant == DINO_V3_VITB_M01_ARCHITECTURE_VARIANT:
            # M1 preserves C04's 16-complex-channel GN footprint while keeping
            # I/Q together until the A1 input boundary.
            self.complex_boundary_normalization = ComplexGroupRMSNorm(
                channels=PRETRAINED_EARLY_FUSION_MIXER_CHANNELS,
                groups=16,
            )
            self.complex_boundary_activation = ComplexModReLU(
                channels=PRETRAINED_EARLY_FUSION_MIXER_CHANNELS
            )
            self.input_normalization = nn.Identity()
            self.input_activation = nn.Identity()
        else:
            self.input_normalization = nn.GroupNorm(
                PRETRAINED_EARLY_FUSION_INPUT_GROUPS,
                PRETRAINED_EARLY_FUSION_INPUT_CHANNELS,
            )
            self.input_activation = nn.GELU()
        self.use_direct_patch_bridge = (
            architecture_variant in _DINO_V3_DIRECT_PATCH_BRIDGE_ARCHITECTURE_VARIANTS
        )
        self.use_dino_gradient_checkpointing = (
            architecture_variant == DINO_V3_VITL_C07_ARCHITECTURE_VARIANT
        )
        patch_input_channels = 192 if self.use_direct_patch_bridge else 96
        self.radar_adapter: nn.Module
        if self.use_direct_patch_bridge:
            # C04 keeps A1's spatial projection and residual block, then lets
            # the fresh DINO patch convolution consume its 192-channel map.
            self.radar_adapter = _A1RadarPrePatchAdapter()
        else:
            # C01/C03 retain the full A1 package at the radar/DINO boundary.
            self.radar_adapter = _A1RadarAdapter(output_channels=96, use_groupnorm_output=True)
        self.dino_v3_body = DINOv3ViTModel(
            DINOv3ViTConfig(
                patch_size=2,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_channels=patch_input_channels,
                image_size=32,
                num_register_tokens=DINO_V3_VITB_REGISTER_TOKEN_COUNT,
                rope_theta=100.0,
            )
        )
        self.encoder_feature_channels = hidden_size
        self.final_spatial_shape = DINO_V3_VITB_PATCH_GRID
        self._patch_token_offset = 1 + DINO_V3_VITB_REGISTER_TOKEN_COUNT
        self.use_dense_feature_fusion = (
            architecture_variant == DINO_V3_VITB_C03_ARCHITECTURE_VARIANT
        )
        self.use_cls_token_fusion = architecture_variant == DINO_V3_VITB_C06_ARCHITECTURE_VARIANT
        if self.use_dense_feature_fusion:
            # C03 starts from an equal readout of blocks 9--12. These four
            # scalars are the only new trainable structure; the body, A1 and
            # spatial readout remain the C01 implementation.
            self.dense_feature_fusion_logits = nn.Parameter(
                torch.zeros(len(DINO_V3_VITB_DENSE_FUSION_LAYER_INDICES))
            )
        else:
            self.register_parameter("dense_feature_fusion_logits", None)
        # Pre-activation readout compression is intentionally fresh and is the
        # only decoder change in C01.
        self.spatial_readout = nn.Sequential(
            nn.GroupNorm(8, hidden_size),
            nn.ReLU(),
            nn.Conv2d(
                hidden_size,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.feature_projection = nn.Linear(64 * 4 * 4, 512)
        self.cls_token_projection: nn.Linear | None
        if self.use_cls_token_fusion:
            # C06 keeps the C01 patch-map readout and adds only the pretrained
            # global token through a fresh 768-to-512 residual projection.
            self.cls_token_projection = nn.Linear(hidden_size, 512)
        else:
            self.cls_token_projection = None
        self.feature_normalization = nn.LayerNorm(512)
        self.feature_activation = nn.GELU()
        self.pose_head = nn.Sequential(nn.ReLU(), nn.Linear(512, 2 * 24 * 3))
        self.use_joint_query_residual_decoder = architecture_variant in {
            DINO_V3_VITB_D1_ARCHITECTURE_VARIANT,
            DINO_V3_VITB_D3_ARCHITECTURE_VARIANT,
        }
        self.use_position_conditioned_joint_query_decoder = requires_position_conditioning
        self.joint_query_residual_decoder: (
            JointQueryResidualDecoder | PositionConditionedJointQueryResidualDecoder | None
        )
        if architecture_variant == DINO_V3_VITB_D1_ARCHITECTURE_VARIANT:
            self.joint_query_residual_decoder = JointQueryResidualDecoder(
                token_channels=hidden_size,
                hidden_channels=256,
                query_count=2 * 24,
                attention_heads=8,
                ffn_channels=512,
            )
        elif architecture_variant == DINO_V3_VITB_D3_ARCHITECTURE_VARIANT:
            if joint_query_position_mean_m is None or joint_query_position_std_m is None:
                raise PretrainedEarlyFusionError("D3 position statistics are unavailable")
            self.joint_query_residual_decoder = PositionConditionedJointQueryResidualDecoder(
                coordinate_mean_m=joint_query_position_mean_m,
                coordinate_std_m=joint_query_position_std_m,
                token_channels=hidden_size,
                hidden_channels=256,
                query_count=2 * 24,
                attention_heads=8,
                ffn_channels=512,
            )
        else:
            self.joint_query_residual_decoder = None

    def complex_rd_maps(self, raw_iq: Tensor) -> Tensor:
        """Return the shared Doppler-only complex map as ``[B,116,64,32]``."""

        PretrainedRadarEarlyFusionRoute._require_raw_iq(raw_iq)
        raw_complex = torch.view_as_complex(raw_iq.contiguous())
        spectrum = self.doppler_processor.process_doppler(raw_complex)
        if tuple(spectrum.shape[1:]) != (64, ANTENNA_COUNT, RAW_IQ_TAIL[2]):
            raise PretrainedEarlyFusionError("C01 Doppler prefix produced unexpected spectrum")
        return spectrum.permute(0, 2, 1, 3).contiguous()

    @torch.no_grad()
    def initialize_m02_complex_frontend(
        self,
        real_weight: Tensor,
        imaginary_weight: Tensor,
    ) -> None:
        """Apply M2's sole C04-derived state to its Block-A center tap."""

        if self.complex_frontend is None:
            raise PretrainedEarlyFusionError("M2 complex frontend is unavailable")
        if self.architecture_variant != DINO_V3_VITB_M02_ARCHITECTURE_VARIANT:
            raise PretrainedEarlyFusionError("M2 initializer requires the M2 architecture variant")
        self.complex_frontend.initialize_from_c04_mixer(real_weight, imaginary_weight)

    @torch.no_grad()
    def initialize_m03_complex_frontend(
        self,
        real_weight: Tensor,
        imaginary_weight: Tensor,
    ) -> None:
        """Apply M3's C04-derived Block-A center tap and residual identities."""

        if self.complex_frontend is None:
            raise PretrainedEarlyFusionError("M3 complex frontend is unavailable")
        if self.architecture_variant != DINO_V3_VITB_M03_ARCHITECTURE_VARIANT:
            raise PretrainedEarlyFusionError("M3 initializer requires the M3 architecture variant")
        self.complex_frontend.initialize_from_c04_mixer(real_weight, imaginary_weight)

    @torch.no_grad()
    def initialize_m05_growth_block_c(self) -> None:
        """Initialize M5's only new material as a complex identity Block C."""

        if self.complex_frontend is None:
            raise PretrainedEarlyFusionError("M5 complex frontend is unavailable")
        if self.architecture_variant not in {
            DINO_V3_VITB_M05_ARCHITECTURE_VARIANT,
            DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT,
        }:
            raise PretrainedEarlyFusionError("M5 initializer requires the M5 architecture variant")
        if (
            self.complex_frontend.block_c is None
            or self.complex_frontend.block_c_activation is None
        ):
            raise PretrainedEarlyFusionError("M5 Block C is unavailable")
        self.complex_frontend.block_c.initialize_complex_identity_center()
        self.complex_frontend.block_c_activation.bias.zero_()
        if self.complex_frontend.block_c_alpha is not None:
            self.complex_frontend.block_c_alpha.zero_()

    def _patch_tokens_to_map(self, tokens: Tensor) -> Tensor:
        expected_token_count = self._patch_token_offset + math.prod(DINO_V3_VITB_PATCH_GRID)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            expected_token_count,
            self.dino_v3_hidden_size,
        ):
            raise PretrainedEarlyFusionError(
                "C01 DINOv3 body produced an unexpected token sequence"
            )
        patch_tokens = tokens[:, self._patch_token_offset :, :]
        return patch_tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            self.dino_v3_hidden_size,
            *DINO_V3_VITB_PATCH_GRID,
        )

    def dense_feature_fusion_weights(self) -> Tensor:
        """Return C03's normalized block-9--12 dense-readout weights."""

        if self.dense_feature_fusion_logits is None:
            raise PretrainedEarlyFusionError("C01 does not use dense feature fusion")
        return torch.softmax(self.dense_feature_fusion_logits, dim=0)

    def _cls_token_feature(self, tokens: Tensor) -> Tensor:
        """Project C06's final normalized CLS token, excluding register tokens."""

        if self.cls_token_projection is None:
            raise PretrainedEarlyFusionError("C01 does not use CLS token fusion")
        expected_token_count = self._patch_token_offset + math.prod(DINO_V3_VITB_PATCH_GRID)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            expected_token_count,
            self.dino_v3_hidden_size,
        ):
            raise PretrainedEarlyFusionError(
                "C06 DINOv3 body produced an unexpected token sequence"
            )
        return self.cls_token_projection(tokens[:, 0, :])

    def _dino_token_sequence(self, adapter_outputs: Tensor) -> Tensor:
        """Read C01's final tokens or C03's learned last-four fusion.

        C03 deliberately bypasses only ``DINOv3ViTModel.forward`` so it can
        retain four intermediate block outputs. The per-block calls mirror the
        upstream forward order and reuse its pretrained final LayerNorm.
        """

        if self.use_dino_gradient_checkpointing:
            return self._checkpointed_dino_token_sequence(adapter_outputs)
        if not self.use_dense_feature_fusion:
            return cast(Tensor, self.dino_v3_body(adapter_outputs).last_hidden_state)

        pixel_values = adapter_outputs.to(
            self.dino_v3_body.embeddings.patch_embeddings.weight.dtype
        )
        hidden_states = self.dino_v3_body.embeddings(pixel_values)
        position_embeddings = self.dino_v3_body.rope_embeddings(pixel_values)
        selected_states: list[Tensor] = []
        for layer_index, layer_module in enumerate(self.dino_v3_body.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
            )
            if layer_index in DINO_V3_VITB_DENSE_FUSION_LAYER_INDICES:
                selected_states.append(hidden_states)
        if len(selected_states) != len(DINO_V3_VITB_DENSE_FUSION_LAYER_INDICES):
            raise PretrainedEarlyFusionError("C03 DINOv3 body does not expose blocks 9 through 12")
        weights = self.dense_feature_fusion_weights()
        normalized_states = [self.dino_v3_body.norm(states) for states in selected_states]
        fused_tokens = normalized_states[0] * weights[0]
        for weight, states in zip(weights[1:], normalized_states[1:], strict=True):
            fused_tokens = fused_tokens + states * weight
        return fused_tokens

    def _checkpointed_dino_token_sequence(self, adapter_outputs: Tensor) -> Tensor:
        """Run C07's 24-block DINO body with per-block activation checkpointing."""

        pixel_values = adapter_outputs.to(
            self.dino_v3_body.embeddings.patch_embeddings.weight.dtype
        )
        hidden_states = self.dino_v3_body.embeddings(pixel_values)
        position_embeddings = self.dino_v3_body.rope_embeddings(pixel_values)
        for layer_module in self.dino_v3_body.layer:
            if self.training and torch.is_grad_enabled():

                def run_layer(states: Tensor, layer: nn.Module = layer_module) -> Tensor:
                    return layer(
                        states,
                        attention_mask=None,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = checkpoint(run_layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                )
        return self.dino_v3_body.norm(hidden_states)

    def _mixer_boundary_inputs(self, mixed_real: Tensor, mixed_imaginary: Tensor) -> Tensor:
        """Apply C04's real boundary or M1's structured complex boundary package."""

        if self.complex_boundary_normalization is None:
            return self.input_activation(
                self.input_normalization(torch.cat((mixed_real, mixed_imaginary), dim=1))
            )
        if self.complex_boundary_activation is None:
            raise PretrainedEarlyFusionError("M1 complex boundary activation is unavailable")
        normalized_real, normalized_imaginary = self.complex_boundary_normalization(
            mixed_real, mixed_imaginary
        )
        activated_real, activated_imaginary = self.complex_boundary_activation(
            normalized_real, normalized_imaginary
        )
        return torch.cat((activated_real, activated_imaginary), dim=1)

    def set_parameter_free_complex_boundary_transform(self, transform: nn.Module | None) -> None:
        """Install an opt-in parameter-free transform before the real M2 boundary."""

        if transform is not None and (
            any(True for _ in transform.parameters()) or any(True for _ in transform.buffers())
        ):
            raise PretrainedEarlyFusionError(
                "complex boundary transform must not add parameters or buffers"
            )
        self._complex_boundary_transform = transform

    def encode_rd_maps(self, rd_maps: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the C01 complex RD map into a 16x16 DINOv3 feature map."""

        if (
            rd_maps.ndim != 4
            or tuple(rd_maps.shape[1:]) != (ANTENNA_COUNT, 64, RAW_IQ_TAIL[2])
            or not torch.is_complex(rd_maps)
        ):
            raise PretrainedEarlyFusionError("RD maps must be complex [B,116,64,32]")
        if self.complex_frontend is not None:
            mixed_real, mixed_imaginary = self.complex_frontend(rd_maps.real, rd_maps.imag)
        elif self.antenna_mixer is not None:
            mixed_real, mixed_imaginary = self.antenna_mixer(rd_maps.real, rd_maps.imag)
        else:  # pragma: no cover - every registered route constructs one frontend
            raise PretrainedEarlyFusionError("DINOv3 route has no complex frontend")
        if self._complex_boundary_transform is not None:
            mixed_real, mixed_imaginary = self._complex_boundary_transform(
                mixed_real, mixed_imaginary
            )
        adapter_inputs = self._mixer_boundary_inputs(mixed_real, mixed_imaginary)
        adapter_outputs = self.radar_adapter(adapter_inputs)
        adapter_channels = 192 if self.use_direct_patch_bridge else 96
        if tuple(adapter_outputs.shape[1:]) != (adapter_channels, 32, 32):
            raise PretrainedEarlyFusionError(
                "C-series A1 adapter produced an unexpected patch interface"
            )
        tokens = self._dino_token_sequence(adapter_outputs)
        encoder_features = self._patch_tokens_to_map(tokens)
        if tuple(encoder_features.shape[1:]) != (
            self.dino_v3_hidden_size,
            *DINO_V3_VITB_PATCH_GRID,
        ):
            raise PretrainedEarlyFusionError("C01 DINOv3 body produced an unexpected feature map")
        deep_readout = self.spatial_readout(encoder_features)
        if tuple(deep_readout.shape[1:]) != (64, 4, 4):
            raise PretrainedEarlyFusionError(
                "C01 spatial readout produced an unexpected feature map"
            )
        feature = self.feature_projection(deep_readout.flatten(start_dim=1))
        if self.use_cls_token_fusion:
            feature = feature + self._cls_token_feature(tokens)
        feature = self.feature_activation(self.feature_normalization(feature))
        return encoder_features, feature

    def first_block_activation_statistics(self, raw_iq: Tensor) -> dict[str, object]:
        """Record the post-surgery first-block scale without changing model state."""

        captured: Tensor | None = None

        def capture(_module: nn.Module, _inputs: tuple[Tensor, ...], outputs: Tensor) -> None:
            nonlocal captured
            if not isinstance(outputs, Tensor):
                raise PretrainedEarlyFusionError("C01 first DINOv3 block returned a non-tensor")
            captured = outputs.detach()

        was_training = self.training
        handle = self.dino_v3_body.layer[0].register_forward_hook(capture)
        try:
            self.eval()
            with torch.no_grad():
                self(raw_iq)
        finally:
            handle.remove()
            self.train(was_training)
        if captured is None:  # pragma: no cover - defensive against an upstream body change
            raise PretrainedEarlyFusionError("C01 first DINOv3 block hook did not execute")
        return {
            "shape": list(captured.shape),
            "mean": float(captured.mean().item()),
            "std": float(captured.std(unbiased=False).item()),
            "abs_max": float(captured.abs().max().item()),
        }

    def training_epoch_diagnostics(self) -> dict[str, dict[str, float]] | None:
        """Expose post-epoch M2 kernel growth without inspecting heldout data."""

        if self.complex_frontend is None:
            return None
        return {"learned_complex_frontend": self.complex_frontend.epoch_diagnostics()}

    def architecture_diagnostics(self) -> dict[str, dict[str, float]] | None:
        """Expose terminal-only diagnostics for the finite DINO architecture variants."""

        payload: dict[str, dict[str, float]] = {}
        if self.use_dense_feature_fusion:
            weights = self.dense_feature_fusion_weights().detach()
            payload["dense_last_four_token_fusion"] = {
                f"block_{layer_index + 1}_weight": float(weight)
                for layer_index, weight in zip(
                    DINO_V3_VITB_DENSE_FUSION_LAYER_INDICES, weights, strict=True
                )
            }
        if self.complex_frontend is not None:
            payload["learned_complex_frontend"] = self.complex_frontend.terminal_diagnostics()
        return payload or None

    def parameter_count(self) -> int:
        """Return the number of trainable C01 route parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, raw_iq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoder_features, features = self.encode_rd_maps(self.complex_rd_maps(raw_iq))
        base_joints = self.pose_head(features).reshape(raw_iq.shape[0], *self.output_shape)
        if self.joint_query_residual_decoder is None:
            joints = base_joints
        elif self.architecture_variant == DINO_V3_VITB_D3_ARCHITECTURE_VARIANT:
            if not isinstance(
                self.joint_query_residual_decoder, PositionConditionedJointQueryResidualDecoder
            ):
                raise PretrainedEarlyFusionError("D3 position-conditioned decoder is unavailable")
            patch_tokens = encoder_features.flatten(start_dim=2).transpose(1, 2)
            residual_joints = self.joint_query_residual_decoder(
                patch_tokens,
                base_joints.flatten(start_dim=1, end_dim=2),
            ).reshape(raw_iq.shape[0], *self.output_shape)
            joints = base_joints + residual_joints
        else:
            # ``_patch_tokens_to_map`` preserves the raster order of DINO's
            # 16x16 patch sequence, so flattening this map restores the exact
            # final patch tokens while keeping the incumbent P0 path intact.
            patch_tokens = encoder_features.flatten(start_dim=2).transpose(1, 2)
            residual_joints = self.joint_query_residual_decoder(patch_tokens).reshape(
                raw_iq.shape[0], *self.output_shape
            )
            joints = base_joints + residual_joints
        if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(joints).all()):
            raise PretrainedEarlyFusionError("C01 DINOv3 route produced non-finite tensors")
        return joints, features, encoder_features


def build_pretrained_early_fusion_route(
    *,
    processor_config: RadarEarlyFusionProcessorConfig,
    pose_head_variant: str,
    backbone: str,
    adapter: str,
    readout: str,
    use_context: bool,
    range_preserving: bool,
    channel_phase_rad: Tensor | None = None,
    architecture_variant: str = "n08",
    joint_query_position_mean_m: tuple[float, float, float] | None = None,
    joint_query_position_std_m: tuple[float, float, float] | None = None,
    output_shape: tuple[int, int, int] = POSE_SHAPE,
) -> PretrainedRadarEarlyFusionRoute | DinoV3RadarEarlyFusionRoute:
    """Build only one finite registered pretrained EarlyFusion route."""

    if architecture_variant in {
        DINO_V3_VITB_C01_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_C03_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_C04_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_C06_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M01_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M02_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M03_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_M05_ALPHA_GATED_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D0_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D1_ARCHITECTURE_VARIANT,
        DINO_V3_VITB_D3_ARCHITECTURE_VARIANT,
        DINO_V3_VITS_C05_ARCHITECTURE_VARIANT,
        DINO_V3_VITL_C07_ARCHITECTURE_VARIANT,
    }:
        return DinoV3RadarEarlyFusionRoute(
            processor_config=processor_config,
            pose_head_variant=pose_head_variant,
            backbone=backbone,
            adapter=adapter,
            readout=readout,
            use_context=use_context,
            range_preserving=range_preserving,
            channel_phase_rad=channel_phase_rad,
            architecture_variant=architecture_variant,
            joint_query_position_mean_m=joint_query_position_mean_m,
            joint_query_position_std_m=joint_query_position_std_m,
            output_shape=output_shape,
        )
    return PretrainedRadarEarlyFusionRoute(
        processor_config=processor_config,
        pose_head_variant=pose_head_variant,
        backbone=backbone,
        adapter=adapter,
        readout=readout,
        use_context=use_context,
        range_preserving=range_preserving,
        channel_phase_rad=channel_phase_rad,
        architecture_variant=architecture_variant,
        output_shape=output_shape,
    )
