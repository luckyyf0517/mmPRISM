"""Complex antenna mixing and range-Doppler feature processing."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RAW_IQ_TAIL = (64, 116, 32, 2)

ANTENNA_COUNT = 116

LATENT_ANTENNA_CHANNELS = 128

POSE_SHAPE = (2, 24, 3)

COMPLEX_GROUP_RMS_GROUPS = 16

COMPLEX_GROUP_RMS_EPSILON = 1e-6

COMPLEX_MODRELU_EPSILON = 1e-6

EARLY_FUSION_ENCODER_SCALES = {
    "small": (64, 96, 160, 256),
    "large": (192, 288, 480, 768),
}

RD_CONTEXT_CHANNELS = 96

RD_CONTEXT_SUPPORTED_CHANNELS = frozenset(
    {
        *(widths[1] for widths in EARLY_FUSION_ENCODER_SCALES.values()),
        128,
        192,
        512,
    }
)

RD_CONTEXT_SPATIAL_SHAPE = (16, 16)

RD_CONTEXT_TOKEN_DIM = 192


class RadarEarlyFusionError(RuntimeError):
    """Raised when the EarlyFusion route receives an unsupported tensor."""


class RadarEarlyFusionProcessorConfig(Protocol):
    """The issued Processor fields consumed by the Doppler-only prefix."""

    @property
    def remove_static(self) -> bool: ...

    @property
    def range_bin_count(self) -> int: ...

    @property
    def apply_doppler_window(self) -> bool: ...


class ComplexAntennaMixer(nn.Module):
    """One shared complex antenna map at every Doppler/range location."""

    def __init__(
        self,
        *,
        input_channels: int = ANTENNA_COUNT,
        output_channels: int = LATENT_ANTENNA_CHANNELS,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.real_weight = nn.Parameter(torch.empty(output_channels, input_channels))
        self.imag_weight = nn.Parameter(torch.empty(output_channels, input_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = math.sqrt(1.0 / (2.0 * self.input_channels))
        nn.init.normal_(self.real_weight, mean=0.0, std=standard_deviation)
        nn.init.normal_(self.imag_weight, mean=0.0, std=standard_deviation)

    def forward(self, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        if real.ndim != 4 or imaginary.shape != real.shape or real.shape[1] != self.input_channels:
            raise RadarEarlyFusionError(
                "complex antenna mixer expects matching [B,116,Doppler,range] tensors"
            )
        mixed_real = torch.einsum("ka,badr->bkdr", self.real_weight, real)
        mixed_real = mixed_real - torch.einsum("ka,badr->bkdr", self.imag_weight, imaginary)
        mixed_imaginary = torch.einsum("ka,badr->bkdr", self.real_weight, imaginary)
        mixed_imaginary = mixed_imaginary + torch.einsum("ka,badr->bkdr", self.imag_weight, real)
        return mixed_real, mixed_imaginary


class ComplexConv2d(nn.Module):
    """Bias-free complex 2D convolution over a paired real/IQ feature map."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if (
            input_channels <= 0
            or output_channels <= 0
            or kernel_size <= 0
            or padding < 0
            or dilation <= 0
        ):
            raise ValueError(
                "complex convolution requires positive channels/kernel/dilation "
                "and nonnegative padding"
            )
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.real_weight = nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.imag_weight = nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = math.sqrt(1.0 / (2.0 * self.input_channels * self.kernel_size**2))
        nn.init.normal_(self.real_weight, mean=0.0, std=standard_deviation)
        nn.init.normal_(self.imag_weight, mean=0.0, std=standard_deviation)

    def forward(self, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        if real.ndim != 4 or imaginary.shape != real.shape or real.shape[1] != self.input_channels:
            raise RadarEarlyFusionError(
                "complex convolution expects matching [B,channels,Doppler,range] tensors"
            )
        real_from_real = F.conv2d(
            real, self.real_weight, padding=self.padding, dilation=self.dilation
        )
        real_from_imaginary = F.conv2d(
            imaginary, self.imag_weight, padding=self.padding, dilation=self.dilation
        )
        imaginary_from_real = F.conv2d(
            real, self.imag_weight, padding=self.padding, dilation=self.dilation
        )
        imaginary_from_imaginary = F.conv2d(
            imaginary, self.real_weight, padding=self.padding, dilation=self.dilation
        )
        return (
            real_from_real - real_from_imaginary,
            imaginary_from_imaginary + imaginary_from_real,
        )

    @torch.no_grad()
    def initialize_center_tap(self, real_weight: Tensor, imaginary_weight: Tensor) -> None:
        """Set one complex pointwise map at the central kernel tap and zero the rest."""

        expected_shape = (self.output_channels, self.input_channels)
        if (
            tuple(real_weight.shape) != expected_shape
            or tuple(imaginary_weight.shape) != expected_shape
            or not torch.is_floating_point(real_weight)
            or not torch.is_floating_point(imaginary_weight)
        ):
            raise RadarEarlyFusionError(
                "center-tap initialization requires matching floating "
                "[out_channels,input_channels] weights"
            )
        center = self.kernel_size // 2
        self.real_weight.zero_()
        self.imag_weight.zero_()
        self.real_weight[:, :, center, center].copy_(
            real_weight.to(device=self.real_weight.device, dtype=self.real_weight.dtype)
        )
        self.imag_weight[:, :, center, center].copy_(
            imaginary_weight.to(device=self.imag_weight.device, dtype=self.imag_weight.dtype)
        )

    @torch.no_grad()
    def initialize_complex_identity_center(self) -> None:
        """Set a square complex convolution to a center-tap complex identity."""

        if self.input_channels != self.output_channels:
            raise RadarEarlyFusionError(
                "complex identity initialization requires square channel dimensions"
            )
        identity = torch.eye(
            self.output_channels,
            device=self.real_weight.device,
            dtype=self.real_weight.dtype,
        )
        self.initialize_center_tap(identity, torch.zeros_like(identity))

    @torch.no_grad()
    def noncenter_weight_l2(self) -> Tensor:
        """Return the combined real/IQ L2 norm excluding the central spatial tap."""

        center = self.kernel_size // 2
        total_squared = self.real_weight.square().sum() + self.imag_weight.square().sum()
        center_squared = (
            self.real_weight[:, :, center, center].square().sum()
            + self.imag_weight[:, :, center, center].square().sum()
        )
        return torch.sqrt((total_squared - center_squared).clamp_min(0.0))


class ComplexGroupRMSNorm(nn.Module):
    """Normalize I/Q jointly with one positive RMS scale per complex group."""

    def __init__(
        self,
        *,
        channels: int = LATENT_ANTENNA_CHANNELS,
        groups: int = COMPLEX_GROUP_RMS_GROUPS,
        eps: float = COMPLEX_GROUP_RMS_EPSILON,
    ) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError("complex RMS normalization requires channels divisible by groups")
        if eps <= 0.0:
            raise ValueError("complex RMS normalization epsilon must be positive")
        self.channels = channels
        self.groups = groups
        self.eps = eps

    def forward(self, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        if (
            real.ndim != 4
            or imaginary.shape != real.shape
            or real.shape[1] != self.channels
            or not torch.is_floating_point(real)
            or not torch.is_floating_point(imaginary)
        ):
            raise RadarEarlyFusionError(
                "complex RMS normalization expects matching floating [B,128,Doppler,range] tensors"
            )
        batch_size, _, doppler_bins, range_bins = real.shape
        channels_per_group = self.channels // self.groups
        grouped_real = real.reshape(
            batch_size, self.groups, channels_per_group, doppler_bins, range_bins
        )
        grouped_imaginary = imaginary.reshape(
            batch_size, self.groups, channels_per_group, doppler_bins, range_bins
        )
        # ``mean(...)/2`` is exactly the RMS denominator over all I/Q scalars:
        # sum(Re² + Im²) / (2 * complex_channels * Doppler * range).
        squared_rms = 0.5 * (grouped_real.square() + grouped_imaginary.square()).mean(
            dim=(2, 3, 4), keepdim=True
        )
        scale = torch.sqrt(squared_rms + self.eps)
        return (
            (grouped_real / scale).reshape_as(real),
            (grouped_imaginary / scale).reshape_as(imaginary),
        )


class ComplexModReLU(nn.Module):
    """Apply a channel-wise modReLU while preserving each active I/Q phase."""

    def __init__(
        self,
        *,
        channels: int = LATENT_ANTENNA_CHANNELS,
        eps: float = COMPLEX_MODRELU_EPSILON,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("complex modReLU requires a positive channel count")
        if eps <= 0.0:
            raise ValueError("complex modReLU epsilon must be positive")
        self.channels = channels
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        if (
            real.ndim != 4
            or imaginary.shape != real.shape
            or real.shape[1] != self.channels
            or not torch.is_floating_point(real)
            or not torch.is_floating_point(imaginary)
        ):
            raise RadarEarlyFusionError(
                "complex modReLU expects matching floating [B,channels,Doppler,range] tensors"
            )
        squared_magnitude = real.square() + imaginary.square()
        active = squared_magnitude >= self.eps**2
        # Clamp before sqrt so the inactive branch has finite zero gradients.
        magnitude = torch.sqrt(squared_magnitude.clamp_min(self.eps**2))
        thresholded = torch.relu(magnitude + self.bias.reshape(1, -1, 1, 1))
        scale = torch.where(active, thresholded / magnitude, torch.zeros_like(magnitude))
        return real * scale, imaginary * scale


class RDContextBlock(nn.Module):
    """Issued global RD-token interaction placed after a 16×16 Stage-2 map."""

    def __init__(
        self,
        *,
        channels: int = RD_CONTEXT_CHANNELS,
        spatial_shape: tuple[int, int] = RD_CONTEXT_SPATIAL_SHAPE,
        token_dim: int = RD_CONTEXT_TOKEN_DIM,
    ) -> None:
        super().__init__()
        height, width = spatial_shape
        if channels not in RD_CONTEXT_SUPPORTED_CHANNELS:
            raise ValueError("RD Context Block channels are not an issued Stage-2 width")
        if spatial_shape != RD_CONTEXT_SPATIAL_SHAPE:
            raise ValueError("RD Context Block is fixed to the Stage-2 [*,16,16] contract")
        if token_dim != RD_CONTEXT_TOKEN_DIM or token_dim % 4 != 0:
            raise ValueError("RD Context Block requires the fixed 192-D token contract")
        self.channels = channels
        self.spatial_shape = spatial_shape
        self.token_dim = token_dim
        self.input_projection = nn.Linear(channels, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=6,
            dim_feedforward=768,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-LN is required by the issued block and PyTorch cannot use its
        # nested-tensor optimization in that mode. State this explicitly so a
        # normal construction does not emit a misleading runtime warning.
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output_projection = nn.Linear(token_dim, channels)
        self.register_buffer(
            "position_encoding",
            self._fixed_position_encoding(height, width, token_dim),
            persistent=False,
        )

    @staticmethod
    def _axis_position_encoding(length: int, dimensions: int) -> Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequency_indices = torch.arange(0, dimensions, 2, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10_000.0) * frequency_indices / dimensions)
        encoding = torch.zeros(length, dimensions, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        return encoding

    @classmethod
    def _fixed_position_encoding(cls, height: int, width: int, token_dim: int) -> Tensor:
        axis_dimensions = token_dim // 2
        doppler_encoding = cls._axis_position_encoding(height, axis_dimensions)
        range_encoding = cls._axis_position_encoding(width, axis_dimensions)
        return torch.cat(
            (
                doppler_encoding[:, None, :].expand(height, width, axis_dimensions),
                range_encoding[None, :, :].expand(height, width, axis_dimensions),
            ),
            dim=-1,
        ).reshape(1, height * width, token_dim)

    def _to_tokens(self, inputs: Tensor) -> Tensor:
        if (
            inputs.ndim != 4
            or inputs.shape[1] != self.channels
            or tuple(inputs.shape[2:]) != self.spatial_shape
        ):
            raise RadarEarlyFusionError(
                f"RD Context Block expects [B,{self.channels},16,16] Stage-2 features"
            )
        return inputs.flatten(start_dim=2).transpose(1, 2)

    def _from_tokens(self, tokens: Tensor) -> Tensor:
        height, width = self.spatial_shape
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (height * width, self.channels):
            raise RadarEarlyFusionError("RD Context Block produced an invalid token shape")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.channels, height, width)

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = self.input_projection(self._to_tokens(inputs))
        position_encoding = self.position_encoding.to(device=tokens.device, dtype=tokens.dtype)
        tokens = self.transformer(tokens + position_encoding)
        return inputs + self._from_tokens(self.output_projection(tokens))
