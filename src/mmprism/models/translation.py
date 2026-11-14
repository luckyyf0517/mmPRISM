from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from mmprism.contracts import (
    POINTCLOUD_ONLY_INPUT_MODE,
    POSE_ONLY_INPUT_MODE,
    POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
    RADAR_FEATURE_ONLY_INPUT_MODE,
    WAVELLM_INPUT_MODES,
)
from mmprism.models.stgcn import DualHandPoseEncoder
from mmprism.models.wavellm_fusion_profiles import (
    WAVELLM_FUSION_COMPACT_CONFIDENCE_V1,
    WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1,
    WAVELLM_FUSION_VARIANTS,
)
from mmprism.models.wavellm_local_geometry import build_reader_geometry_branch
from mmprism.models.wavellm_residual_fusion import (
    POSE_ANCHORED_RESIDUAL_FEATURE_DIM,
    POSE_ANCHORED_RESIDUAL_HIDDEN_DIM,
    PoseAnchoredResidualFeatureProjector,
    PoseAnchoredResidualFusion,
)


class _LanguageModelAPI(Protocol):
    def get_input_embeddings(self) -> nn.Module: ...

    def generate(self, **kwargs: Any) -> Tensor: ...


class _PoseEncoderAPI(Protocol):
    joint_count: int
    coordinate_dim: int
    output_dim: int

    def __call__(self, pose: Tensor, confidence: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class ModalityEncoding:
    embeddings: Tensor
    attention_mask: Tensor
    pose_gate: Tensor


@dataclass(frozen=True)
class TranslationOutput:
    loss: Tensor
    logits: Tensor
    pose_gate: Tensor
    encoder_attention_mask: Tensor


class RadarFeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"radar features must have shape [batch,time,{self.input_dim}], "
                f"got {tuple(features.shape)}"
            )
        return cast(Tensor, self.projection(features))


class ConfidenceAwareFusion(nn.Module):
    def __init__(self, *, joint_count: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.joint_count = joint_count
        confidence_dim = 2 * joint_count
        self.confidence_gate = nn.Sequential(
            nn.Linear(confidence_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self, pose_embeddings: Tensor, radar_embeddings: Tensor, confidence: Tensor
    ) -> tuple[Tensor, Tensor]:
        if pose_embeddings.shape != radar_embeddings.shape or pose_embeddings.ndim != 3:
            raise ValueError(
                "pose and radar embeddings must have identical [batch,time,hidden] shape"
            )
        if confidence.ndim != 4 or confidence.shape[2:] != (2, self.joint_count):
            raise ValueError(f"confidence must have shape [batch,time,2,{self.joint_count}]")
        if confidence.shape[:2] != pose_embeddings.shape[:2]:
            raise ValueError("confidence and modality batch/time dimensions must match")
        confidence = confidence.to(dtype=pose_embeddings.dtype)
        mean_confidence = confidence.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
        learned_pose_gate = self.confidence_gate(confidence.flatten(start_dim=2))
        pose_gate = mean_confidence * learned_pose_gate
        fused = pose_gate * pose_embeddings + (1 - pose_gate) * radar_embeddings
        return self.output(fused), pose_gate


class LearnablePoseOutputScale(nn.Module):
    """A global trainable gain for encoder tokens before language-model input."""

    def __init__(self, initial_scale: float) -> None:
        super().__init__()
        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError("pose output scale must be finite and > 0")
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale), dtype=torch.float32))

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError("pose embeddings must have shape [batch,time,hidden]")
        return embeddings * self.log_scale.exp().to(dtype=embeddings.dtype)


class GeometryGuidedMT5(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        *,
        hidden_size: int,
        input_mode: str = POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
        radar_feature_dim: int | None = None,
        joint_count: int = 24,
        coordinate_dim: int = 3,
        pose_channels: tuple[int, ...] = (64, 128),
        temporal_kernel_size: int = 5,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        pose_encoder: nn.Module | None = None,
        pointcloud_encoder: nn.Module | None = None,
        fusion_variant: str = WAVELLM_FUSION_COMPACT_CONFIDENCE_V1,
        pose_output_scale_initial: float | None = None,
        reader_geometry_branch: str | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be within [0,1)")
        if input_mode not in WAVELLM_INPUT_MODES:
            raise ValueError(f"unsupported translation input mode: {input_mode!r}")
        if input_mode == POSE_ONLY_INPUT_MODE and radar_feature_dim is not None:
            raise ValueError("pose_only model must not declare radar_feature_dim")
        if input_mode != POSE_ONLY_INPUT_MODE and (
            not isinstance(radar_feature_dim, int) or radar_feature_dim < 1
        ):
            raise ValueError("feature-consuming model requires positive radar_feature_dim")
        if fusion_variant not in WAVELLM_FUSION_VARIANTS:
            raise ValueError(f"unsupported WaveLLM fusion variant: {fusion_variant!r}")
        if (
            fusion_variant != WAVELLM_FUSION_COMPACT_CONFIDENCE_V1
            and input_mode != POSE_PLUS_RADAR_FEATURE_INPUT_MODE
        ):
            raise ValueError("non-default fusion variants require pose_plus_radar_feature mode")
        if fusion_variant == WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1 and (
            radar_feature_dim != POSE_ANCHORED_RESIDUAL_FEATURE_DIM
            or hidden_size != POSE_ANCHORED_RESIDUAL_HIDDEN_DIM
        ):
            raise ValueError(
                "pose_anchored_residual_pf_v1 requires 1024-D contextual features and 768-D mT5"
            )
        if pose_output_scale_initial is not None and input_mode in {
            RADAR_FEATURE_ONLY_INPUT_MODE,
            POINTCLOUD_ONLY_INPUT_MODE,
        }:
            raise ValueError("radar_feature_only model must not declare pose output scaling")
        model_config = getattr(language_model, "config", None)
        model_hidden_size = getattr(model_config, "d_model", None)
        if model_hidden_size != hidden_size:
            raise ValueError(
                f"language model hidden size {model_hidden_size!r} does not match {hidden_size}"
            )
        if not callable(getattr(language_model, "get_input_embeddings", None)):
            raise TypeError("language model must expose get_input_embeddings")
        if not callable(getattr(language_model, "generate", None)):
            raise TypeError("language model must expose generate")
        self.language_model = language_model
        self.pointcloud_encoder: nn.Module | None = None
        if input_mode == POINTCLOUD_ONLY_INPUT_MODE:
            if pointcloud_encoder is None:
                raise ValueError("pointcloud_only model requires a point-cloud encoder")
            self.pointcloud_encoder = pointcloud_encoder
        if input_mode not in {RADAR_FEATURE_ONLY_INPUT_MODE, POINTCLOUD_ONLY_INPUT_MODE}:
            if pose_encoder is None:
                pose_encoder = DualHandPoseEncoder(
                    coordinate_dim=coordinate_dim,
                    joint_count=joint_count,
                    channels=pose_channels,
                    output_dim=hidden_size,
                    temporal_kernel_size=temporal_kernel_size,
                    dropout=dropout,
                )
            encoder_contract = cast(_PoseEncoderAPI, pose_encoder)
            if encoder_contract.joint_count != joint_count:
                raise ValueError("pose encoder joint count does not match the model contract")
            if encoder_contract.coordinate_dim != coordinate_dim:
                raise ValueError(
                    "pose encoder coordinate dimension does not match the model contract"
                )
            if encoder_contract.output_dim != hidden_size:
                raise ValueError("pose encoder output dimension does not match mT5 hidden size")
            self.pose_encoder = pose_encoder
            if pose_output_scale_initial is not None:
                self.pose_output_scaler = LearnablePoseOutputScale(pose_output_scale_initial)
        elif pose_encoder is not None:
            raise ValueError("radar_feature_only model must not receive a pose encoder")
        self.input_mode = input_mode
        self.radar_feature_dim = radar_feature_dim
        self.fusion_variant = fusion_variant
        if input_mode != POSE_ONLY_INPUT_MODE:
            self.radar_projector: nn.Module
            if fusion_variant == WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1:
                self.radar_projector = PoseAnchoredResidualFeatureProjector()
            else:
                self.radar_projector = RadarFeatureProjector(
                    cast(int, radar_feature_dim), hidden_size, dropout
                )
        if input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
            self.fusion: nn.Module
            if fusion_variant == WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1:
                self.fusion = PoseAnchoredResidualFusion()
            else:
                self.fusion = ConfidenceAwareFusion(
                    joint_count=joint_count, hidden_size=hidden_size, dropout=dropout
                )
        self.joint_count = joint_count
        self.coordinate_dim = coordinate_dim
        self.label_smoothing = label_smoothing
        self.reader_geometry_branch: nn.Module | None = None
        if reader_geometry_branch is not None:
            if input_mode != POSE_ONLY_INPUT_MODE:
                raise ValueError("reader geometry branches require pose_only input mode")
            self.reader_geometry_branch = build_reader_geometry_branch(
                reader_geometry_branch, hidden_size=hidden_size
            )

    def encode_modalities(
        self,
        pose: Tensor,
        pose_confidence: Tensor,
        radar_features: Tensor | None = None,
        frame_attention_mask: Tensor | None = None,
        pointcloud: Tensor | None = None,
    ) -> ModalityEncoding:
        if self.input_mode == POINTCLOUD_ONLY_INPUT_MODE:
            if pointcloud is None:
                raise ValueError("pointcloud_only model requires a point cloud tensor")
            encoder = cast(nn.Module, self.pointcloud_encoder)
            features = encoder(pointcloud, frame_attention_mask)
            if frame_attention_mask is None:
                frame_attention_mask = torch.ones(
                    features.shape[:2], dtype=torch.long, device=features.device
                )
            mask = frame_attention_mask.to(device=features.device, dtype=features.dtype)
            embeddings = self.radar_projector(features * mask[:, :, None])
            return ModalityEncoding(
                embeddings=embeddings,
                attention_mask=frame_attention_mask.to(device=embeddings.device, dtype=torch.long),
                pose_gate=torch.zeros_like(embeddings),
            )
        if self.input_mode == RADAR_FEATURE_ONLY_INPUT_MODE:
            if radar_features is None:
                raise ValueError("radar_feature_only model requires radar_features")
            if frame_attention_mask is None:
                frame_attention_mask = torch.ones(
                    radar_features.shape[:2], dtype=torch.long, device=radar_features.device
                )
            if frame_attention_mask.shape != radar_features.shape[:2]:
                raise ValueError("frame_attention_mask must have shape [batch,time]")
            mask = frame_attention_mask.to(device=radar_features.device, dtype=radar_features.dtype)
            radar_embeddings = self.radar_projector(radar_features * mask[:, :, None])
            return ModalityEncoding(
                embeddings=radar_embeddings,
                attention_mask=frame_attention_mask.to(
                    device=radar_embeddings.device, dtype=torch.long
                ),
                pose_gate=torch.zeros_like(radar_embeddings),
            )
        if frame_attention_mask is None:
            frame_attention_mask = torch.ones(pose.shape[:2], dtype=torch.long, device=pose.device)
        if frame_attention_mask.shape != pose.shape[:2]:
            raise ValueError("frame_attention_mask must have shape [batch,time]")
        mask = frame_attention_mask.to(device=pose.device, dtype=pose.dtype)
        pose = pose * mask[:, :, None, None, None]
        pose_confidence = pose_confidence * mask[:, :, None, None]
        pose_encoder = cast(_PoseEncoderAPI, self.pose_encoder)
        pose_embeddings = pose_encoder(pose, pose_confidence)
        pose_output_scaler = getattr(self, "pose_output_scaler", None)
        if isinstance(pose_output_scaler, LearnablePoseOutputScale):
            pose_embeddings = pose_output_scaler(pose_embeddings)
        if self.reader_geometry_branch is not None:
            pose_embeddings = pose_embeddings + self.reader_geometry_branch(
                pose, frame_attention_mask.to(dtype=torch.bool)
            )
        if self.input_mode == POSE_ONLY_INPUT_MODE:
            if radar_features is not None:
                raise ValueError("pose_only model must not receive radar_features")
            fused = pose_embeddings
            pose_gate = torch.ones_like(fused)
        else:
            if radar_features is None:
                raise ValueError("pose_plus_radar_feature model requires radar_features")
            radar_features = radar_features * mask[:, :, None]
            if self.fusion_variant == WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1:
                residual_projector = cast(
                    PoseAnchoredResidualFeatureProjector, self.radar_projector
                )
                residual_fusion = cast(PoseAnchoredResidualFusion, self.fusion)
                radar_embeddings = residual_projector(radar_features, frame_attention_mask)
                fused = residual_fusion(pose_embeddings, radar_embeddings, frame_attention_mask)
                pose_gate = torch.ones_like(fused)
            else:
                radar_embeddings = cast(RadarFeatureProjector, self.radar_projector)(radar_features)
                fused, pose_gate = cast(ConfidenceAwareFusion, self.fusion)(
                    pose_embeddings, radar_embeddings, pose_confidence
                )
        return ModalityEncoding(
            embeddings=fused,
            attention_mask=frame_attention_mask.to(device=fused.device, dtype=torch.long),
            pose_gate=pose_gate,
        )

    def _encoder_inputs(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        modalities: ModalityEncoding,
    ) -> tuple[Tensor, Tensor]:
        if prompt_input_ids.ndim != 2 or prompt_attention_mask.shape != prompt_input_ids.shape:
            raise ValueError(
                "prompt IDs and attention mask must have identical [batch,token] shape"
            )
        if prompt_input_ids.shape[0] != modalities.embeddings.shape[0]:
            raise ValueError("prompt and modality batch dimensions must match")
        language_model = cast(_LanguageModelAPI, self.language_model)
        embedding_layer = language_model.get_input_embeddings()
        prompt_embeddings = cast(Tensor, embedding_layer(prompt_input_ids))
        if prompt_embeddings.shape[-1] != modalities.embeddings.shape[-1]:
            raise ValueError("prompt and modality embedding dimensions must match")
        inputs_embeds = torch.cat((prompt_embeddings, modalities.embeddings), dim=1)
        attention_mask = torch.cat(
            (
                prompt_attention_mask.to(device=inputs_embeds.device, dtype=torch.long),
                modalities.attention_mask,
            ),
            dim=1,
        )
        return inputs_embeds, attention_mask

    def forward(
        self,
        *,
        pose: Tensor,
        pose_confidence: Tensor,
        radar_features: Tensor | None = None,
        pointcloud: Tensor | None = None,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        labels: Tensor,
        frame_attention_mask: Tensor | None = None,
    ) -> TranslationOutput:
        modalities = self.encode_modalities(
            pose, pose_confidence, radar_features, frame_attention_mask, pointcloud
        )
        return self.forward_from_modalities(
            modalities=modalities,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            labels=labels,
        )

    def forward_from_modalities(
        self,
        *,
        modalities: ModalityEncoding,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        labels: Tensor,
    ) -> TranslationOutput:
        """Decode labels from an already constructed modality sequence."""

        if labels.ndim != 2 or labels.shape[0] != modalities.embeddings.shape[0]:
            raise ValueError("labels must have shape [batch,target_token]")
        inputs_embeds, attention_mask = self._encoder_inputs(
            prompt_input_ids, prompt_attention_mask, modalities
        )
        outputs: Any = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        loss = functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        return TranslationOutput(
            loss=loss,
            logits=logits,
            pose_gate=modalities.pose_gate,
            encoder_attention_mask=attention_mask,
        )

    def generate(
        self,
        *,
        pose: Tensor,
        pose_confidence: Tensor,
        radar_features: Tensor | None = None,
        pointcloud: Tensor | None = None,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        frame_attention_mask: Tensor | None = None,
        max_new_tokens: int,
        num_beams: int,
    ) -> Tensor:
        modalities = self.encode_modalities(
            pose, pose_confidence, radar_features, frame_attention_mask, pointcloud
        )
        inputs_embeds, attention_mask = self._encoder_inputs(
            prompt_input_ids, prompt_attention_mask, modalities
        )
        language_model = cast(_LanguageModelAPI, self.language_model)
        generated = language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        if not isinstance(generated, Tensor):
            raise TypeError("language model generation did not return a tensor")
        return generated
