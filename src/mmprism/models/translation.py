from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mmprism.models.stgcn import DualHandPoseEncoder


class _LanguageModelAPI(Protocol):
    def get_input_embeddings(self) -> nn.Module: ...

    def generate(self, **kwargs: Any) -> Tensor: ...


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
            raise ValueError(
                f"confidence must have shape [batch,time,2,{self.joint_count}]"
            )
        if confidence.shape[:2] != pose_embeddings.shape[:2]:
            raise ValueError("confidence and modality batch/time dimensions must match")
        confidence = confidence.to(dtype=pose_embeddings.dtype)
        mean_confidence = confidence.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
        learned_pose_gate = self.confidence_gate(confidence.flatten(start_dim=2))
        pose_gate = mean_confidence * learned_pose_gate
        fused = pose_gate * pose_embeddings + (1 - pose_gate) * radar_embeddings
        return self.output(fused), pose_gate


class GeometryGuidedMT5(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        *,
        hidden_size: int,
        radar_feature_dim: int,
        joint_count: int = 24,
        coordinate_dim: int = 3,
        pose_channels: tuple[int, ...] = (64, 128),
        temporal_kernel_size: int = 5,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be within [0,1)")
        model_config = getattr(language_model, "config", None)
        model_hidden_size = getattr(model_config, "d_model", None)
        if model_hidden_size != hidden_size:
            raise ValueError(
                f"language model hidden size {model_hidden_size!r} does not match {hidden_size}"
            )
        if not callable(getattr(language_model, "get_input_embeddings", None)):
            raise ValueError("language model must expose get_input_embeddings")
        if not callable(getattr(language_model, "generate", None)):
            raise ValueError("language model must expose generate")
        self.language_model = language_model
        self.pose_encoder = DualHandPoseEncoder(
            coordinate_dim=coordinate_dim,
            joint_count=joint_count,
            channels=pose_channels,
            output_dim=hidden_size,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.radar_projector = RadarFeatureProjector(radar_feature_dim, hidden_size, dropout)
        self.fusion = ConfidenceAwareFusion(
            joint_count=joint_count, hidden_size=hidden_size, dropout=dropout
        )
        self.joint_count = joint_count
        self.coordinate_dim = coordinate_dim
        self.label_smoothing = label_smoothing

    def encode_modalities(
        self,
        pose: Tensor,
        pose_confidence: Tensor,
        radar_features: Tensor,
        frame_attention_mask: Tensor | None = None,
    ) -> ModalityEncoding:
        if frame_attention_mask is None:
            frame_attention_mask = torch.ones(
                pose.shape[:2], dtype=torch.long, device=pose.device
            )
        if frame_attention_mask.shape != pose.shape[:2]:
            raise ValueError("frame_attention_mask must have shape [batch,time]")
        mask = frame_attention_mask.to(device=pose.device, dtype=pose.dtype)
        pose = pose * mask[:, :, None, None, None]
        pose_confidence = pose_confidence * mask[:, :, None, None]
        radar_features = radar_features * mask[:, :, None]
        pose_embeddings = self.pose_encoder(pose, pose_confidence)
        radar_embeddings = self.radar_projector(radar_features)
        fused, pose_gate = self.fusion(
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
        radar_features: Tensor,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        labels: Tensor,
        frame_attention_mask: Tensor | None = None,
    ) -> TranslationOutput:
        if labels.ndim != 2 or labels.shape[0] != pose.shape[0]:
            raise ValueError("labels must have shape [batch,target_token]")
        modalities = self.encode_modalities(
            pose, pose_confidence, radar_features, frame_attention_mask
        )
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
        radar_features: Tensor,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        frame_attention_mask: Tensor | None = None,
        max_new_tokens: int,
        num_beams: int,
    ) -> Tensor:
        modalities = self.encode_modalities(
            pose, pose_confidence, radar_features, frame_attention_mask
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
            raise RuntimeError("language model generation did not return a tensor")
        return generated
