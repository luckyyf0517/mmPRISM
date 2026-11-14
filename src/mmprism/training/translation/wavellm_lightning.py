from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from mmprism.contracts import wavellm_input_mode_uses_radar_feature
from mmprism.data.csl_daily_contextual_pf_delivery import (
    CslDailyContextualPFWaveLLMDeliveryError,
    validate_csl_daily_contextual_pf_wavellm_payload_contract,
)
from mmprism.data.csl_daily_contextual_pf_split_view import (
    CslDailyContextualPFExternalSplitView,
    CslDailyContextualPFSplitViewError,
    load_csl_daily_contextual_pf_external_split_view,
    resolve_csl_daily_contextual_pf_parquet_references,
)
from mmprism.data.parquet_delivery import (
    ParquetDeliveryError,
    ParquetDeliveryValidation,
    ParquetSignLanguageTranslationDataset,
    load_published_parquet_delivery_validation,
)
from mmprism.data.sign_language_translation import (
    SignLanguageTranslationBatch,
    SignLanguageTranslationSample,
    collate_sign_language_translation_samples,
)
from mmprism.models.translation import GeometryGuidedMT5, ModalityEncoding, TranslationOutput
from mmprism.models.wavellm_factory import WaveLLMModelFactoryError, describe_wavellm_pose_encoder
from mmprism.models.wavellm_fusion_profiles import WAVELLM_FUSION_COMPACT_CONFIDENCE_V1
from mmprism.training.translation.wavellm_conditional_discrimination import (
    ConditionalNegativePairs,
    conditional_pair_loss,
    select_conditional_negative_pairs,
    sequence_log_probability,
)
from mmprism.training.translation.wavellm_run_config import WaveLLMRunConfig

WAVELLM_LIGHTNING_CHECKPOINT_SCHEMA = "mmprism.wavellm_lightning_checkpoint.v1"

WAVELLM_LIGHTNING_HEARTBEAT_SCHEMA = "mmprism.wavellm_lightning_heartbeat.v1"

WAVELLM_LIGHTNING_PROGRESS_SCHEMA = "mmprism.wavellm_lightning_progress.v1"

WARMUP_OPTIMIZER_STEPS = 100

HEARTBEAT_BATCH_INTERVAL = 10


class WaveLLMLightningError(RuntimeError):
    """Raised when the WaveLLM Lightning lifecycle contract fails."""


def translation_parquet_source_identity(
    canonical_root: Path,
    validation: ParquetDeliveryValidation,
    *,
    input_mode: str,
    split_view: CslDailyContextualPFExternalSplitView | None = None,
    contextual_input_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind one canonical translation build and, when used, its membership view."""

    identity: dict[str, object] = {
        "backend": "parquet_delivery",
        "delivery_root": str(canonical_root),
        "build_id": validation.build_id,
        "index_sha256": validation.index_sha256,
        "schema_fingerprint": validation.schema_fingerprint,
        "input_mode": input_mode,
    }
    if split_view is not None:
        identity["split_view"] = split_view.checkpoint_identity()
    if contextual_input_contract is not None:
        identity["contextual_input_contract"] = dict(contextual_input_contract)
    return identity


def contextual_pf_input_contract_observation(
    canonical_root: Path,
    config: WaveLLMRunConfig,
    *,
    split_view: CslDailyContextualPFExternalSplitView | None,
) -> dict[str, object] | None:
    """Bind a named contextual tensor product before opening a consumer route."""

    contract = config.contextual_input_contract
    if contract is None:
        return None
    if config.initialization is not None:
        expected_view = Path(config.initialization.expected_split_view_root).resolve()
        if split_view is None or split_view.root != expected_view:
            raise WaveLLMLightningError(
                "legacy contextual P+F requires its declared official external split view"
            )
    try:
        return validate_csl_daily_contextual_pf_wavellm_payload_contract(
            canonical_root,
            pose_contract=contract.pose_contract,
            feature_identity=contract.feature_identity,
        )
    except CslDailyContextualPFWaveLLMDeliveryError as error:
        raise WaveLLMLightningError(
            f"contextual canonical payload violates its configured contract: {error}"
        ) from error


class TranslationDatasetLike(Protocol):
    def __len__(self) -> int: ...

    def load_sample(self, index: int) -> SignLanguageTranslationSample: ...


class TokenizerLike(Protocol):
    pad_token_id: int | None

    def __call__(self, texts: Sequence[str], **kwargs: object) -> Mapping[str, Tensor]: ...

    def batch_decode(self, token_ids: Tensor, *, skip_special_tokens: bool) -> Sequence[str]: ...


class _DatasetAdapter(Dataset[SignLanguageTranslationSample]):
    def __init__(self, source: TranslationDatasetLike) -> None:
        if len(source) < 1:
            raise WaveLLMLightningError("WaveLLM dataset must contain at least one sample")
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> SignLanguageTranslationSample:
        return self.source.load_sample(index)


class WaveLLMDataModule(L.LightningDataModule):
    """DataLoader adapter for L2 and published Parquet translation datasets."""

    def __init__(
        self,
        train_dataset: TranslationDatasetLike,
        validation_dataset: TranslationDatasetLike,
        config: WaveLLMRunConfig,
        *,
        seed: int,
        source_identity: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.train_source = train_dataset
        self.validation_source = validation_dataset
        self.source_identity = dict(source_identity or {})
        group_ids = getattr(train_dataset, "sample_group_ids", None)
        self.train_group_ids = dict(group_ids) if isinstance(group_ids, Mapping) else {}
        self._train_dataset = _DatasetAdapter(train_dataset)
        self._validation_dataset = _DatasetAdapter(validation_dataset)

    @classmethod
    def from_mesh_fk_l2_handoff(
        cls,
        handoff_root: str | Path,
        l2_root: str | Path,
        config: WaveLLMRunConfig,
        *,
        seed: int,
    ) -> WaveLLMDataModule:
        """Build split views from one validated handoff-bound L2 source.

        The closed reader re-validates every handoff/build/index/member/
        normalization binding at construction; both splits must come from
        the SAME handoff root or the second construction fails its digest
        checks.  The canonical split-agnostic source identity becomes the
        run/checkpoint binding so reload/evaluation can reject drift.
        """

        from mmprism.data.wavellm_mesh_fk_l2_dataset import (
            MeshFkL2DatasetConfig,
            MeshFkL2EpisodeDataset,
        )

        handoff_path = Path(handoff_root).expanduser().resolve()
        l2_path = Path(l2_root).expanduser().resolve()
        dataset_config = MeshFkL2DatasetConfig(
            handoff_root=handoff_path,
            l2_root=l2_path,
            split="train",
            max_frames=config.data.max_frames,
            train_augmentation=config.data.train_augmentation,
            legacy_sequence_format=config.data.legacy_sequence_format,
            pose_frame=config.data.pose_frame,
            pose_coordinate_frame=config.data.pose_coordinate_frame,
            pose_normalization_profile=(
                Path(config.data.pose_normalization_profile)
                if config.data.pose_normalization_profile is not None
                else None
            ),
            full24_normalization_profile=(
                Path(config.data.full24_normalization_profile)
                if config.data.full24_normalization_profile is not None
                else None
            ),
            hybrid24_normalization_profile=(
                Path(config.data.hybrid24_normalization_profile)
                if config.data.hybrid24_normalization_profile is not None
                else None
            ),
            hybrid24v2_normalization_profile=(
                Path(config.data.hybrid24v2_normalization_profile)
                if config.data.hybrid24v2_normalization_profile is not None
                else None
            ),
            unisign_hand_bridge_normalization_profile=(
                Path(config.data.unisign_hand_bridge_normalization_profile)
                if config.data.unisign_hand_bridge_normalization_profile is not None
                else None
            ),
            full24_temporal_sampling_profile=config.data.full24_temporal_sampling_profile,
            full24_temporal_train_phase=config.data.full24_temporal_train_phase,
            split_assignments_path=(
                Path(config.data.split_assignments_path)
                if config.data.split_assignments_path is not None
                else None
            ),
        )
        train = MeshFkL2EpisodeDataset(dataset_config)
        validation = MeshFkL2EpisodeDataset(replace(dataset_config, split="validation"))
        identity = dict(train.checkpoint_identity())
        return cls(
            train,
            validation,
            config,
            seed=seed,
            source_identity=identity,
        )

    @classmethod
    def from_translation_parquet_delivery(
        cls,
        delivery_root: str | Path,
        config: WaveLLMRunConfig,
        *,
        seed: int,
        split_view_root: str | Path | None = None,
    ) -> WaveLLMDataModule:
        """Build train/validation readers from one canonical episode delivery.

        Without ``split_view_root`` the delivery's embedded split remains the
        compatibility default.  An external view changes only semantic role
        membership; it resolves every selected sample through the canonical
        delivery index and never copies Parquet payloads.
        """

        root = Path(delivery_root).expanduser().resolve()
        try:
            validation = load_published_parquet_delivery_validation(root)
            split_view = (
                load_csl_daily_contextual_pf_external_split_view(split_view_root)
                if split_view_root is not None
                else None
            )
            if split_view is None:
                train = ParquetSignLanguageTranslationDataset(
                    root,
                    split="train",
                    verify_checksums=config.data.verify_checksums,
                    accepted_validation=validation,
                    consumer_input_mode=config.input_mode,
                )
                held = ParquetSignLanguageTranslationDataset(
                    root,
                    split="validation",
                    verify_checksums=config.data.verify_checksums,
                    accepted_validation=validation,
                    consumer_input_mode=config.input_mode,
                )
            else:
                train = ParquetSignLanguageTranslationDataset(
                    root,
                    indexed_samples=resolve_csl_daily_contextual_pf_parquet_references(
                        root,
                        split_view,
                        role="train",
                    ),
                    verify_checksums=config.data.verify_checksums,
                    accepted_validation=validation,
                    consumer_input_mode=config.input_mode,
                )
                held = ParquetSignLanguageTranslationDataset(
                    root,
                    indexed_samples=resolve_csl_daily_contextual_pf_parquet_references(
                        root,
                        split_view,
                        role="validation",
                    ),
                    verify_checksums=config.data.verify_checksums,
                    accepted_validation=validation,
                    consumer_input_mode=config.input_mode,
                )
        except (CslDailyContextualPFSplitViewError, ParquetDeliveryError) as error:
            raise WaveLLMLightningError(
                f"unable to open WaveLLM Parquet delivery: {error}"
            ) from error
        contextual_contract = contextual_pf_input_contract_observation(
            root,
            config,
            split_view=split_view,
        )
        return cls(
            train,
            held,
            config,
            seed=seed,
            source_identity=translation_parquet_source_identity(
                root,
                validation,
                input_mode=train.input_mode,
                split_view=split_view,
                contextual_input_contract=contextual_contract,
            ),
        )

    @classmethod
    def from_ce_frozen_daily_predpose_delivery(
        cls,
        delivery_root: str | Path,
        config: WaveLLMRunConfig,
        *,
        seed: int,
    ) -> WaveLLMDataModule:
        """Build the dedicated frozen-CE PredPose train/validation view.

        This compact delivery owns its explicit sample/range and train-only
        Yoke normalization contract.  It is intentionally not routed through
        the legacy fourfold or generic Parquet producer identities.
        """

        from mmprism.data.ce_frozen_daily_predpose import (
            CeFrozenDailyPredPoseDataset,
            CeFrozenDailyPredPoseError,
        )

        try:
            selected_head = config.data.ce_frozen_predpose_head
            if selected_head is None:
                raise CeFrozenDailyPredPoseError(
                    "CE-frozen PredPose data requires an explicit selected pose head"
                )
            train = CeFrozenDailyPredPoseDataset(
                delivery_root,
                split="train",
                selected_head=selected_head,
                normalization_profile_path=config.data.ce_frozen_predpose_normalization_profile,
            )
            validation = CeFrozenDailyPredPoseDataset(
                delivery_root,
                split="validation",
                selected_head=selected_head,
                normalization_profile_path=config.data.ce_frozen_predpose_normalization_profile,
            )
        except CeFrozenDailyPredPoseError as error:
            raise WaveLLMLightningError(
                f"unable to open CE-frozen PredPose delivery: {error}"
            ) from error
        identity = train.checkpoint_identity()
        if validation.checkpoint_identity() != identity:
            raise WaveLLMLightningError("PredPose train and validation identities differ")
        return cls(train, validation, config, seed=seed, source_identity=identity)

    @classmethod
    def from_ce_frozen_daily_gtpose_delivery(
        cls,
        delivery_root: str | Path,
        config: WaveLLMRunConfig,
        *,
        seed: int,
    ) -> WaveLLMDataModule:
        """Build the strict GT paired-control train/validation view.

        The GT delivery owns its explicit target/range identity.  Its reader
        applies GT-derived Yoke geometry while binding the Direct PredPose
        baseline's frozen affine profile through the task configuration.
        """

        from mmprism.data.ce_frozen_daily_gtpose import (
            CeDailyGtPoseStyleJointAugmentationConfig,
            CeDailyGtPoseStyleJointTrainingDataset,
            CeFrozenDailyGtPoseDataset,
            CeFrozenDailyGtPoseError,
        )

        try:
            if config.data.ce_frozen_predpose_head != "direct":
                raise CeFrozenDailyGtPoseError(
                    "CE-frozen GT paired control requires the Direct baseline head"
                )
            profile = config.data.ce_frozen_predpose_normalization_profile
            train = CeFrozenDailyGtPoseDataset(
                delivery_root,
                split="train",
                normalization_profile_path=profile,
            )
            validation = CeFrozenDailyGtPoseDataset(
                delivery_root,
                split="validation",
                normalization_profile_path=profile,
            )
            if isinstance(
                config.data.train_augmentation,
                CeDailyGtPoseStyleJointAugmentationConfig,
            ):
                train = CeDailyGtPoseStyleJointTrainingDataset(
                    train,
                    config.data.train_augmentation,
                )
        except CeFrozenDailyGtPoseError as error:
            raise WaveLLMLightningError(
                f"unable to open CE-frozen GT paired delivery: {error}"
            ) from error
        validation_identity = validation.checkpoint_identity()
        identity = train.checkpoint_identity()
        base_identity = dict(identity)
        base_identity.pop("train_augmentation", None)
        if validation_identity != base_identity:
            raise WaveLLMLightningError("GT train and validation identities differ")
        return cls(train, validation, config, seed=seed, source_identity=base_identity)

    @property
    def train_sample_count(self) -> int:
        return len(self._train_dataset)

    @property
    def validation_sample_count(self) -> int:
        return len(self._validation_dataset)

    def _loader(
        self,
        dataset: Dataset[SignLanguageTranslationSample],
        *,
        shuffle: bool,
    ) -> DataLoader[SignLanguageTranslationBatch]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            collate_fn=partial(
                collate_sign_language_translation_samples,
                max_frames=self.config.data.max_frames,
            ),
            generator=generator,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.data.num_workers > 0,
        )
        return cast(DataLoader[SignLanguageTranslationBatch], loader)

    def train_dataloader(self) -> DataLoader[SignLanguageTranslationBatch]:
        return self._loader(self._train_dataset, shuffle=self.config.data.shuffle)

    def val_dataloader(self) -> DataLoader[SignLanguageTranslationBatch]:
        return self._loader(self._validation_dataset, shuffle=False)

    def set_train_epoch(self, epoch: int) -> None:
        setter = getattr(self.train_source, "set_epoch", None)
        if callable(setter):
            setter(epoch)


@dataclass(frozen=True, slots=True)
class _TensorBatch:
    sample_ids: tuple[str, ...]
    pose: Tensor
    pose_confidence: Tensor
    radar_features: Tensor | None
    frame_attention_mask: Tensor
    captions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ConditionalBatchStats:
    generation_loss: Tensor
    pair_loss: Tensor | None
    positive_score_sum: Tensor
    negative_score_sum: Tensor
    margin_sum: Tensor
    pair_count: int
    length_difference_sum: int


def _tensor_batch(
    batch: SignLanguageTranslationBatch,
    device: torch.device,
    *,
    input_mode: str,
) -> _TensorBatch:
    non_blocking = device.type == "cuda"
    return _TensorBatch(
        sample_ids=batch.sample_ids,
        pose=torch.from_numpy(batch.pose).to(device, non_blocking=non_blocking),
        pose_confidence=torch.from_numpy(batch.pose_confidence).to(
            device, non_blocking=non_blocking
        ),
        radar_features=(
            torch.from_numpy(batch.radar_feature).to(device, non_blocking=non_blocking)
            if wavellm_input_mode_uses_radar_feature(input_mode) and batch.radar_feature is not None
            else None
        ),
        frame_attention_mask=torch.from_numpy(batch.frame_mask).to(
            device=device, dtype=torch.long, non_blocking=non_blocking
        ),
        captions=batch.captions,
    )


def tensor_batch(
    batch: SignLanguageTranslationBatch,
    device: torch.device,
    *,
    input_mode: str,
) -> _TensorBatch:
    """Transfer a translation batch for supported evaluation and annotation consumers."""

    return _tensor_batch(batch, device, input_mode=input_mode)


def _token_tensor(payload: Mapping[str, Tensor], key: str, device: torch.device) -> Tensor:
    tensor = payload.get(key)
    if not isinstance(tensor, Tensor):
        raise WaveLLMLightningError(f"tokenizer output is missing tensor {key}")
    return tensor.to(device)


def warmup_cosine_multiplier(
    optimizer_step: int,
    *,
    total_optimizer_steps: int,
    warmup_optimizer_steps: int = WARMUP_OPTIMIZER_STEPS,
) -> float:
    if optimizer_step < 0:
        raise WaveLLMLightningError("optimizer_step must be non-negative")
    if total_optimizer_steps < 1:
        raise WaveLLMLightningError("total_optimizer_steps must be positive")
    if warmup_optimizer_steps < 0:
        raise WaveLLMLightningError("warmup_optimizer_steps must be non-negative")
    if warmup_optimizer_steps > 0 and optimizer_step < warmup_optimizer_steps:
        return optimizer_step / warmup_optimizer_steps
    decay_steps = max(1, total_optimizer_steps - warmup_optimizer_steps)
    progress = min(
        1.0,
        max(0.0, (optimizer_step - warmup_optimizer_steps) / decay_steps),
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _distributed_totals(
    loss_sum: Tensor, sample_count: Tensor, elapsed_seconds: float
) -> tuple[float, int, float]:
    totals = torch.stack(
        (
            loss_sum.detach().to(dtype=torch.float64),
            sample_count.detach().to(dtype=torch.float64),
            torch.tensor(elapsed_seconds, device=loss_sum.device, dtype=torch.float64),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals[:2], op=dist.ReduceOp.SUM)
        dist.all_reduce(totals[2], op=dist.ReduceOp.MAX)
    return float(totals[0].item()), int(totals[1].item()), float(totals[2].item())


def _gather_objects(value: Mapping[str, object]) -> list[dict[str, object]]:
    if not dist.is_available() or not dist.is_initialized():
        return [dict(value)]
    gathered: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(value))
    if any(not isinstance(row, dict) for row in gathered):
        raise WaveLLMLightningError("distributed runtime summary is invalid")
    return [cast(dict[str, object], row) for row in gathered]


class WaveLLMLightningModule(L.LightningModule):
    """Ordinary FP32 Lightning wrapper around the maintained GeometryGuidedMT5."""

    def __init__(
        self,
        model: GeometryGuidedMT5,
        tokenizer: TokenizerLike,
        config: WaveLLMRunConfig,
        *,
        source_identity: Mapping[str, object] | None = None,
        initialization_evidence: Mapping[str, object] | None = None,
        fusion_training_evidence: Mapping[str, object] | None = None,
        train_group_ids: Mapping[str, str] | None = None,
        warmup_optimizer_steps: int = WARMUP_OPTIMIZER_STEPS,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.train()
        if config.model.freeze_language_model:
            self.model.language_model.eval()
        self.tokenizer = tokenizer
        self.config = config
        if warmup_optimizer_steps < 0:
            raise WaveLLMLightningError("warmup_optimizer_steps must be non-negative")
        self.warmup_optimizer_steps = warmup_optimizer_steps
        pose_encoder_profile = getattr(config, "pose_encoder_profile", None)
        try:
            self.pose_encoder_descriptor = (
                describe_wavellm_pose_encoder(
                    pose_encoder_profile,
                    self.model.pose_encoder,
                ).to_dict()
                if pose_encoder_profile is not None
                else None
            )
        except WaveLLMModelFactoryError as error:
            raise WaveLLMLightningError(str(error)) from error
        self.source_identity = dict(source_identity or {})
        self.initialization_evidence = (
            dict(initialization_evidence) if initialization_evidence is not None else None
        )
        self.fusion_training_evidence = (
            dict(fusion_training_evidence) if fusion_training_evidence is not None else None
        )
        self.train_group_ids = dict(train_group_ids or {})
        self._enforce_frozen_backbone_mode()
        self.total_optimizer_steps = 0
        self.optimizer_step_count = 0
        self.history: list[dict[str, object]] = []
        self.generation_summaries: list[dict[str, object]] = []
        self.resource_summaries: list[dict[str, object]] = []
        self._train_loss_sum: Tensor | None = None
        self._train_sample_count: Tensor | None = None
        self._validation_loss_sum: Tensor | None = None
        self._validation_sample_count: Tensor | None = None
        self._train_started = 0.0
        self._validation_started = 0.0
        self._fit_started = 0.0
        self._generation_batch: _TensorBatch | None = None
        self._train_finalized_epoch: int | None = None
        self._last_train_batch_end: float | None = None
        self._loader_wait_seconds = 0.0
        self._loader_wait_batches = 0
        self._gpu_utilization_sum = 0
        self._gpu_utilization_max = 0
        self._gpu_utilization_samples = 0
        self._progress_output_root: Path | None = None
        self._optimizer_group_summary: list[dict[str, object]] = []
        self._conditional_epoch_sums: Tensor | None = None

    def configure_progress_output(self, output_root: str | Path) -> None:
        """Configure the lifecycle directory used for live rank-zero status."""

        self._progress_output_root = Path(output_root).expanduser().resolve()

    def _enforce_frozen_backbone_mode(self) -> None:
        if self.fusion_training_evidence is None:
            return
        if self.fusion_training_evidence.get("freeze_inherited_backbone") is not True:
            return
        self.model.language_model.eval()
        self.model.pose_encoder.eval()

    def forward(self, batch: SignLanguageTranslationBatch) -> Tensor:
        tensor_batch = _tensor_batch(batch, self.device, input_mode=self.config.input_mode)
        return self._forward_loss(tensor_batch)

    def transfer_batch_to_device(
        self,
        batch: SignLanguageTranslationBatch,
        device: torch.device,
        dataloader_idx: int,
    ) -> SignLanguageTranslationBatch:
        del device, dataloader_idx
        # The public batch is a frozen NumPy dataclass; _tensor_batch owns transfer.
        return batch

    def _tokens(self, texts: Sequence[str], *, max_length: int) -> Mapping[str, Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def _forward_output(
        self, batch: _TensorBatch
    ) -> tuple[TranslationOutput, Tensor, ModalityEncoding, Tensor, Tensor]:
        prompt_tokens = self._tokens(
            [self.config.data.prompt] * len(batch.sample_ids),
            max_length=self.config.data.max_prompt_length,
        )
        prompt = _token_tensor(
            prompt_tokens,
            "input_ids",
            batch.pose.device,
        )
        prompt_mask = _token_tensor(
            prompt_tokens,
            "attention_mask",
            batch.pose.device,
        )
        target_tokens = self._tokens(batch.captions, max_length=self.config.data.max_target_length)
        labels = _token_tensor(target_tokens, "input_ids", batch.pose.device)
        pad_token_id = self.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            raise WaveLLMLightningError("tokenizer pad_token_id must be an integer")
        labels = labels.masked_fill(labels == pad_token_id, -100)
        if not bool(torch.any(labels != -100)):
            raise WaveLLMLightningError("target tokenization produced no supervised tokens")
        if (
            wavellm_input_mode_uses_radar_feature(self.config.input_mode)
            and batch.radar_features is None
        ):
            raise WaveLLMLightningError("fusion batch is missing radar features")
        modalities = self.model.encode_modalities(
            batch.pose,
            batch.pose_confidence,
            batch.radar_features,
            batch.frame_attention_mask,
        )
        output = self.model.forward_from_modalities(
            modalities=modalities,
            prompt_input_ids=prompt,
            prompt_attention_mask=prompt_mask,
            labels=labels,
        )
        # Stash for subclass distillation losses (read-only; no behavior change).
        self._last_forward_output = output
        self._last_forward_labels = labels
        self._last_forward_modalities = modalities
        self._last_forward_prompt = prompt
        self._last_forward_prompt_mask = prompt_mask
        return output, labels, modalities, prompt, prompt_mask

    @staticmethod
    def _negative_modalities(
        modalities: ModalityEncoding, negative_indices: Tensor
    ) -> ModalityEncoding:
        return ModalityEncoding(
            embeddings=modalities.embeddings.index_select(0, negative_indices),
            attention_mask=modalities.attention_mask.index_select(0, negative_indices),
            pose_gate=modalities.pose_gate.index_select(0, negative_indices),
        )

    def _conditional_pairs(self, batch: _TensorBatch) -> ConditionalNegativePairs:
        config = self.config.conditional_discrimination
        if config is None:
            raise WaveLLMLightningError("conditional discrimination is not configured")
        try:
            group_ids = tuple(self.train_group_ids[sample_id] for sample_id in batch.sample_ids)
        except KeyError as error:
            raise WaveLLMLightningError(
                f"conditional discrimination lacks group ID for {error.args[0]}"
            ) from error
        valid_frame_counts = tuple(
            int(value) for value in batch.frame_attention_mask.detach().sum(dim=1).cpu().tolist()
        )
        return select_conditional_negative_pairs(
            sample_ids=batch.sample_ids,
            captions=batch.captions,
            group_ids=group_ids,
            valid_frame_counts=valid_frame_counts,
            tie_seed=config.tie_seed,
        )

    def _forward_training_loss(
        self, batch: _TensorBatch
    ) -> tuple[Tensor, _ConditionalBatchStats | None]:
        output, labels, modalities, prompt, prompt_mask = self._forward_output(batch)
        generation_loss = output.loss
        if not isinstance(generation_loss, Tensor) or not bool(torch.isfinite(generation_loss)):
            raise WaveLLMLightningError("WaveLLM generation loss is non-finite")
        config = self.config.conditional_discrimination
        if config is None:
            return generation_loss, None
        pairs = self._conditional_pairs(batch)
        pair_count = len(pairs.target_indices)
        zero = generation_loss.detach().new_zeros(())
        if pair_count == 0:
            return generation_loss, _ConditionalBatchStats(
                generation_loss=generation_loss.detach(),
                pair_loss=None,
                positive_score_sum=zero,
                negative_score_sum=zero,
                margin_sum=zero,
                pair_count=0,
                length_difference_sum=0,
            )
        target_indices = torch.tensor(
            pairs.target_indices, device=batch.pose.device, dtype=torch.long
        )
        negative_indices = torch.tensor(
            pairs.negative_indices, device=batch.pose.device, dtype=torch.long
        )
        paired_labels = labels.index_select(0, target_indices)
        positive_scores = sequence_log_probability(
            output.logits.index_select(0, target_indices), paired_labels
        )
        devices = (
            [batch.pose.device.index]
            if batch.pose.device.type == "cuda" and batch.pose.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=devices, enabled=True):
            negative_output = self.model.forward_from_modalities(
                modalities=self._negative_modalities(modalities, negative_indices),
                prompt_input_ids=prompt.index_select(0, target_indices),
                prompt_attention_mask=prompt_mask.index_select(0, target_indices),
                labels=paired_labels,
            )
        negative_scores = sequence_log_probability(negative_output.logits, paired_labels)
        pair_loss = conditional_pair_loss(positive_scores, negative_scores)
        total_loss = generation_loss + config.coefficient * pair_loss
        if not bool(torch.isfinite(pair_loss)) or not bool(torch.isfinite(total_loss)):
            raise WaveLLMLightningError("conditional discrimination loss is non-finite")
        return total_loss, _ConditionalBatchStats(
            generation_loss=generation_loss.detach(),
            pair_loss=pair_loss.detach(),
            positive_score_sum=positive_scores.detach().sum(),
            negative_score_sum=negative_scores.detach().sum(),
            margin_sum=(positive_scores.detach() - negative_scores.detach()).sum(),
            pair_count=pair_count,
            length_difference_sum=sum(pairs.length_differences),
        )

    def _forward_loss(self, batch: _TensorBatch) -> Tensor:
        output, _labels, _modalities, _prompt, _prompt_mask = self._forward_output(batch)
        loss = output.loss
        if not isinstance(loss, Tensor):
            raise WaveLLMLightningError("WaveLLM output is missing tensor loss")
        if not bool(torch.isfinite(loss)):
            raise WaveLLMLightningError("WaveLLM Lightning loss is non-finite")
        return loss

    def _step(
        self, batch: SignLanguageTranslationBatch, *, split: str
    ) -> tuple[Tensor, _TensorBatch]:
        tensor_batch = _tensor_batch(batch, self.device, input_mode=self.config.input_mode)
        loss = self._forward_loss(tensor_batch)
        batch_size = len(tensor_batch.sample_ids)
        self.log(
            f"{split}_loss",
            loss,
            on_step=split == "train",
            on_epoch=True,
            prog_bar=False,
            logger=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss, tensor_batch

    def training_step(self, batch: SignLanguageTranslationBatch, batch_idx: int) -> Tensor:
        del batch_idx
        tensor_batch = _tensor_batch(batch, self.device, input_mode=self.config.input_mode)
        loss, conditional = self._forward_training_loss(tensor_batch)
        count = len(tensor_batch.sample_ids)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=False,
            sync_dist=True,
            batch_size=count,
        )
        assert self._train_loss_sum is not None and self._train_sample_count is not None
        self._train_loss_sum += loss.detach().double() * count
        self._train_sample_count += count
        if conditional is not None:
            assert self._conditional_epoch_sums is not None
            pair_count = conditional.pair_count
            pair_loss_sum = (
                conditional.pair_loss.double() * pair_count
                if conditional.pair_loss is not None
                else loss.detach().new_zeros((), dtype=torch.float64)
            )
            self._conditional_epoch_sums += torch.stack(
                (
                    conditional.generation_loss.double() * count,
                    pair_loss_sum,
                    pair_loss_sum * self.config.conditional_discrimination.coefficient,
                    conditional.positive_score_sum.double(),
                    conditional.negative_score_sum.double(),
                    conditional.margin_sum.double(),
                    loss.detach().new_tensor(
                        conditional.length_difference_sum, dtype=torch.float64
                    ),
                    loss.detach().new_tensor(pair_count, dtype=torch.float64),
                    loss.detach().new_tensor(count, dtype=torch.float64),
                )
            )
        return loss

    def validation_step(self, batch: SignLanguageTranslationBatch, batch_idx: int) -> Tensor:
        loss, tensor_batch = self._step(batch, split="validation")
        assert self._validation_loss_sum is not None and self._validation_sample_count is not None
        count = len(tensor_batch.sample_ids)
        self._validation_loss_sum += loss.detach().double() * count
        self._validation_sample_count += count
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._generation_batch = tensor_batch
        return loss

    def _optimizer_parameter_groups(
        self,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Partition the official language module from the fresh pose branch."""

        optimization = self.config.optimization
        language_parameters = [
            parameter
            for parameter in self.model.language_model.parameters()
            if parameter.requires_grad
        ]
        language_ids = {id(parameter) for parameter in language_parameters}
        pose_branch_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in language_ids
        ]
        all_trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if len({id(parameter) for parameter in all_trainable}) != len(all_trainable):
            raise WaveLLMLightningError("WaveLLM model exposes duplicate trainable parameters")
        if optimization.uses_dual_learning_rates:
            if not language_parameters or not pose_branch_parameters:
                raise WaveLLMLightningError(
                    "dual learning rates require trainable language and fresh pose "
                    "branch parameters"
                )
            assert optimization.language_model_learning_rate is not None
            assert optimization.pose_branch_learning_rate is not None
            groups = [
                {
                    "params": pose_branch_parameters,
                    "lr": optimization.pose_branch_learning_rate,
                },
                {
                    "params": language_parameters,
                    "lr": optimization.language_model_learning_rate,
                },
            ]
            summary = [
                {
                    "name": "fresh_pose_branch_and_connector",
                    "learning_rate": optimization.pose_branch_learning_rate,
                    "parameter_count": sum(
                        parameter.numel() for parameter in pose_branch_parameters
                    ),
                },
                {
                    "name": "official_language_model",
                    "learning_rate": optimization.language_model_learning_rate,
                    "parameter_count": sum(parameter.numel() for parameter in language_parameters),
                },
            ]
            return groups, summary
        if not all_trainable:
            raise WaveLLMLightningError("WaveLLM model has no trainable parameters")
        return [
            {
                "params": all_trainable,
                "lr": optimization.learning_rate,
            }
        ], [
            {
                "name": "all_trainable_parameters",
                "learning_rate": optimization.learning_rate,
                "parameter_count": sum(parameter.numel() for parameter in all_trainable),
            }
        ]

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps < 1:
            raise WaveLLMLightningError("Lightning resolved no optimizer steps")
        self.total_optimizer_steps = total_steps
        parameter_groups, group_summary = self._optimizer_parameter_groups()
        self._optimizer_group_summary = group_summary
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
            betas=(self.config.optimization.beta1, self.config.optimization.beta2),
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_multiplier(
                step,
                total_optimizer_steps=total_steps,
                warmup_optimizer_steps=self.warmup_optimizer_steps,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine_learning_rate",
            },
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        self.optimizer_step_count += 1

    def on_train_start(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._fit_started = time.perf_counter()
        self._loader_wait_seconds = 0.0
        self._loader_wait_batches = 0
        self._gpu_utilization_sum = 0
        self._gpu_utilization_max = 0
        self._gpu_utilization_samples = 0
        self._publish_heartbeat(status="running")

    def on_train_epoch_start(self) -> None:
        self._enforce_frozen_backbone_mode()
        data_module = getattr(self.trainer, "datamodule", None)
        set_epoch = getattr(data_module, "set_train_epoch", None)
        if callable(set_epoch):
            set_epoch(int(self.current_epoch))
        self._train_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._train_sample_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self._train_started = time.perf_counter()
        self._train_finalized_epoch = None
        self._conditional_epoch_sums = (
            torch.zeros(9, device=self.device, dtype=torch.float64)
            if self.config.conditional_discrimination is not None
            else None
        )
        self._last_train_batch_end = None

    def on_train_batch_start(
        self,
        batch: SignLanguageTranslationBatch,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        now = time.perf_counter()
        previous = (
            self._train_started
            if self._last_train_batch_end is None
            else self._last_train_batch_end
        )
        self._loader_wait_seconds += max(0.0, now - previous)
        self._loader_wait_batches += 1

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, object] | None,
        batch: SignLanguageTranslationBatch,
        batch_idx: int,
    ) -> None:
        del outputs, batch
        self._last_train_batch_end = time.perf_counter()
        if self.device.type == "cuda" and batch_idx % HEARTBEAT_BATCH_INTERVAL == 0:
            utilization_reader = getattr(torch.cuda, "utilization", None)
            if callable(utilization_reader):
                utilization = int(utilization_reader(self.device))
                self._gpu_utilization_sum += utilization
                self._gpu_utilization_max = max(self._gpu_utilization_max, utilization)
                self._gpu_utilization_samples += 1
        if batch_idx == 0 or (batch_idx + 1) % HEARTBEAT_BATCH_INTERVAL == 0:
            self._publish_heartbeat(status="running")

    def _finalize_train_epoch(self) -> None:
        if self._train_finalized_epoch == int(self.current_epoch):
            return
        assert self._train_loss_sum is not None and self._train_sample_count is not None
        loss_sum, count, seconds = _distributed_totals(
            self._train_loss_sum,
            self._train_sample_count,
            time.perf_counter() - self._train_started,
        )
        if count < 1 or not math.isfinite(loss_sum):
            raise WaveLLMLightningError("WaveLLM training epoch summary is non-finite")
        self.history.append(
            {
                "epoch": int(self.current_epoch),
                "train_loss": loss_sum / count,
                "train_sample_count": count,
                "train_seconds": seconds,
                "train_samples_per_second": count / seconds,
            }
        )
        if self._conditional_epoch_sums is not None:
            totals = self._conditional_epoch_sums.detach().clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            pair_count = float(totals[7].item())
            sample_count = float(totals[8].item())
            coefficient = self.config.conditional_discrimination.coefficient
            self.history[-1]["conditional_discrimination"] = {
                "generation_loss": float(totals[0].item() / sample_count),
                "pair_loss": (float(totals[1].item() / pair_count) if pair_count > 0 else None),
                "weighted_contribution": (
                    float(totals[2].item() / pair_count) if pair_count > 0 else 0.0
                ),
                "coefficient": coefficient,
                "paired_fraction": pair_count / sample_count,
                "pair_count": int(pair_count),
                "mean_length_difference_frames": (
                    float(totals[6].item() / pair_count) if pair_count > 0 else None
                ),
                "positive_score": (
                    float(totals[3].item() / pair_count) if pair_count > 0 else None
                ),
                "negative_score": (
                    float(totals[4].item() / pair_count) if pair_count > 0 else None
                ),
                "positive_minus_negative_margin": (
                    float(totals[5].item() / pair_count) if pair_count > 0 else None
                ),
            }
        self._train_finalized_epoch = int(self.current_epoch)

    def on_train_epoch_end(self) -> None:
        self._finalize_train_epoch()

    def on_validation_epoch_start(self) -> None:
        self._finalize_train_epoch()
        self._validation_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._validation_sample_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self._validation_started = time.perf_counter()
        self._generation_batch = None

    def on_validation_epoch_end(self) -> None:
        assert self._validation_loss_sum is not None and self._validation_sample_count is not None
        loss_sum, count, seconds = _distributed_totals(
            self._validation_loss_sum,
            self._validation_sample_count,
            time.perf_counter() - self._validation_started,
        )
        if count < 1 or not math.isfinite(loss_sum):
            raise WaveLLMLightningError("WaveLLM validation epoch summary is non-finite")
        if not self.history or self.history[-1]["epoch"] != int(self.current_epoch):
            raise WaveLLMLightningError("WaveLLM train/validation epoch history drift")
        self.history[-1].update(
            {
                "validation_loss": loss_sum / count,
                "validation_sample_count": count,
                "validation_seconds": seconds,
                "validation_samples_per_second": count / seconds,
            }
        )
        if self.trainer.is_global_zero:
            self._generate_summary()
            self._publish_progress()
            self._publish_heartbeat(status="running")

    def _generate_summary(self) -> None:
        batch = self._generation_batch
        if batch is None:
            raise WaveLLMLightningError("validation produced no generation batch")
        limit = min(2, len(batch.sample_ids))
        prompt_tokens = self._tokens(
            [self.config.data.prompt] * limit,
            max_length=self.config.data.max_prompt_length,
        )
        prompt_ids = _token_tensor(prompt_tokens, "input_ids", batch.pose.device)
        prompt_mask = _token_tensor(prompt_tokens, "attention_mask", batch.pose.device)
        with torch.inference_mode():
            token_ids = self.model.generate(
                pose=batch.pose[:limit],
                pose_confidence=batch.pose_confidence[:limit],
                frame_attention_mask=batch.frame_attention_mask[:limit],
                prompt_input_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                max_new_tokens=self.config.generation.max_new_tokens,
                num_beams=self.config.generation.num_beams,
                **(
                    {"radar_features": batch.radar_features[:limit]}
                    if wavellm_input_mode_uses_radar_feature(self.config.input_mode)
                    and batch.radar_features is not None
                    else {}
                ),
            )
        if token_ids.ndim != 2 or token_ids.shape[0] != limit:
            raise WaveLLMLightningError("WaveLLM generation returned an invalid shape")
        decoded = tuple(
            text.strip()
            for text in self.tokenizer.batch_decode(
                token_ids.detach().cpu(), skip_special_tokens=True
            )
        )
        if len(decoded) != limit:
            raise WaveLLMLightningError("WaveLLM generation count mismatch")
        self.generation_summaries.append(
            {
                "epoch": int(self.current_epoch),
                "sample_ids": list(batch.sample_ids[:limit]),
                "predictions": list(decoded),
                **(
                    {"references": list(batch.captions[:limit])}
                    if self.config.evaluation.save_references
                    else {}
                ),
            }
        )

    def on_train_end(self) -> None:
        fit_seconds = time.perf_counter() - self._fit_started
        utilization_mean: float | None = None
        utilization_max: int | None = None
        peak_allocated = 0
        peak_reserved = 0
        if self.device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
            if self._gpu_utilization_samples > 0:
                utilization_mean = self._gpu_utilization_sum / self._gpu_utilization_samples
                utilization_max = self._gpu_utilization_max
        rows = _gather_objects(
            {
                "rank": int(self.global_rank),
                "device": str(self.device),
                "fit_seconds": fit_seconds,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "gpu_utilization_mean_percent": utilization_mean,
                "gpu_utilization_max_percent": utilization_max,
                "gpu_utilization_sample_count": self._gpu_utilization_samples,
                "loader_wait_seconds": self._loader_wait_seconds,
                "loader_wait_batch_count": self._loader_wait_batches,
                "loader_wait_fraction": (
                    self._loader_wait_seconds / fit_seconds if fit_seconds > 0 else None
                ),
            }
        )
        if self.trainer.is_global_zero:
            self.resource_summaries = rows
            self._publish_heartbeat(status="completed")

    def _history_metric(self, key: str) -> float | None:
        for row in reversed(self.history):
            value = row.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                return float(value)
        return None

    def _running_train_metrics(self) -> tuple[float | None, float | None]:
        if (
            self._train_finalized_epoch is not None
            or self._train_loss_sum is None
            or self._train_sample_count is None
        ):
            return None, None
        count = float(self._train_sample_count.detach().item())
        loss_sum = float(self._train_loss_sum.detach().item())
        seconds = time.perf_counter() - self._train_started
        if count <= 0 or seconds <= 0 or not math.isfinite(loss_sum):
            return None, None
        world_size = max(1, int(self.trainer.world_size))
        return loss_sum / count, count * world_size / seconds

    def _recent_generation_samples(self) -> list[dict[str, object]]:
        if not self.generation_summaries:
            return []
        latest = self.generation_summaries[-1]
        epoch = latest.get("epoch")
        sample_ids = latest.get("sample_ids")
        predictions = latest.get("predictions")
        references = latest.get("references")
        if not isinstance(sample_ids, list) or not isinstance(predictions, list):
            return []
        reference_rows = references if isinstance(references, list) else []
        return [
            {
                "epoch": epoch if isinstance(epoch, int) else int(self.current_epoch),
                "sample_id": sample_id,
                "prediction": predictions[index],
                "reference": (reference_rows[index] if index < len(reference_rows) else None),
            }
            for index, sample_id in enumerate(sample_ids)
            if index < len(predictions)
        ]

    def _memory_snapshot(self) -> dict[str, object]:
        peak_allocated = 0
        peak_reserved = 0
        if self.device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        return {
            "device": str(self.device),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }

    def _monitor_snapshot(self, *, schema_version: str, status: str) -> dict[str, object]:
        running_train_loss, running_train_throughput = self._running_train_metrics()
        train_loss = (
            running_train_loss
            if running_train_loss is not None
            else self._history_metric("train_loss")
        )
        train_throughput = (
            running_train_throughput
            if running_train_throughput is not None
            else self._history_metric("train_samples_per_second")
        )
        fit_seconds = time.perf_counter() - self._fit_started
        return {
            "schema_version": schema_version,
            "status": status,
            "updated_at_unix_seconds": time.time(),
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "loss": {
                "train": train_loss,
                "validation": self._history_metric("validation_loss"),
            },
            "throughput": {
                "train_samples_per_second": train_throughput,
                "validation_samples_per_second": self._history_metric(
                    "validation_samples_per_second"
                ),
            },
            "loader_wait": {
                "seconds": self._loader_wait_seconds,
                "batch_count": self._loader_wait_batches,
                "fraction": (self._loader_wait_seconds / fit_seconds if fit_seconds > 0 else None),
            },
            "memory": self._memory_snapshot(),
            "recent_generation_samples": self._recent_generation_samples(),
        }

    def _publish_heartbeat(self, *, status: str) -> None:
        root = self._progress_output_root
        if root is None or not self.trainer.is_global_zero:
            return
        _atomic_json(
            root / "heartbeat.json",
            self._monitor_snapshot(
                schema_version=WAVELLM_LIGHTNING_HEARTBEAT_SCHEMA,
                status=status,
            ),
        )

    def _publish_progress(self) -> None:
        root = self._progress_output_root
        if root is None or not self.trainer.is_global_zero:
            return
        _atomic_json(
            root / "progress.json",
            self._monitor_snapshot(
                schema_version=WAVELLM_LIGHTNING_PROGRESS_SCHEMA,
                status="validation_complete",
            ),
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        binding: dict[str, object] = {
            "schema_version": WAVELLM_LIGHTNING_CHECKPOINT_SCHEMA,
            "config_fingerprint": self.config.fingerprint,
            "model_fingerprint": self.config.model_fingerprint,
            "training_fingerprint": self.config.training_fingerprint,
            "input_mode": self.config.input_mode,
            "source_identity": self.source_identity,
            "warmup_optimizer_steps": self.warmup_optimizer_steps,
            "total_optimizer_steps": self.total_optimizer_steps,
            "optimizer_step_count": self.optimizer_step_count,
            "history": self.history,
            "generation": self.generation_summaries,
        }
        if self.pose_encoder_descriptor is not None:
            binding["pose_encoder_descriptor"] = self.pose_encoder_descriptor
        if self.config.data.train_augmentation is not None:
            binding["train_augmentation"] = self.config.data.train_augmentation.to_dict()
        if self.config.conditional_discrimination is not None:
            binding["conditional_discrimination"] = self.config.conditional_discrimination.to_dict()
        if self.config.fusion_variant != WAVELLM_FUSION_COMPACT_CONFIDENCE_V1:
            binding["fusion_variant"] = self.config.fusion_variant
        if self.config.contextual_input_contract is not None:
            binding["contextual_input_contract"] = self.config.contextual_input_contract.to_dict()
        if self.initialization_evidence is not None:
            binding["initialization"] = self.initialization_evidence
        if self.fusion_training_evidence is not None:
            binding["fusion_training"] = self.fusion_training_evidence
        checkpoint["mmprism_wavellm"] = binding

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        binding = checkpoint.get("mmprism_wavellm")
        if not isinstance(binding, Mapping):
            raise WaveLLMLightningError("checkpoint lacks WaveLLM Lightning binding")
        expected = {
            "schema_version": WAVELLM_LIGHTNING_CHECKPOINT_SCHEMA,
            "config_fingerprint": self.config.fingerprint,
            "model_fingerprint": self.config.model_fingerprint,
            "training_fingerprint": self.config.training_fingerprint,
            "input_mode": self.config.input_mode,
            "source_identity": self.source_identity,
            "warmup_optimizer_steps": self.warmup_optimizer_steps,
        }
        if self.pose_encoder_descriptor is not None:
            expected["pose_encoder_descriptor"] = self.pose_encoder_descriptor
        if self.config.data.train_augmentation is not None:
            expected["train_augmentation"] = self.config.data.train_augmentation.to_dict()
        if self.config.conditional_discrimination is not None:
            expected["conditional_discrimination"] = (
                self.config.conditional_discrimination.to_dict()
            )
        if self.config.fusion_variant != WAVELLM_FUSION_COMPACT_CONFIDENCE_V1:
            expected["fusion_variant"] = self.config.fusion_variant
        if self.config.contextual_input_contract is not None:
            expected["contextual_input_contract"] = self.config.contextual_input_contract.to_dict()
        if self.initialization_evidence is not None:
            expected["initialization"] = self.initialization_evidence
        if self.fusion_training_evidence is not None:
            expected["fusion_training"] = self.fusion_training_evidence
        for key, value in expected.items():
            if binding.get(key) != value:
                raise WaveLLMLightningError(f"checkpoint WaveLLM binding mismatch for {key}")
        history = binding.get("history")
        generation = binding.get("generation")
        if isinstance(history, list):
            self.history = [dict(row) for row in history if isinstance(row, Mapping)]
        if isinstance(generation, list):
            self.generation_summaries = [
                dict(row) for row in generation if isinstance(row, Mapping)
            ]
        optimizer_step_count = binding.get("optimizer_step_count")
        if isinstance(optimizer_step_count, int) and not isinstance(optimizer_step_count, bool):
            self.optimizer_step_count = optimizer_step_count


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _bounded_val_check_interval(
    config: WaveLLMRunConfig,
    accumulate_grad_batches: int,
    device_count: int,
    training_batches: int | None,
) -> int | float:
    """Validation interval for bounded runs, clamped to one epoch.

    The interval is the run length in micro-batches (max_steps x
    accumulation) so bounded runs validate at their end, but Lightning
    rejects an interval above the per-epoch training batch count — at the
    campaign micro-batch 32 one epoch holds fewer batches than a 50-step
    preflight spans.  Without a batch count the safe fallback is epoch
    boundaries.
    """

    interval = int(config.optimization.max_steps or 0) * max(1, accumulate_grad_batches)
    if training_batches is None or training_batches < 1:
        return 1.0
    return max(1, min(interval, training_batches))


def build_wavellm_lightning_trainer(
    config: WaveLLMRunConfig,
    *,
    accelerator: str,
    devices: int | list[int],
    accumulate_grad_batches: int,
    output_root: str | Path,
    deterministic: bool = False,
    training_batches: int | None = None,
    validation_every_n_epochs: int = 1,
) -> L.Trainer:
    if accelerator not in {"cpu", "gpu"}:
        raise WaveLLMLightningError("accelerator must be cpu or gpu")
    device_count = len(devices) if not isinstance(devices, int) else devices
    if device_count < 1:
        raise WaveLLMLightningError("devices must be positive")
    if not isinstance(devices, int) and any(device < 0 for device in devices):
        raise WaveLLMLightningError("device indices must be non-negative")
    if accumulate_grad_batches < 1:
        raise WaveLLMLightningError("accumulate_grad_batches must be positive")
    if validation_every_n_epochs < 1:
        raise WaveLLMLightningError("validation_every_n_epochs must be positive")
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise WaveLLMLightningError("GPU acceleration requested but CUDA is unavailable")
    root = Path(output_root).expanduser().resolve()
    checkpoint_cadence = (
        {"every_n_epochs": validation_every_n_epochs, "save_on_train_epoch_end": False}
        if validation_every_n_epochs != 1
        else {}
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="best",
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
        **checkpoint_cadence,
    )
    latest_checkpoint = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="last",
        monitor=None,
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
        auto_insert_metric_name=False,
        **checkpoint_cadence,
    )
    strategy: str | DDPStrategy = "auto"
    if device_count > 1:
        strategy = DDPStrategy(
            process_group_backend="nccl" if accelerator == "gpu" else "gloo",
            find_unused_parameters=False,
            start_method="popen",
        )
    return L.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision="32-true",
        max_epochs=config.optimization.epochs,
        max_steps=(
            config.optimization.max_steps if config.optimization.max_steps is not None else -1
        ),
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=config.optimization.gradient_clip_norm,
        gradient_clip_algorithm="norm",
        deterministic=deterministic,
        default_root_dir=root,
        callbacks=[best_checkpoint, latest_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=(
            _bounded_val_check_interval(
                config, accumulate_grad_batches, device_count, training_batches
            )
            if config.optimization.max_steps is not None and validation_every_n_epochs == 1
            else 1.0
        ),
        check_val_every_n_epoch=validation_every_n_epochs,
        log_every_n_steps=1,
        use_distributed_sampler=True,
    )
