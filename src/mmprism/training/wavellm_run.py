from __future__ import annotations

import json
import math
import os
import random
import re
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset

from mmprism.artifacts import (
    RunArtifactWriter,
    RunInput,
    aggregate_prediction_shards,
    sha256_file,
    validate_split_bindings,
    write_prediction_shard,
)
from mmprism.assets import (
    ModelAssetSetConfig,
    ResolvedModelAsset,
    load_model_asset_config,
    resolve_model_asset,
)
from mmprism.assets.models import MODEL_ASSET_COLLECTION_NAME, MODEL_ASSET_MANIFEST_NAME
from mmprism.config import ExperimentConfig, RuntimeConfig, Task
from mmprism.data import (
    SignLanguageTranslationBatch,
    SignLanguageTranslationManifest,
    SignLanguageTranslationSample,
    collate_sign_language_translation_samples,
)
from mmprism.evaluation import LANGUAGE_METRIC_PROTOCOL, LanguageMetricAccumulator
from mmprism.models import GeometryGuidedMT5
from mmprism.runtime import build_run_plan, collect_runtime_report
from mmprism.training.distributed import (
    DistributedContext,
    DistributedRunError,
    prediction_sampler,
    set_training_sampler_epoch,
    training_sampler,
)
from mmprism.training.resume import (
    TrainingStateError,
    load_epoch_training_state,
    save_epoch_training_state,
)
from mmprism.training.wavellm_run_config import (
    WaveLLMRunConfig,
    WaveLLMRunError,
    load_wavellm_run_config,
)

WAVELLM_CHECKPOINT_SCHEMA = "mmprism.wavellm_checkpoint.v1"
WAVELLM_HISTORY_SCHEMA = "mmprism.wavellm_history.v1"
WAVELLM_PREDICTION_SCHEMA = "mmprism.translation_prediction.v1"
WAVELLM_PERFORMANCE_SCHEMA = "mmprism.wavellm_performance.v1"
WAVELLM_RUN_RESULT_SCHEMA = "mmprism.wavellm_run_result.v1"
WAVELLM_RUNTIME_SCHEMA = "mmprism.wavellm_runtime.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class _TranslationDataset(Dataset[SignLanguageTranslationSample]):
    def __init__(self, manifest: SignLanguageTranslationManifest) -> None:
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> SignLanguageTranslationSample:
        return self.manifest.load_sample(index)


@dataclass(frozen=True, slots=True)
class _TensorBatch:
    sample_ids: tuple[str, ...]
    pose: Tensor
    pose_confidence: Tensor
    radar_features: Tensor
    frame_attention_mask: Tensor
    captions: tuple[str, ...]
    coordinate_frame: str


def _require_formal_runtime(runtime_report: Mapping[str, Any], project_root: Path) -> None:
    reported_root = runtime_report.get("project_root")
    if not isinstance(reported_root, str) or Path(reported_root).resolve() != project_root:
        raise WaveLLMRunError("formal WaveLLM runs require matching project-root provenance")
    git = runtime_report.get("git")
    if (
        not isinstance(git, Mapping)
        or not isinstance(git.get("commit"), str)
        or not _GIT_COMMIT.fullmatch(git["commit"])
    ):
        raise WaveLLMRunError("formal WaveLLM runs require Git commit provenance")
    if git.get("dirty") is not False:
        raise WaveLLMRunError("formal WaveLLM runs require a clean Git worktree")


def _model_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "32-true":
        return torch.float32
    if device.type != "cuda":
        raise WaveLLMRunError(f"precision {precision} requires CUDA")
    return torch.float16 if precision == "16-mixed" else torch.bfloat16


def _seed_runtime(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


@contextmanager
def _precision_context(device: torch.device, precision: str) -> Iterator[None]:
    if device.type == "cuda" and precision in {"16-mixed", "bf16-mixed"}:
        dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _asset_identity(asset: ResolvedModelAsset) -> dict[str, object]:
    manifest_sha256 = asset.verification.get("asset_manifest_sha256")
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        raise WaveLLMRunError("resolved mT5 asset has invalid manifest provenance")
    collection_sha256 = asset.collection_manifest_sha256
    if not isinstance(collection_sha256, str) or not _SHA256.fullmatch(collection_sha256):
        raise WaveLLMRunError("resolved mT5 asset collection has invalid provenance")
    return {
        "asset_id": asset.spec.asset_id,
        "repo_id": asset.spec.repo_id,
        "revision": asset.spec.revision,
        "loader": asset.spec.loader,
        "asset_manifest_sha256": manifest_sha256,
        "collection_manifest_sha256": collection_sha256,
    }


def _load_model_and_tokenizer(
    asset: ResolvedModelAsset,
    config: WaveLLMRunConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GeometryGuidedMT5, Any]:
    if asset.spec.loader != "transformers_mt5":
        raise WaveLLMRunError(
            f"asset {asset.spec.asset_id!r} must declare loader transformers_mt5"
        )
    try:
        from transformers import AutoTokenizer, MT5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            asset.path, local_files_only=True
        )
        language_model = MT5ForConditionalGeneration.from_pretrained(
            asset.path,
            local_files_only=True,
            dtype=dtype,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise WaveLLMRunError(f"unable to load pinned mT5 asset: {error}") from error
    if language_model.config.d_model != config.model.hidden_size:
        raise WaveLLMRunError(
            "configured hidden size does not match the pinned mT5 asset: "
            f"{config.model.hidden_size} != {language_model.config.d_model}"
        )
    if tokenizer.pad_token_id is None:
        raise WaveLLMRunError("mT5 tokenizer must define pad_token_id")
    model = GeometryGuidedMT5(
        language_model,
        hidden_size=config.model.hidden_size,
        radar_feature_dim=config.model.radar_feature_dim,
        joint_count=config.model.joint_count,
        coordinate_dim=config.model.coordinate_dim,
        pose_channels=config.model.pose_channels,
        temporal_kernel_size=config.model.temporal_kernel_size,
        dropout=config.model.dropout,
        label_smoothing=config.model.label_smoothing,
    )
    if config.model.freeze_language_model:
        model.language_model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model, tokenizer


def _runtime_payload(
    model: GeometryGuidedMT5,
    runtime: RuntimeConfig,
    device: torch.device,
    asset: ResolvedModelAsset,
    distributed: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": WAVELLM_RUNTIME_SCHEMA,
        "seed": runtime.seed,
        "deterministic": runtime.deterministic,
        "precision": runtime.precision,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "parameter_count": _parameter_counts(model),
        "model_asset": _asset_identity(asset),
        "distributed": dict(distributed),
    }


def _cuda_memory_payload(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _validate_manifest_for_model(
    manifest: SignLanguageTranslationManifest,
    config: WaveLLMRunConfig,
    *,
    role: str,
) -> None:
    expected = (
        config.model.radar_feature_dim,
        config.model.joint_count,
        config.model.coordinate_dim,
    )
    observed = (
        manifest.radar_feature_dim,
        manifest.joint_count,
        manifest.coordinate_dim,
    )
    if observed != expected:
        raise WaveLLMRunError(
            f"{role} manifest feature/joint/coordinate dimensions are {observed}, "
            f"model expects {expected}"
        )
    longest = max(record.frame_count for record in manifest.records)
    if longest > config.data.max_frames:
        raise WaveLLMRunError(
            f"{role} manifest contains {longest} frames, model maximum is "
            f"{config.data.max_frames}"
        )


def _validate_split_separation(
    train_manifest: SignLanguageTranslationManifest,
    validation_manifest: SignLanguageTranslationManifest,
) -> None:
    train_sample_ids = {record.sample_id for record in train_manifest.records}
    validation_sample_ids = {record.sample_id for record in validation_manifest.records}
    sample_overlap = train_sample_ids & validation_sample_ids
    if sample_overlap:
        raise WaveLLMRunError(
            f"train and validation manifests overlap on {len(sample_overlap)} sample IDs"
        )

    train_sequence_ids = {
        record.sequence_id
        for record in train_manifest.records
        if record.sequence_id is not None
    }
    validation_sequence_ids = {
        record.sequence_id
        for record in validation_manifest.records
        if record.sequence_id is not None
    }
    sequence_overlap = train_sequence_ids & validation_sequence_ids
    if sequence_overlap:
        raise WaveLLMRunError(
            "train and validation manifests overlap on "
            f"{len(sequence_overlap)} sequence IDs"
        )


def _loader(
    manifest: SignLanguageTranslationManifest,
    config: WaveLLMRunConfig,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
    distributed: DistributedContext | None = None,
    exact_distributed_coverage: bool = False,
) -> DataLoader[SignLanguageTranslationBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = _TranslationDataset(manifest)
    sampler = None
    if distributed is not None:
        sampler = (
            prediction_sampler(dataset, distributed)
            if exact_distributed_coverage
            else training_sampler(dataset, distributed, shuffle=shuffle, seed=seed)
        )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=partial(
            collate_sign_language_translation_samples,
            max_frames=config.data.max_frames,
        ),
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
    return cast(DataLoader[SignLanguageTranslationBatch], loader)


def _tensor_batch(batch: SignLanguageTranslationBatch, device: torch.device) -> _TensorBatch:
    non_blocking = device.type == "cuda"
    return _TensorBatch(
        sample_ids=batch.sample_ids,
        pose=torch.from_numpy(batch.pose).to(device, non_blocking=non_blocking),
        pose_confidence=torch.from_numpy(batch.pose_confidence).to(
            device, non_blocking=non_blocking
        ),
        radar_features=torch.from_numpy(batch.radar_feature).to(
            device, non_blocking=non_blocking
        ),
        frame_attention_mask=torch.from_numpy(batch.frame_mask).to(
            device=device, dtype=torch.long, non_blocking=non_blocking
        ),
        captions=batch.captions,
        coordinate_frame=batch.coordinate_frame,
    )


def _token_tensor(payload: Any, key: str, device: torch.device) -> Tensor:
    if not isinstance(payload, Mapping) or not isinstance(payload.get(key), Tensor):
        raise WaveLLMRunError(f"tokenizer output is missing tensor {key}")
    return cast(Tensor, payload[key]).to(device)


def _prompt_tokens(
    tokenizer: Any,
    config: WaveLLMRunConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tokens = tokenizer(
        [config.data.prompt] * batch_size,
        padding=True,
        truncation=True,
        max_length=config.data.max_prompt_length,
        return_tensors="pt",
    )
    return (
        _token_tensor(tokens, "input_ids", device),
        _token_tensor(tokens, "attention_mask", device),
    )


def _target_labels(
    tokenizer: Any,
    captions: tuple[str, ...],
    config: WaveLLMRunConfig,
    *,
    device: torch.device,
) -> Tensor:
    tokens = tokenizer(
        list(captions),
        padding=True,
        truncation=True,
        max_length=config.data.max_target_length,
        return_tensors="pt",
    )
    labels = _token_tensor(tokens, "input_ids", device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int):
        raise WaveLLMRunError("mT5 tokenizer must define integer pad_token_id")
    labels = labels.masked_fill(labels == pad_token_id, -100)
    if not bool(torch.any(labels != -100)):
        raise WaveLLMRunError("target tokenization produced no supervised tokens")
    return labels


def _forward_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: _TensorBatch,
    config: WaveLLMRunConfig,
) -> Tensor:
    prompt_ids, prompt_mask = _prompt_tokens(
        tokenizer, config, batch_size=len(batch.sample_ids), device=batch.pose.device
    )
    labels = _target_labels(tokenizer, batch.captions, config, device=batch.pose.device)
    output = model(
        pose=batch.pose,
        pose_confidence=batch.pose_confidence,
        radar_features=batch.radar_features,
        frame_attention_mask=batch.frame_attention_mask,
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        labels=labels,
    )
    loss = getattr(output, "loss", None)
    if not isinstance(loss, Tensor):
        raise WaveLLMRunError("WaveLLM forward output is missing tensor loss")
    return loss


def _validation_loss(
    model: GeometryGuidedMT5,
    tokenizer: Any,
    loader: DataLoader[SignLanguageTranslationBatch],
    config: WaveLLMRunConfig,
    *,
    device: torch.device,
    precision: str,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for numpy_batch in loader:
            batch = _tensor_batch(numpy_batch, device)
            with _precision_context(device, precision):
                loss = _forward_loss(model, tokenizer, batch, config)
            value = float(loss.detach().float().cpu().item())
            if not math.isfinite(value):
                raise WaveLLMRunError("validation produced non-finite token loss")
            loss_sum += value * len(batch.sample_ids)
            sample_count += len(batch.sample_ids)
    if sample_count == 0:
        raise WaveLLMRunError("validation loader produced no samples")
    return loss_sum / sample_count


def _train_model(
    model: GeometryGuidedMT5,
    forward_model: torch.nn.Module,
    tokenizer: Any,
    train_loader: DataLoader[SignLanguageTranslationBatch],
    validation_loader: DataLoader[SignLanguageTranslationBatch],
    config: WaveLLMRunConfig,
    *,
    writer: RunArtifactWriter,
    state_bindings: Mapping[str, str],
    resume_metadata_path: Path | None,
    resume_tensor_path: Path | None,
    device: torch.device,
    precision: str,
    distributed: DistributedContext,
) -> tuple[list[dict[str, object]], int, int, str | None]:
    named_parameters = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    trainable = [parameter for _, parameter in named_parameters]
    if not named_parameters:
        raise WaveLLMRunError("WaveLLM run has no trainable parameters")
    optimization = config.optimization
    optimizer = torch.optim.AdamW(
        trainable,
        lr=optimization.learning_rate,
        betas=(optimization.beta1, optimization.beta2),
        weight_decay=optimization.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and precision == "16-mixed")
    loader_generator = train_loader.generator
    if loader_generator is None:
        raise WaveLLMRunError("training loader has no reproducible generator")
    history: list[dict[str, object]] = []
    global_step = 0
    start_epoch = 1
    resumed_from: str | None = None
    scope = _checkpoint_scope(config)
    model_state_names = set(_checkpoint_state(model, scope))
    if resume_metadata_path is not None and resume_tensor_path is not None:
        try:
            state = load_epoch_training_state(
                resume_metadata_path,
                resume_tensor_path,
                model=model,
                expected_model_state_names=model_state_names,
                named_parameters=named_parameters,
                optimizer=optimizer,
                scaler=scaler,
                loader_generator=loader_generator,
                device=device,
                expected_bindings=state_bindings,
                target_epochs=optimization.epochs,
                target_max_steps=optimization.max_steps,
            )
        except TrainingStateError as error:
            raise WaveLLMRunError(str(error)) from error
        history = list(state.history)
        global_step = state.global_step
        start_epoch = state.completed_epoch + 1
        resumed_from = state.source_run_id
    if start_epoch > optimization.epochs:
        raise WaveLLMRunError("resume state has already reached the configured epoch target")
    if optimization.max_steps is not None and global_step >= optimization.max_steps:
        raise WaveLLMRunError("resume state has already reached the configured step target")
    initial_global_step = global_step
    stop = False
    for epoch in range(start_epoch, optimization.epochs + 1):
        set_training_sampler_epoch(train_loader, epoch - 1)
        forward_model.train()
        if config.model.freeze_language_model:
            model.language_model.eval()
        loss_sum = 0.0
        sample_count = 0
        gradient_norm_sum = 0.0
        epoch_steps = 0
        for numpy_batch in train_loader:
            batch = _tensor_batch(numpy_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(device, precision):
                loss = _forward_loss(forward_model, tokenizer, batch, config)
            loss_value = float(loss.detach().float().cpu().item())
            if not math.isfinite(loss_value):
                raise WaveLLMRunError(f"non-finite training loss at step {global_step + 1}")
            torch.autograd.backward(scaler.scale(loss))
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, optimization.gradient_clip_norm
            )
            gradient_norm_value = float(gradient_norm.detach().float().cpu().item())
            if not math.isfinite(gradient_norm_value):
                raise WaveLLMRunError(
                    f"non-finite gradient norm at step {global_step + 1}"
                )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss_value * len(batch.sample_ids)
            sample_count += len(batch.sample_ids)
            gradient_norm_sum += gradient_norm_value
            global_step += 1
            epoch_steps += 1
            if optimization.max_steps is not None and global_step >= optimization.max_steps:
                stop = True
                break
        if epoch_steps == 0 or sample_count == 0:
            raise WaveLLMRunError("training loader produced no optimization batches")
        (
            global_loss_sum,
            global_sample_count,
            global_gradient_norm_sum,
            global_epoch_steps,
        ) = distributed.sum_values(
            (loss_sum, sample_count, gradient_norm_sum, epoch_steps)
        )
        validation_loss = distributed.rank_zero_call(
            lambda: _validation_loss(
                model,
                tokenizer,
                validation_loader,
                config,
                device=device,
                precision=precision,
            ),
            stage=f"epoch-{epoch} validation",
        )
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "steps": epoch_steps,
                "train_token_cross_entropy": global_loss_sum / global_sample_count,
                "mean_preclip_gradient_norm": (
                    global_gradient_norm_sum / global_epoch_steps
                ),
                "validation_token_cross_entropy": validation_loss,
            }
        )
        if not distributed.enabled and epoch_steps == len(train_loader):
            try:
                save_epoch_training_state(
                    writer,
                    model_state=_checkpoint_state(model, scope),
                    named_parameters=named_parameters,
                    optimizer=optimizer,
                    scaler=scaler,
                    loader_generator=loader_generator,
                    device=device,
                    bindings=state_bindings,
                    completed_epoch=epoch,
                    global_step=global_step,
                    configured_epochs=optimization.epochs,
                    configured_max_steps=optimization.max_steps,
                    history=history,
                )
            except TrainingStateError as error:
                raise WaveLLMRunError(str(error)) from error
        if stop:
            break
    return history, global_step, global_step - initial_global_step, resumed_from


def _checkpoint_scope(config: WaveLLMRunConfig) -> str:
    return "adapter_only" if config.model.freeze_language_model else "full_model"


def _checkpoint_state(
    model: GeometryGuidedMT5, scope: str
) -> dict[str, Tensor]:
    state = model.state_dict()
    if scope == "adapter_only":
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("language_model.")
        }
    if not state:
        raise WaveLLMRunError("WaveLLM checkpoint state is empty")
    return {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}


def _save_checkpoint(
    model: GeometryGuidedMT5, destination: Path, *, scope: str
) -> str:
    if destination.exists():
        raise WaveLLMRunError(f"checkpoint already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        save_file(_checkpoint_state(model, scope), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, RuntimeError, ValueError) as error:
        raise WaveLLMRunError(f"unable to save WaveLLM checkpoint: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(destination)


def _checkpoint_payload(
    *,
    writer: RunArtifactWriter,
    config: WaveLLMRunConfig,
    coordinate_frame: str,
    weights_sha256: str,
    global_step: int,
    epochs_executed: int,
    model: GeometryGuidedMT5,
    asset: ResolvedModelAsset,
    runtime_report: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    runtime_payload: Mapping[str, object],
    model_state_sha256: str,
) -> dict[str, object]:
    git = cast(Mapping[str, Any], runtime_report["git"])
    return {
        "schema_version": WAVELLM_CHECKPOINT_SCHEMA,
        "run_id": writer.run_id,
        "weights": {
            "filename": "checkpoint.safetensors",
            "sha256": weights_sha256,
            "format": "safetensors",
            "scope": _checkpoint_scope(config),
        },
        "model": config.model.to_dict(),
        "model_config_sha256": config.model_fingerprint,
        "task_config_sha256": config.fingerprint,
        "model_asset": _asset_identity(asset),
        "coordinate_frame": coordinate_frame,
        "pose_units": "m",
        "global_step": global_step,
        "epochs_executed": epochs_executed,
        "selection": "final_step",
        "parameter_count": _parameter_counts(model),
        "model_state_sha256": model_state_sha256,
        "runtime": dict(runtime_payload),
        "git_commit": git["commit"],
        "input_sha256": dict(sorted(input_hashes.items())),
    }


def _load_checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WaveLLMRunError(f"unable to read checkpoint metadata: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != WAVELLM_CHECKPOINT_SCHEMA
    ):
        raise WaveLLMRunError("checkpoint metadata has an unsupported schema")
    return payload


def _load_checkpoint(
    model: GeometryGuidedMT5,
    *,
    weights_path: Path,
    metadata_path: Path,
    config: WaveLLMRunConfig,
    coordinate_frame: str,
    asset: ResolvedModelAsset,
) -> str:
    metadata = _load_checkpoint_metadata(metadata_path)
    weights = metadata.get("weights")
    if not isinstance(weights, Mapping):
        raise WaveLLMRunError("checkpoint metadata is missing weights provenance")
    expected_sha256 = weights.get("sha256")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise WaveLLMRunError("checkpoint metadata has an invalid weights SHA-256")
    if weights.get("format") != "safetensors":
        raise WaveLLMRunError("checkpoint metadata must declare safetensors weights")
    scope = weights.get("scope")
    expected_scope = _checkpoint_scope(config)
    if scope != expected_scope:
        raise WaveLLMRunError(
            f"checkpoint scope {scope!r} does not match configured scope {expected_scope!r}"
        )
    if metadata.get("pose_units") != "m":
        raise WaveLLMRunError("checkpoint pose units must be metres")
    observed_sha256 = sha256_file(weights_path)
    if observed_sha256 != expected_sha256:
        raise WaveLLMRunError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    if metadata.get("model_config_sha256") != config.model_fingerprint:
        raise WaveLLMRunError("checkpoint model configuration does not match the run config")
    if metadata.get("task_config_sha256") != config.fingerprint:
        raise WaveLLMRunError("checkpoint task configuration does not match the run config")
    if metadata.get("coordinate_frame") != coordinate_frame:
        raise WaveLLMRunError("checkpoint and evaluation manifest coordinate frames do not match")
    if metadata.get("model_asset") != _asset_identity(asset):
        raise WaveLLMRunError("checkpoint and resolved mT5 asset identities do not match")
    try:
        state = load_file(weights_path, device="cpu")
        if scope == "full_model":
            model.load_state_dict(state, strict=True)
        else:
            expected_adapter_keys = {
                name for name in model.state_dict() if not name.startswith("language_model.")
            }
            if set(state) != expected_adapter_keys:
                raise WaveLLMRunError(
                    "adapter checkpoint tensor inventory does not match the model"
                )
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys or any(
                not name.startswith("language_model.") for name in incompatible.missing_keys
            ):
                raise WaveLLMRunError("adapter checkpoint produced incompatible model keys")
    except WaveLLMRunError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise WaveLLMRunError(f"unable to load WaveLLM checkpoint: {error}") from error
    return observed_sha256


def _prediction_records(
    model: GeometryGuidedMT5,
    tokenizer: Any,
    loader: DataLoader[SignLanguageTranslationBatch],
    accumulator: LanguageMetricAccumulator,
    config: WaveLLMRunConfig,
    *,
    device: torch.device,
    precision: str,
    checkpoint_sha256: str,
) -> Iterator[Mapping[str, object]]:
    model.eval()
    with torch.inference_mode():
        for numpy_batch in loader:
            batch = _tensor_batch(numpy_batch, device)
            prompt_ids, prompt_mask = _prompt_tokens(
                tokenizer, config, batch_size=len(batch.sample_ids), device=device
            )
            with _precision_context(device, precision):
                generated_ids = model.generate(
                    pose=batch.pose,
                    pose_confidence=batch.pose_confidence,
                    radar_features=batch.radar_features,
                    frame_attention_mask=batch.frame_attention_mask,
                    prompt_input_ids=prompt_ids,
                    prompt_attention_mask=prompt_mask,
                    max_new_tokens=config.generation.max_new_tokens,
                    num_beams=config.generation.num_beams,
                )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
                raise WaveLLMRunError(
                    "tokenizer batch_decode must return one text string per sample"
                )
            predictions = tuple(item.strip() for item in decoded)
            if len(predictions) != len(batch.sample_ids):
                raise WaveLLMRunError("generation output count does not match the input batch")
            per_sample = accumulator.update(batch.captions, predictions)
            for sample_id, reference, prediction, metric in zip(
                batch.sample_ids,
                batch.captions,
                predictions,
                per_sample,
                strict=True,
            ):
                record: dict[str, object] = {
                    "schema_version": WAVELLM_PREDICTION_SCHEMA,
                    "sample_id": sample_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "prediction": prediction,
                    "exact_match": metric.exact_match,
                    "character_edit_distance": metric.character_edit_distance,
                    "reference_character_count": metric.reference_character_count,
                    "prediction_character_count": metric.prediction_character_count,
                }
                if config.evaluation.save_references:
                    record["reference"] = reference
                yield record


def _resolved_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _prepare_run(
    experiment_config: ExperimentConfig,
    task_config: WaveLLMRunConfig,
    asset_config: ModelAssetSetConfig,
    model_root: str | Path,
    *,
    source_experiment_config: str | Path,
    source_task_config: str | Path,
    source_asset_config: str | Path,
    input_specs: Sequence[tuple[str, str, str | Path]],
    project_root: Path,
    command: Sequence[str],
    runtime_report: Mapping[str, Any] | None,
    created_at: datetime | None,
    distributed: DistributedContext,
) -> tuple[
    RunArtifactWriter,
    dict[str, Path],
    dict[str, str],
    Mapping[str, Any],
    ResolvedModelAsset,
]:
    root = project_root.expanduser().resolve()
    if experiment_config.task is not Task.SIGN_LANGUAGE_TRANSLATION:
        raise WaveLLMRunError("WaveLLM runs require task=sign_language_translation")
    if not task_config.data.verify_checksums:
        raise WaveLLMRunError("formal WaveLLM runs require data.verify_checksums=true")
    task_config_path = _resolved_path(source_task_config, root)
    asset_config_path = _resolved_path(source_asset_config, root)
    if load_wavellm_run_config(task_config_path).fingerprint != task_config.fingerprint:
        raise WaveLLMRunError("source WaveLLM configuration does not match the loaded config")
    if load_model_asset_config(asset_config_path).fingerprint != asset_config.fingerprint:
        raise WaveLLMRunError("source model asset configuration does not match the loaded config")
    resolved_model_root = _resolved_path(model_root, root)
    resolved_asset = resolve_model_asset(
        asset_config, resolved_model_root, task_config.model.asset_id
    )
    identities = distributed.all_gather_object(_asset_identity(resolved_asset))
    if any(identity != identities[0] for identity in identities[1:]):
        raise WaveLLMRunError("resolved mT5 asset identity differs across ranks")

    def initialize() -> dict[str, object]:
        report = dict(collect_runtime_report(root) if runtime_report is None else runtime_report)
        _require_formal_runtime(report, root)
        plan = build_run_plan(
            experiment_config,
            root,
            created_at=created_at,
            runtime_report=report,
        )
        paths = {name: _resolved_path(path, root) for name, _, path in input_specs}
        model_input_paths = {
            "model_asset_collection": resolved_model_root / MODEL_ASSET_COLLECTION_NAME,
            "model_asset_manifest": resolved_asset.path / MODEL_ASSET_MANIFEST_NAME,
        }
        run_inputs = [
            RunInput.capture(name="wavellm_config", kind="config", path=task_config_path),
            RunInput.capture(name="model_asset_config", kind="config", path=asset_config_path),
            *(
                RunInput.capture(name=name, kind="model", path=path)
                for name, path in model_input_paths.items()
            ),
            *(
                RunInput.capture(name=name, kind=kind, path=paths[name])
                for name, kind, _ in input_specs
            ),
        ]
        writer = RunArtifactWriter.initialize(
            plan,
            source_config=_resolved_path(source_experiment_config, root),
            inputs=run_inputs,
            command=command,
        )
        try:
            writer.write_json_artifact("wavellm.resolved.json", task_config.to_dict())
        except Exception as error:
            _finalize_failed_run(writer, error)
            raise
        return {
            "run_dir": str(writer.run_dir),
            "run_id": writer.run_id,
            "paths": {name: str(path) for name, path in paths.items()},
            "hashes": {item.name: item.sha256 for item in run_inputs},
            "runtime_report": report,
        }

    shared = distributed.rank_zero_call(initialize, stage="formal run initialization")
    if not isinstance(shared, Mapping):
        raise WaveLLMRunError("distributed run initialization returned invalid metadata")
    run_dir = shared.get("run_dir")
    run_id = shared.get("run_id")
    raw_paths = shared.get("paths")
    raw_hashes = shared.get("hashes")
    report = shared.get("runtime_report")
    if (
        not isinstance(run_dir, str)
        or not isinstance(run_id, str)
        or not isinstance(raw_paths, Mapping)
        or not isinstance(raw_hashes, Mapping)
        or not isinstance(report, Mapping)
    ):
        raise WaveLLMRunError("distributed run initialization metadata is incomplete")
    paths = {str(name): Path(str(path)) for name, path in raw_paths.items()}
    hashes = {str(name): str(value) for name, value in raw_hashes.items()}
    return RunArtifactWriter(Path(run_dir), run_id), paths, hashes, report, resolved_asset


def _finalize_failed_run(writer: RunArtifactWriter, error: BaseException) -> None:
    with suppress(Exception):
        writer.finalize(status="failed", failure=f"{type(error).__name__}: {error}")


def train_wavellm(
    experiment_config: ExperimentConfig,
    task_config: WaveLLMRunConfig,
    asset_config: ModelAssetSetConfig,
    model_root: str | Path,
    *,
    source_experiment_config: str | Path,
    source_task_config: str | Path,
    source_asset_config: str | Path,
    train_manifest_path: str | Path,
    validation_manifest_path: str | Path,
    split_assignments_path: str | Path,
    project_root: Path,
    command: Sequence[str],
    resume_state_metadata_path: str | Path | None = None,
    resume_state_tensors_path: str | Path | None = None,
    runtime_report: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    run_started = time.perf_counter()
    if (resume_state_metadata_path is None) != (resume_state_tensors_path is None):
        raise WaveLLMRunError("resume requires both state metadata and Safetensors")
    root = project_root.expanduser().resolve()
    resolved_experiment = experiment_config.resolved(root)
    try:
        distributed = DistributedContext.from_environment(resolved_experiment.runtime)
    except DistributedRunError as error:
        raise WaveLLMRunError(str(error)) from error
    if distributed.enabled and resume_state_metadata_path is not None:
        raise WaveLLMRunError(
            "DDP resume is unsupported until every rank's RNG and sampler state is captured"
        )
    try:
        distributed.initialize()
    except DistributedRunError as error:
        raise WaveLLMRunError(str(error)) from error
    resume_inputs: tuple[tuple[str, str, str | Path], ...] = ()
    if resume_state_metadata_path is not None and resume_state_tensors_path is not None:
        resume_inputs = (
            ("resume_state_metadata", "checkpoint", resume_state_metadata_path),
            ("resume_state_tensors", "checkpoint", resume_state_tensors_path),
        )
    try:
        writer, paths, input_hashes, report, asset = _prepare_run(
            experiment_config,
            task_config,
            asset_config,
            model_root,
            source_experiment_config=source_experiment_config,
            source_task_config=source_task_config,
            source_asset_config=source_asset_config,
            input_specs=(
                ("train_manifest", "manifest", train_manifest_path),
                ("validation_manifest", "manifest", validation_manifest_path),
                ("split_assignments", "split", split_assignments_path),
            )
            + resume_inputs,
            project_root=project_root,
            command=command,
            runtime_report=runtime_report,
            created_at=created_at,
            distributed=distributed,
        )
    except Exception:
        distributed.close()
        raise
    try:
        train_manifest = SignLanguageTranslationManifest(
            paths["train_manifest"],
            data_root=resolved_experiment.paths.data_root,
            verify_checksums=task_config.data.verify_checksums,
        )
        validation_manifest = SignLanguageTranslationManifest(
            paths["validation_manifest"],
            data_root=resolved_experiment.paths.data_root,
            verify_checksums=task_config.data.verify_checksums,
        )
        _validate_manifest_for_model(train_manifest, task_config, role="train")
        _validate_manifest_for_model(validation_manifest, task_config, role="validation")
        _validate_split_separation(train_manifest, validation_manifest)
        if train_manifest.coordinate_frame != validation_manifest.coordinate_frame:
            raise WaveLLMRunError("train and validation coordinate frames do not match")
        validate_split_bindings(
            {
                "train_manifest": (record.sample_id for record in train_manifest.records),
                "validation_manifest": (
                    record.sample_id for record in validation_manifest.records
                ),
            },
            paths["split_assignments"],
            {"train_manifest": "train", "validation_manifest": "validation"},
        )

        device = distributed.device
        dtype = _model_dtype(device, resolved_experiment.runtime.precision)
        _seed_runtime(resolved_experiment.runtime.seed, resolved_experiment.runtime.deterministic)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, tokenizer = _load_model_and_tokenizer(
            asset, task_config, device=device, dtype=dtype
        )
        distributed_topology = distributed.topology_payload()
        actual_runtime = _runtime_payload(
            model,
            resolved_experiment.runtime,
            device,
            asset,
            distributed_topology,
        )
        distributed.rank_zero_call(
            lambda: str(writer.write_json_artifact("wavellm.runtime.json", actual_runtime)),
            stage="runtime artifact publication",
        )
        forward_model = distributed.wrap_model(model)
        train_loader = _loader(
            train_manifest,
            task_config,
            shuffle=task_config.data.shuffle,
            seed=resolved_experiment.runtime.seed,
            device=device,
            distributed=distributed,
        )
        validation_loader = _loader(
            validation_manifest,
            task_config,
            shuffle=False,
            seed=resolved_experiment.runtime.seed,
            device=device,
        )
        git = cast(Mapping[str, Any], report["git"])
        state_bindings = {
            "task": "sign_language_translation",
            "training_config_sha256": task_config.training_fingerprint,
            "model_config_sha256": task_config.model_fingerprint,
            "model_asset_config_sha256": input_hashes["model_asset_config"],
            "model_asset_collection_sha256": input_hashes["model_asset_collection"],
            "model_asset_manifest_sha256": input_hashes["model_asset_manifest"],
            "train_manifest_sha256": input_hashes["train_manifest"],
            "validation_manifest_sha256": input_hashes["validation_manifest"],
            "split_assignments_sha256": input_hashes["split_assignments"],
            "coordinate_frame": train_manifest.coordinate_frame,
            "checkpoint_scope": _checkpoint_scope(task_config),
            "runtime_seed": str(resolved_experiment.runtime.seed),
            "runtime_precision": resolved_experiment.runtime.precision,
            "runtime_deterministic": str(
                resolved_experiment.runtime.deterministic
            ).lower(),
            "device_type": device.type,
            "git_commit": str(git["commit"]),
        }
        training_started = time.perf_counter()
        history, global_step, steps_this_run, resumed_from = _train_model(
            model,
            forward_model,
            tokenizer,
            train_loader,
            validation_loader,
            task_config,
            writer=writer,
            state_bindings=state_bindings,
            resume_metadata_path=paths.get("resume_state_metadata"),
            resume_tensor_path=paths.get("resume_state_tensors"),
            device=device,
            precision=resolved_experiment.runtime.precision,
            distributed=distributed,
        )
        local_training_seconds = time.perf_counter() - training_started
        training_seconds = distributed.max_value(local_training_seconds)

        scope = _checkpoint_scope(task_config)
        model_state_sha256 = distributed.assert_consistent_state(
            _checkpoint_state(model, scope)
        )
        weights_path = writer.artifact_path("checkpoint.safetensors")

        def publish_checkpoint() -> str:
            weights_sha256 = _save_checkpoint(model, weights_path, scope=scope)
            writer.register_artifact("checkpoint.safetensors")
            writer.write_json_artifact(
                "checkpoint.json",
                _checkpoint_payload(
                    writer=writer,
                    config=task_config,
                    coordinate_frame=train_manifest.coordinate_frame,
                    weights_sha256=weights_sha256,
                    global_step=global_step,
                    epochs_executed=len(history),
                    model=model,
                    asset=asset,
                    runtime_report=report,
                    input_hashes=input_hashes,
                    runtime_payload=actual_runtime,
                    model_state_sha256=model_state_sha256,
                ),
            )
            return weights_sha256

        weights_sha256 = distributed.rank_zero_call(
            publish_checkpoint, stage="checkpoint publication"
        )

        accumulator = LanguageMetricAccumulator()
        prediction_loader = _loader(
            validation_manifest,
            task_config,
            shuffle=False,
            seed=resolved_experiment.runtime.seed,
            device=device,
            distributed=distributed,
            exact_distributed_coverage=True,
        )
        prediction_started = time.perf_counter()
        write_prediction_shard(
            writer.run_dir,
            run_id=writer.run_id,
            prediction_schema=WAVELLM_PREDICTION_SCHEMA,
            rank=distributed.rank,
            world_size=distributed.world_size,
            records=_prediction_records(
                model,
                tokenizer,
                prediction_loader,
                accumulator,
                task_config,
                device=device,
                precision=resolved_experiment.runtime.precision,
                checkpoint_sha256=weights_sha256,
            ),
        )
        local_prediction_seconds = time.perf_counter() - prediction_started
        distributed.barrier()
        distributed.rank_zero_call(
            lambda: aggregate_prediction_shards(
                writer,
                prediction_schema=WAVELLM_PREDICTION_SCHEMA,
                world_size=distributed.world_size,
                expected_sample_ids=(
                    record.sample_id for record in validation_manifest.records
                ),
            ).record_count,
            stage="prediction aggregation",
        )
        distributed.barrier()
        prediction_seconds = distributed.max_value(local_prediction_seconds)
        merged_accumulator = LanguageMetricAccumulator()
        for state in distributed.all_gather_object(accumulator.state_dict()):
            merged_accumulator.merge_state(state)
        metrics = merged_accumulator.values()
        metric_values: dict[str, int | float] = {
            **metrics,
            "global_step": global_step,
            "epochs_executed": len(history),
        }
        rank_performance = distributed.all_gather_object(
            {
                **distributed.rank_payload(),
                "optimizer_steps_this_run": steps_this_run,
                "training_seconds": local_training_seconds,
                "prediction_samples": accumulator.sample_count,
                "prediction_seconds": local_prediction_seconds,
                "cuda_memory": _cuda_memory_payload(device),
                "model_state_sha256": model_state_sha256,
            }
        )
        counts = _parameter_counts(model)
        end_to_end_seconds = distributed.max_value(time.perf_counter() - run_started)

        def finalize_run() -> dict[str, int | float]:
            writer.write_json_artifact(
                "history.json",
                {
                    "schema_version": WAVELLM_HISTORY_SCHEMA,
                    "run_id": writer.run_id,
                    "task_config_sha256": task_config.fingerprint,
                    "global_step": global_step,
                    "epochs_executed": len(history),
                    "resumed_from_run_id": resumed_from,
                    "records": history,
                    "final_validation": metrics,
                },
            )
            writer.write_json_artifact(
                "performance.json",
                {
                    "schema_version": WAVELLM_PERFORMANCE_SCHEMA,
                    "mode": "train",
                    "device": str(device),
                    "precision": resolved_experiment.runtime.precision,
                    "parameter_count": counts,
                    "checkpoint_scope": scope,
                    "optimizer_steps": global_step,
                    "optimizer_steps_this_run": steps_this_run,
                    "training_seconds": training_seconds,
                    "optimizer_steps_per_second": steps_this_run / training_seconds,
                    "prediction_samples": merged_accumulator.sample_count,
                    "prediction_seconds": prediction_seconds,
                    "prediction_samples_per_second": (
                        merged_accumulator.sample_count / prediction_seconds
                    ),
                    "end_to_end_seconds": end_to_end_seconds,
                    "cuda_memory": _cuda_memory_payload(device),
                    "distributed": {
                        **distributed_topology,
                        "rank_performance": rank_performance,
                        "model_state_sha256": model_state_sha256,
                    },
                },
            )
            writer.write_metrics(
                protocol_id=LANGUAGE_METRIC_PROTOCOL,
                split="validation",
                values=metric_values,
                sample_count=merged_accumulator.sample_count,
            )
            writer.finalize(status="completed")
            return metric_values

        metric_values = distributed.rank_zero_call(
            finalize_run, stage="final artifact publication"
        )
    except KeyboardInterrupt as error:
        if distributed.is_rank_zero:
            writer.finalize(status="aborted", failure="interrupted by operator")
        raise error
    except Exception as error:
        if distributed.is_rank_zero:
            _finalize_failed_run(writer, error)
        if isinstance(error, WaveLLMRunError):
            raise
        raise WaveLLMRunError(f"WaveLLM training failed: {error}") from error
    finally:
        distributed.close()
    return {
        "schema_version": WAVELLM_RUN_RESULT_SCHEMA,
        "mode": "train",
        "status": "completed",
        "run_id": writer.run_id,
        "run_dir": str(writer.run_dir),
        "metrics": metric_values,
    }


def evaluate_wavellm(
    experiment_config: ExperimentConfig,
    task_config: WaveLLMRunConfig,
    asset_config: ModelAssetSetConfig,
    model_root: str | Path,
    *,
    source_experiment_config: str | Path,
    source_task_config: str | Path,
    source_asset_config: str | Path,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_metadata_path: str | Path,
    split_assignments_path: str | Path,
    split: str,
    project_root: Path,
    command: Sequence[str],
    runtime_report: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    run_started = time.perf_counter()
    if split not in {"train", "validation", "test"}:
        raise WaveLLMRunError("evaluation split must be train, validation, or test")
    root = project_root.expanduser().resolve()
    resolved_experiment = experiment_config.resolved(root)
    try:
        distributed = DistributedContext.from_environment(resolved_experiment.runtime)
        distributed.initialize()
    except DistributedRunError as error:
        raise WaveLLMRunError(str(error)) from error
    try:
        writer, paths, _, _, asset = _prepare_run(
            experiment_config,
            task_config,
            asset_config,
            model_root,
            source_experiment_config=source_experiment_config,
            source_task_config=source_task_config,
            source_asset_config=source_asset_config,
            input_specs=(
                ("evaluation_manifest", "manifest", manifest_path),
                ("split_assignments", "split", split_assignments_path),
                ("checkpoint_weights", "checkpoint", checkpoint_path),
                ("checkpoint_metadata", "checkpoint", checkpoint_metadata_path),
            ),
            project_root=project_root,
            command=command,
            runtime_report=runtime_report,
            created_at=created_at,
            distributed=distributed,
        )
    except Exception:
        distributed.close()
        raise
    try:
        manifest = SignLanguageTranslationManifest(
            paths["evaluation_manifest"],
            data_root=resolved_experiment.paths.data_root,
            verify_checksums=task_config.data.verify_checksums,
        )
        _validate_manifest_for_model(manifest, task_config, role="evaluation")
        validate_split_bindings(
            {"evaluation_manifest": (record.sample_id for record in manifest.records)},
            paths["split_assignments"],
            {"evaluation_manifest": split},
        )
        device = distributed.device
        dtype = _model_dtype(device, resolved_experiment.runtime.precision)
        _seed_runtime(resolved_experiment.runtime.seed, resolved_experiment.runtime.deterministic)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, tokenizer = _load_model_and_tokenizer(
            asset, task_config, device=device, dtype=dtype
        )
        checkpoint_sha256 = _load_checkpoint(
            model,
            weights_path=paths["checkpoint_weights"],
            metadata_path=paths["checkpoint_metadata"],
            config=task_config,
            coordinate_frame=manifest.coordinate_frame,
            asset=asset,
        )
        distributed_topology = distributed.topology_payload()
        actual_runtime = _runtime_payload(
            model,
            resolved_experiment.runtime,
            device,
            asset,
            distributed_topology,
        )
        distributed.rank_zero_call(
            lambda: str(writer.write_json_artifact("wavellm.runtime.json", actual_runtime)),
            stage="runtime artifact publication",
        )
        model_state_sha256 = distributed.assert_consistent_state(
            _checkpoint_state(model, _checkpoint_scope(task_config))
        )
        loader = _loader(
            manifest,
            task_config,
            shuffle=False,
            seed=resolved_experiment.runtime.seed,
            device=device,
            distributed=distributed,
            exact_distributed_coverage=True,
        )
        accumulator = LanguageMetricAccumulator()
        prediction_started = time.perf_counter()
        write_prediction_shard(
            writer.run_dir,
            run_id=writer.run_id,
            prediction_schema=WAVELLM_PREDICTION_SCHEMA,
            rank=distributed.rank,
            world_size=distributed.world_size,
            records=_prediction_records(
                model,
                tokenizer,
                loader,
                accumulator,
                task_config,
                device=device,
                precision=resolved_experiment.runtime.precision,
                checkpoint_sha256=checkpoint_sha256,
            ),
        )
        local_prediction_seconds = time.perf_counter() - prediction_started
        distributed.barrier()
        distributed.rank_zero_call(
            lambda: aggregate_prediction_shards(
                writer,
                prediction_schema=WAVELLM_PREDICTION_SCHEMA,
                world_size=distributed.world_size,
                expected_sample_ids=(record.sample_id for record in manifest.records),
            ).record_count,
            stage="prediction aggregation",
        )
        distributed.barrier()
        prediction_seconds = distributed.max_value(local_prediction_seconds)
        merged_accumulator = LanguageMetricAccumulator()
        for state in distributed.all_gather_object(accumulator.state_dict()):
            merged_accumulator.merge_state(state)
        metrics = merged_accumulator.values()
        rank_performance = distributed.all_gather_object(
            {
                **distributed.rank_payload(),
                "prediction_samples": accumulator.sample_count,
                "prediction_seconds": local_prediction_seconds,
                "cuda_memory": _cuda_memory_payload(device),
                "model_state_sha256": model_state_sha256,
            }
        )
        end_to_end_seconds = distributed.max_value(time.perf_counter() - run_started)

        def finalize_run() -> dict[str, int | float]:
            writer.write_metrics(
                protocol_id=LANGUAGE_METRIC_PROTOCOL,
                split=split,
                values=metrics,
                sample_count=merged_accumulator.sample_count,
            )
            writer.write_json_artifact(
                "performance.json",
                {
                    "schema_version": WAVELLM_PERFORMANCE_SCHEMA,
                    "mode": "evaluate",
                    "device": str(device),
                    "precision": resolved_experiment.runtime.precision,
                    "parameter_count": _parameter_counts(model),
                    "checkpoint_scope": _checkpoint_scope(task_config),
                    "prediction_samples": merged_accumulator.sample_count,
                    "prediction_seconds": prediction_seconds,
                    "prediction_samples_per_second": (
                        merged_accumulator.sample_count / prediction_seconds
                    ),
                    "end_to_end_seconds": end_to_end_seconds,
                    "cuda_memory": _cuda_memory_payload(device),
                    "distributed": {
                        **distributed_topology,
                        "rank_performance": rank_performance,
                        "model_state_sha256": model_state_sha256,
                    },
                },
            )
            writer.finalize(status="completed")
            return metrics

        metrics = distributed.rank_zero_call(
            finalize_run, stage="final artifact publication"
        )
    except KeyboardInterrupt as error:
        if distributed.is_rank_zero:
            writer.finalize(status="aborted", failure="interrupted by operator")
        raise error
    except Exception as error:
        if distributed.is_rank_zero:
            _finalize_failed_run(writer, error)
        if isinstance(error, WaveLLMRunError):
            raise
        raise WaveLLMRunError(f"WaveLLM evaluation failed: {error}") from error
    finally:
        distributed.close()
    return {
        "schema_version": WAVELLM_RUN_RESULT_SCHEMA,
        "mode": "evaluate",
        "status": "completed",
        "run_id": writer.run_id,
        "run_dir": str(writer.run_dir),
        "metrics": metrics,
    }
