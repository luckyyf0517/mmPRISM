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
from mmprism.config import ExperimentConfig, RuntimeConfig, Task
from mmprism.data import (
    PoseReconstructionBatch,
    PoseReconstructionManifest,
    PoseReconstructionSample,
    collate_pose_reconstruction_samples,
)
from mmprism.evaluation import POSE_METRIC_PROTOCOL, PoseMetricAccumulator
from mmprism.evaluation.pose import masked_pose_l1_metres
from mmprism.models import CubeNetSpatialEncoder, OmniHandCubeNet, TemporalTransformerAggregator
from mmprism.runtime import build_run_plan, collect_runtime_report
from mmprism.training.distributed import (
    DistributedContext,
    DistributedRunError,
    prediction_sampler,
    set_training_sampler_epoch,
    training_sampler,
)
from mmprism.training.omnihand_run_config import (
    OmniHandRunConfig,
    OmniHandRunError,
    load_omnihand_run_config,
)
from mmprism.training.resume import (
    TrainingStateError,
    load_epoch_training_state,
    save_epoch_training_state,
)

OMNIHAND_CHECKPOINT_SCHEMA = "mmprism.omnihand_checkpoint.v1"
OMNIHAND_HISTORY_SCHEMA = "mmprism.omnihand_history.v1"
OMNIHAND_PREDICTION_SCHEMA = "mmprism.pose_prediction.v1"
OMNIHAND_PERFORMANCE_SCHEMA = "mmprism.omnihand_performance.v1"
OMNIHAND_RUN_RESULT_SCHEMA = "mmprism.omnihand_run_result.v1"
OMNIHAND_RUNTIME_SCHEMA = "mmprism.omnihand_runtime.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class _PoseDataset(Dataset[PoseReconstructionSample]):
    def __init__(self, manifest: PoseReconstructionManifest) -> None:
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> PoseReconstructionSample:
        return self.manifest.load_sample(index)


@dataclass(frozen=True, slots=True)
class _TensorBatch:
    sample_ids: tuple[str, ...]
    radar_cube: Tensor
    frame_mask: Tensor
    pose_target: Tensor
    pose_valid: Tensor
    coordinate_frame: str


def _require_formal_runtime(runtime_report: Mapping[str, Any], project_root: Path) -> None:
    reported_root = runtime_report.get("project_root")
    if not isinstance(reported_root, str) or Path(reported_root).resolve() != project_root:
        raise OmniHandRunError("formal OmniHand runs require matching project-root provenance")
    git = runtime_report.get("git")
    if (
        not isinstance(git, Mapping)
        or not isinstance(git.get("commit"), str)
        or not _GIT_COMMIT.fullmatch(git["commit"])
    ):
        raise OmniHandRunError("formal OmniHand runs require Git commit provenance")
    if git.get("dirty") is not False:
        raise OmniHandRunError("formal OmniHand runs require a clean Git worktree")


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


def _build_model(config: OmniHandRunConfig) -> OmniHandCubeNet:
    spatial_config = config.model.spatial
    temporal_config = config.model.temporal
    spatial = CubeNetSpatialEncoder(
        in_channels=spatial_config.in_channels,
        stem_channels=spatial_config.stem_channels,
        stage_channels=spatial_config.stage_channels,
        stage_depths=spatial_config.stage_depths,
        channel_attention=spatial_config.channel_attention,
        spatial_attention=spatial_config.spatial_attention,
        se_attention=spatial_config.se_attention,
        use_pafpn=spatial_config.use_pafpn,
        fpn_channels=spatial_config.fpn_channels,
    )
    temporal = TemporalTransformerAggregator(
        spatial.feature_dim,
        max_frames=temporal_config.max_frames,
        layers=temporal_config.layers,
        heads=temporal_config.heads,
        feedforward_dim=temporal_config.feedforward_dim,
        dropout=temporal_config.dropout,
    )
    return OmniHandCubeNet(
        spatial,
        temporal,
        joint_count=config.model.joint_count,
        coordinate_dim=config.model.coordinate_dim,
    )


def _runtime_payload(
    model: OmniHandCubeNet,
    runtime: RuntimeConfig,
    device: torch.device,
    distributed: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": OMNIHAND_RUNTIME_SCHEMA,
        "seed": runtime.seed,
        "deterministic": runtime.deterministic,
        "precision": runtime.precision,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
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
    manifest: PoseReconstructionManifest,
    config: OmniHandRunConfig,
    *,
    role: str,
) -> None:
    doppler_channels = manifest.radar_spatial_shape[0]
    if doppler_channels != config.model.spatial.in_channels:
        raise OmniHandRunError(
            f"{role} manifest has {doppler_channels} Doppler channels, "
            f"model expects {config.model.spatial.in_channels}"
        )
    longest = max(record.radar_cube_shape[0] for record in manifest.records)
    if longest > config.model.temporal.max_frames:
        raise OmniHandRunError(
            f"{role} manifest contains {longest} frames, model maximum is "
            f"{config.model.temporal.max_frames}"
        )


def _loader(
    manifest: PoseReconstructionManifest,
    config: OmniHandRunConfig,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
    distributed: DistributedContext | None = None,
    exact_distributed_coverage: bool = False,
) -> DataLoader[PoseReconstructionBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = _PoseDataset(manifest)
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
            collate_pose_reconstruction_samples,
            max_frames=config.model.temporal.max_frames,
        ),
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
    return cast(DataLoader[PoseReconstructionBatch], loader)


def _tensor_batch(batch: PoseReconstructionBatch, device: torch.device) -> _TensorBatch:
    non_blocking = device.type == "cuda"
    return _TensorBatch(
        sample_ids=batch.sample_ids,
        radar_cube=torch.from_numpy(batch.radar_cube).to(device, non_blocking=non_blocking),
        frame_mask=torch.from_numpy(batch.frame_mask).to(device, non_blocking=non_blocking),
        pose_target=torch.from_numpy(batch.pose_target).to(device, non_blocking=non_blocking),
        pose_valid=torch.from_numpy(batch.pose_valid).to(device, non_blocking=non_blocking),
        coordinate_frame=batch.coordinate_frame,
    )


def _evaluate_summary(
    model: OmniHandCubeNet,
    loader: DataLoader[PoseReconstructionBatch],
    *,
    device: torch.device,
    precision: str,
    pck_threshold_mm: float,
) -> dict[str, float]:
    accumulator = PoseMetricAccumulator(pck_threshold_mm=pck_threshold_mm)
    model.eval()
    with torch.inference_mode():
        for numpy_batch in loader:
            batch = _tensor_batch(numpy_batch, device)
            with _precision_context(device, precision):
                prediction = model(batch.radar_cube, batch.frame_mask).joints
            accumulator.update(prediction.float(), batch.pose_target.float(), batch.pose_valid)
    return accumulator.values()


def _train_model(
    model: OmniHandCubeNet,
    forward_model: torch.nn.Module,
    train_loader: DataLoader[PoseReconstructionBatch],
    validation_loader: DataLoader[PoseReconstructionBatch],
    config: OmniHandRunConfig,
    *,
    writer: RunArtifactWriter,
    state_bindings: Mapping[str, str],
    resume_metadata_path: Path | None,
    resume_tensor_path: Path | None,
    device: torch.device,
    precision: str,
    distributed: DistributedContext,
) -> tuple[list[dict[str, object]], int, int, str | None]:
    optimization = config.optimization
    named_parameters = tuple(model.named_parameters())
    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in named_parameters),
        lr=optimization.learning_rate,
        betas=(optimization.beta1, optimization.beta2),
        weight_decay=optimization.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and precision == "16-mixed")
    loader_generator = train_loader.generator
    if loader_generator is None:
        raise OmniHandRunError("training loader has no reproducible generator")
    history: list[dict[str, object]] = []
    global_step = 0
    start_epoch = 1
    resumed_from: str | None = None
    if resume_metadata_path is not None and resume_tensor_path is not None:
        try:
            state = load_epoch_training_state(
                resume_metadata_path,
                resume_tensor_path,
                model=model,
                expected_model_state_names=set(model.state_dict()),
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
            raise OmniHandRunError(str(error)) from error
        history = list(state.history)
        global_step = state.global_step
        start_epoch = state.completed_epoch + 1
        resumed_from = state.source_run_id
    if start_epoch > optimization.epochs:
        raise OmniHandRunError("resume state has already reached the configured epoch target")
    if optimization.max_steps is not None and global_step >= optimization.max_steps:
        raise OmniHandRunError("resume state has already reached the configured step target")
    initial_global_step = global_step
    stop = False
    for epoch in range(start_epoch, optimization.epochs + 1):
        set_training_sampler_epoch(train_loader, epoch - 1)
        forward_model.train()
        coordinate_error_sum = 0.0
        coordinate_count = 0
        gradient_norm_sum = 0.0
        epoch_steps = 0
        for numpy_batch in train_loader:
            batch = _tensor_batch(numpy_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(device, precision):
                prediction = forward_model(batch.radar_cube, batch.frame_mask).joints
                loss = masked_pose_l1_metres(
                    prediction.float(), batch.pose_target.float(), batch.pose_valid
                )
            if not bool(torch.isfinite(loss)):
                raise OmniHandRunError(f"non-finite training loss at step {global_step + 1}")
            torch.autograd.backward(scaler.scale(loss))
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimization.gradient_clip_norm
            )
            gradient_norm_value = float(gradient_norm.detach().float().cpu().item())
            if not math.isfinite(gradient_norm_value):
                raise OmniHandRunError(f"non-finite gradient norm at step {global_step + 1}")
            scaler.step(optimizer)
            scaler.update()

            valid_coordinates = int(batch.pose_valid.sum().item()) * 3
            coordinate_error_sum += float(loss.detach().cpu().item()) * valid_coordinates
            coordinate_count += valid_coordinates
            gradient_norm_sum += gradient_norm_value
            global_step += 1
            epoch_steps += 1
            if optimization.max_steps is not None and global_step >= optimization.max_steps:
                stop = True
                break

        if epoch_steps == 0 or coordinate_count == 0:
            raise OmniHandRunError("training loader produced no optimization batches")
        (
            global_coordinate_error_sum,
            global_coordinate_count,
            global_gradient_norm_sum,
            global_epoch_steps,
        ) = distributed.sum_values(
            (
                coordinate_error_sum,
                coordinate_count,
                gradient_norm_sum,
                epoch_steps,
            )
        )
        validation = distributed.rank_zero_call(
            lambda: _evaluate_summary(
                model,
                validation_loader,
                device=device,
                precision=precision,
                pck_threshold_mm=config.evaluation.pck_threshold_mm,
            ),
            stage=f"epoch-{epoch} validation",
        )
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "steps": epoch_steps,
                "train_masked_pose_l1_metres": (
                    global_coordinate_error_sum / global_coordinate_count
                ),
                "mean_preclip_gradient_norm": (
                    global_gradient_norm_sum / global_epoch_steps
                ),
                "validation": validation,
            }
        )
        if not distributed.enabled and epoch_steps == len(train_loader):
            try:
                save_epoch_training_state(
                    writer,
                    model_state=model.state_dict(),
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
                raise OmniHandRunError(str(error)) from error
        if stop:
            break
    return history, global_step, global_step - initial_global_step, resumed_from


def _save_checkpoint(model: OmniHandCubeNet, destination: Path) -> str:
    if destination.exists():
        raise OmniHandRunError(f"checkpoint already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    try:
        save_file(state, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, RuntimeError, ValueError) as error:
        raise OmniHandRunError(f"unable to save OmniHand checkpoint: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(destination)


def _checkpoint_payload(
    *,
    writer: RunArtifactWriter,
    config: OmniHandRunConfig,
    coordinate_frame: str,
    weights_sha256: str,
    global_step: int,
    epochs_executed: int,
    model: OmniHandCubeNet,
    runtime_report: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    runtime_payload: Mapping[str, object],
    model_state_sha256: str,
) -> dict[str, object]:
    git = cast(Mapping[str, Any], runtime_report["git"])
    return {
        "schema_version": OMNIHAND_CHECKPOINT_SCHEMA,
        "run_id": writer.run_id,
        "weights": {
            "filename": "checkpoint.safetensors",
            "sha256": weights_sha256,
            "format": "safetensors",
        },
        "model": config.model.to_dict(),
        "model_config_sha256": config.model_fingerprint,
        "task_config_sha256": config.fingerprint,
        "coordinate_frame": coordinate_frame,
        "pose_units": "m",
        "global_step": global_step,
        "epochs_executed": epochs_executed,
        "selection": "final_step",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_state_sha256": model_state_sha256,
        "runtime": dict(runtime_payload),
        "git_commit": git["commit"],
        "input_sha256": dict(sorted(input_hashes.items())),
    }


def _load_checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OmniHandRunError(f"unable to read checkpoint metadata: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != OMNIHAND_CHECKPOINT_SCHEMA
    ):
        raise OmniHandRunError("checkpoint metadata has an unsupported schema")
    return payload


def _load_checkpoint(
    model: OmniHandCubeNet,
    *,
    weights_path: Path,
    metadata_path: Path,
    config: OmniHandRunConfig,
    coordinate_frame: str,
) -> str:
    metadata = _load_checkpoint_metadata(metadata_path)
    weights = metadata.get("weights")
    if not isinstance(weights, Mapping):
        raise OmniHandRunError("checkpoint metadata is missing weights provenance")
    expected_sha256 = weights.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise OmniHandRunError("checkpoint metadata has an invalid weights SHA-256")
    if not _SHA256.fullmatch(expected_sha256):
        raise OmniHandRunError("checkpoint metadata has an invalid weights SHA-256")
    if weights.get("format") != "safetensors":
        raise OmniHandRunError("checkpoint metadata must declare safetensors weights")
    if metadata.get("pose_units") != "m":
        raise OmniHandRunError("checkpoint pose units must be metres")
    observed_sha256 = sha256_file(weights_path)
    if observed_sha256 != expected_sha256:
        raise OmniHandRunError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    if metadata.get("model_config_sha256") != config.model_fingerprint:
        raise OmniHandRunError("checkpoint model configuration does not match the run config")
    if metadata.get("coordinate_frame") != coordinate_frame:
        raise OmniHandRunError("checkpoint and evaluation manifest coordinate frames do not match")
    try:
        state = load_file(weights_path, device="cpu")
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise OmniHandRunError(f"unable to load OmniHand checkpoint: {error}") from error
    return observed_sha256


def _prediction_records(
    model: OmniHandCubeNet,
    loader: DataLoader[PoseReconstructionBatch],
    accumulator: PoseMetricAccumulator,
    *,
    device: torch.device,
    precision: str,
    checkpoint_sha256: str,
    save_targets: bool,
) -> Iterator[Mapping[str, object]]:
    model.eval()
    with torch.inference_mode():
        for numpy_batch in loader:
            batch = _tensor_batch(numpy_batch, device)
            with _precision_context(device, precision):
                prediction = model(batch.radar_cube, batch.frame_mask).joints.float()
            target = batch.pose_target.float()
            per_sample = accumulator.update(prediction, target, batch.pose_valid)
            prediction_cpu = prediction.cpu()
            target_cpu = target.cpu()
            valid_cpu = batch.pose_valid.cpu()
            for index, sample_id in enumerate(batch.sample_ids):
                record: dict[str, object] = {
                    "schema_version": OMNIHAND_PREDICTION_SCHEMA,
                    "sample_id": sample_id,
                    "coordinate_frame": batch.coordinate_frame,
                    "pose_units": "m",
                    "checkpoint_sha256": checkpoint_sha256,
                    "absolute_mpjpe_mm": float(per_sample[index].cpu().item()),
                    "prediction_m": prediction_cpu[index].tolist(),
                    "valid_joint_mask": valid_cpu[index].tolist(),
                }
                if save_targets:
                    record["target_m"] = target_cpu[index].tolist()
                yield record


def _resolved_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _prepare_run(
    experiment_config: ExperimentConfig,
    task_config: OmniHandRunConfig,
    *,
    source_experiment_config: str | Path,
    source_task_config: str | Path,
    input_specs: Sequence[tuple[str, str, str | Path]],
    project_root: Path,
    command: Sequence[str],
    runtime_report: Mapping[str, Any] | None,
    created_at: datetime | None,
    distributed: DistributedContext,
) -> tuple[RunArtifactWriter, dict[str, Path], dict[str, str], Mapping[str, Any]]:
    root = project_root.expanduser().resolve()
    if experiment_config.task is not Task.POSE_RECONSTRUCTION:
        raise OmniHandRunError("OmniHand runs require task=pose_reconstruction")
    if not task_config.data.verify_checksums:
        raise OmniHandRunError("formal OmniHand runs require data.verify_checksums=true")

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
        task_config_path = _resolved_path(source_task_config, root)
        if load_omnihand_run_config(task_config_path).fingerprint != task_config.fingerprint:
            raise OmniHandRunError(
                "source OmniHand configuration does not match the loaded config"
            )
        run_inputs = [
            RunInput.capture(name="omnihand_config", kind="config", path=task_config_path),
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
            writer.write_json_artifact("omnihand.resolved.json", task_config.to_dict())
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
        raise OmniHandRunError("distributed run initialization returned invalid metadata")
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
        raise OmniHandRunError("distributed run initialization metadata is incomplete")
    paths = {str(name): Path(str(path)) for name, path in raw_paths.items()}
    hashes = {str(name): str(value) for name, value in raw_hashes.items()}
    return RunArtifactWriter(Path(run_dir), run_id), paths, hashes, report


def _finalize_failed_run(writer: RunArtifactWriter, error: BaseException) -> None:
    with suppress(Exception):
        writer.finalize(status="failed", failure=f"{type(error).__name__}: {error}")


def train_omnihand(
    experiment_config: ExperimentConfig,
    task_config: OmniHandRunConfig,
    *,
    source_experiment_config: str | Path,
    source_task_config: str | Path,
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
        raise OmniHandRunError("resume requires both state metadata and Safetensors")
    root = project_root.expanduser().resolve()
    resolved_experiment = experiment_config.resolved(root)
    try:
        distributed = DistributedContext.from_environment(resolved_experiment.runtime)
    except DistributedRunError as error:
        raise OmniHandRunError(str(error)) from error
    if distributed.enabled and resume_state_metadata_path is not None:
        raise OmniHandRunError(
            "DDP resume is unsupported until every rank's RNG and sampler state is captured"
        )
    try:
        distributed.initialize()
    except DistributedRunError as error:
        raise OmniHandRunError(str(error)) from error
    resume_inputs: tuple[tuple[str, str, str | Path], ...] = ()
    if resume_state_metadata_path is not None and resume_state_tensors_path is not None:
        resume_inputs = (
            ("resume_state_metadata", "checkpoint", resume_state_metadata_path),
            ("resume_state_tensors", "checkpoint", resume_state_tensors_path),
        )
    try:
        writer, paths, input_hashes, report = _prepare_run(
            experiment_config,
            task_config,
            source_experiment_config=source_experiment_config,
            source_task_config=source_task_config,
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
        train_manifest = PoseReconstructionManifest(
            paths["train_manifest"],
            data_root=resolved_experiment.paths.data_root,
            verify_checksums=task_config.data.verify_checksums,
        )
        validation_manifest = PoseReconstructionManifest(
            paths["validation_manifest"],
            data_root=resolved_experiment.paths.data_root,
            verify_checksums=task_config.data.verify_checksums,
        )
        _validate_manifest_for_model(train_manifest, task_config, role="train")
        _validate_manifest_for_model(validation_manifest, task_config, role="validation")
        overlap = {record.sample_id for record in train_manifest.records} & {
            record.sample_id for record in validation_manifest.records
        }
        if overlap:
            raise OmniHandRunError(
                f"train and validation manifests overlap on {len(overlap)} sample IDs"
            )
        if train_manifest.coordinate_frame != validation_manifest.coordinate_frame:
            raise OmniHandRunError("train and validation coordinate frames do not match")
        if train_manifest.radar_spatial_shape != validation_manifest.radar_spatial_shape:
            raise OmniHandRunError("train and validation radar spatial shapes do not match")
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
        _seed_runtime(resolved_experiment.runtime.seed, resolved_experiment.runtime.deterministic)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = _build_model(task_config).to(device)
        distributed_topology = distributed.topology_payload()
        actual_runtime = _runtime_payload(
            model, resolved_experiment.runtime, device, distributed_topology
        )
        distributed.rank_zero_call(
            lambda: str(
                writer.write_json_artifact("omnihand.runtime.json", actual_runtime)
            ),
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
            "task": "pose_reconstruction",
            "training_config_sha256": task_config.training_fingerprint,
            "model_config_sha256": task_config.model_fingerprint,
            "train_manifest_sha256": input_hashes["train_manifest"],
            "validation_manifest_sha256": input_hashes["validation_manifest"],
            "split_assignments_sha256": input_hashes["split_assignments"],
            "coordinate_frame": train_manifest.coordinate_frame,
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

        model_state_sha256 = distributed.assert_consistent_state(model.state_dict())
        weights_path = writer.artifact_path("checkpoint.safetensors")

        def publish_checkpoint() -> str:
            weights_sha256 = _save_checkpoint(model, weights_path)
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

        accumulator = PoseMetricAccumulator(
            pck_threshold_mm=task_config.evaluation.pck_threshold_mm
        )
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
            prediction_schema=OMNIHAND_PREDICTION_SCHEMA,
            rank=distributed.rank,
            world_size=distributed.world_size,
            records=_prediction_records(
                model,
                prediction_loader,
                accumulator,
                device=device,
                precision=resolved_experiment.runtime.precision,
                checkpoint_sha256=weights_sha256,
                save_targets=task_config.evaluation.save_targets,
            ),
        )
        local_prediction_seconds = time.perf_counter() - prediction_started
        distributed.barrier()
        distributed.rank_zero_call(
            lambda: aggregate_prediction_shards(
                writer,
                prediction_schema=OMNIHAND_PREDICTION_SCHEMA,
                world_size=distributed.world_size,
                expected_sample_ids=(
                    record.sample_id for record in validation_manifest.records
                ),
            ).record_count,
            stage="prediction aggregation",
        )
        distributed.barrier()
        prediction_seconds = distributed.max_value(local_prediction_seconds)
        merged_accumulator = PoseMetricAccumulator(
            pck_threshold_mm=task_config.evaluation.pck_threshold_mm
        )
        for state in distributed.all_gather_object(accumulator.state_dict()):
            merged_accumulator.merge_state(state)
        metrics = merged_accumulator.values()
        metric_values: dict[str, int | float] = {
            **metrics,
            "pck_threshold_mm": task_config.evaluation.pck_threshold_mm,
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
        end_to_end_seconds = distributed.max_value(time.perf_counter() - run_started)

        def finalize_run() -> dict[str, int | float]:
            writer.write_json_artifact(
                "history.json",
                {
                    "schema_version": OMNIHAND_HISTORY_SCHEMA,
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
                    "schema_version": OMNIHAND_PERFORMANCE_SCHEMA,
                    "mode": "train",
                    "device": str(device),
                    "precision": resolved_experiment.runtime.precision,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
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
                protocol_id=POSE_METRIC_PROTOCOL,
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
        if isinstance(error, OmniHandRunError):
            raise
        raise OmniHandRunError(f"OmniHand training failed: {error}") from error
    finally:
        distributed.close()
    return {
        "schema_version": OMNIHAND_RUN_RESULT_SCHEMA,
        "mode": "train",
        "status": "completed",
        "run_id": writer.run_id,
        "run_dir": str(writer.run_dir),
        "metrics": metric_values,
    }


def evaluate_omnihand(
    experiment_config: ExperimentConfig,
    task_config: OmniHandRunConfig,
    *,
    source_experiment_config: str | Path,
    source_task_config: str | Path,
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
        raise OmniHandRunError("evaluation split must be train, validation, or test")
    root = project_root.expanduser().resolve()
    resolved_experiment = experiment_config.resolved(root)
    try:
        distributed = DistributedContext.from_environment(resolved_experiment.runtime)
        distributed.initialize()
    except DistributedRunError as error:
        raise OmniHandRunError(str(error)) from error
    try:
        writer, paths, _, _ = _prepare_run(
            experiment_config,
            task_config,
            source_experiment_config=source_experiment_config,
            source_task_config=source_task_config,
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
        manifest = PoseReconstructionManifest(
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
        _seed_runtime(resolved_experiment.runtime.seed, resolved_experiment.runtime.deterministic)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = _build_model(task_config)
        checkpoint_sha256 = _load_checkpoint(
            model,
            weights_path=paths["checkpoint_weights"],
            metadata_path=paths["checkpoint_metadata"],
            config=task_config,
            coordinate_frame=manifest.coordinate_frame,
        )
        model.to(device)
        distributed_topology = distributed.topology_payload()
        actual_runtime = _runtime_payload(
            model, resolved_experiment.runtime, device, distributed_topology
        )
        distributed.rank_zero_call(
            lambda: str(
                writer.write_json_artifact("omnihand.runtime.json", actual_runtime)
            ),
            stage="runtime artifact publication",
        )
        model_state_sha256 = distributed.assert_consistent_state(model.state_dict())
        loader = _loader(
            manifest,
            task_config,
            shuffle=False,
            seed=resolved_experiment.runtime.seed,
            device=device,
            distributed=distributed,
            exact_distributed_coverage=True,
        )
        accumulator = PoseMetricAccumulator(
            pck_threshold_mm=task_config.evaluation.pck_threshold_mm
        )
        prediction_started = time.perf_counter()
        write_prediction_shard(
            writer.run_dir,
            run_id=writer.run_id,
            prediction_schema=OMNIHAND_PREDICTION_SCHEMA,
            rank=distributed.rank,
            world_size=distributed.world_size,
            records=_prediction_records(
                model,
                loader,
                accumulator,
                device=device,
                precision=resolved_experiment.runtime.precision,
                checkpoint_sha256=checkpoint_sha256,
                save_targets=task_config.evaluation.save_targets,
            ),
        )
        local_prediction_seconds = time.perf_counter() - prediction_started
        distributed.barrier()
        distributed.rank_zero_call(
            lambda: aggregate_prediction_shards(
                writer,
                prediction_schema=OMNIHAND_PREDICTION_SCHEMA,
                world_size=distributed.world_size,
                expected_sample_ids=(record.sample_id for record in manifest.records),
            ).record_count,
            stage="prediction aggregation",
        )
        distributed.barrier()
        prediction_seconds = distributed.max_value(local_prediction_seconds)
        merged_accumulator = PoseMetricAccumulator(
            pck_threshold_mm=task_config.evaluation.pck_threshold_mm
        )
        for state in distributed.all_gather_object(accumulator.state_dict()):
            merged_accumulator.merge_state(state)
        metrics = merged_accumulator.values()
        metric_values: dict[str, int | float] = {
            **metrics,
            "pck_threshold_mm": task_config.evaluation.pck_threshold_mm,
        }
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
                protocol_id=POSE_METRIC_PROTOCOL,
                split=split,
                values=metric_values,
                sample_count=merged_accumulator.sample_count,
            )
            writer.write_json_artifact(
                "performance.json",
                {
                    "schema_version": OMNIHAND_PERFORMANCE_SCHEMA,
                    "mode": "evaluate",
                    "device": str(device),
                    "precision": resolved_experiment.runtime.precision,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
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
        if isinstance(error, OmniHandRunError):
            raise
        raise OmniHandRunError(f"OmniHand evaluation failed: {error}") from error
    finally:
        distributed.close()
    return {
        "schema_version": OMNIHAND_RUN_RESULT_SCHEMA,
        "mode": "evaluate",
        "status": "completed",
        "run_id": writer.run_id,
        "run_dir": str(writer.run_dir),
        "metrics": metric_values,
    }
