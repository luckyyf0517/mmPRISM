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

from mmprism.artifacts import RunArtifactWriter, RunInput, sha256_file
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
from mmprism.training.omnihand_run_config import (
    OmniHandRunConfig,
    OmniHandRunError,
    load_omnihand_run_config,
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


def _resolve_device(runtime: RuntimeConfig) -> torch.device:
    accelerator = runtime.accelerator.lower()
    if accelerator not in {"auto", "cpu", "cuda", "gpu"}:
        raise OmniHandRunError("OmniHand runtime.accelerator must be auto, cpu, cuda, or gpu")
    if isinstance(runtime.devices, str) and runtime.devices != "auto":
        raise OmniHandRunError("OmniHand runtime.devices supports only auto or one device index")
    if isinstance(runtime.devices, tuple) and len(runtime.devices) != 1:
        raise OmniHandRunError("OmniHand v1 formal runs currently require exactly one device")

    if accelerator == "cpu":
        if isinstance(runtime.devices, tuple):
            raise OmniHandRunError("CPU runs cannot select a CUDA device index")
        device = torch.device("cpu")
    else:
        cuda_requested = accelerator in {"cuda", "gpu"} or isinstance(runtime.devices, tuple)
        if not torch.cuda.is_available():
            if cuda_requested:
                raise OmniHandRunError("CUDA was requested but is unavailable")
            device = torch.device("cpu")
        else:
            index = runtime.devices[0] if isinstance(runtime.devices, tuple) else 0
            if index >= torch.cuda.device_count():
                raise OmniHandRunError(f"CUDA device index {index} is unavailable")
            device = torch.device("cuda", index)

    if device.type == "cpu" and runtime.precision != "32-true":
        raise OmniHandRunError("CPU OmniHand runs require runtime.precision=32-true")
    if (
        device.type == "cuda"
        and runtime.precision == "bf16-mixed"
        and not torch.cuda.is_bf16_supported()
    ):
        raise OmniHandRunError("the selected CUDA device does not support bfloat16")
    return device


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
) -> DataLoader[PoseReconstructionBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        _PoseDataset(manifest),
        batch_size=config.data.batch_size,
        shuffle=shuffle,
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
    train_loader: DataLoader[PoseReconstructionBatch],
    validation_loader: DataLoader[PoseReconstructionBatch],
    config: OmniHandRunConfig,
    *,
    device: torch.device,
    precision: str,
) -> tuple[list[dict[str, object]], int]:
    optimization = config.optimization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        betas=(optimization.beta1, optimization.beta2),
        weight_decay=optimization.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and precision == "16-mixed")
    history: list[dict[str, object]] = []
    global_step = 0
    stop = False
    for epoch in range(1, optimization.epochs + 1):
        model.train()
        coordinate_error_sum = 0.0
        coordinate_count = 0
        gradient_norm_sum = 0.0
        epoch_steps = 0
        for numpy_batch in train_loader:
            batch = _tensor_batch(numpy_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(device, precision):
                prediction = model(batch.radar_cube, batch.frame_mask).joints
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
        validation = _evaluate_summary(
            model,
            validation_loader,
            device=device,
            precision=precision,
            pck_threshold_mm=config.evaluation.pck_threshold_mm,
        )
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "steps": epoch_steps,
                "train_masked_pose_l1_metres": coordinate_error_sum / coordinate_count,
                "mean_preclip_gradient_norm": gradient_norm_sum / epoch_steps,
                "validation": validation,
            }
        )
        if stop:
            break
    return history, global_step


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
) -> tuple[RunArtifactWriter, dict[str, Path], dict[str, str], Mapping[str, Any]]:
    root = project_root.expanduser().resolve()
    if experiment_config.task is not Task.POSE_RECONSTRUCTION:
        raise OmniHandRunError("OmniHand runs require task=pose_reconstruction")
    if not task_config.data.verify_checksums:
        raise OmniHandRunError("formal OmniHand runs require data.verify_checksums=true")
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
        raise OmniHandRunError("source OmniHand configuration does not match the loaded config")
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
    hashes = {item.name: item.sha256 for item in run_inputs}
    try:
        writer.write_json_artifact("omnihand.resolved.json", task_config.to_dict())
    except Exception as error:
        _finalize_failed_run(writer, error)
        raise
    return writer, paths, hashes, report


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
    project_root: Path,
    command: Sequence[str],
    runtime_report: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    run_started = time.perf_counter()
    writer, paths, input_hashes, report = _prepare_run(
        experiment_config,
        task_config,
        source_experiment_config=source_experiment_config,
        source_task_config=source_task_config,
        input_specs=(
            ("train_manifest", "manifest", train_manifest_path),
            ("validation_manifest", "manifest", validation_manifest_path),
        ),
        project_root=project_root,
        command=command,
        runtime_report=runtime_report,
        created_at=created_at,
    )
    try:
        resolved_experiment = experiment_config.resolved(project_root.expanduser().resolve())
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

        device = _resolve_device(resolved_experiment.runtime)
        _seed_runtime(resolved_experiment.runtime.seed, resolved_experiment.runtime.deterministic)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = _build_model(task_config).to(device)
        actual_runtime = _runtime_payload(model, resolved_experiment.runtime, device)
        writer.write_json_artifact("omnihand.runtime.json", actual_runtime)
        train_loader = _loader(
            train_manifest,
            task_config,
            shuffle=task_config.data.shuffle,
            seed=resolved_experiment.runtime.seed,
            device=device,
        )
        validation_loader = _loader(
            validation_manifest,
            task_config,
            shuffle=False,
            seed=resolved_experiment.runtime.seed,
            device=device,
        )
        training_started = time.perf_counter()
        history, global_step = _train_model(
            model,
            train_loader,
            validation_loader,
            task_config,
            device=device,
            precision=resolved_experiment.runtime.precision,
        )
        training_seconds = time.perf_counter() - training_started

        weights_path = writer.artifact_path("checkpoint.safetensors")
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
            ),
        )

        accumulator = PoseMetricAccumulator(
            pck_threshold_mm=task_config.evaluation.pck_threshold_mm
        )
        prediction_started = time.perf_counter()
        writer.write_jsonl_artifact(
            "predictions.jsonl",
            _prediction_records(
                model,
                validation_loader,
                accumulator,
                device=device,
                precision=resolved_experiment.runtime.precision,
                checkpoint_sha256=weights_sha256,
                save_targets=task_config.evaluation.save_targets,
            ),
        )
        prediction_seconds = time.perf_counter() - prediction_started
        metrics = accumulator.values()
        metric_values: dict[str, int | float] = {
            **metrics,
            "pck_threshold_mm": task_config.evaluation.pck_threshold_mm,
            "global_step": global_step,
            "epochs_executed": len(history),
        }
        writer.write_json_artifact(
            "history.json",
            {
                "schema_version": OMNIHAND_HISTORY_SCHEMA,
                "run_id": writer.run_id,
                "task_config_sha256": task_config.fingerprint,
                "global_step": global_step,
                "epochs_executed": len(history),
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
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "optimizer_steps": global_step,
                "training_seconds": training_seconds,
                "optimizer_steps_per_second": global_step / training_seconds,
                "prediction_samples": accumulator.sample_count,
                "prediction_seconds": prediction_seconds,
                "prediction_samples_per_second": accumulator.sample_count / prediction_seconds,
                "end_to_end_seconds": time.perf_counter() - run_started,
                "cuda_memory": _cuda_memory_payload(device),
            },
        )
        writer.write_metrics(
            protocol_id=POSE_METRIC_PROTOCOL,
            split="validation",
            values=metric_values,
            sample_count=accumulator.sample_count,
        )
        writer.finalize(status="completed")
    except KeyboardInterrupt as error:
        writer.finalize(status="aborted", failure="interrupted by operator")
        raise error
    except Exception as error:
        _finalize_failed_run(writer, error)
        if isinstance(error, OmniHandRunError):
            raise
        raise OmniHandRunError(f"OmniHand training failed: {error}") from error
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
    split: str,
    project_root: Path,
    command: Sequence[str],
    runtime_report: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    run_started = time.perf_counter()
    if split not in {"train", "validation", "test"}:
        raise OmniHandRunError("evaluation split must be train, validation, or test")
    writer, paths, _, _ = _prepare_run(
        experiment_config,
        task_config,
        source_experiment_config=source_experiment_config,
        source_task_config=source_task_config,
        input_specs=(
            ("evaluation_manifest", "manifest", manifest_path),
            ("checkpoint_weights", "checkpoint", checkpoint_path),
            ("checkpoint_metadata", "checkpoint", checkpoint_metadata_path),
        ),
        project_root=project_root,
        command=command,
        runtime_report=runtime_report,
        created_at=created_at,
    )
    try:
        resolved_experiment = experiment_config.resolved(project_root.expanduser().resolve())
        manifest = PoseReconstructionManifest(
            paths["evaluation_manifest"],
            data_root=resolved_experiment.paths.data_root,
            verify_checksums=task_config.data.verify_checksums,
        )
        _validate_manifest_for_model(manifest, task_config, role="evaluation")
        device = _resolve_device(resolved_experiment.runtime)
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
        writer.write_json_artifact(
            "omnihand.runtime.json",
            _runtime_payload(model, resolved_experiment.runtime, device),
        )
        loader = _loader(
            manifest,
            task_config,
            shuffle=False,
            seed=resolved_experiment.runtime.seed,
            device=device,
        )
        accumulator = PoseMetricAccumulator(
            pck_threshold_mm=task_config.evaluation.pck_threshold_mm
        )
        prediction_started = time.perf_counter()
        writer.write_jsonl_artifact(
            "predictions.jsonl",
            _prediction_records(
                model,
                loader,
                accumulator,
                device=device,
                precision=resolved_experiment.runtime.precision,
                checkpoint_sha256=checkpoint_sha256,
                save_targets=task_config.evaluation.save_targets,
            ),
        )
        prediction_seconds = time.perf_counter() - prediction_started
        metrics = accumulator.values()
        metric_values: dict[str, int | float] = {
            **metrics,
            "pck_threshold_mm": task_config.evaluation.pck_threshold_mm,
        }
        writer.write_metrics(
            protocol_id=POSE_METRIC_PROTOCOL,
            split=split,
            values=metric_values,
            sample_count=accumulator.sample_count,
        )
        writer.write_json_artifact(
            "performance.json",
            {
                "schema_version": OMNIHAND_PERFORMANCE_SCHEMA,
                "mode": "evaluate",
                "device": str(device),
                "precision": resolved_experiment.runtime.precision,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "prediction_samples": accumulator.sample_count,
                "prediction_seconds": prediction_seconds,
                "prediction_samples_per_second": accumulator.sample_count / prediction_seconds,
                "end_to_end_seconds": time.perf_counter() - run_started,
                "cuda_memory": _cuda_memory_payload(device),
            },
        )
        writer.finalize(status="completed")
    except KeyboardInterrupt as error:
        writer.finalize(status="aborted", failure="interrupted by operator")
        raise error
    except Exception as error:
        _finalize_failed_run(writer, error)
        if isinstance(error, OmniHandRunError):
            raise
        raise OmniHandRunError(f"OmniHand evaluation failed: {error}") from error
    return {
        "schema_version": OMNIHAND_RUN_RESULT_SCHEMA,
        "mode": "evaluate",
        "status": "completed",
        "run_id": writer.run_id,
        "run_dir": str(writer.run_dir),
        "metrics": metric_values,
    }
