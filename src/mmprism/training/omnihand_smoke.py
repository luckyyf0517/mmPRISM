from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from mmprism.evaluation.pose import (
    POSE_METRIC_PROTOCOL,
    masked_pose_l1_metres,
    pose_metric_tensors,
)
from mmprism.models.cubenet import (
    CubeNetSpatialEncoder,
    OmniHandCubeNet,
    TemporalTransformerAggregator,
)
from mmprism.training.omnihand_config import OmniHandSmokeConfig, OmniHandSmokeError

OMNIHAND_SMOKE_REPORT_SCHEMA = "mmprism.omnihand_vertical_smoke.v1"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _strict_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise OmniHandSmokeError(f"OmniHand smoke report is not strict JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def write_omnihand_smoke_report(payload: Mapping[str, object], destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(_strict_json_bytes(payload))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except RuntimeError as error:
        raise OmniHandSmokeError(f"Invalid torch device {device!r}: {error}") from error
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise OmniHandSmokeError("CUDA was requested but is not available")
    return resolved


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    raise OmniHandSmokeError(f"dtype {name} is not supported on device {device}")


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def _tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _batch_fingerprint(batch: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(batch):
        digest.update(name.encode("ascii"))
        digest.update(_tensor_sha256(batch[name]).encode("ascii"))
    return digest.hexdigest()


def _build_batch(
    config: OmniHandSmokeConfig,
    *,
    step: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    batch_size = config.batch.batch_size
    frames = config.batch.frame_count
    channels = config.model.spatial.in_channels
    range_bins, azimuth_bins, elevation_bins = config.batch.spatial_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.runtime.seed + step)
    radar_cube = torch.rand(
        batch_size,
        frames,
        channels,
        range_bins,
        azimuth_bins,
        elevation_bins,
        generator=generator,
        dtype=torch.float32,
    )
    target = (
        torch.rand(
            batch_size,
            2,
            config.model.joint_count,
            config.model.coordinate_dim,
            generator=generator,
            dtype=torch.float32,
        )
        - 0.5
    ) * 0.4
    frame_mask = torch.ones(batch_size, frames, dtype=torch.bool)
    if batch_size > 1 and frames > 1:
        frame_mask[-1, -1] = False
        radar_cube[-1, -1] = 0
    valid_joints = torch.ones(batch_size, 2, config.model.joint_count, dtype=torch.bool)
    if batch_size > 1:
        valid_joints[-1, 0, -1] = False
    return {
        "radar_cube": radar_cube.to(device=device, dtype=dtype),
        "frame_mask": frame_mask.to(device),
        "target": target.to(device=device, dtype=dtype),
        "valid_joints": valid_joints.to(device),
    }


def build_omnihand_model(config: OmniHandSmokeConfig) -> OmniHandCubeNet:
    spatial = config.model.spatial
    temporal = config.model.temporal
    spatial_encoder = CubeNetSpatialEncoder(
        in_channels=spatial.in_channels,
        stem_channels=spatial.stem_channels,
        stage_channels=spatial.stage_channels,
        stage_depths=spatial.stage_depths,
        channel_attention=spatial.channel_attention,
        spatial_attention=spatial.spatial_attention,
        se_attention=spatial.se_attention,
        use_pafpn=spatial.use_pafpn,
        fpn_channels=spatial.fpn_channels,
    )
    temporal_aggregator = TemporalTransformerAggregator(
        spatial_encoder.feature_dim,
        max_frames=temporal.max_frames,
        layers=temporal.layers,
        heads=temporal.heads,
        feedforward_dim=temporal.feedforward_dim,
        dropout=temporal.dropout,
    )
    return OmniHandCubeNet(
        spatial_encoder,
        temporal_aggregator,
        joint_count=config.model.joint_count,
        coordinate_dim=config.model.coordinate_dim,
    )


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _tracked_parameters(model: OmniHandCubeNet) -> dict[str, torch.nn.Parameter]:
    requested = {
        "spatial_encoder.stem.conv.weight",
        "temporal_aggregator.encoder.layers.0.self_attn.in_proj_weight",
        "pose_head.1.weight",
    }
    parameters: dict[str, torch.nn.Parameter] = {
        name: parameter for name, parameter in model.named_parameters() if name in requested
    }
    if set(parameters) != requested:
        missing = sorted(requested - set(parameters))
        raise OmniHandSmokeError(f"Unable to track expected model parameters: {missing}")
    return parameters


def _gradient_norms(model: OmniHandCubeNet) -> dict[str, float]:
    totals = {"spatial_encoder": 0.0, "temporal_aggregator": 0.0, "pose_head": 0.0}
    for name, parameter in model.named_parameters():
        prefix = name.partition(".")[0]
        if prefix in totals and parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            totals[prefix] += float(torch.sum(gradient * gradient).item())
    norms = {name: math.sqrt(value) for name, value in totals.items()}
    if any(not math.isfinite(value) or value <= 0 for value in norms.values()):
        raise OmniHandSmokeError(f"Non-finite or zero model gradient norm: {norms}")
    return norms


def _runtime_payload(
    runtime_report: Mapping[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, object]:
    git = runtime_report.get("git")
    git_payload = git if isinstance(git, Mapping) else {}
    packages = runtime_report.get("packages")
    package_payload = packages if isinstance(packages, Mapping) else {}
    device_payload: dict[str, object] = {"requested_type": device.type, "resolved": str(device)}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_payload.update(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    return {
        "python": runtime_report.get("python"),
        "platform": runtime_report.get("platform"),
        "git": {
            "commit": git_payload.get("commit"),
            "dirty": git_payload.get("dirty"),
        },
        "packages": {
            "numpy": package_payload.get("numpy"),
            "torch": package_payload.get("torch") or _package_version("torch"),
        },
        "torch_cuda": torch.version.cuda,
        "dtype": str(dtype).removeprefix("torch."),
        "device": device_payload,
    }


def _float(value: Tensor) -> float:
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise OmniHandSmokeError("OmniHand smoke produced a non-finite scalar")
    return result


def run_omnihand_smoke(
    config: OmniHandSmokeConfig,
    *,
    device: str,
    runtime_report: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, object]:
    git = runtime_report.get("git")
    if not isinstance(git, Mapping) or not isinstance(git.get("commit"), str):
        raise OmniHandSmokeError("OmniHand smoke requires Git commit provenance")
    if git.get("dirty") is not False:
        raise OmniHandSmokeError("OmniHand smoke requires a clean Git worktree")
    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(config.runtime.dtype, resolved_device)
    _seed_everything(config.runtime.seed, config.runtime.deterministic)
    model = build_omnihand_model(config).to(device=resolved_device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        betas=(config.optimization.beta1, config.optimization.beta2),
    )
    tracked = _tracked_parameters(model)
    initial_parameters = {
        name: parameter.detach().float().cpu().clone() for name, parameter in tracked.items()
    }
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    started = time.perf_counter()
    steps: list[dict[str, object]] = []
    final_batch: dict[str, Tensor] | None = None
    for step in range(config.optimization.steps):
        batch = _build_batch(config, step=step, device=resolved_device, dtype=dtype)
        final_batch = batch
        optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(device_type="cuda", dtype=dtype)
            if resolved_device.type == "cuda" and dtype is torch.bfloat16
            else nullcontext()
        ):
            output = model.forward(batch["radar_cube"], batch["frame_mask"])
            loss = masked_pose_l1_metres(output.joints, batch["target"], batch["valid_joints"])
        if not bool(torch.isfinite(loss)):
            raise OmniHandSmokeError(f"Non-finite pose loss at step {step}")
        torch.autograd.backward(loss)
        gradient_norms = _gradient_norms(model)
        optimizer.step()
        metric_tensors = pose_metric_tensors(
            output.joints.detach(),
            batch["target"],
            batch["valid_joints"],
            pck_threshold_mm=config.metrics.pck_threshold_mm,
        )
        steps.append(
            {
                "step": step,
                "input_batch_sha256": _batch_fingerprint(batch),
                "loss_l1_metres": _float(loss),
                "absolute_mpjpe_mm": _float(metric_tensors["absolute_mpjpe_mm"]),
                "root_relative_mpjpe_mm": _float(metric_tensors["root_relative_mpjpe_mm"]),
                "root_relative_pck": _float(metric_tensors["root_relative_pck"]),
                "gradient_norms": gradient_norms,
                "joints_shape": list(output.joints.shape),
                "frame_feature_shape": list(output.frame_features.shape),
                "sequence_feature_shape": list(output.sequence_features.shape),
            }
        )
    elapsed = time.perf_counter() - started
    if final_batch is None:
        raise OmniHandSmokeError("OmniHand smoke executed no optimization steps")

    parameter_deltas = {
        name: float(
            torch.max(torch.abs(parameter.detach().float().cpu() - initial_parameters[name])).item()
        )
        for name, parameter in tracked.items()
    }
    if any(not math.isfinite(value) or value <= 0 for value in parameter_deltas.values()):
        raise OmniHandSmokeError(f"Expected tracked model parameters to update: {parameter_deltas}")

    model.eval()
    with torch.inference_mode():
        sequence_output = model.forward(final_batch["radar_cube"], final_batch["frame_mask"])
        changed_padding = final_batch["radar_cube"].clone()
        invalid_frames = ~final_batch["frame_mask"].bool()
        changed_padding[invalid_frames] = 1000.0
        changed_output = model.forward(changed_padding, final_batch["frame_mask"])
        padding_difference = torch.max(
            torch.abs(sequence_output.joints.float() - changed_output.joints.float())
        )
        single_frame_output = model.forward_single_frame(final_batch["radar_cube"][:, 0])
        final_metrics = pose_metric_tensors(
            sequence_output.joints,
            final_batch["target"],
            final_batch["valid_joints"],
            pck_threshold_mm=config.metrics.pck_threshold_mm,
        )
    padding_max_abs_difference = _float(padding_difference)
    if padding_max_abs_difference > 1e-3:
        raise OmniHandSmokeError(
            f"masked temporal padding changed the pose prediction: {padding_max_abs_difference}"
        )
    if not bool(torch.all(torch.isfinite(single_frame_output.joints))):
        raise OmniHandSmokeError("single-frame path produced non-finite joints")

    per_sample = final_metrics["per_sample_absolute_mpjpe_mm"]
    sample_metrics = [
        {
            "sample_id": f"synthetic_pose_smoke_{index:02d}",
            "absolute_mpjpe_mm": _float(per_sample[index]),
        }
        for index in range(config.batch.batch_size)
    ]
    cuda_memory: dict[str, int] | None = None
    if resolved_device.type == "cuda":
        cuda_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(resolved_device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(resolved_device),
        }
    return {
        "schema_version": OMNIHAND_SMOKE_REPORT_SCHEMA,
        "smoke_id": config.smoke_id,
        "status": "passed",
        "scope": "engineering_vertical_smoke_not_paper_evidence",
        "config_fingerprint": config.fingerprint,
        "command": list(command),
        "runtime": _runtime_payload(runtime_report, resolved_device, dtype),
        "architecture": {
            "input_axes": [
                "batch",
                "time",
                "doppler",
                "range",
                "azimuth",
                "elevation",
            ],
            "output_axes": ["batch", "hand", "joint", "coordinate"],
            "output_units": "m",
            "depthwise_separable_residual": True,
            "pafpn": config.model.spatial.use_pafpn,
            "attention": {
                "channel": config.model.spatial.channel_attention,
                "spatial": config.model.spatial.spatial_attention,
                "se": config.model.spatial.se_attention,
            },
            "temporal_aggregation": "mask_aware_transformer_cls_mean_attention_v1",
        },
        "parameter_count": _parameter_counts(model),
        "training": {
            "objective": "masked_coordinate_l1_metres_v1",
            "steps": steps,
            "parameter_max_abs_delta": parameter_deltas,
        },
        "metrics": {
            "protocol": POSE_METRIC_PROTOCOL,
            "pck_threshold_mm": config.metrics.pck_threshold_mm,
            "absolute_mpjpe_mm": _float(final_metrics["absolute_mpjpe_mm"]),
            "root_relative_mpjpe_mm": _float(final_metrics["root_relative_mpjpe_mm"]),
            "root_relative_pck": _float(final_metrics["root_relative_pck"]),
            "samples": sample_metrics,
        },
        "single_frame": {
            "joints_shape": list(single_frame_output.joints.shape),
            "finite": True,
        },
        "temporal_mask_counterfactual": {
            "masked_padding_max_abs_difference": padding_max_abs_difference,
        },
        "elapsed_seconds": elapsed,
        "samples_per_second": (config.batch.batch_size * config.optimization.steps / elapsed),
        "cuda_memory": cuda_memory,
    }
