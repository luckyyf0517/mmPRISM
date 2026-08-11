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

from mmprism.assets import ModelAssetSetConfig, resolve_model_asset
from mmprism.models import GeometryGuidedMT5
from mmprism.training.mt5_config import MT5SmokeConfig, MT5SmokeError

MT5_SMOKE_REPORT_SCHEMA = "mmprism.mt5_vertical_smoke.v1"


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
        raise MT5SmokeError(f"mT5 smoke report is not strict JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def write_mt5_smoke_report(
    payload: Mapping[str, object], destination: str | Path
) -> Path:
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
        raise MT5SmokeError(f"Invalid torch device {device!r}: {error}") from error
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise MT5SmokeError("CUDA was requested but is not available")
    return resolved


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    raise MT5SmokeError(f"dtype {name} is not supported on device {device}")


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


def _build_batch(
    config: MT5SmokeConfig, *, step: int, device: torch.device, dtype: torch.dtype
) -> dict[str, Tensor]:
    batch = config.batch.batch_size
    frames = config.batch.frame_count
    model = config.model
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.runtime.seed + step)
    pose = torch.randn(
        batch,
        frames,
        2,
        model.joint_count,
        model.coordinate_dim,
        generator=generator,
        dtype=torch.float32,
    )
    confidence = torch.rand(
        batch,
        frames,
        2,
        model.joint_count,
        generator=generator,
        dtype=torch.float32,
    )
    radar = torch.randn(
        batch,
        frames,
        model.radar_feature_dim,
        generator=generator,
        dtype=torch.float32,
    )
    frame_mask = torch.ones(batch, frames, dtype=torch.long)
    if batch > 1 and frames > 1:
        frame_mask[-1, -1] = 0
        pose[-1, -1] = 0
        confidence[-1, -1] = 0
        radar[-1, -1] = 0
    return {
        "pose": pose.to(device=device, dtype=dtype),
        "pose_confidence": confidence.to(device=device, dtype=dtype),
        "radar_features": radar.to(device=device, dtype=dtype),
        "frame_attention_mask": frame_mask.to(device),
    }


def _batch_fingerprint(batch: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(batch):
        digest.update(name.encode("ascii"))
        digest.update(_tensor_sha256(batch[name]).encode("ascii"))
    return digest.hexdigest()


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _tracked_parameters(model: GeometryGuidedMT5) -> dict[str, Tensor]:
    requested = {
        "pose_encoder.output_projection.0.weight",
        "radar_projector.projection.0.weight",
        "fusion.confidence_gate.0.weight",
    }
    parameters: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if name in requested:
            parameters[name] = parameter
    if set(parameters) != requested:
        missing = sorted(requested - set(parameters))
        raise MT5SmokeError(f"Unable to track expected adapter parameters: {missing}")
    return parameters


def _gradient_norms(model: GeometryGuidedMT5) -> dict[str, float]:
    totals = {"pose_encoder": 0.0, "radar_projector": 0.0, "fusion": 0.0}
    for name, parameter in model.named_parameters():
        prefix = name.partition(".")[0]
        if prefix in totals and parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            totals[prefix] += float(torch.sum(gradient * gradient).item())
    norms = {name: math.sqrt(value) for name, value in totals.items()}
    if any(not math.isfinite(value) or value <= 0 for value in norms.values()):
        raise MT5SmokeError(f"Non-finite or zero adapter gradient norm: {norms}")
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
            "transformers": package_payload.get("transformers")
            or _package_version("transformers"),
            "sentencepiece": _package_version("sentencepiece"),
        },
        "torch_cuda": torch.version.cuda,
        "dtype": str(dtype).removeprefix("torch."),
        "device": device_payload,
    }


def run_mt5_smoke(
    config: MT5SmokeConfig,
    asset_config: ModelAssetSetConfig,
    model_root: str | Path,
    *,
    device: str,
    runtime_report: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, object]:
    git = runtime_report.get("git")
    if not isinstance(git, Mapping) or not isinstance(git.get("commit"), str):
        raise MT5SmokeError("mT5 smoke requires Git commit provenance")
    if git.get("dirty") is not False:
        raise MT5SmokeError("mT5 smoke requires a clean Git worktree")
    resolved_asset = resolve_model_asset(asset_config, model_root, config.model.asset_id)
    if resolved_asset.spec.loader != "transformers_mt5":
        raise MT5SmokeError(
            f"asset {config.model.asset_id!r} must declare loader transformers_mt5"
        )
    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(config.runtime.dtype, resolved_device)
    _seed_everything(config.runtime.seed, config.runtime.deterministic)

    try:
        from transformers import AutoTokenizer, MT5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            resolved_asset.path, local_files_only=True
        )
        language_model = MT5ForConditionalGeneration.from_pretrained(
            resolved_asset.path,
            local_files_only=True,
            dtype=dtype,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise MT5SmokeError(f"Unable to load pinned mT5 asset: {error}") from error
    if language_model.config.d_model != config.model.hidden_size:
        raise MT5SmokeError(
            "Configured hidden size does not match the pinned mT5 asset: "
            f"{config.model.hidden_size} != {language_model.config.d_model}"
        )

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
    model.to(device=resolved_device, dtype=dtype)
    model.train()
    if config.model.freeze_language_model:
        model.language_model.eval()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise MT5SmokeError("mT5 smoke has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        betas=(config.optimization.beta1, config.optimization.beta2),
    )
    tracked = _tracked_parameters(model)
    initial_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in tracked.items()
    }
    prompts = [config.batch.prompt] * config.batch.batch_size
    prompt_tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    target_tokens = tokenizer(
        list(config.batch.targets),
        padding=True,
        truncation=True,
        max_length=config.batch.max_target_length,
        return_tensors="pt",
    )
    prompt_input_ids = prompt_tokens["input_ids"].to(resolved_device)
    prompt_attention_mask = prompt_tokens["attention_mask"].to(resolved_device)
    labels = target_tokens["input_ids"].to(resolved_device)
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    started = time.perf_counter()
    steps: list[dict[str, object]] = []
    final_batch: dict[str, Tensor] | None = None
    for step in range(config.optimization.steps):
        batch = _build_batch(
            config, step=step, device=resolved_device, dtype=dtype
        )
        final_batch = batch
        optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(device_type="cuda", dtype=dtype)
            if resolved_device.type == "cuda" and dtype is torch.bfloat16
            else nullcontext()
        ):
            output = model(
                **batch,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                labels=labels,
            )
        loss_value = float(output.loss.detach().float().item())
        if not math.isfinite(loss_value):
            raise MT5SmokeError(f"Training step {step} produced non-finite loss")
        output.loss.backward()
        gradient_norms = _gradient_norms(model)
        optimizer.step()
        steps.append(
            {
                "step": step,
                "loss": loss_value,
                "gradient_norms": gradient_norms,
                "input_batch_sha256": _batch_fingerprint(batch),
                "encoder_shape": list(output.encoder_attention_mask.shape),
                "logits_shape": list(output.logits.shape),
                "pose_gate_mean": float(output.pose_gate.detach().float().mean().item()),
            }
        )
    if final_batch is None:
        raise MT5SmokeError("No smoke training batch was executed")

    deltas = {
        name: float(
            torch.max(
                torch.abs(parameter.detach().float().cpu() - initial_parameters[name])
            ).item()
        )
        for name, parameter in tracked.items()
    }
    if any(not math.isfinite(delta) or delta <= 0 for delta in deltas.values()):
        raise MT5SmokeError(f"Expected adapter parameters did not update: {deltas}")

    model.eval()
    with torch.inference_mode():
        low_confidence = torch.zeros_like(final_batch["pose_confidence"])
        high_confidence = torch.ones_like(final_batch["pose_confidence"])
        low_encoding = model.encode_modalities(
            final_batch["pose"],
            low_confidence,
            final_batch["radar_features"],
            final_batch["frame_attention_mask"],
        )
        high_encoding = model.encode_modalities(
            final_batch["pose"],
            high_confidence,
            final_batch["radar_features"],
            final_batch["frame_attention_mask"],
        )
        generated_ids = model.generate(
            **final_batch,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            max_new_tokens=config.generation.max_new_tokens,
            num_beams=config.generation.num_beams,
        )
    embedding_difference = float(
        torch.max(
            torch.abs(high_encoding.embeddings.float() - low_encoding.embeddings.float())
        ).item()
    )
    low_gate_mean = float(low_encoding.pose_gate.float().mean().item())
    high_gate_mean = float(high_encoding.pose_gate.float().mean().item())
    if not (low_gate_mean == 0 and high_gate_mean > low_gate_mean and embedding_difference > 0):
        raise MT5SmokeError("Confidence counterfactual did not alter the fusion path")
    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if len(predictions) != config.batch.batch_size:
        raise MT5SmokeError("Generation output count does not match the smoke batch")

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed = time.perf_counter() - started
    memory: dict[str, int] | None = None
    if resolved_device.type == "cuda":
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(resolved_device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(resolved_device),
        }
    report: dict[str, object] = {
        "schema_version": MT5_SMOKE_REPORT_SCHEMA,
        "status": "passed",
        "scope": "engineering_vertical_smoke_not_paper_evidence",
        "smoke_id": config.smoke_id,
        "config_fingerprint": config.fingerprint,
        "asset_config_fingerprint": asset_config.fingerprint,
        "model_asset": {
            "asset_id": resolved_asset.spec.asset_id,
            "repo_id": resolved_asset.spec.repo_id,
            "revision": resolved_asset.spec.revision,
            "loader": resolved_asset.spec.loader,
            "asset_manifest_sha256": resolved_asset.verification[
                "asset_manifest_sha256"
            ],
            "collection_manifest_sha256": resolved_asset.collection_manifest_sha256,
            "size_bytes": resolved_asset.verification["size_bytes"],
        },
        "runtime": _runtime_payload(runtime_report, resolved_device, dtype),
        "command": list(command),
        "parameter_count": _parameter_counts(model),
        "loss_protocol": {
            "name": "token_cross_entropy",
            "ignore_index": -100,
            "label_smoothing": config.model.label_smoothing,
        },
        "training": {
            "steps": steps,
            "parameter_max_abs_delta": deltas,
        },
        "confidence_counterfactual": {
            "zero_confidence_pose_gate_mean": low_gate_mean,
            "unit_confidence_pose_gate_mean": high_gate_mean,
            "embedding_max_abs_difference": embedding_difference,
        },
        "generation": {
            "max_new_tokens": config.generation.max_new_tokens,
            "num_beams": config.generation.num_beams,
            "generated_token_shape": list(generated_ids.shape),
            "samples": [
                {
                    "sample_id": f"synthetic_smoke_{index:02d}",
                    "reference": reference,
                    "prediction": prediction,
                }
                for index, (reference, prediction) in enumerate(
                    zip(config.batch.targets, predictions, strict=True)
                )
            ],
        },
        "elapsed_seconds": elapsed,
        "cuda_memory": memory,
    }
    _strict_json_bytes(report)
    return report
