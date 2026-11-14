"""Stage-one model construction, dual loss, optimizer, schedule and recoverable updates."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from mmprism.models.dense_token_deterministic_mhr import MhrTargetNormalization
from mmprism.models.dense_token_dual_supervision import (
    DenseTokenDualSupervisionModel,
    build_dual_decoder_from_direct_seed,
)
from mmprism.models.radar_early_fusion_pretrained import (
    DINO_V3_VITB_M02_ARCHITECTURE_VARIANT as FRONTEND_ARCHITECTURE,
)
from mmprism.models.radar_early_fusion_pretrained import (
    DinoV3RadarEarlyFusionRoute as RadarRoute,
)
from mmprism.models.radar_early_fusion_pretrained import (
    build_pretrained_early_fusion_route,
)
from mmprism.release.paths import resolve
from mmprism.training.reconstruction.dataset import (
    MHR_STATE_DIMENSION,
    WindowDataset,
    _normalization_arrays,
    collate_windows,
    load_prepared_segments,
)
from mmprism.training.reconstruction.dense_token_deterministic_mhr import split_complete_mhr_state
from mmprism.training.reconstruction.dense_token_dual_supervision import (
    DualSupervisionLoss,
    masked_dual_supervision_loss,
)
from mmprism.training.reconstruction.dense_token_dual_supervision import (
    HoldCosineSchedule as CosineSchedule,
)
from mmprism.training.reconstruction.mixed_stream import DailySimulationStream, MixedCycleLoader

CHECKPOINT_SCHEMA = "mmprism.reconstruction.training_checkpoint"


class TrainingError(RuntimeError):
    """The reconstruction training state cannot continue."""


def _section_path(config, section, key):
    return resolve(config[section][key])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _initialize_frontend(route: RadarRoute) -> dict[str, object]:
    import torch

    frontend = route.complex_frontend
    _require(frontend is not None, "reconstruction frontend is absent")
    _require(frontend.block_c is None, "reconstruction requires the two-block complex frontend")
    standard_deviation = math.sqrt(1.0 / (2.0 * frontend.block_a.input_channels))
    real = torch.empty(
        (frontend.block_a.output_channels, frontend.block_a.input_channels), dtype=torch.float32
    ).normal_(mean=0.0, std=standard_deviation)
    imaginary = torch.empty_like(real).normal_(mean=0.0, std=standard_deviation)
    with torch.no_grad():
        frontend.block_a.initialize_center_tap(real, imaginary)
        frontend.block_b.initialize_complex_identity_center()
        frontend.block_a_activation.bias.zero_()
        frontend.block_b_activation.bias.zero_()
    center = frontend.block_a.kernel_size // 2
    noncenter = frontend.block_a.real_weight.detach().clone()
    noncenter[..., center, center] = 0.0
    noncenter_imaginary = frontend.block_a.imag_weight.detach().clone()
    noncenter_imaginary[..., center, center] = 0.0
    _require(
        float(noncenter.abs().max()) == 0.0 and float(noncenter_imaginary.abs().max()) == 0.0,
        "fresh reconstruction Block A has a nonzero non-center tap",
    )
    return {
        "block_a": "fresh complex fan-in pointwise center tap; zero non-center taps",
        "block_b": "complex identity center tap; zero non-center taps",
        "modrelu_biases": "zero",
        "block_a_center_real_std": float(
            frontend.block_a.real_weight[..., center, center].detach().std()
        ),
        "block_a_center_imaginary_std": float(
            frontend.block_a.imag_weight[..., center, center].detach().std()
        ),
    }


def _build_fresh_route(config: Mapping[str, Any]) -> tuple[RadarRoute, dict[str, object]]:
    processor = _mapping(config.get("processor"), "processor")
    route = build_pretrained_early_fusion_route(
        processor_config=SimpleNamespace(
            remove_static=bool(processor["remove_static"]),
            range_bin_count=int(processor["range_bin_count"]),
            apply_doppler_window=bool(processor["apply_doppler_window"]),
        ),
        pose_head_variant="relu_linear",
        backbone="dinov3_vit_base",
        adapter="a1",
        readout="stage4",
        use_context=False,
        range_preserving=False,
        architecture_variant=FRONTEND_ARCHITECTURE,
        output_shape=(2, 24, 3),
    )
    _require(isinstance(route, RadarRoute), "fresh DINO reconstruction route differs")
    _require(
        route.architecture_variant == FRONTEND_ARCHITECTURE,
        "fresh route is not the reconstruction chassis",
    )
    return (route, _initialize_frontend(route))


def _load_backbone_body(route: RadarRoute, config: Mapping[str, Any]) -> dict[str, object]:
    from safetensors.torch import load_file

    path = _section_path(config, "upstream", "official_dinov3_vitb_weights")
    _require(path.is_file(), f"official DINOv3 ViT-B weights are missing: {path}")
    source = load_file(str(path), device="cpu")
    target = route.dino_v3_body.state_dict()
    fresh_patch = {"embeddings.patch_embeddings.weight", "embeddings.patch_embeddings.bias"}
    _require(
        set(source) == set(target),
        "official DINOv3 state must exactly match the registered ViT-B body",
    )
    before_non_dino = {
        name: value.detach().clone()
        for (name, value) in route.state_dict().items()
        if not name.startswith("dino_v3_body.")
    }
    body = {name: value for (name, value) in source.items() if name not in fresh_patch}
    incompatible = route.dino_v3_body.load_state_dict(body, strict=False)
    _require(
        set(incompatible.missing_keys) == fresh_patch and (not incompatible.unexpected_keys),
        "official DINO body load left unexpected route tensors",
    )
    after_non_dino = route.state_dict()
    _require(
        all(
            torch_equal(before, after_non_dino[name]) for (name, before) in before_non_dino.items()
        ),
        "official DINO load modified a non-DINO tensor",
    )
    return {
        "weights_path": str(path),
        "source_tensor_count": len(source),
        "loaded_transformer_and_norm_tensor_count": len(body),
        "fresh_radar_patch_embedding_tensors": sorted(fresh_patch),
        "non_dino_tensors_unchanged_by_external_load": True,
        "excluded_image_specific_head": "not present in registered body state",
    }


def torch_equal(first: Any, second: Any) -> bool:
    import torch

    return (
        isinstance(first, torch.Tensor)
        and isinstance(second, torch.Tensor)
        and bool(torch.equal(first, second))
    )


def _configure_route_trainability(route: RadarRoute) -> None:
    frozen_prefixes = (
        "doppler_processor.",
        "pose_head.",
        "spatial_readout.",
        "feature_projection.",
        "feature_normalization.",
    )
    frozen_names = {"dino_v3_body.embeddings.mask_token"}
    for name, parameter in route.named_parameters():
        parameter.requires_grad_(
            name not in frozen_names
            and (not any(name.startswith(prefix) for prefix in frozen_prefixes))
        )


def _build_base_model(
    config: Mapping[str, Any], *, load_official_dino: bool
) -> tuple[Any, dict[str, object]]:
    model_config = _mapping(config.get("model"), "model")
    (route, frontend_audit) = _build_fresh_route(config)
    dino_audit: dict[str, object]
    if load_official_dino:
        dino_audit = _load_backbone_body(route, config)
    else:
        dino_audit = {"restored_from_prepared_initial_state": True}
    _configure_route_trainability(route)
    (decoder, reader_audit) = build_dual_decoder_from_direct_seed(
        mhr_head_seed=int(model_config["mhr_head_seed"]),
        token_channels=int(model_config["token_shape"][1]),
        hidden_channels=int(model_config["hidden_channels"]),
        attention_heads=int(model_config["attention_heads"]),
        mlp_ratio=int(model_config["mlp_ratio"]),
        double_stream_blocks=int(model_config["double_stream_blocks"]),
        fused_single_stream_blocks=int(model_config["fused_single_stream_blocks"]),
        maximum_frames=int(_mapping(config.get("data"), "data")["train_window_frames"]),
    )
    model = DenseTokenDualSupervisionModel(route=route, decoder=decoder)
    trainable = {name: value for (name, value) in model.named_parameters() if value.requires_grad}
    required = (
        "route.complex_frontend.",
        "route.radar_adapter.",
        "route.dino_v3_body.embeddings.patch_embeddings.",
        "route.dino_v3_body.layer.0.",
        "route.dino_v3_body.layer.11.",
        "decoder.token_projection.",
        "decoder.motion_queries",
        "decoder.double_blocks.",
        "decoder.single_blocks.",
        "decoder.direct_coarse_head.",
        "decoder.direct_left_hand_head.",
        "decoder.direct_right_hand_head.",
        "decoder.upper_global_head.",
        "decoder.left_hand_head.",
        "decoder.right_hand_head.",
        "decoder.static_head.",
    )
    for prefix in required:
        _require(
            any(name.startswith(prefix) for name in trainable), f"trainable path absent: {prefix}"
        )
    _require(
        not any(name.startswith("route.pose_head.") for name in trainable),
        "unused incumbent pose head entered optimizer",
    )
    return (
        model,
        {
            "fresh_frontend": frontend_audit,
            "dino_initialization": dino_audit,
            "reader_initialization": reader_audit,
            "trainable_parameter_count": int(sum(value.numel() for value in trainable.values())),
            "trainable_tensor_count": len(trainable),
            "forbidden_task_checkpoints_loaded": False,
        },
    )


def _batch_terms(
    *,
    model: Any,
    batch: Mapping[str, Any],
    normalization: MhrTargetNormalization,
    fk: Any,
    device: Any,
    capture_activations: bool = False,
) -> tuple[DualSupervisionLoss, Any, Any]:
    import torch

    raw = batch["raw_radar"].to(device=device, dtype=torch.float32, non_blocking=True)
    temporal = batch["temporal_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    target_mask = batch["target_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    target_global = batch["target_global_pose"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    target_state = batch["target_mhr191"].to(device=device, dtype=torch.float32, non_blocking=True)
    prediction = model(raw, valid_mask=temporal, capture_activations=capture_activations)
    fk_full24 = decode_complete_mhr_full24(
        prediction.mhr_state.state, temporal_mask=temporal, fk=fk
    )
    terms = masked_dual_supervision_loss(
        direct_coarse18_m=prediction.direct.coarse18_m.reshape(*raw.shape[:2], 18),
        direct_hand120_wrist_relative_m=prediction.direct.hand120_wrist_relative_m.reshape(
            *raw.shape[:2], 120
        ),
        fk_decoded_full24_m=fk_full24,
        predicted_mhr191=prediction.mhr_state.state,
        target_global_m=target_global,
        target_mhr191=target_state,
        valid_mask=target_mask,
        capacity_frames=int(temporal.shape[1]),
        normalization=normalization,
    )
    return (terms, prediction, temporal)


def _optimizer(model: Any, config: Mapping[str, Any]) -> tuple[Any, dict[str, object]]:
    import torch

    optimization = _mapping(config.get("optimization"), "optimization")
    trainable = [(name, value) for (name, value) in model.named_parameters() if value.requires_grad]
    _require(trainable, "reconstruction has no trainable tensors")
    optimizer = torch.optim.AdamW(
        [value for (_, value) in trainable],
        lr=float(optimization["learning_rate"]),
        betas=(float(optimization["beta1"]), float(optimization["beta2"])),
        weight_decay=float(optimization["weight_decay"]),
    )
    covered = {id(value) for group in optimizer.param_groups for value in group["params"]}
    _require(covered == {id(value) for (_, value) in trainable}, "optimizer coverage differs")
    return (
        optimizer,
        {
            "kind": "AdamW",
            "learning_rate": float(optimization["learning_rate"]),
            "weight_decay": float(optimization["weight_decay"]),
            "betas": [float(optimization["beta1"]), float(optimization["beta2"])],
            "trainable_parameter_count": int(sum(value.numel() for (_, value) in trainable)),
            "trainable_tensor_count": len(trainable),
        },
    )


def _set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _all_reduce(values: Any, *, ddp: bool) -> Any:
    if ddp:
        from torch import distributed

        distributed.all_reduce(values)
    return values


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: CosineSchedule,
    output_root: Path,
    completed_step: int,
    sampler_cycle: int,
    consumed_batches_in_cycle: int,
    rng_states: Sequence[Mapping[str, Any]],
    rank: int,
) -> None:
    if rank != 0:
        return
    import torch

    output = _checkpoint_path(output_root, completed_step)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "kind": "reconstruction_training_checkpoint",
            "completed_optimizer_steps": completed_step,
            "sampler_cycle": sampler_cycle,
            "consumed_batches_in_cycle": consumed_batches_in_cycle,
            "model_state_dict": getattr(model, "module", model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_states": list(rng_states),
        },
        temporary,
    )
    temporary.replace(output)
    latest = output_root / "checkpoints" / "reconstruction-last.pt"
    latest_temporary = latest.with_name(f".{latest.name}.tmp.{os.getpid()}")
    shutil.copyfile(output, latest_temporary)
    latest_temporary.replace(latest)


def _load_resume_checkpoint(
    *, model: Any, optimizer: Any, path: Path, rank: int, world_size: int
) -> tuple[int, int, int, CosineSchedule]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Original and public checkpoints share these payload fields; their labels may differ.
    _require(isinstance(payload, Mapping), "resume checkpoint differs")
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    scheduler_state = payload.get("scheduler_state_dict")
    _require(
        isinstance(model_state, Mapping)
        and isinstance(optimizer_state, Mapping)
        and isinstance(scheduler_state, Mapping),
        "resume state is malformed",
    )
    getattr(model, "module", model).load_state_dict(
        cast(Mapping[str, Any], model_state), strict=True
    )
    optimizer.load_state_dict(cast(Mapping[str, Any], optimizer_state))
    scheduler = CosineSchedule.from_state_dict(dict(scheduler_state))
    states = payload.get("rng_states")
    _require(isinstance(states, list) and len(states) == world_size, "resume RNG differs")
    _require(isinstance(states[rank], Mapping), "rank resume RNG differs")
    restore_rng_state(cast(Mapping[str, Any], states[rank]))
    try:
        completed = int(payload["completed_optimizer_steps"])
        cycle = int(payload["sampler_cycle"])
        offset = int(payload["consumed_batches_in_cycle"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError(f"resume progress is malformed: {error}") from error
    _require(completed == scheduler.completed_optimizer_steps, "resume scheduler step differs")
    return (completed, cycle, offset, scheduler)


def _training_schedule(config: Mapping[str, Any]) -> CosineSchedule:
    optimization = _mapping(config.get("optimization"), "optimization")
    return CosineSchedule(
        hold_steps=int(optimization["hold_learning_rate_steps"]),
        max_optimizer_steps=int(optimization["max_optimizer_steps"]),
        peak_learning_rate=float(optimization["learning_rate"]),
        final_learning_rate=float(optimization["final_learning_rate"]),
    )


def distributed_context() -> tuple[bool, int, int, Any]:
    import torch
    from torch import distributed

    if "RANK" not in os.environ:
        return (False, 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    _require(torch.cuda.is_available(), "distributed reconstruction requires an accelerator")
    torch.cuda.set_device(local_rank)
    if not distributed.is_initialized():
        distributed.init_process_group(backend="nccl")
    return (True, rank, world_size, torch.device(f"cuda:{local_rank}"))


def barrier(ddp: bool) -> None:
    if ddp:
        from torch import distributed

        distributed.barrier()


def destroy_distributed(ddp: bool) -> None:
    if ddp:
        from torch import distributed

        distributed.destroy_process_group()


def set_seed(seed: int, *, device: Any) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise TrainingError(
            f"cannot restore reconstruction checkpoint random state: {error}"
        ) from error


def gather_rng_states(*, ddp: bool, local_state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not ddp:
        return [local_state]
    from torch import distributed

    gathered: list[Mapping[str, Any] | None] = [None] * distributed.get_world_size()
    distributed.all_gather_object(gathered, dict(local_state))
    _require(all(value is not None for value in gathered), "reconstructionRNG gather is incomplete")
    return [cast(Mapping[str, Any], value) for value in gathered]


def decode_complete_mhr_full24(
    state: Tensor, *, temporal_mask: Tensor, fk: Any, fixed_standard_body: bool = False
) -> Tensor:
    from mmprism.models.mhr_param_fk import map_keypoints_70_to_full24_torch

    _require(
        state.ndim == 3 and state.shape[-1] == MHR_STATE_DIMENSION, "MHR-FK state must be [B,T,191]"
    )
    _require(
        temporal_mask.shape == state.shape[:2] and temporal_mask.dtype == torch.bool,
        "MHR-FK temporal mask differs",
    )
    _require(bool(temporal_mask.any(dim=1).all()), "every MHR-FK item needs one temporal frame")
    selected = state[temporal_mask]
    _require(bool(torch.isfinite(selected).all()), "MHR-FK state is non-finite")
    parts = split_complete_mhr_state(selected)
    morphology = torch.zeros_like(parts.morphology) if fixed_standard_body else parts.morphology
    lambda_to_standard = (
        torch.ones_like(parts.lambda_to_standard)
        if fixed_standard_body
        else parts.lambda_to_standard
    )
    points70 = fk.decode_points70_complete(
        torch.cat((parts.left_hand_articulation, parts.right_hand_articulation), dim=-1),
        parts.upper_articulation,
        parts.global_rotation,
        morphology,
    )
    decoded = map_keypoints_70_to_full24_torch(points70)
    decoded = (
        decoded * lambda_to_standard[:, None, None, :] + parts.camera_translation[:, None, None, :]
    ).to(dtype=state.dtype)
    flat_mask = temporal_mask.reshape(-1)
    flat_indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    flat = torch.zeros(
        (state.shape[0] * state.shape[1], 2, 24, 3), dtype=state.dtype, device=state.device
    )
    result = flat.index_copy(0, flat_indices, decoded).reshape(*state.shape[:2], 2, 24, 3)
    _require(bool(torch.isfinite(result).all()), "MHR-FK decoded full24 is non-finite")
    return result


def _fast_validation(
    *,
    model: Any,
    dataset: WindowDataset,
    fk: Any,
    device: Any,
    rank: int,
    world_size: int,
    ddp: bool,
) -> dict[str, Any]:
    import torch

    was_training = model.training
    model.eval()
    aggregates = torch.zeros((2, 10), dtype=torch.float64, device=device)
    with torch.inference_mode():
        for segment_index in range(rank, len(dataset), world_size):
            segment = dataset.segments[segment_index]
            for start in range(0, segment.frame_count, dataset.window_frames):
                item = dataset.fixed_window(segment_index, start)
                raw = torch.from_numpy(np.asarray(item["raw_radar"])).unsqueeze(0).to(device)
                temporal = (
                    torch.from_numpy(np.asarray(item["temporal_mask"])).unsqueeze(0).to(device)
                )
                target_mask = (
                    torch.from_numpy(np.asarray(item["target_mask"])).unsqueeze(0).to(device)
                )
                target = (
                    torch.from_numpy(np.asarray(item["target_global_pose"])).unsqueeze(0).to(device)
                )
                output = model(
                    raw.to(dtype=torch.float32), valid_mask=temporal.to(dtype=torch.bool)
                )
                mhr_full = decode_complete_mhr_full24(
                    output.mhr_state.state, temporal_mask=temporal.to(dtype=torch.bool), fk=fk
                )
                for head, prediction in enumerate((output.direct.full24_m, mhr_full)):
                    physical = target[:, :, :, 2, 1] <= target[:, :, :, 1, 1]
                    valid_sides = physical & target_mask[:, :, None]
                    if bool(valid_sides.any()):
                        point_weights = valid_sides[:, :, :, None].to(dtype=torch.float64)
                        full_error = (prediction - target).norm(dim=-1).to(dtype=torch.float64)
                        aggregates[head, 0] += (full_error * point_weights).sum()
                        aggregates[head, 1] += point_weights.sum() * 24.0
                        wrist_weights = valid_sides.to(dtype=torch.float64)
                        aggregates[head, 2] += (full_error[:, :, :, 2] * wrist_weights).sum()
                        aggregates[head, 3] += wrist_weights.sum()
                        local_error = (
                            (
                                prediction[:, :, :, 4:]
                                - prediction[:, :, :, 2:3]
                                - (target[:, :, :, 4:] - target[:, :, :, 2:3])
                            )
                            .norm(dim=-1)
                            .to(dtype=torch.float64)
                        )
                        aggregates[head, 4] += (local_error * point_weights).sum()
                        aggregates[head, 5] += point_weights.sum() * 20.0
                        aggregates[head, 8] += wrist_weights.sum()
                    aggregates[head, 9] += target_mask.sum()
    _all_reduce(aggregates, ddp=ddp)
    result: dict[str, Any] = {}
    for name, values in zip(("direct", "mhr_fk"), aggregates, strict=True):
        _require(bool((values[[1, 3, 5, 8, 9]] > 0).all()), f"validation support is empty: {name}")
        result[name] = {
            "raw_global_full24_mpjpe_mm": float(values[0].item() / values[1].item() * 1000.0),
            "wrist_mpjpe_mm": float(values[2].item() / values[3].item() * 1000.0),
            "wrist_relative_hand20_mpjpe_mm": float(values[4].item() / values[5].item() * 1000.0),
            "valid_hands": int(values[8].item()),
            "valid_frames": int(values[9].item()),
            "hand20_denominator_points_per_valid_hand": 20,
            "mode": "deterministic non-overlapping 32-frame full-validation chunks",
        }
    if was_training:
        model.train()
    return result


def _checkpoint_path(output_root: Path, step: int) -> Path:
    return output_root / "checkpoints" / f"reconstruction-step-{step:05d}.pt"


def _normalization(target_root: Path, *, device: Any) -> MhrTargetNormalization:
    values = _normalization_arrays(target_root)
    return MhrTargetNormalization(
        **{key: torch.from_numpy(value).to(device) for key, value in values.items()}
    )


def _new_fk(config: Mapping[str, Any], *, device: Any) -> Any:
    from mmprism.models.mhr_param_fk import MhrParamFK

    return MhrParamFK(
        _section_path(config, "upstream", "mhr_rig"),
        _section_path(config, "upstream", "mhr_fk_constants"),
        device=device,
    )


def _log(output_root: Path, record: Mapping[str, Any], *, rank: int) -> None:
    """Use one journal for training and validation, with the same stdout record."""
    if rank == 0:
        text = json.dumps(record, sort_keys=True, allow_nan=False)
        print(text, flush=True)
        with (output_root / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")


def _configure_caches(output_root: Path, rank: int) -> None:
    import tempfile

    for variable, directory in (
        ("TMPDIR", "tmp"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        path = output_root / ".cache" / str(rank) / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]


def _prepared_initial_state(config, output_root, *, ddp, rank) -> Path:
    if "initial_state" in config["upstream"]:
        return _section_path(config, "upstream", "initial_state")
    path = output_root / "initial_state.pt"
    if rank == 0 and not path.is_file():
        mixed = bool(config.get("mixing"))
        torch.manual_seed(int(config["optimization"]["seed"]))
        model, audit = build_model(config, load_official_dino=not mixed)
        if mixed:
            warm = torch.load(
                _section_path(config, "upstream", "warm_start_checkpoint"),
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            _require(
                int(warm["completed_optimizer_steps"]) == 80_000,
                "mixed continuation requires the real-only step-80000 checkpoint",
            )
            incompatible = model.load_state_dict(warm["model_state_dict"], strict=False)
            fresh = {name for name in model.state_dict() if name.startswith("error_head.")}
            _require(
                not incompatible.unexpected_keys and set(incompatible.missing_keys) == fresh,
                "warm-start tensor inventory differs",
            )
        temporary = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                },
                "initialization": audit,
            },
            temporary,
        )
        temporary.replace(path)
    barrier(ddp)
    return path


def _close_iterator(iterator) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def train(config: Mapping[str, Any], *, resume_key: str | None = None) -> Path:
    """Train on aligned real and freshly simulated windows; paths are logical keys.

    Required resource keys: output_root, upstream.binding_root, cache.target_root,
    upstream.initial_state, and the renderer's upstream assets. The prepared initial
    state uses the original model_state_dict payload. This API deliberately has no
    console command because the full training corpus is not distributed.
    """
    from torch.nn.parallel import DistributedDataParallel

    from mmprism.data.reconstruction_membership import deterministic_cycle_batches

    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_caches(output_root, int(os.environ.get("LOCAL_RANK", "0")))
    ddp, rank, world_size, device = distributed_context()
    iterator = None
    try:
        runtime = _mapping(config.get("runtime"), "runtime")
        optimization = _mapping(config.get("optimization"), "optimization")
        _require(
            ddp and world_size == int(runtime["world_size"]),
            "training requires the configured distributed ranks",
        )
        initial_path = _prepared_initial_state(config, output_root, ddp=ddp, rank=rank)
        set_seed(int(optimization["seed"]), device=device)
        segments = load_prepared_segments(_section_path(config, "upstream", "binding_root"))
        target_root = _section_path(config, "cache", "target_root")
        dataset = WindowDataset(
            config=config, segments=segments, role="train", target_root=target_root
        )
        mixed = bool(config.get("mixing"))
        diagnostic_batch = (
            None if mixed else _fixed_diagnostic_batch(dataset, int(runtime["per_device_batch"]))
        )
        model, _audit = build_model(config, load_official_dino=False)
        initial = torch.load(initial_path, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(initial["model_state_dict"], strict=True)
        del initial
        model = model.to(device)
        wrapped = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index
        )
        optimizer, _audit = _optimizer(wrapped, config)
        schedule = _training_schedule(config)
        if resume_key is None:
            completed_step, sampler_cycle, batch_offset = 0, 0, 0
        else:
            completed_step, sampler_cycle, batch_offset, schedule = _load_resume_checkpoint(
                model=wrapped,
                optimizer=optimizer,
                path=resolve(resume_key),
                rank=rank,
                world_size=world_size,
            )

        def learning_rate() -> float:
            warmup = int(optimization.get("warmup_steps", 0)) if mixed else 0
            factor = 1.0 if warmup <= 0 else min(1.0, (completed_step + 1) / warmup)
            return schedule.next_learning_rate() * max(factor, 1e-3)

        _set_optimizer_learning_rate(optimizer, learning_rate())
        mixing = _mapping(config["mixing"], "mixing") if mixed else {}
        normalization = _normalization(target_root, device=device)
        fk = _new_fk(config, device=device)
        streams = []
        validation_stream = None
        if mixed:
            donor_indices = [
                i for i, segment in enumerate(dataset.segments) if segment.corpus == "old"
            ]
            streams = [
                DailySimulationStream(
                    config=config, device=device, donor_dataset=dataset, donor_indices=donor_indices
                )
                for _ in range(max(1, int(mixing["prefetch_workers"])))
            ]
            for number, stream in enumerate(streams):
                for warm in range(4):
                    stream.draw(global_draw=2_000_000_000 + 10 * number + warm, step=0, rank=0)
                stream.rendered_windows = 0
        started = time.monotonic()
        local = np.zeros(6, dtype=np.float64)
        local_count = 0
        requested_stop = optimization.get("stop_at_step")
        requested_stop = int(requested_stop) if requested_stop is not None else None
        stopped_non_finite = False
        while completed_step < schedule.max_optimizer_steps:
            if iterator is None:
                # mixed_stream.py:471-617 replaces the real-only loader.
                batches_per_cycle = len(
                    deterministic_cycle_batches(
                        segment_count=len(dataset),
                        world_size=world_size,
                        per_device_batch=int(runtime["per_device_batch"]),
                        base_seed=int(optimization["sampler_seed"]),
                        sampler_cycle=sampler_cycle,
                    )[rank]
                )
                _require(0 <= batch_offset < batches_per_cycle, "resume sampler offset differs")
                if mixed:
                    loader = MixedCycleLoader(
                        real_dataset=dataset,
                        sim_streams=streams,
                        world_size=world_size,
                        rank=rank,
                        per_device_batch=int(runtime["per_device_batch"]),
                        real_count=int(mixing["per_device_real"]),
                        sim_count=int(mixing["per_device_simulation"]),
                        base_seed=int(optimization["sampler_seed"]),
                        sampler_cycle=sampler_cycle,
                        start_batch=batch_offset,
                        inline=int(mixing["prefetch_workers"]) == 0,
                        data_workers=int(mixing["data_workers"]),
                    )
                else:
                    from mmprism.training.reconstruction.dataset import RankCycleBatchSampler

                    sampler = RankCycleBatchSampler(
                        segment_count=len(dataset),
                        world_size=world_size,
                        rank=rank,
                        per_device_batch=int(runtime["per_device_batch"]),
                        base_seed=int(optimization["sampler_seed"]),
                        sampler_cycle=sampler_cycle,
                        start_batch=batch_offset,
                    )
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        num_workers=int(runtime["train_num_workers"]),
                        pin_memory=bool(runtime["pin_memory"]),
                        persistent_workers=bool(runtime["persistent_workers"]),
                        prefetch_factor=int(runtime["prefetch_factor"]),
                        collate_fn=collate_windows,
                    )
                iterator = iter(loader)
            batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            terms, _prediction, _temporal = training_terms(
                model=wrapped,
                batch=batch,
                normalization=normalization,
                fk=fk,
                device=device,
                config=config,
            )
            terms.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [value for value in wrapped.parameters() if value.requires_grad],
                float(optimization["gradient_clip_norm"]),
            )
            if mixed and not (torch.isfinite(terms.total) and torch.isfinite(gradient_norm)):
                stopped_non_finite = True
                _log(
                    output_root,
                    {"event": "stopped_non_finite", "completed_optimizer_steps": completed_step},
                    rank=rank,
                )
                barrier(ddp)
                break
            optimizer.step()
            completed_step += 1
            schedule.step()
            _set_optimizer_learning_rate(optimizer, learning_rate())
            batch_offset += 1
            local += np.asarray(
                [
                    float(terms.total.detach().item()),
                    float(terms.direct_xyz.coarse_coordinate_l1.detach().item()),
                    float(terms.direct_xyz.hand_coordinate_l1.detach().item()),
                    float(terms.fk_xyz.coarse_coordinate_l1.detach().item()),
                    float(terms.fk_xyz.hand_coordinate_l1.detach().item()),
                    float(terms.mhr_parameter.total.detach().item()),
                ],
                dtype=np.float64,
            )
            local_count += 1
            if batch_offset == batches_per_cycle:
                _close_iterator(iterator)
                iterator = None
                sampler_cycle += 1
                batch_offset = 0
            if completed_step % 100 == 0:
                summary = torch.tensor(
                    [*local.tolist(), float(local_count)], dtype=torch.float64, device=device
                )
                _all_reduce(summary, ddp=ddp)
                names = (
                    "total_loss",
                    "direct_coarse_l1",
                    "direct_hand_l1",
                    "fk_coarse_l1",
                    "fk_hand_l1",
                    "mhr_parameter_smooth_l1",
                )
                record = {
                    name: float(summary[i].item() / summary[-1].item())
                    for i, name in enumerate(names)
                }
                _log(
                    output_root,
                    {
                        "event": "train",
                        "completed_optimizer_steps": completed_step,
                        "elapsed_seconds": time.monotonic() - started,
                        "next_update_learning_rate": learning_rate(),
                        **record,
                    },
                    rank=rank,
                )
                local.fill(0.0)
                local_count = 0
            stop = requested_stop is not None and completed_step >= requested_stop
            if completed_step % int(optimization["checkpoint_every_steps"]) == 0 or stop:
                states = gather_rng_states(ddp=ddp, local_state=capture_rng_state())
                _save_checkpoint(
                    model=wrapped,
                    optimizer=optimizer,
                    scheduler=schedule,
                    output_root=output_root,
                    completed_step=completed_step,
                    sampler_cycle=sampler_cycle,
                    consumed_batches_in_cycle=batch_offset,
                    rng_states=states,
                    rank=rank,
                )
                barrier(ddp)
            if stop:
                break
            if not mixed and completed_step in {10_000, 20_000, 30_000, 40_000, 50_000}:
                diagnostic = _gradient_interaction_diagnostic(
                    model=wrapped,
                    batch=diagnostic_batch,
                    normalization=normalization,
                    fk=fk,
                    device=device,
                )
                _log(
                    output_root,
                    {
                        "event": "gradient_interaction",
                        "completed_optimizer_steps": completed_step,
                        "diagnostic": diagnostic,
                    },
                    rank=rank,
                )
                barrier(ddp)
            if completed_step % int(optimization["validate_every_steps"]) == 0:
                validation = WindowDataset(
                    config=config, segments=segments, role="validation", target_root=target_root
                )
                metrics = _validation_by_pool(
                    model=wrapped,
                    dataset=validation,
                    fk=fk,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    ddp=ddp,
                )
                simulated_metrics = None
                if mixed:
                    from mmprism.training.reconstruction.mixed_stream import simulation_validation

                    if validation_stream is None:
                        validation_stream = DailySimulationStream(
                            config=config,
                            device=device,
                            donor_dataset=dataset,
                            donor_indices=donor_indices,
                        )
                        validation_stream.specs = validation_stream.sim._membership(
                            validation_stream.cfg
                        ).role("validation")
                    simulated_metrics = simulation_validation(
                        model=wrapped,
                        stream=validation_stream,
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        all_reduce=lambda tensor: _all_reduce(tensor, ddp=ddp),
                    )
                _log(
                    output_root,
                    {
                        "event": "validation",
                        "completed_optimizer_steps": completed_step,
                        "metrics": metrics,
                        "simulation_validation": simulated_metrics,
                    },
                    rank=rank,
                )
                barrier(ddp)
        if rank == 0:
            terminal = completed_step == schedule.max_optimizer_steps
            checkpoint = _checkpoint_path(output_root, completed_step)
            if terminal:
                _require(checkpoint.is_file(), "terminal checkpoint is missing")
                (output_root / "selected_checkpoint.json").write_text(
                    json.dumps(
                        {
                            "selection": "declared terminal endpoint; validation is diagnostic",
                            "selected_completed_optimizer_steps": completed_step,
                            "selected_checkpoint": checkpoint.relative_to(output_root).as_posix(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            status = (
                "completed"
                if terminal
                else "stopped_non_finite"
                if stopped_non_finite
                else "stopped_at_requested_step"
            )
            (output_root / "training_run.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "completed_optimizer_steps": completed_step,
                        "max_optimizer_steps": schedule.max_optimizer_steps,
                        "next_update_learning_rate": learning_rate(),
                    },
                    indent=2,
                )
                + "\n"
            )
        barrier(ddp)
        return output_root
    finally:
        if iterator is not None:
            _close_iterator(iterator)
        destroy_distributed(ddp)


def build_model(config: Mapping[str, Any], *, load_official_dino: bool) -> tuple[Any, dict]:
    """Build the spectral reconstruction network, with its continuation error head."""
    from mmprism.models.dense_token_spectral_incoherent import IncoherentSpectralBranch
    from mmprism.training.reconstruction.mixed_stream import install_error_head

    model, audit = _build_base_model(config, load_official_dino=load_official_dino)
    model.spatial_branch = IncoherentSpectralBranch(hidden=int(config["model"]["hidden_channels"]))
    encode = model.encode_dense_tokens

    def encode_with_residual(raw_radar: Tensor, *, valid_mask: Tensor) -> Any:
        output = encode(raw_radar, valid_mask=valid_mask)
        model.spectral_residual = model.spatial_branch(raw_radar)
        return output

    model.encode_dense_tokens = encode_with_residual

    def projection_hook(module: Any, inputs: Any, output: Tensor) -> Tensor:
        return output + model.spectral_residual[:, :, None, :]

    model.decoder.token_projection.register_forward_hook(projection_hook)
    if config.get("error_head"):
        audit["error_head"] = install_error_head(model, config=config)
    return model, audit


def training_terms(
    *,
    model: Any,
    batch: Mapping[str, Any],
    normalization: Any,
    fk: Any,
    device: Any,
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    """Apply the calibrated base auxiliary or the complete mixed continuation objective."""
    from mmprism.training.reconstruction.hand_palm_frame_torch import (
        intrinsic_unit_bone_cosine_loss,
    )
    from mmprism.training.reconstruction.mixed_stream import AugmentedTerms, mixed_batch_terms

    calibration = json.loads(resolve(config["objective"]["auxiliary_calibration"]).read_text())
    weight = float(calibration["fixed_auxiliary_weight"])
    if config.get("mixing"):
        return mixed_batch_terms(
            model=model,
            batch=batch,
            normalization=normalization,
            fk=fk,
            device=device,
            contribution_weights={
                "auxiliary_weight": weight,
                "error_head_weight": float(config["objective"]["error_head_weight"]),
            },
            n_real=int(config["mixing"]["per_device_real"]),
        )
    terms, prediction, temporal = _batch_terms(
        model=model, batch=batch, normalization=normalization, fk=fk, device=device
    )
    target = batch["target_global_pose"].to(device=device, dtype=torch.float32, non_blocking=True)
    valid = batch["target_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    auxiliary = intrinsic_unit_bone_cosine_loss(prediction.direct.full24_m, target, valid)
    model.intrinsic_auxiliary_last = auxiliary.detach()
    return (
        AugmentedTerms(terms, terms.total + weight * auxiliary, auxiliary, auxiliary.new_zeros(())),
        prediction,
        temporal,
    )


def _validation_by_pool(*, dataset: WindowDataset, **kwargs: Any) -> dict[str, Any]:
    """Compute per-pool validation and weighted overall metrics."""
    import copy

    def pool_name(segment: Any) -> str:
        return "old" if segment.corpus == "old" else segment.source_id

    per = {}
    for name in sorted(
        {pool_name(segment) for segment in dataset.segments}, key=lambda name: (name != "old", name)
    ):
        view = copy.copy(dataset)
        view.segments = tuple(segment for segment in dataset.segments if pool_name(segment) == name)
        per[name] = _fast_validation(dataset=view, **kwargs)
    result = {}
    for head in ("direct", "mhr_fk"):
        hands = sum(per[name][head]["valid_hands"] for name in per)
        row = {
            key: sum(per[name][head][key] * per[name][head]["valid_hands"] for name in per) / hands
            for key in (
                "raw_global_full24_mpjpe_mm",
                "wrist_mpjpe_mm",
                "wrist_relative_hand20_mpjpe_mm",
            )
        }
        row.update(
            valid_hands=hands,
            valid_frames=sum(per[name][head]["valid_frames"] for name in per),
            hand20_denominator_points_per_valid_hand=20,
            mode="pooled deterministic non-overlapping 32-frame chunks",
        )
        result[head] = row
    result["pools"] = per
    return result


def _gradient_vector(*, prediction: Any, temporal: Any, tap: str) -> Any:
    _require(prediction.dense_tokens.grad is not None, "post-DINO gradient is unavailable")
    _require(
        prediction.normalized_reader_tokens.grad is not None, "shared-query gradient is unavailable"
    )
    if tap == "post_dino_dense_tokens":
        values = prediction.dense_tokens.grad[temporal]
    else:
        index = {
            "normalized_upper_query": 16,
            "normalized_left_query": 17,
            "normalized_right_query": 18,
        }.get(tap)
        _require(index is not None, f"unknown gradient tap: {tap}")
        values = prediction.normalized_reader_tokens.grad[:, :, int(index)][temporal]
    _require(values.numel() > 0 and bool(values.isfinite().all()), f"invalid gradient tap: {tap}")
    return values.detach().reshape(-1).float().clone()


def _clear_activation_gradients(prediction: Any) -> None:
    prediction.dense_tokens.grad = None
    prediction.normalized_reader_tokens.grad = None


def _gradient_interaction_diagnostic(
    *,
    model: Any,
    batch: Mapping[str, Any],
    normalization: MhrTargetNormalization,
    fk: Any,
    device: Any,
) -> dict[str, object]:
    """Measure separate fixed-batch task gradients without an optimizer update."""
    base_model = getattr(model, "module", model)
    terms, prediction, temporal = _batch_terms(
        model=base_model,
        batch=batch,
        normalization=normalization,
        fk=fk,
        device=device,
        capture_activations=True,
    )
    taps = (
        "post_dino_dense_tokens",
        "normalized_upper_query",
        "normalized_left_query",
        "normalized_right_query",
    )
    losses = {
        "direct_xyz": terms.direct_xyz.total,
        "fk_xyz": terms.fk_xyz.total,
        "mhr_parameter": terms.mhr_parameter.total,
    }
    vectors: dict[str, dict[str, Any]] = {}
    for name, loss in losses.items():
        base_model.zero_grad(set_to_none=True)
        _clear_activation_gradients(prediction)
        loss.backward(retain_graph=True)
        vectors[name] = {
            tap: _gradient_vector(prediction=prediction, temporal=temporal, tap=tap) for tap in taps
        }
    base_model.zero_grad(set_to_none=True)
    _clear_activation_gradients(prediction)
    details: dict[str, object] = {}
    for tap in taps:
        direct = vectors["direct_xyz"][tap]
        fk_vector = vectors["fk_xyz"][tap]
        parameter = vectors["mhr_parameter"][tap]
        direct_norm = float(direct.norm().item())
        fk_norm = float(fk_vector.norm().item())
        parameter_norm = float(parameter.norm().item())
        denominator = max(direct_norm * fk_norm, 1e-20)
        details[tap] = {
            "direct_coordinate_gradient_norm": direct_norm,
            "fk_coordinate_gradient_norm": fk_norm,
            "mhr_parameter_gradient_norm": parameter_norm,
            "direct_fk_gradient_cosine": float(torch_dot(direct, fk_vector) / denominator),
            "direct_to_fk_gradient_norm_ratio": direct_norm / max(fk_norm, 1e-20),
            "valid_activation_elements": int(direct.numel()),
        }
    return {
        "fixed_train_only_batch": {
            "sample_ids": list(batch["sample_id"]),
            "segment_indices": [int(value) for value in batch["segment_index"]],
            "window_starts": [int(value) for value in batch["window_start"]],
            "excluded_from_optimizer_update": True,
        },
        "taps": details,
        "unweighted_losses": {
            "direct_xyz": float(terms.direct_xyz.total.detach().item()),
            "fk_xyz": float(terms.fk_xyz.total.detach().item()),
            "mhr_parameter": float(terms.mhr_parameter.total.detach().item()),
        },
    }


def torch_dot(first: Any, second: Any) -> float:
    import torch

    return float(torch.dot(first, second).item())


def _fixed_diagnostic_batch(dataset: WindowDataset, per_device_batch: int) -> dict[str, Any]:
    _require(len(dataset) >= per_device_batch, "diagnostic batch exceeds train support")
    return collate_windows(
        [dataset.fixed_window(index, start=0) for index in range(per_device_batch)]
    )
