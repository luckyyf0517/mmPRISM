from __future__ import annotations

import json
import math
import os
import random
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from mmprism.artifacts import RunArtifactWriter, sha256_file

TRAINING_STATE_SCHEMA_VERSION = "mmprism.training_state.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TrainingStateError(RuntimeError):
    """Raised when an epoch-boundary training state is incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class LoadedTrainingState:
    source_run_id: str
    completed_epoch: int
    global_step: int
    history: tuple[dict[str, object], ...]


def _utc_text(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise TrainingStateError("training-state timestamps must be timezone-aware")
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TrainingStateError(f"training-state metadata is not strict JSON: {error}") from error
    return (serialized + "\n").encode("utf-8")


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise TrainingStateError(
            f"training-state artifact already exists: {destination}"
        ) from error
    temporary.unlink()


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: object, location: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingStateError(f"{location} must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TrainingStateError(f"{location} keys must be non-empty strings")
            result[key] = _json_value(item, f"{location}.{key}")
        return dict(sorted(result.items()))
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)]
    raise TrainingStateError(f"{location} has unsupported value type {type(value).__name__}")


def _tuple_tree(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _parameter_inventory(
    named_parameters: Sequence[tuple[str, nn.Parameter]], optimizer: Optimizer
) -> tuple[dict[int, tuple[str, nn.Parameter]], list[dict[str, object]]]:
    if not named_parameters:
        raise TrainingStateError("training state requires optimizer parameters")
    names = [name for name, _ in named_parameters]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise TrainingStateError("optimizer parameter names must be non-empty and unique")
    by_id = {id(parameter): (name, parameter) for name, parameter in named_parameters}
    if len(by_id) != len(named_parameters):
        raise TrainingStateError("optimizer parameter inventory contains aliases")

    observed: list[int] = []
    groups: list[dict[str, object]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise TrainingStateError("optimizer parameter group params must be a list")
        group_names: list[str] = []
        for parameter in parameters:
            entry = by_id.get(id(parameter))
            if entry is None or entry[1] is not parameter:
                raise TrainingStateError("optimizer contains an unnamed parameter")
            observed.append(id(parameter))
            group_names.append(entry[0])
        options = {
            str(key): _json_value(value, f"optimizer.group[{group_index}].{key}")
            for key, value in group.items()
            if key != "params"
        }
        groups.append({"parameters": group_names, "options": dict(sorted(options.items()))})
    if len(observed) != len(set(observed)) or set(observed) != set(by_id):
        raise TrainingStateError("optimizer parameter groups do not exactly match named parameters")
    return by_id, groups


def _model_tensor_payload(
    model_state: Mapping[str, Tensor], tensors: dict[str, Tensor]
) -> dict[str, str]:
    if not model_state:
        raise TrainingStateError("training state requires model tensors")
    manifest: dict[str, str] = {}
    for index, name in enumerate(sorted(model_state)):
        tensor = model_state[name]
        if not isinstance(name, str) or not name or not isinstance(tensor, Tensor):
            raise TrainingStateError("model state must map non-empty names to tensors")
        key = f"model.{index:06d}"
        tensors[key] = tensor.detach().cpu().contiguous().clone()
        manifest[name] = key
    return manifest


def _optimizer_payload(
    optimizer: Optimizer,
    by_id: Mapping[int, tuple[str, nn.Parameter]],
    groups: list[dict[str, object]],
    tensors: dict[str, Tensor],
) -> dict[str, object]:
    states: dict[str, object] = {}
    for state_index, (parameter, state) in enumerate(
        sorted(optimizer.state.items(), key=lambda item: by_id[id(item[0])][0])
    ):
        entry = by_id.get(id(parameter))
        if entry is None or entry[1] is not parameter:
            raise TrainingStateError("optimizer state contains an unnamed parameter")
        parameter_name = entry[0]
        if not isinstance(state, Mapping):
            raise TrainingStateError(f"optimizer state for {parameter_name!r} must be a mapping")
        tensor_state: dict[str, object] = {}
        scalar_state: dict[str, object] = {}
        for value_index, (key, value) in enumerate(sorted(state.items())):
            if not isinstance(key, str) or not key:
                raise TrainingStateError("optimizer state keys must be non-empty strings")
            if isinstance(value, Tensor):
                tensor_key = f"optimizer.{state_index:06d}.{value_index:03d}"
                tensors[tensor_key] = value.detach().cpu().contiguous().clone()
                tensor_state[key] = {
                    "key": tensor_key,
                    "device_type": value.device.type,
                }
            else:
                scalar_state[key] = _json_value(
                    value, f"optimizer.state.{parameter_name}.{key}"
                )
        states[parameter_name] = {
            "tensors": tensor_state,
            "scalars": scalar_state,
        }
    optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    return {
        "type": optimizer_type,
        "parameter_groups": groups,
        "states": states,
    }


def _rng_payload(
    tensors: dict[str, Tensor],
    *,
    loader_generator: torch.Generator,
    device: torch.device,
) -> dict[str, object]:
    tensors["rng.torch_cpu"] = torch.get_rng_state().contiguous().clone()
    tensors["rng.loader"] = loader_generator.get_state().contiguous().clone()
    cuda_key: str | None = None
    if device.type == "cuda":
        cuda_key = "rng.torch_cuda"
        tensors[cuda_key] = torch.cuda.get_rng_state(device).cpu().contiguous().clone()

    numpy_state = np.random.get_state()
    return {
        "python": _json_value(random.getstate(), "rng.python"),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu_tensor": "rng.torch_cpu",
        "loader_tensor": "rng.loader",
        "torch_cuda_tensor": cuda_key,
        "device_type": device.type,
    }


def _validate_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    if not bindings:
        raise TrainingStateError("training state requires compatibility bindings")
    result: dict[str, str] = {}
    for key, value in bindings.items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise TrainingStateError("training-state bindings must map non-empty strings")
        result[key] = value
    return dict(sorted(result.items()))


def save_epoch_training_state(
    writer: RunArtifactWriter,
    *,
    model_state: Mapping[str, Tensor],
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    optimizer: Optimizer,
    scaler: GradScaler,
    loader_generator: torch.Generator,
    device: torch.device,
    bindings: Mapping[str, str],
    completed_epoch: int,
    global_step: int,
    configured_epochs: int,
    configured_max_steps: int | None,
    history: Sequence[Mapping[str, object]],
    created_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Publish and register one immutable, complete epoch-boundary state pair."""

    if isinstance(completed_epoch, bool) or completed_epoch < 1:
        raise TrainingStateError("completed_epoch must be a positive integer")
    if isinstance(global_step, bool) or global_step < 1:
        raise TrainingStateError("global_step must be a positive integer")
    if isinstance(configured_epochs, bool) or configured_epochs < completed_epoch:
        raise TrainingStateError("configured_epochs must cover completed_epoch")
    if configured_max_steps is not None and (
        isinstance(configured_max_steps, bool) or configured_max_steps < global_step
    ):
        raise TrainingStateError("configured_max_steps must cover global_step")
    if len(history) != completed_epoch:
        raise TrainingStateError("training history must contain every completed epoch")
    if not history or history[-1].get("epoch") != completed_epoch:
        raise TrainingStateError("training history does not end at completed_epoch")

    parameter_by_id, groups = _parameter_inventory(named_parameters, optimizer)
    tensors: dict[str, Tensor] = {}
    model_manifest = _model_tensor_payload(model_state, tensors)
    optimizer_manifest = _optimizer_payload(optimizer, parameter_by_id, groups, tensors)
    rng = _rng_payload(tensors, loader_generator=loader_generator, device=device)

    width = max(5, len(str(completed_epoch)))
    stem = f"training-state.epoch-{completed_epoch:0{width}d}"
    tensor_name = f"{stem}.safetensors"
    metadata_name = f"{stem}.json"
    tensor_path = writer.artifact_path(tensor_name)
    metadata_path = writer.artifact_path(metadata_name)
    if tensor_path.exists() or metadata_path.exists():
        raise TrainingStateError(f"training-state epoch {completed_epoch} already exists")

    temporary = tensor_path.with_name(
        f".{tensor_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        save_file(
            tensors,
            temporary,
            metadata={
                "schema_version": TRAINING_STATE_SCHEMA_VERSION,
                "run_id": writer.run_id,
            },
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        tensor_sha256 = sha256_file(temporary)
        tensor_size = temporary.stat().st_size
        _publish_no_replace(temporary, tensor_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingStateError(f"unable to save training-state tensors: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)

    payload = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "source_run_id": writer.run_id,
        "created_at": _utc_text(created_at),
        "progress": {
            "completed_epoch": completed_epoch,
            "global_step": global_step,
            "configured_epochs": configured_epochs,
            "configured_max_steps": configured_max_steps,
            "resume_granularity": "completed_epoch",
        },
        "bindings": _validate_bindings(bindings),
        "tensors": {
            "filename": tensor_name,
            "format": "safetensors",
            "sha256": tensor_sha256,
            "size_bytes": tensor_size,
        },
        "model_tensors": model_manifest,
        "optimizer": optimizer_manifest,
        "scaler": _json_value(scaler.state_dict(), "scaler"),
        "rng": rng,
        "history": [dict(record) for record in history],
    }
    _write_json_no_replace(metadata_path, payload)
    writer.register_artifacts((tensor_name, metadata_name))
    return metadata_path, tensor_path


def _read_metadata(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        payload: object = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingStateError(f"unable to read training-state metadata: {error}") from error
    expected_keys = {
        "schema_version",
        "source_run_id",
        "created_at",
        "progress",
        "bindings",
        "tensors",
        "model_tensors",
        "optimizer",
        "scaler",
        "rng",
        "history",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise TrainingStateError("training-state metadata keys do not match the contract")
    if payload.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
        raise TrainingStateError("training-state metadata has an unsupported schema")
    if raw != _json_bytes(payload):
        raise TrainingStateError("training-state metadata is not canonical strict JSON")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise TrainingStateError("training-state created_at is invalid")
    try:
        datetime.fromisoformat(created_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise TrainingStateError("training-state created_at is invalid") from error
    return payload


def _integer(payload: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingStateError(f"training-state {key} must be an integer >= {minimum}")
    return value


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingStateError(f"{location} must be a mapping")
    return value


def _validate_tensor_artifact(metadata: Mapping[str, Any], tensor_path: Path) -> None:
    tensor_artifact = _mapping(metadata.get("tensors"), "training-state tensors")
    if set(tensor_artifact) != {"filename", "format", "sha256", "size_bytes"}:
        raise TrainingStateError("training-state tensor metadata keys do not match the contract")
    if tensor_artifact.get("filename") != tensor_path.name:
        raise TrainingStateError("training-state tensor filename mismatch")
    if tensor_artifact.get("format") != "safetensors":
        raise TrainingStateError("training-state tensors must use Safetensors")
    expected_hash = tensor_artifact.get("sha256")
    if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
        raise TrainingStateError("training-state tensor SHA-256 is invalid")
    expected_size = _integer(tensor_artifact, "size_bytes", minimum=0)
    before = tensor_path.stat()
    observed_hash = sha256_file(tensor_path)
    after = tensor_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise TrainingStateError("training-state tensors changed while hashing")
    if observed_hash != expected_hash or after.st_size != expected_size:
        raise TrainingStateError("training-state tensor checksum or size mismatch")


def _restore_model(
    model: nn.Module,
    tensors: Mapping[str, Tensor],
    manifest_value: object,
    expected_model_state_names: set[str],
) -> set[str]:
    manifest = _mapping(manifest_value, "training-state model_tensors")
    if set(manifest) != expected_model_state_names:
        raise TrainingStateError("training-state model tensor inventory is incompatible")
    tensor_keys = list(manifest.values())
    if any(not isinstance(key, str) for key in tensor_keys) or len(set(tensor_keys)) != len(
        tensor_keys
    ):
        raise TrainingStateError("training-state model tensor references must be unique strings")
    state: dict[str, Tensor] = {}
    for name, tensor_key in manifest.items():
        if not isinstance(tensor_key, str) or tensor_key not in tensors:
            raise TrainingStateError(f"training-state model tensor is missing for {name!r}")
        state[name] = tensors[tensor_key]
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise TrainingStateError(f"unable to restore model training state: {error}") from error
    allowed_missing = set(model.state_dict()) - expected_model_state_names
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise TrainingStateError("training-state model keys are incompatible")
    return set(tensor_keys)


def _current_group_payload(
    optimizer: Optimizer, by_id: Mapping[int, tuple[str, nn.Parameter]]
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group["params"]
        names = [by_id[id(parameter)][0] for parameter in parameters]
        options = {
            str(key): _json_value(value, f"optimizer.group[{group_index}].{key}")
            for key, value in group.items()
            if key != "params"
        }
        groups.append({"parameters": names, "options": dict(sorted(options.items()))})
    return groups


def _restore_optimizer(
    optimizer: Optimizer,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    tensors: Mapping[str, Tensor],
    optimizer_value: object,
) -> set[str]:
    by_id, current_groups = _parameter_inventory(named_parameters, optimizer)
    by_name = {name: parameter for name, parameter in named_parameters}
    payload = _mapping(optimizer_value, "training-state optimizer")
    if set(payload) != {"type", "parameter_groups", "states"}:
        raise TrainingStateError("training-state optimizer keys do not match the contract")
    optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    if payload.get("type") != optimizer_type:
        raise TrainingStateError("training-state optimizer type is incompatible")
    if payload.get("parameter_groups") != current_groups:
        raise TrainingStateError("training-state optimizer parameter groups are incompatible")
    states = _mapping(payload.get("states"), "training-state optimizer states")
    if not set(states).issubset(by_name):
        raise TrainingStateError("training-state optimizer names are incompatible")

    restored: dict[Tensor, dict[str, object]] = {}
    referenced_tensors: set[str] = set()
    for name, state_value in states.items():
        state = _mapping(state_value, f"training-state optimizer state {name}")
        if set(state) != {"tensors", "scalars"}:
            raise TrainingStateError("training-state optimizer state keys are invalid")
        tensor_state = _mapping(state.get("tensors"), f"optimizer state tensors {name}")
        scalar_state = _mapping(state.get("scalars"), f"optimizer state scalars {name}")
        values: dict[str, object] = dict(scalar_state)
        parameter = by_name[name]
        for key, tensor_value in tensor_state.items():
            tensor_metadata = _mapping(tensor_value, f"optimizer tensor {name}.{key}")
            if set(tensor_metadata) != {"key", "device_type"}:
                raise TrainingStateError("optimizer tensor metadata keys are invalid")
            tensor_key = tensor_metadata.get("key")
            device_type = tensor_metadata.get("device_type")
            if not isinstance(tensor_key, str) or tensor_key not in tensors:
                raise TrainingStateError(f"optimizer tensor is missing for {name}.{key}")
            if tensor_key in referenced_tensors:
                raise TrainingStateError("optimizer tensor references must be unique")
            if device_type not in {"cpu", "cuda"}:
                raise TrainingStateError("optimizer tensor device type is invalid")
            target_device = torch.device("cpu") if device_type == "cpu" else parameter.device
            values[key] = tensors[tensor_key].to(target_device).clone()
            referenced_tensors.add(tensor_key)
        restored[parameter] = values
    optimizer.state.clear()
    optimizer.state.update(restored)
    del by_id
    return referenced_tensors


def _restore_rng(
    tensors: Mapping[str, Tensor],
    rng_value: object,
    *,
    loader_generator: torch.Generator,
    device: torch.device,
) -> set[str]:
    rng = _mapping(rng_value, "training-state rng")
    expected_keys = {
        "python",
        "numpy",
        "torch_cpu_tensor",
        "loader_tensor",
        "torch_cuda_tensor",
        "device_type",
    }
    if set(rng) != expected_keys or rng.get("device_type") != device.type:
        raise TrainingStateError("training-state RNG device or keys are incompatible")
    cpu_key = rng.get("torch_cpu_tensor")
    loader_key = rng.get("loader_tensor")
    cuda_key = rng.get("torch_cuda_tensor")
    if not isinstance(cpu_key, str) or not isinstance(loader_key, str):
        raise TrainingStateError("training-state RNG tensor references are invalid")
    if cpu_key not in tensors or loader_key not in tensors:
        raise TrainingStateError("training-state RNG tensors are missing")
    if device.type == "cuda":
        if not isinstance(cuda_key, str) or cuda_key not in tensors:
            raise TrainingStateError("training-state CUDA RNG tensor is missing")
    elif cuda_key is not None:
        raise TrainingStateError("CPU training state must not contain a CUDA RNG tensor")

    numpy_value = _mapping(rng.get("numpy"), "training-state NumPy RNG")
    if set(numpy_value) != {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise TrainingStateError("training-state NumPy RNG keys are invalid")
    algorithm = numpy_value.get("algorithm")
    keys = numpy_value.get("keys")
    if not isinstance(algorithm, str) or not isinstance(keys, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in keys
    ):
        raise TrainingStateError("training-state NumPy RNG values are invalid")
    position = _integer(numpy_value, "position", minimum=0)
    has_gauss = _integer(numpy_value, "has_gauss", minimum=0)
    cached_gaussian = numpy_value.get("cached_gaussian")
    if not isinstance(cached_gaussian, (int, float)) or isinstance(cached_gaussian, bool):
        raise TrainingStateError("training-state NumPy cached Gaussian is invalid")

    try:
        python_state = _tuple_tree(rng.get("python"))
        if not isinstance(python_state, tuple):
            raise TypeError("Python RNG state is not a tuple")
        random.setstate(python_state)
        np.random.set_state(
            (
                algorithm,
                np.asarray(keys, dtype=np.uint32),
                position,
                has_gauss,
                float(cached_gaussian),
            )
        )
        torch.set_rng_state(tensors[cpu_key].cpu())
        loader_generator.set_state(tensors[loader_key].cpu())
        if device.type == "cuda":
            assert isinstance(cuda_key, str)
            torch.cuda.set_rng_state(tensors[cuda_key].cpu(), device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingStateError(f"unable to restore RNG state: {error}") from error
    return {cpu_key, loader_key, *(set() if cuda_key is None else {cuda_key})}


def load_epoch_training_state(
    metadata_path: str | Path,
    tensor_path: str | Path,
    *,
    model: nn.Module,
    expected_model_state_names: set[str],
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    optimizer: Optimizer,
    scaler: GradScaler,
    loader_generator: torch.Generator,
    device: torch.device,
    expected_bindings: Mapping[str, str],
    target_epochs: int,
    target_max_steps: int | None,
) -> LoadedTrainingState:
    """Validate and restore one complete epoch boundary without pickle deserialization."""

    metadata_source = Path(metadata_path).expanduser().resolve()
    tensor_source = Path(tensor_path).expanduser().resolve()
    metadata = _read_metadata(metadata_source)
    _validate_tensor_artifact(metadata, tensor_source)
    if metadata.get("bindings") != _validate_bindings(expected_bindings):
        raise TrainingStateError("training-state compatibility bindings do not match")

    source_run_id = metadata.get("source_run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise TrainingStateError("training-state source_run_id is invalid")
    progress = _mapping(metadata.get("progress"), "training-state progress")
    if set(progress) != {
        "completed_epoch",
        "global_step",
        "configured_epochs",
        "configured_max_steps",
        "resume_granularity",
    }:
        raise TrainingStateError("training-state progress keys do not match the contract")
    if progress.get("resume_granularity") != "completed_epoch":
        raise TrainingStateError("unsupported training-state resume granularity")
    completed_epoch = _integer(progress, "completed_epoch", minimum=1)
    global_step = _integer(progress, "global_step", minimum=1)
    configured_epochs = _integer(progress, "configured_epochs", minimum=1)
    configured_max_steps_value = progress.get("configured_max_steps")
    if configured_max_steps_value is None:
        configured_max_steps = None
    else:
        configured_max_steps = _integer(progress, "configured_max_steps", minimum=1)
    if completed_epoch > configured_epochs:
        raise TrainingStateError("training-state completed epoch exceeds its configured target")
    if configured_max_steps is not None and global_step > configured_max_steps:
        raise TrainingStateError("training-state global step exceeds its configured target")
    if isinstance(target_epochs, bool) or target_epochs < configured_epochs:
        raise TrainingStateError("resume epoch target may only increase")
    if target_max_steps is not None and (
        isinstance(target_max_steps, bool) or target_max_steps < 1
    ):
        raise TrainingStateError("resume step target must be a positive integer or null")
    if configured_max_steps is None:
        if target_max_steps is not None:
            raise TrainingStateError("resume step target may only increase")
    elif target_max_steps is not None and target_max_steps < configured_max_steps:
        raise TrainingStateError("resume step target may only increase")
    history_value = metadata.get("history")
    if not isinstance(history_value, list) or len(history_value) != completed_epoch:
        raise TrainingStateError("training-state history length is invalid")
    if any(not isinstance(record, Mapping) for record in history_value):
        raise TrainingStateError("training-state history records must be mappings")
    history = tuple(dict(record) for record in history_value)
    if not history or history[-1].get("epoch") != completed_epoch:
        raise TrainingStateError("training-state history does not end at completed_epoch")

    try:
        tensors = load_file(tensor_source, device="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingStateError(f"unable to load training-state tensors: {error}") from error
    model_tensor_keys = _restore_model(
        model, tensors, metadata.get("model_tensors"), expected_model_state_names
    )
    optimizer_tensor_keys = _restore_optimizer(
        optimizer, named_parameters, tensors, metadata.get("optimizer")
    )
    scaler_value = _mapping(metadata.get("scaler"), "training-state scaler")
    try:
        scaler.load_state_dict(dict(scaler_value))
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingStateError(f"unable to restore gradient scaler: {error}") from error
    rng_tensor_keys = _restore_rng(
        tensors,
        metadata.get("rng"),
        loader_generator=loader_generator,
        device=device,
    )
    referenced_groups = (model_tensor_keys, optimizer_tensor_keys, rng_tensor_keys)
    referenced = set().union(*referenced_groups)
    reference_count = sum(len(group) for group in referenced_groups)
    if reference_count != len(referenced) or referenced != set(tensors):
        raise TrainingStateError("training-state tensor inventory is not exact")
    return LoadedTrainingState(
        source_run_id=source_run_id,
        completed_epoch=completed_epoch,
        global_step=global_step,
        history=history,
    )
