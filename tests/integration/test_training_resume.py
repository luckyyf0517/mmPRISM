from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.amp.grad_scaler import GradScaler

from mmprism.artifacts import RunArtifactWriter
from mmprism.training.resume import (
    LoadedTrainingState,
    TrainingStateError,
    load_epoch_training_state,
    save_epoch_training_state,
)


class _TestWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_id = "resume-source-run"
        self.registered: tuple[str, ...] = ()
        run_dir.mkdir()

    def artifact_path(self, name: str) -> Path:
        return self.run_dir / name

    def register_artifacts(self, names: tuple[str, ...]) -> None:
        self.registered = names


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _trained_components() -> tuple[nn.Linear, torch.optim.AdamW, GradScaler, torch.Generator]:
    torch.manual_seed(19)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    generator = torch.Generator().manual_seed(23)
    return model, optimizer, GradScaler("cuda", enabled=False), generator


def _fresh_components() -> tuple[nn.Linear, torch.optim.AdamW, GradScaler, torch.Generator]:
    torch.manual_seed(29)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(31)
    return model, optimizer, GradScaler("cuda", enabled=False), generator


def _load(
    metadata_path: Path,
    tensor_path: Path,
    *,
    bindings: dict[str, str],
    target_epochs: int = 3,
    target_max_steps: int | None = 20,
) -> tuple[nn.Linear, LoadedTrainingState]:
    model, optimizer, scaler, generator = _fresh_components()
    state = load_epoch_training_state(
        metadata_path,
        tensor_path,
        model=model,
        expected_model_state_names=set(model.state_dict()),
        named_parameters=tuple(model.named_parameters()),
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=generator,
        device=torch.device("cpu"),
        expected_bindings=bindings,
        target_epochs=target_epochs,
        target_max_steps=target_max_steps,
    )
    return model, state


def _copy_state(metadata_path: Path, tensor_path: Path, destination: Path) -> tuple[Path, Path]:
    destination.mkdir()
    metadata_copy = destination / metadata_path.name
    tensor_copy = destination / tensor_path.name
    shutil.copy2(metadata_path, metadata_copy)
    shutil.copy2(tensor_path, tensor_copy)
    return metadata_copy, tensor_copy


def test_epoch_training_state_validates_compatibility_and_exact_tensor_inventory(
    tmp_path: Path,
) -> None:
    source_model, optimizer, scaler, generator = _trained_components()
    bindings = {"git_commit": "a" * 40, "task": "fixture"}
    writer = _TestWriter(tmp_path / "source")
    metadata_path, tensor_path = save_epoch_training_state(
        cast(RunArtifactWriter, writer),
        model_state=source_model.state_dict(),
        named_parameters=tuple(source_model.named_parameters()),
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=generator,
        device=torch.device("cpu"),
        bindings=bindings,
        completed_epoch=1,
        global_step=1,
        configured_epochs=2,
        configured_max_steps=10,
        history=({"epoch": 1, "global_step": 1},),
    )

    restored_model, state = _load(metadata_path, tensor_path, bindings=bindings)
    assert state.source_run_id == "resume-source-run"
    assert state.completed_epoch == 1
    assert state.global_step == 1
    assert all(
        torch.equal(source_model.state_dict()[name], restored_model.state_dict()[name])
        for name in source_model.state_dict()
    )
    assert writer.registered == (tensor_path.name, metadata_path.name)

    with pytest.raises(TrainingStateError, match="compatibility bindings"):
        _load(metadata_path, tensor_path, bindings={**bindings, "task": "changed"})
    with pytest.raises(TrainingStateError, match="epoch target may only increase"):
        _load(metadata_path, tensor_path, bindings=bindings, target_epochs=1)
    with pytest.raises(TrainingStateError, match="step target may only increase"):
        _load(metadata_path, tensor_path, bindings=bindings, target_max_steps=9)

    tampered_metadata, tampered_tensors = _copy_state(
        metadata_path, tensor_path, tmp_path / "tampered"
    )
    content = bytearray(tampered_tensors.read_bytes())
    content[-1] ^= 1
    tampered_tensors.write_bytes(content)
    with pytest.raises(TrainingStateError, match="checksum or size mismatch"):
        _load(tampered_metadata, tampered_tensors, bindings=bindings)

    extra_metadata, extra_tensors = _copy_state(
        metadata_path, tensor_path, tmp_path / "extra"
    )
    tensor_payload = load_file(extra_tensors)
    tensor_payload["unexpected"] = torch.ones(1)
    save_file(tensor_payload, extra_tensors)
    metadata_payload = json.loads(extra_metadata.read_text(encoding="utf-8"))
    metadata_payload["tensors"]["sha256"] = _sha256(extra_tensors)
    metadata_payload["tensors"]["size_bytes"] = extra_tensors.stat().st_size
    _canonical_write(extra_metadata, metadata_payload)
    with pytest.raises(TrainingStateError, match="tensor inventory is not exact"):
        _load(extra_metadata, extra_tensors, bindings=bindings)
