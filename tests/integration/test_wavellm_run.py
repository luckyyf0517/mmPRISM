from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file

transformers = pytest.importorskip("transformers")

from mmprism.assets import ResolvedModelAsset, load_model_asset_config  # noqa: E402
from mmprism.config import load_experiment_config  # noqa: E402
from mmprism.models import GeometryGuidedMT5  # noqa: E402
from mmprism.runtime import discover_project_root  # noqa: E402
from mmprism.training import WaveLLMRunError, load_wavellm_run_config  # noqa: E402
from mmprism.training import wavellm_run as run_module  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    data_root: Path,
    *,
    sample_id: str,
    seed: int,
    frames: int,
    input_mode: str = "pose_plus_radar_feature",
) -> dict[str, object]:
    array_root = data_root / "arrays"
    array_root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    arrays = {
        "pose": (generator.normal(size=(frames, 2, 24, 3)) * 0.01).astype(np.float32),
        "pose_confidence": generator.random((frames, 2, 24), dtype=np.float32),
        "frame_mask": np.ones(frames, dtype=np.bool_),
    }
    if input_mode == "pose_plus_radar_feature":
        arrays["radar_feature"] = generator.normal(size=(frames, 16)).astype(np.float32)
    paths: dict[str, Path] = {}
    for name, array in arrays.items():
        path = array_root / f"{sample_id}.{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = path
    modalities: dict[str, object] = {
        name: {
            "uri": path.relative_to(data_root).as_posix(),
            "shape": list(arrays[name].shape),
            "dtype": str(arrays[name].dtype),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    modalities["caption"] = {"text": f"target {sample_id}"}
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": f"sequence-{sample_id}",
        "subject_id": f"subject-{sample_id}",
        "dataset": "wavellm-run-fixture",
        "modalities": modalities,
        "acquisition": {
            "sample_protocol": "mmprism.sign_language_translation.sample_v2",
            "input_mode": input_mode,
            "pose_units": "m",
            "pose_coordinate_frame": "fixture_radar_cartesian_v1",
            **(
                {"radar_feature_protocol": "mmprism.radar_feature.sequence_v1"}
                if input_mode == "pose_plus_radar_feature"
                else {}
            ),
        },
        "provenance": {"purpose": "integration-test"},
    }


def _manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _split_assignments(path: Path, assignments: dict[str, str]) -> Path:
    records = (
        {
            "schema_version": "mmprism.split_assignment.v1",
            "sample_id": sample_id,
            "group_id": f"{index + 1:064x}",
            "split": split,
        }
        for index, (sample_id, split) in enumerate(sorted(assignments.items()))
    )
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _experiment(path: Path, root: Path, *, name: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mmprism.experiment.v1",
                "name": name,
                "task": "sign_language_translation",
                "paths": {
                    "data_root": str(root / "data"),
                    "artifact_root": str(root / "artifacts"),
                    "cache_root": str(root / "cache"),
                },
                "runtime": {
                    "seed": 23,
                    "accelerator": "cpu",
                    "devices": "auto",
                    "precision": "32-true",
                    "deterministic": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _task_config(
    path: Path,
    *,
    epochs: int = 1,
    max_steps: int | None = 1,
    input_mode: str = "pose_plus_radar_feature",
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mmprism.wavellm_run.v2",
                "input_mode": input_mode,
                "model": {
                    "asset_id": "tiny_mt5",
                    "hidden_size": 32,
                    "radar_feature_dim": 16,
                    "joint_count": 24,
                    "coordinate_dim": 3,
                    "pose_channels": [8, 16],
                    "temporal_kernel_size": 3,
                    "dropout": 0.0,
                    "label_smoothing": 0.0,
                    "freeze_language_model": True,
                },
                "data": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "verify_checksums": True,
                    "shuffle": True,
                    "max_frames": 3,
                    "prompt": "Translate signs:",
                    "max_prompt_length": 16,
                    "max_target_length": 16,
                },
                "optimization": {
                    "epochs": epochs,
                    "max_steps": max_steps,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "beta1": 0.9,
                    "beta2": 0.98,
                    "gradient_clip_norm": 1.0,
                },
                "generation": {"max_new_tokens": 3, "num_beams": 1},
                "evaluation": {"save_references": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if input_mode == "pose_only":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        model = payload["model"]
        assert isinstance(model, dict)
        model.pop("radar_feature_dim")
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _asset_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mmprism.model_asset_config.v1",
                "asset_set_id": "tiny_mt5_v1",
                "models": [
                    {
                        "asset_id": "tiny_mt5",
                        "repo_id": "tests/tiny-mt5",
                        "revision": "b" * 40,
                        "destination": "tiny_mt5",
                        "loader": "transformers_mt5",
                        "required_files": ["config.json"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding and truncation and return_tensors == "pt"
        encoded = [
            [2 + (ord(character) % 60) for character in text][: max_length - 1] + [1]
            for text in texts
        ]
        width = max(len(row) for row in encoded)
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(encoded):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(
        self, token_ids: torch.Tensor, *, skip_special_tokens: bool
    ) -> list[str]:
        assert skip_special_tokens
        return [
            " ".join(str(int(token)) for token in row if int(token) > 1)
            for row in token_ids.detach().cpu()
        ]


def _fake_model_and_tokenizer(
    asset: ResolvedModelAsset,
    config: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GeometryGuidedMT5, _TinyTokenizer]:
    del asset
    language_config = transformers.MT5Config(
        vocab_size=64,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    language_model = transformers.MT5ForConditionalGeneration(language_config)
    model = GeometryGuidedMT5(
        language_model,
        hidden_size=config.model.hidden_size,
        input_mode=config.input_mode,
        radar_feature_dim=config.model.radar_feature_dim,
        joint_count=config.model.joint_count,
        coordinate_dim=config.model.coordinate_dim,
        pose_channels=config.model.pose_channels,
        temporal_kernel_size=config.model.temporal_kernel_size,
        dropout=config.model.dropout,
        label_smoothing=config.model.label_smoothing,
    )
    model.language_model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model, _TinyTokenizer()


def _runtime(project_root: Path) -> dict[str, object]:
    return {
        "python": "3.12.test",
        "platform": "test",
        "project_root": str(project_root),
        "git": {"commit": "a" * 40, "dirty": False},
        "packages": {"torch": "test", "safetensors": "test", "transformers": "test"},
    }


def test_formal_wavellm_train_adapter_prediction_and_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = discover_project_root(Path(__file__))
    data_root = tmp_path / "data"
    train_manifest = _manifest(
        tmp_path / "train.jsonl",
        [
            _record(data_root, sample_id="train-001", seed=1, frames=3),
            _record(data_root, sample_id="train-002", seed=2, frames=2),
        ],
    )
    validation_manifest = _manifest(
        tmp_path / "validation.jsonl",
        [
            _record(data_root, sample_id="validation-001", seed=3, frames=3),
            _record(data_root, sample_id="validation-002", seed=4, frames=2),
        ],
    )
    split_assignments = _split_assignments(
        tmp_path / "assignments.jsonl",
        {
            "train-001": "train",
            "train-002": "train",
            "validation-001": "validation",
            "validation-002": "validation",
        },
    )
    task_path = _task_config(tmp_path / "wavellm.yaml")
    task_config = load_wavellm_run_config(task_path)
    asset_path = _asset_config(tmp_path / "assets.yaml")
    asset_config = load_model_asset_config(asset_path)
    model_root = tmp_path / "models"
    model_path = model_root / "tiny_mt5"
    model_path.mkdir(parents=True)
    asset_manifest = model_path / "mmprism_model_asset.json"
    collection_manifest = model_root / "mmprism_model_assets.json"
    asset_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    collection_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    resolved_asset = ResolvedModelAsset(
        spec=asset_config.models[0],
        path=model_path,
        verification={"asset_manifest_sha256": _sha256(asset_manifest)},
        collection_manifest_sha256=_sha256(collection_manifest),
    )
    monkeypatch.setattr(run_module, "resolve_model_asset", lambda *args: resolved_asset)
    monkeypatch.setattr(run_module, "_load_model_and_tokenizer", _fake_model_and_tokenizer)

    train_experiment_path = _experiment(
        tmp_path / "train-experiment.yaml", tmp_path, name="wavellm-train-fixture"
    )
    train_result = run_module.train_wavellm(
        load_experiment_config(train_experiment_path),
        task_config,
        asset_config,
        model_root,
        source_experiment_config=train_experiment_path,
        source_task_config=task_path,
        source_asset_config=asset_path,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        project_root=project_root,
        command=("mmprism", "wavellm-train", "fixture"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 21, 0, tzinfo=UTC),
    )

    train_run = Path(str(train_result["run_dir"]))
    expected_artifacts = {
        "checkpoint.json",
        "checkpoint.safetensors",
        "config.resolved.json",
        "environment.json",
        "history.json",
        "inputs.json",
        "metrics.json",
        "performance.json",
        "predictions.index.json",
        "predictions.jsonl",
        "predictions.rank-00000-of-00001.json",
        "predictions.rank-00000-of-00001.jsonl",
        "run.json",
        "training-state.epoch-00001.json",
        "training-state.epoch-00001.safetensors",
        "wavellm.resolved.json",
        "wavellm.runtime.json",
    }
    assert {path.name for path in train_run.iterdir()} == expected_artifacts
    train_inputs = json.loads((train_run / "inputs.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in train_inputs["inputs"]} == {
        "model_asset_collection",
        "model_asset_config",
        "model_asset_manifest",
        "split_assignments",
        "train_manifest",
        "validation_manifest",
        "wavellm_config",
    }
    run_payload = json.loads((train_run / "run.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads(
        (train_run / "checkpoint.json").read_text(encoding="utf-8")
    )
    prediction_index = json.loads(
        (train_run / "predictions.index.json").read_text(encoding="utf-8")
    )
    train_predictions = (train_run / "predictions.jsonl").read_bytes()
    checkpoint_state = load_file(train_run / "checkpoint.safetensors")
    assert run_payload["status"] == "completed"
    assert prediction_index["prediction_schema"] == "mmprism.translation_prediction.v1"
    assert prediction_index["record_count"] == 2
    assert prediction_index["coverage"] == {
        "expected": 2,
        "extra": 0,
        "missing": 0,
        "observed": 2,
    }
    assert checkpoint_payload["weights"]["scope"] == "adapter_only"
    assert checkpoint_payload["weights"]["sha256"] == _sha256(
        train_run / "checkpoint.safetensors"
    )
    assert checkpoint_state
    assert not any(name.startswith("language_model.") for name in checkpoint_state)
    performance = json.loads((train_run / "performance.json").read_text(encoding="utf-8"))
    assert performance["mode"] == "train"
    assert performance["optimizer_steps"] == 1
    assert performance["optimizer_steps_this_run"] == 1
    assert performance["prediction_samples"] == 2
    assert performance["parameter_count"]["trainable"] > 0
    assert performance["parameter_count"]["frozen"] > 0
    training_state = json.loads(
        (train_run / "training-state.epoch-00001.json").read_text(encoding="utf-8")
    )
    assert training_state["schema_version"] == "mmprism.training_state.v1"
    assert training_state["progress"] == {
        "completed_epoch": 1,
        "configured_epochs": 1,
        "configured_max_steps": 1,
        "global_step": 1,
        "resume_granularity": "completed_epoch",
    }

    evaluation_experiment_path = _experiment(
        tmp_path / "evaluation-experiment.yaml", tmp_path, name="wavellm-eval-fixture"
    )
    evaluation_result = run_module.evaluate_wavellm(
        load_experiment_config(evaluation_experiment_path),
        task_config,
        asset_config,
        model_root,
        source_experiment_config=evaluation_experiment_path,
        source_task_config=task_path,
        source_asset_config=asset_path,
        manifest_path=validation_manifest,
        checkpoint_path=train_run / "checkpoint.safetensors",
        checkpoint_metadata_path=train_run / "checkpoint.json",
        split_assignments_path=split_assignments,
        split="validation",
        project_root=project_root,
        command=("mmprism", "wavellm-evaluate", "fixture"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 21, 5, tzinfo=UTC),
    )

    evaluation_run = Path(str(evaluation_result["run_dir"]))
    evaluation_inputs = json.loads(
        (evaluation_run / "inputs.json").read_text(encoding="utf-8")
    )
    evaluation_metrics = json.loads(
        (evaluation_run / "metrics.json").read_text(encoding="utf-8")
    )
    assert {item["name"] for item in evaluation_inputs["inputs"]} == {
        "checkpoint_metadata",
        "checkpoint_weights",
        "evaluation_manifest",
        "model_asset_collection",
        "model_asset_config",
        "model_asset_manifest",
        "split_assignments",
        "wavellm_config",
    }
    assert evaluation_metrics["protocol_id"] == "mmprism.language_metric.character_v1"
    assert evaluation_metrics["split"] == "validation"
    assert evaluation_metrics["sample_count"] == 2
    assert (evaluation_run / "predictions.jsonl").read_bytes() == train_predictions
    assert evaluation_result["metrics"] == evaluation_metrics["values"]

    tampered_checkpoint = tmp_path / "tampered.safetensors"
    tampered_checkpoint.write_bytes(
        (train_run / "checkpoint.safetensors").read_bytes() + b"x"
    )
    tampered_experiment_path = _experiment(
        tmp_path / "tampered-experiment.yaml", tmp_path, name="wavellm-eval-tampered"
    )
    with pytest.raises(WaveLLMRunError, match="SHA-256 mismatch"):
        run_module.evaluate_wavellm(
            load_experiment_config(tampered_experiment_path),
            task_config,
            asset_config,
            model_root,
            source_experiment_config=tampered_experiment_path,
            source_task_config=task_path,
            source_asset_config=asset_path,
            manifest_path=validation_manifest,
            checkpoint_path=tampered_checkpoint,
            checkpoint_metadata_path=train_run / "checkpoint.json",
            split_assignments_path=split_assignments,
            split="validation",
            project_root=project_root,
            command=("mmprism", "wavellm-evaluate", "tampered"),
            runtime_report=_runtime(project_root),
            created_at=datetime(2026, 8, 11, 21, 10, tzinfo=UTC),
        )
    failed_runs = list((tmp_path / "artifacts" / "wavellm-eval-tampered").iterdir())
    assert len(failed_runs) == 1
    failed_payload = json.loads((failed_runs[0] / "run.json").read_text(encoding="utf-8"))
    assert failed_payload["status"] == "failed"
    assert "SHA-256 mismatch" in failed_payload["failure"]


def test_pose_only_wavellm_train_evaluate_and_checkpoint_mode_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = discover_project_root(Path(__file__))
    data_root = tmp_path / "data"
    train_manifest = _manifest(
        tmp_path / "pose-only-train.jsonl",
        [
            _record(
                data_root,
                sample_id="pose-only-train-001",
                seed=11,
                frames=3,
                input_mode="pose_only",
            )
        ],
    )
    validation_manifest = _manifest(
        tmp_path / "pose-only-validation.jsonl",
        [
            _record(
                data_root,
                sample_id="pose-only-validation-001",
                seed=12,
                frames=2,
                input_mode="pose_only",
            )
        ],
    )
    split_assignments = _split_assignments(
        tmp_path / "pose-only-assignments.jsonl",
        {
            "pose-only-train-001": "train",
            "pose-only-validation-001": "validation",
        },
    )
    task_path = _task_config(
        tmp_path / "pose-only.yaml", input_mode="pose_only"
    )
    task_config = load_wavellm_run_config(task_path)
    asset_path = _asset_config(tmp_path / "assets.yaml")
    asset_config = load_model_asset_config(asset_path)
    model_root = tmp_path / "models"
    model_path = model_root / "tiny_mt5"
    model_path.mkdir(parents=True)
    asset_manifest = model_path / "mmprism_model_asset.json"
    collection_manifest = model_root / "mmprism_model_assets.json"
    asset_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    collection_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    resolved_asset = ResolvedModelAsset(
        spec=asset_config.models[0],
        path=model_path,
        verification={"asset_manifest_sha256": _sha256(asset_manifest)},
        collection_manifest_sha256=_sha256(collection_manifest),
    )
    monkeypatch.setattr(run_module, "resolve_model_asset", lambda *args: resolved_asset)
    monkeypatch.setattr(run_module, "_load_model_and_tokenizer", _fake_model_and_tokenizer)

    train_experiment = _experiment(
        tmp_path / "pose-only-train-experiment.yaml", tmp_path, name="wavellm-pose-only"
    )
    train_result = run_module.train_wavellm(
        load_experiment_config(train_experiment),
        task_config,
        asset_config,
        model_root,
        source_experiment_config=train_experiment,
        source_task_config=task_path,
        source_asset_config=asset_path,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        project_root=project_root,
        command=("mmprism", "wavellm-train", "pose-only"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 12, 9, 0, tzinfo=UTC),
    )
    train_run = Path(str(train_result["run_dir"]))
    checkpoint = json.loads((train_run / "checkpoint.json").read_text(encoding="utf-8"))
    state = load_file(train_run / "checkpoint.safetensors")

    assert checkpoint["input_mode"] == "pose_only"
    assert "radar_feature_dim" not in checkpoint["model"]
    assert not any(
        name.startswith(("radar_projector.", "fusion.")) for name in state
    )
    assert not list((data_root / "arrays").glob("*.radar_feature.npy"))

    evaluation_experiment = _experiment(
        tmp_path / "pose-only-eval-experiment.yaml", tmp_path, name="wavellm-pose-only-eval"
    )
    result = run_module.evaluate_wavellm(
        load_experiment_config(evaluation_experiment),
        task_config,
        asset_config,
        model_root,
        source_experiment_config=evaluation_experiment,
        source_task_config=task_path,
        source_asset_config=asset_path,
        manifest_path=validation_manifest,
        checkpoint_path=train_run / "checkpoint.safetensors",
        checkpoint_metadata_path=train_run / "checkpoint.json",
        split_assignments_path=split_assignments,
        split="validation",
        project_root=project_root,
        command=("mmprism", "wavellm-evaluate", "pose-only"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 12, 9, 5, tzinfo=UTC),
    )
    assert result["status"] == "completed"

    invalid_metadata = dict(checkpoint)
    invalid_metadata["input_mode"] = "pose_plus_radar_feature"
    invalid_metadata_path = tmp_path / "invalid-mode-checkpoint.json"
    invalid_metadata_path.write_text(
        json.dumps(invalid_metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    mismatch_experiment = _experiment(
        tmp_path / "mode-mismatch.yaml", tmp_path, name="wavellm-mode-mismatch"
    )
    with pytest.raises(WaveLLMRunError, match="input mode"):
        run_module.evaluate_wavellm(
            load_experiment_config(mismatch_experiment),
            task_config,
            asset_config,
            model_root,
            source_experiment_config=mismatch_experiment,
            source_task_config=task_path,
            source_asset_config=asset_path,
            manifest_path=validation_manifest,
            checkpoint_path=train_run / "checkpoint.safetensors",
            checkpoint_metadata_path=invalid_metadata_path,
            split_assignments_path=split_assignments,
            split="validation",
            project_root=project_root,
            command=("mmprism", "wavellm-evaluate", "mode-mismatch"),
            runtime_report=_runtime(project_root),
            created_at=datetime(2026, 8, 12, 9, 10, tzinfo=UTC),
        )


def test_wavellm_epoch_resume_matches_uninterrupted_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = discover_project_root(Path(__file__))
    data_root = tmp_path / "data"
    train_manifest = _manifest(
        tmp_path / "resume-train.jsonl",
        [
            _record(data_root, sample_id=f"train-{index:03d}", seed=index, frames=3)
            for index in range(1, 5)
        ],
    )
    validation_manifest = _manifest(
        tmp_path / "resume-validation.jsonl",
        [
            _record(data_root, sample_id=f"validation-{index:03d}", seed=index + 10, frames=3)
            for index in range(1, 3)
        ],
    )
    split_assignments = _split_assignments(
        tmp_path / "resume-assignments.jsonl",
        {
            **{f"train-{index:03d}": "train" for index in range(1, 5)},
            **{f"validation-{index:03d}": "validation" for index in range(1, 3)},
        },
    )
    segment_task_path = _task_config(
        tmp_path / "wavellm-segment.yaml", epochs=1, max_steps=None
    )
    target_task_path = _task_config(
        tmp_path / "wavellm-target.yaml", epochs=2, max_steps=None
    )
    asset_path = _asset_config(tmp_path / "assets.yaml")
    asset_config = load_model_asset_config(asset_path)
    model_root = tmp_path / "models"
    model_path = model_root / "tiny_mt5"
    model_path.mkdir(parents=True)
    asset_manifest = model_path / "mmprism_model_asset.json"
    collection_manifest = model_root / "mmprism_model_assets.json"
    asset_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    collection_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    resolved_asset = ResolvedModelAsset(
        spec=asset_config.models[0],
        path=model_path,
        verification={"asset_manifest_sha256": _sha256(asset_manifest)},
        collection_manifest_sha256=_sha256(collection_manifest),
    )
    monkeypatch.setattr(run_module, "resolve_model_asset", lambda *args: resolved_asset)
    monkeypatch.setattr(run_module, "_load_model_and_tokenizer", _fake_model_and_tokenizer)
    segment_experiment = _experiment(
        tmp_path / "segment-experiment.yaml", tmp_path, name="wavellm-resume-segment"
    )
    full_experiment = _experiment(
        tmp_path / "full-experiment.yaml", tmp_path, name="wavellm-resume-full"
    )
    resumed_experiment = _experiment(
        tmp_path / "resumed-experiment.yaml", tmp_path, name="wavellm-resume-restored"
    )

    segment = run_module.train_wavellm(
        load_experiment_config(segment_experiment),
        load_wavellm_run_config(segment_task_path),
        asset_config,
        model_root,
        source_experiment_config=segment_experiment,
        source_task_config=segment_task_path,
        source_asset_config=asset_path,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        project_root=project_root,
        command=("mmprism", "wavellm-train", "segment"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
    )
    full = run_module.train_wavellm(
        load_experiment_config(full_experiment),
        load_wavellm_run_config(target_task_path),
        asset_config,
        model_root,
        source_experiment_config=full_experiment,
        source_task_config=target_task_path,
        source_asset_config=asset_path,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        project_root=project_root,
        command=("mmprism", "wavellm-train", "full"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 23, 5, tzinfo=UTC),
    )
    segment_run = Path(str(segment["run_dir"]))
    with pytest.raises(WaveLLMRunError, match="requires both"):
        run_module.train_wavellm(
            load_experiment_config(resumed_experiment),
            load_wavellm_run_config(target_task_path),
            asset_config,
            model_root,
            source_experiment_config=resumed_experiment,
            source_task_config=target_task_path,
            source_asset_config=asset_path,
            train_manifest_path=train_manifest,
            validation_manifest_path=validation_manifest,
            split_assignments_path=split_assignments,
            resume_state_metadata_path=segment_run / "training-state.epoch-00001.json",
            project_root=project_root,
            command=("mmprism", "wavellm-train", "incomplete-resume"),
            runtime_report=_runtime(project_root),
        )
    resumed = run_module.train_wavellm(
        load_experiment_config(resumed_experiment),
        load_wavellm_run_config(target_task_path),
        asset_config,
        model_root,
        source_experiment_config=resumed_experiment,
        source_task_config=target_task_path,
        source_asset_config=asset_path,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        resume_state_metadata_path=segment_run / "training-state.epoch-00001.json",
        resume_state_tensors_path=segment_run / "training-state.epoch-00001.safetensors",
        project_root=project_root,
        command=("mmprism", "wavellm-train", "resumed"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 23, 10, tzinfo=UTC),
    )

    full_run = Path(str(full["run_dir"]))
    resumed_run = Path(str(resumed["run_dir"]))
    full_state = load_file(full_run / "checkpoint.safetensors")
    resumed_state = load_file(resumed_run / "checkpoint.safetensors")
    assert set(full_state) == set(resumed_state)
    assert all(torch.equal(full_state[name], resumed_state[name]) for name in full_state)
    full_history = json.loads((full_run / "history.json").read_text(encoding="utf-8"))
    resumed_history = json.loads(
        (resumed_run / "history.json").read_text(encoding="utf-8")
    )
    assert resumed_history["records"] == full_history["records"]
    assert resumed_history["global_step"] == full_history["global_step"] == 4
    assert resumed_history["resumed_from_run_id"] == segment_run.name
    resumed_performance = json.loads(
        (resumed_run / "performance.json").read_text(encoding="utf-8")
    )
    assert resumed_performance["optimizer_steps"] == 4
    assert resumed_performance["optimizer_steps_this_run"] == 2
    resumed_inputs = json.loads((resumed_run / "inputs.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in resumed_inputs["inputs"]} >= {
        "resume_state_metadata",
        "resume_state_tensors",
    }


def test_wavellm_rejects_sequence_leakage_and_checkpoint_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = discover_project_root(Path(__file__))
    data_root = tmp_path / "data"
    train_record = _record(data_root, sample_id="train-001", seed=1, frames=2)
    validation_record = _record(data_root, sample_id="validation-001", seed=2, frames=2)
    validation_record["sequence_id"] = train_record["sequence_id"]
    train_manifest = _manifest(tmp_path / "train.jsonl", [train_record])
    leaking_manifest = _manifest(tmp_path / "leaking.jsonl", [validation_record])
    split_assignments = _split_assignments(
        tmp_path / "assignments.jsonl",
        {"train-001": "train", "validation-001": "validation"},
    )
    task_path = _task_config(tmp_path / "wavellm.yaml")
    task_config = load_wavellm_run_config(task_path)
    asset_path = _asset_config(tmp_path / "assets.yaml")
    asset_config = load_model_asset_config(asset_path)
    model_root = tmp_path / "models"
    model_path = model_root / "tiny_mt5"
    model_path.mkdir(parents=True)
    asset_manifest = model_path / "mmprism_model_asset.json"
    collection_manifest = model_root / "mmprism_model_assets.json"
    asset_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    collection_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    resolved_asset = ResolvedModelAsset(
        spec=asset_config.models[0],
        path=model_path,
        verification={"asset_manifest_sha256": _sha256(asset_manifest)},
        collection_manifest_sha256=_sha256(collection_manifest),
    )
    monkeypatch.setattr(run_module, "resolve_model_asset", lambda *args: resolved_asset)
    monkeypatch.setattr(run_module, "_load_model_and_tokenizer", _fake_model_and_tokenizer)

    leakage_experiment = _experiment(
        tmp_path / "leakage-experiment.yaml", tmp_path, name="wavellm-leakage-fixture"
    )
    with pytest.raises(WaveLLMRunError, match="sequence IDs"):
        run_module.train_wavellm(
            load_experiment_config(leakage_experiment),
            task_config,
            asset_config,
            model_root,
            source_experiment_config=leakage_experiment,
            source_task_config=task_path,
            source_asset_config=asset_path,
            train_manifest_path=train_manifest,
            validation_manifest_path=leaking_manifest,
            split_assignments_path=split_assignments,
            project_root=project_root,
            command=("mmprism", "wavellm-train", "leakage"),
            runtime_report=_runtime(project_root),
            created_at=datetime(2026, 8, 11, 22, 0, tzinfo=UTC),
        )

    validation_record["sequence_id"] = "sequence-validation-001"
    validation_manifest = _manifest(tmp_path / "validation.jsonl", [validation_record])
    train_experiment = _experiment(
        tmp_path / "train-experiment.yaml", tmp_path, name="wavellm-contract-train"
    )
    train_result = run_module.train_wavellm(
        load_experiment_config(train_experiment),
        task_config,
        asset_config,
        model_root,
        source_experiment_config=train_experiment,
        source_task_config=task_path,
        source_asset_config=asset_path,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        project_root=project_root,
        command=("mmprism", "wavellm-train", "contract"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 22, 5, tzinfo=UTC),
    )
    train_run = Path(str(train_result["run_dir"]))
    checkpoint_metadata = json.loads(
        (train_run / "checkpoint.json").read_text(encoding="utf-8")
    )
    checkpoint_metadata["pose_units"] = "mm"
    invalid_metadata = tmp_path / "checkpoint-invalid-units.json"
    invalid_metadata.write_text(
        json.dumps(checkpoint_metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    evaluation_experiment = _experiment(
        tmp_path / "evaluation-experiment.yaml", tmp_path, name="wavellm-contract-eval"
    )
    with pytest.raises(WaveLLMRunError, match="pose units"):
        run_module.evaluate_wavellm(
            load_experiment_config(evaluation_experiment),
            task_config,
            asset_config,
            model_root,
            source_experiment_config=evaluation_experiment,
            source_task_config=task_path,
            source_asset_config=asset_path,
            manifest_path=validation_manifest,
            checkpoint_path=train_run / "checkpoint.safetensors",
            checkpoint_metadata_path=invalid_metadata,
            split_assignments_path=split_assignments,
            split="validation",
            project_root=project_root,
            command=("mmprism", "wavellm-evaluate", "invalid-units"),
            runtime_report=_runtime(project_root),
            created_at=datetime(2026, 8, 11, 22, 10, tzinfo=UTC),
        )

    altered_task_payload = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    altered_task_payload["generation"]["max_new_tokens"] = 4
    altered_task_path = tmp_path / "wavellm-altered-generation.yaml"
    altered_task_path.write_text(
        yaml.safe_dump(altered_task_payload, sort_keys=False), encoding="utf-8"
    )
    altered_task_config = load_wavellm_run_config(altered_task_path)
    task_drift_experiment = _experiment(
        tmp_path / "task-drift-experiment.yaml", tmp_path, name="wavellm-task-drift-eval"
    )
    with pytest.raises(WaveLLMRunError, match="task configuration"):
        run_module.evaluate_wavellm(
            load_experiment_config(task_drift_experiment),
            altered_task_config,
            asset_config,
            model_root,
            source_experiment_config=task_drift_experiment,
            source_task_config=altered_task_path,
            source_asset_config=asset_path,
            manifest_path=validation_manifest,
            checkpoint_path=train_run / "checkpoint.safetensors",
            checkpoint_metadata_path=train_run / "checkpoint.json",
            split_assignments_path=split_assignments,
            split="validation",
            project_root=project_root,
            command=("mmprism", "wavellm-evaluate", "task-drift"),
            runtime_report=_runtime(project_root),
            created_at=datetime(2026, 8, 11, 22, 15, tzinfo=UTC),
        )
