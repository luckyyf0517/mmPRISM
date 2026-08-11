from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mmprism.config import load_experiment_config  # noqa: E402
from mmprism.runtime import discover_project_root  # noqa: E402
from mmprism.training import OmniHandRunError, load_omnihand_run_config  # noqa: E402
from mmprism.training.omnihand_run import evaluate_omnihand, train_omnihand  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(data_root: Path, *, sample_id: str, seed: int, frames: int) -> dict[str, object]:
    array_root = data_root / "arrays"
    array_root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    arrays = {
        "radar_cube": generator.random((frames, 4, 8, 6, 4), dtype=np.float32),
        "pose_gt": (generator.normal(size=(2, 24, 3)) * 0.01).astype(np.float32),
        "frame_mask": np.ones(frames, dtype=np.bool_),
        "pose_valid": np.ones((2, 24), dtype=np.bool_),
    }
    paths: dict[str, Path] = {}
    for name, array in arrays.items():
        path = array_root / f"{sample_id}.{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = path
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": f"sequence-{sample_id}",
        "subject_id": f"subject-{sample_id}",
        "dataset": "omnihand-run-fixture",
        "modalities": {
            name: {
                "uri": path.relative_to(data_root).as_posix(),
                "shape": list(arrays[name].shape),
                "dtype": str(arrays[name].dtype),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "acquisition": {
            "sample_protocol": "mmprism.pose_reconstruction.sample_v1",
            "radar_cube_protocol": "mmprism.radar_cube.power_v1",
            "pose_units": "m",
            "pose_coordinate_frame": "fixture_radar_cartesian_v1",
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
                "task": "pose_reconstruction",
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


def _task_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mmprism.omnihand_run.v1",
                "model": {
                    "spatial": {
                        "in_channels": 4,
                        "stem_channels": 8,
                        "stage_channels": [8, 16],
                        "stage_depths": [1, 1],
                        "channel_attention": True,
                        "spatial_attention": True,
                        "se_attention": True,
                        "use_pafpn": True,
                        "fpn_channels": 16,
                    },
                    "temporal": {
                        "max_frames": 3,
                        "layers": 1,
                        "heads": 4,
                        "feedforward_dim": 32,
                        "dropout": 0.0,
                    },
                    "joint_count": 24,
                    "coordinate_dim": 3,
                },
                "data": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "verify_checksums": True,
                    "shuffle": True,
                },
                "optimization": {
                    "epochs": 1,
                    "max_steps": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "beta1": 0.9,
                    "beta2": 0.98,
                    "gradient_clip_norm": 1.0,
                },
                "evaluation": {"pck_threshold_mm": 40.0, "save_targets": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _runtime(project_root: Path) -> dict[str, object]:
    return {
        "python": "3.12.test",
        "platform": "test",
        "project_root": str(project_root),
        "git": {"commit": "a" * 40, "dirty": False},
        "packages": {"torch": "test", "safetensors": "test"},
    }


def test_formal_omnihand_train_checkpoint_prediction_and_evaluate(tmp_path: Path) -> None:
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
    task_path = _task_config(tmp_path / "omnihand.yaml")
    task_config = load_omnihand_run_config(task_path)
    train_experiment_path = _experiment(
        tmp_path / "train-experiment.yaml", tmp_path, name="omnihand-train-fixture"
    )

    train_result = train_omnihand(
        load_experiment_config(train_experiment_path),
        task_config,
        source_experiment_config=train_experiment_path,
        source_task_config=task_path,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        project_root=project_root,
        command=("mmprism", "omnihand-train", "fixture"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 20, 0, tzinfo=UTC),
    )

    train_run = Path(str(train_result["run_dir"]))
    expected_train_artifacts = {
        "checkpoint.json",
        "checkpoint.safetensors",
        "config.resolved.json",
        "environment.json",
        "history.json",
        "inputs.json",
        "metrics.json",
        "omnihand.resolved.json",
        "omnihand.runtime.json",
        "performance.json",
        "predictions.jsonl",
        "run.json",
    }
    assert {path.name for path in train_run.iterdir()} == expected_train_artifacts
    train_inputs = json.loads((train_run / "inputs.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in train_inputs["inputs"]} == {
        "omnihand_config",
        "split_assignments",
        "train_manifest",
        "validation_manifest",
    }
    run_payload = json.loads((train_run / "run.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((train_run / "checkpoint.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (train_run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert run_payload["status"] == "completed"
    assert checkpoint_payload["weights"]["sha256"] == _sha256(train_run / "checkpoint.safetensors")
    assert checkpoint_payload["runtime"]["device"] == "cpu"
    performance = json.loads((train_run / "performance.json").read_text(encoding="utf-8"))
    assert performance["mode"] == "train"
    assert performance["optimizer_steps"] == 1
    assert performance["prediction_samples"] == 2
    assert performance["cuda_memory"] is None
    assert performance["end_to_end_seconds"] > 0
    assert [record["sample_id"] for record in predictions] == [
        "validation-001",
        "validation-002",
    ]
    assert all(record["target_m"] for record in predictions)

    evaluation_experiment_path = _experiment(
        tmp_path / "evaluation-experiment.yaml", tmp_path, name="omnihand-eval-fixture"
    )
    evaluation_result = evaluate_omnihand(
        load_experiment_config(evaluation_experiment_path),
        task_config,
        source_experiment_config=evaluation_experiment_path,
        source_task_config=task_path,
        manifest_path=validation_manifest,
        checkpoint_path=train_run / "checkpoint.safetensors",
        checkpoint_metadata_path=train_run / "checkpoint.json",
        split_assignments_path=split_assignments,
        split="validation",
        project_root=project_root,
        command=("mmprism", "omnihand-evaluate", "fixture"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 11, 20, 5, tzinfo=UTC),
    )

    evaluation_run = Path(str(evaluation_result["run_dir"]))
    evaluation_inputs = json.loads((evaluation_run / "inputs.json").read_text(encoding="utf-8"))
    evaluation_metrics = json.loads((evaluation_run / "metrics.json").read_text(encoding="utf-8"))
    input_names = {item["name"] for item in evaluation_inputs["inputs"]}
    assert input_names == {
        "checkpoint_metadata",
        "checkpoint_weights",
        "evaluation_manifest",
        "omnihand_config",
        "split_assignments",
    }
    assert evaluation_metrics["protocol_id"] == "mmprism.pose_metric.dual_hand_metric_v1"
    assert evaluation_metrics["split"] == "validation"
    assert evaluation_metrics["sample_count"] == 2
    assert evaluation_result["metrics"] == evaluation_metrics["values"]
    evaluation_performance = json.loads(
        (evaluation_run / "performance.json").read_text(encoding="utf-8")
    )
    assert evaluation_performance["mode"] == "evaluate"
    assert evaluation_performance["prediction_samples"] == 2

    tampered_checkpoint = tmp_path / "tampered.safetensors"
    tampered_checkpoint.write_bytes((train_run / "checkpoint.safetensors").read_bytes() + b"x")
    tampered_experiment_path = _experiment(
        tmp_path / "tampered-experiment.yaml", tmp_path, name="omnihand-eval-tampered"
    )
    with pytest.raises(OmniHandRunError, match="SHA-256 mismatch"):
        evaluate_omnihand(
            load_experiment_config(tampered_experiment_path),
            task_config,
            source_experiment_config=tampered_experiment_path,
            source_task_config=task_path,
            manifest_path=validation_manifest,
            checkpoint_path=tampered_checkpoint,
            checkpoint_metadata_path=train_run / "checkpoint.json",
            split_assignments_path=split_assignments,
            split="validation",
            project_root=project_root,
            command=("mmprism", "omnihand-evaluate", "tampered"),
            runtime_report=_runtime(project_root),
            created_at=datetime(2026, 8, 11, 20, 10, tzinfo=UTC),
        )
    failed_runs = list((tmp_path / "artifacts" / "omnihand-eval-tampered").iterdir())
    assert len(failed_runs) == 1
    failed_payload = json.loads((failed_runs[0] / "run.json").read_text(encoding="utf-8"))
    assert failed_payload["status"] == "failed"
    assert "SHA-256 mismatch" in failed_payload["failure"]
