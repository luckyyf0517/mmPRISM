from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import yaml
from safetensors.torch import load_file

from mmprism.config import RuntimeConfig, load_experiment_config
from mmprism.runtime import discover_project_root
from mmprism.training import load_omnihand_run_config
from mmprism.training.distributed import (
    DistributedContext,
    DistributedRunError,
    ExactDistributedSampler,
    tensor_state_sha256,
)
from mmprism.training.omnihand_run import train_omnihand


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
    modalities: dict[str, object] = {}
    for name, array in arrays.items():
        path = array_root / f"{sample_id}.{name}.npy"
        np.save(path, array, allow_pickle=False)
        modalities[name] = {
            "uri": path.relative_to(data_root).as_posix(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(path),
        }
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": f"sequence-{sample_id}",
        "subject_id": f"subject-{sample_id}",
        "dataset": "distributed-omnihand-fixture",
        "modalities": modalities,
        "acquisition": {
            "sample_protocol": "mmprism.pose_reconstruction.sample_v1",
            "radar_cube_protocol": "mmprism.radar_cube.power_v1",
            "pose_units": "m",
            "pose_coordinate_frame": "fixture_radar_cartesian_v1",
        },
        "provenance": {"purpose": "distributed-integration-test"},
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _split_assignments(path: Path, assignments: dict[str, str]) -> Path:
    records = [
        {
            "schema_version": "mmprism.split_assignment.v1",
            "sample_id": sample_id,
            "group_id": f"{index + 1:064x}",
            "split": split,
        }
        for index, (sample_id, split) in enumerate(sorted(assignments.items()))
    ]
    return _write_jsonl(path, records)


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


def _task(path: Path, *, batch_size: int) -> Path:
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
                    "batch_size": batch_size,
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _distributed_worker(rank: int, world_size: int, port: int, payload: dict[str, str]) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    torch.set_num_threads(1)
    experiment_path = Path(payload["experiment"])
    task_path = Path(payload["task"])
    train_omnihand(
        load_experiment_config(experiment_path),
        load_omnihand_run_config(task_path),
        source_experiment_config=experiment_path,
        source_task_config=task_path,
        train_manifest_path=payload["train_manifest"],
        validation_manifest_path=payload["validation_manifest"],
        split_assignments_path=payload["split_assignments"],
        project_root=Path(payload["project_root"]),
        command=("torchrun", "-m", "mmprism", "omnihand-train", "fixture"),
        runtime_report=_runtime(Path(payload["project_root"])),
        created_at=datetime(2026, 8, 12, 1, 0, tzinfo=UTC),
    )


def test_distributed_runtime_rejects_partial_environment() -> None:
    runtime = RuntimeConfig(accelerator="cpu")
    with pytest.raises(DistributedRunError, match="missing LOCAL_RANK, WORLD_SIZE"):
        DistributedContext.from_environment(runtime, environment={"RANK": "0"})


def test_exact_distributed_sampler_has_no_padding() -> None:
    dataset = list(range(5))
    samplers = [
        ExactDistributedSampler(dataset, rank=rank, world_size=3) for rank in range(3)
    ]
    indices = [list(sampler) for sampler in samplers]
    assert indices == [[0, 3], [1, 4], [2]]
    assert sorted(index for shard in indices for index in shard) == list(range(5))


def test_tensor_state_hash_accepts_scalar_buffers() -> None:
    state = {
        "scalar": torch.tensor(7, dtype=torch.int64),
        "matrix": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }

    digest = tensor_state_sha256(state)

    assert len(digest) == 64
    assert digest == tensor_state_sha256({"matrix": state["matrix"], "scalar": state["scalar"]})


def test_two_process_gloo_training_publishes_one_formal_run(tmp_path: Path) -> None:
    project_root = discover_project_root(Path(__file__))
    data_root = tmp_path / "data"
    train_records = [
        _record(data_root, sample_id="train-001", seed=1, frames=3),
        _record(data_root, sample_id="train-002", seed=2, frames=3),
    ]
    validation_records = [
        _record(data_root, sample_id="validation-001", seed=3, frames=3),
        _record(data_root, sample_id="validation-002", seed=4, frames=2),
        _record(data_root, sample_id="validation-003", seed=5, frames=3),
    ]
    train_manifest = _write_jsonl(tmp_path / "train.jsonl", train_records)
    validation_manifest = _write_jsonl(tmp_path / "validation.jsonl", validation_records)
    split_assignments = _split_assignments(
        tmp_path / "assignments.jsonl",
        {
            **{str(record["sample_id"]): "train" for record in train_records},
            **{str(record["sample_id"]): "validation" for record in validation_records},
        },
    )
    distributed_experiment = _experiment(
        tmp_path / "distributed-experiment.yaml",
        tmp_path,
        name="omnihand-ddp-fixture",
    )
    distributed_task = _task(tmp_path / "distributed-task.yaml", batch_size=1)
    payload = {
        "project_root": str(project_root),
        "experiment": str(distributed_experiment),
        "task": str(distributed_task),
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "split_assignments": str(split_assignments),
    }
    mp.spawn(_distributed_worker, args=(2, _free_port(), payload), nprocs=2, join=True)

    run_parent = tmp_path / "artifacts" / "omnihand-ddp-fixture"
    runs = list(run_parent.iterdir())
    assert len(runs) == 1
    run = runs[0]
    run_payload = json.loads((run / "run.json").read_text(encoding="utf-8"))
    prediction_index = json.loads(
        (run / "predictions.index.json").read_text(encoding="utf-8")
    )
    performance = json.loads((run / "performance.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "completed"
    assert prediction_index["world_size"] == 2
    assert prediction_index["record_count"] == 3
    assert prediction_index["coverage"] == {
        "expected": 3,
        "extra": 0,
        "missing": 0,
        "observed": 3,
    }
    assert len(list(run.glob("checkpoint.safetensors"))) == 1
    assert len(list(run.glob("predictions.rank-*-of-00002.jsonl"))) == 2
    assert len(list(run.glob("predictions.rank-*-of-00002.json"))) == 2
    assert not list(run.glob("training-state.*"))
    rank_performance = performance["distributed"]["rank_performance"]
    assert [item["rank"] for item in rank_performance] == [0, 1]
    assert [item["optimizer_steps_this_run"] for item in rank_performance] == [1, 1]
    assert sorted(item["prediction_samples"] for item in rank_performance) == [1, 2]
    assert len({item["model_state_sha256"] for item in rank_performance}) == 1
    assert checkpoint["model_state_sha256"] == rank_performance[0]["model_state_sha256"]

    reference_experiment = _experiment(
        tmp_path / "reference-experiment.yaml",
        tmp_path,
        name="omnihand-single-reference",
    )
    reference_task = _task(tmp_path / "reference-task.yaml", batch_size=2)
    reference_result = train_omnihand(
        load_experiment_config(reference_experiment),
        load_omnihand_run_config(reference_task),
        source_experiment_config=reference_experiment,
        source_task_config=reference_task,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        split_assignments_path=split_assignments,
        project_root=project_root,
        command=("mmprism", "omnihand-train", "single-reference"),
        runtime_report=_runtime(project_root),
        created_at=datetime(2026, 8, 12, 1, 5, tzinfo=UTC),
    )
    distributed_state = load_file(run / "checkpoint.safetensors")
    reference_state = load_file(Path(str(reference_result["run_dir"])) / "checkpoint.safetensors")
    assert set(distributed_state) == set(reference_state)
    for name in distributed_state:
        torch.testing.assert_close(
            distributed_state[name], reference_state[name], rtol=1e-4, atol=1e-4
        )
