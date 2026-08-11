from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from mmprism.artifacts import (
    PrepareError,
    RunInput,
    build_prepare_report,
    validate_split_bindings,
)
from mmprism.cli import main
from mmprism.config import load_experiment_config


def _runtime(project_root: Path, *, dirty: bool = False) -> dict[str, object]:
    return {
        "project_root": str(project_root.resolve()),
        "git": {"commit": "a" * 40, "dirty": dirty},
        "python": "3.12",
    }


def _config(project_root: Path) -> Path:
    path = project_root / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mmprism.experiment.v1",
                "name": "prepare-fixture",
                "task": "pose_reconstruction",
                "paths": {
                    "data_root": "data",
                    "artifact_root": "artifacts",
                    "cache_root": "cache",
                },
                "runtime": {
                    "seed": 17,
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


def _manifest(path: Path, sample_ids: tuple[str, ...]) -> Path:
    records = (
        {
            "schema_version": "mmprism.sample.v1",
            "sample_id": sample_id,
            "sequence_id": f"sequence-{sample_id}",
            "dataset": "prepare-fixture",
            "modalities": {"caption": {"text": sample_id}},
        }
        for sample_id in sample_ids
    )
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _assignments(path: Path, assignments: dict[str, str]) -> Path:
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


def _inputs(
    train_manifest: Path,
    validation_manifest: Path,
    split_assignments: Path,
) -> tuple[RunInput, ...]:
    return (
        RunInput.capture(name="train_manifest", kind="manifest", path=train_manifest),
        RunInput.capture(
            name="validation_manifest", kind="manifest", path=validation_manifest
        ),
        RunInput.capture(name="split_assignments", kind="split", path=split_assignments),
    )


def test_prepare_is_side_effect_free_and_binds_every_manifest(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    config_path = _config(tmp_path)
    train_manifest = _manifest(tmp_path / "train.jsonl", ("train-001", "train-002"))
    validation_manifest = _manifest(tmp_path / "validation.jsonl", ("validation-001",))
    split_assignments = _assignments(
        tmp_path / "assignments.jsonl",
        {
            "train-001": "train",
            "train-002": "train",
            "validation-001": "validation",
            "unused-test-001": "test",
        },
    )

    report = build_prepare_report(
        load_experiment_config(config_path),
        source_config=config_path,
        inputs=_inputs(train_manifest, validation_manifest, split_assignments),
        split_bindings={
            "train_manifest": "train",
            "validation_manifest": "validation",
        },
        project_root=tmp_path,
        runtime_report=_runtime(tmp_path),
        created_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
    )

    assert report["schema_version"] == "mmprism.prepare_report.v1"
    assert report["status"] == "passed"
    assert report["side_effect_free"] is True
    validation = report["validation"]
    assert isinstance(validation, dict)
    assert validation["manifest_record_count"] == 3
    split = validation["split"]
    assert isinstance(split, dict)
    assert split["bound_sample_count"] == 3
    assert split["unbound_assignment_count"] == 1
    plan = report["plan"]
    assert isinstance(plan, dict)
    assert not Path(str(plan["run_dir"])).exists()
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "cache").exists()


def test_prepare_rejects_dirty_git_and_wrong_split(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    config_path = _config(tmp_path)
    train_manifest = _manifest(tmp_path / "train.jsonl", ("train-001",))
    validation_manifest = _manifest(tmp_path / "validation.jsonl", ("validation-001",))
    split_assignments = _assignments(
        tmp_path / "assignments.jsonl",
        {
            "train-001": "validation",
            "validation-001": "validation",
            "unused-train-001": "train",
        },
    )
    inputs = _inputs(train_manifest, validation_manifest, split_assignments)
    bindings = {"train_manifest": "train", "validation_manifest": "validation"}

    with pytest.raises(PrepareError, match="clean Git"):
        build_prepare_report(
            load_experiment_config(config_path),
            source_config=config_path,
            inputs=inputs,
            split_bindings=bindings,
            project_root=tmp_path,
            runtime_report=_runtime(tmp_path, dirty=True),
        )
    with pytest.raises(PrepareError, match="wrong_split=1"):
        build_prepare_report(
            load_experiment_config(config_path),
            source_config=config_path,
            inputs=inputs,
            split_bindings=bindings,
            project_root=tmp_path,
            runtime_report=_runtime(tmp_path),
        )


def test_split_bindings_reject_incomplete_coverage_and_manifest_overlap(
    tmp_path: Path,
) -> None:
    split_assignments = _assignments(
        tmp_path / "assignments.jsonl",
        {"shared-001": "train"},
    )
    manifest_samples = {
        "train_manifest": ("shared-001",),
        "validation_manifest": ("shared-001",),
    }

    with pytest.raises(PrepareError, match="cover every manifest exactly once"):
        validate_split_bindings(
            manifest_samples,
            split_assignments,
            {"train_manifest": "train"},
        )
    with pytest.raises(PrepareError, match="manifest inputs overlap"):
        validate_split_bindings(
            manifest_samples,
            split_assignments,
            {"train_manifest": "train", "validation_manifest": "train"},
        )


def test_artifact_preflight_import_does_not_load_training_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from mmprism.artifacts import build_prepare_report; "
                "assert build_prepare_report is not None; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'lightning' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_prepare_cli_uses_the_same_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "data").mkdir()
    config_path = _config(tmp_path)
    manifest = _manifest(tmp_path / "manifest.jsonl", ("sample-001",))
    assignments = _assignments(tmp_path / "assignments.jsonl", {"sample-001": "test"})
    monkeypatch.setattr(
        "mmprism.artifacts.prepare.collect_runtime_report",
        lambda project_root: _runtime(Path(project_root)),
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = main(
            [
                "prepare",
                str(config_path),
                "--project-root",
                str(tmp_path),
                "--input",
                f"manifest:evaluation_manifest={manifest}",
                "--input",
                f"split:split_assignments={assignments}",
                "--split-binding",
                "evaluation_manifest=test",
            ]
        )

    assert exit_code == 0
    payload = json.loads(output.getvalue())
    assert payload["status"] == "passed"
    assert payload["validation"]["split"]["bindings"]["evaluation_manifest"] == {
        "sample_count": 1,
        "split": "test",
    }
