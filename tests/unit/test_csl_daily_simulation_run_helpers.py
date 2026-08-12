from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmprism.data.csl_daily_simulation_run import (
    ArrayArtifact,
    CslDailySimulationConfig,
    CslDailySimulationError,
    PoseInputEntry,
    build_manifest_row,
    frame_validity,
    load_csl_daily_simulation_config,
    load_pose_manifest,
    select_target_pose,
)


def _config(**overrides: object) -> CslDailySimulationConfig:
    payload: dict[str, object] = {
        "schema_version": "mmprism.csl_daily_simulation.v1",
        "radar": {"radar_config_id": "iwr1843_sim_v1"},
        "input": {"pose_manifest_path": "/in/poses.jsonl", "pose_root": "/in"},
        "output": {"output_root": "/out"},
        "dataset": {"name": "csl-daily", "coordinate_frame": "fixture_camera_v1"},
    }
    payload.update(overrides)
    return CslDailySimulationConfig.from_mapping(payload)  # type: ignore[arg-type]


def _entry() -> PoseInputEntry:
    return PoseInputEntry(
        sample_id="seq-001",
        pose_uri="poses/seq-001.npy",
        pose_sha256="a" * 64,
        sequence_id=None,
        subject_id=None,
    )


def test_frame_validity_marks_any_nan_coordinate_invalid() -> None:
    pose = np.zeros((3, 2, 24, 3), dtype=np.float32)
    pose[1, 0, 5, 2] = np.nan
    pose[2] = np.inf
    validity = frame_validity(pose)
    assert validity.dtype == np.bool_
    assert validity.tolist() == [True, False, False]


def test_select_target_pose_picks_central_decimated_frame() -> None:
    valid_pose = np.arange(10 * 2 * 24 * 3, dtype=np.float32).reshape(10, 2, 24, 3)
    # 3 output frames -> central index 1 -> valid-frame index 1 * decimation.
    target = select_target_pose(valid_pose, output_frames=3, decimation=3)
    np.testing.assert_array_equal(target, valid_pose[3])
    # Index is clamped to the available valid frames.
    clamped = select_target_pose(valid_pose, output_frames=9, decimation=3)
    np.testing.assert_array_equal(clamped, valid_pose[-1])


def test_build_manifest_row_matches_pose_reconstruction_contract() -> None:
    artifacts = {
        "radar_cube": ArrayArtifact(
            "radar_cube/seq-001.npy", (3, 64, 32, 32, 32), "float32", "b" * 64
        ),
        "pose_gt": ArrayArtifact("pose_gt/seq-001.npy", (2, 24, 3), "float32", "c" * 64),
        "frame_mask": ArrayArtifact("frame_mask/seq-001.npy", (3,), "bool", "d" * 64),
        "pose_valid": ArrayArtifact("pose_valid/seq-001.npy", (2, 24), "bool", "e" * 64),
    }
    row = build_manifest_row(
        entry=_entry(),
        config=_config(),
        artifacts=artifacts,
        pose_manifest_sha256="f" * 64,
    )
    assert row["schema_version"] == "mmprism.sample.v1"
    assert row["sequence_id"] == "seq-001"  # falls back to sample_id
    assert "subject_id" not in row
    assert row["modalities"]["radar_cube"]["shape"] == [3, 64, 32, 32, 32]
    acquisition = row["acquisition"]
    assert acquisition["sample_protocol"] == "mmprism.pose_reconstruction.sample_v1"
    assert acquisition["radar_cube_protocol"] == "mmprism.radar_cube.power_v1"
    assert acquisition["pose_units"] == "m"
    assert acquisition["radar_config_id"] == "iwr1843_sim_v1"
    provenance = row["provenance"]
    assert provenance["source_pose_manifest_sha256"] == "f" * 64
    assert provenance["source_pose_sha256"] == "a" * 64


def test_config_rejects_unknown_radar_and_non_cpu() -> None:
    with pytest.raises(CslDailySimulationError, match="unknown radar_config_id"):
        _config(radar={"radar_config_id": "nope"})
    with pytest.raises(CslDailySimulationError, match="device"):
        _config(runtime={"device": "cuda:0"})
    with pytest.raises(CslDailySimulationError, match="schema_version"):
        _config(schema_version="wrong")


def test_config_fingerprint_is_stable() -> None:
    assert _config().fingerprint() == _config().fingerprint()
    changed = _config(preprocessing={"decimation": 2})
    assert changed.fingerprint() != _config().fingerprint()


def test_load_pose_manifest_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "poses.jsonl"
    manifest.write_text(
        '{"sample_id": "seq-001", "pose_uri": "a.npy", "pose_sha256": "%s"}\n' % ("a" * 64),
        encoding="utf-8",
    )
    entries = load_pose_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].sample_id == "seq-001"

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"sample_id": "x", "pose_uri": "../a.npy", "pose_sha256": "%s"}\n' % ("a" * 64),
        encoding="utf-8",
    )
    with pytest.raises(CslDailySimulationError, match="relative path"):
        load_pose_manifest(bad)

    duplicate = tmp_path / "duplicate.jsonl"
    line = '{"sample_id": "x", "pose_uri": "a.npy", "pose_sha256": "%s"}\n' % ("a" * 64)
    duplicate.write_text(line + line, encoding="utf-8")
    with pytest.raises(CslDailySimulationError, match="duplicate sample_id"):
        load_pose_manifest(duplicate)


def test_versioned_config_loads_with_injected_variables() -> None:
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "data" / "csl_daily_simulation.yaml"
    )
    config = load_csl_daily_simulation_config(
        config_path, variables={"MMPRISM_DATA_ROOT": "/data"}
    )
    assert config.radar_config_id == "iwr1843_sim_v1"
    assert str(config.pose_manifest_path).startswith("/data/")
    assert str(config.output_root).startswith("/data/")
    assert config.device == "cpu"
    assert config.decimation == 3
    assert config.min_valid_frames == 2

    with pytest.raises(CslDailySimulationError, match="no supplied value"):
        load_csl_daily_simulation_config(config_path, variables={})
