from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")

from mmprism.contracts import validate_dual_hand_pose, validate_radar_cube  # noqa: E402
from mmprism.data.csl_daily_simulation_run import (  # noqa: E402
    RUN_RECORD_SCHEMA,
    SIMULATION_PROTOCOL,
    load_csl_daily_simulation_config,
    run_csl_daily_simulation,
)
from mmprism.data.pose_reconstruction import PoseReconstructionManifest  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pose(
    pose_root: Path,
    *,
    sample_id: str,
    seed: int,
    frames: int = 10,
    nan_frames: tuple[int, ...] = (),
    keep_frames: int | None = None,
) -> dict[str, str]:
    generator = np.random.default_rng(seed)
    pose = (
        generator.normal(loc=[0.0, 0.0, 1.5], scale=0.15, size=(frames, 2, 24, 3))
    ).astype(np.float32)
    for index in nan_frames:
        pose[index] = np.nan
    if keep_frames is not None:
        pose[keep_frames:] = np.nan
    path = pose_root / f"{sample_id}.npy"
    np.save(path, pose, allow_pickle=False)
    return {
        "sample_id": sample_id,
        "pose_uri": f"{sample_id}.npy",
        "pose_sha256": _sha256(path),
    }


def _write_config(path: Path, tmp_path: Path) -> Path:
    payload = {
        "schema_version": "mmprism.csl_daily_simulation.v1",
        "radar": {"radar_config_id": "iwr1843_sim_v1"},
        "input": {
            "pose_manifest_path": "${POSE_MANIFEST}",
            "pose_root": "${POSE_ROOT}",
        },
        "output": {
            "output_root": "${OUTPUT_ROOT}",
            "manifest_name": "manifest.jsonl",
            "run_record_name": "run_record.json",
        },
        "dataset": {
            "name": "csl-daily-fixture",
            "coordinate_frame": "fixture_rtmw3d_camera_v1",
        },
        "preprocessing": {"min_valid_frames": 2, "min_output_frames": 1},
        "runtime": {"device": "cpu", "precision": "float32", "frames_per_batch": 2},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_csl_daily_simulation_run_end_to_end(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    pose_root.mkdir()
    entries = [
        # Clean sequence: 10 frames -> 3 decimated frames.
        _write_pose(pose_root, sample_id="seq-good", seed=1),
        # Partially NaN sequence: invalid frames are dropped, still materializes.
        _write_pose(pose_root, sample_id="seq-partial", seed=2, nan_frames=(4, 5)),
        # Only one valid frame: recorded failure, never a crash.
        _write_pose(pose_root, sample_id="seq-bad", seed=3, keep_frames=1),
    ]
    pose_manifest = tmp_path / "pose_manifest.jsonl"
    pose_manifest.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path / "simulation.yaml", tmp_path)
    output_root = tmp_path / "output"
    config = load_csl_daily_simulation_config(
        config_path,
        variables={
            "POSE_MANIFEST": str(pose_manifest),
            "POSE_ROOT": str(pose_root),
            "OUTPUT_ROOT": str(output_root),
        },
    )

    result = run_csl_daily_simulation(config)

    assert result.emitted_count == 2
    assert result.failed_count == 1
    outcomes = {outcome.sample_id: outcome for outcome in result.outcomes}
    assert outcomes["seq-good"].status == "emitted"
    assert outcomes["seq-good"].output_frames == 3
    assert outcomes["seq-partial"].status == "emitted"
    assert outcomes["seq-partial"].valid_frames == 8
    assert outcomes["seq-bad"].status == "failed"
    assert "valid frames" in (outcomes["seq-bad"].reason or "")

    # The emitted manifest loads through the real loader, checksums included.
    manifest = PoseReconstructionManifest(result.manifest_path, data_root=output_root)
    assert len(manifest) == 2
    assert manifest.radar_spatial_shape == (64, 32, 32, 32)
    assert manifest.coordinate_frame == "fixture_rtmw3d_camera_v1"
    sample = manifest.load_sample(0)
    assert sample.radar_cube.shape == (3, 64, 32, 32, 32)
    assert sample.pose_target.shape == (2, 24, 3)
    assert bool(sample.frame_mask.all())
    assert bool(sample.pose_valid.all())
    validate_radar_cube(sample.radar_cube, leading_axes=("time",))
    validate_dual_hand_pose(
        sample.pose_target, coordinate_frame=manifest.coordinate_frame
    )

    # Provenance: radar_config_id and source pose-manifest hash on every row.
    rows = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["seq-good", "seq-partial"]
    for row in rows:
        assert row["acquisition"]["radar_config_id"] == "iwr1843_sim_v1"
        assert row["acquisition"]["sample_protocol"] == (
            "mmprism.pose_reconstruction.sample_v1"
        )
        provenance = row["provenance"]
        assert provenance["radar_config_id"] == "iwr1843_sim_v1"
        assert provenance["source_pose_manifest_sha256"] == _sha256(pose_manifest)
        assert provenance["simulation_protocol"] == SIMULATION_PROTOCOL

    # Run record: config hash, counts, per-sample status, failure recorded.
    run_record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert run_record["schema_version"] == RUN_RECORD_SCHEMA
    assert run_record["config_sha256"] == result.config_fingerprint
    assert run_record["radar_config_id"] == "iwr1843_sim_v1"
    assert run_record["pose_manifest"]["sha256"] == _sha256(pose_manifest)
    assert run_record["counts"]["entries"] == 3
    assert run_record["counts"]["emitted"] == 2
    assert run_record["counts"]["failed"] == 1
    statuses = {sample["sample_id"]: sample["status"] for sample in run_record["samples"]}
    assert statuses == {
        "seq-good": "emitted",
        "seq-partial": "emitted",
        "seq-bad": "failed",
    }
    assert run_record["outputs"]["manifest_sha256"] == _sha256(result.manifest_path)

    # No-clobber: a second run against the same output root refuses.
    with pytest.raises(Exception, match="no-clobber"):
        run_csl_daily_simulation(config)
