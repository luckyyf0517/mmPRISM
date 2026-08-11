from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mmprism.data import (
    PoseReconstructionDataError,
    PoseReconstructionManifest,
    collate_pose_reconstruction_samples,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    root: Path,
    *,
    sample_id: str,
    frames: int,
    negative_cube: bool = False,
    pose_units: str = "m",
) -> dict[str, object]:
    arrays = root / "arrays"
    arrays.mkdir(exist_ok=True)
    generator = np.random.default_rng(frames)
    radar = generator.random((frames, 4, 8, 6, 4), dtype=np.float32)
    if negative_cube:
        radar[0, 0, 0, 0, 0] = -1
    pose = generator.normal(size=(2, 24, 3)).astype(np.float32)
    frame_mask = np.ones(frames, dtype=np.bool_)
    frame_mask[-1] = False
    pose_valid = np.ones((2, 24), dtype=np.bool_)
    paths = {
        "radar_cube": arrays / f"{sample_id}.radar.npy",
        "pose_gt": arrays / f"{sample_id}.pose.npy",
        "frame_mask": arrays / f"{sample_id}.frames.npy",
        "pose_valid": arrays / f"{sample_id}.valid.npy",
    }
    np.save(paths["radar_cube"], radar, allow_pickle=False)
    np.save(paths["pose_gt"], pose, allow_pickle=False)
    np.save(paths["frame_mask"], frame_mask, allow_pickle=False)
    np.save(paths["pose_valid"], pose_valid, allow_pickle=False)
    shapes = {
        "radar_cube": list(radar.shape),
        "pose_gt": [2, 24, 3],
        "frame_mask": [frames],
        "pose_valid": [2, 24],
    }
    dtypes = {
        "radar_cube": "float32",
        "pose_gt": "float32",
        "frame_mask": "bool",
        "pose_valid": "bool",
    }
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": f"sequence-{sample_id}",
        "subject_id": "subject-fixture",
        "dataset": "pose-fixture",
        "modalities": {
            name: {
                "uri": path.relative_to(root).as_posix(),
                "shape": shapes[name],
                "dtype": dtypes[name],
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "acquisition": {
            "sample_protocol": "mmprism.pose_reconstruction.sample_v1",
            "radar_cube_protocol": "mmprism.radar_cube.power_v1",
            "pose_units": pose_units,
            "pose_coordinate_frame": "radar_cartesian_v1",
        },
        "provenance": {"purpose": "unit-test"},
    }


def _manifest(root: Path, records: list[dict[str, object]]) -> Path:
    path = root / "manifest.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_pose_manifest_loads_checksums_and_collates_variable_time(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        [
            _record(tmp_path, sample_id="sample-001", frames=3),
            _record(tmp_path, sample_id="sample-002", frames=2),
        ],
    )

    manifest = PoseReconstructionManifest(manifest_path, data_root=tmp_path)
    first = manifest.load_sample(0)
    second = manifest.load_sample(1)
    batch = collate_pose_reconstruction_samples((first, second), max_frames=4)

    assert len(manifest) == 2
    assert manifest.radar_spatial_shape == (4, 8, 6, 4)
    assert manifest.coordinate_frame == "radar_cartesian_v1"
    assert batch.sample_ids == ("sample-001", "sample-002")
    assert batch.radar_cube.shape == (2, 3, 4, 8, 6, 4)
    assert batch.frame_mask.tolist() == [[True, True, False], [True, False, False]]
    assert np.count_nonzero(batch.radar_cube[~batch.frame_mask]) == 0
    assert batch.pose_target.shape == (2, 2, 24, 3)
    assert batch.pose_valid.shape == (2, 2, 24)


def test_pose_manifest_rejects_checksum_mismatch(tmp_path: Path) -> None:
    record = _record(tmp_path, sample_id="sample-001", frames=3)
    modalities = record["modalities"]
    assert isinstance(modalities, dict)
    radar = modalities["radar_cube"]
    assert isinstance(radar, dict)
    radar["sha256"] = "0" * 64

    with pytest.raises(PoseReconstructionDataError, match="SHA-256 mismatch"):
        PoseReconstructionManifest(_manifest(tmp_path, [record]), data_root=tmp_path)


def test_pose_manifest_rejects_physical_contract_violations(tmp_path: Path) -> None:
    negative = _record(tmp_path, sample_id="sample-negative", frames=3, negative_cube=True)
    negative_manifest = PoseReconstructionManifest(
        _manifest(tmp_path, [negative]), data_root=tmp_path
    )
    with pytest.raises(PoseReconstructionDataError, match="finite non-negative"):
        negative_manifest.load_sample(0)

    wrong_units = _record(tmp_path, sample_id="sample-units", frames=3, pose_units="mm")
    with pytest.raises(PoseReconstructionDataError, match="units must be metres"):
        PoseReconstructionManifest(_manifest(tmp_path, [wrong_units]), data_root=tmp_path)


def test_pose_collate_rejects_sequences_longer_than_model_limit(tmp_path: Path) -> None:
    record = _record(tmp_path, sample_id="sample-001", frames=3)
    manifest = PoseReconstructionManifest(_manifest(tmp_path, [record]), data_root=tmp_path)

    with pytest.raises(PoseReconstructionDataError, match="model maximum"):
        collate_pose_reconstruction_samples((manifest.load_sample(0),), max_frames=2)
