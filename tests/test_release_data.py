"""Data-integrity and evaluation regressions using small local numeric fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmprism.data.compact_radar import PackedRadar
from mmprism.data.reconstruction_membership import deterministic_cycle_batches
from mmprism.release import paths as path_context
from mmprism.release.paths import Paths
from mmprism.release.scoring import evaluate
from mmprism.release.verify import verify_data


def test_compact_radar_matches_array_slices(tmp_path: Path) -> None:
    values = np.arange(48, dtype=np.float16).reshape(4, 3, 4)
    path = tmp_path / "radar.npy"
    np.save(path, values)
    reader = PackedRadar(path)
    np.testing.assert_array_equal(reader[1:3], values[1:3])
    np.testing.assert_array_equal(reader[-2:], values[-2:])
    np.testing.assert_array_equal(reader[4:4], values[4:4])


def test_truncated_radar_cannot_return_uninitialized_values(tmp_path: Path) -> None:
    path = tmp_path / "radar.npy"
    np.save(path, np.ones((4, 3, 4), dtype=np.float16))
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    reader = PackedRadar(path)
    np.testing.assert_array_equal(reader[:1], np.ones((1, 3, 4), dtype=np.float16))
    with pytest.raises(ValueError, match="truncated radar window"):
        reader[:]


def test_compact_radar_rejects_wrong_dtype(tmp_path: Path) -> None:
    path = tmp_path / "radar.npy"
    np.save(path, np.ones((4, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="float16"):
        PackedRadar(path)


@pytest.mark.parametrize("segment_count", [1, 3])
def test_sampler_fills_every_rank_with_a_small_roster(segment_count: int) -> None:
    batches = deterministic_cycle_batches(
        segment_count=segment_count,
        world_size=4,
        per_device_batch=2,
        base_seed=7,
        sampler_cycle=0,
    )
    assert len(batches) == 4
    assert all(len(rank_batches) == 1 and len(rank_batches[0]) == 2 for rank_batches in batches)
    first_global_batch = [index for rank_batches in batches for index in rank_batches[0]]
    assert set(first_global_batch[:segment_count]) == set(range(segment_count))
    assert all(0 <= index < segment_count for index in first_global_batch)


@pytest.fixture
def small_pack(tmp_path: Path) -> Paths:
    data = tmp_path / "data"
    config = tmp_path / "configs"
    config.mkdir()
    pool = data / "pool/p01"
    pool.mkdir(parents=True)
    segment = "p01:00001"
    radar = data / "radar/p01" / segment / "mmwave.fp16.npy"
    radar.parent.mkdir(parents=True)
    np.save(radar, np.zeros((1, 64, 116, 32, 2), dtype=np.float16))
    pose = np.zeros((1, 2, 24, 3), dtype=np.float32)
    pose[..., 2] = 1.0
    np.save(pool / "pose_camera_m.npy", pose)
    np.save(pool / "valid_mask.npy", np.ones(1, dtype=bool))
    (pool / "manifest.jsonl").write_text(
        json.dumps({"segment_id": segment, "frames": 1, "frame_offset": 0}) + "\n"
    )
    reader_base = data / "reader_base"
    reader_base.mkdir()
    for name in ("config.json", "tokenizer_config.json", "spiece.model"):
        (reader_base / name).write_text("{}\n")
    (data / "intrinsics.json").write_text(
        json.dumps({"K": np.eye(3).tolist(), "width": 16, "height": 16})
    )
    (data / "cells.json").write_text(
        json.dumps({"cells": {"D": {"groups": [{"person": "p01", "segment_ids": [segment]}]}}})
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "signers": ["p01"],
                "per_person": {"p01": {"evaluation_segments": 1, "calibration_segments": 0}},
                "radar_files": [
                    {
                        "segment_id": segment,
                        "frames": 1,
                        "bytes": radar.stat().st_size,
                        "path": radar.relative_to(data).as_posix(),
                    }
                ],
                "sampled_sha256": {},
            }
        )
    )
    (config / "paths.toml").write_text(
        f'data_root = "{data}"\noutput_root = "{tmp_path / "outputs"}"\n'
    )
    (config / "data_layout.toml").write_text(
        '[pack]\nmanifest = "manifest.json"\npool = "pool/{signer}"\n'
        'radar = "radar/{signer}/{segment}/mmwave.fp16.npy"\nreader_base = "reader_base"\n'
        'splits = "cells.json"\nintrinsics = "intrinsics.json"\n'
        '[outputs]\nreconstruction = "reconstruction/{mode}/{signer}/{segment}.npz"\n'
        'pose_signer_report = "evaluation/pose/{mode}/{signer}.json"\n'
    )
    return Paths(config_dir=config, environ={})


def test_verify_data_accepts_complete_pack(small_pack: Paths) -> None:
    assert all(row.get("matches", row["exists"]) for row in verify_data(small_pack))


@pytest.mark.parametrize(
    "resource,member", [("pool", "valid_mask.npy"), ("reader_base", "spiece.model")]
)
def test_verify_data_rejects_missing_required_member(
    small_pack: Paths, resource: str, member: str
) -> None:
    missing = small_pack.resolve(f"pack.{resource}", signer="p01") / member
    missing.unlink()
    results = verify_data(small_pack)
    assert any(row["path"] == str(missing) and not row["exists"] for row in results)


def test_verify_data_reports_corrupt_header(small_pack: Paths) -> None:
    radar = small_pack.resolve("pack.radar", signer="p01", segment="p01:00001")
    radar.write_bytes(b"incomplete array")
    result = next(row for row in verify_data(small_pack) if row["key"] == "pack.radar")
    assert not result["matches"]
    assert any("cannot read radar header" in error for error in result["failures"])


def test_repeated_signer_does_not_double_evaluation_support(
    small_pack: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(path_context, "_default", small_pack)
    prediction = np.load(small_pack.resolve("pack.pool", signer="p01") / "pose_camera_m.npy")
    path = small_pack.resolve(
        "outputs.reconstruction", mode="zero-shot", signer="p01", segment="p01:00001"
    )
    path.parent.mkdir(parents=True)
    np.savez(path, pose=prediction)
    result = evaluate(["p01", "p01"], mode="zero-shot")
    assert result["segments"] == 1
    assert result["pooled"]["joints"] == 48
    assert result["per_person"]["p01"]["segments"] == 1
