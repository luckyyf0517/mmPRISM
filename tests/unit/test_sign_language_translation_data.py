from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mmprism.data import (
    SignLanguageTranslationDataError,
    SignLanguageTranslationManifest,
    collate_sign_language_translation_samples,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    root: Path,
    *,
    sample_id: str,
    frames: int,
    feature_dim: int = 8,
    caption: str = "target",
    include_mask: bool = True,
) -> dict[str, object]:
    arrays = root / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(frames)
    values: dict[str, np.ndarray] = {
        "pose": generator.normal(size=(frames, 2, 24, 3)).astype(np.float32),
        "pose_confidence": generator.random((frames, 2, 24), dtype=np.float32),
        "radar_feature": generator.normal(size=(frames, feature_dim)).astype(np.float32),
    }
    if include_mask:
        mask = np.ones(frames, dtype=np.bool_)
        if frames > 1:
            mask[-1] = False
        values["frame_mask"] = mask
    modalities: dict[str, object] = {"caption": {"text": caption}}
    for name, value in values.items():
        path = arrays / f"{sample_id}.{name}.npy"
        np.save(path, value, allow_pickle=False)
        modalities[name] = {
            "uri": path.relative_to(root).as_posix(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256(path),
        }
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": f"sequence-{sample_id}",
        "dataset": "translation-fixture",
        "modalities": modalities,
        "acquisition": {
            "sample_protocol": "mmprism.sign_language_translation.sample_v1",
            "radar_feature_protocol": "mmprism.radar_feature.sequence_v1",
            "pose_units": "m",
            "pose_coordinate_frame": "fixture_radar_cartesian_v1",
        },
    }


def _manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_translation_manifest_loads_and_zero_pads_variable_sequences(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path / "manifest.jsonl",
        [
            _record(tmp_path, sample_id="sample-001", frames=3),
            _record(
                tmp_path,
                sample_id="sample-002",
                frames=2,
                caption="second target",
                include_mask=False,
            ),
        ],
    )
    manifest = SignLanguageTranslationManifest(manifest_path, data_root=tmp_path)

    assert len(manifest) == 2
    assert manifest.radar_feature_dim == 8
    assert manifest.joint_count == 24
    assert manifest.coordinate_dim == 3
    first = manifest.load_sample(0)
    second = manifest.load_sample(1)
    batch = collate_sign_language_translation_samples((first, second), max_frames=4)

    assert batch.pose.shape == (2, 3, 2, 24, 3)
    assert batch.pose_confidence.shape == (2, 3, 2, 24)
    assert batch.radar_feature.shape == (2, 3, 8)
    assert batch.frame_mask.tolist() == [[True, True, False], [True, True, False]]
    assert np.count_nonzero(batch.pose[0, 2]) == 0
    assert np.count_nonzero(batch.radar_feature[1, 2]) == 0
    assert batch.captions == ("target", "second target")


def test_translation_manifest_rejects_tamper_confidence_and_contract_drift(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, sample_id="sample-001", frames=2)
    manifest_path = _manifest(tmp_path / "manifest.jsonl", [record])
    pose_path = tmp_path / "arrays" / "sample-001.pose.npy"
    pose_path.write_bytes(pose_path.read_bytes() + b"x")
    with pytest.raises(SignLanguageTranslationDataError, match="SHA-256 mismatch"):
        SignLanguageTranslationManifest(manifest_path, data_root=tmp_path)

    record = _record(tmp_path, sample_id="sample-002", frames=2)
    confidence_path = tmp_path / "arrays" / "sample-002.pose_confidence.npy"
    confidence = np.load(confidence_path, allow_pickle=False)
    confidence[0, 0, 0] = 1.5
    np.save(confidence_path, confidence, allow_pickle=False)
    modalities = record["modalities"]
    assert isinstance(modalities, dict)
    confidence_ref = modalities["pose_confidence"]
    assert isinstance(confidence_ref, dict)
    confidence_ref["sha256"] = _sha256(confidence_path)
    manifest = SignLanguageTranslationManifest(
        _manifest(tmp_path / "confidence.jsonl", [record]), data_root=tmp_path
    )
    with pytest.raises(SignLanguageTranslationDataError, match=r"within \[0,1\]"):
        manifest.load_sample(0)

    record = _record(tmp_path, sample_id="sample-003", frames=2)
    acquisition = record["acquisition"]
    assert isinstance(acquisition, dict)
    acquisition["radar_feature_protocol"] = "unknown"
    with pytest.raises(SignLanguageTranslationDataError, match="radar feature protocol"):
        SignLanguageTranslationManifest(
            _manifest(tmp_path / "protocol.jsonl", [record]), data_root=tmp_path
        )
