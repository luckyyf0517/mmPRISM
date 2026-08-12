from __future__ import annotations

import builtins
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from mmprism.cli import main
from mmprism.data import (
    MAX_PART_ROWS,
    MAX_PARTS_PER_CHUNK,
    POSE_RECONSTRUCTION_PRODUCT,
    SIGN_LANGUAGE_TRANSLATION_PRODUCT,
    ParquetDeliveryConfig,
    ParquetDeliveryError,
    ParquetPoseReconstructionDataset,
    ParquetSignLanguageTranslationDataset,
    PoseReconstructionManifest,
    SignLanguageTranslationManifest,
    collate_pose_reconstruction_samples,
    collate_sign_language_translation_samples,
    load_parquet_delivery_config,
    materialize_parquet_delivery,
    parquet_delivery,
    plan_parquet_layout,
    validate_parquet_delivery,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict[str, object]:
    return {"git": {"commit": "a" * 40, "dirty": False}, "python": "3.12"}


def _write_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _write_assignments(path: Path, assignments: dict[str, str]) -> Path:
    rows = []
    for index, sample_id in enumerate(sorted(assignments)):
        rows.append(
            {
                "schema_version": "mmprism.split_assignment.v1",
                "sample_id": sample_id,
                "group_id": hashlib.sha256(f"group-{index}".encode()).hexdigest(),
                "split": assignments[sample_id],
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _pose_record(root: Path, sample_id: str, frames: int) -> dict[str, object]:
    arrays = root / "arrays"
    arrays.mkdir(exist_ok=True)
    generator = np.random.default_rng(frames)
    values = {
        "radar_cube": generator.random((frames, 2, 3, 2, 2), dtype=np.float32),
        "pose_gt": generator.normal(size=(2, 24, 3)).astype(np.float32),
        "frame_mask": np.ones(frames, dtype=np.bool_),
        "pose_valid": np.ones((2, 24), dtype=np.bool_),
    }
    values["frame_mask"][-1] = False
    modalities: dict[str, object] = {}
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
        "subject_id": "subject-fixture",
        "dataset": "pose-fixture",
        "modalities": modalities,
        "acquisition": {
            "sample_protocol": "mmprism.pose_reconstruction.sample_v1",
            "radar_cube_protocol": "mmprism.radar_cube.power_v1",
            "pose_units": "m",
            "pose_coordinate_frame": "fixture_radar_cartesian_v1",
            "fps": 30,
        },
        "provenance": {
            "source_archive_id": "archive_001",
            "source_archive_sha256": "b" * 64,
            "source_member": f"{sample_id}.mp4",
            "source_member_crc32": 7,
        },
    }


def _translation_record(root: Path, sample_id: str, frames: int) -> dict[str, object]:
    arrays = root / "arrays"
    arrays.mkdir(exist_ok=True)
    generator = np.random.default_rng(frames)
    values = {
        "pose": generator.normal(size=(frames, 2, 24, 3)).astype(np.float32),
        "pose_confidence": generator.random((frames, 2, 24), dtype=np.float32),
        "radar_feature": generator.normal(size=(frames, 5)).astype(np.float32),
        "frame_mask": np.ones(frames, dtype=np.bool_),
    }
    values["frame_mask"][-1] = False
    modalities: dict[str, object] = {"caption": {"text": f"caption {sample_id}"}}
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
        "subject_id": "subject-fixture",
        "dataset": "translation-fixture",
        "modalities": modalities,
        "acquisition": {
            "sample_protocol": "mmprism.sign_language_translation.sample_v2",
            "input_mode": "pose_plus_radar_feature",
            "radar_feature_protocol": "mmprism.radar_feature.sequence_v1",
            "pose_units": "m",
            "pose_coordinate_frame": "fixture_radar_cartesian_v1",
        },
    }


def _pose_only_translation_record(
    root: Path, sample_id: str, frames: int
) -> dict[str, object]:
    arrays = root / "arrays"
    arrays.mkdir(exist_ok=True)
    generator = np.random.default_rng(frames)
    values = {
        "pose": generator.normal(size=(frames, 2, 24, 3)).astype(np.float32),
        "pose_confidence": generator.random((frames, 2, 24), dtype=np.float32),
        "frame_mask": np.ones(frames, dtype=np.bool_),
    }
    values["frame_mask"][-1] = False
    modalities: dict[str, object] = {"caption": {"text": f"caption {sample_id}"}}
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
        "subject_id": "subject-fixture",
        "dataset": "translation-fixture",
        "modalities": modalities,
        "acquisition": {
            "sample_protocol": "mmprism.sign_language_translation.sample_v2",
            "input_mode": "pose_only",
            "pose_units": "m",
            "pose_coordinate_frame": "fixture_radar_cartesian_v1",
        },
    }


def _config(
    root: Path,
    product: str,
    manifest: Path,
    assignments: Path,
    *,
    part_rows: int = 2,
    parts_per_chunk: int = 1,
) -> ParquetDeliveryConfig:
    return ParquetDeliveryConfig(
        product=product,
        data_root=root,
        source_manifest_path=manifest,
        split_assignment_path=assignments,
        processed_root=root / "processed",
        expected_source_manifest_sha256=_sha256(manifest),
        expected_split_assignment_sha256=_sha256(assignments),
        source_scope="partial",
        part_rows=part_rows,
        parts_per_chunk=parts_per_chunk,
    )


def _write_delivery_config(
    root: Path,
    product: str,
    manifest: Path,
    assignments: Path,
) -> Path:
    path = root / "delivery.yaml"
    path.write_text(
        f"""schema_version: mmprism.parquet_delivery_config.v1
source:
  data_root: {root}
  manifest: {manifest.relative_to(root).as_posix()}
  expected_manifest_sha256: {_sha256(manifest)}
  split_assignments: {assignments.relative_to(root).as_posix()}
  expected_split_assignment_sha256: {_sha256(assignments)}
  scope: partial
delivery:
  product: {product}
  processed_root: processed
  part_rows: 1024
  parts_per_chunk: 64
validation:
  minimum_free_bytes: 0
  verify_source_checksums: true
""",
        encoding="utf-8",
    )
    return path


def test_pose_delivery_round_trip_matches_source_adapter(tmp_path: Path) -> None:
    records = [
        _pose_record(tmp_path, "sample-003", 3),
        _pose_record(tmp_path, "sample-001", 2),
        _pose_record(tmp_path, "sample-002", 4),
    ]
    manifest_path = _write_manifest(tmp_path / "source.jsonl", records)
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl",
        {"sample-001": "train", "sample-002": "train", "sample-003": "validation"},
    )
    config = _config(
        tmp_path,
        POSE_RECONSTRUCTION_PRODUCT,
        manifest_path,
        assignments_path,
    )

    result = materialize_parquet_delivery(config, runtime_report=_runtime())
    delivery = ParquetPoseReconstructionDataset(result.root, split="train")
    source = PoseReconstructionManifest(manifest_path, data_root=tmp_path)
    source_by_id = {record.sample_id: index for index, record in enumerate(source.records)}
    parquet_samples = [delivery.load_sample(index) for index in range(len(delivery))]
    source_samples = [
        source.load_sample(source_by_id[sample.sample_id]) for sample in parquet_samples
    ]

    assert delivery.sample_ids == ("sample-001", "sample-002")
    assert result.sample_count == 3
    assert result.part_count == 2
    assert (result.root / "splits/train/chunk-00000/part-00000.parquet").is_file()
    assert (result.root / "splits/validation/chunk-00000/part-00000.parquet").is_file()
    delivery_metadata = json.loads((result.root / "delivery.json").read_text(encoding="utf-8"))
    build = delivery_metadata["build"]
    assert build["git"] == {"commit": "a" * 40, "dirty": False}
    assert "project_root" not in build["runtime_environment"]
    assert build["resolved_delivery_config"] == config.portable_dict()
    assert build["randomness"] == "none_deterministic_placement_v1"
    for parquet_sample, source_sample in zip(parquet_samples, source_samples, strict=True):
        assert np.array_equal(parquet_sample.radar_cube, source_sample.radar_cube)
        assert np.array_equal(parquet_sample.frame_mask, source_sample.frame_mask)
        assert np.array_equal(parquet_sample.pose_target, source_sample.pose_target)
        assert np.array_equal(parquet_sample.pose_valid, source_sample.pose_valid)

    parquet_batch = collate_pose_reconstruction_samples(parquet_samples, max_frames=4)
    source_batch = collate_pose_reconstruction_samples(source_samples, max_frames=4)
    assert np.array_equal(parquet_batch.radar_cube, source_batch.radar_cube)
    assert np.array_equal(parquet_batch.frame_mask, source_batch.frame_mask)
    assert np.array_equal(parquet_batch.pose_target, source_batch.pose_target)


def test_translation_delivery_round_trip_matches_source_adapter(tmp_path: Path) -> None:
    records = [
        _translation_record(tmp_path, "sample-003", 3),
        _translation_record(tmp_path, "sample-001", 2),
        _translation_record(tmp_path, "sample-002", 4),
    ]
    manifest_path = _write_manifest(tmp_path / "source.jsonl", records)
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl",
        {"sample-001": "train", "sample-002": "train", "sample-003": "validation"},
    )
    result = materialize_parquet_delivery(
        _config(
            tmp_path,
            SIGN_LANGUAGE_TRANSLATION_PRODUCT,
            manifest_path,
            assignments_path,
        ),
        runtime_report=_runtime(),
    )
    delivery = ParquetSignLanguageTranslationDataset(result.root, split="train")
    source = SignLanguageTranslationManifest(manifest_path, data_root=tmp_path)
    source_by_id = {record.sample_id: index for index, record in enumerate(source.records)}
    parquet_samples = [delivery.load_sample(index) for index in range(len(delivery))]
    source_samples = [
        source.load_sample(source_by_id[sample.sample_id]) for sample in parquet_samples
    ]

    assert delivery.sample_ids == ("sample-001", "sample-002")
    for parquet_sample, source_sample in zip(parquet_samples, source_samples, strict=True):
        assert np.array_equal(parquet_sample.pose, source_sample.pose)
        assert np.array_equal(parquet_sample.pose_confidence, source_sample.pose_confidence)
        assert parquet_sample.radar_feature is not None
        assert source_sample.radar_feature is not None
        assert np.array_equal(parquet_sample.radar_feature, source_sample.radar_feature)
        assert np.array_equal(parquet_sample.frame_mask, source_sample.frame_mask)
        assert parquet_sample.caption == source_sample.caption

    parquet_batch = collate_sign_language_translation_samples(parquet_samples, max_frames=4)
    source_batch = collate_sign_language_translation_samples(source_samples, max_frames=4)
    assert np.array_equal(parquet_batch.pose, source_batch.pose)
    assert np.array_equal(parquet_batch.pose_confidence, source_batch.pose_confidence)
    assert parquet_batch.radar_feature is not None
    assert source_batch.radar_feature is not None
    assert np.array_equal(parquet_batch.radar_feature, source_batch.radar_feature)
    assert parquet_batch.captions == source_batch.captions


def test_pose_only_translation_delivery_omits_radar_feature_end_to_end(tmp_path: Path) -> None:
    records = [
        _pose_only_translation_record(tmp_path, "sample-003", 3),
        _pose_only_translation_record(tmp_path, "sample-001", 2),
        _pose_only_translation_record(tmp_path, "sample-002", 4),
    ]
    manifest_path = _write_manifest(tmp_path / "source.jsonl", records)
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl",
        {"sample-001": "train", "sample-002": "train", "sample-003": "validation"},
    )
    result = materialize_parquet_delivery(
        _config(
            tmp_path,
            SIGN_LANGUAGE_TRANSLATION_PRODUCT,
            manifest_path,
            assignments_path,
        ),
        runtime_report=_runtime(),
    )
    delivery_metadata = json.loads((result.root / "delivery.json").read_text(encoding="utf-8"))
    schema_metadata = json.loads((result.root / "schema.json").read_text(encoding="utf-8"))
    assert delivery_metadata["input_mode"] == "pose_only"
    assert "radar_feature_dim" not in delivery_metadata["static_dimensions"]
    assert schema_metadata["input_mode"] == "pose_only"
    assert "radar_feature" not in schema_metadata["arrow_schema"]
    assert not list((tmp_path / "arrays").glob("*.radar_feature.npy"))

    delivery = ParquetSignLanguageTranslationDataset(result.root, split="train")
    source = SignLanguageTranslationManifest(manifest_path, data_root=tmp_path)
    source_by_id = {record.sample_id: index for index, record in enumerate(source.records)}
    parquet_samples = [delivery.load_sample(index) for index in range(len(delivery))]
    source_samples = [
        source.load_sample(source_by_id[sample.sample_id]) for sample in parquet_samples
    ]

    assert delivery.input_mode == "pose_only"
    assert delivery.radar_feature_dim is None
    for parquet_sample, source_sample in zip(parquet_samples, source_samples, strict=True):
        assert np.array_equal(parquet_sample.pose, source_sample.pose)
        assert np.array_equal(parquet_sample.pose_confidence, source_sample.pose_confidence)
        assert parquet_sample.radar_feature is None
        assert source_sample.radar_feature is None
        assert np.array_equal(parquet_sample.frame_mask, source_sample.frame_mask)
        assert parquet_sample.caption == source_sample.caption

    parquet_batch = collate_sign_language_translation_samples(parquet_samples, max_frames=4)
    source_batch = collate_sign_language_translation_samples(source_samples, max_frames=4)
    assert parquet_batch.radar_feature is None
    assert source_batch.radar_feature is None
    assert np.array_equal(parquet_batch.pose, source_batch.pose)
    assert np.array_equal(parquet_batch.pose_confidence, source_batch.pose_confidence)
    assert parquet_batch.captions == source_batch.captions


def test_pose_only_delivery_rejects_input_mode_metadata_tampering(tmp_path: Path) -> None:
    record = _pose_only_translation_record(tmp_path, "sample-001", 2)
    manifest_path = _write_manifest(tmp_path / "source.jsonl", [record])
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl", {"sample-001": "train"}
    )
    result = materialize_parquet_delivery(
        _config(
            tmp_path,
            SIGN_LANGUAGE_TRANSLATION_PRODUCT,
            manifest_path,
            assignments_path,
        ),
        runtime_report=_runtime(),
    )
    delivery_path = result.root / "delivery.json"
    payload = json.loads(delivery_path.read_text(encoding="utf-8"))
    payload["input_mode"] = "pose_plus_radar_feature"
    delivery_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ParquetDeliveryError, match="static dimensions do not match input mode"):
        validate_parquet_delivery(result.root, verify_checksums=False)

def test_layout_enforces_part_and_chunk_bounds_deterministically() -> None:
    part_ids = [f"sample-{index:04d}" for index in range(MAX_PART_ROWS + 1)]
    parts = plan_parquet_layout({"train": list(reversed(part_ids))})

    assert [len(part.sample_ids) for part in parts] == [MAX_PART_ROWS, 1]
    assert parts[0].relative_path.as_posix() == "splits/train/chunk-00000/part-00000.parquet"
    assert parts[1].relative_path.as_posix() == "splits/train/chunk-00000/part-00001.parquet"
    assert parts[0].sample_ids[0] == "sample-0000"

    chunks = plan_parquet_layout(
        {"train": [f"sample-{index:04d}" for index in range(MAX_PARTS_PER_CHUNK + 1)]},
        part_rows=1,
        parts_per_chunk=MAX_PARTS_PER_CHUNK,
    )
    assert chunks[MAX_PARTS_PER_CHUNK - 1].relative_path.as_posix().endswith(
        "chunk-00000/part-00063.parquet"
    )
    assert chunks[MAX_PARTS_PER_CHUNK].relative_path.as_posix().endswith(
        "chunk-00001/part-00000.parquet"
    )


def test_delivery_rejects_no_clobber_and_tampered_part(tmp_path: Path) -> None:
    records = [
        _pose_record(tmp_path, "sample-001", 2),
        _pose_record(tmp_path, "sample-002", 2),
    ]
    manifest_path = _write_manifest(tmp_path / "source.jsonl", records)
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl",
        {"sample-001": "train", "sample-002": "validation"},
    )
    config = _config(
        tmp_path,
        POSE_RECONSTRUCTION_PRODUCT,
        manifest_path,
        assignments_path,
    )
    result = materialize_parquet_delivery(config, runtime_report=_runtime())

    with pytest.raises(ParquetDeliveryError, match="already exists"):
        materialize_parquet_delivery(config, runtime_report=_runtime())

    part = result.root / "splits/train/chunk-00000/part-00000.parquet"
    part.write_bytes(part.read_bytes() + b"tamper")
    with pytest.raises(ParquetDeliveryError, match="byte count mismatch"):
        validate_parquet_delivery(result.root)


def test_materializer_requires_clean_runtime_and_model_ready_source(tmp_path: Path) -> None:
    record = _translation_record(tmp_path, "sample-001", 2)
    modalities = record["modalities"]
    assert isinstance(modalities, dict)
    del modalities["radar_feature"]
    manifest_path = _write_manifest(tmp_path / "source.jsonl", [record])
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl", {"sample-001": "train"}
    )
    config = _config(
        tmp_path,
        SIGN_LANGUAGE_TRANSLATION_PRODUCT,
        manifest_path,
        assignments_path,
    )
    with pytest.raises(ParquetDeliveryError, match="clean Git"):
        materialize_parquet_delivery(
            config,
            runtime_report={"git": {"commit": "a" * 40, "dirty": True}},
        )
    with pytest.raises(ParquetDeliveryError, match="missing modalities"):
        materialize_parquet_delivery(config, runtime_report=_runtime())


def test_delivery_config_is_portable_and_binds_exact_inputs(tmp_path: Path) -> None:
    record = _pose_record(tmp_path, "sample-001", 2)
    manifest_path = _write_manifest(tmp_path / "source.jsonl", [record])
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl", {"sample-001": "train"}
    )
    config_path = tmp_path / "delivery.yaml"
    config_path.write_text(
        f"""schema_version: mmprism.parquet_delivery_config.v1
source:
  data_root: {tmp_path}
  manifest: source.jsonl
  expected_manifest_sha256: {_sha256(manifest_path)}
  split_assignments: assignments.jsonl
  expected_split_assignment_sha256: {_sha256(assignments_path)}
  scope: partial
delivery:
  product: pose_reconstruction
  processed_root: processed
  part_rows: 1024
  parts_per_chunk: 64
validation:
  minimum_free_bytes: 0
  verify_source_checksums: true
""",
        encoding="utf-8",
    )

    loaded = load_parquet_delivery_config(config_path)
    assert loaded.data_root == tmp_path.resolve()
    assert loaded.source_manifest_path == manifest_path.resolve()
    assert loaded.expected_source_manifest_sha256 == _sha256(manifest_path)

    invalid_config = config_path.read_text(encoding="utf-8").replace(
        "manifest: source.jsonl", "manifest: ../source.jsonl"
    )
    config_path.write_text(invalid_config, encoding="utf-8")
    with pytest.raises(ParquetDeliveryError, match="safe portable relative"):
        load_parquet_delivery_config(config_path)


def test_delivery_cli_plan_and_validate_use_their_own_command_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_pose_record(tmp_path, "sample-001", 2)]
    manifest_path = _write_manifest(tmp_path / "source.jsonl", records)
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl", {"sample-001": "train"}
    )
    config_path = _write_delivery_config(
        tmp_path,
        POSE_RECONSTRUCTION_PRODUCT,
        manifest_path,
        assignments_path,
    )
    monkeypatch.setattr("mmprism.cli.collect_runtime_report", lambda _: _runtime())

    plan_output = io.StringIO()
    with redirect_stdout(plan_output):
        plan_exit = main(
            [
                "parquet-delivery-plan",
                str(config_path),
                "--project-root",
                str(tmp_path),
            ]
        )

    assert plan_exit == 0
    plan_payload = json.loads(plan_output.getvalue())
    assert plan_payload["product"] == POSE_RECONSTRUCTION_PRODUCT
    assert plan_payload["sample_count"] == 1

    result = materialize_parquet_delivery(
        load_parquet_delivery_config(config_path), runtime_report=_runtime()
    )
    validate_output = io.StringIO()
    with redirect_stdout(validate_output):
        validate_exit = main(["parquet-delivery-validate", str(result.root)])

    assert validate_exit == 0
    assert json.loads(validate_output.getvalue())["status"] == "passed"


def test_delivery_validation_rejects_inventory_row_group_drift_and_unlisted_part(
    tmp_path: Path,
) -> None:
    records = [
        _pose_record(tmp_path, "sample-001", 2),
        _pose_record(tmp_path, "sample-002", 2),
    ]
    manifest_path = _write_manifest(tmp_path / "source.jsonl", records)
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl",
        {"sample-001": "train", "sample-002": "validation"},
    )
    result = materialize_parquet_delivery(
        _config(
            tmp_path,
            POSE_RECONSTRUCTION_PRODUCT,
            manifest_path,
            assignments_path,
        ),
        runtime_report=_runtime(),
    )
    inventory_path = result.root / "inventories" / "parts.jsonl"
    inventory_rows = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    inventory_rows[0]["row_group_count"] = 2
    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory_rows),
        encoding="utf-8",
    )
    with pytest.raises(ParquetDeliveryError, match="row-group count mismatch"):
        validate_parquet_delivery(result.root, verify_checksums=False)

    inventory_rows[0]["row_group_count"] = 1
    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory_rows),
        encoding="utf-8",
    )
    part_path = result.root / "splits/train/chunk-00000/part-00001.parquet"
    part_path.write_bytes(
        (result.root / "splits/train/chunk-00000/part-00000.parquet").read_bytes()
    )
    with pytest.raises(ParquetDeliveryError, match="inventory coverage mismatch"):
        validate_parquet_delivery(result.root, verify_checksums=False)


def test_delivery_validation_rejects_metadata_contract_drift(tmp_path: Path) -> None:
    record = _pose_record(tmp_path, "sample-001", 2)
    manifest_path = _write_manifest(tmp_path / "source.jsonl", [record])
    assignments_path = _write_assignments(
        tmp_path / "assignments.jsonl", {"sample-001": "train"}
    )
    result = materialize_parquet_delivery(
        _config(
            tmp_path,
            POSE_RECONSTRUCTION_PRODUCT,
            manifest_path,
            assignments_path,
        ),
        runtime_report=_runtime(),
    )
    delivery_path = result.root / "delivery.json"
    delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
    delivery["row_policy"]["row_groups_per_part"] = 2
    delivery_path.write_text(json.dumps(delivery), encoding="utf-8")

    with pytest.raises(ParquetDeliveryError, match="exactly one row group"):
        validate_parquet_delivery(result.root, verify_checksums=False)


def test_parquet_dependency_error_names_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("blocked for dependency-boundary test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ParquetDeliveryError, match="data-parquet"):
        parquet_delivery._require_pyarrow()
