import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mmprism.artifacts import (
    PREDICTION_AGGREGATION_SCHEMA_VERSION,
    PREDICTION_SHARD_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    ArtifactError,
    RunArtifactWriter,
    aggregate_prediction_shards,
    write_prediction_shard,
)

PREDICTION_SCHEMA = "fixture.prediction.v1"


def _writer(root: Path, *, status: str = "initialized") -> RunArtifactWriter:
    run_id = "fixture-run"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": run_id,
                "status": status,
                "artifacts": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return RunArtifactWriter(run_dir=run_dir, run_id=run_id)


def _record(sample_id: str, value: int) -> dict[str, object]:
    return {
        "schema_version": PREDICTION_SCHEMA,
        "sample_id": sample_id,
        "value": value,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_rank_local_shards_are_independent_and_aggregate_deterministically(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    original_run = (writer.run_dir / "run.json").read_bytes()
    created_at = datetime(2026, 8, 11, 23, 0, tzinfo=UTC)

    first = write_prediction_shard(
        writer.run_dir,
        run_id=writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=0,
        world_size=3,
        records=(_record("sample-c", 3), _record("sample-a", 1)),
        created_at=created_at,
    )
    second = write_prediction_shard(
        writer.run_dir,
        run_id=writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=1,
        world_size=3,
        records=(_record("sample-b", 2),),
        created_at=created_at,
    )
    empty = write_prediction_shard(
        writer.run_dir,
        run_id=writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=2,
        world_size=3,
        records=(),
        created_at=created_at,
    )

    assert (writer.run_dir / "run.json").read_bytes() == original_run
    assert (first.record_count, second.record_count, empty.record_count) == (2, 1, 0)
    assert empty.predictions_path.read_bytes() == b""
    first_receipt = _read_json(first.receipt_path)
    assert first_receipt["schema_version"] == PREDICTION_SHARD_SCHEMA_VERSION
    assert first_receipt["run_id"] == writer.run_id
    assert first_receipt["rank"] == 0
    assert first_receipt["world_size"] == 3
    assert first_receipt["record_count"] == 2
    assert first_receipt["shard"] == {
        "name": first.predictions_path.name,
        "sha256": hashlib.sha256(first.predictions_path.read_bytes()).hexdigest(),
        "size_bytes": first.predictions_path.stat().st_size,
    }

    result = aggregate_prediction_shards(
        writer,
        prediction_schema=PREDICTION_SCHEMA,
        world_size=3,
        expected_sample_ids=("sample-b", "sample-c", "sample-a"),
        created_at=created_at,
    )

    prediction_lines = result.predictions_path.read_text(encoding="utf-8").splitlines()
    predictions = [json.loads(line) for line in prediction_lines]
    assert [record["sample_id"] for record in predictions] == [
        "sample-a",
        "sample-b",
        "sample-c",
    ]
    index = _read_json(result.index_path)
    assert index["schema_version"] == PREDICTION_AGGREGATION_SCHEMA_VERSION
    assert index["record_count"] == 3
    assert index["coverage"] == {"expected": 3, "extra": 0, "missing": 0, "observed": 3}
    assert [shard["record_count"] for shard in index["shards"]] == [2, 1, 0]  # type: ignore[index]
    run = _read_json(writer.run_dir / "run.json")
    registered = set(run["artifacts"])  # type: ignore[arg-type]
    assert registered == {
        "predictions.rank-00000-of-00003.jsonl",
        "predictions.rank-00000-of-00003.json",
        "predictions.rank-00001-of-00003.jsonl",
        "predictions.rank-00001-of-00003.json",
        "predictions.rank-00002-of-00003.jsonl",
        "predictions.rank-00002-of-00003.json",
        "predictions.jsonl",
        "predictions.index.json",
    }
    assert not any(path.name.startswith(".") for path in writer.run_dir.iterdir())


def test_aggregation_rejects_cross_rank_duplicates_without_registering(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    original_run = (writer.run_dir / "run.json").read_bytes()
    for rank in range(2):
        write_prediction_shard(
            writer.run_dir,
            run_id=writer.run_id,
            prediction_schema=PREDICTION_SCHEMA,
            rank=rank,
            world_size=2,
            records=(_record("duplicate", rank),),
        )

    with pytest.raises(ArtifactError, match="duplicate prediction sample_id"):
        aggregate_prediction_shards(
            writer,
            prediction_schema=PREDICTION_SCHEMA,
            world_size=2,
            expected_sample_ids=("duplicate",),
        )

    assert not (writer.run_dir / "predictions.jsonl").exists()
    assert not (writer.run_dir / "predictions.index.json").exists()
    assert (writer.run_dir / "run.json").read_bytes() == original_run


def test_aggregation_rejects_missing_extra_and_incomplete_rank_sets(tmp_path: Path) -> None:
    missing_writer = _writer(tmp_path / "missing")
    write_prediction_shard(
        missing_writer.run_dir,
        run_id=missing_writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=0,
        world_size=2,
        records=(_record("one", 1),),
    )
    with pytest.raises(ArtifactError, match="missing prediction rank artifact"):
        aggregate_prediction_shards(
            missing_writer,
            prediction_schema=PREDICTION_SCHEMA,
            world_size=2,
            expected_sample_ids=("one",),
        )

    extra_writer = _writer(tmp_path / "extra")
    write_prediction_shard(
        extra_writer.run_dir,
        run_id=extra_writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=0,
        world_size=1,
        records=(_record("one", 1),),
    )
    (extra_writer.run_dir / "predictions.rank-unexpected.jsonl").write_bytes(b"")
    with pytest.raises(ArtifactError, match="unexpected prediction rank artifact"):
        aggregate_prediction_shards(
            extra_writer,
            prediction_schema=PREDICTION_SCHEMA,
            world_size=1,
            expected_sample_ids=("one",),
        )

    incomplete_writer = _writer(tmp_path / "incomplete")
    write_prediction_shard(
        incomplete_writer.run_dir,
        run_id=incomplete_writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=0,
        world_size=1,
        records=(_record("one", 1),),
    )
    with pytest.raises(ArtifactError, match="missing 1 expected samples"):
        aggregate_prediction_shards(
            incomplete_writer,
            prediction_schema=PREDICTION_SCHEMA,
            world_size=1,
            expected_sample_ids=("one", "two"),
        )


def test_aggregation_rejects_shard_tampering(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    shard = write_prediction_shard(
        writer.run_dir,
        run_id=writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=0,
        world_size=1,
        records=(_record("one", 1),),
    )
    tampered = shard.predictions_path.read_bytes().replace(b'"value":1', b'"value":2')
    shard.predictions_path.write_bytes(tampered)

    with pytest.raises(ArtifactError, match="checksum or size mismatch"):
        aggregate_prediction_shards(
            writer,
            prediction_schema=PREDICTION_SCHEMA,
            world_size=1,
            expected_sample_ids=("one",),
        )


def test_aggregation_rejects_receipt_identity_tampering(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    shard = write_prediction_shard(
        writer.run_dir,
        run_id=writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=0,
        world_size=1,
        records=(_record("one", 1),),
    )
    receipt = _read_json(shard.receipt_path)
    receipt["rank"] = 1
    shard.receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ArtifactError, match="receipt rank mismatch"):
        aggregate_prediction_shards(
            writer,
            prediction_schema=PREDICTION_SCHEMA,
            world_size=1,
            expected_sample_ids=("one",),
        )


def test_shard_writer_rejects_invalid_records_collisions_and_finalized_runs(
    tmp_path: Path,
) -> None:
    invalid_writer = _writer(tmp_path / "invalid")
    with pytest.raises(ArtifactError, match="schema_version"):
        write_prediction_shard(
            invalid_writer.run_dir,
            run_id=invalid_writer.run_id,
            prediction_schema=PREDICTION_SCHEMA,
            rank=0,
            world_size=1,
            records=({"schema_version": "wrong", "sample_id": "one"},),
        )
    with pytest.raises(ArtifactError, match="strict JSON"):
        write_prediction_shard(
            invalid_writer.run_dir,
            run_id=invalid_writer.run_id,
            prediction_schema=PREDICTION_SCHEMA,
            rank=0,
            world_size=1,
            records=(
                {
                    "schema_version": PREDICTION_SCHEMA,
                    "sample_id": "one",
                    "value": float("nan"),
                },
            ),
        )
    with pytest.raises(ArtifactError, match="duplicate sample_id"):
        write_prediction_shard(
            invalid_writer.run_dir,
            run_id=invalid_writer.run_id,
            prediction_schema=PREDICTION_SCHEMA,
            rank=0,
            world_size=1,
            records=(_record("one", 1), _record("one", 2)),
        )
    assert not any(
        path.name.startswith("predictions.rank-") for path in invalid_writer.run_dir.iterdir()
    )

    write_prediction_shard(
        invalid_writer.run_dir,
        run_id=invalid_writer.run_id,
        prediction_schema=PREDICTION_SCHEMA,
        rank=0,
        world_size=1,
        records=(_record("one", 1),),
    )
    with pytest.raises(ArtifactError, match="already exists"):
        write_prediction_shard(
            invalid_writer.run_dir,
            run_id=invalid_writer.run_id,
            prediction_schema=PREDICTION_SCHEMA,
            rank=0,
            world_size=1,
            records=(_record("one", 1),),
        )

    finalized_writer = _writer(tmp_path / "finalized", status="completed")
    with pytest.raises(ArtifactError, match="initialized run"):
        write_prediction_shard(
            finalized_writer.run_dir,
            run_id=finalized_writer.run_id,
            prediction_schema=PREDICTION_SCHEMA,
            rank=0,
            world_size=1,
            records=(),
        )
