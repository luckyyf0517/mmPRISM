from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mmprism.artifacts.run import (
    RUN_SCHEMA_VERSION,
    ArtifactError,
    RunArtifactWriter,
    sha256_file,
)

PREDICTION_SHARD_SCHEMA_VERSION = "mmprism.prediction_shard.v1"
PREDICTION_AGGREGATION_SCHEMA_VERSION = "mmprism.prediction_aggregation.v1"
PREDICTION_INDEX_NAME = "predictions.index.json"
PREDICTION_NAME = "predictions.jsonl"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD_PREFIX = "predictions.rank-"


@dataclass(frozen=True, slots=True)
class PredictionShard:
    rank: int
    world_size: int
    record_count: int
    sample_id_sha256: str
    predictions_path: Path
    receipt_path: Path


@dataclass(frozen=True, slots=True)
class PredictionAggregation:
    record_count: int
    sample_id_sha256: str
    predictions_path: Path
    index_path: Path


def _utc_text(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ArtifactError("prediction artifact timestamps must be timezone-aware")
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _strict_json_line(record: Mapping[str, Any], location: str) -> bytes:
    try:
        return json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"{location} is not strict JSON: {error}") from error


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"prediction artifact is not strict JSON: {error}") from error
    return (serialized + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise ArtifactError(f"prediction artifact already exists: {destination}") from error
    temporary.unlink()


def _validate_rank(rank: int, world_size: int) -> None:
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ArtifactError("prediction world_size must be a positive integer")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank >= world_size
    ):
        raise ArtifactError("prediction rank must be an integer within [0, world_size)")


def _validate_prediction_schema(prediction_schema: str) -> str:
    if not isinstance(prediction_schema, str) or not prediction_schema.strip():
        raise ArtifactError("prediction schema must be a non-empty string")
    return prediction_schema.strip()


def _shard_names(rank: int, world_size: int) -> tuple[str, str]:
    _validate_rank(rank, world_size)
    width = max(5, len(str(world_size)))
    stem = f"predictions.rank-{rank:0{width}d}-of-{world_size:0{width}d}"
    return f"{stem}.jsonl", f"{stem}.json"


def _sample_id_bytes(sample_id: object, location: str) -> bytes:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ArtifactError(f"{location}.sample_id must be a non-empty string")
    return sample_id.encode("utf-8")


def _update_sample_digest(digest: Any, sample_id_bytes: bytes) -> None:
    digest.update(len(sample_id_bytes).to_bytes(8, byteorder="big", signed=False))
    digest.update(sample_id_bytes)


def _read_run(run_dir: Path, run_id: str) -> Mapping[str, Any]:
    run_path = run_dir / "run.json"
    try:
        payload: object = json.loads(run_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid run metadata: {run_path}") from error
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"run metadata must be a mapping: {run_path}")
    if payload.get("schema_version") != RUN_SCHEMA_VERSION or payload.get("run_id") != run_id:
        raise ArtifactError(f"prediction run identity mismatch: {run_path}")
    if payload.get("status") != "initialized":
        raise ArtifactError("prediction artifacts require an initialized run")
    return payload


def _validate_prediction_record(
    record: object,
    *,
    prediction_schema: str,
    location: str,
) -> tuple[Mapping[str, Any], bytes, bytes]:
    if not isinstance(record, Mapping):
        raise ArtifactError(f"{location} must be a mapping")
    if record.get("schema_version") != prediction_schema:
        raise ArtifactError(f"{location}.schema_version must equal {prediction_schema!r}")
    sample_id = _sample_id_bytes(record.get("sample_id"), location)
    line = _strict_json_line(record, location)
    return record, sample_id, line


def write_prediction_shard(
    run_dir: str | Path,
    *,
    run_id: str,
    prediction_schema: str,
    rank: int,
    world_size: int,
    records: Iterable[Mapping[str, Any]],
    created_at: datetime | None = None,
) -> PredictionShard:
    """Write one immutable rank-local shard without modifying shared run metadata."""

    _validate_rank(rank, world_size)
    schema = _validate_prediction_schema(prediction_schema)
    destination_dir = Path(run_dir).expanduser().resolve()
    _read_run(destination_dir, run_id)
    prediction_name, receipt_name = _shard_names(rank, world_size)
    prediction_path = destination_dir / prediction_name
    receipt_path = destination_dir / receipt_name
    if prediction_path.exists() or receipt_path.exists():
        raise ArtifactError(
            f"prediction shard artifact already exists for rank {rank}: {prediction_path}"
        )

    temporary = prediction_path.with_name(
        f".{prediction_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    record_count = 0
    sample_ids: set[bytes] = set()
    sample_digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            for index, candidate in enumerate(records):
                _, sample_id, line = _validate_prediction_record(
                    candidate,
                    prediction_schema=schema,
                    location=f"prediction shard rank {rank} record {index}",
                )
                if sample_id in sample_ids:
                    raise ArtifactError(
                        "prediction shard contains duplicate sample_id "
                        f"{candidate.get('sample_id')!r}"
                    )
                sample_ids.add(sample_id)
                _update_sample_digest(sample_digest, sample_id)
                handle.write(line + b"\n")
                record_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        shard_sha256 = sha256_file(temporary)
        shard_size = temporary.stat().st_size
        receipt = {
            "schema_version": PREDICTION_SHARD_SCHEMA_VERSION,
            "run_id": run_id,
            "prediction_schema": schema,
            "rank": rank,
            "world_size": world_size,
            "record_count": record_count,
            "sample_id_sha256": sample_digest.hexdigest(),
            "created_at": _utc_text(created_at),
            "shard": {
                "name": prediction_name,
                "sha256": shard_sha256,
                "size_bytes": shard_size,
            },
        }
        _publish_no_replace(temporary, prediction_path)
        _atomic_write_bytes(receipt_path, _pretty_json_bytes(receipt))
    finally:
        temporary.unlink(missing_ok=True)

    return PredictionShard(
        rank=rank,
        world_size=world_size,
        record_count=record_count,
        sample_id_sha256=sample_digest.hexdigest(),
        predictions_path=prediction_path,
        receipt_path=receipt_path,
    )


def _required_integer(payload: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"prediction shard receipt {key} must be an integer >= {minimum}")
    return value


def _required_sha256(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ArtifactError(f"prediction shard receipt {key} must be a lowercase SHA-256")
    return value


def _read_receipt(path: Path) -> Mapping[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid prediction shard receipt: {path}") from error
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"prediction shard receipt must be a mapping: {path}")
    expected_keys = {
        "schema_version",
        "run_id",
        "prediction_schema",
        "rank",
        "world_size",
        "record_count",
        "sample_id_sha256",
        "created_at",
        "shard",
    }
    if set(payload) != expected_keys:
        raise ArtifactError(f"prediction shard receipt keys do not match the contract: {path}")
    return payload


def _validate_receipt(
    path: Path,
    *,
    run_id: str,
    prediction_schema: str,
    rank: int,
    world_size: int,
    prediction_name: str,
) -> Mapping[str, Any]:
    receipt = _read_receipt(path)
    if receipt.get("schema_version") != PREDICTION_SHARD_SCHEMA_VERSION:
        raise ArtifactError(f"unsupported prediction shard receipt schema: {path}")
    if receipt.get("run_id") != run_id:
        raise ArtifactError(f"prediction shard receipt run ID mismatch: {path}")
    if receipt.get("prediction_schema") != prediction_schema:
        raise ArtifactError(f"prediction shard receipt prediction schema mismatch: {path}")
    if _required_integer(receipt, "rank") != rank:
        raise ArtifactError(f"prediction shard receipt rank mismatch: {path}")
    if _required_integer(receipt, "world_size", minimum=1) != world_size:
        raise ArtifactError(f"prediction shard receipt world size mismatch: {path}")
    _required_integer(receipt, "record_count")
    _required_sha256(receipt, "sample_id_sha256")
    created_at = receipt.get("created_at")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise ArtifactError(f"prediction shard receipt created_at is invalid: {path}")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ArtifactError(f"prediction shard receipt created_at is invalid: {path}") from error
    if parsed_created_at.utcoffset() is None:
        raise ArtifactError(f"prediction shard receipt created_at is invalid: {path}")
    shard = receipt.get("shard")
    if not isinstance(shard, Mapping) or set(shard) != {"name", "sha256", "size_bytes"}:
        raise ArtifactError(f"prediction shard receipt file metadata is invalid: {path}")
    if shard.get("name") != prediction_name:
        raise ArtifactError(f"prediction shard receipt filename mismatch: {path}")
    _required_sha256(shard, "sha256")
    _required_integer(shard, "size_bytes")
    return receipt


def _expected_sample_ids(values: Iterable[str]) -> tuple[set[str], set[str]]:
    expected: set[str] = set()
    for index, value in enumerate(values):
        _sample_id_bytes(value, f"expected sample {index}")
        if value in expected:
            raise ArtifactError(f"expected sample IDs contain duplicate {value!r}")
        expected.add(value)
    if not expected:
        raise ArtifactError("prediction aggregation requires expected sample IDs")
    return expected, set(expected)


def _unexpected_rank_artifacts(run_dir: Path, expected_names: set[str]) -> list[str]:
    observed = {
        path.name
        for path in run_dir.glob(f"{_SHARD_PREFIX}*")
        if not path.name.startswith(".")
    }
    return sorted(observed - expected_names)


def _artifact_metadata(path: Path) -> dict[str, int | str]:
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_shard_into_database(
    connection: sqlite3.Connection,
    path: Path,
    receipt: Mapping[str, Any],
    *,
    prediction_schema: str,
    expected: set[str],
    remaining: set[str],
) -> None:
    before = path.stat()
    shard_hash = hashlib.sha256()
    sample_digest = hashlib.sha256()
    record_count = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            shard_hash.update(raw_line)
            if not raw_line.endswith(b"\n"):
                raise ArtifactError(f"prediction shard line lacks newline: {path}:{line_number}")
            try:
                candidate: object = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArtifactError(f"invalid prediction JSON at {path}:{line_number}") from error
            record, sample_id_bytes, canonical = _validate_prediction_record(
                candidate,
                prediction_schema=prediction_schema,
                location=f"{path}:{line_number}",
            )
            if raw_line != canonical + b"\n":
                raise ArtifactError(
                    f"prediction shard record is not canonical: {path}:{line_number}"
                )
            sample_id = str(record["sample_id"])
            if sample_id not in expected:
                raise ArtifactError(f"unexpected prediction sample_id {sample_id!r}")
            if sample_id not in remaining:
                raise ArtifactError(f"duplicate prediction sample_id {sample_id!r}")
            remaining.remove(sample_id)
            _update_sample_digest(sample_digest, sample_id_bytes)
            try:
                connection.execute(
                    "INSERT INTO predictions(sample_key, payload) VALUES (?, ?)",
                    (sqlite3.Binary(sample_id_bytes), sqlite3.Binary(canonical)),
                )
            except sqlite3.IntegrityError as error:
                raise ArtifactError(f"duplicate prediction sample_id {sample_id!r}") from error
            record_count += 1
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ArtifactError(f"prediction shard changed while aggregating: {path}")

    shard = receipt["shard"]
    assert isinstance(shard, Mapping)
    if shard_hash.hexdigest() != shard["sha256"] or after.st_size != shard["size_bytes"]:
        raise ArtifactError(f"prediction shard checksum or size mismatch: {path}")
    if record_count != receipt["record_count"]:
        raise ArtifactError(f"prediction shard record count mismatch: {path}")
    if sample_digest.hexdigest() != receipt["sample_id_sha256"]:
        raise ArtifactError(f"prediction shard sample-ID digest mismatch: {path}")


def _write_merged_predictions(
    connection: sqlite3.Connection,
    destination: Path,
) -> tuple[int, str]:
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    record_count = 0
    sample_digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            cursor = connection.execute(
                "SELECT sample_key, payload FROM predictions ORDER BY sample_key"
            )
            for sample_key, payload in cursor:
                key = bytes(sample_key)
                _update_sample_digest(sample_digest, key)
                handle.write(bytes(payload) + b"\n")
                record_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return record_count, sample_digest.hexdigest()


def aggregate_prediction_shards(
    writer: RunArtifactWriter,
    *,
    prediction_schema: str,
    world_size: int,
    expected_sample_ids: Iterable[str],
    created_at: datetime | None = None,
) -> PredictionAggregation:
    """Verify all rank shards, merge exact sample coverage, and register one result set."""

    _validate_rank(0, world_size)
    schema = _validate_prediction_schema(prediction_schema)
    _read_run(writer.run_dir, writer.run_id)
    expected, remaining = _expected_sample_ids(expected_sample_ids)

    shard_names = [_shard_names(rank, world_size) for rank in range(world_size)]
    expected_names = {name for names in shard_names for name in names}
    unexpected = _unexpected_rank_artifacts(writer.run_dir, expected_names)
    if unexpected:
        raise ArtifactError(f"unexpected prediction rank artifact: {unexpected[0]}")
    missing = sorted(name for name in expected_names if not (writer.run_dir / name).is_file())
    if missing:
        raise ArtifactError(f"missing prediction rank artifact: {missing[0]}")
    for name in (PREDICTION_NAME, PREDICTION_INDEX_NAME):
        if (writer.run_dir / name).exists():
            raise ArtifactError(f"prediction aggregation artifact already exists: {name}")

    shard_index: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".prediction-aggregate-", dir=writer.run_dir) as temp:
        database_path = Path(temp) / "records.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                "CREATE TABLE predictions ("
                "sample_key BLOB PRIMARY KEY, payload BLOB NOT NULL"
                ") WITHOUT ROWID"
            )
            for rank, (prediction_name, receipt_name) in enumerate(shard_names):
                prediction_path = writer.run_dir / prediction_name
                receipt_path = writer.run_dir / receipt_name
                receipt = _validate_receipt(
                    receipt_path,
                    run_id=writer.run_id,
                    prediction_schema=schema,
                    rank=rank,
                    world_size=world_size,
                    prediction_name=prediction_name,
                )
                _read_shard_into_database(
                    connection,
                    prediction_path,
                    receipt,
                    prediction_schema=schema,
                    expected=expected,
                    remaining=remaining,
                )
                shard_index.append(
                    {
                        "rank": rank,
                        "record_count": receipt["record_count"],
                        "sample_id_sha256": receipt["sample_id_sha256"],
                        "predictions": _artifact_metadata(prediction_path),
                        "receipt": _artifact_metadata(receipt_path),
                    }
                )
            if remaining:
                first_missing = min(remaining, key=lambda value: value.encode("utf-8"))
                raise ArtifactError(
                    f"prediction aggregation is missing {len(remaining)} expected samples; "
                    f"first: {first_missing!r}"
                )
            connection.commit()
            predictions_path = writer.run_dir / PREDICTION_NAME
            record_count, sample_id_sha256 = _write_merged_predictions(
                connection, predictions_path
            )
        finally:
            connection.close()

    if record_count != len(expected):
        raise ArtifactError("merged prediction count does not match expected sample coverage")
    index_path = writer.run_dir / PREDICTION_INDEX_NAME
    index_payload = {
        "schema_version": PREDICTION_AGGREGATION_SCHEMA_VERSION,
        "run_id": writer.run_id,
        "prediction_schema": schema,
        "world_size": world_size,
        "record_count": record_count,
        "sample_id_sha256": sample_id_sha256,
        "created_at": _utc_text(created_at),
        "coverage": {
            "expected": len(expected),
            "observed": record_count,
            "missing": 0,
            "extra": 0,
        },
        "shards": shard_index,
        "merged": _artifact_metadata(predictions_path),
    }
    _atomic_write_bytes(index_path, _pretty_json_bytes(index_payload))
    writer.register_artifacts(
        tuple(name for names in shard_names for name in names)
        + (PREDICTION_NAME, PREDICTION_INDEX_NAME)
    )
    return PredictionAggregation(
        record_count=record_count,
        sample_id_sha256=sample_id_sha256,
        predictions_path=predictions_path,
        index_path=index_path,
    )


def write_single_rank_predictions(
    writer: RunArtifactWriter,
    *,
    prediction_schema: str,
    records: Iterable[Mapping[str, Any]],
    expected_sample_ids: Iterable[str],
) -> PredictionAggregation:
    """Exercise the distributed artifact contract for a world-size-one formal run."""

    write_prediction_shard(
        writer.run_dir,
        run_id=writer.run_id,
        prediction_schema=prediction_schema,
        rank=0,
        world_size=1,
        records=records,
    )
    return aggregate_prediction_shards(
        writer,
        prediction_schema=prediction_schema,
        world_size=1,
        expected_sample_ids=expected_sample_ids,
    )
