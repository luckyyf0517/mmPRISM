from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import numpy as np

from mmprism.contracts import (
    POSE_ONLY_INPUT_MODE,
    POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
    TRANSLATION_INPUT_MODES,
    ManifestError,
    SampleRecord,
    SplitAssignment,
    translation_delivery_supports_wavellm_input_mode,
    translation_input_mode_modalities,
    validate_manifest,
    validate_split_assignments,
)
from mmprism.data.online_pose_control import (
    ONLINE_POSE_AUDIT_MODALITY,
    ONLINE_POSE_CAPTION_MODALITY,
    ONLINE_POSE_CONTROL_SAMPLE_PROTOCOL,
    ONLINE_POSE_MODALITY,
    OnlinePoseControlDataError,
    OnlinePoseControlSample,
    validate_online_pose_arrays,
)
from mmprism.data.pose_reconstruction import POSE_RECONSTRUCTION_SAMPLE_PROTOCOL
from mmprism.data.sign_language_translation import (
    SIGN_LANGUAGE_TRANSLATION_SAMPLE_PROTOCOL,
    TRANSLATION_POSE_UNITS,
    SignLanguageTranslationSample,
)

PARQUET_DELIVERY_SCHEMA = "mmprism.parquet_delivery.v3"

LEGACY_PARQUET_DELIVERY_SCHEMA = "mmprism.parquet_delivery.v2"

PARQUET_DELIVERY_CONFIG_SCHEMA = "mmprism.parquet_delivery_config.v1"

PARQUET_DELIVERY_SCHEMA_FILE = "mmprism.parquet_delivery_schema.v3"

LEGACY_PARQUET_DELIVERY_SCHEMA_FILE = "mmprism.parquet_delivery_schema.v2"

PARQUET_DELIVERY_INVENTORY_SCHEMA = "mmprism.parquet_delivery_inventory.v1"

PARQUET_DELIVERY_INDEX_SCHEMA = "mmprism.parquet_delivery_index.v2"

LEGACY_PARQUET_DELIVERY_INDEX_SCHEMA = "mmprism.parquet_delivery_index.v1"

PARQUET_DELIVERY_READER_PART_CACHE_SIZE = 2

PARQUET_DELIVERY_VALIDATION_SCHEMA = "mmprism.parquet_delivery_validation.v1"

PARQUET_ONE_SAMPLE_PER_ROW_GROUP = "one_sample_per_row_group_v1"

PARQUET_LEGACY_PART_ROW_GROUP = "one_row_group_per_part_v1"

POSE_RECONSTRUCTION_PRODUCT = "pose_reconstruction"

SIGN_LANGUAGE_TRANSLATION_PRODUCT = "sign_language_translation"

ONLINE_POSE_CONTROL_PRODUCT = "online_pose_control"

SUPPORTED_PRODUCTS = frozenset(
    (
        ONLINE_POSE_CONTROL_PRODUCT,
        POSE_RECONSTRUCTION_PRODUCT,
        SIGN_LANGUAGE_TRANSLATION_PRODUCT,
    )
)

PRODUCT_PROTOCOLS = {
    ONLINE_POSE_CONTROL_PRODUCT: ONLINE_POSE_CONTROL_SAMPLE_PROTOCOL,
    POSE_RECONSTRUCTION_PRODUCT: POSE_RECONSTRUCTION_SAMPLE_PROTOCOL,
    SIGN_LANGUAGE_TRANSLATION_PRODUCT: SIGN_LANGUAGE_TRANSLATION_SAMPLE_PROTOCOL,
}

MAX_PART_ROWS = 1024

MAX_PARTS_PER_CHUNK = 64

_BUILD_ID_PATTERN = re.compile(r"build-[0-9a-f]{24}")

_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ParquetDeliveryError(RuntimeError):
    """Raised when final Parquet delivery data violates its strict contract."""


@dataclass(frozen=True, slots=True)
class ParquetDeliveryConfig:
    """Portable policy plus local source/destination roots for one immutable build."""

    product: str
    data_root: Path
    source_manifest_path: Path
    split_assignment_path: Path
    processed_root: Path
    expected_source_manifest_sha256: str
    expected_split_assignment_sha256: str
    source_scope: Literal["partial", "complete"]
    part_rows: int = MAX_PART_ROWS
    parts_per_chunk: int = MAX_PARTS_PER_CHUNK
    minimum_free_bytes: int = 0
    verify_source_checksums: bool = True

    def portable_dict(self) -> dict[str, object]:
        return {
            "schema_version": PARQUET_DELIVERY_CONFIG_SCHEMA,
            "product": self.product,
            "expected_source_manifest_sha256": self.expected_source_manifest_sha256,
            "expected_split_assignment_sha256": self.expected_split_assignment_sha256,
            "source_scope": self.source_scope,
            "part_rows": self.part_rows,
            "parts_per_chunk": self.parts_per_chunk,
            "minimum_free_bytes": self.minimum_free_bytes,
            "verify_source_checksums": self.verify_source_checksums,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.portable_dict()))


@dataclass(frozen=True, slots=True)
class ParquetDeliveryPartPlan:
    split: str
    chunk_index: int
    part_index: int
    sample_ids: tuple[str, ...]

    @property
    def relative_path(self) -> PurePosixPath:
        return PurePosixPath(
            "splits",
            self.split,
            f"chunk-{self.chunk_index:05d}",
            f"part-{self.part_index:05d}.parquet",
        )


@dataclass(frozen=True, slots=True)
class ParquetDeliveryPlan:
    config: ParquetDeliveryConfig
    build_id: str
    git_commit: str
    runtime_report: dict[str, object]
    source_manifest_sha256: str
    split_assignment_sha256: str
    records_by_id: dict[str, SampleRecord]
    assignments_by_id: dict[str, SplitAssignment]
    parts: tuple[ParquetDeliveryPartPlan, ...]
    datasets: tuple[str, ...]
    input_mode: str | None
    static_dimensions: dict[str, int]
    coordinate_frame: str
    pose_units: str
    estimated_payload_bytes: int
    estimated_staging_bytes: int
    required_free_bytes: int

    @property
    def delivery_root(self) -> Path:
        return (
            self.config.processed_root.expanduser().resolve()
            / self.config.product
            / PARQUET_DELIVERY_SCHEMA
            / self.build_id
        )

    @property
    def sample_count(self) -> int:
        return len(self.records_by_id)

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted({part.split for part in self.parts}))


@dataclass(frozen=True, slots=True)
class ParquetDeliveryValidation:
    root: Path
    product: str
    build_id: str
    sample_count: int
    part_count: int
    split_counts: dict[str, int]
    inventory_sha256: str
    index_sha256: str
    schema_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PARQUET_DELIVERY_VALIDATION_SCHEMA,
            "status": "passed",
            "root": ".",
            "product": self.product,
            "build_id": self.build_id,
            "sample_count": self.sample_count,
            "part_count": self.part_count,
            "split_counts": self.split_counts,
            "inventory_sha256": self.inventory_sha256,
            "index_sha256": self.index_sha256,
            "schema_fingerprint": self.schema_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ParquetSampleLocation:
    """Checksum-bound physical location for one indexed delivery sample."""

    part_path: str
    row_index: int
    row_group_index: int | None = None


class ParquetIndexedSample(Protocol):
    """Structural reference to one sample in a one-row-group delivery."""

    @property
    def sample_id(self) -> str: ...

    @property
    def part_path(self) -> str: ...

    @property
    def row_index(self) -> int: ...

    @property
    def row_group_index(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _DeliveryMetadata:
    root: Path
    schema_version: str
    product: str
    build_id: str
    source_manifest_sha256: str
    split_assignment_sha256: str
    part_rows: int
    parts_per_chunk: int
    row_group_layout: str
    input_mode: str | None
    coordinate_frame: str
    pose_units: str
    static_dimensions: dict[str, int]
    delivery: dict[str, object]


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    sample_id: str
    relative_path: PurePosixPath
    row_index: int
    row_group_index: int | None
    group_id: str | None = None


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise ParquetDeliveryError(
            "Parquet delivery requires optional data-parquet; run uv sync --extra data-parquet"
        ) from error
    return pa, pq


def _require_text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParquetDeliveryError(f"{location} must be a non-empty string")
    return value.strip()


def _require_sha256(value: object, location: str) -> str:
    text = _require_text(value, location)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ParquetDeliveryError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _require_git_commit(value: object, location: str) -> str:
    text = _require_text(value, location)
    if not _GIT_COMMIT_PATTERN.fullmatch(text):
        raise ParquetDeliveryError(f"{location} must be a lowercase 40-character Git commit")
    return text


def _require_integer(
    value: object,
    location: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParquetDeliveryError(f"{location} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ParquetDeliveryError(f"{location} must be in {bound}")
    return value


def _reject_unknown(payload: Mapping[str, object], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ParquetDeliveryError(f"unknown keys in {location}: {', '.join(unknown)}")


def _relative_path(value: object, location: str) -> PurePosixPath:
    text = _require_text(value, location)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ParquetDeliveryError(f"{location} must be a safe portable relative path")
    return path


def _resolve_relative(root: Path, relative: PurePosixPath, location: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ParquetDeliveryError(f"{location} escapes delivery root")
    return candidate


def _runtime_commit(runtime_report: Mapping[str, object], *, require_clean: bool) -> str:
    git = runtime_report.get("git")
    if not isinstance(git, Mapping):
        raise ParquetDeliveryError("runtime report has no Git provenance")
    try:
        commit = _require_git_commit(git.get("commit"), "runtime report Git commit")
    except ParquetDeliveryError as error:
        raise ParquetDeliveryError("runtime report has no valid Git commit") from error
    if require_clean and git.get("dirty") is not False:
        raise ParquetDeliveryError("final Parquet delivery requires a clean Git worktree")
    return commit


def _portable_runtime_report(runtime_report: Mapping[str, object]) -> dict[str, object]:
    """Keep runtime provenance without serializing machine-local roots."""

    git = runtime_report.get("git")
    if not isinstance(git, Mapping):
        raise ParquetDeliveryError("runtime report has no Git provenance")
    commit = _runtime_commit(runtime_report, require_clean=False)
    dirty = git.get("dirty")
    if not isinstance(dirty, bool):
        raise ParquetDeliveryError("runtime report Git dirty state must be boolean")

    packages = runtime_report.get("packages")
    portable_packages: dict[str, str | None] = {}
    if packages is not None:
        if not isinstance(packages, Mapping):
            raise ParquetDeliveryError("runtime report packages must be an object")
        for name, version in packages.items():
            if not isinstance(name, str) or not name:
                raise ParquetDeliveryError("runtime report package names must be non-empty strings")
            if version is not None and not isinstance(version, str):
                raise ParquetDeliveryError(
                    f"runtime report package version for {name!r} must be text or null"
                )
            portable_packages[name] = version

    portable: dict[str, object] = {
        "python": runtime_report.get("python")
        if isinstance(runtime_report.get("python"), str)
        else None,
        "platform": runtime_report.get("platform")
        if isinstance(runtime_report.get("platform"), str)
        else None,
        "packages": dict(sorted(portable_packages.items())),
        "git": {"commit": commit, "dirty": dirty},
    }
    return portable


def _read_source_records(path: Path) -> dict[str, SampleRecord]:
    try:
        validate_manifest(path)
    except ManifestError as error:
        raise ParquetDeliveryError(f"invalid source manifest {path}: {error}") from error

    records: dict[str, SampleRecord] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = SampleRecord.from_mapping(
                    json.loads(raw_line), f"source line {line_number}"
                )
            except (json.JSONDecodeError, ManifestError) as error:
                raise ParquetDeliveryError(
                    f"invalid source manifest record at {path}:{line_number}: {error}"
                ) from error
            if record.sample_id in records:
                raise ParquetDeliveryError(
                    f"duplicate source manifest sample_id {record.sample_id!r}"
                )
            records[record.sample_id] = record
    if not records:
        raise ParquetDeliveryError("source manifest contains no records")
    return records


def _read_assignments(path: Path, expected_sample_ids: set[str]) -> dict[str, SplitAssignment]:
    try:
        validate_split_assignments(path, expected_sample_ids=expected_sample_ids)
    except ValueError as error:
        raise ParquetDeliveryError(f"invalid split assignment {path}: {error}") from error

    assignments: dict[str, SplitAssignment] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                assignment = SplitAssignment.from_mapping(
                    json.loads(raw_line), f"split line {line_number}"
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ParquetDeliveryError(
                    f"invalid split assignment at {path}:{line_number}: {error}"
                ) from error
            assignments[assignment.sample_id] = assignment
    return assignments


def _layout_parts(
    sample_ids_by_split: Mapping[str, Sequence[str]],
    *,
    part_rows: int,
    parts_per_chunk: int,
) -> tuple[ParquetDeliveryPartPlan, ...]:
    plans: list[ParquetDeliveryPartPlan] = []
    for split in sorted(sample_ids_by_split):
        sample_ids = tuple(sorted(sample_ids_by_split[split], key=lambda value: value.encode()))
        for global_part_index, start in enumerate(range(0, len(sample_ids), part_rows)):
            chunk_index, part_index = divmod(global_part_index, parts_per_chunk)
            plans.append(
                ParquetDeliveryPartPlan(
                    split=split,
                    chunk_index=chunk_index,
                    part_index=part_index,
                    sample_ids=sample_ids[start : start + part_rows],
                )
            )
    return tuple(plans)


def plan_parquet_layout(
    sample_ids_by_split: Mapping[str, Sequence[str]],
    *,
    part_rows: int = MAX_PART_ROWS,
    parts_per_chunk: int = MAX_PARTS_PER_CHUNK,
) -> tuple[ParquetDeliveryPartPlan, ...]:
    """Plan deterministic split-homogeneous files without loading optional dependencies."""

    _require_integer(part_rows, "part_rows", minimum=1, maximum=MAX_PART_ROWS)
    _require_integer(
        parts_per_chunk,
        "parts_per_chunk",
        minimum=1,
        maximum=MAX_PARTS_PER_CHUNK,
    )
    seen: set[str] = set()
    for split, sample_ids in sample_ids_by_split.items():
        _require_text(split, "split")
        if not sample_ids:
            raise ParquetDeliveryError(f"split {split!r} contains no sample IDs")
        if len(set(sample_ids)) != len(sample_ids):
            raise ParquetDeliveryError(f"split {split!r} contains duplicate sample IDs")
        overlap = seen.intersection(sample_ids)
        if overlap:
            raise ParquetDeliveryError(
                f"sample IDs occur in multiple splits: {sorted(overlap)[:5]}"
            )
        seen.update(sample_ids)
    return _layout_parts(
        sample_ids_by_split,
        part_rows=part_rows,
        parts_per_chunk=parts_per_chunk,
    )


def _schema_for_plan(plan: ParquetDeliveryPlan) -> Any:
    pa, _ = _require_pyarrow()
    fields = [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("product", pa.string(), nullable=False),
        pa.field("build_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=True),
        pa.field("subject_id", pa.string(), nullable=True),
        pa.field("group_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("source_archive_id", pa.string(), nullable=True),
        pa.field("source_archive_sha256", pa.string(), nullable=True),
        pa.field("source_member", pa.string(), nullable=True),
        pa.field("source_member_crc32", pa.uint32(), nullable=True),
        pa.field("caption", pa.string(), nullable=True),
        pa.field("caption_language", pa.string(), nullable=True),
        pa.field("frame_count", pa.int32(), nullable=False),
        pa.field("fps", pa.float32(), nullable=True),
        pa.field("coordinate_frame", pa.string(), nullable=False),
        pa.field("pose_units", pa.string(), nullable=False),
        pa.field("annotation_config_fingerprint", pa.string(), nullable=True),
        pa.field("source_manifest_sha256", pa.string(), nullable=False),
        pa.field("split_assignment_sha256", pa.string(), nullable=False),
    ]

    def fixed(value_type: Any, size: int) -> Any:
        return pa.list_(value_type, list_size=size)

    if plan.config.product == POSE_RECONSTRUCTION_PRODUCT:
        dims = plan.static_dimensions
        coordinate = fixed(pa.float32(), dims["coordinate_dim"])
        joints = fixed(coordinate, dims["joints"])
        hands = fixed(joints, dims["hands"])
        valid_joints = fixed(pa.bool_(), dims["joints"])
        valid_hands = fixed(valid_joints, dims["hands"])
        elevation = fixed(pa.float32(), dims["elevation_bins"])
        azimuth = fixed(elevation, dims["azimuth_bins"])
        range_bins = fixed(azimuth, dims["range_bins"])
        doppler = fixed(range_bins, dims["doppler_bins"])
        fields.extend(
            (
                pa.field("radar_cube", pa.list_(doppler), nullable=False),
                pa.field("frame_mask", pa.list_(pa.bool_()), nullable=False),
                pa.field("pose_gt", hands, nullable=False),
                pa.field("pose_valid", valid_hands, nullable=False),
            )
        )
    elif plan.config.product == ONLINE_POSE_CONTROL_PRODUCT:
        dims = plan.static_dimensions
        coordinate = fixed(pa.float32(), dims["coordinate_dim"])
        joints = fixed(coordinate, dims["joints"])
        hands = fixed(joints, dims["hands"])
        joint_float = fixed(pa.float32(), dims["joints"])
        hand_float = fixed(joint_float, dims["hands"])
        joint_bool = fixed(pa.bool_(), dims["joints"])
        hand_bool = fixed(joint_bool, dims["hands"])
        joint_gap = fixed(pa.uint16(), dims["joints"])
        hand_gap = fixed(joint_gap, dims["hands"])
        fields.extend(
            (
                pa.field("scale_mode", pa.string(), nullable=False),
                pa.field("scene_placement", pa.string(), nullable=False),
                pa.field("timing_source", pa.string(), nullable=False),
                pa.field("evidence_role", pa.string(), nullable=False),
                pa.field("annotation_build_id", pa.string(), nullable=False),
                pa.field("annotation_producer_commit", pa.string(), nullable=False),
                pa.field("acceptance_report_sha256", pa.string(), nullable=False),
                pa.field("eligibility_manifest_sha256", pa.string(), nullable=False),
                pa.field("pose_sha256", pa.string(), nullable=False),
                pa.field("audit_sha256", pa.string(), nullable=False),
                pa.field("pose", pa.list_(hands), nullable=False),
                pa.field("reliability", pa.list_(hand_float), nullable=False),
                pa.field("xy_valid", pa.list_(hand_bool), nullable=False),
                pa.field("z_valid", pa.list_(hand_bool), nullable=False),
                pa.field("observed", pa.list_(hand_bool), nullable=False),
                pa.field("valid", pa.list_(hand_bool), nullable=False),
                pa.field("fitted", pa.list_(hand_bool), nullable=False),
                pa.field("imputed", pa.list_(hand_bool), nullable=False),
                pa.field("gap_length", pa.list_(hand_gap), nullable=False),
                pa.field("frame_mask", pa.list_(pa.bool_()), nullable=False),
            )
        )
    else:
        dims = plan.static_dimensions
        if plan.input_mode not in TRANSLATION_INPUT_MODES:
            raise ParquetDeliveryError("translation Parquet schema requires a supported input mode")
        coordinate = fixed(pa.float32(), dims["coordinate_dim"])
        joints = fixed(coordinate, dims["joints"])
        hands = fixed(joints, dims["hands"])
        confidence_joints = fixed(pa.float32(), dims["joints"])
        confidence_hands = fixed(confidence_joints, dims["hands"])
        fields.extend(
            (
                pa.field("pose", pa.list_(hands), nullable=False),
                pa.field("pose_confidence", pa.list_(confidence_hands), nullable=False),
                pa.field("frame_mask", pa.list_(pa.bool_()), nullable=False),
            )
        )
        if plan.input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
            feature = fixed(pa.float32(), dims["radar_feature_dim"])
            fields.insert(
                len(fields) - 1,
                pa.field("radar_feature", pa.list_(feature), nullable=False),
            )
    return pa.schema(fields)


def _schema_fingerprint(schema: Any) -> str:
    return _sha256_bytes(schema.remove_metadata().serialize().to_pybytes())


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return _sha256_bytes("".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8"))


def _validate_sha256sums(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ParquetDeliveryError("delivery SHA256SUMS is missing")
    listed: set[PurePosixPath] = set()
    with sums_path.open("r", encoding="ascii") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest, separator, raw_path = raw_line.rstrip("\n").partition("  ")
            if not separator:
                raise ParquetDeliveryError(f"invalid SHA256SUMS line {line_number}")
            expected_digest = _require_sha256(digest, f"SHA256SUMS line {line_number}")
            relative = _relative_path(raw_path, f"SHA256SUMS line {line_number}")
            if relative in listed:
                raise ParquetDeliveryError(f"duplicate SHA256SUMS path: {relative}")
            listed.add(relative)
            path = _resolve_relative(root, relative, f"SHA256SUMS line {line_number}")
            if not path.is_file() or _sha256_file(path) != expected_digest:
                raise ParquetDeliveryError(f"SHA256SUMS mismatch for {relative}")
    actual = {
        PurePosixPath(path.relative_to(root).as_posix())
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != actual:
        raise ParquetDeliveryError(
            "SHA256SUMS coverage mismatch: "
            f"missing={len(actual - listed)}, extra={len(listed - actual)}"
        )


def _read_json(path: Path, location: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParquetDeliveryError(f"unable to read {location}: {error}") from error
    if not isinstance(value, Mapping):
        raise ParquetDeliveryError(f"{location} must be a JSON object")
    return value


def _read_jsonl(path: Path, location: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                value = json.loads(raw_line)
                if not isinstance(value, Mapping):
                    raise ParquetDeliveryError(f"{location}:{line_number} must be a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ParquetDeliveryError(f"unable to read {location}: {error}") from error
    if not rows:
        raise ParquetDeliveryError(f"{location} contains no rows")
    return rows


def _load_delivery_metadata(root: Path, *, allow_incomplete: bool) -> _DeliveryMetadata:
    payload = _read_json(root / "delivery.json", "delivery metadata")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        PARQUET_DELIVERY_SCHEMA,
        LEGACY_PARQUET_DELIVERY_SCHEMA,
    }:
        raise ParquetDeliveryError("unsupported delivery schema version")
    status = _require_text(payload.get("status"), "delivery.status")
    if status != "complete" and not allow_incomplete:
        raise ParquetDeliveryError(f"delivery is not complete: {status}")
    product = _require_text(payload.get("product"), "delivery.product")
    if product not in SUPPORTED_PRODUCTS:
        raise ParquetDeliveryError(f"unsupported delivery product: {product}")
    if payload.get("product_protocol") != PRODUCT_PROTOCOLS[product]:
        raise ParquetDeliveryError("delivery product protocol is inconsistent")
    input_mode_value = payload.get("input_mode")
    if product == SIGN_LANGUAGE_TRANSLATION_PRODUCT:
        input_mode = _require_text(input_mode_value, "delivery.input_mode")
        if input_mode not in TRANSLATION_INPUT_MODES:
            raise ParquetDeliveryError("delivery.input_mode is unsupported")
    elif input_mode_value is not None:
        raise ParquetDeliveryError("non-translation delivery must not declare input_mode")
    else:
        input_mode = None
    raw_pose_units = payload.get("pose_units")
    pose_units = (
        "model_camera_pseudo_units"
        if raw_pose_units is None and product == ONLINE_POSE_CONTROL_PRODUCT
        else "m"
        if raw_pose_units is None
        else _require_text(raw_pose_units, "delivery.pose_units")
    )
    if product == SIGN_LANGUAGE_TRANSLATION_PRODUCT and pose_units not in TRANSLATION_POSE_UNITS:
        raise ParquetDeliveryError("delivery.pose_units is unsupported for translation")
    expected_pose_units = (
        "model_camera_pseudo_units" if product == ONLINE_POSE_CONTROL_PRODUCT else "m"
    )
    if product != SIGN_LANGUAGE_TRANSLATION_PRODUCT and pose_units != expected_pose_units:
        raise ParquetDeliveryError("delivery.pose_units does not match the product")
    build_id = _require_text(payload.get("build_id"), "delivery.build_id")
    if not _BUILD_ID_PATTERN.fullmatch(build_id):
        raise ParquetDeliveryError("delivery.build_id has invalid format")
    policy = payload.get("row_policy")
    if not isinstance(policy, Mapping):
        raise ParquetDeliveryError("delivery.row_policy must be an object")
    part_rows = _require_integer(
        policy.get("maximum_rows_per_part"),
        "delivery.row_policy.maximum_rows_per_part",
        minimum=1,
        maximum=MAX_PART_ROWS,
    )
    parts_per_chunk = _require_integer(
        policy.get("maximum_parts_per_chunk"),
        "delivery.row_policy.maximum_parts_per_chunk",
        minimum=1,
        maximum=MAX_PARTS_PER_CHUNK,
    )
    if schema_version == PARQUET_DELIVERY_SCHEMA:
        _reject_unknown(
            policy,
            {
                "maximum_rows_per_part",
                "maximum_parts_per_chunk",
                "sample_layout",
                "row_group_layout",
                "rows_per_row_group",
            },
            "delivery.row_policy",
        )
        if policy.get("sample_layout") != "one_sample_per_row_v1":
            raise ParquetDeliveryError("delivery must use exactly one sample per row")
        if policy.get("row_group_layout") != PARQUET_ONE_SAMPLE_PER_ROW_GROUP:
            raise ParquetDeliveryError("delivery must use one row group per sample")
        if (
            _require_integer(
                policy.get("rows_per_row_group"),
                "delivery.row_policy.rows_per_row_group",
                minimum=1,
            )
            != 1
        ):
            raise ParquetDeliveryError("delivery row groups must contain exactly one row")
        row_group_layout = PARQUET_ONE_SAMPLE_PER_ROW_GROUP
    else:
        _reject_unknown(
            policy,
            {
                "maximum_rows_per_part",
                "maximum_parts_per_chunk",
                "row_groups_per_part",
            },
            "delivery.row_policy",
        )
        if (
            _require_integer(
                policy.get("row_groups_per_part"),
                "delivery.row_policy.row_groups_per_part",
                minimum=1,
            )
            != 1
        ):
            raise ParquetDeliveryError("legacy delivery must use exactly one row group per part")
        row_group_layout = PARQUET_LEGACY_PART_ROW_GROUP
    if payload.get("compression") != "zstd":
        raise ParquetDeliveryError("delivery compression must be zstd")
    writer = payload.get("writer")
    if not isinstance(writer, Mapping) or not isinstance(writer.get("pyarrow"), str):
        raise ParquetDeliveryError("delivery writer metadata must include a PyArrow version")
    dimensions = payload.get("static_dimensions")
    if not isinstance(dimensions, Mapping):
        raise ParquetDeliveryError("delivery.static_dimensions must be an object")
    static_dimensions = {
        _require_text(key, "delivery.static_dimensions key"): _require_integer(
            value, f"delivery.static_dimensions.{key}", minimum=1
        )
        for key, value in dimensions.items()
    }
    if product == SIGN_LANGUAGE_TRANSLATION_PRODUCT:
        required_dimensions = {"hands", "joints", "coordinate_dim"}
        if input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
            required_dimensions.add("radar_feature_dim")
        if set(static_dimensions) != required_dimensions:
            raise ParquetDeliveryError(
                "translation delivery static dimensions do not match input mode"
            )
    elif product == ONLINE_POSE_CONTROL_PRODUCT and set(static_dimensions) != {
        "hands",
        "joints",
        "coordinate_dim",
    }:
        raise ParquetDeliveryError("online-pose delivery static dimensions are invalid")
    build = payload.get("build")
    if not isinstance(build, Mapping):
        raise ParquetDeliveryError("delivery.build must be an object")
    created_at = build.get("created_at")
    if not isinstance(created_at, str):
        raise ParquetDeliveryError("delivery.build.created_at must be an RFC 3339 timestamp")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ParquetDeliveryError(
            "delivery.build.created_at must be an RFC 3339 timestamp"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ParquetDeliveryError("delivery.build.created_at must include a timezone")
    git_commit = _require_git_commit(build.get("git_commit"), "delivery.build.git_commit")
    runtime_environment = build.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        raise ParquetDeliveryError("delivery.build.runtime_environment must be an object")
    normalized_runtime = _portable_runtime_report(dict(runtime_environment))
    if dict(runtime_environment) != normalized_runtime:
        raise ParquetDeliveryError("delivery.build.runtime_environment is not portable")
    runtime_git = normalized_runtime["git"]
    if not isinstance(runtime_git, Mapping):
        raise ParquetDeliveryError("delivery.build runtime Git provenance is invalid")
    if build.get("git") != runtime_git or git_commit != runtime_git["commit"]:
        raise ParquetDeliveryError("delivery.build Git provenance is inconsistent")
    resolved_config = build.get("resolved_delivery_config")
    if not isinstance(resolved_config, Mapping):
        raise ParquetDeliveryError("delivery.build.resolved_delivery_config must be an object")
    source_scope = _require_text(payload.get("source_scope"), "delivery.source_scope")
    if source_scope not in {"partial", "complete"}:
        raise ParquetDeliveryError("delivery.source_scope must be partial or complete")
    minimum_free_bytes = _require_integer(
        resolved_config.get("minimum_free_bytes"),
        "delivery.build.resolved_delivery_config.minimum_free_bytes",
        minimum=0,
    )
    verify_source_checksums = resolved_config.get("verify_source_checksums")
    if not isinstance(verify_source_checksums, bool):
        raise ParquetDeliveryError(
            "delivery.build.resolved_delivery_config.verify_source_checksums must be boolean"
        )
    expected_config = {
        "schema_version": PARQUET_DELIVERY_CONFIG_SCHEMA,
        "product": product,
        "expected_source_manifest_sha256": payload.get("source_manifest_sha256"),
        "expected_split_assignment_sha256": payload.get("split_assignment_sha256"),
        "source_scope": source_scope,
        "part_rows": part_rows,
        "parts_per_chunk": parts_per_chunk,
        "minimum_free_bytes": minimum_free_bytes,
        "verify_source_checksums": verify_source_checksums,
    }
    if dict(resolved_config) != expected_config:
        raise ParquetDeliveryError("delivery.build resolved config is inconsistent")
    config_fingerprint = _require_sha256(
        build.get("config_fingerprint"), "delivery.build.config_fingerprint"
    )
    if config_fingerprint != _sha256_bytes(_canonical_json_bytes(expected_config)):
        raise ParquetDeliveryError("delivery.build config fingerprint mismatch")
    if build.get("randomness") != "none_deterministic_placement_v1":
        raise ParquetDeliveryError("delivery.build randomness policy is unsupported")
    return _DeliveryMetadata(
        root=root,
        schema_version=schema_version,
        product=product,
        build_id=build_id,
        source_manifest_sha256=_require_sha256(
            payload.get("source_manifest_sha256"), "delivery.source_manifest_sha256"
        ),
        split_assignment_sha256=_require_sha256(
            payload.get("split_assignment_sha256"),
            "delivery.split_assignment_sha256",
        ),
        part_rows=part_rows,
        parts_per_chunk=parts_per_chunk,
        row_group_layout=row_group_layout,
        input_mode=input_mode,
        coordinate_frame=_require_text(
            payload.get("coordinate_frame"), "delivery.coordinate_frame"
        ),
        pose_units=pose_units,
        static_dimensions=static_dimensions,
        delivery=dict(payload),
    )


def _schema_for_metadata(metadata: _DeliveryMetadata) -> Any:
    config = ParquetDeliveryConfig(
        product=metadata.product,
        data_root=Path("."),
        source_manifest_path=Path("source_manifest.jsonl"),
        split_assignment_path=Path("split_assignments.jsonl"),
        processed_root=metadata.root,
        expected_source_manifest_sha256=metadata.source_manifest_sha256,
        expected_split_assignment_sha256=metadata.split_assignment_sha256,
        source_scope=cast(
            Literal["partial", "complete"],
            _require_text(metadata.delivery.get("source_scope"), "delivery.source_scope"),
        ),
        part_rows=metadata.part_rows,
        parts_per_chunk=metadata.parts_per_chunk,
    )
    plan = ParquetDeliveryPlan(
        config=config,
        build_id=metadata.build_id,
        git_commit="0" * 40,
        runtime_report={
            "python": None,
            "platform": None,
            "packages": {},
            "git": {"commit": "0" * 40, "dirty": False},
        },
        source_manifest_sha256=metadata.source_manifest_sha256,
        split_assignment_sha256=metadata.split_assignment_sha256,
        records_by_id={},
        assignments_by_id={},
        parts=(),
        datasets=(),
        input_mode=metadata.input_mode,
        static_dimensions=metadata.static_dimensions,
        coordinate_frame=metadata.coordinate_frame,
        pose_units=metadata.pose_units,
        estimated_payload_bytes=0,
        estimated_staging_bytes=0,
        required_free_bytes=0,
    )
    return _schema_for_plan(plan)


def _validate_schema(root: Path, metadata: _DeliveryMetadata) -> tuple[Any, str]:
    schema_payload = _read_json(root / "schema.json", "delivery schema")
    expected_schema_file = (
        PARQUET_DELIVERY_SCHEMA_FILE
        if metadata.schema_version == PARQUET_DELIVERY_SCHEMA
        else LEGACY_PARQUET_DELIVERY_SCHEMA_FILE
    )
    if schema_payload.get("schema_version") != expected_schema_file:
        raise ParquetDeliveryError("unsupported delivery schema file version")
    if schema_payload.get("product") != metadata.product:
        raise ParquetDeliveryError("schema product does not match delivery product")
    if schema_payload.get("product_protocol") != PRODUCT_PROTOCOLS[metadata.product]:
        raise ParquetDeliveryError("schema product protocol does not match delivery product")
    if schema_payload.get("input_mode") != metadata.input_mode:
        raise ParquetDeliveryError("schema input mode does not match delivery product")
    schema_pose_units = schema_payload.get("pose_units")
    if schema_pose_units is not None and schema_pose_units != metadata.pose_units:
        raise ParquetDeliveryError("schema pose units do not match delivery product")
    expected_schema = _schema_for_metadata(metadata)
    fingerprint = _schema_fingerprint(expected_schema)
    if schema_payload.get("schema_fingerprint") != fingerprint:
        raise ParquetDeliveryError("schema fingerprint mismatch")
    if metadata.delivery.get("schema_fingerprint") != fingerprint:
        raise ParquetDeliveryError("delivery schema fingerprint mismatch")
    if schema_payload.get("arrow_schema") != str(expected_schema.remove_metadata()):
        raise ParquetDeliveryError("schema Arrow definition mismatch")
    return expected_schema, fingerprint


def _validate_frozen_inputs(
    root: Path,
    metadata: _DeliveryMetadata,
    *,
    verify_checksums: bool,
) -> tuple[dict[str, SampleRecord], dict[str, SplitAssignment]]:
    inputs = metadata.delivery.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ParquetDeliveryError("delivery.inputs must be an object")
    source_input = inputs.get("source_manifest")
    split_input = inputs.get("split_assignments")
    if not isinstance(source_input, Mapping) or not isinstance(split_input, Mapping):
        raise ParquetDeliveryError(
            "delivery inputs must include source_manifest and split_assignments"
        )

    source_relative = _relative_path(source_input.get("path"), "inputs.source_manifest.path")
    split_relative = _relative_path(split_input.get("path"), "inputs.split_assignments.path")
    source_path = _resolve_relative(root, source_relative, "inputs.source_manifest.path")
    split_path = _resolve_relative(root, split_relative, "inputs.split_assignments.path")
    if verify_checksums:
        if _sha256_file(source_path) != metadata.source_manifest_sha256:
            raise ParquetDeliveryError("copied source manifest SHA-256 mismatch")
        if _sha256_file(split_path) != metadata.split_assignment_sha256:
            raise ParquetDeliveryError("copied split assignment SHA-256 mismatch")
    records = _read_source_records(source_path)
    assignments = _read_assignments(split_path, set(records))
    if metadata.product == SIGN_LANGUAGE_TRANSLATION_PRODUCT:
        if metadata.input_mode not in TRANSLATION_INPUT_MODES:
            raise ParquetDeliveryError("translation delivery input mode is unsupported")
        required, forbidden = translation_input_mode_modalities(metadata.input_mode)
        for record in records.values():
            acquisition = record.acquisition or {}
            if acquisition.get("sample_protocol") != SIGN_LANGUAGE_TRANSLATION_SAMPLE_PROTOCOL:
                raise ParquetDeliveryError(
                    f"translation source protocol mismatch for {record.sample_id}"
                )
            if acquisition.get("input_mode") != metadata.input_mode:
                raise ParquetDeliveryError(
                    f"translation source input mode mismatch for {record.sample_id}"
                )
            if acquisition.get("pose_units") != metadata.pose_units:
                raise ParquetDeliveryError(
                    f"translation source pose units mismatch for {record.sample_id}"
                )
            modalities = set(record.modalities)
            if set(required) - modalities or set(forbidden) & modalities:
                raise ParquetDeliveryError(
                    f"translation source modalities do not match input mode for {record.sample_id}"
                )
    elif metadata.product == ONLINE_POSE_CONTROL_PRODUCT:
        required_modalities = {
            ONLINE_POSE_MODALITY,
            ONLINE_POSE_AUDIT_MODALITY,
            ONLINE_POSE_CAPTION_MODALITY,
        }
        for record in records.values():
            if set(record.modalities) != required_modalities:
                raise ParquetDeliveryError(
                    f"online-pose source modalities mismatch for {record.sample_id}"
                )
    return records, assignments


def _validate_inventory_row(
    row: Mapping[str, object],
    *,
    root: Path,
    metadata: _DeliveryMetadata,
    expected_schema: Any,
    schema_fingerprint: str,
    verify_checksums: bool,
) -> tuple[PurePosixPath, list[dict[str, object]]]:
    if row.get("schema_version") != PARQUET_DELIVERY_INVENTORY_SCHEMA:
        raise ParquetDeliveryError("unsupported delivery inventory row schema")
    split = _require_text(row.get("split"), "inventory.split")
    chunk_index = _require_integer(row.get("chunk_index"), "inventory.chunk_index", minimum=0)
    part_index = _require_integer(row.get("part_index"), "inventory.part_index", minimum=0)
    relative = _relative_path(row.get("path"), "inventory.path")
    expected_relative = PurePosixPath(
        "splits", split, f"chunk-{chunk_index:05d}", f"part-{part_index:05d}.parquet"
    )
    if relative != expected_relative:
        raise ParquetDeliveryError("inventory part path does not match split/chunk/part identity")
    path = _resolve_relative(root, relative, "inventory.path")
    if not path.is_file():
        raise ParquetDeliveryError(f"inventory Parquet file is missing: {relative}")
    row_count = _require_integer(
        row.get("row_count"), "inventory.row_count", minimum=1, maximum=metadata.part_rows
    )
    declared_row_group_count = _require_integer(
        row.get("row_group_count"), "inventory.row_group_count", minimum=1
    )
    declared_bytes = _require_integer(row.get("byte_count"), "inventory.byte_count", minimum=1)
    if declared_bytes != path.stat().st_size:
        raise ParquetDeliveryError(f"inventory byte count mismatch for {relative}")
    if verify_checksums and (
        _require_sha256(row.get("sha256"), "inventory.sha256") != _sha256_file(path)
    ):
        raise ParquetDeliveryError(f"inventory SHA-256 mismatch for {relative}")
    if row.get("schema_fingerprint") != schema_fingerprint:
        raise ParquetDeliveryError(f"inventory schema fingerprint mismatch for {relative}")

    _, pq = _require_pyarrow()
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != row_count:
        raise ParquetDeliveryError(f"Parquet row count mismatch for {relative}")
    expected_row_groups = (
        row_count if metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP else 1
    )
    if parquet.metadata.num_row_groups != expected_row_groups:
        raise ParquetDeliveryError(
            f"Parquet row-group layout mismatch for {relative}: expected "
            f"{expected_row_groups}, got {parquet.metadata.num_row_groups}"
        )
    if declared_row_group_count != parquet.metadata.num_row_groups:
        raise ParquetDeliveryError(f"inventory row-group count mismatch for {relative}")
    if metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP and any(
        parquet.metadata.row_group(index).num_rows != 1
        for index in range(parquet.metadata.num_row_groups)
    ):
        raise ParquetDeliveryError(
            f"Parquet row groups must each contain one sample row: {relative}"
        )
    if not parquet.schema_arrow.remove_metadata().equals(expected_schema.remove_metadata()):
        raise ParquetDeliveryError(f"Parquet schema mismatch for {relative}")
    table = parquet.read()
    rows = cast(list[dict[str, object]], table.to_pylist())
    sample_ids = tuple(_require_text(item.get("sample_id"), "Parquet sample_id") for item in rows)
    if row.get("sample_id_digest") != _sample_id_digest(sample_ids):
        raise ParquetDeliveryError(f"inventory sample-ID digest mismatch for {relative}")
    return relative, rows


def _validate_payload_row(
    row: Mapping[str, object], metadata: _DeliveryMetadata, sample_id: str
) -> None:
    frame_count = _require_integer(row.get("frame_count"), "Parquet frame_count", minimum=1)
    if metadata.product == POSE_RECONSTRUCTION_PRODUCT:
        dims = metadata.static_dimensions
        try:
            radar_cube = np.asarray(row["radar_cube"], dtype=np.float32)
            frame_mask = np.asarray(row["frame_mask"], dtype=np.bool_)
            pose = np.asarray(row["pose_gt"], dtype=np.float32)
            pose_valid = np.asarray(row["pose_valid"], dtype=np.bool_)
        except (KeyError, TypeError, ValueError) as error:
            raise ParquetDeliveryError(f"invalid pose payload for {sample_id}") from error
        expected_cube = (
            frame_count,
            dims["doppler_bins"],
            dims["range_bins"],
            dims["azimuth_bins"],
            dims["elevation_bins"],
        )
        if radar_cube.shape != expected_cube or frame_mask.shape != (frame_count,):
            raise ParquetDeliveryError(f"pose payload shape mismatch for {sample_id}")
        if pose.shape != (2, 24, 3) or pose_valid.shape != (2, 24):
            raise ParquetDeliveryError(f"pose target shape mismatch for {sample_id}")
        if not bool(np.all(np.isfinite(radar_cube))) or bool(np.any(radar_cube < 0)):
            raise ParquetDeliveryError(f"pose cube contract mismatch for {sample_id}")
        if not bool(np.all(np.isfinite(pose))) or not bool(np.any(frame_mask)):
            raise ParquetDeliveryError(f"pose validity contract mismatch for {sample_id}")
        if not bool(np.any(pose_valid)):
            raise ParquetDeliveryError(f"pose target mask is empty for {sample_id}")
        return

    if metadata.product == ONLINE_POSE_CONTROL_PRODUCT:
        _online_pose_sample_from_row(row, metadata, sample_id)
        return

    dims = metadata.static_dimensions
    _require_text(row.get("caption"), f"Parquet caption for {sample_id}")
    try:
        pose = np.asarray(row["pose"], dtype=np.float32)
        confidence = np.asarray(row["pose_confidence"], dtype=np.float32)
        frame_mask = np.asarray(row["frame_mask"], dtype=np.bool_)
    except (KeyError, TypeError, ValueError) as error:
        raise ParquetDeliveryError(f"invalid translation payload for {sample_id}") from error
    radar_feature: np.ndarray | None = None
    if metadata.input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
        try:
            radar_feature = np.asarray(row["radar_feature"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ParquetDeliveryError(
                f"invalid feature translation payload for {sample_id}"
            ) from error
    elif metadata.input_mode == POSE_ONLY_INPUT_MODE:
        if "radar_feature" in row:
            raise ParquetDeliveryError(
                f"pose_only translation payload has a radar feature for {sample_id}"
            )
    else:
        raise ParquetDeliveryError("translation delivery input mode is unsupported")
    if pose.shape != (frame_count, 2, dims["joints"], dims["coordinate_dim"]):
        raise ParquetDeliveryError(f"translation pose shape mismatch for {sample_id}")
    if confidence.shape != (frame_count, 2, dims["joints"]):
        raise ParquetDeliveryError(f"translation confidence shape mismatch for {sample_id}")
    if radar_feature is not None and radar_feature.shape != (
        frame_count,
        dims["radar_feature_dim"],
    ):
        raise ParquetDeliveryError(f"translation radar feature shape mismatch for {sample_id}")
    if frame_mask.shape != (frame_count,) or not bool(np.any(frame_mask)):
        raise ParquetDeliveryError(f"translation frame mask mismatch for {sample_id}")
    if not all(
        bool(np.all(np.isfinite(value)))
        for value in (pose, confidence, radar_feature)
        if value is not None
    ):
        raise ParquetDeliveryError(f"translation payload must be finite for {sample_id}")
    if bool(np.any((confidence < 0) | (confidence > 1))):
        raise ParquetDeliveryError(f"translation confidence must be within [0,1] for {sample_id}")


def _online_pose_sample_from_row(
    row: Mapping[str, object], metadata: _DeliveryMetadata, sample_id: str
) -> OnlinePoseControlSample:
    try:
        arrays = {
            "pose": np.asarray(row["pose"], dtype=np.float32),
            "reliability": np.asarray(row["reliability"], dtype=np.float32),
            "xy_valid": np.asarray(row["xy_valid"], dtype=np.bool_),
            "z_valid": np.asarray(row["z_valid"], dtype=np.bool_),
            "observed": np.asarray(row["observed"], dtype=np.bool_),
            "valid": np.asarray(row["valid"], dtype=np.bool_),
            "fitted": np.asarray(row["fitted"], dtype=np.bool_),
            "imputed": np.asarray(row["imputed"], dtype=np.bool_),
            "gap_length": np.asarray(row["gap_length"], dtype=np.uint16),
            "frame_mask": np.asarray(row["frame_mask"], dtype=np.bool_),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ParquetDeliveryError(f"invalid online-pose payload for {sample_id}") from error
    return _online_pose_sample_from_values(row, arrays, metadata, sample_id)


def _online_pose_sample_from_values(
    row: Mapping[str, object],
    arrays: dict[str, np.ndarray],
    metadata: _DeliveryMetadata,
    sample_id: str,
) -> OnlinePoseControlSample:
    frames = _require_integer(row.get("frame_count"), "Parquet frame_count", minimum=1)
    if arrays["pose"].ndim < 1 or arrays["pose"].shape[0] != frames:
        raise ParquetDeliveryError(f"online-pose frame count mismatch for {sample_id}")
    try:
        validate_online_pose_arrays(sample_id=sample_id, **arrays)
    except OnlinePoseControlDataError as error:
        raise ParquetDeliveryError(
            f"invalid online-pose payload for {sample_id}: {error}"
        ) from error
    evidence_role = _require_text(row.get("evidence_role"), "Parquet evidence_role")
    if evidence_role != "synthetic_csl_daily_control":
        raise ParquetDeliveryError(f"online-pose evidence role mismatch for {sample_id}")
    return OnlinePoseControlSample(
        sample_id=sample_id,
        caption=_require_text(row.get("caption"), f"Parquet caption for {sample_id}"),
        coordinate_frame=metadata.coordinate_frame,
        scale_mode=_require_text(row.get("scale_mode"), "Parquet scale_mode"),
        scene_placement=_require_text(row.get("scene_placement"), "Parquet scene_placement"),
        timing_source=_require_text(row.get("timing_source"), "Parquet timing_source"),
        evidence_role=evidence_role,
        annotation_build_id=_require_text(
            row.get("annotation_build_id"), "Parquet annotation_build_id"
        ),
        annotation_producer_commit=_require_git_commit(
            row.get("annotation_producer_commit"), "Parquet annotation_producer_commit"
        ),
        acceptance_report_sha256=_require_sha256(
            row.get("acceptance_report_sha256"), "Parquet acceptance_report_sha256"
        ),
        eligibility_manifest_sha256=_require_sha256(
            row.get("eligibility_manifest_sha256"), "Parquet eligibility_manifest_sha256"
        ),
        pose_sha256=_require_sha256(row.get("pose_sha256"), "Parquet pose_sha256"),
        audit_sha256=_require_sha256(row.get("audit_sha256"), "Parquet audit_sha256"),
        **arrays,
    )


def _validate_online_pose_row_binding(row: Mapping[str, object], record: SampleRecord) -> None:
    pose = record.modalities[ONLINE_POSE_MODALITY]
    audit = record.modalities[ONLINE_POSE_AUDIT_MODALITY]
    caption = record.modalities[ONLINE_POSE_CAPTION_MODALITY]
    provenance = record.provenance or {}
    acquisition = record.acquisition or {}
    expected = {
        "caption": caption.text,
        "frame_count": acquisition.get("frame_count"),
        "scale_mode": provenance.get("scale_mode"),
        "scene_placement": provenance.get("scene_placement"),
        "timing_source": provenance.get("timing_source"),
        "evidence_role": provenance.get("evidence_role"),
        "annotation_build_id": provenance.get("annotation_build_id"),
        "annotation_producer_commit": provenance.get("annotation_producer_commit"),
        "annotation_config_fingerprint": provenance.get("annotation_config_fingerprint"),
        "acceptance_report_sha256": provenance.get("acceptance_report_sha256"),
        "eligibility_manifest_sha256": provenance.get("eligibility_manifest_sha256"),
        "pose_sha256": pose.sha256,
        "audit_sha256": audit.sha256,
    }
    mismatched = sorted(key for key, value in expected.items() if row.get(key) != value)
    if mismatched:
        raise ParquetDeliveryError(
            f"online-pose row provenance mismatch for {record.sample_id}: {', '.join(mismatched)}"
        )


def _validate_index_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    root: Path,
    metadata: _DeliveryMetadata,
    observed_rows: Mapping[str, tuple[PurePosixPath, int, int, str, str]],
) -> None:
    if len(rows) != len(observed_rows):
        raise ParquetDeliveryError(
            f"sample index count mismatch: expected {len(observed_rows)}, got {len(rows)}"
        )
    seen: set[str] = set()
    for row in rows:
        expected_index_schema = (
            PARQUET_DELIVERY_INDEX_SCHEMA
            if metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP
            else LEGACY_PARQUET_DELIVERY_INDEX_SCHEMA
        )
        if row.get("schema_version") != expected_index_schema:
            raise ParquetDeliveryError("unsupported delivery index row schema")
        sample_id = _require_text(row.get("sample_id"), "index.sample_id")
        if sample_id in seen:
            raise ParquetDeliveryError(f"duplicate sample index ID: {sample_id}")
        seen.add(sample_id)
        group_id = _require_sha256(row.get("group_id"), "index.group_id")
        split = _require_text(row.get("split"), "index.split")
        relative = _relative_path(row.get("path"), "index.path")
        row_index = _require_integer(row.get("row_index"), "index.row_index", minimum=0)
        row_group_index = (
            _require_integer(row.get("row_group_index"), "index.row_group_index", minimum=0)
            if metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP
            else 0
        )
        if (
            metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP
            and row_group_index != row_index
        ):
            raise ParquetDeliveryError(
                f"index row/row-group identity mismatch for sample {sample_id}"
            )
        observed = observed_rows.get(sample_id)
        if observed is None:
            raise ParquetDeliveryError(f"index references unknown sample {sample_id}")
        if (relative, row_index, row_group_index, split, group_id) != observed:
            raise ParquetDeliveryError(f"index placement mismatch for sample {sample_id}")
        _resolve_relative(root, relative, "index.path")
    if seen != set(observed_rows):
        raise ParquetDeliveryError("sample index coverage mismatch")


def validate_parquet_delivery(
    root: str | Path,
    *,
    allow_incomplete: bool = False,
    verify_checksums: bool = True,
) -> ParquetDeliveryValidation:
    """Validate a completed delivery and all copied input, index, and part identities."""

    delivery_root = Path(root).expanduser().resolve()
    if not delivery_root.is_dir():
        raise ParquetDeliveryError(f"delivery root does not exist: {delivery_root}")
    metadata = _load_delivery_metadata(delivery_root, allow_incomplete=allow_incomplete)
    expected_schema, schema_fingerprint = _validate_schema(delivery_root, metadata)
    records, assignments = _validate_frozen_inputs(
        delivery_root,
        metadata,
        verify_checksums=verify_checksums,
    )
    inventory_relative = _relative_path(
        metadata.delivery.get("inventory_path"), "delivery.inventory_path"
    )
    index_relative = _relative_path(metadata.delivery.get("index_path"), "delivery.index_path")
    inventory_path = _resolve_relative(delivery_root, inventory_relative, "delivery.inventory_path")
    index_path = _resolve_relative(delivery_root, index_relative, "delivery.index_path")
    inventory_rows = _read_jsonl(inventory_path, "delivery inventory")
    index_rows = _read_jsonl(index_path, "delivery sample index")

    observed: dict[str, tuple[PurePosixPath, int, int, str, str]] = {}
    sample_ids_by_split: dict[str, list[str]] = defaultdict(list)
    expected_parts: list[ParquetDeliveryPartPlan] = []
    inventory_paths: set[PurePosixPath] = set()
    for inventory_row in inventory_rows:
        relative, parquet_rows = _validate_inventory_row(
            inventory_row,
            root=delivery_root,
            metadata=metadata,
            expected_schema=expected_schema,
            schema_fingerprint=schema_fingerprint,
            verify_checksums=verify_checksums,
        )
        if relative in inventory_paths:
            raise ParquetDeliveryError(f"duplicate inventory part path: {relative}")
        inventory_paths.add(relative)
        split = _require_text(inventory_row.get("split"), "inventory.split")
        expected_parts.append(
            ParquetDeliveryPartPlan(
                split=split,
                chunk_index=cast(int, inventory_row["chunk_index"]),
                part_index=cast(int, inventory_row["part_index"]),
                sample_ids=tuple(
                    _require_text(row.get("sample_id"), "Parquet sample_id") for row in parquet_rows
                ),
            )
        )
        for row_index, row in enumerate(parquet_rows):
            sample_id = _require_text(row.get("sample_id"), "Parquet sample_id")
            if sample_id in observed:
                raise ParquetDeliveryError(f"duplicate Parquet sample ID: {sample_id}")
            assignment = assignments.get(sample_id)
            record = records.get(sample_id)
            if assignment is None or record is None:
                raise ParquetDeliveryError(f"Parquet sample has no frozen input: {sample_id}")
            if row.get("schema_version") != metadata.schema_version:
                raise ParquetDeliveryError(f"Parquet row schema mismatch for {sample_id}")
            if row.get("product") != metadata.product or row.get("build_id") != metadata.build_id:
                raise ParquetDeliveryError(f"Parquet row product/build mismatch for {sample_id}")
            if row.get("split") != assignment.split or row.get("group_id") != assignment.group_id:
                raise ParquetDeliveryError(f"Parquet row split/group mismatch for {sample_id}")
            if row.get("dataset_id") != record.dataset:
                raise ParquetDeliveryError(f"Parquet row dataset mismatch for {sample_id}")
            if row.get("source_manifest_sha256") != metadata.source_manifest_sha256:
                raise ParquetDeliveryError(f"Parquet source manifest mismatch for {sample_id}")
            if row.get("split_assignment_sha256") != metadata.split_assignment_sha256:
                raise ParquetDeliveryError(f"Parquet split assignment mismatch for {sample_id}")
            if (
                row.get("coordinate_frame") != metadata.coordinate_frame
                or row.get("pose_units") != metadata.pose_units
            ):
                raise ParquetDeliveryError(f"Parquet coordinate contract mismatch for {sample_id}")
            _validate_payload_row(row, metadata, sample_id)
            if metadata.product == ONLINE_POSE_CONTROL_PRODUCT:
                _validate_online_pose_row_binding(row, record)
            row_group_index = (
                row_index if metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP else 0
            )
            observed[sample_id] = (
                relative,
                row_index,
                row_group_index,
                assignment.split,
                assignment.group_id,
            )
            sample_ids_by_split[assignment.split].append(sample_id)

    splits_root = delivery_root / "splits"
    actual_part_paths = (
        {
            PurePosixPath(path.relative_to(delivery_root).as_posix())
            for path in splits_root.rglob("*.parquet")
        }
        if splits_root.is_dir()
        else set()
    )
    if inventory_paths != actual_part_paths:
        raise ParquetDeliveryError(
            "Parquet part inventory coverage mismatch: "
            f"missing={len(actual_part_paths - inventory_paths)}, "
            f"extra={len(inventory_paths - actual_part_paths)}"
        )

    if set(observed) != set(records):
        missing = sorted(set(records) - set(observed))
        extra = sorted(set(observed) - set(records))
        raise ParquetDeliveryError(
            f"Parquet sample coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    canonical_layout = plan_parquet_layout(
        sample_ids_by_split,
        part_rows=metadata.part_rows,
        parts_per_chunk=metadata.parts_per_chunk,
    )
    if len(canonical_layout) != len(expected_parts):
        raise ParquetDeliveryError("Parquet inventory part count differs from canonical layout")
    for expected, actual in zip(canonical_layout, expected_parts, strict=True):
        if (
            expected.split,
            expected.chunk_index,
            expected.part_index,
            expected.sample_ids,
        ) != (
            actual.split,
            actual.chunk_index,
            actual.part_index,
            actual.sample_ids,
        ):
            raise ParquetDeliveryError("Parquet part placement is not deterministic")
    _validate_index_rows(
        index_rows,
        root=delivery_root,
        metadata=metadata,
        observed_rows=observed,
    )

    split_counts = {
        split: len(sample_ids) for split, sample_ids in sorted(sample_ids_by_split.items())
    }
    declared_count = _require_integer(
        metadata.delivery.get("sample_count"), "delivery.sample_count", minimum=1
    )
    if declared_count != len(observed):
        raise ParquetDeliveryError("delivery sample count mismatch")
    if not allow_incomplete:
        validation_payload = _read_json(
            _resolve_relative(
                delivery_root,
                _relative_path(
                    metadata.delivery.get("validation_path"), "delivery.validation_path"
                ),
                "delivery.validation_path",
            ),
            "delivery validation report",
        )
        if validation_payload.get("schema_version") != PARQUET_DELIVERY_VALIDATION_SCHEMA:
            raise ParquetDeliveryError("unsupported delivery validation report schema")
        if validation_payload.get("status") != "passed":
            raise ParquetDeliveryError("delivery validation report is not passed")
    if verify_checksums and not allow_incomplete:
        _validate_sha256sums(delivery_root)
    return ParquetDeliveryValidation(
        root=delivery_root,
        product=metadata.product,
        build_id=metadata.build_id,
        sample_count=len(observed),
        part_count=len(canonical_layout),
        split_counts=split_counts,
        inventory_sha256=_sha256_file(inventory_path),
        index_sha256=_sha256_file(index_path),
        schema_fingerprint=schema_fingerprint,
    )


def load_published_parquet_delivery_validation(
    root: str | Path,
) -> ParquetDeliveryValidation:
    """Bind a published validation receipt without opening Parquet payload parts."""

    delivery_root = Path(root).expanduser().resolve()
    if not delivery_root.is_dir():
        raise ParquetDeliveryError(f"delivery root does not exist: {delivery_root}")
    metadata = _load_delivery_metadata(delivery_root, allow_incomplete=False)
    _, schema_fingerprint = _validate_schema(delivery_root, metadata)
    validation_path = _resolve_relative(
        delivery_root,
        _relative_path(metadata.delivery.get("validation_path"), "delivery.validation_path"),
        "delivery.validation_path",
    )
    payload = _read_json(validation_path, "delivery validation report")
    raw_split_counts = payload.get("split_counts")
    if not isinstance(raw_split_counts, Mapping):
        raise ParquetDeliveryError("published validation split_counts must be an object")
    split_counts = {
        _require_text(name, "validation split name"): _require_integer(
            count, f"validation split_counts.{name}", minimum=0
        )
        for name, count in raw_split_counts.items()
    }
    inventory_path = _resolve_relative(
        delivery_root,
        _relative_path(metadata.delivery.get("inventory_path"), "delivery.inventory_path"),
        "delivery.inventory_path",
    )
    index_path = _resolve_relative(
        delivery_root,
        _relative_path(metadata.delivery.get("index_path"), "delivery.index_path"),
        "delivery.index_path",
    )
    validation = ParquetDeliveryValidation(
        root=delivery_root,
        product=_require_text(payload.get("product"), "validation.product"),
        build_id=_require_text(payload.get("build_id"), "validation.build_id"),
        sample_count=_require_integer(
            payload.get("sample_count"), "validation.sample_count", minimum=1
        ),
        part_count=_require_integer(payload.get("part_count"), "validation.part_count", minimum=1),
        split_counts=split_counts,
        inventory_sha256=_require_sha256(
            payload.get("inventory_sha256"), "validation.inventory_sha256"
        ),
        index_sha256=_require_sha256(payload.get("index_sha256"), "validation.index_sha256"),
        schema_fingerprint=_require_sha256(
            payload.get("schema_fingerprint"), "validation.schema_fingerprint"
        ),
    )
    if payload != validation.to_dict():
        raise ParquetDeliveryError("published delivery validation report is invalid")
    declared_count = _require_integer(
        metadata.delivery.get("sample_count"), "delivery.sample_count", minimum=1
    )
    if (
        validation.product != metadata.product
        or validation.build_id != metadata.build_id
        or validation.sample_count != declared_count
        or validation.schema_fingerprint != schema_fingerprint
        or validation.inventory_sha256 != _sha256_file(inventory_path)
        or validation.index_sha256 != _sha256_file(index_path)
    ):
        raise ParquetDeliveryError(
            "published validation identity does not match delivery metadata/index"
        )
    return validation


class _ParquetDeliveryDataset:
    """Dependency-light random access over one validated delivery split."""

    expected_product: str

    def __init__(
        self,
        delivery_root: str | Path,
        *,
        split: str | None = None,
        indexed_samples: Sequence[ParquetIndexedSample] | None = None,
        verify_checksums: bool = True,
        accepted_validation: ParquetDeliveryValidation | None = None,
    ) -> None:
        if (split is None) == (indexed_samples is None):
            raise ParquetDeliveryError("provide exactly one of split or indexed_samples")
        root = Path(delivery_root).expanduser().resolve()
        metadata = _load_delivery_metadata(root, allow_incomplete=False)
        if metadata.product != self.expected_product:
            raise ParquetDeliveryError(
                f"expected {self.expected_product} delivery, got {metadata.product}"
            )
        validation = (
            (
                load_published_parquet_delivery_validation(root)
                if indexed_samples is not None
                else validate_parquet_delivery(root, verify_checksums=verify_checksums)
            )
            if accepted_validation is None
            else accepted_validation
        )
        if validation.root != root:
            raise ParquetDeliveryError("accepted delivery validation root mismatch")
        validation_report_path = _resolve_relative(
            root,
            _relative_path(metadata.delivery.get("validation_path"), "delivery.validation_path"),
            "delivery.validation_path",
        )
        published_validation = _read_json(validation_report_path, "delivery validation report")
        if published_validation != validation.to_dict():
            raise ParquetDeliveryError("accepted delivery validation report mismatch")
        if validation.product != metadata.product:
            raise ParquetDeliveryError("accepted delivery validation product mismatch")
        if validation.build_id != metadata.build_id:
            raise ParquetDeliveryError("accepted delivery validation build mismatch")
        if validation.sample_count != _require_integer(
            metadata.delivery.get("sample_count"), "delivery.sample_count", minimum=1
        ):
            raise ParquetDeliveryError("accepted delivery validation sample count mismatch")
        schema_payload = _read_json(root / "schema.json", "delivery schema")
        if validation.schema_fingerprint != _require_sha256(
            schema_payload.get("schema_fingerprint"), "schema.schema_fingerprint"
        ):
            raise ParquetDeliveryError("accepted delivery validation schema mismatch")
        inventory_path = _resolve_relative(
            root,
            _relative_path(metadata.delivery.get("inventory_path"), "delivery.inventory_path"),
            "delivery.inventory_path",
        )
        index_path = _resolve_relative(
            root,
            _relative_path(metadata.delivery.get("index_path"), "delivery.index_path"),
            "delivery.index_path",
        )
        if validation.inventory_sha256 != _sha256_file(inventory_path):
            raise ParquetDeliveryError("accepted delivery validation inventory mismatch")
        if validation.index_sha256 != _sha256_file(index_path):
            raise ParquetDeliveryError("accepted delivery validation index mismatch")
        self.delivery_root = root
        self.validation = validation
        self.metadata = metadata
        self.split = split
        self._entries = (
            self._entries_for_indexed_samples(indexed_samples, metadata)
            if indexed_samples is not None
            else self._entries_for_split(root, _require_text(split, "split"), metadata)
        )
        self._sample_ids = tuple(entry.sample_id for entry in self._entries)
        self._sample_locations = tuple(
            ParquetSampleLocation(
                part_path=entry.relative_path.as_posix(),
                row_index=entry.row_index,
                row_group_index=entry.row_group_index,
            )
            for entry in self._entries
        )
        self._cached_part_rows: OrderedDict[PurePosixPath, list[dict[str, object]]] = OrderedDict()
        self._cached_part_readers: OrderedDict[PurePosixPath, Any] = OrderedDict()
        self._sample_access_count = 0
        self._part_load_count = 0
        self._part_cache_hit_count = 0

    @staticmethod
    def _entries_for_split(
        root: Path,
        split: str,
        metadata: _DeliveryMetadata,
    ) -> tuple[_IndexEntry, ...]:
        index_rows = _read_jsonl(root / "indices" / "sample_index.jsonl", "delivery sample index")
        entries: list[_IndexEntry] = []
        for row in index_rows:
            if row.get("split") != split:
                continue
            row_index = _require_integer(row.get("row_index"), "index.row_index", minimum=0)
            if metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP:
                if row.get("schema_version") != PARQUET_DELIVERY_INDEX_SCHEMA:
                    raise ParquetDeliveryError("unsupported delivery index row schema")
                row_group_index = _require_integer(
                    row.get("row_group_index"), "index.row_group_index", minimum=0
                )
                if row_group_index != row_index:
                    raise ParquetDeliveryError("delivery index row and row-group identities differ")
            else:
                if row.get("schema_version") != LEGACY_PARQUET_DELIVERY_INDEX_SCHEMA:
                    raise ParquetDeliveryError("unsupported legacy delivery index row schema")
                row_group_index = None
            entries.append(
                _IndexEntry(
                    sample_id=_require_text(row.get("sample_id"), "index.sample_id"),
                    relative_path=_relative_path(row.get("path"), "index.path"),
                    row_index=row_index,
                    row_group_index=row_group_index,
                    group_id=_require_sha256(row.get("group_id"), "index.group_id"),
                )
            )
        if not entries:
            raise ParquetDeliveryError(f"delivery has no samples for split {split!r}")
        entries.sort(key=lambda entry: entry.sample_id)
        return tuple(entries)

    @staticmethod
    def _entries_for_indexed_samples(
        indexed_samples: Sequence[ParquetIndexedSample],
        metadata: _DeliveryMetadata,
    ) -> tuple[_IndexEntry, ...]:
        if not indexed_samples:
            raise ParquetDeliveryError("indexed_samples must not be empty")
        if metadata.row_group_layout != PARQUET_ONE_SAMPLE_PER_ROW_GROUP:
            raise ParquetDeliveryError("indexed_samples require one sample per row group")

        entries: list[_IndexEntry] = []
        sample_ids: set[str] = set()
        physical_locations: set[tuple[PurePosixPath, int, int]] = set()
        for position, reference in enumerate(indexed_samples):
            location = f"indexed_samples[{position}]"
            try:
                sample_id_value = reference.sample_id
                part_path_value = reference.part_path
                row_index_value = reference.row_index
                row_group_index_value = reference.row_group_index
            except AttributeError as error:
                raise ParquetDeliveryError(
                    f"{location} must provide sample_id, part_path, row_index, and row_group_index"
                ) from error
            sample_id = _require_text(sample_id_value, f"{location}.sample_id")
            relative_path = _relative_path(part_path_value, f"{location}.part_path")
            row_index = _require_integer(row_index_value, f"{location}.row_index", minimum=0)
            row_group_index = _require_integer(
                row_group_index_value,
                f"{location}.row_group_index",
                minimum=0,
            )
            if row_index != row_group_index:
                raise ParquetDeliveryError(f"{location} row_index must equal row_group_index")
            physical_location = (relative_path, row_index, row_group_index)
            if sample_id in sample_ids:
                raise ParquetDeliveryError(f"duplicate indexed sample ID: {sample_id}")
            if physical_location in physical_locations:
                raise ParquetDeliveryError(
                    f"duplicate indexed physical location: {relative_path} "
                    f"row group {row_group_index}"
                )
            sample_ids.add(sample_id)
            physical_locations.add(physical_location)
            entries.append(
                _IndexEntry(
                    sample_id=sample_id,
                    relative_path=relative_path,
                    row_index=row_index,
                    row_group_index=row_group_index,
                )
            )
        return tuple(entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    @property
    def sample_locations(self) -> tuple[ParquetSampleLocation, ...]:
        return self._sample_locations

    @property
    def parquet_io_stats(self) -> dict[str, int]:
        return {
            "sample_access_count": self._sample_access_count,
            "part_load_count": self._part_load_count,
            "part_cache_hit_count": self._part_cache_hit_count,
            "resident_part_count": (
                len(self._cached_part_readers)
                if self.metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP
                else len(self._cached_part_rows)
            ),
        }

    def _entry_for_access(self, index: int) -> _IndexEntry:
        entry = self._entries[index]
        self._sample_access_count += 1
        return entry

    def _row(self, index: int) -> dict[str, object]:
        entry = self._entry_for_access(index)
        if self.metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP:
            return self._single_row_group(entry)
        rows = self._cached_part_rows.get(entry.relative_path)
        if rows is None:
            _, pq = _require_pyarrow()
            path = _resolve_relative(self.delivery_root, entry.relative_path, "index.path")
            rows = cast(list[dict[str, object]], pq.read_table(path).to_pylist())
            self._part_load_count += 1
            self._cached_part_rows[entry.relative_path] = rows
            if len(self._cached_part_rows) > PARQUET_DELIVERY_READER_PART_CACHE_SIZE:
                self._cached_part_rows.popitem(last=False)
        else:
            self._part_cache_hit_count += 1
            self._cached_part_rows.move_to_end(entry.relative_path)
        try:
            row = rows[entry.row_index]
        except IndexError as error:
            raise ParquetDeliveryError(
                f"index row is outside its Parquet part for {entry.sample_id}"
            ) from error
        if row.get("sample_id") != entry.sample_id:
            raise ParquetDeliveryError(f"index sample mismatch for {entry.sample_id}")
        return row

    def _part_reader(self, entry: _IndexEntry) -> Any:
        parquet = self._cached_part_readers.get(entry.relative_path)
        if parquet is None:
            _, pq = _require_pyarrow()
            path = _resolve_relative(self.delivery_root, entry.relative_path, "index.path")
            parquet = pq.ParquetFile(path)
            self._part_load_count += 1
            self._cached_part_readers[entry.relative_path] = parquet
            if len(self._cached_part_readers) > PARQUET_DELIVERY_READER_PART_CACHE_SIZE:
                _, evicted = self._cached_part_readers.popitem(last=False)
                evicted.close()
        else:
            self._part_cache_hit_count += 1
            self._cached_part_readers.move_to_end(entry.relative_path)
        return parquet

    def _single_row_group_table(
        self,
        entry: _IndexEntry,
        *,
        columns: Sequence[str] | None = None,
    ) -> Any:
        parquet = self._part_reader(entry)
        if entry.row_group_index is None:
            raise ParquetDeliveryError(f"index row group is missing for {entry.sample_id}")
        table = (
            parquet.read_row_group(entry.row_group_index)
            if columns is None
            else parquet.read_row_group(entry.row_group_index, columns=columns)
        )
        if table.num_rows != 1:
            raise ParquetDeliveryError(
                f"indexed row group must contain one sample for {entry.sample_id}"
            )
        return table

    def _single_row_group(self, entry: _IndexEntry) -> dict[str, object]:
        rows = cast(
            list[dict[str, object]],
            self._single_row_group_table(entry).to_pylist(),
        )
        row = rows[0]
        if row.get("sample_id") != entry.sample_id:
            raise ParquetDeliveryError(f"index sample mismatch for {entry.sample_id}")
        return row

    def close(self) -> None:
        for parquet in self._cached_part_readers.values():
            parquet.close()
        self._cached_part_readers.clear()
        self._cached_part_rows.clear()


class ParquetSignLanguageTranslationDataset(_ParquetDeliveryDataset):
    """Parquet source exposing the current dependency-light translation sample contract."""

    expected_product = SIGN_LANGUAGE_TRANSLATION_PRODUCT

    def __init__(
        self,
        delivery_root: str | Path,
        *,
        split: str | None = None,
        indexed_samples: Sequence[ParquetIndexedSample] | None = None,
        verify_checksums: bool = True,
        accepted_validation: ParquetDeliveryValidation | None = None,
        consumer_input_mode: str | None = None,
    ) -> None:
        super().__init__(
            delivery_root,
            split=split,
            indexed_samples=indexed_samples,
            verify_checksums=verify_checksums,
            accepted_validation=accepted_validation,
        )
        dims = self.metadata.static_dimensions
        self.joint_count = dims["joints"]
        self.coordinate_dim = dims["coordinate_dim"]
        if self.metadata.input_mode not in TRANSLATION_INPUT_MODES:
            raise ParquetDeliveryError("translation delivery input mode is unsupported")
        self.input_mode = self.metadata.input_mode
        self.consumer_input_mode = consumer_input_mode or self.input_mode
        try:
            supported = translation_delivery_supports_wavellm_input_mode(
                self.input_mode, self.consumer_input_mode
            )
        except ValueError as error:
            raise ParquetDeliveryError(str(error)) from error
        if not supported:
            raise ParquetDeliveryError(
                f"translation delivery {self.input_mode!r} cannot supply WaveLLM mode "
                f"{self.consumer_input_mode!r}"
            )
        self.radar_feature_dim = (
            dims["radar_feature_dim"]
            if self.input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE
            else None
        )
        self.coordinate_frame = self.metadata.coordinate_frame
        self.pose_units = self.metadata.pose_units

    def load_sample(self, index: int) -> SignLanguageTranslationSample:
        columns = [
            "sample_id",
            "caption",
            "pose",
            "pose_confidence",
            "frame_mask",
            "frame_count",
        ]
        if self.consumer_input_mode != POSE_ONLY_INPUT_MODE:
            columns.append("radar_feature")
        if self.metadata.row_group_layout == PARQUET_ONE_SAMPLE_PER_ROW_GROUP:
            entry = self._entry_for_access(index)
            rows = cast(
                list[dict[str, object]],
                self._single_row_group_table(entry, columns=columns).to_pylist(),
            )
            row = rows[0]
            if row.get("sample_id") != entry.sample_id:
                raise ParquetDeliveryError(f"index sample mismatch for {entry.sample_id}")
        else:
            row = self._row(index)
        sample_id = _require_text(row.get("sample_id"), "Parquet sample_id")
        caption = _require_text(row.get("caption"), f"Parquet caption for {sample_id}")
        try:
            pose = np.asarray(row["pose"], dtype=np.float32)
            confidence = np.asarray(row["pose_confidence"], dtype=np.float32)
            frame_mask = np.asarray(row["frame_mask"], dtype=np.bool_)
        except (KeyError, TypeError, ValueError) as error:
            raise ParquetDeliveryError(
                f"invalid translation Parquet payload for {sample_id}"
            ) from error
        radar_feature: np.ndarray | None = None
        if self.consumer_input_mode != POSE_ONLY_INPUT_MODE:
            try:
                radar_feature = np.asarray(row["radar_feature"], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as error:
                raise ParquetDeliveryError(
                    f"invalid feature translation Parquet payload for {sample_id}"
                ) from error
        elif self.input_mode == POSE_ONLY_INPUT_MODE and "radar_feature" in row:
            raise ParquetDeliveryError(
                f"pose_only translation Parquet payload has a radar feature for {sample_id}"
            )
        frames = _require_integer(row.get("frame_count"), "Parquet frame_count", minimum=1)
        if pose.shape != (frames, 2, self.joint_count, self.coordinate_dim):
            raise ParquetDeliveryError(f"translation pose shape mismatch for {sample_id}")
        if confidence.shape != (frames, 2, self.joint_count):
            raise ParquetDeliveryError(f"translation confidence shape mismatch for {sample_id}")
        if radar_feature is not None and radar_feature.shape != (
            frames,
            self.radar_feature_dim,
        ):
            raise ParquetDeliveryError(f"translation radar feature shape mismatch for {sample_id}")
        if frame_mask.shape != (frames,) or not bool(np.any(frame_mask)):
            raise ParquetDeliveryError(f"translation frame mask mismatch for {sample_id}")
        if not all(
            bool(np.all(np.isfinite(value)))
            for value in (pose, confidence, radar_feature)
            if value is not None
        ):
            raise ParquetDeliveryError(f"translation payload must be finite for {sample_id}")
        if bool(np.any((confidence < 0) | (confidence > 1))):
            raise ParquetDeliveryError(
                f"translation confidence must be within [0,1] for {sample_id}"
            )
        return SignLanguageTranslationSample(
            sample_id=sample_id,
            pose=pose,
            pose_confidence=confidence,
            radar_feature=radar_feature,
            frame_mask=frame_mask,
            caption=caption,
            coordinate_frame=self.coordinate_frame,
            pose_units=self.pose_units,
        )
