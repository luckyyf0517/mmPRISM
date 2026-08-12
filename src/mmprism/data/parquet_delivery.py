from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
import yaml

from mmprism.config import expand_environment
from mmprism.contracts import (
    POSE_ONLY_INPUT_MODE,
    POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
    TRANSLATION_INPUT_MODES,
    ManifestError,
    SampleRecord,
    SplitAssignment,
    translation_input_mode_modalities,
    validate_manifest,
    validate_split_assignments,
)
from mmprism.data.pose_reconstruction import (
    FRAME_MASK_MODALITY,
    POSE_RECONSTRUCTION_SAMPLE_PROTOCOL,
    POSE_TARGET_MODALITY,
    POSE_VALID_MODALITY,
    RADAR_CUBE_MODALITY,
    PoseReconstructionDataError,
    PoseReconstructionManifest,
    PoseReconstructionSample,
)
from mmprism.data.sign_language_translation import (
    SIGN_LANGUAGE_TRANSLATION_SAMPLE_PROTOCOL,
    TRANSLATION_CAPTION_MODALITY,
    TRANSLATION_FRAME_MASK_MODALITY,
    TRANSLATION_POSE_CONFIDENCE_MODALITY,
    TRANSLATION_POSE_MODALITY,
    TRANSLATION_RADAR_FEATURE_MODALITY,
    SignLanguageTranslationDataError,
    SignLanguageTranslationManifest,
    SignLanguageTranslationSample,
)
from mmprism.runtime import collect_runtime_report

PARQUET_DELIVERY_SCHEMA = "mmprism.parquet_delivery.v2"
PARQUET_DELIVERY_CONFIG_SCHEMA = "mmprism.parquet_delivery_config.v1"
PARQUET_DELIVERY_SCHEMA_FILE = "mmprism.parquet_delivery_schema.v2"
PARQUET_DELIVERY_INVENTORY_SCHEMA = "mmprism.parquet_delivery_inventory.v1"
PARQUET_DELIVERY_INDEX_SCHEMA = "mmprism.parquet_delivery_index.v1"
PARQUET_DELIVERY_VALIDATION_SCHEMA = "mmprism.parquet_delivery_validation.v1"

POSE_RECONSTRUCTION_PRODUCT = "pose_reconstruction"
SIGN_LANGUAGE_TRANSLATION_PRODUCT = "sign_language_translation"
SUPPORTED_PRODUCTS = frozenset(
    (POSE_RECONSTRUCTION_PRODUCT, SIGN_LANGUAGE_TRANSLATION_PRODUCT)
)
PRODUCT_PROTOCOLS = {
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
class _DeliveryMetadata:
    root: Path
    product: str
    build_id: str
    source_manifest_sha256: str
    split_assignment_sha256: str
    part_rows: int
    parts_per_chunk: int
    input_mode: str | None
    coordinate_frame: str
    static_dimensions: dict[str, int]
    delivery: dict[str, object]


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    sample_id: str
    group_id: str
    split: str
    relative_path: PurePosixPath
    row_index: int


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
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
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


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ParquetDeliveryError(f"{location} must be a mapping")
    return value


def _reject_unknown(
    payload: Mapping[str, object], allowed: set[str], location: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ParquetDeliveryError(f"unknown keys in {location}: {', '.join(unknown)}")


def _relative_path(value: object, location: str) -> PurePosixPath:
    text = _require_text(value, location)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ParquetDeliveryError(f"{location} must be a safe portable relative path")
    return path


def _config_relative_path(value: object, location: str) -> Path:
    portable = _relative_path(value, location)
    return Path(portable)


def load_parquet_delivery_config(path: str | Path) -> ParquetDeliveryConfig:
    """Load strict portable Parquet delivery config with frozen input bindings."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ParquetDeliveryError(
            f"unable to load Parquet delivery config {config_path}: {error}"
        ) from error
    try:
        payload = _mapping(expand_environment(raw), "root")
    except ValueError as error:
        raise ParquetDeliveryError(str(error)) from error
    _reject_unknown(payload, {"schema_version", "source", "delivery", "validation"}, "root")
    if payload.get("schema_version") != PARQUET_DELIVERY_CONFIG_SCHEMA:
        raise ParquetDeliveryError(
            f"schema_version must be {PARQUET_DELIVERY_CONFIG_SCHEMA}"
        )
    source = _mapping(payload.get("source"), "source")
    delivery = _mapping(payload.get("delivery"), "delivery")
    validation = _mapping(payload.get("validation"), "validation")
    _reject_unknown(
        source,
        {
            "data_root",
            "manifest",
            "expected_manifest_sha256",
            "split_assignments",
            "expected_split_assignment_sha256",
            "scope",
        },
        "source",
    )
    _reject_unknown(
        delivery,
        {"product", "processed_root", "part_rows", "parts_per_chunk"},
        "delivery",
    )
    _reject_unknown(
        validation,
        {"minimum_free_bytes", "verify_source_checksums"},
        "validation",
    )
    data_root = Path(_require_text(source.get("data_root"), "source.data_root")).expanduser()
    if not data_root.is_absolute():
        raise ParquetDeliveryError("source.data_root must be absolute after environment expansion")
    scope = _require_text(source.get("scope"), "source.scope")
    if scope not in {"partial", "complete"}:
        raise ParquetDeliveryError("source.scope must be partial or complete")
    verify_source_checksums = validation.get("verify_source_checksums")
    if not isinstance(verify_source_checksums, bool):
        raise ParquetDeliveryError("validation.verify_source_checksums must be boolean")
    resolved_root = data_root.resolve()
    return ParquetDeliveryConfig(
        product=_require_text(delivery.get("product"), "delivery.product"),
        data_root=resolved_root,
        source_manifest_path=(
            resolved_root / _config_relative_path(source.get("manifest"), "source.manifest")
        ).resolve(),
        split_assignment_path=(
            resolved_root
            / _config_relative_path(source.get("split_assignments"), "source.split_assignments")
        ).resolve(),
        processed_root=(
            resolved_root
            / _config_relative_path(delivery.get("processed_root"), "delivery.processed_root")
        ).resolve(),
        expected_source_manifest_sha256=_require_sha256(
            source.get("expected_manifest_sha256"), "source.expected_manifest_sha256"
        ),
        expected_split_assignment_sha256=_require_sha256(
            source.get("expected_split_assignment_sha256"),
            "source.expected_split_assignment_sha256",
        ),
        source_scope=cast(Literal["partial", "complete"], scope),
        part_rows=_require_integer(
            delivery.get("part_rows"),
            "delivery.part_rows",
            minimum=1,
            maximum=MAX_PART_ROWS,
        ),
        parts_per_chunk=_require_integer(
            delivery.get("parts_per_chunk"),
            "delivery.parts_per_chunk",
            minimum=1,
            maximum=MAX_PARTS_PER_CHUNK,
        ),
        minimum_free_bytes=_require_integer(
            validation.get("minimum_free_bytes"),
            "validation.minimum_free_bytes",
            minimum=0,
        ),
        verify_source_checksums=verify_source_checksums,
    )


def _resolve_relative(root: Path, relative: PurePosixPath, location: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ParquetDeliveryError(f"{location} escapes delivery root")
    return candidate


def _nearest_existing_directory(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists():
        if current.parent == current:
            raise ParquetDeliveryError(f"cannot locate existing parent for {path}")
        current = current.parent
    if not current.is_dir():
        raise ParquetDeliveryError(f"existing destination parent is not a directory: {current}")
    return current


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


def _validate_config(config: ParquetDeliveryConfig) -> None:
    if config.product not in SUPPORTED_PRODUCTS:
        raise ParquetDeliveryError(
            f"unsupported delivery product {config.product!r}; expected one of "
            f"{sorted(SUPPORTED_PRODUCTS)}"
        )
    if config.source_scope not in {"partial", "complete"}:
        raise ParquetDeliveryError("source_scope must be partial or complete")
    _require_integer(config.part_rows, "part_rows", minimum=1, maximum=MAX_PART_ROWS)
    _require_integer(
        config.parts_per_chunk,
        "parts_per_chunk",
        minimum=1,
        maximum=MAX_PARTS_PER_CHUNK,
    )
    _require_integer(config.minimum_free_bytes, "minimum_free_bytes", minimum=0)
    _require_sha256(
        config.expected_source_manifest_sha256,
        "expected_source_manifest_sha256",
    )
    _require_sha256(
        config.expected_split_assignment_sha256,
        "expected_split_assignment_sha256",
    )
    if not config.data_root.expanduser().resolve().is_dir():
        raise ParquetDeliveryError(f"data root does not exist: {config.data_root}")
    if not config.source_manifest_path.expanduser().resolve().is_file():
        raise ParquetDeliveryError(
            f"source manifest does not exist: {config.source_manifest_path}"
        )
    if not config.split_assignment_path.expanduser().resolve().is_file():
        raise ParquetDeliveryError(
            f"split assignments do not exist: {config.split_assignment_path}"
        )


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


def _estimate_array_bytes(record: SampleRecord, modality: str) -> int:
    reference = record.modalities.get(modality)
    if reference is None or reference.shape is None or reference.dtype is None:
        raise ParquetDeliveryError(
            f"sample {record.sample_id} modality {modality} requires shape and dtype"
        )
    try:
        item_size = np.dtype(reference.dtype).itemsize
    except TypeError as error:
        raise ParquetDeliveryError(
            f"sample {record.sample_id} modality {modality} has unsupported dtype "
            f"{reference.dtype!r}"
        ) from error
    element_count = int(np.prod(reference.shape, dtype=np.int64))
    return item_size * element_count


def _estimate_payload_bytes(
    records: Mapping[str, SampleRecord],
    product: str,
    *,
    input_mode: str | None,
) -> int:
    if product == POSE_RECONSTRUCTION_PRODUCT:
        modalities: tuple[str, ...] = (
            RADAR_CUBE_MODALITY,
            POSE_TARGET_MODALITY,
            FRAME_MASK_MODALITY,
            POSE_VALID_MODALITY,
        )
    elif input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
        modalities = (
            TRANSLATION_POSE_MODALITY,
            TRANSLATION_POSE_CONFIDENCE_MODALITY,
            TRANSLATION_RADAR_FEATURE_MODALITY,
            TRANSLATION_FRAME_MASK_MODALITY,
        )
    elif input_mode == POSE_ONLY_INPUT_MODE:
        modalities = (
            TRANSLATION_POSE_MODALITY,
            TRANSLATION_POSE_CONFIDENCE_MODALITY,
            TRANSLATION_FRAME_MASK_MODALITY,
        )
    else:
        raise ParquetDeliveryError(
            f"translation delivery requires a supported input mode, got {input_mode!r}"
        )
    total = 0
    for record in records.values():
        for modality in modalities:
            if modality in record.modalities:
                total += _estimate_array_bytes(record, modality)
        caption = record.modalities.get(TRANSLATION_CAPTION_MODALITY)
        if caption is not None and caption.text is not None:
            total += len(caption.text.encode("utf-8"))
    return total


def _load_source_adapter(
    plan: ParquetDeliveryPlan,
    *,
    verify_checksums: bool,
) -> PoseReconstructionManifest | SignLanguageTranslationManifest:
    config = plan.config
    try:
        if config.product == POSE_RECONSTRUCTION_PRODUCT:
            return PoseReconstructionManifest(
                config.source_manifest_path,
                data_root=config.data_root,
                verify_checksums=verify_checksums,
            )
        translation_manifest = SignLanguageTranslationManifest(
            config.source_manifest_path,
            data_root=config.data_root,
            verify_checksums=verify_checksums,
        )
        if translation_manifest.input_mode not in TRANSLATION_INPUT_MODES:
            raise ParquetDeliveryError(
                "translation source has an unsupported input mode: "
                f"{translation_manifest.input_mode!r}"
            )
        return translation_manifest
    except (PoseReconstructionDataError, SignLanguageTranslationDataError) as error:
        raise ParquetDeliveryError(
            f"source data is not model-ready for {config.product}: {error}"
        ) from error


def _adapter_contract(
    adapter: PoseReconstructionManifest | SignLanguageTranslationManifest,
    product: str,
) -> tuple[dict[str, int], str, str | None]:
    if product == POSE_RECONSTRUCTION_PRODUCT:
        pose_adapter = cast(PoseReconstructionManifest, adapter)
        doppler, range_bins, azimuth, elevation = pose_adapter.radar_spatial_shape
        return (
            {
                "doppler_bins": doppler,
                "range_bins": range_bins,
                "azimuth_bins": azimuth,
                "elevation_bins": elevation,
                "hands": 2,
                "joints": 24,
                "coordinate_dim": 3,
            },
            pose_adapter.coordinate_frame,
            None,
        )
    translation_adapter = cast(SignLanguageTranslationManifest, adapter)
    input_mode = translation_adapter.input_mode
    if input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE and (
        translation_adapter.radar_feature_dim is None
    ):
        raise ParquetDeliveryError(
            "feature translation Parquet delivery requires a radar feature dimension"
        )
    if input_mode == POSE_ONLY_INPUT_MODE and translation_adapter.radar_feature_dim is not None:
        raise ParquetDeliveryError(
            "pose_only translation Parquet delivery must not have a radar feature dimension"
        )
    dimensions = {
        "hands": 2,
        "joints": translation_adapter.joint_count,
        "coordinate_dim": translation_adapter.coordinate_dim,
    }
    if input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
        dimensions["radar_feature_dim"] = cast(int, translation_adapter.radar_feature_dim)
    return (
        dimensions,
        translation_adapter.coordinate_frame,
        input_mode,
    )


def plan_parquet_delivery(
    config: ParquetDeliveryConfig,
    *,
    runtime_report: Mapping[str, object] | None = None,
) -> ParquetDeliveryPlan:
    """Validate frozen source inputs and compute deterministic delivery placement."""

    _validate_config(config)
    source_manifest_path = config.source_manifest_path.expanduser().resolve()
    split_assignment_path = config.split_assignment_path.expanduser().resolve()
    report = (
        dict(collect_runtime_report(Path.cwd()))
        if runtime_report is None
        else dict(runtime_report)
    )
    git_commit = _runtime_commit(report, require_clean=False)
    source_hash = _sha256_file(source_manifest_path)
    split_hash = _sha256_file(split_assignment_path)
    if source_hash != config.expected_source_manifest_sha256:
        raise ParquetDeliveryError(
            "source manifest SHA-256 mismatch: "
            f"expected {config.expected_source_manifest_sha256}, got {source_hash}"
        )
    if split_hash != config.expected_split_assignment_sha256:
        raise ParquetDeliveryError(
            "split assignment SHA-256 mismatch: "
            f"expected {config.expected_split_assignment_sha256}, got {split_hash}"
        )
    records = _read_source_records(source_manifest_path)
    assignments = _read_assignments(split_assignment_path, set(records))
    adapter = _load_source_adapter(
        ParquetDeliveryPlan(
            config=config,
            build_id="build-" + "0" * 24,
            git_commit=git_commit,
            runtime_report=_portable_runtime_report(report),
            source_manifest_sha256=source_hash,
            split_assignment_sha256=split_hash,
            records_by_id=records,
            assignments_by_id=assignments,
            parts=(),
            datasets=(),
            input_mode=None,
            static_dimensions={},
            coordinate_frame="",
            estimated_payload_bytes=0,
            estimated_staging_bytes=0,
            required_free_bytes=0,
        ),
        verify_checksums=False,
    )
    adapter_ids = {record.sample_id for record in adapter.records}
    if adapter_ids != set(records):
        raise ParquetDeliveryError(
            "source adapter records differ from frozen source manifest: "
            f"missing={len(set(records) - adapter_ids)}, extra={len(adapter_ids - set(records))}"
        )
    static_dimensions, coordinate_frame, input_mode = _adapter_contract(adapter, config.product)

    ids_by_split: dict[str, list[str]] = defaultdict(list)
    for sample_id, assignment in assignments.items():
        ids_by_split[assignment.split].append(sample_id)
    parts = plan_parquet_layout(
        ids_by_split,
        part_rows=config.part_rows,
        parts_per_chunk=config.parts_per_chunk,
    )
    payload_bytes = _estimate_payload_bytes(
        records,
        config.product,
        input_mode=input_mode,
    )
    staging_bytes = payload_bytes * 2 + len(records) * 2048
    required_free_bytes = config.minimum_free_bytes + staging_bytes
    config_fingerprint = config.fingerprint
    build_digest = _sha256_bytes(
        "\0".join((source_hash, split_hash, config_fingerprint, git_commit)).encode("utf-8")
    )
    return ParquetDeliveryPlan(
        config=config,
        build_id=f"build-{build_digest[:24]}",
        git_commit=git_commit,
        runtime_report=_portable_runtime_report(report),
        source_manifest_sha256=source_hash,
        split_assignment_sha256=split_hash,
        records_by_id=records,
        assignments_by_id=assignments,
        parts=parts,
        datasets=tuple(sorted({record.dataset for record in records.values()})),
        input_mode=input_mode,
        static_dimensions=static_dimensions,
        coordinate_frame=coordinate_frame,
        estimated_payload_bytes=payload_bytes,
        estimated_staging_bytes=staging_bytes,
        required_free_bytes=required_free_bytes,
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
    else:
        dims = plan.static_dimensions
        if plan.input_mode not in TRANSLATION_INPUT_MODES:
            raise ParquetDeliveryError(
                "translation Parquet schema requires a supported input mode"
            )
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


def _optional_text_from(
    record: SampleRecord,
    *,
    keys: Sequence[str],
) -> str | None:
    for mapping in (record.provenance, record.acquisition):
        if mapping is None:
            continue
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _optional_uint32_from(record: SampleRecord, *, keys: Sequence[str]) -> int | None:
    for mapping in (record.provenance, record.acquisition):
        if mapping is None:
            continue
        for key in keys:
            value = mapping.get(key)
            if (
                not isinstance(value, bool)
                and isinstance(value, int)
                and 0 <= value <= 2**32 - 1
            ):
                return value
    return None


def _optional_float32_from(record: SampleRecord, *, keys: Sequence[str]) -> float | None:
    for mapping in (record.acquisition, record.provenance):
        if mapping is None:
            continue
        for key in keys:
            value = mapping.get(key)
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                float_value = float(value)
                if np.isfinite(float_value) and float_value > 0:
                    return float_value
    return None


def _common_row(
    plan: ParquetDeliveryPlan,
    record: SampleRecord,
    assignment: SplitAssignment,
    *,
    frame_count: int,
    caption: str | None,
) -> dict[str, object]:
    return {
        "schema_version": PARQUET_DELIVERY_SCHEMA,
        "product": plan.config.product,
        "build_id": plan.build_id,
        "sample_id": record.sample_id,
        "sequence_id": record.sequence_id,
        "subject_id": record.subject_id,
        "group_id": assignment.group_id,
        "split": assignment.split,
        "dataset_id": record.dataset,
        "source_archive_id": _optional_text_from(record, keys=("source_archive_id", "archive_id")),
        "source_archive_sha256": _optional_text_from(
            record, keys=("source_archive_sha256", "archive_sha256")
        ),
        "source_member": _optional_text_from(record, keys=("source_member", "member")),
        "source_member_crc32": _optional_uint32_from(
            record, keys=("source_member_crc32", "member_crc32")
        ),
        "caption": caption,
        "caption_language": _optional_text_from(record, keys=("caption_language", "language")),
        "frame_count": frame_count,
        "fps": _optional_float32_from(record, keys=("fps", "frame_rate")),
        "coordinate_frame": plan.coordinate_frame,
        "pose_units": "m",
        "annotation_config_fingerprint": _optional_text_from(
            record, keys=("annotation_config_fingerprint", "config_fingerprint")
        ),
        "source_manifest_sha256": plan.source_manifest_sha256,
        "split_assignment_sha256": plan.split_assignment_sha256,
    }


def _row_for_sample(
    plan: ParquetDeliveryPlan,
    adapter: PoseReconstructionManifest | SignLanguageTranslationManifest,
    adapter_index_by_id: Mapping[str, int],
    sample_id: str,
) -> dict[str, object]:
    record = plan.records_by_id[sample_id]
    assignment = plan.assignments_by_id[sample_id]
    index = adapter_index_by_id[sample_id]
    try:
        if plan.config.product == POSE_RECONSTRUCTION_PRODUCT:
            pose_adapter = cast(PoseReconstructionManifest, adapter)
            pose_sample = pose_adapter.load_sample(index)
            row = _common_row(
                plan,
                record,
                assignment,
                frame_count=int(pose_sample.radar_cube.shape[0]),
                caption=None,
            )
            row.update(
                {
                    "radar_cube": pose_sample.radar_cube.tolist(),
                    "frame_mask": pose_sample.frame_mask.tolist(),
                    "pose_gt": pose_sample.pose_target.tolist(),
                    "pose_valid": pose_sample.pose_valid.tolist(),
                }
            )
            return row
        translation_adapter = cast(SignLanguageTranslationManifest, adapter)
        translation_sample = translation_adapter.load_sample(index)
        if (
            plan.input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE
            and translation_sample.radar_feature is None
        ):
            raise ParquetDeliveryError(
                "feature translation Parquet delivery requires radar features"
            )
        if (
            plan.input_mode == POSE_ONLY_INPUT_MODE
            and translation_sample.radar_feature is not None
        ):
            raise ParquetDeliveryError(
                "pose_only translation Parquet delivery must not load radar features"
            )
        row = _common_row(
            plan,
            record,
            assignment,
            frame_count=int(translation_sample.pose.shape[0]),
            caption=translation_sample.caption,
        )
        row.update(
            {
                "pose": translation_sample.pose.tolist(),
                "pose_confidence": translation_sample.pose_confidence.tolist(),
                "frame_mask": translation_sample.frame_mask.tolist(),
            }
        )
        if plan.input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
            row["radar_feature"] = cast(
                np.ndarray, translation_sample.radar_feature
            ).tolist()
        return row
    except (PoseReconstructionDataError, SignLanguageTranslationDataError) as error:
        raise ParquetDeliveryError(
            f"source sample {sample_id} is not model-ready: {error}"
        ) from error


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical_json_bytes(row) for row in rows))


def _copy_frozen_input(source: Path, destination: Path, expected_sha256: str) -> dict[str, object]:
    shutil.copyfile(source, destination)
    observed_sha256 = _sha256_file(destination)
    if observed_sha256 != expected_sha256:
        raise ParquetDeliveryError(
            f"frozen input copy hash mismatch for {source}: expected {expected_sha256}, "
            f"got {observed_sha256}"
        )
    return {
        "path": destination.name,
        "sha256": observed_sha256,
        "byte_count": destination.stat().st_size,
    }


def _inventory_row(
    part: ParquetDeliveryPartPlan,
    path: Path,
    *,
    schema_fingerprint: str,
) -> dict[str, object]:
    _, pq = _require_pyarrow()
    metadata = pq.ParquetFile(path).metadata
    return {
        "schema_version": PARQUET_DELIVERY_INVENTORY_SCHEMA,
        "split": part.split,
        "chunk_index": part.chunk_index,
        "part_index": part.part_index,
        "path": part.relative_path.as_posix(),
        "row_count": len(part.sample_ids),
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
        "sample_id_digest": _sample_id_digest(part.sample_ids),
        "row_group_count": metadata.num_row_groups,
        "schema_fingerprint": schema_fingerprint,
    }


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return _sha256_bytes("".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8"))


def _delivery_payload(
    plan: ParquetDeliveryPlan,
    *,
    schema_fingerprint: str,
    pyarrow_version: str,
    inputs: Mapping[str, Mapping[str, object]],
    status: str,
) -> dict[str, object]:
    return {
        "schema_version": PARQUET_DELIVERY_SCHEMA,
        "status": status,
        "product": plan.config.product,
        "product_protocol": PRODUCT_PROTOCOLS[plan.config.product],
        "build_id": plan.build_id,
        "source_scope": plan.config.source_scope,
        "source_manifest_sha256": plan.source_manifest_sha256,
        "split_assignment_sha256": plan.split_assignment_sha256,
        "source_datasets": list(plan.datasets),
        "input_mode": plan.input_mode,
        "sample_count": plan.sample_count,
        "splits": list(plan.splits),
        "row_policy": {
            "maximum_rows_per_part": plan.config.part_rows,
            "maximum_parts_per_chunk": plan.config.parts_per_chunk,
            "row_groups_per_part": 1,
        },
        "compression": "zstd",
        "writer": {"pyarrow": pyarrow_version},
        "static_dimensions": plan.static_dimensions,
        "coordinate_frame": plan.coordinate_frame,
        "schema_fingerprint": schema_fingerprint,
        "build": {
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "git_commit": plan.git_commit,
            "git": plan.runtime_report["git"],
            "runtime_environment": plan.runtime_report,
            "resolved_delivery_config": plan.config.portable_dict(),
            "randomness": "none_deterministic_placement_v1",
            "config_fingerprint": plan.config.fingerprint,
            "estimated_payload_bytes": plan.estimated_payload_bytes,
            "estimated_staging_bytes": plan.estimated_staging_bytes,
            "required_free_bytes": plan.required_free_bytes,
        },
        "inputs": dict(inputs),
        "inventory_path": "inventories/parts.jsonl",
        "index_path": "indices/sample_index.jsonl",
        "schema_path": "schema.json",
        "validation_path": "validation/report.json",
    }


def _schema_payload(plan: ParquetDeliveryPlan, schema: Any) -> dict[str, object]:
    return {
        "schema_version": PARQUET_DELIVERY_SCHEMA_FILE,
        "product": plan.config.product,
        "product_protocol": PRODUCT_PROTOCOLS[plan.config.product],
        "input_mode": plan.input_mode,
        "static_dimensions": plan.static_dimensions,
        "coordinate_frame": plan.coordinate_frame,
        "schema_fingerprint": _schema_fingerprint(schema),
        "arrow_schema": str(schema.remove_metadata()),
    }


def _write_sha256sums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".staging-" not in path.name
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files
    ]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def _validate_sha256sums(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ParquetDeliveryError("delivery SHA256SUMS is missing")
    listed: set[PurePosixPath] = set()
    with sums_path.open("r", encoding="ascii") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest, separator, raw_path = raw_line.rstrip("\n").partition("  ")
            if not separator:
                raise ParquetDeliveryError(
                    f"invalid SHA256SUMS line {line_number}"
                )
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


def _write_part(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    schema: Any,
) -> None:
    pa, pq = _require_pyarrow()
    table = pa.Table.from_pylist(rows, schema=schema)
    if table.num_rows != len(rows):
        raise ParquetDeliveryError("Parquet table row-count mismatch before writing")
    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=len(rows),
        version="2.6",
    )


def materialize_parquet_delivery(
    config: ParquetDeliveryConfig,
    *,
    runtime_report: Mapping[str, object] | None = None,
) -> ParquetDeliveryValidation:
    """Build and atomically publish a final immutable Parquet delivery."""

    report = (
        dict(collect_runtime_report(Path.cwd()))
        if runtime_report is None
        else dict(runtime_report)
    )
    _runtime_commit(report, require_clean=True)
    plan = plan_parquet_delivery(config, runtime_report=report)
    target = plan.delivery_root
    if target.exists():
        raise ParquetDeliveryError(f"delivery destination already exists: {target}")
    storage_parent = _nearest_existing_directory(target.parent)
    available = shutil.disk_usage(storage_parent).free
    if available < plan.required_free_bytes:
        raise ParquetDeliveryError(
            f"insufficient free space for delivery: required {plan.required_free_bytes}, "
            f"available {available}"
        )

    pa, _ = _require_pyarrow()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{plan.build_id}.staging-", dir=target.parent)
    )
    try:
        adapter = _load_source_adapter(plan, verify_checksums=config.verify_source_checksums)
        if config.product == POSE_RECONSTRUCTION_PRODUCT:
            pose_adapter = cast(PoseReconstructionManifest, adapter)
            adapter_index_by_id = {
                record.sample_id: index for index, record in enumerate(pose_adapter.records)
            }
        else:
            translation_adapter = cast(SignLanguageTranslationManifest, adapter)
            adapter_index_by_id = {
                record.sample_id: index
                for index, record in enumerate(translation_adapter.records)
            }
        if set(adapter_index_by_id) != set(plan.records_by_id):
            raise ParquetDeliveryError("source adapter changed after delivery planning")
        schema = _schema_for_plan(plan)
        schema_fingerprint = _schema_fingerprint(schema)
        inputs = {
            "source_manifest": _copy_frozen_input(
                config.source_manifest_path.expanduser().resolve(),
                staging_root / "source_manifest.jsonl",
                plan.source_manifest_sha256,
            ),
            "split_assignments": _copy_frozen_input(
                config.split_assignment_path.expanduser().resolve(),
                staging_root / "split_assignments.jsonl",
                plan.split_assignment_sha256,
            ),
        }
        _write_json(staging_root / "schema.json", _schema_payload(plan, schema))
        _write_json(
            staging_root / "delivery.json",
            _delivery_payload(
                plan,
                schema_fingerprint=schema_fingerprint,
                pyarrow_version=pa.__version__,
                inputs=inputs,
                status="staging",
            ),
        )

        inventory_rows: list[dict[str, object]] = []
        index_rows: list[dict[str, object]] = []
        for part in plan.parts:
            rows = [
                _row_for_sample(plan, adapter, adapter_index_by_id, sample_id)
                for sample_id in part.sample_ids
            ]
            output_path = staging_root / Path(part.relative_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_part(output_path, rows, schema)
            inventory_rows.append(
                _inventory_row(part, output_path, schema_fingerprint=schema_fingerprint)
            )
            for row_index, sample_id in enumerate(part.sample_ids):
                assignment = plan.assignments_by_id[sample_id]
                index_rows.append(
                    {
                        "schema_version": PARQUET_DELIVERY_INDEX_SCHEMA,
                        "sample_id": sample_id,
                        "group_id": assignment.group_id,
                        "split": part.split,
                        "path": part.relative_path.as_posix(),
                        "row_index": row_index,
                    }
                )
        _write_jsonl(staging_root / "inventories" / "parts.jsonl", inventory_rows)
        _write_jsonl(staging_root / "indices" / "sample_index.jsonl", index_rows)

        preliminary = validate_parquet_delivery(staging_root, allow_incomplete=True)
        _write_json(staging_root / "validation" / "report.json", preliminary.to_dict())
        _write_json(
            staging_root / "delivery.json",
            _delivery_payload(
                plan,
                schema_fingerprint=schema_fingerprint,
                pyarrow_version=pa.__version__,
                inputs=inputs,
                status="complete",
            ),
        )
        _write_sha256sums(staging_root)
        validation = validate_parquet_delivery(staging_root)
        os.replace(staging_root, target)
        return ParquetDeliveryValidation(
            root=target,
            product=validation.product,
            build_id=validation.build_id,
            sample_count=validation.sample_count,
            part_count=validation.part_count,
            split_counts=validation.split_counts,
            inventory_sha256=validation.inventory_sha256,
            index_sha256=validation.index_sha256,
            schema_fingerprint=validation.schema_fingerprint,
        )
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


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
                    raise ParquetDeliveryError(
                        f"{location}:{line_number} must be a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ParquetDeliveryError(f"unable to read {location}: {error}") from error
    if not rows:
        raise ParquetDeliveryError(f"{location} contains no rows")
    return rows


def _load_delivery_metadata(root: Path, *, allow_incomplete: bool) -> _DeliveryMetadata:
    payload = _read_json(root / "delivery.json", "delivery metadata")
    if payload.get("schema_version") != PARQUET_DELIVERY_SCHEMA:
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
    if _require_integer(
        policy.get("row_groups_per_part"),
        "delivery.row_policy.row_groups_per_part",
        minimum=1,
    ) != 1:
        raise ParquetDeliveryError("delivery must use exactly one row group per part")
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
    build = payload.get("build")
    if not isinstance(build, Mapping):
        raise ParquetDeliveryError("delivery.build must be an object")
    created_at = build.get("created_at")
    if not isinstance(created_at, str):
        raise ParquetDeliveryError("delivery.build.created_at must be an RFC 3339 timestamp")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
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
        input_mode=input_mode,
        coordinate_frame=_require_text(
            payload.get("coordinate_frame"), "delivery.coordinate_frame"
        ),
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
        estimated_payload_bytes=0,
        estimated_staging_bytes=0,
        required_free_bytes=0,
    )
    return _schema_for_plan(plan)


def _validate_schema(root: Path, metadata: _DeliveryMetadata) -> tuple[Any, str]:
    schema_payload = _read_json(root / "schema.json", "delivery schema")
    if schema_payload.get("schema_version") != PARQUET_DELIVERY_SCHEMA_FILE:
        raise ParquetDeliveryError("unsupported delivery schema file version")
    if schema_payload.get("product") != metadata.product:
        raise ParquetDeliveryError("schema product does not match delivery product")
    if schema_payload.get("product_protocol") != PRODUCT_PROTOCOLS[metadata.product]:
        raise ParquetDeliveryError("schema product protocol does not match delivery product")
    if schema_payload.get("input_mode") != metadata.input_mode:
        raise ParquetDeliveryError("schema input mode does not match delivery product")
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
            modalities = set(record.modalities)
            if set(required) - modalities or set(forbidden) & modalities:
                raise ParquetDeliveryError(
                    f"translation source modalities do not match input mode for {record.sample_id}"
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
    declared_bytes = _require_integer(
        row.get("byte_count"), "inventory.byte_count", minimum=1
    )
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
    if parquet.metadata.num_row_groups != 1:
        raise ParquetDeliveryError(f"Parquet file must contain exactly one row group: {relative}")
    if declared_row_group_count != parquet.metadata.num_row_groups:
        raise ParquetDeliveryError(f"inventory row-group count mismatch for {relative}")
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
        raise ParquetDeliveryError(
            f"translation confidence must be within [0,1] for {sample_id}"
        )


def _validate_index_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    root: Path,
    observed_rows: Mapping[str, tuple[PurePosixPath, int, str, str]],
) -> None:
    if len(rows) != len(observed_rows):
        raise ParquetDeliveryError(
            f"sample index count mismatch: expected {len(observed_rows)}, got {len(rows)}"
        )
    seen: set[str] = set()
    for row in rows:
        if row.get("schema_version") != PARQUET_DELIVERY_INDEX_SCHEMA:
            raise ParquetDeliveryError("unsupported delivery index row schema")
        sample_id = _require_text(row.get("sample_id"), "index.sample_id")
        if sample_id in seen:
            raise ParquetDeliveryError(f"duplicate sample index ID: {sample_id}")
        seen.add(sample_id)
        group_id = _require_sha256(row.get("group_id"), "index.group_id")
        split = _require_text(row.get("split"), "index.split")
        relative = _relative_path(row.get("path"), "index.path")
        row_index = _require_integer(row.get("row_index"), "index.row_index", minimum=0)
        observed = observed_rows.get(sample_id)
        if observed is None:
            raise ParquetDeliveryError(f"index references unknown sample {sample_id}")
        if (relative, row_index, split, group_id) != observed:
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

    observed: dict[str, tuple[PurePosixPath, int, str, str]] = {}
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
                    _require_text(row.get("sample_id"), "Parquet sample_id")
                    for row in parquet_rows
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
            if row.get("schema_version") != PARQUET_DELIVERY_SCHEMA:
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
                or row.get("pose_units") != "m"
            ):
                raise ParquetDeliveryError(f"Parquet coordinate contract mismatch for {sample_id}")
            _validate_payload_row(row, metadata, sample_id)
            observed[sample_id] = (relative, row_index, assignment.split, assignment.group_id)
            sample_ids_by_split[assignment.split].append(sample_id)

    splits_root = delivery_root / "splits"
    actual_part_paths = {
        PurePosixPath(path.relative_to(delivery_root).as_posix())
        for path in splits_root.rglob("*.parquet")
    } if splits_root.is_dir() else set()
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
    _validate_index_rows(index_rows, root=delivery_root, observed_rows=observed)

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


class _ParquetDeliveryDataset:
    """Dependency-light random access over one validated delivery split."""

    expected_product: str

    def __init__(
        self,
        delivery_root: str | Path,
        *,
        split: str,
        verify_checksums: bool = True,
    ) -> None:
        root = Path(delivery_root).expanduser().resolve()
        validation = validate_parquet_delivery(root, verify_checksums=verify_checksums)
        metadata = _load_delivery_metadata(root, allow_incomplete=False)
        if metadata.product != self.expected_product:
            raise ParquetDeliveryError(
                f"expected {self.expected_product} delivery, got {metadata.product}"
            )
        self.delivery_root = root
        self.validation = validation
        self.metadata = metadata
        self.split = split
        self._entries = self._entries_for_split(root, split)
        self._cached_part_rows: dict[PurePosixPath, list[dict[str, object]]] = {}

    @staticmethod
    def _entries_for_split(root: Path, split: str) -> tuple[_IndexEntry, ...]:
        index_rows = _read_jsonl(root / "indices" / "sample_index.jsonl", "delivery sample index")
        entries: list[_IndexEntry] = []
        for row in index_rows:
            if row.get("split") != split:
                continue
            entries.append(
                _IndexEntry(
                    sample_id=_require_text(row.get("sample_id"), "index.sample_id"),
                    group_id=_require_sha256(row.get("group_id"), "index.group_id"),
                    split=split,
                    relative_path=_relative_path(row.get("path"), "index.path"),
                    row_index=_require_integer(row.get("row_index"), "index.row_index", minimum=0),
                )
            )
        if not entries:
            raise ParquetDeliveryError(f"delivery has no samples for split {split!r}")
        entries.sort(key=lambda entry: entry.sample_id)
        return tuple(entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(entry.sample_id for entry in self._entries)

    def _row(self, index: int) -> dict[str, object]:
        entry = self._entries[index]
        rows = self._cached_part_rows.get(entry.relative_path)
        if rows is None:
            _, pq = _require_pyarrow()
            path = _resolve_relative(self.delivery_root, entry.relative_path, "index.path")
            rows = cast(list[dict[str, object]], pq.read_table(path).to_pylist())
            self._cached_part_rows[entry.relative_path] = rows
        try:
            row = rows[entry.row_index]
        except IndexError as error:
            raise ParquetDeliveryError(
                f"index row is outside its Parquet part for {entry.sample_id}"
            ) from error
        if row.get("sample_id") != entry.sample_id:
            raise ParquetDeliveryError(f"index sample mismatch for {entry.sample_id}")
        return row


class ParquetPoseReconstructionDataset(_ParquetDeliveryDataset):
    """Parquet source exposing the current dependency-light pose sample contract."""

    expected_product = POSE_RECONSTRUCTION_PRODUCT

    def __init__(
        self,
        delivery_root: str | Path,
        *,
        split: str,
        verify_checksums: bool = True,
    ) -> None:
        super().__init__(delivery_root, split=split, verify_checksums=verify_checksums)
        dims = self.metadata.static_dimensions
        self.radar_spatial_shape = (
            dims["doppler_bins"],
            dims["range_bins"],
            dims["azimuth_bins"],
            dims["elevation_bins"],
        )
        self.coordinate_frame = self.metadata.coordinate_frame

    def load_sample(self, index: int) -> PoseReconstructionSample:
        row = self._row(index)
        sample_id = _require_text(row.get("sample_id"), "Parquet sample_id")
        try:
            radar_cube = np.asarray(row["radar_cube"], dtype=np.float32)
            frame_mask = np.asarray(row["frame_mask"], dtype=np.bool_)
            pose_target = np.asarray(row["pose_gt"], dtype=np.float32)
            pose_valid = np.asarray(row["pose_valid"], dtype=np.bool_)
        except (KeyError, TypeError, ValueError) as error:
            raise ParquetDeliveryError(f"invalid pose Parquet payload for {sample_id}") from error
        frame_count = _require_integer(
            row.get("frame_count"), "Parquet frame_count", minimum=1
        )
        expected_cube_shape = (frame_count, *self.radar_spatial_shape)
        if (
            radar_cube.shape != expected_cube_shape
            or frame_mask.shape != (expected_cube_shape[0],)
        ):
            raise ParquetDeliveryError(f"pose Parquet tensor shape mismatch for {sample_id}")
        if pose_target.shape != (2, 24, 3) or pose_valid.shape != (2, 24):
            raise ParquetDeliveryError(f"pose Parquet target shape mismatch for {sample_id}")
        if not bool(np.all(np.isfinite(radar_cube))) or bool(np.any(radar_cube < 0)):
            raise ParquetDeliveryError(
                f"pose Parquet cube is not finite non-negative for {sample_id}"
            )
        if not bool(np.all(np.isfinite(pose_target))) or not bool(np.any(frame_mask)):
            raise ParquetDeliveryError(f"pose Parquet validity mismatch for {sample_id}")
        if not bool(np.any(pose_valid)):
            raise ParquetDeliveryError(f"pose Parquet has no valid target joint for {sample_id}")
        return PoseReconstructionSample(
            sample_id=sample_id,
            radar_cube=radar_cube,
            frame_mask=frame_mask,
            pose_target=pose_target,
            pose_valid=pose_valid,
            coordinate_frame=self.coordinate_frame,
        )


class ParquetSignLanguageTranslationDataset(_ParquetDeliveryDataset):
    """Parquet source exposing the current dependency-light translation sample contract."""

    expected_product = SIGN_LANGUAGE_TRANSLATION_PRODUCT

    def __init__(
        self,
        delivery_root: str | Path,
        *,
        split: str,
        verify_checksums: bool = True,
    ) -> None:
        super().__init__(delivery_root, split=split, verify_checksums=verify_checksums)
        dims = self.metadata.static_dimensions
        self.joint_count = dims["joints"]
        self.coordinate_dim = dims["coordinate_dim"]
        if self.metadata.input_mode not in TRANSLATION_INPUT_MODES:
            raise ParquetDeliveryError("translation delivery input mode is unsupported")
        self.input_mode = self.metadata.input_mode
        self.radar_feature_dim = (
            dims["radar_feature_dim"]
            if self.input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE
            else None
        )
        self.coordinate_frame = self.metadata.coordinate_frame

    def load_sample(self, index: int) -> SignLanguageTranslationSample:
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
        if self.input_mode == POSE_PLUS_RADAR_FEATURE_INPUT_MODE:
            try:
                radar_feature = np.asarray(row["radar_feature"], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as error:
                raise ParquetDeliveryError(
                    f"invalid feature translation Parquet payload for {sample_id}"
                ) from error
        elif "radar_feature" in row:
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
        )
