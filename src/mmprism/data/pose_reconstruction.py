from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mmprism.contracts import ManifestError, ModalityRef, SampleRecord

RADAR_CUBE_POWER_PROTOCOL = "mmprism.radar_cube.power_v1"
POSE_RECONSTRUCTION_SAMPLE_PROTOCOL = "mmprism.pose_reconstruction.sample_v1"
RADAR_CUBE_MODALITY = "radar_cube"
POSE_TARGET_MODALITY = "pose_gt"
FRAME_MASK_MODALITY = "frame_mask"
POSE_VALID_MODALITY = "pose_valid"


class PoseReconstructionDataError(RuntimeError):
    """Raised when model-ready pose reconstruction data violates its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_local_array(
    reference: ModalityRef,
    *,
    modality: str,
    sample_id: str,
    data_root: Path,
) -> Path:
    if reference.uri is None or "://" in reference.uri:
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} must use a local relative URI"
        )
    path = (data_root / reference.uri).resolve()
    if not path.is_relative_to(data_root):
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} escapes the data root"
        )
    if path.suffix != ".npy":
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} must reference one .npy array"
        )
    if not path.is_file():
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} is missing: {path}"
        )
    if reference.sha256 is None:
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} requires SHA-256 provenance"
        )
    return path


def _require_metadata(
    reference: ModalityRef,
    *,
    modality: str,
    sample_id: str,
    shape: tuple[int, ...] | None = None,
    dimensions: int | None = None,
    dtype: str,
) -> tuple[int, ...]:
    if reference.shape is None:
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} requires shape metadata"
        )
    if any(size <= 0 for size in reference.shape):
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} dimensions must be positive"
        )
    if shape is not None and reference.shape != shape:
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} must have shape {shape}, got {reference.shape}"
        )
    if dimensions is not None and len(reference.shape) != dimensions:
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} must have {dimensions} dimensions"
        )
    if reference.dtype != dtype:
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} must have dtype {dtype}, "
            f"got {reference.dtype!r}"
        )
    return reference.shape


@dataclass(frozen=True, slots=True)
class PoseReconstructionRecord:
    sample_id: str
    sequence_id: str | None
    subject_id: str | None
    dataset: str
    radar_cube_path: Path
    pose_target_path: Path
    frame_mask_path: Path | None
    pose_valid_path: Path | None
    radar_cube_shape: tuple[int, int, int, int, int]
    coordinate_frame: str
    checksums: dict[str, str]


@dataclass(frozen=True, slots=True)
class PoseReconstructionSample:
    sample_id: str
    radar_cube: NDArray[np.float32]
    frame_mask: NDArray[np.bool_]
    pose_target: NDArray[np.float32]
    pose_valid: NDArray[np.bool_]
    coordinate_frame: str


@dataclass(frozen=True, slots=True)
class PoseReconstructionBatch:
    sample_ids: tuple[str, ...]
    radar_cube: NDArray[np.float32]
    frame_mask: NDArray[np.bool_]
    pose_target: NDArray[np.float32]
    pose_valid: NDArray[np.bool_]
    coordinate_frame: str


def _optional_mask_reference(
    record: SampleRecord,
    *,
    modality: str,
    expected_shape: tuple[int, ...],
    data_root: Path,
) -> tuple[Path | None, str | None]:
    reference = record.modalities.get(modality)
    if reference is None:
        return None, None
    _require_metadata(
        reference,
        modality=modality,
        sample_id=record.sample_id,
        shape=expected_shape,
        dtype="bool",
    )
    path = _resolve_local_array(
        reference,
        modality=modality,
        sample_id=record.sample_id,
        data_root=data_root,
    )
    return path, reference.sha256


def _pose_record(record: SampleRecord, data_root: Path) -> PoseReconstructionRecord:
    required = {RADAR_CUBE_MODALITY, POSE_TARGET_MODALITY}
    optional = {FRAME_MASK_MODALITY, POSE_VALID_MODALITY}
    missing = sorted(required - set(record.modalities))
    unexpected = sorted(set(record.modalities) - required - optional)
    if missing:
        raise PoseReconstructionDataError(
            f"sample {record.sample_id} is missing modalities: {', '.join(missing)}"
        )
    if unexpected:
        raise PoseReconstructionDataError(
            f"sample {record.sample_id} has unsupported modalities: {', '.join(unexpected)}"
        )

    acquisition = record.acquisition or {}
    if acquisition.get("sample_protocol") != POSE_RECONSTRUCTION_SAMPLE_PROTOCOL:
        raise PoseReconstructionDataError(
            f"sample {record.sample_id} has an unsupported pose sample protocol"
        )
    if acquisition.get("radar_cube_protocol") != RADAR_CUBE_POWER_PROTOCOL:
        raise PoseReconstructionDataError(
            f"sample {record.sample_id} has an unsupported radar cube protocol"
        )
    if acquisition.get("pose_units") != "m":
        raise PoseReconstructionDataError(f"sample {record.sample_id} pose units must be metres")
    coordinate_frame = acquisition.get("pose_coordinate_frame")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise PoseReconstructionDataError(
            f"sample {record.sample_id} requires an explicit pose coordinate frame"
        )

    radar_reference = record.modalities[RADAR_CUBE_MODALITY]
    radar_shape = _require_metadata(
        radar_reference,
        modality=RADAR_CUBE_MODALITY,
        sample_id=record.sample_id,
        dimensions=5,
        dtype="float32",
    )
    pose_reference = record.modalities[POSE_TARGET_MODALITY]
    _require_metadata(
        pose_reference,
        modality=POSE_TARGET_MODALITY,
        sample_id=record.sample_id,
        shape=(2, 24, 3),
        dtype="float32",
    )
    radar_path = _resolve_local_array(
        radar_reference,
        modality=RADAR_CUBE_MODALITY,
        sample_id=record.sample_id,
        data_root=data_root,
    )
    pose_path = _resolve_local_array(
        pose_reference,
        modality=POSE_TARGET_MODALITY,
        sample_id=record.sample_id,
        data_root=data_root,
    )
    frame_mask_path, frame_mask_sha256 = _optional_mask_reference(
        record,
        modality=FRAME_MASK_MODALITY,
        expected_shape=(radar_shape[0],),
        data_root=data_root,
    )
    pose_valid_path, pose_valid_sha256 = _optional_mask_reference(
        record,
        modality=POSE_VALID_MODALITY,
        expected_shape=(2, 24),
        data_root=data_root,
    )
    checksums = {
        RADAR_CUBE_MODALITY: radar_reference.sha256 or "",
        POSE_TARGET_MODALITY: pose_reference.sha256 or "",
    }
    if frame_mask_sha256 is not None:
        checksums[FRAME_MASK_MODALITY] = frame_mask_sha256
    if pose_valid_sha256 is not None:
        checksums[POSE_VALID_MODALITY] = pose_valid_sha256
    return PoseReconstructionRecord(
        sample_id=record.sample_id,
        sequence_id=record.sequence_id,
        subject_id=record.subject_id,
        dataset=record.dataset,
        radar_cube_path=radar_path,
        pose_target_path=pose_path,
        frame_mask_path=frame_mask_path,
        pose_valid_path=pose_valid_path,
        radar_cube_shape=(
            radar_shape[0],
            radar_shape[1],
            radar_shape[2],
            radar_shape[3],
            radar_shape[4],
        ),
        coordinate_frame=coordinate_frame.strip(),
        checksums=checksums,
    )


def _load_array(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[np.generic],
    sample_id: str,
    modality: str,
) -> NDArray[np.generic]:
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise PoseReconstructionDataError(
            f"unable to load sample {sample_id} modality {modality}: {error}"
        ) from error
    if not isinstance(value, np.ndarray):
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} is not one NumPy array"
        )
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise PoseReconstructionDataError(
            f"sample {sample_id} modality {modality} metadata mismatch: "
            f"expected {expected_shape}/{expected_dtype.name}, got {value.shape}/{value.dtype.name}"
        )
    return value


class PoseReconstructionManifest:
    """Validated local model-ready records without path inference or torch imports."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        data_root: str | Path,
        verify_checksums: bool = True,
    ) -> None:
        self.path = Path(manifest_path).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.path.is_file():
            raise PoseReconstructionDataError(f"manifest does not exist: {self.path}")
        if not self.data_root.is_dir():
            raise PoseReconstructionDataError(f"data root does not exist: {self.data_root}")

        records: list[PoseReconstructionRecord] = []
        sample_ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload: object = json.loads(raw_line)
                    record = SampleRecord.from_mapping(payload, f"line {line_number}")
                except (json.JSONDecodeError, ManifestError) as error:
                    raise PoseReconstructionDataError(
                        f"invalid pose manifest record at {self.path}:{line_number}: {error}"
                    ) from error
                if record.sample_id in sample_ids:
                    raise PoseReconstructionDataError(
                        f"duplicate sample_id {record.sample_id!r} at line {line_number}"
                    )
                sample_ids.add(record.sample_id)
                records.append(_pose_record(record, self.data_root))
        if not records:
            raise PoseReconstructionDataError("pose reconstruction manifest is empty")

        spatial_shapes = {record.radar_cube_shape[1:] for record in records}
        coordinate_frames = {record.coordinate_frame for record in records}
        if len(spatial_shapes) != 1:
            raise PoseReconstructionDataError(
                "all radar cubes in one manifest must share Doppler/range/azimuth/elevation shape"
            )
        if len(coordinate_frames) != 1:
            raise PoseReconstructionDataError(
                "all pose targets in one manifest must share one coordinate frame"
            )
        self.records = tuple(records)
        self.radar_spatial_shape = next(iter(spatial_shapes))
        self.coordinate_frame = next(iter(coordinate_frames))
        if verify_checksums:
            self.verify_checksums()

    def __len__(self) -> int:
        return len(self.records)

    def verify_checksums(self) -> None:
        for record in self.records:
            paths = {
                RADAR_CUBE_MODALITY: record.radar_cube_path,
                POSE_TARGET_MODALITY: record.pose_target_path,
            }
            if record.frame_mask_path is not None:
                paths[FRAME_MASK_MODALITY] = record.frame_mask_path
            if record.pose_valid_path is not None:
                paths[POSE_VALID_MODALITY] = record.pose_valid_path
            for modality, path in paths.items():
                observed = _sha256_file(path)
                expected = record.checksums[modality]
                if observed != expected:
                    raise PoseReconstructionDataError(
                        f"sample {record.sample_id} modality {modality} SHA-256 mismatch: "
                        f"expected {expected}, got {observed}"
                    )

    def load_sample(self, index: int) -> PoseReconstructionSample:
        record = self.records[index]
        radar_cube = cast(
            NDArray[np.float32],
            _load_array(
                record.radar_cube_path,
                expected_shape=record.radar_cube_shape,
                expected_dtype=np.dtype(np.float32),
                sample_id=record.sample_id,
                modality=RADAR_CUBE_MODALITY,
            ),
        )
        pose_target = cast(
            NDArray[np.float32],
            _load_array(
                record.pose_target_path,
                expected_shape=(2, 24, 3),
                expected_dtype=np.dtype(np.float32),
                sample_id=record.sample_id,
                modality=POSE_TARGET_MODALITY,
            ),
        )
        if not bool(np.all(np.isfinite(radar_cube))) or bool(np.any(radar_cube < 0)):
            raise PoseReconstructionDataError(
                f"sample {record.sample_id} radar cube must be finite non-negative power"
            )
        if not bool(np.all(np.isfinite(pose_target))):
            raise PoseReconstructionDataError(
                f"sample {record.sample_id} pose target must be finite"
            )

        if record.frame_mask_path is None:
            frame_mask = np.ones(record.radar_cube_shape[0], dtype=np.bool_)
        else:
            frame_mask = cast(
                NDArray[np.bool_],
                _load_array(
                    record.frame_mask_path,
                    expected_shape=(record.radar_cube_shape[0],),
                    expected_dtype=np.dtype(np.bool_),
                    sample_id=record.sample_id,
                    modality=FRAME_MASK_MODALITY,
                ),
            )
        if record.pose_valid_path is None:
            pose_valid = np.ones((2, 24), dtype=np.bool_)
        else:
            pose_valid = cast(
                NDArray[np.bool_],
                _load_array(
                    record.pose_valid_path,
                    expected_shape=(2, 24),
                    expected_dtype=np.dtype(np.bool_),
                    sample_id=record.sample_id,
                    modality=POSE_VALID_MODALITY,
                ),
            )
        if not bool(np.any(frame_mask)):
            raise PoseReconstructionDataError(
                f"sample {record.sample_id} must contain at least one valid frame"
            )
        if not bool(np.any(pose_valid)):
            raise PoseReconstructionDataError(
                f"sample {record.sample_id} must contain at least one valid pose joint"
            )
        return PoseReconstructionSample(
            sample_id=record.sample_id,
            radar_cube=np.asarray(radar_cube, dtype=np.float32),
            frame_mask=np.asarray(frame_mask, dtype=np.bool_),
            pose_target=np.asarray(pose_target, dtype=np.float32),
            pose_valid=np.asarray(pose_valid, dtype=np.bool_),
            coordinate_frame=record.coordinate_frame,
        )


def collate_pose_reconstruction_samples(
    samples: Sequence[PoseReconstructionSample],
    *,
    max_frames: int,
) -> PoseReconstructionBatch:
    if not samples:
        raise PoseReconstructionDataError("cannot collate an empty pose batch")
    if max_frames < 1:
        raise PoseReconstructionDataError("max_frames must be positive")
    spatial_shapes = {sample.radar_cube.shape[1:] for sample in samples}
    coordinate_frames = {sample.coordinate_frame for sample in samples}
    if len(spatial_shapes) != 1 or len(coordinate_frames) != 1:
        raise PoseReconstructionDataError(
            "one pose batch requires aligned radar shapes and coordinate frames"
        )
    batch_frames = max(sample.radar_cube.shape[0] for sample in samples)
    if batch_frames > max_frames:
        raise PoseReconstructionDataError(
            f"pose batch contains {batch_frames} frames, model maximum is {max_frames}"
        )
    spatial_shape = next(iter(spatial_shapes))
    radar_cube = np.zeros((len(samples), batch_frames, *spatial_shape), dtype=np.float32)
    frame_mask = np.zeros((len(samples), batch_frames), dtype=np.bool_)
    for index, sample in enumerate(samples):
        frames = sample.radar_cube.shape[0]
        radar_cube[index, :frames] = sample.radar_cube
        frame_mask[index, :frames] = sample.frame_mask
        radar_cube[index, :frames][~sample.frame_mask] = 0
    return PoseReconstructionBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        radar_cube=radar_cube,
        frame_mask=frame_mask,
        pose_target=np.stack([sample.pose_target for sample in samples]),
        pose_valid=np.stack([sample.pose_valid for sample in samples]),
        coordinate_frame=next(iter(coordinate_frames)),
    )
