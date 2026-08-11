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

SIGN_LANGUAGE_TRANSLATION_SAMPLE_PROTOCOL = "mmprism.sign_language_translation.sample_v1"
RADAR_FEATURE_SEQUENCE_PROTOCOL = "mmprism.radar_feature.sequence_v1"
TRANSLATION_POSE_MODALITY = "pose"
TRANSLATION_POSE_CONFIDENCE_MODALITY = "pose_confidence"
TRANSLATION_RADAR_FEATURE_MODALITY = "radar_feature"
TRANSLATION_FRAME_MASK_MODALITY = "frame_mask"
TRANSLATION_CAPTION_MODALITY = "caption"


class SignLanguageTranslationDataError(RuntimeError):
    """Raised when model-ready translation data violates its contract."""


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
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} must use a local relative URI"
        )
    path = (data_root / reference.uri).resolve()
    if not path.is_relative_to(data_root):
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} escapes the data root"
        )
    if path.suffix != ".npy":
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} must reference one .npy array"
        )
    if not path.is_file():
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} is missing: {path}"
        )
    if reference.sha256 is None:
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} requires SHA-256 provenance"
        )
    return path


def _array_shape(
    reference: ModalityRef,
    *,
    modality: str,
    sample_id: str,
    dimensions: int,
    dtype: str,
) -> tuple[int, ...]:
    if reference.shape is None or len(reference.shape) != dimensions:
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} must have {dimensions} dimensions"
        )
    if any(size <= 0 for size in reference.shape):
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} dimensions must be positive"
        )
    if reference.dtype != dtype:
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} must have dtype {dtype}, "
            f"got {reference.dtype!r}"
        )
    return reference.shape


@dataclass(frozen=True, slots=True)
class SignLanguageTranslationRecord:
    sample_id: str
    sequence_id: str | None
    subject_id: str | None
    dataset: str
    pose_path: Path
    pose_confidence_path: Path
    radar_feature_path: Path
    frame_mask_path: Path | None
    caption: str
    frame_count: int
    joint_count: int
    coordinate_dim: int
    radar_feature_dim: int
    coordinate_frame: str
    checksums: dict[str, str]


@dataclass(frozen=True, slots=True)
class SignLanguageTranslationSample:
    sample_id: str
    pose: NDArray[np.float32]
    pose_confidence: NDArray[np.float32]
    radar_feature: NDArray[np.float32]
    frame_mask: NDArray[np.bool_]
    caption: str
    coordinate_frame: str


@dataclass(frozen=True, slots=True)
class SignLanguageTranslationBatch:
    sample_ids: tuple[str, ...]
    pose: NDArray[np.float32]
    pose_confidence: NDArray[np.float32]
    radar_feature: NDArray[np.float32]
    frame_mask: NDArray[np.bool_]
    captions: tuple[str, ...]
    coordinate_frame: str


def _translation_record(
    record: SampleRecord, data_root: Path
) -> SignLanguageTranslationRecord:
    required = {
        TRANSLATION_POSE_MODALITY,
        TRANSLATION_POSE_CONFIDENCE_MODALITY,
        TRANSLATION_RADAR_FEATURE_MODALITY,
        TRANSLATION_CAPTION_MODALITY,
    }
    optional = {TRANSLATION_FRAME_MASK_MODALITY}
    missing = sorted(required - set(record.modalities))
    unexpected = sorted(set(record.modalities) - required - optional)
    if missing:
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} is missing modalities: {', '.join(missing)}"
        )
    if unexpected:
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} has unsupported modalities: {', '.join(unexpected)}"
        )

    acquisition = record.acquisition or {}
    if acquisition.get("sample_protocol") != SIGN_LANGUAGE_TRANSLATION_SAMPLE_PROTOCOL:
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} has an unsupported translation sample protocol"
        )
    if acquisition.get("radar_feature_protocol") != RADAR_FEATURE_SEQUENCE_PROTOCOL:
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} has an unsupported radar feature protocol"
        )
    if acquisition.get("pose_units") != "m":
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} pose units must be metres"
        )
    coordinate_frame = acquisition.get("pose_coordinate_frame")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} requires an explicit pose coordinate frame"
        )

    pose_reference = record.modalities[TRANSLATION_POSE_MODALITY]
    confidence_reference = record.modalities[TRANSLATION_POSE_CONFIDENCE_MODALITY]
    radar_reference = record.modalities[TRANSLATION_RADAR_FEATURE_MODALITY]
    caption_reference = record.modalities[TRANSLATION_CAPTION_MODALITY]
    pose_shape = _array_shape(
        pose_reference,
        modality=TRANSLATION_POSE_MODALITY,
        sample_id=record.sample_id,
        dimensions=4,
        dtype="float32",
    )
    if pose_shape[1] != 2:
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} pose must contain exactly two hands"
        )
    confidence_shape = _array_shape(
        confidence_reference,
        modality=TRANSLATION_POSE_CONFIDENCE_MODALITY,
        sample_id=record.sample_id,
        dimensions=3,
        dtype="float32",
    )
    radar_shape = _array_shape(
        radar_reference,
        modality=TRANSLATION_RADAR_FEATURE_MODALITY,
        sample_id=record.sample_id,
        dimensions=2,
        dtype="float32",
    )
    expected_confidence_shape = (pose_shape[0], pose_shape[1], pose_shape[2])
    if confidence_shape != expected_confidence_shape:
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} confidence shape must be "
            f"{expected_confidence_shape}, got {confidence_shape}"
        )
    if radar_shape[0] != pose_shape[0]:
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} radar and pose frame counts must match"
        )
    if caption_reference.text is None or not caption_reference.text.strip():
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} caption must be non-empty inline text"
        )
    if any(
        value is not None
        for value in (
            caption_reference.shape,
            caption_reference.dtype,
            caption_reference.sha256,
        )
    ):
        raise SignLanguageTranslationDataError(
            f"sample {record.sample_id} inline caption must not declare array metadata"
        )

    paths = {
        TRANSLATION_POSE_MODALITY: _resolve_local_array(
            pose_reference,
            modality=TRANSLATION_POSE_MODALITY,
            sample_id=record.sample_id,
            data_root=data_root,
        ),
        TRANSLATION_POSE_CONFIDENCE_MODALITY: _resolve_local_array(
            confidence_reference,
            modality=TRANSLATION_POSE_CONFIDENCE_MODALITY,
            sample_id=record.sample_id,
            data_root=data_root,
        ),
        TRANSLATION_RADAR_FEATURE_MODALITY: _resolve_local_array(
            radar_reference,
            modality=TRANSLATION_RADAR_FEATURE_MODALITY,
            sample_id=record.sample_id,
            data_root=data_root,
        ),
    }
    checksums = {
        TRANSLATION_POSE_MODALITY: pose_reference.sha256 or "",
        TRANSLATION_POSE_CONFIDENCE_MODALITY: confidence_reference.sha256 or "",
        TRANSLATION_RADAR_FEATURE_MODALITY: radar_reference.sha256 or "",
    }
    frame_mask_path: Path | None = None
    frame_reference = record.modalities.get(TRANSLATION_FRAME_MASK_MODALITY)
    if frame_reference is not None:
        frame_shape = _array_shape(
            frame_reference,
            modality=TRANSLATION_FRAME_MASK_MODALITY,
            sample_id=record.sample_id,
            dimensions=1,
            dtype="bool",
        )
        if frame_shape != (pose_shape[0],):
            raise SignLanguageTranslationDataError(
                f"sample {record.sample_id} frame mask must have shape {(pose_shape[0],)}"
            )
        frame_mask_path = _resolve_local_array(
            frame_reference,
            modality=TRANSLATION_FRAME_MASK_MODALITY,
            sample_id=record.sample_id,
            data_root=data_root,
        )
        checksums[TRANSLATION_FRAME_MASK_MODALITY] = frame_reference.sha256 or ""

    return SignLanguageTranslationRecord(
        sample_id=record.sample_id,
        sequence_id=record.sequence_id,
        subject_id=record.subject_id,
        dataset=record.dataset,
        pose_path=paths[TRANSLATION_POSE_MODALITY],
        pose_confidence_path=paths[TRANSLATION_POSE_CONFIDENCE_MODALITY],
        radar_feature_path=paths[TRANSLATION_RADAR_FEATURE_MODALITY],
        frame_mask_path=frame_mask_path,
        caption=caption_reference.text.strip(),
        frame_count=pose_shape[0],
        joint_count=pose_shape[2],
        coordinate_dim=pose_shape[3],
        radar_feature_dim=radar_shape[1],
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
        raise SignLanguageTranslationDataError(
            f"unable to load sample {sample_id} modality {modality}: {error}"
        ) from error
    if not isinstance(value, np.ndarray):
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} is not one NumPy array"
        )
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise SignLanguageTranslationDataError(
            f"sample {sample_id} modality {modality} metadata mismatch: "
            f"expected {expected_shape}/{expected_dtype.name}, got {value.shape}/{value.dtype.name}"
        )
    return value


class SignLanguageTranslationManifest:
    """Validated model-ready multimodal translation records without torch imports."""

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
            raise SignLanguageTranslationDataError(f"manifest does not exist: {self.path}")
        if not self.data_root.is_dir():
            raise SignLanguageTranslationDataError(
                f"data root does not exist: {self.data_root}"
            )

        records: list[SignLanguageTranslationRecord] = []
        sample_ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload: object = json.loads(raw_line)
                    record = SampleRecord.from_mapping(payload, f"line {line_number}")
                except (json.JSONDecodeError, ManifestError) as error:
                    raise SignLanguageTranslationDataError(
                        f"invalid translation manifest record at "
                        f"{self.path}:{line_number}: {error}"
                    ) from error
                if record.sample_id in sample_ids:
                    raise SignLanguageTranslationDataError(
                        f"duplicate sample_id {record.sample_id!r} at line {line_number}"
                    )
                sample_ids.add(record.sample_id)
                records.append(_translation_record(record, self.data_root))
        if not records:
            raise SignLanguageTranslationDataError("translation manifest is empty")

        feature_dims = {record.radar_feature_dim for record in records}
        joint_counts = {record.joint_count for record in records}
        coordinate_dims = {record.coordinate_dim for record in records}
        coordinate_frames = {record.coordinate_frame for record in records}
        if any(len(values) != 1 for values in (feature_dims, joint_counts, coordinate_dims)):
            raise SignLanguageTranslationDataError(
                "all translation records must share feature, joint, and coordinate dimensions"
            )
        if len(coordinate_frames) != 1:
            raise SignLanguageTranslationDataError(
                "all translation records must share one pose coordinate frame"
            )
        self.records = tuple(records)
        self.radar_feature_dim = next(iter(feature_dims))
        self.joint_count = next(iter(joint_counts))
        self.coordinate_dim = next(iter(coordinate_dims))
        self.coordinate_frame = next(iter(coordinate_frames))
        if verify_checksums:
            self.verify_checksums()

    def __len__(self) -> int:
        return len(self.records)

    def verify_checksums(self) -> None:
        for record in self.records:
            paths = {
                TRANSLATION_POSE_MODALITY: record.pose_path,
                TRANSLATION_POSE_CONFIDENCE_MODALITY: record.pose_confidence_path,
                TRANSLATION_RADAR_FEATURE_MODALITY: record.radar_feature_path,
            }
            if record.frame_mask_path is not None:
                paths[TRANSLATION_FRAME_MASK_MODALITY] = record.frame_mask_path
            for modality, path in paths.items():
                observed = _sha256_file(path)
                expected = record.checksums[modality]
                if observed != expected:
                    raise SignLanguageTranslationDataError(
                        f"sample {record.sample_id} modality {modality} SHA-256 mismatch: "
                        f"expected {expected}, got {observed}"
                    )

    def load_sample(self, index: int) -> SignLanguageTranslationSample:
        record = self.records[index]
        pose = cast(
            NDArray[np.float32],
            _load_array(
                record.pose_path,
                expected_shape=(
                    record.frame_count,
                    2,
                    record.joint_count,
                    record.coordinate_dim,
                ),
                expected_dtype=np.dtype(np.float32),
                sample_id=record.sample_id,
                modality=TRANSLATION_POSE_MODALITY,
            ),
        )
        confidence = cast(
            NDArray[np.float32],
            _load_array(
                record.pose_confidence_path,
                expected_shape=(record.frame_count, 2, record.joint_count),
                expected_dtype=np.dtype(np.float32),
                sample_id=record.sample_id,
                modality=TRANSLATION_POSE_CONFIDENCE_MODALITY,
            ),
        )
        radar_feature = cast(
            NDArray[np.float32],
            _load_array(
                record.radar_feature_path,
                expected_shape=(record.frame_count, record.radar_feature_dim),
                expected_dtype=np.dtype(np.float32),
                sample_id=record.sample_id,
                modality=TRANSLATION_RADAR_FEATURE_MODALITY,
            ),
        )
        if not all(
            bool(np.all(np.isfinite(value))) for value in (pose, confidence, radar_feature)
        ):
            raise SignLanguageTranslationDataError(
                f"sample {record.sample_id} arrays must contain only finite values"
            )
        if bool(np.any((confidence < 0) | (confidence > 1))):
            raise SignLanguageTranslationDataError(
                f"sample {record.sample_id} pose confidence must be within [0,1]"
            )
        if record.frame_mask_path is None:
            frame_mask = np.ones(record.frame_count, dtype=np.bool_)
        else:
            frame_mask = cast(
                NDArray[np.bool_],
                _load_array(
                    record.frame_mask_path,
                    expected_shape=(record.frame_count,),
                    expected_dtype=np.dtype(np.bool_),
                    sample_id=record.sample_id,
                    modality=TRANSLATION_FRAME_MASK_MODALITY,
                ),
            )
        if not bool(np.any(frame_mask)):
            raise SignLanguageTranslationDataError(
                f"sample {record.sample_id} must contain at least one valid frame"
            )
        return SignLanguageTranslationSample(
            sample_id=record.sample_id,
            pose=np.asarray(pose, dtype=np.float32),
            pose_confidence=np.asarray(confidence, dtype=np.float32),
            radar_feature=np.asarray(radar_feature, dtype=np.float32),
            frame_mask=np.asarray(frame_mask, dtype=np.bool_),
            caption=record.caption,
            coordinate_frame=record.coordinate_frame,
        )


def collate_sign_language_translation_samples(
    samples: Sequence[SignLanguageTranslationSample],
    *,
    max_frames: int,
) -> SignLanguageTranslationBatch:
    if not samples:
        raise SignLanguageTranslationDataError("cannot collate an empty translation batch")
    if max_frames < 1:
        raise SignLanguageTranslationDataError("max_frames must be positive")
    pose_shapes = {sample.pose.shape[1:] for sample in samples}
    feature_dims = {sample.radar_feature.shape[1] for sample in samples}
    coordinate_frames = {sample.coordinate_frame for sample in samples}
    if len(pose_shapes) != 1 or len(feature_dims) != 1 or len(coordinate_frames) != 1:
        raise SignLanguageTranslationDataError(
            "one translation batch requires aligned pose, feature, and coordinate contracts"
        )
    batch_frames = max(sample.pose.shape[0] for sample in samples)
    if batch_frames > max_frames:
        raise SignLanguageTranslationDataError(
            f"translation batch contains {batch_frames} frames, model maximum is {max_frames}"
        )
    pose_shape = next(iter(pose_shapes))
    feature_dim = next(iter(feature_dims))
    pose = np.zeros((len(samples), batch_frames, *pose_shape), dtype=np.float32)
    confidence = np.zeros(
        (len(samples), batch_frames, pose_shape[0], pose_shape[1]), dtype=np.float32
    )
    radar_feature = np.zeros(
        (len(samples), batch_frames, feature_dim), dtype=np.float32
    )
    frame_mask = np.zeros((len(samples), batch_frames), dtype=np.bool_)
    for index, sample in enumerate(samples):
        frames = sample.pose.shape[0]
        pose[index, :frames] = sample.pose
        confidence[index, :frames] = sample.pose_confidence
        radar_feature[index, :frames] = sample.radar_feature
        frame_mask[index, :frames] = sample.frame_mask
        pose[index, :frames][~sample.frame_mask] = 0
        confidence[index, :frames][~sample.frame_mask] = 0
        radar_feature[index, :frames][~sample.frame_mask] = 0
    return SignLanguageTranslationBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        pose=pose,
        pose_confidence=confidence,
        radar_feature=radar_feature,
        frame_mask=frame_mask,
        captions=tuple(sample.caption for sample in samples),
        coordinate_frame=next(iter(coordinate_frames)),
    )
