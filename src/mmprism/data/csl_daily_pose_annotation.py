"""Per-frame RTMW3D pose annotation for CSL-Daily sentence image sequences.

Faithful rebuild of the legacy ``run_csl_daily_annotation.py`` pass (forensic
reference only, never imported): each sequence directory holds one image per
frame; the canonical v2 worker currently uses independently inferred frames
(``batch=1``). A separate benchmark-only path can compare MMPose frame batches
without changing the canonical output lineage. Keypoints scoring below the
confidence threshold become NaN, depth
is re-centered on the sequence mean z of body joints 6/7, and the native 133
keypoints reduce to the ``[T, 2, 24, 3]`` dual-arm/hand layout via
:data:`~mmprism.data.csl_news_annotation.LEFT_JOINT_INDICES` /
:data:`~mmprism.data.csl_news_annotation.RIGHT_JOINT_INDICES` (3 arm joints +
21 hand joints per side). Sequences with any NaN arm coordinate or an
all-NaN hand are skipped with a structured QC reason, never written.

Outputs under the configured output root:

- ``poses/<sequence_id>.npy``: float32 ``[T, 2, 24, 3]``, NaN-masked.
- ``poses/<sequence_id>.json``: per-sequence sidecar (QC status, NaN stats,
  depth center, model/estimator identity, artifact checksum).
- ``pose_manifest.jsonl``: strict rows for
  ``mmprism.data.csl_daily_simulation_run.load_pose_manifest`` (only
  ``sample_id``/``pose_uri``/``pose_sha256``/``sequence_id``/``subject_id``;
  the loader rejects any other key). Rebuilt atomically from completed
  sidecars at the end of every run.
- ``pose_qc.jsonl``: rich per-sequence QC rows (frame counts, NaN stats,
  model identity) for every completed or skipped sequence.
- ``runs/run_<timestamp>_<pid>.json``: run record.

Restart semantics: a sequence whose sidecar matches the current config
fingerprint (and, for completed sequences, whose ``.npy`` still matches the
recorded size/SHA-256) is never reprocessed. An existing artifact without a
matching sidecar is a conflict and is never overwritten.

This module is a pure library: it reads no environment variables, does no
logging, and parses no CLI arguments. The pose estimator is injected; the
optional MMPose adapter keeps torch/mmpose imports inside its constructor so
this module imports cleanly without the annotation extra.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from mmprism.artifacts.run import sha256_file
from mmprism.data.csl_news_annotation import (
    LEFT_JOINT_INDICES,
    RIGHT_JOINT_INDICES,
    AnnotationModelConfig,
    MMPoseRtmw3dEstimator,
    _require_model_assets,
)

ANNOTATION_SCHEMA_VERSION = "mmprism.csl_daily_pose_annotation.v1"
ANNOTATION_V2_SCHEMA_VERSION = "mmprism.csl_daily_pose_annotation.v2"
SAMPLE_SCHEMA_VERSION = "mmprism.csl_daily_pose_sample.v1"
SAMPLE_V2_SCHEMA_VERSION = "mmprism.csl_daily_pose_sample.v2"
RUN_RECORD_SCHEMA = "mmprism.csl_daily_pose_annotation_run.v1"

FRAME_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SEQUENCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SUBJECT_ID_PATTERN = re.compile(r"S\d+_(P\d+)_T\d+")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

QC_REASON_LEFT_ARM_NAN = "left_arm_nan"
QC_REASON_RIGHT_ARM_NAN = "right_arm_nan"
QC_REASON_LEFT_HAND_ALL_NAN = "left_hand_all_nan"
QC_REASON_RIGHT_HAND_ALL_NAN = "right_hand_all_nan"


class CslDailyPoseAnnotationError(RuntimeError):
    """Raised when CSL-Daily pose annotation cannot continue safely."""


class CslDailyPoseAnnotationConflictError(CslDailyPoseAnnotationError):
    """Raised when existing derived output must be preserved for review."""


@dataclass(frozen=True)
class CslDailyPoseSourceConfig:
    sequence_root: Path
    source_id: str
    # An optional full source receipt may be attached for a later release, but
    # the research annotation queue intentionally does not wait for one.
    receipt_path: Path | None = None
    receipt_sha256: str | None = None


@dataclass(frozen=True)
class CslDailyPoseTransformConfig:
    confidence_threshold: float
    minimum_valid_joints_per_frame: int = 0
    minimum_valid_frame_ratio: float = 0.0


@dataclass(frozen=True)
class CslDailyPoseRuntimeConfig:
    output_root: Path
    inference_batch_size: int


@dataclass(frozen=True)
class CslDailyPoseAnnotationConfig:
    schema_version: str
    source: CslDailyPoseSourceConfig
    model: AnnotationModelConfig
    transform: CslDailyPoseTransformConfig
    runtime: CslDailyPoseRuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "sequence_root": str(self.source.sequence_root),
                "source_id": self.source.source_id,
                "receipt_path": (
                    str(self.source.receipt_path) if self.source.receipt_path is not None else None
                ),
                "receipt_sha256": self.source.receipt_sha256,
            },
            "model": {
                "mmpose_root": str(self.model.mmpose_root),
                "mmpose_commit": self.model.mmpose_commit,
                "project_dir": str(self.model.project_dir),
                "config_path": str(self.model.config_path),
                "checkpoint_path": str(self.model.checkpoint_path),
                "checkpoint_sha256": self.model.checkpoint_sha256,
                "device": self.model.device,
            },
            "transform": {
                "confidence_threshold": self.transform.confidence_threshold,
                "minimum_valid_joints_per_frame": self.transform.minimum_valid_joints_per_frame,
                "minimum_valid_frame_ratio": self.transform.minimum_valid_frame_ratio,
            },
            "runtime": {
                "output_root": str(self.runtime.output_root),
                "inference_batch_size": self.runtime.inference_batch_size,
            },
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FramePosePrediction:
    """Native RTMW3D whole-body prediction for one frame."""

    keypoints: np.ndarray  # [133, 3] float32
    scores: np.ndarray  # [133] float32


class PoseEstimator(Protocol):
    """Injected per-frame estimator; heavy dependencies live behind this seam."""

    def estimate_frame(self, image_path: Path) -> FramePosePrediction: ...

    def runtime_metadata(self) -> Mapping[str, Any]: ...


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CslDailyPoseAnnotationError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise CslDailyPoseAnnotationError(f"unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CslDailyPoseAnnotationError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _number(payload: Mapping[str, Any], key: str, location: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CslDailyPoseAnnotationError(f"{location}.{key} must be numeric")
    return float(value)


def _path(payload: Mapping[str, Any], key: str, location: str, project_root: Path) -> Path:
    value = Path(_text(payload, key, location)).expanduser()
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def _expand_variables(value: Any, variables: Mapping[str, str]) -> Any:
    """Expand ``${NAME}``/``${NAME:-default}`` placeholders from an explicit mapping."""
    if isinstance(value, Mapping):
        return {key: _expand_variables(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_variables(item, variables) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        if name in variables:
            return variables[name]
        if default is not None:
            return default
        raise CslDailyPoseAnnotationError(
            f"configuration placeholder {name} has no supplied value"
        )

    return _VARIABLE_PATTERN.sub(replace, value)


def load_csl_daily_pose_annotation_config(
    path: str | Path,
    project_root: str | Path,
    *,
    variables: Mapping[str, str],
) -> CslDailyPoseAnnotationConfig:
    """Load and strictly validate the annotation configuration.

    ``${NAME}`` placeholders (for example ``${MMPRISM_DATA_ROOT}``) are
    expanded from the explicit ``variables`` mapping supplied by the caller;
    relative paths resolve against ``project_root``.
    """

    config_path = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CslDailyPoseAnnotationError(
            f"Unable to load annotation config: {error}"
        ) from error
    payload = _mapping(_expand_variables(raw, variables), "root")
    _reject_unknown(payload, {"schema_version", "source", "model", "transform", "runtime"}, "root")
    schema_version = payload.get("schema_version")
    if schema_version not in {ANNOTATION_SCHEMA_VERSION, ANNOTATION_V2_SCHEMA_VERSION}:
        raise CslDailyPoseAnnotationError(
            "schema_version must be "
            f"{ANNOTATION_SCHEMA_VERSION} or {ANNOTATION_V2_SCHEMA_VERSION}"
        )

    source = _mapping(payload.get("source"), "source")
    _reject_unknown(
        source,
        {"sequence_root", "source_id", "receipt_path", "receipt_sha256"},
        "source",
    )
    model = _mapping(payload.get("model"), "model")
    _reject_unknown(
        model,
        {
            "mmpose_root",
            "mmpose_commit",
            "project_dir",
            "config_path",
            "checkpoint_path",
            "checkpoint_sha256",
            "device",
        },
        "model",
    )
    transform = _mapping(payload.get("transform"), "transform")
    _reject_unknown(
        transform,
        {
            "confidence_threshold",
            "minimum_valid_joints_per_frame",
            "minimum_valid_frame_ratio",
        },
        "transform",
    )
    runtime = _mapping(payload.get("runtime"), "runtime")
    _reject_unknown(runtime, {"output_root", "inference_batch_size"}, "runtime")

    threshold = _number(transform, "confidence_threshold", "transform")
    if not 0.0 <= threshold <= 1.0:
        raise CslDailyPoseAnnotationError(
            "transform.confidence_threshold must be in [0, 1]"
        )
    minimum_valid_joints_per_frame = transform.get("minimum_valid_joints_per_frame", 0)
    if (
        isinstance(minimum_valid_joints_per_frame, bool)
        or not isinstance(minimum_valid_joints_per_frame, int)
        or not 0 <= minimum_valid_joints_per_frame <= 48
    ):
        raise CslDailyPoseAnnotationError(
            "transform.minimum_valid_joints_per_frame must be an integer in [0, 48]"
        )
    minimum_valid_frame_ratio = _number(
        {"minimum_valid_frame_ratio": transform.get("minimum_valid_frame_ratio", 0.0)},
        "minimum_valid_frame_ratio",
        "transform",
    )
    if not 0.0 <= minimum_valid_frame_ratio <= 1.0:
        raise CslDailyPoseAnnotationError(
            "transform.minimum_valid_frame_ratio must be in [0, 1]"
        )
    if schema_version == ANNOTATION_SCHEMA_VERSION and (
        minimum_valid_joints_per_frame != 0 or minimum_valid_frame_ratio != 0.0
    ):
        raise CslDailyPoseAnnotationError(
            "v1 annotation configuration cannot declare v2 validity thresholds"
        )
    checksum = _text(model, "checkpoint_sha256", "model").lower()
    if not SHA256_PATTERN.fullmatch(checksum):
        raise CslDailyPoseAnnotationError(
            "model.checkpoint_sha256 must contain 64 hex characters"
        )
    receipt_path: Path | None = None
    receipt_sha256: str | None = None
    declared_receipt_path = source.get("receipt_path")
    declared_receipt_sha256 = source.get("receipt_sha256")
    if declared_receipt_path is not None or declared_receipt_sha256 is not None:
        if declared_receipt_path is None or declared_receipt_sha256 is None:
            raise CslDailyPoseAnnotationError(
                "source receipt_path and receipt_sha256 must be supplied together"
            )
        receipt_path = _path(source, "receipt_path", "source", root)
        receipt_sha256 = _text(source, "receipt_sha256", "source").lower()
        if not SHA256_PATTERN.fullmatch(receipt_sha256):
            raise CslDailyPoseAnnotationError(
                "source.receipt_sha256 must contain 64 hex characters"
            )
        if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha256:
            raise CslDailyPoseAnnotationError(
                "source receipt does not exist or its SHA-256 does not match configuration"
            )
    batch_size = runtime.get("inference_batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1:
        raise CslDailyPoseAnnotationError(
            "runtime.inference_batch_size is fixed at 1: larger RTMW3D batches are "
            "not numerically equivalent in this artifact lineage"
        )

    return CslDailyPoseAnnotationConfig(
        schema_version=schema_version,
        source=CslDailyPoseSourceConfig(
            sequence_root=_path(source, "sequence_root", "source", root),
            source_id=_text(source, "source_id", "source"),
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
        ),
        model=AnnotationModelConfig(
            mmpose_root=_path(model, "mmpose_root", "model", root),
            mmpose_commit=_text(model, "mmpose_commit", "model"),
            project_dir=_path(model, "project_dir", "model", root),
            config_path=_path(model, "config_path", "model", root),
            checkpoint_path=_path(model, "checkpoint_path", "model", root),
            checkpoint_sha256=checksum,
            device=_text(model, "device", "model"),
        ),
        transform=CslDailyPoseTransformConfig(
            confidence_threshold=threshold,
            minimum_valid_joints_per_frame=minimum_valid_joints_per_frame,
            minimum_valid_frame_ratio=minimum_valid_frame_ratio,
        ),
        runtime=CslDailyPoseRuntimeConfig(
            output_root=_path(runtime, "output_root", "runtime", root),
            inference_batch_size=batch_size,
        ),
    )


def subject_id_for_sequence(sequence_id: str) -> str | None:
    """Extract the signer id (``P0004``) from a CSL-Daily sequence id."""
    match = SUBJECT_ID_PATTERN.fullmatch(sequence_id)
    return match.group(1) if match is not None else None


def discover_sequences(sequence_root: str | Path) -> list[Path]:
    """Return all sequence directories under the root, sorted by sequence id."""
    root = Path(sequence_root)
    if not root.is_dir():
        raise CslDailyPoseAnnotationError(f"sequence root does not exist: {root}")
    sequences = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    for sequence in sequences:
        if not SEQUENCE_ID_PATTERN.fullmatch(sequence.name):
            raise CslDailyPoseAnnotationError(
                f"sequence id {sequence.name!r} must match {SEQUENCE_ID_PATTERN.pattern}"
            )
    return sequences


def list_sequence_frames(sequence_dir: str | Path) -> list[Path]:
    """Sorted frame files (jpg/jpeg/png, case-insensitive suffix) of one sequence."""
    directory = Path(sequence_dir)
    frames = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    ]
    return sorted(frames, key=lambda p: p.name)


@dataclass(frozen=True)
class SequenceReduction:
    """Result of the legacy masking/centering/reduction chain for one sequence."""

    pose: np.ndarray  # [T, 2, 24, 3] float32, NaN-masked
    depth_center: float
    nan_count: int
    qc_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SequenceAnnotationV2:
    """Contract-complete pose product for one CSL-Daily image sequence."""

    native_keypoints_3d: np.ndarray  # [T, 133, 3] float32
    native_keypoint_scores: np.ndarray  # [T, 133] float32
    canonical_pose: np.ndarray  # [T, 2, 24, 3] finite float32
    canonical_confidence: np.ndarray  # [T, 2, 24] float32 in [0, 1]
    canonical_valid: np.ndarray  # [T, 2, 24] bool
    canonical_imputed: np.ndarray  # [T, 2, 24] bool
    frame_mask: np.ndarray  # [T] bool; all true before collation padding
    depth_center: float
    invalid_joint_count: int
    valid_frame_ratio: float
    qc_reasons: tuple[str, ...]


QC_REASON_NONFINITE_NATIVE = "nonfinite_native_prediction"
QC_REASON_DEPTH_CENTER_UNAVAILABLE = "depth_center_unavailable"
QC_REASON_INSUFFICIENT_VALID_FRAMES = "insufficient_valid_frames"
QC_REASON_LEFT_HAND_ALL_INVALID = "left_hand_all_invalid"
QC_REASON_RIGHT_HAND_ALL_INVALID = "right_hand_all_invalid"


def _temporally_fill_invalid_joints(
    canonical: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill invalid sequence positions by deterministic linear/edge interpolation."""

    filled = canonical.copy()
    imputed = ~valid.copy()
    time_index = np.arange(canonical.shape[0], dtype=np.float32)
    for side in range(canonical.shape[1]):
        for joint in range(canonical.shape[2]):
            observed = valid[:, side, joint]
            observed_indices = time_index[observed]
            if not observed_indices.size:
                filled[:, side, joint] = 0.0
                continue
            for axis in range(3):
                values = canonical[observed, side, joint, axis]
                filled[:, side, joint, axis] = np.interp(
                    time_index, observed_indices, values
                ).astype(np.float32, copy=False)
    return filled, imputed


def build_sequence_annotation_v2(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    confidence_threshold: float,
    minimum_valid_joints_per_frame: int,
    minimum_valid_frame_ratio: float,
) -> SequenceAnnotationV2:
    """Produce a finite canonical target while retaining source reliability.

    Invalid joints are explicitly represented by ``canonical_valid`` and their
    confidence values. The persisted canonical tensor is filled with zero only
    where invalid, so no NaN can enter simulation or training by accident.
    Native estimator output remains separately preserved for diagnosis.
    """

    native = np.asarray(keypoints, dtype=np.float32)
    native_scores = np.asarray(scores, dtype=np.float32)
    if native.ndim != 3 or native.shape[1:] != (133, 3):
        raise CslDailyPoseAnnotationError(
            f"native keypoints must have shape [T, 133, 3], got {native.shape}"
        )
    if native_scores.shape != native.shape[:2]:
        raise CslDailyPoseAnnotationError(
            f"native scores must have shape [T, 133], got {native_scores.shape}"
        )
    if native.shape[0] == 0:
        raise CslDailyPoseAnnotationError("cannot annotate an empty sequence")

    finite_native = np.isfinite(native).all(axis=-1)
    finite_scores = np.isfinite(native_scores)
    clean_native = np.where(np.isfinite(native), native, 0.0).astype(np.float32, copy=False)
    clean_scores = np.clip(
        np.where(finite_scores, native_scores, 0.0), 0.0, 1.0
    ).astype(np.float32, copy=False)
    shoulder_depth = clean_native[:, [6, 7], 2]
    shoulder_valid = finite_native[:, [6, 7]] & finite_scores[:, [6, 7]]
    valid_depths = shoulder_depth[shoulder_valid]
    depth_center = float(valid_depths.mean(dtype=np.float64)) if valid_depths.size else 0.0
    centered = clean_native.copy()
    centered[..., 2] -= np.float32(depth_center)
    canonical = np.stack(
        (centered[:, LEFT_JOINT_INDICES], centered[:, RIGHT_JOINT_INDICES]), axis=1
    ).astype(np.float32, copy=False)
    confidence = np.stack(
        (clean_scores[:, LEFT_JOINT_INDICES], clean_scores[:, RIGHT_JOINT_INDICES]), axis=1
    ).astype(np.float32, copy=False)
    native_selected_valid = np.stack(
        (finite_native[:, LEFT_JOINT_INDICES], finite_native[:, RIGHT_JOINT_INDICES]), axis=1
    ) & np.stack(
        (finite_scores[:, LEFT_JOINT_INDICES], finite_scores[:, RIGHT_JOINT_INDICES]), axis=1
    )
    valid = native_selected_valid & (confidence >= confidence_threshold)
    canonical, imputed = _temporally_fill_invalid_joints(canonical, valid)
    frame_mask = np.ones(native.shape[0], dtype=np.bool_)
    valid_frame = valid.sum(axis=(1, 2)) >= minimum_valid_joints_per_frame
    valid_frame_ratio = float(valid_frame.mean())

    reasons: list[str] = []
    if not bool(np.isfinite(native).all() and np.isfinite(native_scores).all()):
        reasons.append(QC_REASON_NONFINITE_NATIVE)
    if not valid_depths.size:
        reasons.append(QC_REASON_DEPTH_CENTER_UNAVAILABLE)
    if valid_frame_ratio < minimum_valid_frame_ratio:
        reasons.append(QC_REASON_INSUFFICIENT_VALID_FRAMES)
    if not bool(valid[:, 0, 3:].any()):
        reasons.append(QC_REASON_LEFT_HAND_ALL_INVALID)
    if not bool(valid[:, 1, 3:].any()):
        reasons.append(QC_REASON_RIGHT_HAND_ALL_INVALID)
    return SequenceAnnotationV2(
        native_keypoints_3d=native.copy(),
        native_keypoint_scores=native_scores.copy(),
        canonical_pose=canonical,
        canonical_confidence=confidence,
        canonical_valid=valid,
        canonical_imputed=imputed,
        frame_mask=frame_mask,
        depth_center=depth_center,
        invalid_joint_count=int((~valid).sum()),
        valid_frame_ratio=valid_frame_ratio,
        qc_reasons=tuple(reasons),
    )


def reduce_sequence_poses(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    confidence_threshold: float,
) -> SequenceReduction:
    """Apply the legacy per-sequence chain without mutating the inputs.

    Mirrors ``run_csl_daily_annotation.py``: scores below the threshold set
    the whole keypoint to NaN; the depth center is the plain (NaN-propagating)
    mean z of body joints 6/7 over the whole sequence; the 133 native
    keypoints reduce to ``[T, 2, 24, 3]`` (3 arm + 21 hand joints per side).
    """

    native = np.asarray(keypoints, dtype=np.float32)
    native_scores = np.asarray(scores, dtype=np.float32)
    if native.ndim != 3 or native.shape[1:] != (133, 3):
        raise CslDailyPoseAnnotationError(
            f"native keypoints must have shape [T, 133, 3], got {native.shape}"
        )
    if native_scores.shape != native.shape[:2]:
        raise CslDailyPoseAnnotationError(
            f"native scores must have shape [T, 133], got {native_scores.shape}"
        )
    if native.shape[0] == 0:
        raise CslDailyPoseAnnotationError("cannot reduce an empty sequence")

    masked = native.copy()
    masked[native_scores < confidence_threshold] = np.nan
    # Legacy: depths_center = keypoints_all[:, [6, 7], 2].mean() — a plain
    # mean, so a NaN at joints 6/7 propagates and the sequence fails QC below.
    depth_center = float(masked[:, [6, 7], 2].mean())
    masked[..., 2] -= depth_center
    pose = np.stack(
        (masked[:, LEFT_JOINT_INDICES], masked[:, RIGHT_JOINT_INDICES]), axis=1
    ).astype(np.float32, copy=False)

    reasons: list[str] = []
    if bool(np.isnan(pose[:, 0, :3]).any()):
        reasons.append(QC_REASON_LEFT_ARM_NAN)
    if bool(np.isnan(pose[:, 1, :3]).any()):
        reasons.append(QC_REASON_RIGHT_ARM_NAN)
    if bool(np.isnan(pose[:, 0, 3:]).all()):
        reasons.append(QC_REASON_LEFT_HAND_ALL_NAN)
    if bool(np.isnan(pose[:, 1, 3:]).all()):
        reasons.append(QC_REASON_RIGHT_HAND_ALL_NAN)
    return SequenceReduction(
        pose=pose,
        depth_center=depth_center,
        nan_count=int(np.isnan(pose).sum()),
        qc_reasons=tuple(reasons),
    )


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_npy_atomic(array: np.ndarray, path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CslDailyPoseAnnotationConflictError(
            f"refusing to overwrite existing pose artifact: {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    expected_size = temporary.stat().st_size
    if expected_size <= 0:
        raise CslDailyPoseAnnotationError(f"temporary pose artifact is empty: {temporary}")
    expected_sha256 = sha256_file(temporary)
    if path.exists():
        raise CslDailyPoseAnnotationConflictError(
            f"pose artifact appeared before atomic promotion: {path}"
        )
    temporary.replace(path)
    return expected_size, expected_sha256


def _write_npz_atomic(arrays: Mapping[str, np.ndarray], path: Path) -> tuple[int, str]:
    """Write one no-clobber compressed audit payload and return its identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CslDailyPoseAnnotationConflictError(
            f"refusing to overwrite existing audit artifact: {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
        stream.flush()
        os.fsync(stream.fileno())
    expected_size = temporary.stat().st_size
    if expected_size <= 0:
        raise CslDailyPoseAnnotationError(f"temporary audit artifact is empty: {temporary}")
    expected_sha256 = sha256_file(temporary)
    if path.exists():
        raise CslDailyPoseAnnotationConflictError(
            f"audit artifact appeared before atomic promotion: {path}"
        )
    temporary.replace(path)
    return expected_size, expected_sha256


def _pose_paths(config: CslDailyPoseAnnotationConfig, sequence_id: str) -> tuple[Path, Path]:
    pose_root = config.runtime.output_root / "poses"
    return pose_root / f"{sequence_id}.npy", pose_root / f"{sequence_id}.json"


def _audit_path(config: CslDailyPoseAnnotationConfig, sequence_id: str) -> Path:
    return config.runtime.output_root / "samples" / f"{sequence_id}.npz"


def validate_annotation_v2_audit(
    audit_path: str | Path, *, pose_path: str | Path | None = None
) -> bool:
    """Validate the v2 audit payload and, when present, its canonical pose view."""

    required = {
        "native_keypoints_3d",
        "native_keypoint_scores",
        "canonical_pose",
        "canonical_confidence",
        "canonical_valid",
        "canonical_imputed",
        "frame_mask",
    }
    try:
        with np.load(audit_path, allow_pickle=False) as arrays:
            if set(arrays.files) != required:
                return False
            native = arrays["native_keypoints_3d"]
            native_scores = arrays["native_keypoint_scores"]
            canonical = arrays["canonical_pose"]
            confidence = arrays["canonical_confidence"]
            valid = arrays["canonical_valid"]
            imputed = arrays["canonical_imputed"]
            frame_mask = arrays["frame_mask"]
            frames = native.shape[0] if native.ndim == 3 else 0
            if not (
                frames > 0
                and native.shape == (frames, 133, 3)
                and native.dtype == np.float32
                and native_scores.shape == (frames, 133)
                and native_scores.dtype == np.float32
                and canonical.shape == (frames, 2, 24, 3)
                and canonical.dtype == np.float32
                and confidence.shape == (frames, 2, 24)
                and confidence.dtype == np.float32
                and valid.shape == (frames, 2, 24)
                and valid.dtype == np.bool_
                and imputed.shape == (frames, 2, 24)
                and imputed.dtype == np.bool_
                and frame_mask.shape == (frames,)
                and frame_mask.dtype == np.bool_
                and bool(frame_mask.all())
                and bool(np.isfinite(canonical).all())
                and bool(np.isfinite(confidence).all())
                and bool(((confidence >= 0.0) & (confidence <= 1.0)).all())
                and bool(np.array_equal(imputed, ~valid))
            ):
                return False
            if pose_path is not None:
                pose = np.load(pose_path, allow_pickle=False)
                if pose.shape != canonical.shape or pose.dtype != np.float32:
                    return False
                if not np.array_equal(pose, canonical):
                    return False
    except (OSError, ValueError):
        return False
    return True


def _finished_sidecar_status(
    npy_path: Path, sidecar_path: Path, config: CslDailyPoseAnnotationConfig
) -> str | None:
    """Return ``completed``/``skipped`` for a valid finished record, else None."""
    if not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(sidecar, dict):
        return None
    if sidecar.get("config_fingerprint") != config.fingerprint:
        return None
    expected_sample_schema = (
        SAMPLE_V2_SCHEMA_VERSION
        if config.schema_version == ANNOTATION_V2_SCHEMA_VERSION
        else SAMPLE_SCHEMA_VERSION
    )
    if sidecar.get("schema_version") != expected_sample_schema:
        return None
    status = sidecar.get("status")
    audit_artifact = sidecar.get("audit_artifact")
    if config.schema_version == ANNOTATION_V2_SCHEMA_VERSION:
        audit_path = _audit_path(config, str(sidecar.get("sequence_id", "")))
        if not isinstance(audit_artifact, Mapping) or not audit_path.is_file():
            return None
        if audit_artifact.get("size_bytes") != audit_path.stat().st_size:
            return None
        if audit_artifact.get("sha256") != sha256_file(audit_path):
            return None
        if not validate_annotation_v2_audit(
            audit_path, pose_path=npy_path if status == "completed" else None
        ):
            return None
    if status == "skipped":
        qc = sidecar.get("qc")
        reasons = qc.get("reasons") if isinstance(qc, dict) else None
        return "skipped" if isinstance(reasons, list) and reasons else None
    if status != "completed":
        return None
    artifact = sidecar.get("artifact")
    if not isinstance(artifact, dict) or not npy_path.is_file():
        return None
    if artifact.get("size_bytes") != npy_path.stat().st_size:
        return None
    if artifact.get("sha256") != sha256_file(npy_path):
        return None
    return "completed"


def _model_identity(config: CslDailyPoseAnnotationConfig) -> dict[str, Any]:
    return {
        "mmpose_commit": config.model.mmpose_commit,
        "config_path": str(config.model.config_path),
        "checkpoint_path": str(config.model.checkpoint_path),
        "checkpoint_sha256": config.model.checkpoint_sha256,
        "device": config.model.device,
    }


def _annotate_sequence(
    config: CslDailyPoseAnnotationConfig,
    estimator: PoseEstimator,
    sequence_dir: Path,
) -> str:
    """Process one sequence; returns ``completed`` or ``skipped``.

    Writes the pose artifact and sidecar. Per-sequence unexpected errors
    propagate to the caller, which records them as failed outcomes.
    """

    sequence_id = sequence_dir.name
    npy_path, sidecar_path = _pose_paths(config, sequence_id)
    started = time.monotonic()

    frames = list_sequence_frames(sequence_dir)
    if not frames:
        raise CslDailyPoseAnnotationError(f"sequence {sequence_id} has no image frames")
    keypoints: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for frame_path in frames:
        prediction = estimator.estimate_frame(frame_path)
        keypoints.append(np.asarray(prediction.keypoints, dtype=np.float32))
        scores.append(np.asarray(prediction.scores, dtype=np.float32))
    native_keypoints = np.stack(keypoints)
    native_scores = np.stack(scores)
    is_v2 = config.schema_version == ANNOTATION_V2_SCHEMA_VERSION
    if is_v2:
        v2 = build_sequence_annotation_v2(
            native_keypoints,
            native_scores,
            confidence_threshold=config.transform.confidence_threshold,
            minimum_valid_joints_per_frame=config.transform.minimum_valid_joints_per_frame,
            minimum_valid_frame_ratio=config.transform.minimum_valid_frame_ratio,
        )
        pose = v2.canonical_pose
        depth_center = v2.depth_center
        nan_count = int(np.isnan(pose).sum())
        qc_reasons = v2.qc_reasons
    else:
        reduction = reduce_sequence_poses(
            native_keypoints,
            native_scores,
            confidence_threshold=config.transform.confidence_threshold,
        )
        pose = reduction.pose
        depth_center = reduction.depth_center
        nan_count = reduction.nan_count
        qc_reasons = reduction.qc_reasons

    qc_status = "passed" if not qc_reasons else "skipped"
    artifact: dict[str, Any] | None = None
    audit_artifact: dict[str, Any] | None = None
    if is_v2:
        audit_path = _audit_path(config, sequence_id)
        audit_size, audit_checksum = _write_npz_atomic(
            {
                "native_keypoints_3d": v2.native_keypoints_3d,
                "native_keypoint_scores": v2.native_keypoint_scores,
                "canonical_pose": v2.canonical_pose,
                "canonical_confidence": v2.canonical_confidence,
                "canonical_valid": v2.canonical_valid,
                "canonical_imputed": v2.canonical_imputed,
                "frame_mask": v2.frame_mask,
            },
            audit_path,
        )
        audit_artifact = {
            "path": str(audit_path),
            "size_bytes": audit_size,
            "sha256": audit_checksum,
        }
    if qc_reasons:
        status = "skipped"
    else:
        try:
            size_bytes, checksum = _write_npy_atomic(pose, npy_path)
        except Exception:
            if audit_artifact is not None:
                _audit_path(config, sequence_id).unlink(missing_ok=True)
            raise
        artifact = {
            "path": str(npy_path),
            "size_bytes": size_bytes,
            "sha256": checksum,
        }
        status = "completed"

    sidecar: dict[str, Any] = {
        "schema_version": SAMPLE_V2_SCHEMA_VERSION if is_v2 else SAMPLE_SCHEMA_VERSION,
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_id": sequence_id,
        "sequence_id": sequence_id,
        "subject_id": subject_id_for_sequence(sequence_id),
        "config_fingerprint": config.fingerprint,
        "source": {
            "source_id": config.source.source_id,
            "sequence_root": str(config.source.sequence_root),
            "receipt_path": (
                str(config.source.receipt_path) if config.source.receipt_path is not None else None
            ),
            "receipt_sha256": config.source.receipt_sha256,
            "frame_count": len(frames),
            "frame_files": [path.name for path in frames],
        },
        "transform": {
            "confidence_threshold": config.transform.confidence_threshold,
            "masking_policy": (
                "score_below_threshold_or_nonfinite_to_temporal_interpolation_with_validity_mask"
                if is_v2
                else "score_below_threshold_to_nan"
            ),
            "canonical_fill_value": 0.0 if is_v2 else None,
            "depth_center_policy": "sequence_mean_native_z_joints_6_7",
            "depth_center": depth_center,
            "left_joint_indices": LEFT_JOINT_INDICES.tolist(),
            "right_joint_indices": RIGHT_JOINT_INDICES.tolist(),
        },
        "qc": {
            "status": qc_status,
            "reasons": list(qc_reasons),
            "nan_count": nan_count,
        },
        "model": _model_identity(config),
        "artifact": artifact,
        "audit_artifact": audit_artifact,
        "arrays": {
            "pose": [int(size) for size in pose.shape],
            "dtype": "float32",
            **(
                {
                    "native_keypoints_3d": [int(size) for size in v2.native_keypoints_3d.shape],
                    "native_keypoint_scores": [
                        int(size) for size in v2.native_keypoint_scores.shape
                    ],
                    "canonical_confidence": [int(size) for size in v2.canonical_confidence.shape],
                    "canonical_valid": [int(size) for size in v2.canonical_valid.shape],
                    "canonical_imputed": [int(size) for size in v2.canonical_imputed.shape],
                    "frame_mask": [int(size) for size in v2.frame_mask.shape],
                }
                if is_v2
                else {}
            ),
        },
        "validity": (
            {
                "minimum_valid_joints_per_frame": config.transform.minimum_valid_joints_per_frame,
                "minimum_valid_frame_ratio": config.transform.minimum_valid_frame_ratio,
                "valid_frame_ratio": v2.valid_frame_ratio,
                "invalid_joint_count": v2.invalid_joint_count,
                "imputation_policy": "per_joint_linear_time_interpolation_with_edge_extension",
            }
            if is_v2
            else None
        ),
        "runtime": dict(estimator.runtime_metadata()),
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_atomic(sidecar, sidecar_path)
    return status


def _manifest_row(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Strict row for ``load_pose_manifest``: no keys beyond its contract."""
    sequence_id = sidecar["sequence_id"]
    artifact = sidecar["artifact"]
    assert isinstance(artifact, Mapping)
    row: dict[str, Any] = {
        "sample_id": sidecar["sample_id"],
        "sequence_id": sequence_id,
        "pose_uri": f"poses/{sequence_id}.npy",
        "pose_sha256": artifact["sha256"],
    }
    subject_id = sidecar.get("subject_id")
    if isinstance(subject_id, str) and subject_id:
        row["subject_id"] = subject_id
    return row


def _qc_row(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    source = sidecar["source"]
    transform = sidecar["transform"]
    qc = sidecar["qc"]
    model = sidecar["model"]
    artifact = sidecar.get("artifact")
    assert isinstance(source, Mapping)
    assert isinstance(transform, Mapping)
    assert isinstance(qc, Mapping)
    assert isinstance(model, Mapping)
    return {
        "sample_id": sidecar["sample_id"],
        "sequence_id": sidecar["sequence_id"],
        "subject_id": sidecar.get("subject_id"),
        "status": sidecar["status"],
        "frame_count": source["frame_count"],
        "qc": {"status": qc["status"], "reasons": qc["reasons"], "nan_count": qc["nan_count"]},
        "depth_center": transform["depth_center"],
        "confidence_threshold": transform["confidence_threshold"],
        "model": {
            "mmpose_commit": model["mmpose_commit"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "device": model["device"],
        },
        "config_fingerprint": sidecar["config_fingerprint"],
        "pose_sha256": artifact.get("sha256") if isinstance(artifact, Mapping) else None,
    }


def _collect_finished_sidecars(
    config: CslDailyPoseAnnotationConfig,
) -> list[dict[str, Any]]:
    """Valid finished sidecars (completed or skipped), sorted by sequence id."""
    sidecars: list[dict[str, Any]] = []
    for sequence_dir in discover_sequences(config.source.sequence_root):
        npy_path, sidecar_path = _pose_paths(config, sequence_dir.name)
        if _finished_sidecar_status(npy_path, sidecar_path, config) is None:
            continue
        sidecars.append(json.loads(sidecar_path.read_text(encoding="utf-8")))
    return sidecars


def _rewrite_manifests(
    config: CslDailyPoseAnnotationConfig,
) -> tuple[Path, Path, int]:
    """Rebuild the pose manifest and QC report from finished sidecars."""
    sidecars = _collect_finished_sidecars(config)
    completed = [sidecar for sidecar in sidecars if sidecar.get("status") == "completed"]
    manifest_path = config.runtime.output_root / "pose_manifest.jsonl"
    manifest_payload = "".join(
        json.dumps(_manifest_row(sidecar), sort_keys=True) + "\n" for sidecar in completed
    )
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        stream.write(manifest_payload.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(manifest_path)
    qc_path = config.runtime.output_root / "pose_qc.jsonl"
    qc_temporary = qc_path.with_name(f".{qc_path.name}.tmp.{os.getpid()}")
    qc_payload = "".join(
        json.dumps(_qc_row(sidecar), sort_keys=True) + "\n" for sidecar in sidecars
    )
    with qc_temporary.open("xb") as stream:
        stream.write(qc_payload.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    qc_temporary.replace(qc_path)
    return manifest_path, qc_path, len(completed)


def _run_record(
    config: CslDailyPoseAnnotationConfig,
    estimator: PoseEstimator,
    outcomes: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path | None,
    qc_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = config.runtime.output_root / "runs" / f"run_{timestamp}_{os.getpid()}.json"
    payload: dict[str, Any] = {
        "schema_version": RUN_RECORD_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "model": _model_identity(config),
        "runtime": dict(estimator.runtime_metadata()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "counts": {
            "completed": sum(1 for o in outcomes if o["outcome"] == "completed"),
            "skipped_existing": sum(1 for o in outcomes if o["outcome"] == "skipped_existing"),
            "skipped_qc": sum(1 for o in outcomes if o["outcome"] == "skipped_qc"),
            "failed": sum(1 for o in outcomes if o["outcome"] == "failed"),
        },
        "sequences": list(outcomes),
        "outputs": {
            "pose_manifest_path": str(manifest_path) if manifest_path is not None else None,
            "pose_manifest_sha256": (
                sha256_file(manifest_path) if manifest_path is not None else None
            ),
            "pose_qc_path": str(qc_path) if qc_path is not None else None,
        },
    }
    _write_json_atomic(payload, path)
    return path, payload


def run_csl_daily_pose_annotation(
    config: CslDailyPoseAnnotationConfig,
    *,
    estimator: PoseEstimator,
    max_sequences: int | None = None,
    sequence_ids: Sequence[str] | None = None,
    rewrite_manifests: bool = True,
) -> dict[str, Any]:
    """Annotate discovered sequences, preserving restartable outputs.

    Per-sequence failures are recorded in the run record and never abort the
    run. Returns a JSON-serializable summary payload.
    """

    if max_sequences is not None and max_sequences < 1:
        raise CslDailyPoseAnnotationError("max_sequences must be positive")
    discovered = discover_sequences(config.source.sequence_root)
    by_id = {sequence.name: sequence for sequence in discovered}
    if sequence_ids is not None:
        missing = sorted(set(sequence_ids) - set(by_id))
        if missing:
            raise CslDailyPoseAnnotationError(
                f"requested sequence ids not found: {', '.join(missing)}"
            )
        selected = [by_id[sequence_id] for sequence_id in dict.fromkeys(sequence_ids)]
    else:
        selected = discovered

    outcomes: list[dict[str, Any]] = []
    processed = 0
    limit_reached = False
    for sequence_dir in selected:
        if max_sequences is not None and processed >= max_sequences:
            limit_reached = True
            break
        sequence_id = sequence_dir.name
        npy_path, sidecar_path = _pose_paths(config, sequence_id)
        finished = _finished_sidecar_status(npy_path, sidecar_path, config)
        if finished is not None:
            outcomes.append(
                {
                    "sequence_id": sequence_id,
                    "outcome": "skipped_existing" if finished == "completed" else "skipped_qc",
                }
            )
            continue
        if npy_path.exists() or sidecar_path.exists():
            raise CslDailyPoseAnnotationConflictError(
                "refusing to overwrite an existing incomplete or identity-mismatched "
                f"sequence output: {npy_path} / {sidecar_path}"
            )
        try:
            status = _annotate_sequence(config, estimator, sequence_dir)
        except CslDailyPoseAnnotationConflictError:
            raise
        except Exception as error:
            outcomes.append(
                {
                    "sequence_id": sequence_id,
                    "outcome": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        processed += 1
        outcomes.append(
            {
                "sequence_id": sequence_id,
                "outcome": "completed" if status == "completed" else "skipped_qc",
            }
        )

    if rewrite_manifests:
        manifest_path, qc_path, manifest_rows = _rewrite_manifests(config)
    else:
        manifest_path, qc_path, manifest_rows = None, None, None
    run_record_path, run_record = _run_record(
        config, estimator, outcomes, manifest_path=manifest_path, qc_path=qc_path
    )
    failed = sum(1 for outcome in outcomes if outcome["outcome"] == "failed")
    return {
        "schema_version": RUN_RECORD_SCHEMA,
        "status": "limit_reached" if limit_reached else (
            "completed" if failed == 0 else "completed_with_failures"
        ),
        "sequences_selected": len(selected),
        "processed": processed,
        "skipped_existing": run_record["counts"]["skipped_existing"],
        "skipped_qc": run_record["counts"]["skipped_qc"],
        "failed": failed,
        "manifest_rows": manifest_rows,
        "pose_manifest": str(manifest_path) if manifest_path is not None else None,
        "pose_qc": str(qc_path) if qc_path is not None else None,
        "run_record": str(run_record_path),
    }


def _normalize_prediction(value: Any, final_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > len(final_shape) and array.shape[0] == 1:
        array = array[0]
    if array.shape != final_shape:
        raise CslDailyPoseAnnotationError(
            f"{name} has shape {array.shape}, expected {final_shape}"
        )
    return array.astype(np.float32, copy=False)


class MMPoseRtmw3dFrameEstimator:
    """Optional-dependency RTMW3D adapter for single-person full-frame images.

    All heavy imports (torch, cv2, mmpose) happen inside the constructor so
    the enclosing module stays importable without the annotation extra.
    Model assets are verified against the pinned commit and checkpoint
    SHA-256 before the model loads.
    """

    def __init__(self, config: CslDailyPoseAnnotationConfig) -> None:
        self._config = config
        self._model_assets = _require_model_assets(config.model)
        project_path = str(config.model.project_dir)
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
        MMPoseRtmw3dEstimator._guard_unused_edpose_head()
        try:
            importlib.import_module("rtmpose3d")
            self._cv2 = importlib.import_module("cv2")
            apis = importlib.import_module("mmpose.apis")
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise CslDailyPoseAnnotationError(
                "Annotation dependency import failed after syncing the UV annotation "
                f"extra: {error}"
            ) from error
        self._torch = torch
        cpu_threads = int(os.environ.get("MMPRISM_CPU_THREADS", "4"))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            # PyTorch permits configuring inter-op threads only before its
            # first parallel operation; an already-established value is fine.
            if "cannot set number of interop threads" not in str(error):
                raise
        self._cv2.setNumThreads(1)
        self._inference_topdown = apis.inference_topdown
        # The pinned official RTMW3D checkpoint predates PyTorch's weights_only
        # default change. Mirror the CSL-News worker policy
        # (scripts/run_csl_news_annotation_worker.sh) for this trusted,
        # sha256-pinned asset instead of allowlisting pickle globals.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        self._model = apis.init_model(
            str(config.model.config_path),
            str(config.model.checkpoint_path),
            device=config.model.device,
        )
        self._compose = importlib.import_module("mmengine.dataset").Compose
        self._pseudo_collate = importlib.import_module("mmengine.dataset").pseudo_collate
        self._init_default_scope = importlib.import_module(
            "mmengine.registry"
        ).init_default_scope
        self._pipeline = self._compose(self._model.cfg.test_dataloader.dataset.pipeline)
        if config.model.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def estimate_frame(self, image_path: Path) -> FramePosePrediction:
        frame = self._cv2.imread(str(image_path))
        if frame is None:
            raise CslDailyPoseAnnotationError(f"undecodable image frame: {image_path}")
        height, width = frame.shape[:2]
        # Single-person full-frame inference, as in the legacy script; the
        # public batch=1 inference_topdown path is the numerically verified
        # lineage baseline.
        bbox = np.asarray([[0, 0, width, height]], dtype=np.float32)
        results = self._inference_topdown(self._model, frame, bbox)
        instances = results[0].pred_instances
        return FramePosePrediction(
            keypoints=_normalize_prediction(instances.keypoints, (133, 3), "keypoints"),
            scores=_normalize_prediction(
                instances.keypoint_scores, (133,), "keypoint_scores"
            ),
        )

    def estimate_frames_batched(
        self, image_paths: Sequence[Path], *, batch_size: int
    ) -> list[FramePosePrediction]:
        """Benchmark-only batched inference over a stable image-path order.

        This method deliberately does not alter the configured v2 worker
        behavior. It exists so a Slurm benchmark can measure and compare the
        private MMPose ``test_step`` batch path before any production decision.
        """

        if batch_size < 1:
            raise CslDailyPoseAnnotationError("batch_size must be positive")
        if not image_paths:
            raise CslDailyPoseAnnotationError("cannot infer an empty image-path batch")
        if batch_size == 1:
            return [self.estimate_frame(path) for path in image_paths]

        predictions: list[FramePosePrediction] = []
        scope = self._model.cfg.get("default_scope", "mmpose")
        if scope is not None:
            self._init_default_scope(scope)
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start : start + batch_size]
            data_list = []
            for image_path in paths:
                frame = self._cv2.imread(str(image_path))
                if frame is None:
                    raise CslDailyPoseAnnotationError(
                        f"undecodable image frame: {image_path}"
                    )
                height, width = frame.shape[:2]
                data_info: dict[str, Any] = {
                    "img": frame,
                    "bbox": np.asarray([[0, 0, width, height]], dtype=np.float32),
                    "bbox_score": np.ones(1, dtype=np.float32),
                }
                data_info.update(self._model.dataset_meta)
                data_list.append(self._pipeline(data_info))
            with self._torch.inference_mode():
                results = self._model.test_step(self._pseudo_collate(data_list))
            if len(results) != len(paths):
                raise CslDailyPoseAnnotationError(
                    f"expected {len(paths)} pose instances, received {len(results)}"
                )
            for result in results:
                instances = result.pred_instances
                predictions.append(
                    FramePosePrediction(
                        keypoints=_normalize_prediction(
                            instances.keypoints, (133, 3), "keypoints"
                        ),
                        scores=_normalize_prediction(
                            instances.keypoint_scores, (133,), "keypoint_scores"
                        ),
                    )
                )
        return predictions

    def runtime_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "device": self._config.model.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_threads": self._torch.get_num_threads(),
            "interop_threads": self._torch.get_num_interop_threads(),
            "opencv_threads": self._cv2.getNumThreads(),
            "inference_batch_size": 1,
            **self._model_assets,
        }
        if self._config.model.device.startswith("cuda"):
            metadata.update(
                {
                    "torch_version": self._torch.__version__,
                    "cuda_version": self._torch.version.cuda,
                    "gpu_name": self._torch.cuda.get_device_name(0),
                    "peak_memory_bytes": self._torch.cuda.max_memory_allocated(),
                }
            )
        return metadata


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "CslDailyPoseAnnotationConfig",
    "CslDailyPoseAnnotationConflictError",
    "CslDailyPoseAnnotationError",
    "FramePosePrediction",
    "MMPoseRtmw3dFrameEstimator",
    "PoseEstimator",
    "SequenceReduction",
    "discover_sequences",
    "list_sequence_frames",
    "load_csl_daily_pose_annotation_config",
    "reduce_sequence_poses",
    "run_csl_daily_pose_annotation",
    "subject_id_for_sequence",
]
