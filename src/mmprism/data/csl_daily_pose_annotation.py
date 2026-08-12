"""Per-frame RTMW3D pose annotation for CSL-Daily sentence image sequences.

Faithful rebuild of the legacy ``run_csl_daily_annotation.py`` pass (forensic
reference only, never imported): each sequence directory holds one image per
frame; every frame is estimated independently (inference batch fixed at 1 —
larger RTMW3D batches are not numerically equivalent in this artifact
lineage), keypoints scoring below the confidence threshold become NaN, depth
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
SAMPLE_SCHEMA_VERSION = "mmprism.csl_daily_pose_sample.v1"
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


@dataclass(frozen=True)
class CslDailyPoseTransformConfig:
    confidence_threshold: float


@dataclass(frozen=True)
class CslDailyPoseRuntimeConfig:
    output_root: Path
    inference_batch_size: int


@dataclass(frozen=True)
class CslDailyPoseAnnotationConfig:
    source: CslDailyPoseSourceConfig
    model: AnnotationModelConfig
    transform: CslDailyPoseTransformConfig
    runtime: CslDailyPoseRuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "source": {
                "sequence_root": str(self.source.sequence_root),
                "source_id": self.source.source_id,
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
    if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise CslDailyPoseAnnotationError(
            f"schema_version must be {ANNOTATION_SCHEMA_VERSION}"
        )

    source = _mapping(payload.get("source"), "source")
    _reject_unknown(source, {"sequence_root", "source_id"}, "source")
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
    _reject_unknown(transform, {"confidence_threshold"}, "transform")
    runtime = _mapping(payload.get("runtime"), "runtime")
    _reject_unknown(runtime, {"output_root", "inference_batch_size"}, "runtime")

    threshold = _number(transform, "confidence_threshold", "transform")
    if not 0.0 <= threshold <= 1.0:
        raise CslDailyPoseAnnotationError(
            "transform.confidence_threshold must be in [0, 1]"
        )
    checksum = _text(model, "checkpoint_sha256", "model").lower()
    if not SHA256_PATTERN.fullmatch(checksum):
        raise CslDailyPoseAnnotationError(
            "model.checkpoint_sha256 must contain 64 hex characters"
        )
    batch_size = runtime.get("inference_batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1:
        raise CslDailyPoseAnnotationError(
            "runtime.inference_batch_size is fixed at 1: larger RTMW3D batches are "
            "not numerically equivalent in this artifact lineage"
        )

    return CslDailyPoseAnnotationConfig(
        source=CslDailyPoseSourceConfig(
            sequence_root=_path(source, "sequence_root", "source", root),
            source_id=_text(source, "source_id", "source"),
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
        transform=CslDailyPoseTransformConfig(confidence_threshold=threshold),
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


def _pose_paths(config: CslDailyPoseAnnotationConfig, sequence_id: str) -> tuple[Path, Path]:
    pose_root = config.runtime.output_root / "poses"
    return pose_root / f"{sequence_id}.npy", pose_root / f"{sequence_id}.json"


def _finished_sidecar_status(
    npy_path: Path, sidecar_path: Path, config_fingerprint: str
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
    if sidecar.get("config_fingerprint") != config_fingerprint:
        return None
    status = sidecar.get("status")
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
    reduction = reduce_sequence_poses(
        np.stack(keypoints),
        np.stack(scores),
        confidence_threshold=config.transform.confidence_threshold,
    )

    qc_status = "passed" if not reduction.qc_reasons else "skipped"
    artifact: dict[str, Any] | None = None
    if reduction.qc_reasons:
        status = "skipped"
    else:
        size_bytes, checksum = _write_npy_atomic(reduction.pose, npy_path)
        artifact = {
            "path": str(npy_path),
            "size_bytes": size_bytes,
            "sha256": checksum,
        }
        status = "completed"

    sidecar: dict[str, Any] = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_id": sequence_id,
        "sequence_id": sequence_id,
        "subject_id": subject_id_for_sequence(sequence_id),
        "config_fingerprint": config.fingerprint,
        "source": {
            "source_id": config.source.source_id,
            "sequence_root": str(config.source.sequence_root),
            "frame_count": len(frames),
            "frame_files": [path.name for path in frames],
        },
        "transform": {
            "confidence_threshold": config.transform.confidence_threshold,
            "masking_policy": "score_below_threshold_to_nan",
            "depth_center_policy": "sequence_mean_native_z_joints_6_7",
            "depth_center": reduction.depth_center,
            "left_joint_indices": LEFT_JOINT_INDICES.tolist(),
            "right_joint_indices": RIGHT_JOINT_INDICES.tolist(),
        },
        "qc": {
            "status": qc_status,
            "reasons": list(reduction.qc_reasons),
            "nan_count": reduction.nan_count,
        },
        "model": _model_identity(config),
        "artifact": artifact,
        "arrays": {"pose": [int(size) for size in reduction.pose.shape], "dtype": "float32"},
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
        if _finished_sidecar_status(npy_path, sidecar_path, config.fingerprint) is None:
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
    manifest_path: Path,
    qc_path: Path,
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
            "pose_manifest_path": str(manifest_path),
            "pose_manifest_sha256": sha256_file(manifest_path),
            "pose_qc_path": str(qc_path),
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
        finished = _finished_sidecar_status(npy_path, sidecar_path, config.fingerprint)
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

    manifest_path, qc_path, manifest_rows = _rewrite_manifests(config)
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
        "pose_manifest": str(manifest_path),
        "pose_qc": str(qc_path),
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
        self._model = apis.init_model(
            str(config.model.config_path),
            str(config.model.checkpoint_path),
            device=config.model.device,
        )
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
