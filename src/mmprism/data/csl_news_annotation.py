from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import types
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
import yaml

ANNOTATION_SCHEMA_VERSION = "mmprism.csl_news_pose_annotation.v1"
OUTPUT_SCHEMA_VERSION = "mmprism.csl_news_pose_sample.v1"
ARCHIVE_PATTERN = re.compile(r"^archive_(\d{3})\.zip$")
LEFT_JOINT_INDICES = np.asarray([5, 7, 9, *range(91, 112)], dtype=np.int64)
RIGHT_JOINT_INDICES = np.asarray([6, 8, 10, *range(112, 133)], dtype=np.int64)


class CslNewsAnnotationError(RuntimeError):
    """Raised when CSL-News annotation cannot continue safely."""


@dataclass(frozen=True)
class CslNewsLabel:
    video_name: str
    text: str
    legacy_pose_name: str | None


@dataclass(frozen=True)
class AnnotationSourceConfig:
    archive_root: Path
    labels_path: Path
    source_id: str
    source_revision: str
    expected_archive_count: int


@dataclass(frozen=True)
class AnnotationModelConfig:
    mmpose_root: Path
    mmpose_commit: str
    project_dir: Path
    config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    device: str


@dataclass(frozen=True)
class AnnotationTransformConfig:
    crop_top: int
    crop_left: int
    crop_right: int
    confidence_threshold: float


@dataclass(frozen=True)
class AnnotationRuntimeConfig:
    output_root: Path
    scratch_root: Path
    worker_index: int
    worker_count: int
    poll_seconds: int
    min_free_bytes: int
    max_consecutive_oom: int


@dataclass(frozen=True)
class CslNewsAnnotationConfig:
    source: AnnotationSourceConfig
    model: AnnotationModelConfig
    transform: AnnotationTransformConfig
    runtime: AnnotationRuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "source": {
                "archive_root": str(self.source.archive_root),
                "labels_path": str(self.source.labels_path),
                "source_id": self.source.source_id,
                "source_revision": self.source.source_revision,
                "expected_archive_count": self.source.expected_archive_count,
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
                "crop_top": self.transform.crop_top,
                "crop_left": self.transform.crop_left,
                "crop_right": self.transform.crop_right,
                "confidence_threshold": self.transform.confidence_threshold,
            },
            "runtime": {
                "output_root": str(self.runtime.output_root),
                "scratch_root": str(self.runtime.scratch_root),
                "worker_index": self.runtime.worker_index,
                "worker_count": self.runtime.worker_count,
                "poll_seconds": self.runtime.poll_seconds,
                "min_free_bytes": self.runtime.min_free_bytes,
                "max_consecutive_oom": self.runtime.max_consecutive_oom,
            },
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VideoAnnotation:
    native_keypoints_3d: np.ndarray
    native_keypoint_scores: np.ndarray
    transformed_keypoints_2d: np.ndarray
    frame_indices: np.ndarray
    timestamps_seconds: np.ndarray
    fps: float
    reported_frame_count: int
    width: int
    height: int
    cropped_width: int
    cropped_height: int


class PoseEstimator(Protocol):
    def annotate_video(self, video_path: Path) -> VideoAnnotation: ...

    def runtime_metadata(self) -> Mapping[str, Any]: ...


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CslNewsAnnotationError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CslNewsAnnotationError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CslNewsAnnotationError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _integer(payload: Mapping[str, Any], key: str, location: str, *, minimum: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CslNewsAnnotationError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(payload: Mapping[str, Any], key: str, location: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CslNewsAnnotationError(f"{location}.{key} must be numeric")
    return float(value)


def _path(payload: Mapping[str, Any], key: str, location: str, project_root: Path) -> Path:
    raw = os.path.expandvars(_text(payload, key, location))
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def load_csl_news_annotation_config(
    path: str | Path, project_root: str | Path
) -> CslNewsAnnotationConfig:
    """Load and strictly validate the annotation configuration."""

    config_path = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CslNewsAnnotationError(f"Unable to load annotation config: {error}") from error
    payload = _mapping(raw, "root")
    _reject_unknown(payload, {"schema_version", "source", "model", "transform", "runtime"}, "root")
    if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise CslNewsAnnotationError(
            f"schema_version must be {ANNOTATION_SCHEMA_VERSION}"
        )

    source = _mapping(payload.get("source"), "source")
    _reject_unknown(
        source,
        {"archive_root", "labels_path", "source_id", "source_revision", "expected_archive_count"},
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
        {"crop_top", "crop_left", "crop_right", "confidence_threshold"},
        "transform",
    )
    runtime = _mapping(payload.get("runtime"), "runtime")
    _reject_unknown(
        runtime,
        {
            "output_root",
            "scratch_root",
            "worker_index",
            "worker_count",
            "poll_seconds",
            "min_free_bytes",
            "max_consecutive_oom",
        },
        "runtime",
    )

    worker_index = _integer(runtime, "worker_index", "runtime", minimum=0)
    worker_count = _integer(runtime, "worker_count", "runtime", minimum=1)
    if worker_index >= worker_count:
        raise CslNewsAnnotationError("runtime.worker_index must be less than worker_count")
    threshold = _number(transform, "confidence_threshold", "transform")
    if not 0.0 <= threshold <= 1.0:
        raise CslNewsAnnotationError("transform.confidence_threshold must be in [0, 1]")
    checksum = _text(model, "checkpoint_sha256", "model").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise CslNewsAnnotationError("model.checkpoint_sha256 must contain 64 hex characters")

    return CslNewsAnnotationConfig(
        source=AnnotationSourceConfig(
            archive_root=_path(source, "archive_root", "source", root),
            labels_path=_path(source, "labels_path", "source", root),
            source_id=_text(source, "source_id", "source"),
            source_revision=_text(source, "source_revision", "source"),
            expected_archive_count=_integer(
                source, "expected_archive_count", "source", minimum=1
            ),
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
        transform=AnnotationTransformConfig(
            crop_top=_integer(transform, "crop_top", "transform", minimum=0),
            crop_left=_integer(transform, "crop_left", "transform", minimum=0),
            crop_right=_integer(transform, "crop_right", "transform", minimum=0),
            confidence_threshold=threshold,
        ),
        runtime=AnnotationRuntimeConfig(
            output_root=_path(runtime, "output_root", "runtime", root),
            scratch_root=_path(runtime, "scratch_root", "runtime", root),
            worker_index=worker_index,
            worker_count=worker_count,
            poll_seconds=_integer(runtime, "poll_seconds", "runtime", minimum=1),
            min_free_bytes=_integer(runtime, "min_free_bytes", "runtime", minimum=0),
            max_consecutive_oom=_integer(
                runtime, "max_consecutive_oom", "runtime", minimum=1
            ),
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_csl_news_labels(path: str | Path) -> dict[str, CslNewsLabel]:
    labels_path = Path(path).expanduser().resolve()
    if labels_path.name.endswith(".part") or not labels_path.is_file():
        raise CslNewsAnnotationError(f"Labels must be a complete JSON file: {labels_path}")
    try:
        raw: object = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslNewsAnnotationError(f"Unable to read labels {labels_path}: {error}") from error
    if not isinstance(raw, list):
        raise CslNewsAnnotationError("CSL-News labels must be a JSON list")

    labels: dict[str, CslNewsLabel] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise CslNewsAnnotationError(f"Label record {index} must be a mapping")
        video = item.get("video")
        text = item.get("text")
        pose = item.get("pose")
        if not isinstance(video, str) or not video.strip():
            raise CslNewsAnnotationError(f"Label record {index} has no video name")
        if not isinstance(text, str) or not text.strip():
            raise CslNewsAnnotationError(f"Label record {index} has no text")
        if pose is not None and not isinstance(pose, str):
            raise CslNewsAnnotationError(f"Label record {index} has an invalid pose name")
        video_name = PurePosixPath(video.strip().replace("\\", "/")).name
        if video_name in labels:
            raise CslNewsAnnotationError(f"Duplicate label video name: {video_name}")
        labels[video_name] = CslNewsLabel(video_name, text, pose)
    return labels


def stable_sample_id(source_id: str, archive_name: str, member_name: str) -> str:
    identity = "\0".join((source_id, archive_name, member_name)).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:24]


def canonicalize_hands(
    native_keypoints_3d: np.ndarray,
    native_scores: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply the historical 2x24 arm/hand mapping without mutating native output."""

    keypoints = np.asarray(native_keypoints_3d, dtype=np.float32)
    scores = np.asarray(native_scores, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 3):
        raise CslNewsAnnotationError(
            f"native keypoints must have shape [T, 133, 3], got {keypoints.shape}"
        )
    if scores.shape != keypoints.shape[:2]:
        raise CslNewsAnnotationError(
            f"native scores must have shape [T, 133], got {scores.shape}"
        )
    if keypoints.shape[0] == 0:
        raise CslNewsAnnotationError("Cannot canonicalize an empty video")
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        raise CslNewsAnnotationError("Pose output contains non-finite values")

    depth_center = float(keypoints[:, [6, 7], 2].mean(dtype=np.float64))
    centered = keypoints.copy()
    centered[..., 2] -= depth_center
    canonical_pose = np.stack(
        (centered[:, LEFT_JOINT_INDICES], centered[:, RIGHT_JOINT_INDICES]), axis=1
    ).astype(np.float32, copy=False)
    canonical_scores = np.stack(
        (scores[:, LEFT_JOINT_INDICES], scores[:, RIGHT_JOINT_INDICES]), axis=1
    ).astype(np.float32, copy=False)
    valid = np.logical_and(
        canonical_scores >= confidence_threshold,
        np.isfinite(canonical_pose).all(axis=-1),
    )
    return canonical_pose, canonical_scores, valid, depth_center


def validate_annotation_output(path: str | Path) -> bool:
    artifact = Path(path)
    if not artifact.is_file():
        return False
    required = {
        "native_keypoints_3d",
        "native_keypoint_scores",
        "transformed_keypoints_2d",
        "frame_indices",
        "timestamps_seconds",
        "canonical_pose",
        "canonical_confidence",
        "canonical_valid",
    }
    try:
        with np.load(artifact, allow_pickle=False) as arrays:
            if not required.issubset(arrays.files):
                return False
            native = arrays["native_keypoints_3d"]
            scores = arrays["native_keypoint_scores"]
            transformed = arrays["transformed_keypoints_2d"]
            frame_indices = arrays["frame_indices"]
            timestamps = arrays["timestamps_seconds"]
            canonical = arrays["canonical_pose"]
            confidence = arrays["canonical_confidence"]
            valid = arrays["canonical_valid"]
            frame_count = native.shape[0] if native.ndim == 3 else -1
            return bool(
                frame_count > 0
                and native.shape == (frame_count, 133, 3)
                and scores.shape == (frame_count, 133)
                and transformed.shape == (frame_count, 133, 2)
                and frame_indices.shape == (frame_count,)
                and timestamps.shape == (frame_count,)
                and canonical.shape == (frame_count, 2, 24, 3)
                and confidence.shape == (frame_count, 2, 24)
                and valid.shape == (frame_count, 2, 24)
                and np.isfinite(native).all()
                and np.isfinite(scores).all()
                and np.isfinite(transformed).all()
                and np.isfinite(canonical).all()
                and np.array_equal(frame_indices, np.arange(frame_count))
            )
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz_atomic(payload: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)  # type: ignore[arg-type]
    temporary.replace(path)


def _archive_number(path: Path) -> int:
    match = ARCHIVE_PATTERN.fullmatch(path.name)
    if match is None:
        raise CslNewsAnnotationError(f"Unexpected archive name: {path.name}")
    return int(match.group(1))


def discover_complete_archives(config: CslNewsAnnotationConfig) -> list[Path]:
    if not config.source.archive_root.is_dir():
        raise CslNewsAnnotationError(
            f"Archive root does not exist: {config.source.archive_root}"
        )
    archives = []
    for path in config.source.archive_root.glob("archive_*.zip"):
        archive_number = _archive_number(path)
        if archive_number % config.runtime.worker_count == config.runtime.worker_index:
            archives.append(path.resolve())
    return sorted(archives, key=_archive_number)


def _require_model_assets(config: CslNewsAnnotationConfig) -> dict[str, str]:
    for description, path in (
        ("MMPose source", config.model.mmpose_root),
        ("RTMPose3D project", config.model.project_dir),
        ("model config", config.model.config_path),
        ("checkpoint", config.model.checkpoint_path),
    ):
        if not path.exists():
            raise CslNewsAnnotationError(f"{description} does not exist: {path}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(config.model.mmpose_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CslNewsAnnotationError("Unable to verify the pinned MMPose commit") from error
    if commit != config.model.mmpose_commit:
        raise CslNewsAnnotationError(
            f"MMPose commit mismatch: expected {config.model.mmpose_commit}, got {commit}"
        )
    checkpoint_sha256 = sha256_file(config.model.checkpoint_path)
    if checkpoint_sha256 != config.model.checkpoint_sha256:
        raise CslNewsAnnotationError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {config.model.checkpoint_sha256}, got {checkpoint_sha256}"
        )
    return {
        "mmpose_commit": commit,
        "config_sha256": sha256_file(config.model.config_path),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _normalize_prediction(value: Any, final_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > len(final_shape) and array.shape[0] == 1:
        array = array[0]
    if array.shape != final_shape:
        raise CslNewsAnnotationError(f"{name} has shape {array.shape}, expected {final_shape}")
    return array.astype(np.float32, copy=False)


class MMPoseRtmw3dEstimator:
    """Optional-dependency adapter around the pinned RTMW3D estimator."""

    def __init__(self, config: CslNewsAnnotationConfig) -> None:
        self._config = config
        project_path = str(config.model.project_dir)
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
        self._guard_unused_edpose_head()
        try:
            importlib.import_module("rtmpose3d")
            self._cv2 = importlib.import_module("cv2")
            apis = importlib.import_module("mmpose.apis")
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise CslNewsAnnotationError(
                "Annotation dependency import failed after syncing the UV annotation extra: "
                f"{error}"
            ) from error
        self._torch = torch
        cpu_threads = int(os.environ.get("MMPRISM_CPU_THREADS", "4"))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        self._cv2.setNumThreads(1)
        self._inference_topdown = apis.inference_topdown
        self._model = apis.init_model(
            str(config.model.config_path),
            str(config.model.checkpoint_path),
            device=config.model.device,
        )
        if config.model.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _guard_unused_edpose_head() -> None:
        """Keep MMPose's eager EDPose import from requiring unused MMCV CUDA ops."""

        try:
            importlib.import_module("mmcv._ext")
            return
        except ImportError:
            pass
        module_name = "mmpose.models.heads.transformer_heads"
        if module_name in sys.modules:
            return

        class UnavailableEdPoseHead:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise CslNewsAnnotationError(
                    "EDPose requires full MMCV ops and is unavailable in the annotation profile"
                )

        module = types.ModuleType(module_name)
        module.EDPoseHead = UnavailableEdPoseHead  # type: ignore[attr-defined]
        sys.modules[module_name] = module

    def annotate_video(self, video_path: Path) -> VideoAnnotation:
        capture = self._cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise CslNewsAnnotationError(f"OpenCV cannot open video: {video_path}")
        fps = float(capture.get(self._cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(self._cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise CslNewsAnnotationError(f"Video has invalid FPS {fps}: {video_path}")

        keypoints: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        transformed: list[np.ndarray] = []
        crop = self._config.transform
        try:
            while True:
                readable, frame = capture.read()
                if not readable:
                    break
                if (
                    frame.shape[0] <= crop.crop_top
                    or frame.shape[1] <= crop.crop_left + crop.crop_right
                ):
                    raise CslNewsAnnotationError(
                        f"Crop exceeds frame dimensions {frame.shape[:2]}: {video_path}"
                    )
                right = frame.shape[1] - crop.crop_right if crop.crop_right else frame.shape[1]
                cropped = frame[crop.crop_top :, crop.crop_left : right]
                results = self._inference_topdown(self._model, cropped, None)
                if len(results) != 1:
                    raise CslNewsAnnotationError(
                        f"Expected one pose instance, received {len(results)}"
                    )
                instances = results[0].pred_instances
                keypoints.append(
                    _normalize_prediction(instances.keypoints, (133, 3), "keypoints")
                )
                scores.append(
                    _normalize_prediction(
                        instances.keypoint_scores, (133,), "keypoint_scores"
                    )
                )
                transformed.append(
                    _normalize_prediction(
                        instances.transformed_keypoints,
                        (133, 2),
                        "transformed_keypoints",
                    )
                )
        finally:
            capture.release()
        if not keypoints:
            raise CslNewsAnnotationError(f"Video decoded zero frames: {video_path}")

        frame_count = len(keypoints)
        return VideoAnnotation(
            native_keypoints_3d=np.stack(keypoints).astype(np.float32, copy=False),
            native_keypoint_scores=np.stack(scores).astype(np.float32, copy=False),
            transformed_keypoints_2d=np.stack(transformed).astype(np.float32, copy=False),
            frame_indices=np.arange(frame_count, dtype=np.int64),
            timestamps_seconds=np.arange(frame_count, dtype=np.float64) / fps,
            fps=fps,
            reported_frame_count=reported_frames,
            width=width,
            height=height,
            cropped_width=width - crop.crop_left - crop.crop_right,
            cropped_height=height - crop.crop_top,
        )

    def runtime_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "device": self._config.model.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint_load_policy": os.environ.get(
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
            ),
            "cpu_threads": self._torch.get_num_threads(),
            "interop_threads": self._torch.get_num_interop_threads(),
            "opencv_threads": self._cv2.getNumThreads(),
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


def _safe_video_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [member for member in archive.infolist() if not member.is_dir()]
    unsafe = [
        member.filename
        for member in members
        if (path := PurePosixPath(member.filename.replace("\\", "/"))).is_absolute()
        or ".." in path.parts
    ]
    if unsafe:
        raise CslNewsAnnotationError(f"Archive contains unsafe member paths: {unsafe[:3]}")
    videos = sorted(
        [
            member
            for member in members
            if PurePosixPath(member.filename.replace("\\", "/")).suffix.lower() == ".mp4"
        ],
        key=lambda member: member.filename,
    )
    names = [PurePosixPath(member.filename.replace("\\", "/")).name for member in videos]
    if not videos:
        raise CslNewsAnnotationError("Archive contains no MP4 videos")
    if len(names) != len(set(names)):
        raise CslNewsAnnotationError("Archive contains duplicate video basenames")
    return videos


def _ensure_disk_floor(config: CslNewsAnnotationConfig) -> None:
    for root in (config.runtime.output_root, config.runtime.scratch_root):
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        if free < config.runtime.min_free_bytes:
            raise CslNewsAnnotationError(
                f"Free space {free} at {root} is below floor {config.runtime.min_free_bytes}"
            )


def _extract_video(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
) -> tuple[Path, str]:
    metadata_path = destination.with_suffix(".source.json")
    if destination.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if (
            destination.stat().st_size == member.file_size
            and metadata.get("zip_crc32") == member.CRC
        ):
            checksum = metadata.get("sha256")
            if isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum):
                return destination, checksum

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    try:
        with archive.open(member, "r") as source, temporary.open("wb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise CslNewsAnnotationError(
            f"Unable to extract {member.filename}: {error}"
        ) from error
    temporary.replace(destination)
    checksum = digest.hexdigest()
    _write_json_atomic(
        {
            "member": member.filename,
            "size_bytes": member.file_size,
            "zip_crc32": member.CRC,
            "sha256": checksum,
        },
        metadata_path,
    )
    return destination, checksum


def _artifact_paths(
    config: CslNewsAnnotationConfig, archive_name: str, sample_id: str
) -> tuple[Path, Path]:
    archive_stem = Path(archive_name).stem
    sample_root = config.runtime.output_root / "samples" / archive_stem
    return sample_root / f"{sample_id}.npz", sample_root / f"{sample_id}.json"


def is_completed_annotation_sample(
    npz_path: Path, sidecar_path: Path, config_fingerprint: str
) -> bool:
    if not sidecar_path.is_file() or not validate_annotation_output(npz_path):
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        sidecar.get("status") == "completed"
        and sidecar.get("config_fingerprint") == config_fingerprint
        and sidecar.get("artifact", {}).get("sha256") == sha256_file(npz_path)
    )


def _failure_sidecar(
    config: CslNewsAnnotationConfig,
    archive_name: str,
    sample_id: str,
    member_name: str,
    error: BaseException,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = (
        config.runtime.output_root
        / "failures"
        / Path(archive_name).stem
        / sample_id
        / f"attempt_{timestamp}.json"
    )
    _write_json_atomic(
        {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "failed",
            "generated_at": datetime.now(UTC).isoformat(),
            "config_fingerprint": config.fingerprint,
            "archive": archive_name,
            "member": member_name,
            "sample_id": sample_id,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        path,
    )
    return path


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"time": datetime.now(UTC).isoformat(), "event": event, **payload},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _run_metadata(
    config: CslNewsAnnotationConfig, model_assets: Mapping[str, str], labels_sha256: str
) -> Path:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
        git_dirty = True
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = config.runtime.output_root / "runs" / f"run_{timestamp}_{os.getpid()}.json"
    _write_json_atomic(
        {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "started_at": datetime.now(UTC).isoformat(),
            "config": config.to_dict(),
            "config_fingerprint": config.fingerprint,
            "git": {"commit": git_commit, "dirty": git_dirty},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cpu_threads": os.environ.get("MMPRISM_CPU_THREADS"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            },
            "model_assets": dict(model_assets),
            "labels_sha256": labels_sha256,
        },
        path,
    )
    return path


def run_csl_news_annotation(
    config: CslNewsAnnotationConfig,
    *,
    max_videos: int | None = None,
    once: bool = False,
    archive_id: int | None = None,
    estimator: PoseEstimator | None = None,
) -> dict[str, Any]:
    """Annotate complete archives, preserving restartable outputs and all scratch data."""

    if max_videos is not None and max_videos < 1:
        raise CslNewsAnnotationError("max_videos must be positive")
    if archive_id is not None and not 1 <= archive_id <= config.source.expected_archive_count:
        raise CslNewsAnnotationError("archive_id is outside the configured source range")
    _ensure_disk_floor(config)
    labels = load_csl_news_labels(config.source.labels_path)
    labels_sha256 = sha256_file(config.source.labels_path)
    model_assets = _require_model_assets(config)
    run_metadata_path = _run_metadata(config, model_assets, labels_sha256)
    _emit(
        "annotation_run_started",
        config_fingerprint=config.fingerprint,
        run_metadata=str(run_metadata_path),
        worker_index=config.runtime.worker_index,
        worker_count=config.runtime.worker_count,
    )

    processed = 0
    skipped = 0
    failed = 0
    consecutive_oom = 0
    pose_estimator = estimator
    while True:
        archives = discover_complete_archives(config)
        if archive_id is not None:
            archives = [path for path in archives if _archive_number(path) == archive_id]
            if not archives:
                raise CslNewsAnnotationError(
                    f"Requested archive_{archive_id:03d}.zip is not complete"
                )
        made_progress = False
        for archive_path in archives:
            marker_path = (
                config.runtime.output_root / "archives" / f"{archive_path.stem}.json"
            )
            if marker_path.is_file() and max_videos is None:
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    marker = {}
                if (
                    marker.get("status") in {"completed", "completed_with_failures"}
                    and marker.get("config_fingerprint") == config.fingerprint
                    and marker.get("archive_size_bytes") == archive_path.stat().st_size
                ):
                    continue

            _ensure_disk_floor(config)
            archive_processed = 0
            archive_skipped = 0
            archive_failed = 0
            try:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    members = _safe_video_members(archive)
                    missing_labels = [
                        PurePosixPath(member.filename.replace("\\", "/")).name
                        for member in members
                        if PurePosixPath(member.filename.replace("\\", "/")).name
                        not in labels
                    ]
                    if missing_labels:
                        raise CslNewsAnnotationError(
                            f"Archive {archive_path.name} has {len(missing_labels)} "
                            "videos without labels"
                        )
                    _emit(
                        "archive_started",
                        archive=archive_path.name,
                        video_count=len(members),
                    )
                    for member in members:
                        if max_videos is not None and processed >= max_videos:
                            return {
                                "status": "limit_reached",
                                "processed": processed,
                                "skipped": skipped,
                                "failed": failed,
                                "run_metadata": str(run_metadata_path),
                            }
                        _ensure_disk_floor(config)
                        member_name = member.filename.replace("\\", "/")
                        video_name = PurePosixPath(member_name).name
                        sample_id = stable_sample_id(
                            config.source.source_id, archive_path.name, member_name
                        )
                        npz_path, sidecar_path = _artifact_paths(
                            config, archive_path.name, sample_id
                        )
                        if is_completed_annotation_sample(
                            npz_path, sidecar_path, config.fingerprint
                        ):
                            skipped += 1
                            archive_skipped += 1
                            continue

                        scratch_path = (
                            config.runtime.scratch_root
                            / "videos"
                            / archive_path.stem
                            / f"{sample_id}--{video_name}"
                        )
                        started = time.monotonic()
                        try:
                            extracted_path, video_sha256 = _extract_video(
                                archive, member, scratch_path
                            )
                            if pose_estimator is None:
                                pose_estimator = MMPoseRtmw3dEstimator(config)
                                _emit(
                                    "model_loaded",
                                    runtime=dict(pose_estimator.runtime_metadata()),
                                )
                            annotation = pose_estimator.annotate_video(extracted_path)
                            canonical, confidence, valid, depth_center = canonicalize_hands(
                                annotation.native_keypoints_3d,
                                annotation.native_keypoint_scores,
                                config.transform.confidence_threshold,
                            )
                            _write_npz_atomic(
                                {
                                    "native_keypoints_3d": annotation.native_keypoints_3d,
                                    "native_keypoint_scores": annotation.native_keypoint_scores,
                                    "transformed_keypoints_2d": annotation.transformed_keypoints_2d,
                                    "frame_indices": annotation.frame_indices,
                                    "timestamps_seconds": annotation.timestamps_seconds,
                                    "canonical_pose": canonical,
                                    "canonical_confidence": confidence,
                                    "canonical_valid": valid,
                                },
                                npz_path,
                            )
                            artifact_sha256 = sha256_file(npz_path)
                            sidecar = {
                                "schema_version": OUTPUT_SCHEMA_VERSION,
                                "status": "completed",
                                "generated_at": datetime.now(UTC).isoformat(),
                                "sample_id": sample_id,
                                "config_fingerprint": config.fingerprint,
                                "source": {
                                    "source_id": config.source.source_id,
                                    "source_revision": config.source.source_revision,
                                    "archive": archive_path.name,
                                    "archive_size_bytes": archive_path.stat().st_size,
                                    "member": member_name,
                                    "member_size_bytes": member.file_size,
                                    "member_crc32": member.CRC,
                                    "video_sha256": video_sha256,
                                },
                                "annotation": {
                                    "text": labels[video_name].text,
                                    "legacy_pose_name": labels[video_name].legacy_pose_name,
                                },
                                "video": {
                                    "decoded_frame_count": int(annotation.frame_indices.size),
                                    "reported_frame_count": annotation.reported_frame_count,
                                    "fps": annotation.fps,
                                    "width": annotation.width,
                                    "height": annotation.height,
                                    "cropped_width": annotation.cropped_width,
                                    "cropped_height": annotation.cropped_height,
                                },
                                "transform": {
                                    "crop_top": config.transform.crop_top,
                                    "crop_left": config.transform.crop_left,
                                    "crop_right": config.transform.crop_right,
                                    "bbox_policy": "whole_cropped_frame_xyxy",
                                    "depth_center_policy": "sequence_mean_native_z_joints_6_7",
                                    "depth_center": depth_center,
                                    "left_joint_indices": LEFT_JOINT_INDICES.tolist(),
                                    "right_joint_indices": RIGHT_JOINT_INDICES.tolist(),
                                    "confidence_threshold": config.transform.confidence_threshold,
                                },
                                "model": {
                                    **model_assets,
                                    "device": config.model.device,
                                },
                                "arrays": {
                                    "native_keypoints_3d": list(
                                        annotation.native_keypoints_3d.shape
                                    ),
                                    "native_keypoint_scores": list(
                                        annotation.native_keypoint_scores.shape
                                    ),
                                    "transformed_keypoints_2d": list(
                                        annotation.transformed_keypoints_2d.shape
                                    ),
                                    "canonical_pose": list(canonical.shape),
                                    "canonical_confidence": list(confidence.shape),
                                    "canonical_valid": list(valid.shape),
                                },
                                "runtime": dict(pose_estimator.runtime_metadata()),
                                "artifact": {
                                    "path": str(npz_path),
                                    "size_bytes": npz_path.stat().st_size,
                                    "sha256": artifact_sha256,
                                },
                                "scratch_video": str(extracted_path),
                                "elapsed_seconds": time.monotonic() - started,
                            }
                            _write_json_atomic(sidecar, sidecar_path)
                            processed += 1
                            archive_processed += 1
                            consecutive_oom = 0
                            made_progress = True
                            _emit(
                                "sample_completed",
                                archive=archive_path.name,
                                member=member_name,
                                sample_id=sample_id,
                                frames=int(annotation.frame_indices.size),
                                elapsed_seconds=sidecar["elapsed_seconds"],
                                output=str(npz_path),
                            )
                        except Exception as error:
                            failed += 1
                            archive_failed += 1
                            failure_path = _failure_sidecar(
                                config,
                                archive_path.name,
                                sample_id,
                                member_name,
                                error,
                            )
                            is_oom = (
                                "out of memory" in str(error).lower()
                                or type(error).__name__ == "OutOfMemoryError"
                            )
                            consecutive_oom = consecutive_oom + 1 if is_oom else 0
                            _emit(
                                "sample_failed",
                                archive=archive_path.name,
                                member=member_name,
                                sample_id=sample_id,
                                error=str(error),
                                failure=str(failure_path),
                            )
                            if pose_estimator is None:
                                raise CslNewsAnnotationError(
                                    "Failed before the pose estimator became ready"
                                ) from error
                            if is_oom and consecutive_oom >= config.runtime.max_consecutive_oom:
                                raise CslNewsAnnotationError(
                                    f"Stopping after {consecutive_oom} consecutive "
                                    "CUDA OOM failures"
                                ) from error
            except (OSError, zipfile.BadZipFile, RuntimeError) as error:
                if isinstance(error, CslNewsAnnotationError):
                    raise
                raise CslNewsAnnotationError(
                    f"Unable to process archive {archive_path}: {error}"
                ) from error

            if max_videos is None:
                _write_json_atomic(
                    {
                        "schema_version": ANNOTATION_SCHEMA_VERSION,
                        "status": (
                            "completed_with_failures" if archive_failed else "completed"
                        ),
                        "completed_at": datetime.now(UTC).isoformat(),
                        "config_fingerprint": config.fingerprint,
                        "archive": archive_path.name,
                        "archive_size_bytes": archive_path.stat().st_size,
                        "processed": archive_processed,
                        "skipped": archive_skipped,
                        "failed": archive_failed,
                    },
                    marker_path,
                )
                _emit(
                    "archive_completed",
                    archive=archive_path.name,
                    processed=archive_processed,
                    skipped=archive_skipped,
                    failed=archive_failed,
                )

        if once or archive_id is not None:
            break
        expected_for_worker = sum(
            archive_number % config.runtime.worker_count == config.runtime.worker_index
            for archive_number in range(1, config.source.expected_archive_count + 1)
        )
        completed_marker_count = 0
        for marker_path in (config.runtime.output_root / "archives").glob("archive_*.json"):
            marker_match = re.fullmatch(r"archive_(\d{3})\.json", marker_path.name)
            if marker_match is None:
                continue
            marker_number = int(marker_match.group(1))
            if marker_number % config.runtime.worker_count != config.runtime.worker_index:
                continue
            try:
                marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                marker_payload.get("status") in {"completed", "completed_with_failures"}
                and marker_payload.get("config_fingerprint") == config.fingerprint
            ):
                completed_marker_count += 1
        if completed_marker_count >= expected_for_worker:
            break
        if not made_progress:
            _emit(
                "waiting_for_archives",
                complete_archives=len(archives),
                expected_for_worker=expected_for_worker,
                poll_seconds=config.runtime.poll_seconds,
            )
            time.sleep(config.runtime.poll_seconds)

    return {
        "status": "completed" if failed == 0 else "completed_with_failures",
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "run_metadata": str(run_metadata_path),
    }
