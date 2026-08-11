from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mmprism.data.csl_news_annotation import (
    CslNewsAnnotationConfig,
    CslNewsAnnotationError,
    sha256_file,
)

QC_SCHEMA_VERSION = "mmprism.csl_news_pose_annotation_qc.v1"
MIN_IN_BOUNDS_RATIO = 0.80
MIN_CANONICAL_VALID_RATIO = 0.50
MAX_P99_FRAME_MOTION = 1.0
MIN_SCORE = -0.05
MAX_SCORE = 1.05


def _load_sidecar(path: Path) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid sidecar JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("sidecar must be a JSON object")
    return payload


def _select_evenly(paths: list[Path], sample_count: int) -> list[Path]:
    if sample_count < 1:
        raise CslNewsAnnotationError("sample_count must be positive")
    if len(paths) <= sample_count:
        return paths
    if sample_count == 1:
        return [paths[len(paths) // 2]]
    indices = {
        round(index * (len(paths) - 1) / (sample_count - 1))
        for index in range(sample_count)
    }
    return [paths[index] for index in sorted(indices)]


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    probabilities = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    labels = ("min", "p01", "p05", "median", "p95", "p99", "max")
    return {
        label: float(value)
        for label, value in zip(labels, np.quantile(values, probabilities), strict=True)
    }


def build_csl_news_annotation_qc(
    config: CslNewsAnnotationConfig,
    *,
    sample_count: int = 100,
) -> dict[str, Any]:
    """Measure pose quality on a deterministic, evenly spaced output sample."""

    if sample_count < 1:
        raise CslNewsAnnotationError("sample_count must be positive")

    sidecar_paths = sorted(
        path
        for path in (config.runtime.output_root / "samples").glob("archive_*/*.json")
        if not path.name.startswith(".")
    )
    selected = _select_evenly(sidecar_paths, sample_count) if sidecar_paths else []
    sample_reports: list[dict[str, Any]] = []
    scores: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    in_bounds_masks: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    frame_motions: list[np.ndarray] = []
    resolution_counts: Counter[str] = Counter()
    failure_count = 0
    warning_count = 0
    total_frames = 0

    for sidecar_path in selected:
        artifact_path = sidecar_path.with_suffix(".npz")
        failures: list[str] = []
        warnings: list[str] = []
        report: dict[str, Any] = {
            "sidecar": str(sidecar_path),
            "artifact": str(artifact_path),
            "failures": failures,
            "warnings": warnings,
        }
        try:
            sidecar = _load_sidecar(sidecar_path)
            if sidecar.get("status") != "completed":
                failures.append("sidecar status is not completed")
            if sidecar.get("config_fingerprint") != config.fingerprint:
                failures.append("config fingerprint mismatch")
            annotation = sidecar.get("annotation")
            if (
                not isinstance(annotation, Mapping)
                or not isinstance(annotation.get("text"), str)
                or not annotation["text"].strip()
            ):
                failures.append("annotation text is missing or empty")
            artifact = sidecar.get("artifact")
            expected_sha256 = artifact.get("sha256") if isinstance(artifact, Mapping) else None
            if not artifact_path.is_file():
                failures.append("artifact is missing")
                raise ValueError("artifact is missing")
            checksum_match = (
                isinstance(expected_sha256, str)
                and sha256_file(artifact_path) == expected_sha256
            )
            if not checksum_match:
                failures.append("artifact checksum mismatch")

            with np.load(artifact_path, allow_pickle=False) as arrays:
                native = arrays["native_keypoints_3d"]
                native_scores = arrays["native_keypoint_scores"]
                transformed = arrays["transformed_keypoints_2d"]
                canonical = arrays["canonical_pose"]
                canonical_valid = arrays["canonical_valid"]
                frame_indices = arrays["frame_indices"]
            frame_count = int(native.shape[0])
            expected_shapes = {
                "native_keypoints_3d": (frame_count, 133, 3),
                "native_keypoint_scores": (frame_count, 133),
                "transformed_keypoints_2d": (frame_count, 133, 2),
                "canonical_pose": (frame_count, 2, 24, 3),
                "canonical_valid": (frame_count, 2, 24),
                "frame_indices": (frame_count,),
            }
            actual_shapes = {
                "native_keypoints_3d": native.shape,
                "native_keypoint_scores": native_scores.shape,
                "transformed_keypoints_2d": transformed.shape,
                "canonical_pose": canonical.shape,
                "canonical_valid": canonical_valid.shape,
                "frame_indices": frame_indices.shape,
            }
            shape_failures = [
                name
                for name, expected in expected_shapes.items()
                if actual_shapes[name] != expected
            ]
            if shape_failures:
                failures.append(f"invalid array shapes: {', '.join(shape_failures)}")
                raise ValueError("array shape contract failed")
            if frame_count == 0:
                failures.append("artifact has zero frames")
                raise ValueError("artifact has zero frames")
            finite = bool(
                np.isfinite(native).all()
                and np.isfinite(native_scores).all()
                and np.isfinite(transformed).all()
                and np.isfinite(canonical).all()
            )
            if not finite:
                failures.append("pose arrays contain non-finite values")
            if not np.array_equal(frame_indices, np.arange(frame_count)):
                failures.append("frame indices are not contiguous from zero")

            video = sidecar.get("video")
            if not isinstance(video, Mapping):
                failures.append("video metadata is missing")
                raise ValueError("video metadata is missing")
            reported_frames = video.get("reported_frame_count")
            fps = video.get("fps")
            cropped_width = video.get("cropped_width")
            cropped_height = video.get("cropped_height")
            if (
                isinstance(reported_frames, int)
                and reported_frames > 0
                and reported_frames != frame_count
            ):
                failures.append("decoded and reported frame counts differ")
            if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
                failures.append("FPS is missing or non-positive")
            if (
                not isinstance(cropped_width, int)
                or not isinstance(cropped_height, int)
                or cropped_width <= 0
                or cropped_height <= 0
            ):
                failures.append("cropped video dimensions are invalid")
                raise ValueError("cropped video dimensions are invalid")

            in_bounds = (
                (transformed[..., 0] >= 0)
                & (transformed[..., 0] < cropped_width)
                & (transformed[..., 1] >= 0)
                & (transformed[..., 1] < cropped_height)
            )
            valid_ratio = float(canonical_valid.mean())
            in_bounds_ratio = float(in_bounds.mean())
            motion = (
                np.linalg.norm(np.diff(canonical, axis=0), axis=-1).reshape(-1)
                if frame_count > 1
                else np.asarray([], dtype=np.float32)
            )
            p99_motion = float(np.quantile(motion, 0.99)) if motion.size else 0.0
            score_min = float(native_scores.min())
            score_max = float(native_scores.max())
            if in_bounds_ratio < MIN_IN_BOUNDS_RATIO:
                warnings.append(
                    f"2D in-bounds ratio {in_bounds_ratio:.4f} is below {MIN_IN_BOUNDS_RATIO}"
                )
            if valid_ratio < MIN_CANONICAL_VALID_RATIO:
                warnings.append(
                    f"canonical valid ratio {valid_ratio:.4f} is below "
                    f"{MIN_CANONICAL_VALID_RATIO}"
                )
            if p99_motion > MAX_P99_FRAME_MOTION:
                warnings.append(
                    f"p99 frame motion {p99_motion:.4f} exceeds {MAX_P99_FRAME_MOTION}"
                )
            if score_min < MIN_SCORE or score_max > MAX_SCORE:
                warnings.append(
                    f"native score range [{score_min:.4f}, {score_max:.4f}] exceeds "
                    f"[{MIN_SCORE}, {MAX_SCORE}]"
                )

            total_frames += frame_count
            scores.append(native_scores.reshape(-1))
            valid_masks.append(canonical_valid.reshape(-1))
            in_bounds_masks.append(in_bounds.reshape(-1))
            depths.append(native[..., 2].reshape(-1))
            if motion.size:
                frame_motions.append(motion)
            resolution_counts[f"{cropped_width}x{cropped_height}"] += 1
            report.update(
                {
                    "sample_id": sidecar.get("sample_id"),
                    "frame_count": frame_count,
                    "reported_frame_count": reported_frames,
                    "fps": fps,
                    "cropped_resolution": [cropped_width, cropped_height],
                    "checksum_match": checksum_match,
                    "finite": finite,
                    "canonical_valid_ratio": valid_ratio,
                    "transformed_in_bounds_ratio": in_bounds_ratio,
                    "native_score_min": score_min,
                    "native_score_mean": float(native_scores.mean()),
                    "native_score_max": score_max,
                    "p99_frame_motion": p99_motion,
                }
            )
        except (IndexError, KeyError, OSError, ValueError) as error:
            if not failures:
                failures.append(str(error))
        if failures:
            failure_count += 1
            report["status"] = "failed"
        elif warnings:
            warning_count += 1
            report["status"] = "warning"
        else:
            report["status"] = "passed"
        sample_reports.append(report)

    def concatenate(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)

    aggregate_scores = concatenate(scores)
    aggregate_valid = concatenate(valid_masks)
    aggregate_in_bounds = concatenate(in_bounds_masks)
    aggregate_depths = concatenate(depths)
    aggregate_motion = concatenate(frame_motions)
    global_failures = ["no completed annotation sidecars"] if not selected else []
    status = (
        "failed"
        if failure_count or global_failures
        else "passed_with_warnings"
        if warning_count
        else "passed"
    )
    return {
        "schema_version": QC_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "config_fingerprint": config.fingerprint,
        "selection": {
            "policy": "evenly_spaced_sorted_sidecars",
            "candidate_count": len(sidecar_paths),
            "requested_sample_count": sample_count,
            "checked_sample_count": len(selected),
        },
        "thresholds": {
            "minimum_transformed_in_bounds_ratio": MIN_IN_BOUNDS_RATIO,
            "minimum_canonical_valid_ratio": MIN_CANONICAL_VALID_RATIO,
            "maximum_p99_frame_motion": MAX_P99_FRAME_MOTION,
            "expected_native_score_range": [MIN_SCORE, MAX_SCORE],
        },
        "summary": {
            "global_failures": global_failures,
            "failure_sample_count": failure_count,
            "warning_sample_count": warning_count,
            "passed_sample_count": len(selected) - failure_count - warning_count,
            "total_frame_count": total_frames,
            "cropped_resolution_counts": dict(sorted(resolution_counts.items())),
            "native_score_quantiles": _quantiles(aggregate_scores),
            "canonical_valid_ratio": (
                float(aggregate_valid.mean()) if aggregate_valid.size else None
            ),
            "transformed_in_bounds_ratio": (
                float(aggregate_in_bounds.mean()) if aggregate_in_bounds.size else None
            ),
            "native_depth_quantiles": _quantiles(aggregate_depths),
            "canonical_frame_motion_quantiles": _quantiles(aggregate_motion),
        },
        "samples": sample_reports,
    }


def write_csl_news_annotation_qc(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write an annotation QC report atomically."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
