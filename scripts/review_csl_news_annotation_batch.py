#!/usr/bin/env python3
"""Render a frame-level visual review of RTMW3D batch inference differences."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mmprism.data.csl_news_annotation import (
    MMPoseRtmw3dEstimator,
    VideoAnnotation,
    load_csl_news_annotation_config,
)

BODY_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
HAND_FINGER_ROOTS = (1, 5, 9, 13, 17)
PALETTE = {
    "reference": (65, 230, 130),
    "batch16": (40, 160, 255),
    "batch64": (235, 80, 225),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize batch-1, batch-16, and batch-64 RTMW3D predictions."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=4)
    return parser.parse_args()


def _run_annotation(
    *,
    config_path: Path,
    project_root: Path,
    video: Path,
    batch_size: int,
) -> tuple[VideoAnnotation, dict[str, Any]]:
    config = load_csl_news_annotation_config(config_path, project_root)
    batched_config = replace(
        config,
        runtime=replace(config.runtime, inference_batch_size=batch_size),
    )
    estimator = MMPoseRtmw3dEstimator(batched_config)
    torch = estimator._torch  # Synchronize the pinned inference runtime for timing.
    if batched_config.model.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    annotation = estimator.annotate_video(video)
    if batched_config.model.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    metadata = {
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "frames_per_second": int(annotation.frame_indices.size) / elapsed_seconds,
        "runtime": dict(estimator.runtime_metadata()),
    }
    del estimator
    gc.collect()
    if batched_config.model.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return annotation, metadata


def _draw_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    color: tuple[int, int, int],
    confidence_threshold: float,
    point_radius: int = 2,
) -> None:
    import cv2

    visible = scores >= confidence_threshold

    def point(index: int) -> tuple[int, int]:
        return tuple(np.rint(keypoints[index]).astype(np.int32))

    for left, right in BODY_EDGES:
        if visible[left] and visible[right]:
            cv2.line(image, point(left), point(right), color, 2, cv2.LINE_AA)
    for wrist, offset in ((9, 91), (10, 112)):
        for root in HAND_FINGER_ROOTS:
            chain = (
                wrist,
                offset + root - 1,
                offset + root,
                offset + root + 1,
                offset + root + 2,
            )
            for left, right in zip(chain, chain[1:], strict=False):
                if visible[left] and visible[right]:
                    cv2.line(image, point(left), point(right), color, 1, cv2.LINE_AA)
    for index, is_visible in enumerate(visible):
        if is_visible:
            cv2.circle(image, point(index), point_radius, color, -1, cv2.LINE_AA)


def _label(image: np.ndarray, text: str, *, line: int = 0) -> None:
    import cv2

    origin = (8, 22 + line * 21)
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )


def _read_cropped_frames(
    video: Path,
    *,
    wanted: tuple[int, ...],
    crop_top: int,
    crop_left: int,
    crop_right: int,
) -> dict[int, np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video}")
    frames: dict[int, np.ndarray] = {}
    wanted_set = set(wanted)
    index = 0
    try:
        while wanted_set:
            readable, frame = capture.read()
            if not readable:
                break
            if index in wanted_set:
                right = frame.shape[1] - crop_right if crop_right else frame.shape[1]
                frames[index] = frame[crop_top:, crop_left:right].copy()
                wanted_set.remove(index)
            index += 1
    finally:
        capture.release()
    if wanted_set:
        raise RuntimeError(f"Could not decode requested frame indices: {sorted(wanted_set)}")
    return frames


def _scale_panel(image: np.ndarray, *, width: int = 360) -> np.ndarray:
    import cv2

    height, original_width = image.shape[:2]
    scaled_height = max(1, round(height * width / original_width))
    return cv2.resize(image, (width, scaled_height), interpolation=cv2.INTER_AREA)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    arguments = _parse_args()
    if arguments.frame_count < 1:
        raise ValueError("--frame-count must be positive")
    if not arguments.video.is_file():
        raise FileNotFoundError(f"Video does not exist: {arguments.video}")
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_csl_news_annotation_config(arguments.config, arguments.project_root)

    annotations: dict[int, VideoAnnotation] = {}
    runs: list[dict[str, Any]] = []
    for batch_size in (1, 16, 64):
        annotation, run = _run_annotation(
            config_path=arguments.config,
            project_root=arguments.project_root,
            video=arguments.video,
            batch_size=batch_size,
        )
        annotations[batch_size] = annotation
        runs.append(run)

    reference = annotations[1]
    frame_stats: list[dict[str, Any]] = []
    frame_errors: dict[int, np.ndarray] = {}
    for batch_size in (16, 64):
        candidate = annotations[batch_size]
        if (
            candidate.transformed_keypoints_2d.shape
            != reference.transformed_keypoints_2d.shape
        ):
            raise RuntimeError("Batch result has a different frame/keypoint shape")
        deltas = np.linalg.norm(
            candidate.transformed_keypoints_2d.astype(np.float64)
            - reference.transformed_keypoints_2d.astype(np.float64),
            axis=-1,
        )
        frame_errors[batch_size] = deltas
    combined_frame_error = np.maximum(
        frame_errors[16].max(axis=1), frame_errors[64].max(axis=1)
    )
    selected = np.argsort(-combined_frame_error, kind="stable")[: arguments.frame_count]
    selected_frames = tuple(sorted(int(index) for index in selected))
    frames = _read_cropped_frames(
        arguments.video,
        wanted=selected_frames,
        crop_top=config.transform.crop_top,
        crop_left=config.transform.crop_left,
        crop_right=config.transform.crop_right,
    )

    rows: list[np.ndarray] = []
    for frame_index in selected_frames:
        reference_points = reference.transformed_keypoints_2d[frame_index]
        reference_scores = reference.native_keypoint_scores[frame_index]
        panels: list[np.ndarray] = []
        panel_specs = (
            ("batch=1", 1, False),
            ("batch=16", 16, False),
            ("b1 + b16", 16, True),
            ("batch=64", 64, False),
            ("b1 + b64", 64, True),
        )
        for label, batch_size, overlay in panel_specs:
            panel = frames[frame_index].copy()
            candidate = annotations[batch_size]
            if overlay:
                _draw_pose(
                    panel,
                    reference_points,
                    reference_scores,
                    color=PALETTE["reference"],
                    confidence_threshold=config.transform.confidence_threshold,
                    point_radius=2,
                )
            _draw_pose(
                panel,
                candidate.transformed_keypoints_2d[frame_index],
                candidate.native_keypoint_scores[frame_index],
                color=PALETTE[
                    "reference" if batch_size == 1 else f"batch{batch_size}"
                ],
                confidence_threshold=config.transform.confidence_threshold,
                point_radius=2,
            )
            panel = _scale_panel(panel)
            _label(panel, label)
            if overlay:
                error = frame_errors[batch_size][frame_index]
                _label(
                    panel,
                    f"mean {error.mean():.2f}px / max {error.max():.2f}px",
                    line=1,
                )
            panels.append(panel)
        rows.append(np.hstack(panels))
        frame_stats.append(
            {
                "frame_index": frame_index,
                "batch16": {
                    "mean_joint_distance_px": float(
                        frame_errors[16][frame_index].mean()
                    ),
                    "max_joint_distance_px": float(
                        frame_errors[16][frame_index].max()
                    ),
                    "max_joint_index": int(frame_errors[16][frame_index].argmax()),
                },
                "batch64": {
                    "mean_joint_distance_px": float(
                        frame_errors[64][frame_index].mean()
                    ),
                    "max_joint_distance_px": float(
                        frame_errors[64][frame_index].max()
                    ),
                    "max_joint_index": int(frame_errors[64][frame_index].argmax()),
                },
            }
        )

    import cv2

    gallery = np.vstack(rows)
    gallery_path = output_dir / "batch_1_16_64_review.png"
    if not cv2.imwrite(str(gallery_path), gallery):
        raise RuntimeError(f"Unable to write review image: {gallery_path}")
    summary = {
        "schema_version": "mmprism.csl_news_annotation_batch_review.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "video": str(arguments.video.resolve()),
        "output_image": str(gallery_path),
        "selected_frames": frame_stats,
        "all_frames": {
            str(batch_size): {
                "mean_joint_distance_px": float(errors.mean()),
                "median_joint_distance_px": float(np.median(errors)),
                "p95_joint_distance_px": float(np.percentile(errors, 95)),
                "max_joint_distance_px": float(errors.max()),
            }
            for batch_size, errors in frame_errors.items()
        },
        "runs": runs,
    }
    _write_json(output_dir / "batch_1_16_64_review.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
