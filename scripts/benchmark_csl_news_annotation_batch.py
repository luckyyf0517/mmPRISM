#!/usr/bin/env python3
"""Verify RTMW3D frame-batch parity and measure inference throughput."""

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
    load_csl_news_annotation_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RTMW3D single-frame and mini-batch inference on one video."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _run_one(
    *,
    config_path: Path,
    project_root: Path,
    video: Path,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = load_csl_news_annotation_config(config_path, project_root)
    batched_config = replace(
        config,
        runtime=replace(config.runtime, inference_batch_size=batch_size),
    )
    estimator = MMPoseRtmw3dEstimator(batched_config)
    torch = estimator._torch  # The benchmark must synchronize the pinned inference runtime.
    if batched_config.model.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    annotation = estimator.annotate_video(video)
    if batched_config.model.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    result = {
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "frame_count": int(annotation.frame_indices.size),
        "frames_per_second": int(annotation.frame_indices.size) / elapsed_seconds,
        "runtime": dict(estimator.runtime_metadata()),
    }
    arrays = {
        "native_keypoints_3d": annotation.native_keypoints_3d,
        "native_keypoint_scores": annotation.native_keypoint_scores,
        "transformed_keypoints_2d": annotation.transformed_keypoints_2d,
    }
    del estimator
    gc.collect()
    if batched_config.model.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, arrays


def main() -> int:
    arguments = _parse_args()
    batch_sizes = tuple(dict.fromkeys(arguments.batch_size))
    if any(size < 1 for size in batch_sizes):
        raise ValueError("Every --batch-size must be positive")
    if 1 not in batch_sizes:
        raise ValueError("--batch-size 1 is required as the parity reference")
    if not arguments.video.is_file():
        raise FileNotFoundError(f"Video does not exist: {arguments.video}")

    results: list[dict[str, Any]] = []
    reference: dict[str, np.ndarray] | None = None
    for batch_size in batch_sizes:
        result, arrays = _run_one(
            config_path=arguments.config,
            project_root=arguments.project_root,
            video=arguments.video,
            batch_size=batch_size,
        )
        if reference is None:
            reference = arrays
            result["parity"] = {"reference": True}
        else:
            result["parity"] = {
                name: _maximum_absolute_difference(reference[name], arrays[name])
                for name in sorted(reference)
            }
        results.append(result)

    non_reference = [item for item in results if item["batch_size"] != 1]
    max_difference = max(
        (
            value
            for item in non_reference
            for value in item["parity"].values()
            if isinstance(value, float)
        ),
        default=0.0,
    )
    payload = {
        "schema_version": "mmprism.csl_news_annotation_batch_benchmark.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(arguments.config.resolve()),
        "video": str(arguments.video.resolve()),
        "acceptance": {
            "maximum_absolute_difference": max_difference,
            "tolerance": 1e-5,
            "passed": max_difference <= 1e-5,
        },
        "runs": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if max_difference <= 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
