#!/usr/bin/env python3
"""Measure CSL-Daily RTMW3D batch-64 throughput against the v2 batch-1 path.

This is intentionally a read-only benchmark. It never claims a scheduler
lease and never writes pose, NPZ, QC, or manifest artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mmprism.data.csl_daily_pose_annotation import (
    MMPoseRtmw3dFrameEstimator,
    build_sequence_annotation_v2,
    list_sequence_frames,
    load_csl_daily_pose_annotation_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CSL-Daily RTMW3D batch=1 and batched inference."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--sequence-id", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"shape mismatch: {left.shape} != {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _run(
    estimator: MMPoseRtmw3dFrameEstimator, paths: list[Path], batch_size: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    torch = estimator._torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions = estimator.estimate_frames_batched(paths, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    keypoints = np.stack([item.keypoints for item in predictions]).astype(np.float32, copy=False)
    scores = np.stack([item.scores for item in predictions]).astype(np.float32, copy=False)
    runtime = dict(estimator.runtime_metadata())
    runtime["inference_batch_size"] = batch_size
    return (
        {
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "frame_count": len(paths),
            "frames_per_second": len(paths) / elapsed,
            "runtime": runtime,
        },
        keypoints,
        scores,
    )


def _annotation_summary(
    keypoints: np.ndarray, scores: np.ndarray, config: Any) -> dict[str, Any]:
    annotation = build_sequence_annotation_v2(
        keypoints,
        scores,
        confidence_threshold=config.transform.confidence_threshold,
        minimum_valid_joints_per_frame=config.transform.minimum_valid_joints_per_frame,
        minimum_valid_frame_ratio=config.transform.minimum_valid_frame_ratio,
    )
    return {
        "canonical_pose": annotation.canonical_pose,
        "canonical_confidence": annotation.canonical_confidence,
        "canonical_valid": annotation.canonical_valid,
        "qc_reasons": list(annotation.qc_reasons),
        "valid_frame_ratio": annotation.valid_frame_ratio,
    }


def main() -> int:
    args = _parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for this comparison")
    if args.max_frames_per_sequence < 0:
        raise ValueError("--max-frames-per-sequence must be non-negative")
    config = load_csl_daily_pose_annotation_config(
        args.config, args.project_root, variables=dict(__import__("os").environ)
    )
    estimator = MMPoseRtmw3dFrameEstimator(config)
    sequences: list[dict[str, Any]] = []
    aggregate_frames = 0
    aggregate_b1_seconds = 0.0
    aggregate_batched_seconds = 0.0
    for sequence_id in dict.fromkeys(args.sequence_id):
        paths = list_sequence_frames(config.source.sequence_root / sequence_id)
        if args.max_frames_per_sequence:
            paths = paths[: args.max_frames_per_sequence]
        if not paths:
            raise FileNotFoundError(f"sequence contains no image frames: {sequence_id}")
        baseline, keypoints_b1, scores_b1 = _run(estimator, paths, 1)
        batched, keypoints_batched, scores_batched = _run(estimator, paths, args.batch_size)
        canonical_b1 = _annotation_summary(keypoints_b1, scores_b1, config)
        canonical_batched = _annotation_summary(keypoints_batched, scores_batched, config)
        sequences.append(
            {
                "sequence_id": sequence_id,
                "frame_count": len(paths),
                "batch_1": baseline,
                f"batch_{args.batch_size}": batched,
                "differences": {
                    "native_keypoints_3d_max_abs": _max_abs(keypoints_b1, keypoints_batched),
                    "native_keypoint_scores_max_abs": _max_abs(scores_b1, scores_batched),
                    "canonical_pose_max_abs": _max_abs(
                        canonical_b1["canonical_pose"], canonical_batched["canonical_pose"]
                    ),
                    "canonical_confidence_max_abs": _max_abs(
                        canonical_b1["canonical_confidence"],
                        canonical_batched["canonical_confidence"],
                    ),
                    "canonical_valid_mismatch_count": int(
                        np.count_nonzero(
                            canonical_b1["canonical_valid"]
                            != canonical_batched["canonical_valid"]
                        )
                    ),
                    "qc_reasons_batch_1": canonical_b1["qc_reasons"],
                    f"qc_reasons_batch_{args.batch_size}": canonical_batched["qc_reasons"],
                    "valid_frame_ratio_batch_1": canonical_b1["valid_frame_ratio"],
                    f"valid_frame_ratio_batch_{args.batch_size}": canonical_batched[
                        "valid_frame_ratio"
                    ],
                },
            }
        )
        aggregate_frames += len(paths)
        aggregate_b1_seconds += baseline["elapsed_seconds"]
        aggregate_batched_seconds += batched["elapsed_seconds"]
    payload = {
        "schema_version": "mmprism.csl_daily_annotation_batch_benchmark.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(args.config.resolve()),
        "source_root": str(config.source.sequence_root),
        "batch_size": args.batch_size,
        "max_frames_per_sequence": args.max_frames_per_sequence or None,
        "aggregate": {
            "frame_count": aggregate_frames,
            "batch_1_frames_per_second": aggregate_frames / aggregate_b1_seconds,
            f"batch_{args.batch_size}_frames_per_second": (
                aggregate_frames / aggregate_batched_seconds
            ),
            "throughput_multiplier": aggregate_b1_seconds / aggregate_batched_seconds,
        },
        "sequences": sequences,
        "interpretation": (
            "measurement only; the benchmark does not establish numerical equivalence "
            "or change the canonical annotation_v2 worker configuration"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
