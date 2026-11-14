"""Camera-visible full-pose error, wrist-relative hand error and hand-point accuracy."""

from __future__ import annotations

import argparse
import json

import numpy as np

from mmprism.release.data import SIGNERS, CellDataset, cell_segments
from mmprism.release.paths import resolve
from mmprism.release.settings import write_report

FIELDS = ("abs_sum_m", "joints", "rel_sum_m", "hand_joints", "pck_hit", "frames")


def visible_joints(target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    calibration = json.loads(resolve("pack.intrinsics").read_text())
    projection = target @ np.asarray(calibration["K"], dtype=np.float64).T
    z = projection[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = projection[..., 0] / z
        v = projection[..., 1] / z
    return (
        (z > 0)
        & (u >= 0)
        & (u < calibration["width"])
        & (v >= 0)
        & (v < calibration["height"])
        & valid[:, None, None]
    )


def metrics(counts: dict) -> dict:
    return dict(
        **counts,
        full24_mm=1000.0 * counts["abs_sum_m"] / counts["joints"],
        wrist_relative_hand20_mm=1000.0 * counts["rel_sum_m"] / counts["hand_joints"],
        pck40_wrist_relative=100.0 * counts["pck_hit"] / counts["hand_joints"],
    )


def score_arrays(prediction, target, valid) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or target.shape != (len(valid), 2, 24, 3):
        raise ValueError("Prediction, target and frame-mask shapes disagree")
    if not (np.isfinite(target[valid]).all() and np.isfinite(prediction[valid]).all()):
        raise ValueError("Nonfinite pose on the declared evaluation support")
    error = np.linalg.norm(prediction - target, axis=-1)
    relative = np.linalg.norm(
        (prediction[..., 4:24, :] - prediction[..., 2:3, :])
        - (target[..., 4:24, :] - target[..., 2:3, :]),
        axis=-1,
    )
    selected = visible_joints(target, valid)
    selected_hand = selected[..., 4:24]
    return {
        "abs_sum_m": float(error[selected].sum()),
        "joints": int(selected.sum()),
        "rel_sum_m": float(relative[selected_hand].sum()),
        "hand_joints": int(selected_hand.sum()),
        "pck_hit": int(((relative * 1000.0) < 40.0)[selected_hand].sum()),
        "frames": len(valid),
    }


def evaluate(signers, *, mode: str) -> dict:
    signers = tuple(dict.fromkeys(signers))
    total = dict.fromkeys(FIELDS, 0)
    per_person, per_segment = {}, []
    for signer in signers:
        dataset = CellDataset(cell_segments("D", signer))
        subtotal = dict.fromkeys(FIELDS, 0)
        for index, segment in enumerate(dataset.segments):
            path = resolve(
                "outputs.reconstruction", mode=mode, signer=signer, segment=segment.sample_id
            )
            with np.load(path, allow_pickle=False) as payload:
                target, valid = dataset.targets(index)
                counts = score_arrays(payload["pose"], target, valid)
            for key in FIELDS:
                total[key] += counts[key]
                subtotal[key] += counts[key]
            per_segment.append(dict(segment_id=segment.sample_id, person=signer, **metrics(counts)))
        per_person[signer] = dict(segments=len(dataset), **metrics(subtotal))
        write_report(
            resolve("outputs.pose_signer_report", mode=mode, signer=signer),
            per_person[signer],
        )
    result = {
        "mode": mode,
        "segments": len(per_segment),
        "pooled": metrics(total),
        "per_person": per_person,
        "per_segment": per_segment,
        "validity": "reference frame validity and camera projection inside the image",
    }
    if set(signers) == set(SIGNERS):
        write_report(resolve("outputs.pose_report", mode=mode), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="mmprism-evaluate", description=__doc__)
    _common(parser)

def _common(parser: argparse.ArgumentParser) -> None:
    """The two options the mmprism command parses before the sub-command."""
    parser.add_argument("--data-root", help="root of the supplied data pack (default from config)")
    parser.add_argument("--config-dir", help="directory holding paths.toml (default: configs/)")

    parser.add_argument("--signer", action="append", choices=SIGNERS)
    parser.add_argument("--mode", choices=("zero-shot", "adapted"), default="zero-shot")
    args = parser.parse_args()
    result = evaluate(args.signer or SIGNERS, mode=args.mode)
    print(json.dumps(dict(segments=result["segments"], **result["pooled"]), indent=2))
    return 0
