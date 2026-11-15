"""The camera-pose reference decode, computed instead of asserted.

The paper reports retained performance as a ratio to a camera-pose decode. That decode is itself a
reader adapted on the same signer's calibration segments, run on camera poses rather than on
reconstructed ones. This module produces it with the released code path: it materialises the camera
poses of the calibration and evaluation segments in the same stream format the reconstruction
writes, adapts the reader through ``reader.adapt_reader(mode="camera")``, decodes through
``reader.translate_signer(mode="camera")`` and scores through ``translation_metrics``.

Everything else in the package is untouched: the same nine signers, the same calibration
selection, the same 1,416 evaluation segments, the same stage-three recipe.
"""

from __future__ import annotations

import json

import numpy as np

from mmprism.release.data import SIGNERS, calibration_ids, cell_segments, person_data, person_rows
from mmprism.release.paths import resolve

CAMERA_MODE = "camera"


def pose_stream_path(signer: str, segment: str):
    return resolve("outputs.reconstruction", signer=signer, segment=segment, mode=CAMERA_MODE)


def write_camera_streams(signer: str, *, force: bool = False) -> dict:
    """Write pose_camera_m slices for the calibration and evaluation segments of one signer."""
    rows, poses, masks = person_data(signer)
    wanted = list(calibration_ids(signer)) + [segment.sample_id for segment in cell_segments("D", signer)]
    written = skipped = 0
    for segment in dict.fromkeys(wanted):
        row = rows[segment]
        path = pose_stream_path(signer, segment)
        if path.is_file() and not force:
            skipped += 1
            continue
        start = int(row["frame_offset"])
        stop = start + int(row["frames"])
        pose = np.asarray(poses[start:stop], dtype=np.float32)
        mask = np.asarray(masks[start:stop], dtype=bool)
        finite = np.isfinite(pose).reshape(pose.shape[0], -1).all(axis=1)
        pose = np.where(finite.reshape((-1, 1, 1, 1)), pose, np.nan).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pose=pose, mask=mask.astype(np.uint8))
        written += 1
    return {"signer": signer, "written": written, "skipped": skipped}


def score_reference(signers=SIGNERS, *, device: str = "cpu") -> dict:
    from mmprism.release.translation_metrics import score_translation

    return score_translation(list(signers), mode=CAMERA_MODE, reader="adapted", device=device)


def build_reference(signers=SIGNERS, *, device: str = "cpu", force: bool = False) -> dict:
    """Adapt the reader on camera poses, decode, score, and write the reference file."""
    from mmprism.release.reader import adapt_reader, translate_signer

    for signer in signers:
        write_camera_streams(signer, force=force)
        adapt_reader(signer, mode=CAMERA_MODE, device=device)
        translate_signer(signer, mode=CAMERA_MODE, reader="adapted", device=device)

    result = score_reference(signers, device=device)
    pooled = result["pooled"]
    payload = {
        "description": (
            "Camera-pose reference decode, computed by `mmprism-reference`: the reader adapted on "
            "the camera poses of each signer's calibration segments and decoded on camera poses. "
            "Used as the denominator of relative performance."
        ),
        "mode": CAMERA_MODE,
        "signers": list(signers),
        "metrics": {key: pooled[key] for key in ("bleu4", "rouge_l", "sbert", "simcse")},
        "per_person": result.get("per_person", {}),
    }
    path = resolve("outputs.translation_reference")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"reference": payload, "path": str(path)}


def compare_with_printed(computed: dict, printed_path=None) -> dict:
    from mmprism.release.paths import PACKAGE_ROOT

    source = json.loads((printed_path or (PACKAGE_ROOT / "configs/translation_reference.json")).read_text())
    rows = {}
    for key, value in computed["metrics"].items():
        expected = float(source["metrics"][key])
        rows[key] = {
            "computed": round(float(value), 4),
            "printed": round(expected, 4),
            "delta": round(float(value) - expected, 4),
            "relative": round(float(value) / expected - 1.0, 4),
        }
    return rows
