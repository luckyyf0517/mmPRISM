#!/usr/bin/env python3
"""Assemble the nine-signer evaluation pack from configured source resources.

The source layout supplies assembly.pool, cells, selection, bigsplit, daily,
radar_catalog and labels_{signer}; outputs.data_pack selects the destination.
Only the named annotation arrays, pose tables and selected radar slices are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mmprism.release.data import SIGNERS
from mmprism.release.paths import Paths


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def assemble(paths: Paths) -> dict:
    source_root = paths.root("data_root")
    destination = paths.resolve("outputs.data_pack")
    destination.mkdir(parents=True, exist_ok=True)
    cells = load(paths.resolve("assembly.cells"))["cells"]
    selected = load(paths.resolve("assembly.selection"))["persons"]
    radar_catalog = load(paths.resolve("assembly.radar_catalog"))
    write(
        destination / "splits/calibration_selection.json",
        {"persons": {signer: selected[signer] for signer in SIGNERS}},
    )
    split_cells = {}
    for cell in ("C", "D"):
        groups = []
        for group in cells[cell]["groups"]:
            signer = group["person"]
            if signer not in SIGNERS:
                continue
            ids = group["segment_ids"]
            if cell == "C":
                ids = [sid for sid in ids if sid in set(selected[signer])]
            if ids:
                groups.append({"person": signer, "text": group["text"], "segment_ids": ids})
        split_cells[cell] = {"groups": groups}
    write(
        destination / "splits/cells.json",
        {
            "description": (
                "Our sentence-disjoint repartition of the source corpus; nine-signer subset."
            ),
            "cells": split_cells,
        },
    )
    original = load(paths.resolve("assembly.bigsplit"))
    write(
        destination / "splits/bigsplit.json",
        {
            "description": (
                "Our repartition of the source corpus; not its official training/test split."
            ),
            "normalization": original["normalization"],
            "big_train": original["big_train"],
            "big_test": original["big_test"],
        },
    )
    write(
        destination / "annotations/camera_intrinsics.json",
        {
            "K": [[323.6941, 0.0, 317.4467], [0.0, 322.9638, 171.8694], [0.0, 0.0, 1.0]],
            "width": 640,
            "height": 360,
        },
    )
    report = {"signers": list(SIGNERS), "per_person": {}, "radar_files": [], "sampled_sha256": {}}
    label_cache = {}
    for signer in SIGNERS:
        source = paths.resolve("assembly.pool", signer=signer)
        rows = {
            q["segment_id"]: q
            for line in (source / "manifest.jsonl").read_text().splitlines()
            if (q := json.loads(line))
        }
        pose = np.load(source / "pose_camera_m.npy", mmap_mode="r")
        valid = np.load(source / "valid_mask.npy", mmap_mode="r")
        evaluation = [
            sid
            for group in cells["D"]["groups"]
            if group["person"] == signer
            for sid in group["segment_ids"]
        ]
        calibration = selected[signer]
        if len(calibration) != 50 or len(set(calibration)) != 50:
            raise ValueError(f"{signer}: expected 50 distinct calibration segments")
        if set(calibration) & set(evaluation):
            raise ValueError(f"{signer}: calibration/evaluation overlap")
        ids = sorted([*calibration, *evaluation])
        person_out = destination / "pool" / signer
        person_out.mkdir(parents=True, exist_ok=True)
        output_rows, poses, masks, offset = [], [], [], 0
        for index, sid in enumerate(ids):
            row = rows[sid]
            frames = int(row["frames"])
            relative = Paths._relative(radar_catalog[sid], sid)
            source_radar = (source_root / relative).resolve()
            if not source_radar.is_relative_to(source_root):
                raise ValueError(f"{sid}: source radar escapes data_root")
            packed = np.load(source_radar, mmap_mode="r")
            radar_offset = int(row["radar_ref"]["frame_offset"])
            values = packed[radar_offset : radar_offset + frames]
            if values.dtype != np.float16 or values.shape != (frames, 64, 116, 32, 2):
                raise ValueError(f"{sid}: compact radar shape/dtype differs")
            if not np.isfinite(values).all():
                raise ValueError(f"{sid}: nonfinite compact radar")
            radar_relative = Path("radar") / signer / sid / "mmwave.fp16.npy"
            radar_out = destination / radar_relative
            radar_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(radar_out, values, allow_pickle=False)
            start = int(row["frame_offset"])
            targets = np.asarray(pose[start : start + frames])
            frame_mask = np.asarray(valid[start : start + frames])
            if not np.isfinite(targets[frame_mask]).all():
                raise ValueError(f"{sid}: nonfinite valid reference pose")
            poses.append(targets)
            masks.append(frame_mask)
            output_rows.append(
                {
                    "segment_id": sid,
                    "caption": row["caption"],
                    "sentence_key": row["sentence_key"],
                    "reference_id": row["reference_id"],
                    "frames": frames,
                    "frame_offset": offset,
                    "radar_ref": {
                        "path": radar_relative.as_posix(),
                        "frame_offset": 0,
                        "frames": frames,
                    },
                }
            )
            offset += frames
            report["radar_files"].append(
                {
                    "segment_id": sid,
                    "path": radar_relative.as_posix(),
                    "frames": frames,
                    "bytes": radar_out.stat().st_size,
                }
            )
            if index in (0, len(ids) - 1):
                with radar_out.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                report["sampled_sha256"][radar_relative.as_posix()] = digest
            if (index + 1) % 50 == 0:
                print(f"{signer}: {index + 1}/{len(ids)} radar segments", flush=True)
        (person_out / "manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
        )
        np.save(person_out / "pose_camera_m.npy", np.concatenate(poses))
        np.save(person_out / "valid_mask.npy", np.concatenate(masks))
        label_root = paths.resolve(f"assembly.labels_{signer}")
        if label_root not in label_cache:
            label_cache[label_root] = (
                pq.read_table(label_root / "parts/part-00000.parquet"),
                load(label_root / "segment_inventory.json"),
            )
        table, inventory = label_cache[label_root]
        subset = table.filter(pc.is_in(table["sample_id"], value_set=pa.array(ids)))
        label_ids = set(subset["sample_id"].to_pylist())
        label_out = destination / "labels3d" / signer
        (label_out / "parts").mkdir(parents=True, exist_ok=True)
        pq.write_table(subset.replace_schema_metadata(None), label_out / "parts/part-00000.parquet")
        write(
            label_out / "segment_inventory.json",
            [row for row in inventory if row["sample_id"] in label_ids],
        )
        report["per_person"][signer] = {
            "evaluation_segments": len(evaluation),
            "calibration_segments": len(calibration),
            "evaluation_frames": sum(rows[sid]["frames"] for sid in evaluation),
            "calibration_frames": sum(rows[sid]["frames"] for sid in calibration),
            "label_frames": len(subset),
            "label_segments": len(label_ids),
        }
        print(signer, report["per_person"][signer], flush=True)
    source_daily = paths.resolve("assembly.daily")
    daily_out = destination / "daily/ce_daily_gtpose_paired"
    (daily_out / "arrays").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_daily / "episodes.jsonl", daily_out / "episodes.jsonl")
    for name in ("gt_full24_m", "frame_mask", "target_source_frame_indices", "target_timestamps_s"):
        shutil.copyfile(
            source_daily / "arrays" / f"{name}.npy", daily_out / "arrays" / f"{name}.npy"
        )
    reader_base = paths.resolve("assembly.reader_base")
    reader_out = destination / "assets/reader_base"
    reader_out.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json",
    ):
        shutil.copyfile(reader_base / name, reader_out / name)
        if name.endswith(".json"):
            metadata = load(reader_out / name)
            metadata.pop("_name_or_path", None)
            metadata.pop("name_or_path", None)
            for key in ("special_tokens_map_file", "tokenizer_file"):
                value = metadata.get(key)
                if isinstance(value, str) and Path(value).is_absolute():
                    metadata.pop(key)
            write(reader_out / name, metadata)
    report["evaluation_segments"] = sum(
        q["evaluation_segments"] for q in report["per_person"].values()
    )
    report["calibration_segments"] = 450
    report["evaluation_frames"] = sum(q["evaluation_frames"] for q in report["per_person"].values())
    report["radar_bytes"] = sum(q["bytes"] for q in report["radar_files"])
    write(destination / "manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--data-root")
    args = parser.parse_args()
    report = assemble(Paths(data_root=args.data_root, config_dir=args.config_dir))
    print(
        json.dumps(
            {k: report[k] for k in ("evaluation_segments", "calibration_segments", "radar_bytes")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
