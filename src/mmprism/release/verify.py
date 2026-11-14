"""Existence checks for data, and in-place SHA-256 checks for original weights."""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path
from typing import Any

from mmprism.release.paths import PACKAGE_ROOT, Paths

REQUIRED_PACK_FILES = {
    "pool": ("manifest.jsonl", "pose_camera_m.npy", "valid_mask.npy"),
    "reader_base": ("config.json", "tokenizer_config.json", "spiece.model"),
    "daily_annotations": (
        "episodes.jsonl",
        "arrays/gt_full24_m.npy",
        "arrays/frame_mask.npy",
        "arrays/target_source_frame_indices.npy",
        "arrays/target_timestamps_s.npy",
    ),
}


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash original bytes through a read-only mapping with bounded hash updates."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if path.stat().st_size:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                for offset in range(0, len(mapped), chunk_bytes):
                    digest.update(mapped[offset : offset + chunk_bytes])
    return digest.hexdigest()


def verify_data(paths: Paths, *, signers: list[str] | None = None) -> list[dict[str, Any]]:
    """Check declared files, radar headers and the pack's sampled byte digests."""
    import numpy as np

    manifest = json.loads(paths.resolve("pack.manifest").read_text())
    signers = list(manifest["signers"]) if signers is None else signers
    unknown = set(signers) - set(manifest["signers"])
    if unknown:
        raise ValueError(f"Unknown signers: {', '.join(sorted(unknown))}")
    results = []
    for name, template in paths.layout.get("pack", {}).items():
        if "{segment}" in template:
            continue
        if "{signer}" in template:
            resources = [paths.resolve(f"pack.{name}", signer=signer) for signer in signers]
        else:
            resources = [paths.resolve(f"pack.{name}")]
        for path in resources:
            results.append({"key": f"pack.{name}", "path": str(path), "exists": path.exists()})
            for member in REQUIRED_PACK_FILES.get(name, ()):
                required = path / member
                results.append(
                    {
                        "key": f"pack.{name}/{member}",
                        "path": str(required),
                        "exists": required.is_file(),
                    }
                )
    for signer in signers:
        files = [
            row for row in manifest["radar_files"] if row["segment_id"].startswith(signer + ":")
        ]
        failures = []
        sampled = 0
        for row in files:
            path = paths.resolve("pack.radar", signer=signer, segment=row["segment_id"])
            if not path.is_file():
                failures.append(f"{row['segment_id']}: missing radar file")
                continue
            try:
                with path.open("rb") as stream:
                    version = np.lib.format.read_magic(stream)
                    read_header = (
                        np.lib.format.read_array_header_1_0
                        if version[0] == 1
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, fortran, dtype = read_header(stream)
            except (OSError, ValueError, EOFError) as error:
                failures.append(f"{row['segment_id']}: cannot read radar header: {error}")
                continue
            if shape != (row["frames"], 64, 116, 32, 2) or dtype != np.float16 or fortran:
                failures.append(f"{row['segment_id']}: radar shape or dtype differs")
            if path.stat().st_size != row["bytes"]:
                failures.append(f"{row['segment_id']}: byte count differs")
            expected = manifest["sampled_sha256"].get(row["path"])
            if expected:
                sampled += 1
                if sha256_file(path) != expected:
                    failures.append(f"{row['segment_id']}: sampled digest differs")
        expected_count = sum(
            manifest["per_person"][signer][key]
            for key in ("evaluation_segments", "calibration_segments")
        )
        if len(files) != expected_count:
            failures.append(f"{signer}: radar segment count differs")
        results.append(
            {
                "key": "pack.radar",
                "signer": signer,
                "exists": not failures,
                "matches": not failures,
                "segments": len(files),
                "sampled_digests": sampled,
                "failures": failures,
            }
        )
    return results


def verify_weights(paths: Paths, *, manifest: Path | None = None) -> list[dict[str, Any]]:
    """Check only existence and original-file hashes; never deserialize weights."""
    payload = json.loads((manifest or PACKAGE_ROOT / "manifests/weights_manifest.json").read_text())
    results = []
    for key, item in payload["weights"].items():
        path = paths.resolve(f"weights.{key}")
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        results.append(
            {
                "key": key,
                "path": str(path),
                "exists": exists,
                "sha256": actual,
                "expected_sha256": item["sha256"],
                "matches": exists and actual == item["sha256"],
            }
        )
    return results
