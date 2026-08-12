"""Read-only forensic comparison of historical CSL-News pose arrays.

Historical CSL-News exports can contain the intermediate 59-joint RTMW3D
representation instead of the final 2x24 arm-and-hand tensor.  This module
derives that exact intermediate view from a canonical annotation's preserved
133-joint output, then reports numerical differences without altering either
input collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

LEGACY_59_JOINT_INDICES = np.asarray([*range(17), *range(91, 133)], dtype=np.int64)
LEGACY_59_JOINTS = 59
REPORT_SCHEMA_VERSION = "mmprism.csl_news_legacy_pose_comparison.v1"


class CslNewsLegacyPoseComparisonError(RuntimeError):
    """Raised when historical pose evidence cannot be compared safely."""


@dataclass(frozen=True)
class LegacyPoseComparisonPaths:
    """Locations required for one immutable archive-level comparison."""

    legacy_zip: Path
    samples_root: Path
    report_dir: Path
    expected_archive_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CslNewsLegacyPoseComparisonError(
            f"Unable to read annotation sidecar {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CslNewsLegacyPoseComparisonError(f"Annotation sidecar {path} must be a mapping")
    return payload


def _legacy_stem(sidecar: Mapping[str, Any], path: Path) -> str:
    annotation = sidecar.get("annotation")
    if not isinstance(annotation, Mapping):
        raise CslNewsLegacyPoseComparisonError(f"{path} lacks annotation metadata")
    legacy_name = annotation.get("legacy_pose_name")
    if not isinstance(legacy_name, str) or not legacy_name.strip():
        raise CslNewsLegacyPoseComparisonError(f"{path} lacks annotation.legacy_pose_name")
    return Path(legacy_name).stem


def _sidecars_by_legacy_stem(samples_root: Path) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    if not samples_root.is_dir():
        raise CslNewsLegacyPoseComparisonError(
            f"Canonical sample directory does not exist: {samples_root}"
        )
    sidecars: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in sorted(samples_root.glob("*.json")):
        sidecar = _load_json(path)
        stem = _legacy_stem(sidecar, path)
        if stem in sidecars:
            raise CslNewsLegacyPoseComparisonError(
                f"Duplicate canonical legacy-pose identity {stem}: {sidecars[stem][0]} and {path}"
            )
        sidecars[stem] = (path, sidecar)
    if not sidecars:
        raise CslNewsLegacyPoseComparisonError(f"No annotation sidecars found in {samples_root}")
    return sidecars


def derive_legacy_pose59(
    native_keypoints_3d: np.ndarray, depth_center: float
) -> np.ndarray:
    """Recreate the legacy 17-body-plus-42-hand pose view from native RTMW3D.

    The old annotation script centered every native joint by the sequence-wide
    mean z value of joints 6 and 7 before selecting ``[:17] + [-42:]``.
    """

    native = np.asarray(native_keypoints_3d)
    if native.ndim != 3 or native.shape[1:] != (133, 3):
        raise CslNewsLegacyPoseComparisonError(
            f"native_keypoints_3d must have shape [T, 133, 3], got {native.shape}"
        )
    if native.shape[0] == 0 or not np.isfinite(native).all():
        raise CslNewsLegacyPoseComparisonError("native_keypoints_3d must be finite and non-empty")
    if not np.isfinite(depth_center):
        raise CslNewsLegacyPoseComparisonError("depth_center must be finite")

    centered = native.copy()
    centered[..., 2] -= np.asarray(depth_center, dtype=centered.dtype)
    return centered[:, LEGACY_59_JOINT_INDICES, :]


def _array_error_summary(legacy: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    if legacy.shape != current.shape:
        return {
            "comparable": False,
            "reason": "shape_mismatch",
            "legacy_shape": list(legacy.shape),
            "current_shape": list(current.shape),
        }
    if not np.isfinite(legacy).all() or not np.isfinite(current).all():
        return {
            "comparable": False,
            "reason": "non_finite_values",
            "legacy_shape": list(legacy.shape),
            "current_shape": list(current.shape),
        }

    absolute_error = np.abs(legacy.astype(np.float64) - current.astype(np.float64))
    per_axis = absolute_error.mean(axis=(0, 1))
    per_axis_max = absolute_error.max(axis=(0, 1))
    per_joint_mean = absolute_error.mean(axis=(0, 2))
    per_joint_max = absolute_error.max(axis=(0, 2))
    return {
        "comparable": True,
        "legacy_shape": list(legacy.shape),
        "current_shape": list(current.shape),
        "legacy_dtype": str(legacy.dtype),
        "current_dtype": str(current.dtype),
        "array_equal": bool(np.array_equal(legacy, current)),
        "allclose_rtol_1e-5_atol_1e-6": bool(
            np.allclose(legacy, current, rtol=1e-5, atol=1e-6)
        ),
        "absolute_error": {
            "mean": float(absolute_error.mean()),
            "max": float(absolute_error.max()),
            "mean_by_axis_xyz": [float(value) for value in per_axis],
            "max_by_axis_xyz": [float(value) for value in per_axis_max],
            "mean_by_joint_59": [float(value) for value in per_joint_mean],
            "max_by_joint_59": [float(value) for value in per_joint_max],
        },
    }


def _identity_status(sidecar: Mapping[str, Any], expected_archive_sha256: str) -> str:
    source = sidecar.get("source")
    if not isinstance(source, Mapping):
        return "source_identity_unbound"
    integrity = source.get("integrity")
    if not isinstance(integrity, Mapping):
        return "source_identity_unbound"
    archive_sha256 = integrity.get("archive_sha256")
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        return "source_identity_unbound"
    if archive_sha256 == expected_archive_sha256:
        return "source_identity_bound_current"
    return "source_identity_mismatch"


def _comparison_entry(
    *,
    zip_info: zipfile.ZipInfo,
    legacy: np.ndarray,
    sidecar_path: Path | None,
    sidecar: Mapping[str, Any] | None,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    relative_path = PurePosixPath(zip_info.filename)
    legacy_stem = relative_path.stem
    entry: dict[str, Any] = {
        "legacy_zip_member": zip_info.filename,
        "legacy_pose_stem": legacy_stem,
        "legacy_zip_crc32": f"{zip_info.CRC:08x}",
        "legacy_zip_uncompressed_size_bytes": zip_info.file_size,
        "legacy_shape": list(legacy.shape),
        "legacy_dtype": str(legacy.dtype),
    }
    if sidecar_path is None or sidecar is None:
        entry.update({"status": "missing_canonical_sidecar", "comparison": None})
        return entry

    entry["canonical_sidecar_path"] = str(sidecar_path)
    entry["canonical_sample_id"] = sidecar.get("sample_id")
    entry["source_identity"] = _identity_status(sidecar, expected_archive_sha256)
    source = sidecar.get("source")
    annotation = sidecar.get("annotation")
    transform = sidecar.get("transform")
    source_member = source.get("member") if isinstance(source, Mapping) else None
    entry["source_member"] = source_member
    if not isinstance(source_member, str) or Path(source_member).stem != legacy_stem:
        entry.update({"status": "source_member_mismatch", "comparison": None})
        return entry
    caption = annotation.get("text") if isinstance(annotation, Mapping) else None
    entry["caption_sha256"] = _sha256_text(caption) if isinstance(caption, str) else None

    if not isinstance(transform, Mapping):
        entry.update({"status": "canonical_transform_missing", "comparison": None})
        return entry
    depth_center = transform.get("depth_center")
    if isinstance(depth_center, bool) or not isinstance(depth_center, (int, float)):
        entry.update({"status": "canonical_depth_center_missing", "comparison": None})
        return entry

    npz_path = sidecar_path.with_suffix(".npz")
    if not npz_path.is_file():
        entry.update({"status": "canonical_npz_missing", "comparison": None})
        return entry
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            native = arrays["native_keypoints_3d"]
    except (KeyError, OSError, ValueError) as error:
        entry.update({"status": "canonical_npz_invalid", "error": str(error), "comparison": None})
        return entry

    recomputed_depth_center = float(native[:, [6, 7], 2].mean(dtype=np.float64))
    current = derive_legacy_pose59(native, float(depth_center))
    entry["depth_center"] = {
        "policy": transform.get("depth_center_policy"),
        "reported": float(depth_center),
        "recomputed": recomputed_depth_center,
        "reported_minus_recomputed": float(depth_center - recomputed_depth_center),
    }
    entry["canonical_npz_sha256"] = _sha256(npz_path)
    entry["status"] = "compared"
    entry["comparison"] = _array_error_summary(legacy, current)
    return entry


def _safe_zip_members(infos: Iterable[zipfile.ZipInfo]) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    roots: set[str] = set()
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise CslNewsLegacyPoseComparisonError(f"Unsafe ZIP member path: {info.filename}")
        if len(path.parts) != 2 or path.suffix.lower() != ".npy":
            raise CslNewsLegacyPoseComparisonError(
                "Historical ZIP must contain only archive_NNN/<name>.npy members; "
                f"found {info.filename}"
            )
        roots.add(path.parts[0])
        members.append(info)
    if len(roots) != 1:
        raise CslNewsLegacyPoseComparisonError(
            f"Expected one archive root in ZIP, found {sorted(roots)}"
        )
    if not members:
        raise CslNewsLegacyPoseComparisonError("Historical ZIP contains no .npy pose arrays")
    return members


def build_csl_news_legacy_pose_comparison(
    paths: LegacyPoseComparisonPaths,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a deterministic comparison report without writing to either input."""

    legacy_zip = paths.legacy_zip.expanduser().resolve()
    samples_root = paths.samples_root.expanduser().resolve()
    if not legacy_zip.is_file():
        raise CslNewsLegacyPoseComparisonError(f"Historical ZIP does not exist: {legacy_zip}")
    if len(paths.expected_archive_sha256) != 64:
        raise CslNewsLegacyPoseComparisonError(
            "expected_archive_sha256 must be a SHA-256 hex digest"
        )

    sidecars = _sidecars_by_legacy_stem(samples_root)
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(legacy_zip) as archive:
        infos = _safe_zip_members(archive.infolist())
        zip_root = PurePosixPath(infos[0].filename).parts[0]
        for info in sorted(infos, key=lambda candidate: candidate.filename):
            legacy_stem = PurePosixPath(info.filename).stem
            sidecar_pair = sidecars.get(legacy_stem)
            try:
                legacy = np.load(BytesIO(archive.read(info)), allow_pickle=False)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                entries.append(
                    {
                        "legacy_zip_member": info.filename,
                        "legacy_pose_stem": legacy_stem,
                        "status": "legacy_npy_invalid",
                        "error": str(error),
                        "comparison": None,
                    }
                )
                continue
            entries.append(
                _comparison_entry(
                    zip_info=info,
                    legacy=legacy,
                    sidecar_path=sidecar_pair[0] if sidecar_pair else None,
                    sidecar=sidecar_pair[1] if sidecar_pair else None,
                    expected_archive_sha256=paths.expected_archive_sha256,
                )
            )

    uploaded_stems = {entry["legacy_pose_stem"] for entry in entries}
    missing_historical = sorted(set(sidecars) - uploaded_stems)
    by_status = Counter(str(entry["status"]) for entry in entries)
    by_identity = Counter(
        str(entry.get("source_identity", "not_compared")) for entry in entries
    )
    comparable = [
        entry["comparison"]
        for entry in entries
        if isinstance(entry.get("comparison"), Mapping) and entry["comparison"].get("comparable")
    ]
    strict = [
        entry["comparison"]
        for entry in entries
        if entry.get("source_identity") == "source_identity_bound_current"
        and isinstance(entry.get("comparison"), Mapping)
        and entry["comparison"].get("comparable")
    ]

    def comparison_counts(values: list[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "comparable": len(values),
            "array_equal": sum(bool(value.get("array_equal")) for value in values),
            "allclose_rtol_1e-5_atol_1e-6": sum(
                bool(value.get("allclose_rtol_1e-5_atol_1e-6")) for value in values
            ),
        }

    summary: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "legacy_zip": str(legacy_zip),
            "legacy_zip_sha256": _sha256(legacy_zip),
            "canonical_samples_root": str(samples_root),
            "expected_current_source_archive_sha256": paths.expected_archive_sha256,
            "historical_zip_archive_root": zip_root,
        },
        "coverage": {
            "historical_pose_members": len(entries),
            "canonical_sidecars": len(sidecars),
            "historical_members_missing_canonical_sidecar": by_status[
                "missing_canonical_sidecar"
            ],
            "canonical_sidecars_missing_historical_member": len(missing_historical),
            "canonical_sidecars_missing_historical_member_stems": missing_historical,
        },
        "status_counts": dict(sorted(by_status.items())),
        "source_identity_counts": dict(sorted(by_identity.items())),
        "all_comparable_counts": comparison_counts(comparable),
        "strict_current_source_counts": comparison_counts(strict),
        "notes": [
            "Historical arrays are compared as the legacy 59-joint intermediate "
            "representation, not as final 2x24 tensors.",
            "Only source_identity_bound_current entries are eligible for strict "
            "current-source equivalence conclusions.",
            "source_identity_unbound entries remain numerical forensic references "
            "and are excluded from the strict count.",
        ],
    }
    return summary, entries


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csl_news_legacy_pose_comparison(
    paths: LegacyPoseComparisonPaths,
) -> tuple[Path, Path]:
    """Write a summary and JSONL detail report in the configured report directory."""

    summary, entries = build_csl_news_legacy_pose_comparison(paths)
    report_dir = paths.report_dir.expanduser().resolve()
    summary_path = report_dir / "summary.json"
    entries_path = report_dir / "entries.jsonl"
    _write_text_atomic(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        entries_path,
        "".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries),
    )
    return summary_path, entries_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare immutable historical CSL-News pose arrays with canonical "
            "RTMW3D outputs."
        )
    )
    parser.add_argument("--legacy-zip", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = LegacyPoseComparisonPaths(
        legacy_zip=args.legacy_zip,
        samples_root=args.samples_root,
        report_dir=args.report_dir,
        expected_archive_sha256=args.expected_archive_sha256,
    )
    summary_path, entries_path = write_csl_news_legacy_pose_comparison(paths)
    print(summary_path)
    print(entries_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
