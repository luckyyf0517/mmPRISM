from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from mmprism.config import expand_environment
from mmprism.contracts import SampleRecord, validate_manifest
from mmprism.data.csl_news import csl_news_source_program
from mmprism.data.csl_news_annotation import (
    _source_identity_matches,
    annotation_artifact_stem_matches_sample_id,
    load_csl_news_labels,
    sha256_file,
    stable_sample_id,
    validate_annotation_output,
)
from mmprism.data.csl_news_annotation_audit import IDENTITY_AUDIT_SCHEMA_VERSION
from mmprism.data.csl_news_integrity import (
    CslNewsIntegrityArchive,
    passed_csl_news_integrity_archives,
    read_csl_news_integrity_registry_snapshot,
)

POSE_MANIFEST_CONFIG_SCHEMA = "mmprism.csl_news_pose_manifest.v1"
POSE_MANIFEST_SUMMARY_SCHEMA = "mmprism.csl_news_pose_manifest_snapshot.v1"
POSE_SAMPLE_SCHEMA = "mmprism.csl_news_pose_sample.v1"
ARCHIVE_DIRECTORY_PATTERN = re.compile(r"^archive_(\d{3})$")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SAMPLE_ID_PATTERN = re.compile(r"[0-9a-f]{24}")
EXCLUSION_REASON = "sidecar_artifact_identity_conflict"

ARRAY_DTYPES = {
    "native_keypoints_3d": "float32",
    "native_keypoint_scores": "float32",
    "transformed_keypoints_2d": "float32",
    "frame_indices": "int64",
    "timestamps_seconds": "float64",
    "canonical_pose": "float32",
    "canonical_confidence": "float32",
    "canonical_valid": "bool",
}
SIDECAR_ARRAY_NAMES = (
    "native_keypoints_3d",
    "native_keypoint_scores",
    "transformed_keypoints_2d",
    "canonical_pose",
    "canonical_confidence",
    "canonical_valid",
)


class CslNewsPoseManifestError(RuntimeError):
    """Raised when a canonical CSL-News pose manifest cannot be finalized."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CslNewsPoseManifestError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CslNewsPoseManifestError(
            f"Unknown keys in {location}: {', '.join(unknown)}"
        )


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CslNewsPoseManifestError(
            f"{location}.{key} must be a non-empty string"
        )
    return value.strip()


def _integer(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: int
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CslNewsPoseManifestError(
            f"{location}.{key} must be an integer >= {minimum}"
        )
    return value


def _boolean(payload: Mapping[str, Any], key: str, location: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise CslNewsPoseManifestError(f"{location}.{key} must be a boolean")
    return value


def _relative_path(payload: Mapping[str, Any], key: str, location: str) -> Path:
    path = Path(_text(payload, key, location))
    if path.is_absolute() or ".." in path.parts:
        raise CslNewsPoseManifestError(
            f"{location}.{key} must be relative to source.data_root"
        )
    return path


def _runtime_git_commit(runtime_report: Mapping[str, Any]) -> str:
    git = runtime_report.get("git")
    if not isinstance(git, Mapping):
        raise CslNewsPoseManifestError("runtime report has no Git provenance")
    commit = git.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CslNewsPoseManifestError("runtime report has no valid Git commit")
    if git.get("dirty") is not False:
        raise CslNewsPoseManifestError(
            "formal pose snapshots require a clean Git worktree"
        )
    return commit


@dataclass(frozen=True)
class PoseArtifactExclusion:
    sample_id: str
    archive_directory: str
    sidecar_sha256: str
    artifact_size_bytes: int
    artifact_sha256: str
    declared_artifact_size_bytes: int
    declared_artifact_sha256: str
    audit_report_relative: Path
    audit_report_sha256: str
    reason: str

    def portable_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "archive_directory": self.archive_directory,
            "sidecar_sha256": self.sidecar_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_sha256": self.artifact_sha256,
            "declared_artifact_size_bytes": self.declared_artifact_size_bytes,
            "declared_artifact_sha256": self.declared_artifact_sha256,
            "audit_report": self.audit_report_relative.as_posix(),
            "audit_report_sha256": self.audit_report_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CslNewsPoseManifestConfig:
    data_root: Path
    labels_path_relative: Path
    integrity_registry_relative: Path
    annotation_root_relative: Path
    snapshot_root_relative: Path
    source_id: str
    source_revision: str
    expected_archive_count: int
    dataset_id: str
    annotation_config_fingerprint: str
    verify_artifact_checksum: bool
    validate_artifact_contract: bool
    minimum_free_bytes: int
    exclusions: tuple[PoseArtifactExclusion, ...]

    @property
    def labels_path(self) -> Path:
        return (self.data_root / self.labels_path_relative).resolve()

    @property
    def integrity_registry_path(self) -> Path:
        return (self.data_root / self.integrity_registry_relative).resolve()

    @property
    def annotation_root(self) -> Path:
        return (self.data_root / self.annotation_root_relative).resolve()

    @property
    def snapshot_root(self) -> Path:
        return (self.data_root / self.snapshot_root_relative).resolve()

    def portable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POSE_MANIFEST_CONFIG_SCHEMA,
            "source": {
                "labels_path": self.labels_path_relative.as_posix(),
                "integrity_registry": self.integrity_registry_relative.as_posix(),
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "expected_archive_count": self.expected_archive_count,
            },
            "annotation": {
                "root": self.annotation_root_relative.as_posix(),
                "dataset_id": self.dataset_id,
                "config_fingerprint": self.annotation_config_fingerprint,
            },
            "validation": {
                "verify_artifact_checksum": self.verify_artifact_checksum,
                "validate_artifact_contract": self.validate_artifact_contract,
                "minimum_free_bytes": self.minimum_free_bytes,
                "exclusions": [item.portable_dict() for item in self.exclusions],
            },
            "output": {"snapshot_root": self.snapshot_root_relative.as_posix()},
        }

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.portable_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(serialized).hexdigest()


def load_csl_news_pose_manifest_config(
    path: str | Path,
) -> CslNewsPoseManifestConfig:
    """Load a strict, portable pose-manifest configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CslNewsPoseManifestError(
            f"Unable to load pose-manifest config: {error}"
        ) from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise CslNewsPoseManifestError(str(error)) from error

    root = _mapping(expanded, "root")
    _reject_unknown(
        root, {"schema_version", "source", "annotation", "validation", "output"}, "root"
    )
    if root.get("schema_version") != POSE_MANIFEST_CONFIG_SCHEMA:
        raise CslNewsPoseManifestError(
            f"schema_version must be {POSE_MANIFEST_CONFIG_SCHEMA}"
        )
    source = _mapping(root.get("source"), "source")
    _reject_unknown(
        source,
        {
            "data_root",
            "labels_path",
            "integrity_registry",
            "source_id",
            "source_revision",
            "expected_archive_count",
        },
        "source",
    )
    annotation = _mapping(root.get("annotation"), "annotation")
    _reject_unknown(
        annotation, {"root", "dataset_id", "config_fingerprint"}, "annotation"
    )
    validation = _mapping(root.get("validation"), "validation")
    _reject_unknown(
        validation,
        {
            "verify_artifact_checksum",
            "validate_artifact_contract",
            "minimum_free_bytes",
            "exclusions",
        },
        "validation",
    )
    output = _mapping(root.get("output"), "output")
    _reject_unknown(output, {"snapshot_root"}, "output")

    annotation_fingerprint = _text(
        annotation, "config_fingerprint", "annotation"
    )
    if not SHA256_PATTERN.fullmatch(annotation_fingerprint):
        raise CslNewsPoseManifestError(
            "annotation.config_fingerprint must be a lowercase SHA-256 digest"
        )
    raw_exclusions = validation.get("exclusions", [])
    if not isinstance(raw_exclusions, list):
        raise CslNewsPoseManifestError("validation.exclusions must be a list")
    exclusions: list[PoseArtifactExclusion] = []
    exclusion_keys: set[tuple[str, str]] = set()
    for index, raw_exclusion in enumerate(raw_exclusions):
        location = f"validation.exclusions[{index}]"
        exclusion = _mapping(raw_exclusion, location)
        _reject_unknown(
            exclusion,
            {
                "sample_id",
                "archive_directory",
                "sidecar_sha256",
                "artifact_size_bytes",
                "artifact_sha256",
                "declared_artifact_size_bytes",
                "declared_artifact_sha256",
                "audit_report",
                "audit_report_sha256",
                "reason",
            },
            location,
        )
        sample_id = _text(exclusion, "sample_id", location)
        archive_directory = _text(exclusion, "archive_directory", location)
        if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
            raise CslNewsPoseManifestError(
                f"{location}.sample_id must be a 24-character lowercase hex ID"
            )
        if ARCHIVE_DIRECTORY_PATTERN.fullmatch(archive_directory) is None:
            raise CslNewsPoseManifestError(
                f"{location}.archive_directory must match archive_NNN"
            )

        digests: dict[str, str] = {}
        for digest_key in (
            "sidecar_sha256",
            "artifact_sha256",
            "declared_artifact_sha256",
            "audit_report_sha256",
        ):
            digest = _text(exclusion, digest_key, location)
            if not SHA256_PATTERN.fullmatch(digest):
                raise CslNewsPoseManifestError(
                    f"{location}.{digest_key} must be a lowercase SHA-256 digest"
                )
            digests[digest_key] = digest
        audit_report = _relative_path(exclusion, "audit_report", location)
        reason = _text(exclusion, "reason", location)
        if reason != EXCLUSION_REASON:
            raise CslNewsPoseManifestError(
                f"{location}.reason must be {EXCLUSION_REASON}"
            )
        exclusion_key = (archive_directory, sample_id)
        if exclusion_key in exclusion_keys:
            raise CslNewsPoseManifestError(
                f"duplicate pose exclusion for {archive_directory}/{sample_id}"
            )
        exclusion_keys.add(exclusion_key)
        exclusions.append(
            PoseArtifactExclusion(
                sample_id=sample_id,
                archive_directory=archive_directory,
                sidecar_sha256=digests["sidecar_sha256"],
                artifact_size_bytes=_integer(
                    exclusion, "artifact_size_bytes", location, minimum=1
                ),
                artifact_sha256=digests["artifact_sha256"],
                declared_artifact_size_bytes=_integer(
                    exclusion,
                    "declared_artifact_size_bytes",
                    location,
                    minimum=0,
                ),
                declared_artifact_sha256=digests[
                    "declared_artifact_sha256"
                ],
                audit_report_relative=audit_report,
                audit_report_sha256=digests["audit_report_sha256"],
                reason=reason,
            )
        )

    return CslNewsPoseManifestConfig(
        data_root=Path(_text(source, "data_root", "source")).expanduser().resolve(),
        labels_path_relative=_relative_path(source, "labels_path", "source"),
        integrity_registry_relative=_relative_path(
            source, "integrity_registry", "source"
        ),
        annotation_root_relative=_relative_path(annotation, "root", "annotation"),
        snapshot_root_relative=_relative_path(output, "snapshot_root", "output"),
        source_id=_text(source, "source_id", "source"),
        source_revision=_text(source, "source_revision", "source"),
        expected_archive_count=_integer(
            source, "expected_archive_count", "source", minimum=1
        ),
        dataset_id=_text(annotation, "dataset_id", "annotation"),
        annotation_config_fingerprint=annotation_fingerprint,
        verify_artifact_checksum=_boolean(
            validation, "verify_artifact_checksum", "validation"
        ),
        validate_artifact_contract=_boolean(
            validation, "validate_artifact_contract", "validation"
        ),
        minimum_free_bytes=_integer(
            validation, "minimum_free_bytes", "validation", minimum=0
        ),
        exclusions=tuple(exclusions),
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslNewsPoseManifestError(
            f"Unable to load {description} {path}: {error}"
        ) from error
    return dict(_mapping(payload, description))


def _shape(arrays: Mapping[str, Any], name: str, sidecar_path: Path) -> list[int]:
    value = arrays.get(name)
    if not isinstance(value, list) or not value or not all(
        isinstance(size, int) and size >= 0 for size in value
    ):
        raise CslNewsPoseManifestError(
            f"Invalid {name} shape in annotation sidecar {sidecar_path}"
        )
    return list(value)


def _integrity_provenance(
    registry: Mapping[str, Any],
    archive_id: int,
    entry: CslNewsIntegrityArchive,
) -> dict[str, Any]:
    archives = _mapping(registry.get("archives"), "registry.archives")
    raw_entry = _mapping(
        archives.get(f"{archive_id:03d}"),
        f"registry.archives.{archive_id:03d}",
    )
    audit = _mapping(raw_entry.get("audit"), f"registry.archives.{archive_id:03d}.audit")
    audit_path = _text(audit, "path", "archive audit")
    audit_sha256 = _text(audit, "sha256", "archive audit")
    builder_commit = _text(raw_entry, "builder_commit", "archive entry")
    audited_at = _text(raw_entry, "audited_at", "archive entry")
    if not SHA256_PATTERN.fullmatch(audit_sha256):
        raise CslNewsPoseManifestError(
            f"archive_{archive_id:03d} has an invalid audit SHA-256"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", builder_commit):
        raise CslNewsPoseManifestError(
            f"archive_{archive_id:03d} has an invalid audit builder commit"
        )
    return {
        "archive_sha256": entry.sha256,
        "archive_path_relative": entry.archive_path_relative.as_posix(),
        "archive_source_kind": entry.source_kind,
        "archive_size_bytes": entry.size_bytes,
        "archive_mtime_ns": entry.mtime_ns,
        "archive_video_count": entry.video_count,
        "audit_path": audit_path,
        "audit_sha256": audit_sha256,
        "audited_at": audited_at,
        "audit_builder_commit": builder_commit,
    }


def _verify_sidecar_integrity(
    sidecar_integrity: object,
    current: Mapping[str, Any],
    labels_sha256: str,
    sidecar_path: Path,
) -> bool:
    if sidecar_integrity is None:
        raise CslNewsPoseManifestError(
            f"Sidecar has no source-integrity provenance: {sidecar_path}"
        )
    payload = _mapping(sidecar_integrity, f"{sidecar_path}.source.integrity")
    expected = {
        "archive_sha256": current["archive_sha256"],
        "labels_sha256": labels_sha256,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise CslNewsPoseManifestError(
            f"Sidecar integrity provenance mismatch for {sidecar_path}: "
            + ", ".join(mismatches)
        )
    return True


def _select_current_source_sidecars(
    sidecar_paths: list[Path],
    *,
    annotation_root: Path,
    passed_archives: Mapping[int, CslNewsIntegrityArchive],
    labels_sha256: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    selected_by_sample: dict[tuple[int, str], Path] = {}
    quarantined: list[dict[str, Any]] = []
    for sidecar_path in sidecar_paths:
        match = ARCHIVE_DIRECTORY_PATTERN.fullmatch(sidecar_path.parent.name)
        if match is None:
            raise CslNewsPoseManifestError(
                f"invalid archive output directory: {sidecar_path.parent.name}"
            )
        archive_id = int(match.group(1))
        archive_entry = passed_archives.get(archive_id)
        if archive_entry is None:
            continue
        payload = _load_json(sidecar_path, "annotation sidecar source selection")
        sample_id = _text(payload, "sample_id", "sidecar")
        if not annotation_artifact_stem_matches_sample_id(
            sidecar_path.stem, sample_id
        ):
            raise CslNewsPoseManifestError(
                f"sample ID does not match sidecar filename: {sidecar_path}"
            )
        source = _mapping(payload.get("source"), "sidecar.source")
        expected_integrity = {
            "archive_sha256": archive_entry.sha256,
            "labels_sha256": labels_sha256,
        }
        sidecar_integrity = source.get("integrity")
        if not _source_identity_matches(sidecar_integrity, expected_integrity):
            declared = (
                {
                    "archive_sha256": sidecar_integrity.get("archive_sha256"),
                    "labels_sha256": sidecar_integrity.get("labels_sha256"),
                }
                if isinstance(sidecar_integrity, Mapping)
                else None
            )
            quarantined.append(
                {
                    "sidecar": sidecar_path.relative_to(
                        annotation_root
                    ).as_posix(),
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "sample_id": sample_id,
                    "archive_id": f"{archive_id:03d}",
                    "reason": "source_content_identity_mismatch_or_missing",
                    "declared_source_integrity": declared,
                    "expected_source_identity": expected_integrity,
                }
            )
            continue
        key = (archive_id, sample_id)
        previous = selected_by_sample.get(key)
        if previous is not None:
            raise CslNewsPoseManifestError(
                "multiple sidecars bind the current source identity for "
                f"archive_{archive_id:03d}/{sample_id}: {previous}, {sidecar_path}"
            )
        selected_by_sample[key] = sidecar_path
    return sorted(selected_by_sample.values()), quarantined


def _verify_artifact_exclusion(
    config: CslNewsPoseManifestConfig,
    exclusion: PoseArtifactExclusion,
    sidecar_path: Path,
) -> dict[str, Any]:
    expected_sidecar = (
        config.annotation_root
        / "samples"
        / exclusion.archive_directory
        / f"{exclusion.sample_id}.json"
    )
    if sidecar_path != expected_sidecar:
        raise CslNewsPoseManifestError(
            f"pose exclusion path mismatch: {sidecar_path}"
        )
    artifact_path = sidecar_path.with_suffix(".npz")

    sidecar_stat_before = sidecar_path.stat()
    payload = _load_json(sidecar_path, "excluded annotation sidecar")
    sidecar_sha256 = sha256_file(sidecar_path)
    sidecar_stat_after = sidecar_path.stat()
    if (
        sidecar_stat_before.st_size,
        sidecar_stat_before.st_mtime_ns,
    ) != (sidecar_stat_after.st_size, sidecar_stat_after.st_mtime_ns):
        raise CslNewsPoseManifestError(
            f"excluded sidecar changed during verification: {sidecar_path}"
        )
    if sidecar_sha256 != exclusion.sidecar_sha256:
        raise CslNewsPoseManifestError(
            f"excluded sidecar SHA-256 mismatch: {sidecar_path}"
        )
    if payload.get("sample_id") != exclusion.sample_id:
        raise CslNewsPoseManifestError(
            f"excluded sidecar sample ID mismatch: {sidecar_path}"
        )
    if payload.get("config_fingerprint") != config.annotation_config_fingerprint:
        raise CslNewsPoseManifestError(
            f"excluded sidecar config fingerprint mismatch: {sidecar_path}"
        )
    source = _mapping(payload.get("source"), "excluded sidecar.source")
    if source.get("archive") != f"{exclusion.archive_directory}.zip":
        raise CslNewsPoseManifestError(
            f"excluded sidecar archive mismatch: {sidecar_path}"
        )
    declared = _mapping(payload.get("artifact"), "excluded sidecar.artifact")
    if (
        declared.get("size_bytes") != exclusion.declared_artifact_size_bytes
        or declared.get("sha256") != exclusion.declared_artifact_sha256
    ):
        raise CslNewsPoseManifestError(
            f"excluded sidecar declared identity mismatch: {sidecar_path}"
        )

    artifact_stat_before = artifact_path.stat()
    artifact_sha256 = sha256_file(artifact_path)
    artifact_stat_after = artifact_path.stat()
    if (
        artifact_stat_before.st_size,
        artifact_stat_before.st_mtime_ns,
    ) != (artifact_stat_after.st_size, artifact_stat_after.st_mtime_ns):
        raise CslNewsPoseManifestError(
            f"excluded artifact changed during verification: {artifact_path}"
        )
    if (
        artifact_stat_before.st_size != exclusion.artifact_size_bytes
        or artifact_sha256 != exclusion.artifact_sha256
    ):
        raise CslNewsPoseManifestError(
            f"excluded artifact observed identity mismatch: {artifact_path}"
        )

    report_path = (config.annotation_root / exclusion.audit_report_relative).resolve()
    try:
        report_path.relative_to(config.annotation_root)
    except ValueError as error:
        raise CslNewsPoseManifestError(
            f"pose exclusion report escapes annotation root: {report_path}"
        ) from error
    report_stat_before = report_path.stat()
    report = _load_json(report_path, "annotation identity audit")
    report_sha256 = sha256_file(report_path)
    report_stat_after = report_path.stat()
    if (
        report_stat_before.st_size,
        report_stat_before.st_mtime_ns,
    ) != (report_stat_after.st_size, report_stat_after.st_mtime_ns):
        raise CslNewsPoseManifestError(
            f"pose exclusion report changed during verification: {report_path}"
        )
    if report_sha256 != exclusion.audit_report_sha256:
        raise CslNewsPoseManifestError(
            f"pose exclusion report SHA-256 mismatch: {report_path}"
        )
    if (
        report.get("schema_version") != IDENTITY_AUDIT_SCHEMA_VERSION
        or report.get("status") != "failed"
    ):
        raise CslNewsPoseManifestError(
            f"pose exclusion report has an invalid schema or status: {report_path}"
        )
    report_runtime = _mapping(
        report.get("runtime"), "annotation identity audit.runtime"
    )
    report_git = _mapping(
        report_runtime.get("git"), "annotation identity audit.runtime.git"
    )
    audit_git_commit = report_git.get("commit")
    if (
        not isinstance(audit_git_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", audit_git_commit) is None
        or report_git.get("dirty") is not False
    ):
        raise CslNewsPoseManifestError(
            f"pose exclusion report was not generated from a clean Git commit: {report_path}"
        )
    scope = _mapping(report.get("scope"), "annotation identity audit.scope")
    if scope.get("config_fingerprint") != config.annotation_config_fingerprint:
        raise CslNewsPoseManifestError(
            f"pose exclusion report config fingerprint mismatch: {report_path}"
        )
    invalid_pairs = report.get("invalid_pairs")
    if not isinstance(invalid_pairs, list):
        raise CslNewsPoseManifestError(
            f"pose exclusion report has no invalid_pairs list: {report_path}"
        )
    expected_sidecar_relative = sidecar_path.relative_to(
        config.annotation_root
    ).as_posix()
    matches = [
        item
        for item in invalid_pairs
        if isinstance(item, Mapping)
        and item.get("sidecar") == expected_sidecar_relative
        and item.get("sample_id") == exclusion.sample_id
    ]
    if len(matches) != 1:
        raise CslNewsPoseManifestError(
            f"pose exclusion report does not uniquely bind {expected_sidecar_relative}"
        )
    evidence = matches[0]
    sidecar_identity = _mapping(
        evidence.get("sidecar_identity"), "annotation identity audit.sidecar_identity"
    )
    observed = _mapping(
        evidence.get("observed_artifact"),
        "annotation identity audit.observed_artifact",
    )
    report_declared = _mapping(
        evidence.get("declared_artifact"),
        "annotation identity audit.declared_artifact",
    )
    failures = evidence.get("failures")
    if (
        sidecar_identity.get("sha256") != exclusion.sidecar_sha256
        or observed.get("size_bytes") != exclusion.artifact_size_bytes
        or observed.get("sha256") != exclusion.artifact_sha256
        or report_declared.get("size_bytes")
        != exclusion.declared_artifact_size_bytes
        or report_declared.get("sha256") != exclusion.declared_artifact_sha256
        or not isinstance(failures, list)
        or set(failures)
        != {"artifact_size_mismatch", "artifact_sha256_mismatch"}
    ):
        raise CslNewsPoseManifestError(
            f"pose exclusion report identity mismatch: {report_path}"
        )

    evidence_snapshot_path = (
        Path("exclusion_evidence")
        / f"{exclusion.audit_report_sha256[:12]}_{report_path.name}"
    )
    return {
        "sample_id": exclusion.sample_id,
        "archive_directory": exclusion.archive_directory,
        "reason": exclusion.reason,
        "sidecar_path": expected_sidecar_relative,
        "sidecar_sha256": exclusion.sidecar_sha256,
        "artifact_path": artifact_path.relative_to(config.annotation_root).as_posix(),
        "artifact_size_bytes": exclusion.artifact_size_bytes,
        "artifact_sha256": exclusion.artifact_sha256,
        "declared_artifact_size_bytes": exclusion.declared_artifact_size_bytes,
        "declared_artifact_sha256": exclusion.declared_artifact_sha256,
        "audit_report_source_path": exclusion.audit_report_relative.as_posix(),
        "audit_report_snapshot_path": evidence_snapshot_path.as_posix(),
        "audit_report_sha256": exclusion.audit_report_sha256,
        "audit_git_commit": audit_git_commit,
    }


def _manifest_record(
    *,
    config: CslNewsPoseManifestConfig,
    git_commit: str,
    registry_sha256: str,
    labels_sha256: str,
    sidecar_path: Path,
    sidecar_sha256: str,
    artifact_path: Path,
    artifact_sha256: str,
    payload: Mapping[str, Any],
    source_integrity: Mapping[str, Any],
    sidecar_integrity_present: bool,
) -> dict[str, Any]:
    sample_id = _text(payload, "sample_id", "sidecar")
    source = _mapping(payload.get("source"), "sidecar.source")
    annotation = _mapping(payload.get("annotation"), "sidecar.annotation")
    arrays = _mapping(payload.get("arrays"), "sidecar.arrays")
    model = _mapping(payload.get("model"), "sidecar.model")
    transform = _mapping(payload.get("transform"), "sidecar.transform")
    annotation_run = payload.get("run")
    member_name = _text(source, "member", "sidecar.source")
    video_name = PurePosixPath(member_name.replace("\\", "/")).name
    caption = _text(annotation, "text", "sidecar.annotation")
    archive_name = _text(source, "archive", "sidecar.source")
    artifact_relative = artifact_path.relative_to(config.annotation_root).as_posix()
    sidecar_relative = sidecar_path.relative_to(config.annotation_root).as_posix()
    frame_count = _shape(arrays, "canonical_pose", sidecar_path)[0]
    shapes = {name: _shape(arrays, name, sidecar_path) for name in SIDECAR_ARRAY_NAMES}
    shapes["frame_indices"] = [frame_count]
    shapes["timestamps_seconds"] = [frame_count]
    artifact_uri = artifact_relative
    modalities = {
        name: {
            "uri": f"{artifact_uri}#{name}",
            "shape": shapes[name],
            "dtype": dtype,
            "sha256": artifact_sha256,
        }
        for name, dtype in ARRAY_DTYPES.items()
    }
    modalities["caption"] = {
        "text": caption,
        "dtype": "utf-8",
        "sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
    }
    program = csl_news_source_program(video_name)
    compact_transform = {
        key: transform.get(key)
        for key in (
            "crop_top",
            "crop_left",
            "crop_right",
            "bbox_policy",
            "depth_center_policy",
            "depth_center",
            "confidence_threshold",
        )
    }
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": PurePosixPath(video_name).stem,
        "dataset": config.dataset_id,
        "modalities": modalities,
        "group_keys": {
            "archive": Path(archive_name).stem,
            "source_program": program,
        },
        "acquisition": {
            "source_program": program,
            "subject_id_status": "unavailable_in_source_metadata",
            "scene_status": "unavailable_in_source_metadata",
        },
        "provenance": {
            "source_id": config.source_id,
            "source_revision": config.source_revision,
            "archive_name": archive_name,
            "archive_member": member_name,
            "member_crc32": source.get("member_crc32"),
            "member_size_bytes": source.get("member_size_bytes"),
            "video_sha256": source.get("video_sha256"),
            "labels_path": config.labels_path_relative.as_posix(),
            "labels_sha256": labels_sha256,
            "official_pose_name": annotation.get("legacy_pose_name"),
            "source_integrity": dict(source_integrity),
            "integrity_registry_sha256": registry_sha256,
            "sidecar_source_integrity_present": sidecar_integrity_present,
            "annotation": {
                "schema_version": payload.get("schema_version"),
                "config_fingerprint": payload.get("config_fingerprint"),
                "sidecar_path": sidecar_relative,
                "sidecar_sha256": sidecar_sha256,
                "model": {
                    "mmpose_commit": model.get("mmpose_commit"),
                    "config_sha256": model.get("config_sha256"),
                    "checkpoint_sha256": model.get("checkpoint_sha256"),
                },
                "transform": compact_transform,
                "run": dict(annotation_run)
                if isinstance(annotation_run, Mapping)
                else None,
            },
            "artifact": {
                "path": artifact_relative,
                "sha256": artifact_sha256,
                "size_bytes": artifact_path.stat().st_size,
            },
            "builder_schema": POSE_MANIFEST_CONFIG_SCHEMA,
            "builder_config_fingerprint": config.fingerprint,
            "builder_git_commit": git_commit,
        },
    }


def build_csl_news_pose_manifest_snapshot(
    config: CslNewsPoseManifestConfig,
    *,
    runtime_report: Mapping[str, Any],
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Freeze validated pose/text artifacts visible at scan start."""

    git_commit = _runtime_git_commit(runtime_report)
    if not config.annotation_root.is_dir():
        raise CslNewsPoseManifestError(
            f"annotation root does not exist: {config.annotation_root}"
        )
    if not config.labels_path.is_file():
        raise CslNewsPoseManifestError(f"labels do not exist: {config.labels_path}")
    registry, registry_sha256, registry_bytes = (
        read_csl_news_integrity_registry_snapshot(
            config.integrity_registry_path,
            source_id=config.source_id,
            source_revision=config.source_revision,
        )
    )
    passed_archives = passed_csl_news_integrity_archives(registry)
    registry_source = _mapping(registry.get("source"), "registry.source")
    if registry_source.get("expected_archive_count") != config.expected_archive_count:
        raise CslNewsPoseManifestError(
            "integrity registry expected_archive_count does not match config"
        )

    labels_stat_before = config.labels_path.stat()
    labels = load_csl_news_labels(config.labels_path)
    labels_sha256 = sha256_file(config.labels_path)
    labels_stat_after = config.labels_path.stat()
    if (
        labels_stat_before.st_size,
        labels_stat_before.st_mtime_ns,
    ) != (labels_stat_after.st_size, labels_stat_after.st_mtime_ns):
        raise CslNewsPoseManifestError("labels changed while building the snapshot")
    if registry_source.get("labels_sha256") != labels_sha256:
        raise CslNewsPoseManifestError(
            "integrity registry labels SHA-256 does not match canonical labels"
        )

    config.snapshot_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(config.snapshot_root).free
    if free_bytes < config.minimum_free_bytes:
        raise CslNewsPoseManifestError(
            f"snapshot root has {free_bytes} free bytes, below minimum "
            f"{config.minimum_free_bytes}"
        )
    generated_at = datetime.now(UTC)
    identifier = snapshot_id or generated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier):
        raise CslNewsPoseManifestError("snapshot_id contains unsafe characters")
    destination = config.snapshot_root / f"snapshot_{identifier}"
    temporary = config.snapshot_root / f".snapshot_{identifier}.tmp.{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise CslNewsPoseManifestError(
            f"snapshot destination already exists: {destination}"
        )
    temporary.mkdir()

    sample_root = config.annotation_root / "samples"
    sidecar_paths = sorted(
        path
        for path in sample_root.glob("archive_*/*.json")
        if not path.name.startswith(".")
    )
    npz_paths = sorted(
        path
        for path in sample_root.glob("archive_*/*.npz")
        if not path.name.startswith(".")
    )
    eligible_stems = {f"archive_{archive_id:03d}" for archive_id in passed_archives}
    archive_eligible_sidecars = [
        path for path in sidecar_paths if path.parent.name in eligible_stems
    ]
    eligible_sidecars, source_quarantine = _select_current_source_sidecars(
        archive_eligible_sidecars,
        annotation_root=config.annotation_root,
        passed_archives=passed_archives,
        labels_sha256=labels_sha256,
    )
    exclusion_by_sidecar = {
        Path("samples")
        / exclusion.archive_directory
        / f"{exclusion.sample_id}.json": exclusion
        for exclusion in config.exclusions
    }
    eligible_sidecar_relatives = {
        path.relative_to(config.annotation_root) for path in eligible_sidecars
    }
    unmatched_exclusions = sorted(
        path.as_posix()
        for path in set(exclusion_by_sidecar) - eligible_sidecar_relatives
    )
    if unmatched_exclusions:
        raise CslNewsPoseManifestError(
            "configured pose exclusions are not frozen eligible sidecars: "
            + ", ".join(unmatched_exclusions)
        )
    archive_ineligible_sidecar_count = len(sidecar_paths) - len(
        archive_eligible_sidecars
    )
    archive_ineligible_npz_count = sum(
        path.parent.name not in eligible_stems for path in npz_paths
    )
    ineligible_sidecar_count = archive_ineligible_sidecar_count + len(
        source_quarantine
    )
    frozen_sidecar_identities = {
        path.relative_to(sample_root).with_suffix("") for path in eligible_sidecars
    }
    archive_eligible_sidecar_identities = {
        path.relative_to(sample_root).with_suffix("")
        for path in archive_eligible_sidecars
    }
    frozen_npz_identities = {
        path.relative_to(sample_root).with_suffix("")
        for path in npz_paths
        if path.parent.name in eligible_stems
    }
    source_quarantined_npz_count = len(
        (archive_eligible_sidecar_identities - frozen_sidecar_identities)
        & frozen_npz_identities
    )
    ineligible_npz_count = (
        archive_ineligible_npz_count + source_quarantined_npz_count
    )
    missing_artifacts = sorted(frozen_sidecar_identities - frozen_npz_identities)
    unpaired_npz_at_scan = sorted(
        frozen_npz_identities - archive_eligible_sidecar_identities
    )
    if missing_artifacts:
        raise CslNewsPoseManifestError(
            f"{len(missing_artifacts)} frozen sidecars have no NPZ artifact"
        )
    if not eligible_sidecars:
        raise CslNewsPoseManifestError("no eligible completed pose sidecars are available")
    if len(eligible_sidecars) == len(config.exclusions):
        raise CslNewsPoseManifestError(
            "all eligible pose sidecars are excluded; no manifest records remain"
        )

    manifest_path = temporary / "manifest.jsonl"
    sample_ids: set[str] = set()
    represented_archives: Counter[str] = Counter()
    program_counts: Counter[str] = Counter()
    sidecar_integrity_present_count = 0
    artifact_bytes = 0
    applied_exclusions: list[dict[str, Any]] = []
    with manifest_path.open("w", encoding="utf-8") as manifest_stream:
        for sidecar_path in eligible_sidecars:
            sidecar_relative = sidecar_path.relative_to(config.annotation_root)
            exclusion = exclusion_by_sidecar.get(sidecar_relative)
            if exclusion is not None:
                applied_exclusions.append(
                    _verify_artifact_exclusion(config, exclusion, sidecar_path)
                )
                continue
            match = ARCHIVE_DIRECTORY_PATTERN.fullmatch(sidecar_path.parent.name)
            if match is None:
                raise CslNewsPoseManifestError(
                    f"invalid archive output directory: {sidecar_path.parent.name}"
                )
            archive_id = int(match.group(1))
            archive_entry = passed_archives.get(archive_id)
            if archive_entry is None:
                raise CslNewsPoseManifestError(
                    f"sidecar archive is not integrity-passed: {sidecar_path}"
                )
            current_integrity = _integrity_provenance(
                registry, archive_id, archive_entry
            )
            sidecar_stat_before = sidecar_path.stat()
            payload = _load_json(sidecar_path, "annotation sidecar")
            sidecar_sha256 = sha256_file(sidecar_path)
            sidecar_stat_after = sidecar_path.stat()
            if (
                sidecar_stat_before.st_size,
                sidecar_stat_before.st_mtime_ns,
            ) != (sidecar_stat_after.st_size, sidecar_stat_after.st_mtime_ns):
                raise CslNewsPoseManifestError(
                    f"sidecar changed while building snapshot: {sidecar_path}"
                )
            if payload.get("schema_version") != POSE_SAMPLE_SCHEMA:
                raise CslNewsPoseManifestError(
                    f"unsupported pose sidecar schema: {sidecar_path}"
                )
            if payload.get("status") != "completed":
                raise CslNewsPoseManifestError(
                    f"pose sidecar is not completed: {sidecar_path}"
                )
            if (
                payload.get("config_fingerprint")
                != config.annotation_config_fingerprint
            ):
                raise CslNewsPoseManifestError(
                    f"annotation config fingerprint mismatch: {sidecar_path}"
                )
            sample_id = _text(payload, "sample_id", "sidecar")
            if not annotation_artifact_stem_matches_sample_id(
                sidecar_path.stem, sample_id
            ):
                raise CslNewsPoseManifestError(
                    f"sample ID does not match sidecar filename: {sidecar_path}"
                )
            source = _mapping(payload.get("source"), "sidecar.source")
            archive_name = _text(source, "archive", "sidecar.source")
            if archive_name != f"archive_{archive_id:03d}.zip":
                raise CslNewsPoseManifestError(
                    f"archive identity mismatch in sidecar: {sidecar_path}"
                )
            if source.get("source_id") != config.source_id:
                raise CslNewsPoseManifestError(
                    f"source_id mismatch in sidecar: {sidecar_path}"
                )
            if source.get("source_revision") != config.source_revision:
                raise CslNewsPoseManifestError(
                    f"source_revision mismatch in sidecar: {sidecar_path}"
                )
            member_name = _text(source, "member", "sidecar.source")
            expected_sample_id = stable_sample_id(
                config.source_id, archive_name, member_name
            )
            if sample_id != expected_sample_id:
                raise CslNewsPoseManifestError(
                    f"stable sample ID mismatch in sidecar: {sidecar_path}"
                )
            video_name = PurePosixPath(member_name.replace("\\", "/")).name
            label = labels.get(video_name)
            annotation = _mapping(payload.get("annotation"), "sidecar.annotation")
            if label is None or annotation.get("text") != label.text:
                raise CslNewsPoseManifestError(
                    f"canonical caption mismatch in sidecar: {sidecar_path}"
                )
            if annotation.get("legacy_pose_name") != label.legacy_pose_name:
                raise CslNewsPoseManifestError(
                    f"official pose name mismatch in sidecar: {sidecar_path}"
                )
            integrity_present = _verify_sidecar_integrity(
                source.get("integrity"),
                current_integrity,
                labels_sha256,
                sidecar_path,
            )
            sidecar_integrity_present_count += int(integrity_present)

            artifact_path = sidecar_path.with_suffix(".npz")
            artifact_stat_before = artifact_path.stat()
            if config.validate_artifact_contract and not validate_annotation_output(
                artifact_path
            ):
                raise CslNewsPoseManifestError(
                    f"annotation artifact contract failed: {artifact_path}"
                )
            artifact = _mapping(payload.get("artifact"), "sidecar.artifact")
            expected_artifact_sha256 = _text(
                artifact, "sha256", "sidecar.artifact"
            )
            if not SHA256_PATTERN.fullmatch(expected_artifact_sha256):
                raise CslNewsPoseManifestError(
                    f"invalid artifact SHA-256 in sidecar: {sidecar_path}"
                )
            artifact_sha256 = (
                sha256_file(artifact_path)
                if config.verify_artifact_checksum
                else expected_artifact_sha256
            )
            if artifact_sha256 != expected_artifact_sha256:
                raise CslNewsPoseManifestError(
                    f"annotation artifact checksum mismatch: {artifact_path}"
                )
            if artifact.get("size_bytes") != artifact_stat_before.st_size:
                raise CslNewsPoseManifestError(
                    f"annotation artifact size mismatch: {artifact_path}"
                )
            artifact_stat_after = artifact_path.stat()
            if (
                artifact_stat_before.st_size,
                artifact_stat_before.st_mtime_ns,
            ) != (artifact_stat_after.st_size, artifact_stat_after.st_mtime_ns):
                raise CslNewsPoseManifestError(
                    f"artifact changed while building snapshot: {artifact_path}"
                )
            record = _manifest_record(
                config=config,
                git_commit=git_commit,
                registry_sha256=registry_sha256,
                labels_sha256=labels_sha256,
                sidecar_path=sidecar_path,
                sidecar_sha256=sidecar_sha256,
                artifact_path=artifact_path,
                artifact_sha256=artifact_sha256,
                payload=payload,
                source_integrity=current_integrity,
                sidecar_integrity_present=integrity_present,
            )
            if sample_id in sample_ids:
                raise CslNewsPoseManifestError(
                    f"duplicate stable sample ID: {sample_id}"
                )
            sample_ids.add(sample_id)
            manifest_stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            represented_archives[sidecar_path.parent.name] += 1
            program_counts[csl_news_source_program(video_name)] += 1
            artifact_bytes += artifact_stat_before.st_size

    if len(applied_exclusions) != len(config.exclusions):
        raise CslNewsPoseManifestError(
            "not every configured pose exclusion was applied exactly once"
        )

    exclusion_evidence_checksums: dict[str, str] = {}
    for applied in applied_exclusions:
        exclusion = exclusion_by_sidecar[Path(applied["sidecar_path"])]
        destination_relative = Path(applied["audit_report_snapshot_path"])
        destination_path = temporary / destination_relative
        if destination_relative.as_posix() in exclusion_evidence_checksums:
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = config.annotation_root / exclusion.audit_report_relative
        shutil.copyfile(source_path, destination_path)
        copied_sha256 = sha256_file(destination_path)
        if copied_sha256 != exclusion.audit_report_sha256:
            raise CslNewsPoseManifestError(
                f"copied pose exclusion evidence hash mismatch: {source_path}"
            )
        exclusion_evidence_checksums[destination_relative.as_posix()] = copied_sha256

    contract_summary = validate_manifest(manifest_path)
    if contract_summary.record_count != len(sample_ids):
        raise CslNewsPoseManifestError(
            "manifest contract count differs from builder count"
        )
    manifest_sha256 = sha256_file(manifest_path)
    source_quarantine_path = temporary / "source_identity_quarantine.jsonl"
    source_quarantine_path.write_text(
        "".join(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in source_quarantine
        ),
        encoding="utf-8",
    )
    source_quarantine_sha256 = sha256_file(source_quarantine_path)
    registry_copy_path = temporary / "integrity_registry.json"
    registry_copy_path.write_bytes(registry_bytes)
    registry_copy_sha256 = sha256_file(registry_copy_path)
    if registry_copy_sha256 != registry_sha256:
        raise CslNewsPoseManifestError("copied integrity registry hash mismatch")
    complete = (
        len(passed_archives) == config.expected_archive_count
        and len(sample_ids) + len(applied_exclusions) == len(labels)
    )
    summary = {
        "schema_version": POSE_MANIFEST_SUMMARY_SCHEMA,
        "generated_at": generated_at.isoformat(),
        "status": "complete" if complete else "partial",
        "config_fingerprint": config.fingerprint,
        "portable_config": config.portable_dict(),
        "runtime": dict(runtime_report),
        "source": {
            "source_id": config.source_id,
            "source_revision": config.source_revision,
            "expected_archive_count": config.expected_archive_count,
            "labels_sha256": labels_sha256,
            "label_record_count": len(labels),
            "integrity_registry_path": "integrity_registry.json",
            "integrity_registry_sha256": registry_sha256,
            "passed_archive_count": len(passed_archives),
            "passed_archive_ids": sorted(passed_archives),
        },
        "annotation": {
            "dataset_id": config.dataset_id,
            "config_fingerprint": config.annotation_config_fingerprint,
            "frozen_sidecar_count": len(eligible_sidecars),
            "included_sidecar_count": len(sample_ids),
            "excluded_sidecar_count": len(applied_exclusions),
            "exclusions": applied_exclusions,
            "represented_archive_count": len(represented_archives),
            "represented_archive_sample_counts": dict(sorted(represented_archives.items())),
            "sidecar_integrity_present_count": sidecar_integrity_present_count,
            "sidecar_integrity_missing_count": len(sample_ids)
            - sidecar_integrity_present_count,
            "source_identity_quarantined_sidecar_count": len(source_quarantine),
            "source_identity_quarantine_path": "source_identity_quarantine.jsonl",
            "source_identity_quarantine_sha256": source_quarantine_sha256,
            "archive_ineligible_sidecar_count": archive_ineligible_sidecar_count,
            "ineligible_sidecar_count": ineligible_sidecar_count,
            "archive_ineligible_npz_count": archive_ineligible_npz_count,
            "source_identity_quarantined_npz_count": source_quarantined_npz_count,
            "ineligible_npz_count": ineligible_npz_count,
            "unpaired_npz_at_scan_count": len(unpaired_npz_at_scan),
            "unpaired_npz_at_scan_examples": [
                path.as_posix() for path in unpaired_npz_at_scan[:20]
            ],
        },
        "manifest": {
            "path": "manifest.jsonl",
            "sha256": manifest_sha256,
            "record_count": len(sample_ids),
            "datasets": list(contract_summary.datasets),
            "modalities": list(contract_summary.modalities),
            "program_counts": dict(sorted(program_counts.items())),
        },
        "storage": {
            "artifact_root_relative": config.annotation_root_relative.as_posix(),
            "artifact_bytes_referenced": artifact_bytes,
            "free_bytes_before_snapshot": free_bytes,
        },
    }
    summary_path = temporary / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = {
        **exclusion_evidence_checksums,
        "integrity_registry.json": registry_sha256,
        "manifest.jsonl": manifest_sha256,
        "source_identity_quarantine.jsonl": source_quarantine_sha256,
        "summary.json": sha256_file(summary_path),
    }
    (temporary / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="ascii",
    )
    temporary.replace(destination)
    return {
        "schema_version": POSE_MANIFEST_SUMMARY_SCHEMA,
        "status": summary["status"],
        "snapshot_dir": str(destination),
        "manifest_path": str(destination / "manifest.jsonl"),
        "summary_path": str(destination / "summary.json"),
        "record_count": len(sample_ids),
        "represented_archive_count": len(represented_archives),
        "manifest_sha256": manifest_sha256,
        "integrity_registry_sha256": registry_sha256,
    }


@dataclass(frozen=True)
class CslNewsPoseSample:
    sample_id: str
    caption: str
    arrays: dict[str, np.ndarray]
    provenance: dict[str, Any]


class CslNewsPoseManifest:
    """Dependency-light random-access reader for a canonical pose/text manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        artifact_root: str | Path,
        *,
        verify_checksum: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.verify_checksum = verify_checksum
        if not self.manifest_path.is_file():
            raise CslNewsPoseManifestError(
                f"pose manifest does not exist: {self.manifest_path}"
            )
        if not self.artifact_root.is_dir():
            raise CslNewsPoseManifestError(
                f"pose artifact root does not exist: {self.artifact_root}"
            )
        self._offsets: list[int] = []
        with self.manifest_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    continue
                self._validate_record(raw_line, len(self._offsets) + 1)
                self._offsets.append(offset)
        if not self._offsets:
            raise CslNewsPoseManifestError("pose manifest contains no records")

    def _validate_record(self, raw_line: bytes, line_number: int) -> SampleRecord:
        try:
            payload: object = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CslNewsPoseManifestError(
                f"invalid pose manifest JSON at line {line_number}: {error}"
            ) from error
        try:
            record = SampleRecord.from_mapping(payload, f"line {line_number}")
        except ValueError as error:
            raise CslNewsPoseManifestError(str(error)) from error
        required = set(ARRAY_DTYPES) | {"caption"}
        missing = sorted(required - set(record.modalities))
        if missing:
            raise CslNewsPoseManifestError(
                f"line {line_number} lacks required modalities: {', '.join(missing)}"
            )
        caption = record.modalities["caption"]
        if caption.text is None:
            raise CslNewsPoseManifestError(
                f"line {line_number} caption must be inline text"
            )
        artifact_uris = {
            reference.uri.rpartition("#")[0]
            for name, reference in record.modalities.items()
            if name in ARRAY_DTYPES and reference.uri is not None
        }
        if len(artifact_uris) != 1:
            raise CslNewsPoseManifestError(
                f"line {line_number} arrays must share one NPZ container"
            )
        artifact_hashes = {
            reference.sha256
            for name, reference in record.modalities.items()
            if name in ARRAY_DTYPES
        }
        if len(artifact_hashes) != 1 or None in artifact_hashes:
            raise CslNewsPoseManifestError(
                f"line {line_number} arrays must share one NPZ checksum"
            )
        for name, expected_dtype in ARRAY_DTYPES.items():
            reference = record.modalities[name]
            if reference.uri is None or reference.uri.rpartition("#")[2] != name:
                raise CslNewsPoseManifestError(
                    f"line {line_number} has an invalid {name} NPZ reference"
                )
            if reference.dtype != expected_dtype or reference.shape is None:
                raise CslNewsPoseManifestError(
                    f"line {line_number} has an invalid {name} array contract"
                )
        return record

    def __len__(self) -> int:
        return len(self._offsets)

    def record(self, index: int) -> SampleRecord:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self.manifest_path.open("rb") as stream:
            stream.seek(self._offsets[index])
            raw_line = stream.readline()
        return self._validate_record(raw_line, index + 1)

    def __getitem__(self, index: int) -> CslNewsPoseSample:
        record = self.record(index)
        first_reference = record.modalities["canonical_pose"]
        assert first_reference.uri is not None
        artifact_relative = first_reference.uri.rpartition("#")[0]
        relative_path = Path(artifact_relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CslNewsPoseManifestError(
                f"artifact URI escapes configured root: {artifact_relative}"
            )
        artifact_path = (self.artifact_root / relative_path).resolve()
        try:
            artifact_path.relative_to(self.artifact_root)
        except ValueError as error:
            raise CslNewsPoseManifestError(
                f"artifact URI escapes configured root: {artifact_relative}"
            ) from error
        if not artifact_path.is_file():
            raise CslNewsPoseManifestError(f"pose artifact is missing: {artifact_path}")
        expected_sha256 = first_reference.sha256
        if self.verify_checksum and (
            expected_sha256 is None or sha256_file(artifact_path) != expected_sha256
        ):
            raise CslNewsPoseManifestError(
                f"pose artifact checksum mismatch: {artifact_path}"
            )
        arrays: dict[str, np.ndarray] = {}
        try:
            with np.load(artifact_path, allow_pickle=False) as container:
                for name, expected_dtype in ARRAY_DTYPES.items():
                    reference = record.modalities[name]
                    array = np.asarray(container[name])
                    if tuple(array.shape) != reference.shape:
                        raise CslNewsPoseManifestError(
                            f"{name} shape mismatch for {record.sample_id}"
                        )
                    if array.dtype.name != expected_dtype:
                        raise CslNewsPoseManifestError(
                            f"{name} dtype mismatch for {record.sample_id}"
                        )
                    arrays[name] = array
        except (OSError, ValueError, KeyError) as error:
            raise CslNewsPoseManifestError(
                f"unable to load pose artifact {artifact_path}: {error}"
            ) from error
        caption = record.modalities["caption"].text
        assert caption is not None
        return CslNewsPoseSample(
            sample_id=record.sample_id,
            caption=caption,
            arrays=arrays,
            provenance=dict(record.provenance or {}),
        )
