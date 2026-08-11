from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import yaml

from mmprism.config import expand_environment
from mmprism.contracts import validate_manifest
from mmprism.data.csl_news import csl_news_source_program
from mmprism.data.csl_news_annotation import (
    load_csl_news_labels,
    sha256_file,
    stable_sample_id,
)

SOURCE_MANIFEST_CONFIG_SCHEMA = "mmprism.csl_news_source_manifest.v1"
SOURCE_MANIFEST_SUMMARY_SCHEMA = "mmprism.csl_news_source_manifest_snapshot.v1"
ARCHIVE_PATTERN = re.compile(r"^archive_(\d{3})\.zip$")


class CslNewsSourceManifestError(RuntimeError):
    """Raised when a canonical CSL-News source snapshot cannot be finalized."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CslNewsSourceManifestError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CslNewsSourceManifestError(
            f"Unknown keys in {location}: {', '.join(unknown)}"
        )


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CslNewsSourceManifestError(
            f"{location}.{key} must be a non-empty string"
        )
    return value.strip()


def _integer(
    payload: Mapping[str, Any], key: str, location: str, *, minimum: int
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CslNewsSourceManifestError(
            f"{location}.{key} must be an integer >= {minimum}"
        )
    return value


def _boolean(payload: Mapping[str, Any], key: str, location: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise CslNewsSourceManifestError(f"{location}.{key} must be a boolean")
    return value


def _relative_path(payload: Mapping[str, Any], key: str, location: str) -> Path:
    value = Path(_text(payload, key, location))
    if value.is_absolute() or ".." in value.parts:
        raise CslNewsSourceManifestError(
            f"{location}.{key} must be relative to source.data_root"
        )
    return value


@dataclass(frozen=True)
class CslNewsSourceManifestConfig:
    data_root: Path
    archive_root_relative: Path
    labels_path_relative: Path
    snapshot_root_relative: Path
    source_id: str
    source_revision: str
    expected_archive_count: int
    verify_crc: bool
    minimum_free_bytes: int

    @property
    def archive_root(self) -> Path:
        return (self.data_root / self.archive_root_relative).resolve()

    @property
    def labels_path(self) -> Path:
        return (self.data_root / self.labels_path_relative).resolve()

    @property
    def snapshot_root(self) -> Path:
        return (self.data_root / self.snapshot_root_relative).resolve()

    def portable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_MANIFEST_CONFIG_SCHEMA,
            "source": {
                "archive_root": self.archive_root_relative.as_posix(),
                "labels_path": self.labels_path_relative.as_posix(),
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "expected_archive_count": self.expected_archive_count,
            },
            "validation": {
                "verify_crc": self.verify_crc,
                "minimum_free_bytes": self.minimum_free_bytes,
            },
            "output": {"snapshot_root": self.snapshot_root_relative.as_posix()},
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.portable_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def load_csl_news_source_manifest_config(
    path: str | Path,
) -> CslNewsSourceManifestConfig:
    """Load a strict, portable CSL-News source-manifest configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CslNewsSourceManifestError(
            f"Unable to load source-manifest config: {error}"
        ) from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise CslNewsSourceManifestError(str(error)) from error
    root = _mapping(expanded, "root")
    _reject_unknown(root, {"schema_version", "source", "validation", "output"}, "root")
    if root.get("schema_version") != SOURCE_MANIFEST_CONFIG_SCHEMA:
        raise CslNewsSourceManifestError(
            f"schema_version must be {SOURCE_MANIFEST_CONFIG_SCHEMA}"
        )

    source = _mapping(root.get("source"), "source")
    _reject_unknown(
        source,
        {
            "data_root",
            "archive_root",
            "labels_path",
            "source_id",
            "source_revision",
            "expected_archive_count",
        },
        "source",
    )
    validation = _mapping(root.get("validation"), "validation")
    _reject_unknown(
        validation, {"verify_crc", "minimum_free_bytes"}, "validation"
    )
    output = _mapping(root.get("output"), "output")
    _reject_unknown(output, {"snapshot_root"}, "output")

    data_root = Path(_text(source, "data_root", "source")).expanduser().resolve()
    return CslNewsSourceManifestConfig(
        data_root=data_root,
        archive_root_relative=_relative_path(source, "archive_root", "source"),
        labels_path_relative=_relative_path(source, "labels_path", "source"),
        snapshot_root_relative=_relative_path(output, "snapshot_root", "output"),
        source_id=_text(source, "source_id", "source"),
        source_revision=_text(source, "source_revision", "source"),
        expected_archive_count=_integer(
            source, "expected_archive_count", "source", minimum=1
        ),
        verify_crc=_boolean(validation, "verify_crc", "validation"),
        minimum_free_bytes=_integer(
            validation, "minimum_free_bytes", "validation", minimum=0
        ),
    )


def _is_unsafe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return path.is_absolute() or ".." in path.parts


def _runtime_git_commit(runtime_report: Mapping[str, Any]) -> str:
    git = runtime_report.get("git")
    if not isinstance(git, Mapping):
        raise CslNewsSourceManifestError("runtime report has no Git provenance")
    commit = git.get("commit")
    dirty = git.get("dirty")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CslNewsSourceManifestError("runtime report has no valid Git commit")
    if dirty is not False:
        raise CslNewsSourceManifestError(
            "formal source snapshots require a clean Git worktree"
        )
    return commit


def _archive_files(config: CslNewsSourceManifestConfig) -> list[tuple[int, Path]]:
    if not config.archive_root.is_dir():
        raise CslNewsSourceManifestError(
            f"archive root does not exist: {config.archive_root}"
        )
    archives: list[tuple[int, Path]] = []
    invalid_names: list[str] = []
    for path in config.archive_root.glob("archive_*.zip"):
        match = ARCHIVE_PATTERN.fullmatch(path.name)
        if match is None:
            invalid_names.append(path.name)
            continue
        archive_id = int(match.group(1))
        if not 1 <= archive_id <= config.expected_archive_count:
            invalid_names.append(path.name)
            continue
        archives.append((archive_id, path.resolve()))
    if invalid_names:
        raise CslNewsSourceManifestError(
            f"invalid completed archive names: {', '.join(sorted(invalid_names)[:20])}"
        )
    if not archives:
        raise CslNewsSourceManifestError("no complete CSL-News archives are available")
    return sorted(archives)


def _manifest_record(
    *,
    config: CslNewsSourceManifestConfig,
    git_commit: str,
    archive_name: str,
    archive_sha256: str,
    labels_sha256: str,
    member: zipfile.ZipInfo,
    video_name: str,
    caption: str,
    official_pose_name: str | None,
) -> dict[str, Any]:
    member_name = member.filename.replace("\\", "/")
    sample_id = stable_sample_id(config.source_id, archive_name, member_name)
    program = csl_news_source_program(video_name)
    caption_sha256 = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": PurePosixPath(video_name).stem,
        "dataset": "csl_news",
        "modalities": {
            "video": {"uri": f"zip://{archive_name}!/{quote(member_name, safe='/')}"},
            "caption": {
                "text": caption,
                "dtype": "utf-8",
                "sha256": caption_sha256,
            },
        },
        "group_keys": {"archive": Path(archive_name).stem, "source_program": program},
        "acquisition": {
            "source_program": program,
            "subject_id_status": "unavailable_in_source_metadata",
            "scene_status": "unavailable_in_source_metadata",
        },
        "provenance": {
            "source_id": config.source_id,
            "source_revision": config.source_revision,
            "archive_name": archive_name,
            "archive_sha256": archive_sha256,
            "archive_member": member_name,
            "member_crc32": f"{member.CRC:08x}",
            "member_compressed_size_bytes": member.compress_size,
            "member_uncompressed_size_bytes": member.file_size,
            "labels_path": config.labels_path_relative.as_posix(),
            "labels_sha256": labels_sha256,
            "official_pose_name": official_pose_name,
            "builder_schema": SOURCE_MANIFEST_CONFIG_SCHEMA,
            "config_fingerprint": config.fingerprint,
            "git_commit": git_commit,
        },
    }


def build_csl_news_source_manifest_snapshot(
    config: CslNewsSourceManifestConfig,
    *,
    runtime_report: Mapping[str, Any],
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Build and atomically finalize a manifest for archives complete at scan start."""

    git_commit = _runtime_git_commit(runtime_report)
    archive_files = _archive_files(config)
    part_file_count_at_scan = len(list(config.archive_root.glob("archive_*.zip.part")))
    if not config.labels_path.is_file() or config.labels_path.name.endswith(".part"):
        raise CslNewsSourceManifestError(
            f"labels must be a complete JSON file: {config.labels_path}"
        )
    labels_stat_before = config.labels_path.stat()
    labels = load_csl_news_labels(config.labels_path)
    labels_sha256 = sha256_file(config.labels_path)
    labels_stat_after = config.labels_path.stat()
    if (
        labels_stat_before.st_size,
        labels_stat_before.st_mtime_ns,
    ) != (labels_stat_after.st_size, labels_stat_after.st_mtime_ns):
        raise CslNewsSourceManifestError("labels changed while building the snapshot")

    config.snapshot_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(config.snapshot_root).free
    if free_bytes < config.minimum_free_bytes:
        raise CslNewsSourceManifestError(
            f"snapshot root has {free_bytes} free bytes, below "
            f"minimum {config.minimum_free_bytes}"
        )
    generated_at = datetime.now(UTC)
    identifier = snapshot_id or generated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier):
        raise CslNewsSourceManifestError("snapshot_id contains unsafe characters")
    destination = config.snapshot_root / f"snapshot_{identifier}"
    temporary = config.snapshot_root / f".snapshot_{identifier}.tmp.{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise CslNewsSourceManifestError(
            f"snapshot destination already exists: {destination}"
        )
    temporary.mkdir()
    manifest_path = temporary / "manifest.jsonl"

    seen_video_names: set[str] = set()
    sample_ids: set[str] = set()
    archive_reports: list[dict[str, Any]] = []
    program_counts: Counter[str] = Counter()
    record_count = 0
    total_compressed_member_bytes = 0
    total_uncompressed_member_bytes = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_stream:
        for archive_id, archive_path in archive_files:
            archive_stat_before = archive_path.stat()
            archive_sha256 = sha256_file(archive_path)
            try:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    members = [member for member in archive.infolist() if not member.is_dir()]
                    unsafe_members = [
                        member.filename
                        for member in members
                        if _is_unsafe_member(member.filename)
                    ]
                    encrypted_members = [
                        member.filename for member in members if member.flag_bits & 0x1
                    ]
                    member_counts = Counter(member.filename for member in members)
                    duplicate_members = [
                        name for name, count in member_counts.items() if count > 1
                    ]
                    video_members = sorted(
                        (
                            member
                            for member in members
                            if PurePosixPath(member.filename.replace("\\", "/"))
                            .suffix.lower()
                            == ".mp4"
                        ),
                        key=lambda member: member.filename,
                    )
                    video_names = [
                        PurePosixPath(member.filename.replace("\\", "/")).name
                        for member in video_members
                    ]
                    duplicate_video_names = [
                        name
                        for name, count in Counter(video_names).items()
                        if count > 1
                    ]
                    missing_labels = [name for name in video_names if name not in labels]
                    cross_archive_duplicates = [
                        name for name in video_names if name in seen_video_names
                    ]
                    crc_failure = archive.testzip() if config.verify_crc else None
                    failures: list[str] = []
                    if not video_members:
                        failures.append("archive contains no MP4 videos")
                    if unsafe_members:
                        failures.append(f"{len(unsafe_members)} unsafe member paths")
                    if encrypted_members:
                        failures.append(f"{len(encrypted_members)} encrypted members")
                    if duplicate_members or duplicate_video_names:
                        failures.append("duplicate archive members or video basenames")
                    if missing_labels:
                        failures.append(f"{len(missing_labels)} videos have no JSON label")
                    if cross_archive_duplicates:
                        failures.append(
                            f"{len(cross_archive_duplicates)} video names repeat across archives"
                        )
                    if crc_failure is not None:
                        failures.append(f"ZIP CRC failure: {crc_failure}")
                    if failures:
                        raise CslNewsSourceManifestError(
                            f"{archive_path.name} failed source-manifest validation: "
                            + "; ".join(failures)
                        )

                    archive_program_counts: Counter[str] = Counter()
                    for member, video_name in zip(video_members, video_names, strict=True):
                        label = labels[video_name]
                        record = _manifest_record(
                            config=config,
                            git_commit=git_commit,
                            archive_name=archive_path.name,
                            archive_sha256=archive_sha256,
                            labels_sha256=labels_sha256,
                            member=member,
                            video_name=video_name,
                            caption=label.text,
                            official_pose_name=label.legacy_pose_name,
                        )
                        sample_id = record["sample_id"]
                        if sample_id in sample_ids:
                            raise CslNewsSourceManifestError(
                                f"stable sample ID collision: {sample_id}"
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
                        program = csl_news_source_program(video_name)
                        archive_program_counts[program] += 1
                        program_counts[program] += 1
                        record_count += 1
                    seen_video_names.update(video_names)
                    compressed_bytes = sum(member.compress_size for member in members)
                    uncompressed_bytes = sum(member.file_size for member in members)
                    total_compressed_member_bytes += compressed_bytes
                    total_uncompressed_member_bytes += uncompressed_bytes
                    archive_reports.append(
                        {
                            "archive_id": archive_id,
                            "archive_name": archive_path.name,
                            "archive_path_relative": (
                                config.archive_root_relative / archive_path.name
                            ).as_posix(),
                            "archive_size_bytes": archive_path.stat().st_size,
                            "archive_sha256": archive_sha256,
                            "member_count": len(members),
                            "video_count": len(video_members),
                            "compressed_member_bytes": compressed_bytes,
                            "uncompressed_member_bytes": uncompressed_bytes,
                            "program_counts": dict(sorted(archive_program_counts.items())),
                            "crc_checked": config.verify_crc,
                            "crc_failure": crc_failure,
                        }
                    )
                    archive_stat_after = archive_path.stat()
                    if (
                        archive_stat_before.st_size,
                        archive_stat_before.st_mtime_ns,
                    ) != (archive_stat_after.st_size, archive_stat_after.st_mtime_ns):
                        raise CslNewsSourceManifestError(
                            f"{archive_path.name} changed while building the snapshot"
                        )
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise CslNewsSourceManifestError(
                    f"unable to read {archive_path}: {error}"
                ) from error

    contract_summary = validate_manifest(manifest_path)
    if contract_summary.record_count != record_count:
        raise CslNewsSourceManifestError(
            "manifest contract count differs from builder count"
        )
    manifest_sha256 = sha256_file(manifest_path)
    represented_label_count = len(seen_video_names)
    unrepresented_label_count = len(labels) - represented_label_count
    complete = (
        len(archive_files) == config.expected_archive_count
        and unrepresented_label_count == 0
    )
    summary = {
        "schema_version": SOURCE_MANIFEST_SUMMARY_SCHEMA,
        "generated_at": generated_at.isoformat(),
        "status": "complete" if complete else "partial",
        "config_fingerprint": config.fingerprint,
        "portable_config": config.portable_dict(),
        "runtime": dict(runtime_report),
        "source": {
            "source_id": config.source_id,
            "source_revision": config.source_revision,
            "expected_archive_count": config.expected_archive_count,
            "snapshot_archive_count": len(archive_files),
            "snapshot_archive_ids": [archive_id for archive_id, _ in archive_files],
            "part_file_count_at_scan": part_file_count_at_scan,
            "labels_path_relative": config.labels_path_relative.as_posix(),
            "labels_sha256": labels_sha256,
            "label_record_count": len(labels),
            "represented_label_count": represented_label_count,
            "unrepresented_label_count": unrepresented_label_count,
            "label_coverage_ratio": represented_label_count / len(labels),
        },
        "manifest": {
            "path": "manifest.jsonl",
            "sha256": manifest_sha256,
            "record_count": record_count,
            "datasets": list(contract_summary.datasets),
            "modalities": list(contract_summary.modalities),
            "program_counts": dict(sorted(program_counts.items())),
        },
        "storage": {
            "free_bytes_before_snapshot": free_bytes,
            "total_compressed_member_bytes": total_compressed_member_bytes,
            "total_uncompressed_member_bytes": total_uncompressed_member_bytes,
        },
        "archives": archive_reports,
    }
    summary_path = temporary / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "schema_version": SOURCE_MANIFEST_SUMMARY_SCHEMA,
        "status": summary["status"],
        "snapshot_dir": str(destination),
        "manifest_path": str(destination / "manifest.jsonl"),
        "summary_path": str(destination / "summary.json"),
        "record_count": record_count,
        "snapshot_archive_count": len(archive_files),
        "manifest_sha256": manifest_sha256,
    }
