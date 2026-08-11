from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
import zipfile
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


class CslNewsAuditError(ValueError):
    """Raised when a CSL-News source audit cannot be executed safely."""


@dataclass(frozen=True)
class CslNewsLabelIndex:
    record_count: int
    invalid_record_count: int
    duplicate_video_count: int
    video_names: frozenset[str]
    empty_text_video_names: frozenset[str]


@dataclass(frozen=True)
class CslNewsAuditContext:
    labels_path: Path
    labels_sha256: str
    labels_size_bytes: int
    labels_mtime_ns: int
    label_index: CslNewsLabelIndex


def _require_complete_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name.endswith(".part"):
        raise CslNewsAuditError(f"{description} must be complete, not a .part file: {resolved}")
    if not resolved.is_file():
        raise CslNewsAuditError(f"{description} does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(path: Path) -> CslNewsLabelIndex:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload: object = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslNewsAuditError(f"Invalid CSL-News labels JSON: {path}: {error}") from error

    if not isinstance(payload, list):
        raise CslNewsAuditError(f"CSL-News labels must be a JSON list: {path}")

    video_counts: Counter[str] = Counter()
    empty_text_video_names: set[str] = set()
    invalid_record_count = 0
    for item in payload:
        if not isinstance(item, Mapping):
            invalid_record_count += 1
            continue

        video = item.get("video")
        if not isinstance(video, str) or not video.strip():
            invalid_record_count += 1
            continue

        video_name = PurePosixPath(video.strip().replace("\\", "/")).name
        video_counts[video_name] += 1
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            empty_text_video_names.add(video_name)

    return CslNewsLabelIndex(
        record_count=len(payload),
        invalid_record_count=invalid_record_count,
        duplicate_video_count=sum(count > 1 for count in video_counts.values()),
        video_names=frozenset(video_counts),
        empty_text_video_names=frozenset(empty_text_video_names),
    )


def load_csl_news_audit_context(labels_path: str | Path) -> CslNewsAuditContext:
    """Load immutable label metadata once for a batch of source audits."""

    labels_file = _require_complete_file(Path(labels_path), "CSL-News labels")
    labels_stat = labels_file.stat()
    return CslNewsAuditContext(
        labels_path=labels_file,
        labels_sha256=_sha256(labels_file),
        labels_size_bytes=labels_stat.st_size,
        labels_mtime_ns=labels_stat.st_mtime_ns,
        label_index=_load_labels(labels_file),
    )


def _normalized_member_path(name: str) -> PurePosixPath:
    return PurePosixPath(name.replace("\\", "/"))


def _is_unsafe_member(name: str) -> bool:
    path = _normalized_member_path(name)
    return path.is_absolute() or ".." in path.parts


def csl_news_source_program(video_name: str) -> str:
    if "Common-Concerns" in video_name:
        return "Common-Concerns"
    if "Dragon-TV" in video_name:
        return "Dragon-TV"
    return "unknown"


def verify_zip_crc(archive: zipfile.ZipFile) -> tuple[str | None, str | None]:
    """Read every member and return the first member-level integrity failure."""

    for member in archive.infolist():
        if member.is_dir():
            continue
        try:
            with archive.open(member, "r") as stream:
                while stream.read(8 * 1024 * 1024):
                    pass
        except (EOFError, OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
            return member.filename, f"{type(error).__name__}: {error}"
    return None, None


def _select_probe_members(
    video_members: list[zipfile.ZipInfo], sample_count: int
) -> list[zipfile.ZipInfo]:
    if sample_count < 0:
        raise CslNewsAuditError("decode sample count must be non-negative")
    if sample_count == 0 or not video_members:
        return []
    if sample_count >= len(video_members):
        return video_members
    if sample_count == 1:
        return [video_members[len(video_members) // 2]]

    indices = {
        round(index * (len(video_members) - 1) / (sample_count - 1))
        for index in range(sample_count)
    }
    return [video_members[index] for index in sorted(indices)]


def _probe_video_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    temporary_root: Path,
    probe_index: int,
) -> dict[str, Any]:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as error:
        raise CslNewsAuditError(
            "OpenCV is required when --decode-samples is greater than zero"
        ) from error

    target = temporary_root / f"probe_{probe_index:03d}.mp4"
    with archive.open(member, "r") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)

    capture = cv2.VideoCapture(str(target))
    opened = bool(capture.isOpened())
    decoded_frames = 0
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        while opened:
            readable, _ = capture.read()
            if not readable:
                break
            decoded_frames += 1
    finally:
        capture.release()

    return {
        "member": member.filename,
        "compressed_size_bytes": member.compress_size,
        "uncompressed_size_bytes": member.file_size,
        "opened": opened,
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
        "decoded_frames": decoded_frames,
        "passed": opened and decoded_frames > 0,
    }


def audit_csl_news_archive(
    archive_path: str | Path,
    labels_path: str | Path,
    *,
    source_id: str,
    verify_crc: bool = True,
    decode_sample_count: int = 0,
    scratch_dir: str | Path | None = None,
    audit_context: CslNewsAuditContext | None = None,
) -> dict[str, Any]:
    """Audit one immutable CSL-News archive against the official labels."""

    archive_file = _require_complete_file(Path(archive_path), "CSL-News archive")
    labels_file = _require_complete_file(Path(labels_path), "CSL-News labels")
    if not source_id.strip():
        raise CslNewsAuditError("source_id must be a non-empty string")
    if decode_sample_count < 0:
        raise CslNewsAuditError("decode sample count must be non-negative")

    context = audit_context or load_csl_news_audit_context(labels_file)
    if context.labels_path != labels_file:
        raise CslNewsAuditError(
            "audit context labels path does not match the requested labels file"
        )
    labels_stat = labels_file.stat()
    if (
        labels_stat.st_size != context.labels_size_bytes
        or labels_stat.st_mtime_ns != context.labels_mtime_ns
    ):
        raise CslNewsAuditError("CSL-News labels changed after audit context creation")
    label_index = context.label_index
    scratch_root = Path(scratch_dir).expanduser().resolve() if scratch_dir else None
    if scratch_root is not None:
        scratch_root.mkdir(parents=True, exist_ok=True)

    members: list[zipfile.ZipInfo] = []
    video_members: list[zipfile.ZipInfo] = []
    video_names: list[str] = []
    member_counts: Counter[str] = Counter()
    video_name_counts: Counter[str] = Counter()
    unsafe_members: list[str] = []
    encrypted_members: list[str] = []
    crc_failure: str | None = None
    crc_error: str | None = None
    probes: list[dict[str, Any]] = []
    archive_read_error: str | None = None
    try:
        with zipfile.ZipFile(archive_file, "r") as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            video_members = sorted(
                (
                    member
                    for member in members
                    if _normalized_member_path(member.filename).suffix.lower() == ".mp4"
                ),
                key=lambda member: member.filename,
            )
            video_names = [
                _normalized_member_path(member.filename).name for member in video_members
            ]
            member_counts = Counter(member.filename for member in members)
            video_name_counts = Counter(video_names)
            unsafe_members = sorted(
                member.filename for member in members if _is_unsafe_member(member.filename)
            )
            encrypted_members = sorted(
                member.filename for member in members if member.flag_bits & 0x1
            )
            crc_failure, crc_error = (
                verify_zip_crc(archive) if verify_crc else (None, None)
            )

            selected_members = _select_probe_members(video_members, decode_sample_count)
            if selected_members:
                temporary_parent = str(scratch_root) if scratch_root is not None else None
                with tempfile.TemporaryDirectory(
                    prefix="mmprism_csl_news_probe_", dir=temporary_parent
                ) as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    probes = [
                        _probe_video_member(archive, member, temporary_root, index)
                        for index, member in enumerate(selected_members, start=1)
                    ]
    except (OSError, zipfile.BadZipFile, RuntimeError, zlib.error) as error:
        archive_read_error = f"{type(error).__name__}: {error}"

    archive_video_names = set(video_names)
    missing_labels = sorted(archive_video_names - label_index.video_names)
    empty_text_labels = sorted(archive_video_names & label_index.empty_text_video_names)
    program_counts = Counter(csl_news_source_program(video_name) for video_name in video_names)
    duplicate_members = sorted(name for name, count in member_counts.items() if count > 1)
    duplicate_video_names = sorted(name for name, count in video_name_counts.items() if count > 1)
    failed_probes = [probe["member"] for probe in probes if not probe["passed"]]

    failures: list[str] = []
    if archive_read_error is not None:
        failures.append(f"Unable to read ZIP archive: {archive_read_error}")
    elif not video_members:
        failures.append("archive contains no MP4 videos")
    if crc_failure is not None:
        failures.append(f"ZIP integrity failure: {crc_failure}: {crc_error}")
    if unsafe_members:
        failures.append(f"archive contains {len(unsafe_members)} unsafe member paths")
    if encrypted_members:
        failures.append(f"archive contains {len(encrypted_members)} encrypted members")
    if duplicate_members or duplicate_video_names:
        failures.append("archive contains duplicate member or video names")
    if label_index.invalid_record_count:
        failures.append(f"labels contain {label_index.invalid_record_count} invalid records")
    if label_index.duplicate_video_count:
        failures.append(f"labels contain {label_index.duplicate_video_count} duplicate video keys")
    if missing_labels:
        failures.append(f"{len(missing_labels)} archive videos have no label")
    if empty_text_labels:
        failures.append(f"{len(empty_text_labels)} archive videos have an empty label")
    if failed_probes:
        failures.append(f"{len(failed_probes)} sampled videos failed to decode")

    return {
        "schema_version": "mmprism.csl_news_source_audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_id": source_id.strip(),
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "options": {
            "verify_crc": verify_crc,
            "decode_sample_count": decode_sample_count,
        },
        "archive": {
            "path": str(archive_file),
            "size_bytes": archive_file.stat().st_size,
            "sha256": _sha256(archive_file),
            "member_count": len(members),
            "video_count": len(video_members),
            "compressed_member_bytes": sum(member.compress_size for member in members),
            "uncompressed_member_bytes": sum(member.file_size for member in members),
            "program_counts": dict(sorted(program_counts.items())),
            "unsafe_members": unsafe_members,
            "encrypted_members": encrypted_members,
            "duplicate_members": duplicate_members,
            "duplicate_video_names": duplicate_video_names,
            "read_error": archive_read_error,
            "crc_checked": verify_crc,
            "crc_failure": crc_failure,
            "crc_error": crc_error,
        },
        "labels": {
            "path": str(labels_file),
            "size_bytes": context.labels_size_bytes,
            "sha256": context.labels_sha256,
            "record_count": label_index.record_count,
            "unique_video_count": len(label_index.video_names),
            "invalid_record_count": label_index.invalid_record_count,
            "duplicate_video_count": label_index.duplicate_video_count,
        },
        "coverage": {
            "matched_video_count": len(archive_video_names & label_index.video_names),
            "missing_label_count": len(missing_labels),
            "missing_label_examples": missing_labels[:20],
            "empty_text_count": len(empty_text_labels),
            "empty_text_examples": empty_text_labels[:20],
        },
        "video_probes": probes,
    }


def write_csl_news_audit(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write an audit report atomically."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
