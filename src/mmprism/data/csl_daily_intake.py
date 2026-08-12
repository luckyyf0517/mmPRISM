"""CSL-Daily raw-data intake: incoming-batch validation and atomic promotion.

Implements the CSL-Daily slice (DATA_INTAKE section P0-D) of the upload
staging contract (DATA_INTAKE section 6). An operator stages a batch as::

    <data_root>/incoming/<YYYYMMDD_source_batch>/
      README.md
      UPLOAD_MANIFEST.csv
      SHA256SUMS
      external_sources/csl_daily/          # payload root (default)
        SOURCE_METADATA.json
        sentence/images/<sequence>/*.jpg
        sentence_label/csl2020ct_v2.pkl

``UPLOAD_MANIFEST.csv`` carries the columns required by the intake spec:
``source_id,relative_path,category,dataset,size_bytes,sha256,source_owner,
access_class,original_format,notes``. ``SOURCE_METADATA.json`` records the
P0-D provenance fields: ``dataset_name``, ``dataset_version``,
``download_date`` (ISO 8601), ``source_url`` and ``license``.

Promotion copies (never moves) the validated payload into
``<data_root>/raw/csl_daily/`` through a staging directory, re-verifies every
checksum at the destination, and writes ``INTAKE_RECORD.json`` (schema
``mmprism.csl_daily_intake_record.v1``). Promotion is no-clobber: an existing
destination aborts the run before anything is copied.

All functions are pure over explicit paths: no environment-variable reads, no
logging, no CLI parsing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

INTAKE_RECORD_SCHEMA = "mmprism.csl_daily_intake_record.v1"
INTAKE_VALIDATOR_VERSION = "csl_daily_intake.v1"

DEFAULT_PAYLOAD_ROOT = PurePosixPath("external_sources/csl_daily")
SOURCE_METADATA_FILENAME = "SOURCE_METADATA.json"
INTAKE_RECORD_FILENAME = "INTAKE_RECORD.json"

BATCH_ID_PATTERN = re.compile(r"^\d{8}_[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_PATTERN = re.compile(r"^[^/]+\.jpe?g$", re.IGNORECASE)

MANIFEST_FILENAME = "UPLOAD_MANIFEST.csv"
SHA256SUMS_FILENAME = "SHA256SUMS"
README_FILENAME = "README.md"

MANIFEST_COLUMNS = (
    "source_id",
    "relative_path",
    "category",
    "dataset",
    "size_bytes",
    "sha256",
    "source_owner",
    "access_class",
    "original_format",
    "notes",
)
METADATA_FIELDS = (
    "dataset_name",
    "dataset_version",
    "download_date",
    "source_url",
    "license",
)


class CslDailyIntakeError(RuntimeError):
    """Raised when a CSL-Daily intake step cannot continue safely."""


class CslDailyIntakeDestinationExistsError(CslDailyIntakeError):
    """Raised when promotion would clobber an existing raw dataset."""


@dataclass(frozen=True)
class CslDailyManifestEntry:
    """One row of ``UPLOAD_MANIFEST.csv``."""

    source_id: str
    relative_path: PurePosixPath
    category: str
    dataset: str
    size_bytes: int
    sha256: str
    source_owner: str
    access_class: str
    original_format: str
    notes: str


@dataclass(frozen=True)
class CslDailySourceMetadata:
    """P0-D provenance metadata for the CSL-Daily source package."""

    dataset_name: str
    dataset_version: str
    download_date: str
    source_url: str
    license: str


@dataclass(frozen=True)
class CslDailyIntakeCheck:
    """Outcome of one validation gate."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class CslDailyIntakeReport:
    """Structured result of validating one incoming batch."""

    batch_dir: Path
    batch_id: str
    payload_root: PurePosixPath
    validated_at: str
    ok: bool
    checks: tuple[CslDailyIntakeCheck, ...]
    metadata: CslDailySourceMetadata | None
    manifest_entries: tuple[CslDailyManifestEntry, ...]
    sha256sums: tuple[tuple[PurePosixPath, str], ...]

    def failed_checks(self) -> tuple[CslDailyIntakeCheck, ...]:
        return tuple(check for check in self.checks if not check.ok)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise CslDailyIntakeError(f"Unsafe relative path in batch metadata: {value!r}")
    return path


def _manifest_field(row: Mapping[str, str | None], name: str, location: str) -> str:
    value = row.get(name)
    if value is None or not value.strip():
        raise CslDailyIntakeError(f"{location}: '{name}' must be non-empty")
    return value.strip()


def _parse_manifest(path: Path) -> tuple[CslDailyManifestEntry, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise CslDailyIntakeError("UPLOAD_MANIFEST.csv has no header row")
            missing = [column for column in MANIFEST_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise CslDailyIntakeError(
                    "UPLOAD_MANIFEST.csv is missing columns: " + ", ".join(missing)
                )
            rows = list(reader)
    except UnicodeDecodeError as error:
        raise CslDailyIntakeError(f"UPLOAD_MANIFEST.csv is not valid UTF-8: {error}") from error
    if not rows:
        raise CslDailyIntakeError("UPLOAD_MANIFEST.csv has no data rows")

    entries: list[CslDailyManifestEntry] = []
    for index, row in enumerate(rows):
        location = f"UPLOAD_MANIFEST.csv row {index + 1}"
        relative_path = _safe_relative(_manifest_field(row, "relative_path", location))
        try:
            size_bytes = int(_manifest_field(row, "size_bytes", location))
        except ValueError as error:
            raise CslDailyIntakeError(
                f"{location}: 'size_bytes' must be an integer"
            ) from error
        if size_bytes < 0:
            raise CslDailyIntakeError(f"{location}: 'size_bytes' must be >= 0")
        sha256 = _manifest_field(row, "sha256", location).lower()
        if not SHA256_PATTERN.fullmatch(sha256):
            raise CslDailyIntakeError(f"{location}: 'sha256' must be 64 lowercase hex")
        entries.append(
            CslDailyManifestEntry(
                source_id=_manifest_field(row, "source_id", location),
                relative_path=relative_path,
                category=_manifest_field(row, "category", location),
                dataset=_manifest_field(row, "dataset", location),
                size_bytes=size_bytes,
                sha256=sha256,
                source_owner=_manifest_field(row, "source_owner", location),
                access_class=_manifest_field(row, "access_class", location),
                original_format=_manifest_field(row, "original_format", location),
                notes=(row.get("notes") or "").strip(),
            )
        )
    seen: set[PurePosixPath] = set()
    for entry in entries:
        if entry.relative_path in seen:
            raise CslDailyIntakeError(
                f"Duplicate relative_path in UPLOAD_MANIFEST.csv: {entry.relative_path}"
            )
        seen.add(entry.relative_path)
    return tuple(entries)


def _parse_sha256sums(path: Path) -> tuple[tuple[PurePosixPath, str], ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise CslDailyIntakeError(f"SHA256SUMS is not valid UTF-8: {error}") from error
    entries: list[tuple[PurePosixPath, str]] = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2 or not SHA256_PATTERN.fullmatch(parts[0]):
            raise CslDailyIntakeError(
                f"SHA256SUMS line {index + 1} must be '<sha256>  <relative_path>'"
            )
        relative = parts[1].strip()
        if relative.startswith("*"):
            relative = relative[1:]
        entries.append((_safe_relative(relative), parts[0]))
    if not entries:
        raise CslDailyIntakeError("SHA256SUMS has no entries")
    return tuple(entries)


def _parse_source_metadata(path: Path) -> CslDailySourceMetadata:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslDailyIntakeError(f"Invalid SOURCE_METADATA.json: {error}") from error
    if not isinstance(payload, Mapping):
        raise CslDailyIntakeError("SOURCE_METADATA.json must contain a JSON object")
    values: dict[str, str] = {}
    for field_name in METADATA_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise CslDailyIntakeError(
                f"SOURCE_METADATA.json field '{field_name}' must be a non-empty string"
            )
        values[field_name] = value.strip()
    try:
        date.fromisoformat(values["download_date"])
    except ValueError as error:
        raise CslDailyIntakeError(
            "SOURCE_METADATA.json field 'download_date' must be an ISO 8601 date"
        ) from error
    return CslDailySourceMetadata(
        dataset_name=values["dataset_name"],
        dataset_version=values["dataset_version"],
        download_date=values["download_date"],
        source_url=values["source_url"],
        license=values["license"],
    )


def _check(name: str, ok: bool, detail: str) -> CslDailyIntakeCheck:
    return CslDailyIntakeCheck(name=name, ok=ok, detail=detail)


def _skipped(name: str, reason: str) -> CslDailyIntakeCheck:
    return _check(name, False, f"skipped: {reason}")


def validate_csl_daily_batch(
    batch_dir: str | Path,
    *,
    payload_root: str | PurePosixPath = DEFAULT_PAYLOAD_ROOT,
) -> CslDailyIntakeReport:
    """Validate one staged incoming batch against the CSL-Daily intake gates.

    Returns a structured report; never raises for ordinary validation
    failures (those are recorded as failed checks) and never modifies the
    batch. The original upload package stays read-only.
    """

    batch_path = Path(batch_dir).expanduser().resolve()
    payload = _safe_relative(
        payload_root if isinstance(payload_root, str) else payload_root.as_posix()
    )
    checks: list[CslDailyIntakeCheck] = []

    if not batch_path.is_dir():
        raise CslDailyIntakeError(f"Batch directory does not exist: {batch_path}")
    batch_id = batch_path.name
    checks.append(
        _check(
            "batch_id_format",
            BATCH_ID_PATTERN.fullmatch(batch_id) is not None,
            f"batch directory name {batch_id!r} must match YYYYMMDD_source_batch",
        )
    )

    readme_path = batch_path / README_FILENAME
    checks.append(
        _check(
            "readme_present",
            readme_path.is_file() and readme_path.stat().st_size > 0,
            f"{README_FILENAME} must exist and be non-empty",
        )
    )

    manifest_entries: tuple[CslDailyManifestEntry, ...] = ()
    manifest_path = batch_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        checks.append(_check("manifest_parseable", False, f"missing {MANIFEST_FILENAME}"))
    else:
        try:
            manifest_entries = _parse_manifest(manifest_path)
            checks.append(
                _check(
                    "manifest_parseable",
                    True,
                    f"{len(manifest_entries)} entries parsed",
                )
            )
        except CslDailyIntakeError as error:
            checks.append(_check("manifest_parseable", False, str(error)))

    sha256sums: tuple[tuple[PurePosixPath, str], ...] = ()
    sha256sums_path = batch_path / SHA256SUMS_FILENAME
    if not sha256sums_path.is_file():
        checks.append(_check("sha256sums_parseable", False, f"missing {SHA256SUMS_FILENAME}"))
    else:
        try:
            sha256sums = _parse_sha256sums(sha256sums_path)
            checks.append(
                _check("sha256sums_parseable", True, f"{len(sha256sums)} entries parsed")
            )
        except CslDailyIntakeError as error:
            checks.append(_check("sha256sums_parseable", False, str(error)))

    manifest_ok = manifest_entries != ()
    if not manifest_ok:
        checks.append(_skipped("manifest_files_match", "manifest not parseable"))
    else:
        failures: list[str] = []
        for entry in manifest_entries:
            file_path = batch_path / Path(entry.relative_path.as_posix())
            if not file_path.is_file():
                failures.append(f"missing file: {entry.relative_path}")
                continue
            if file_path.stat().st_size != entry.size_bytes:
                failures.append(f"size mismatch: {entry.relative_path}")
                continue
            if _sha256(file_path) != entry.sha256:
                failures.append(f"sha256 mismatch: {entry.relative_path}")
        checks.append(
            _check(
                "manifest_files_match",
                not failures,
                "; ".join(failures) if failures else "all manifest files verified",
            )
        )

    if not sha256sums:
        checks.append(_skipped("sha256sums_match", "SHA256SUMS not parseable"))
    else:
        failures = []
        for relative_path, digest in sha256sums:
            file_path = batch_path / Path(relative_path.as_posix())
            if not file_path.is_file():
                failures.append(f"missing file: {relative_path}")
            elif _sha256(file_path) != digest:
                failures.append(f"sha256 mismatch: {relative_path}")
        checks.append(
            _check(
                "sha256sums_match",
                not failures,
                "; ".join(failures) if failures else "all SHA256SUMS entries verified",
            )
        )

    if not (manifest_ok and sha256sums):
        checks.append(
            _skipped("manifest_sha256sums_consistent", "manifest or SHA256SUMS not parseable")
        )
    else:
        sums_by_path = dict(sha256sums)
        failures = [
            f"{entry.relative_path} not covered by SHA256SUMS with matching digest"
            for entry in manifest_entries
            if sums_by_path.get(entry.relative_path) != entry.sha256
        ]
        checks.append(
            _check(
                "manifest_sha256sums_consistent",
                not failures,
                "; ".join(failures) if failures else "manifest and SHA256SUMS agree",
            )
        )

    payload_dir = batch_path / Path(payload.as_posix())
    metadata: CslDailySourceMetadata | None = None
    metadata_path = payload_dir / SOURCE_METADATA_FILENAME
    if not metadata_path.is_file():
        checks.append(
            _check(
                "source_metadata",
                False,
                f"missing {payload}/{SOURCE_METADATA_FILENAME}",
            )
        )
    else:
        try:
            metadata = _parse_source_metadata(metadata_path)
            checks.append(_check("source_metadata", True, "provenance metadata parsed"))
        except CslDailyIntakeError as error:
            checks.append(_check("source_metadata", False, str(error)))

    annotation_path = payload_dir / "sentence_label" / "csl2020ct_v2.pkl"
    annotation_detail = "annotation readable"
    annotation_ok = True
    if not annotation_path.is_file():
        annotation_ok = False
        annotation_detail = f"missing {payload}/sentence_label/csl2020ct_v2.pkl"
    else:
        try:
            with annotation_path.open("rb") as stream:
                annotation_payload: object = pickle.load(stream)
        except Exception as error:
            annotation_ok = False
            annotation_detail = f"annotation not readable: {error}"
        else:
            info = (
                annotation_payload.get("info")
                if isinstance(annotation_payload, Mapping)
                else None
            )
            if not isinstance(info, list) or not info:
                annotation_ok = False
                annotation_detail = "annotation must be a mapping with a non-empty 'info' list"
    checks.append(_check("annotation_present", annotation_ok, annotation_detail))

    if not manifest_ok:
        checks.append(_skipped("image_layout", "manifest not parseable"))
    else:
        prefix = f"{payload.as_posix()}/sentence/images/"
        sequences: dict[str, int] = {}
        for entry in manifest_entries:
            posix = entry.relative_path.as_posix()
            if posix.startswith(prefix):
                remainder = posix[len(prefix):]
                parts = remainder.split("/")
                if len(parts) == 2 and IMAGE_PATTERN.fullmatch(parts[1]):
                    sequences.setdefault(parts[0], 0)
        if not sequences:
            checks.append(
                _check(
                    "image_layout",
                    False,
                    "manifest lists no sentence/images/<sequence>/*.jpg files",
                )
            )
        else:
            failures = []
            for sequence in sorted(sequences):
                sequence_dir = payload_dir / "sentence" / "images" / sequence
                if not sequence_dir.is_dir():
                    failures.append(f"missing sequence directory: {sequence}")
                    continue
                frames = [
                    name
                    for name in os.listdir(sequence_dir)
                    if IMAGE_PATTERN.fullmatch(name)
                    and (sequence_dir / name).is_file()
                ]
                if not frames:
                    failures.append(f"sequence has no images: {sequence}")
            checks.append(
                _check(
                    "image_layout",
                    not failures,
                    "; ".join(failures)
                    if failures
                    else f"{len(sequences)} sequence(s) with non-empty image dirs",
                )
            )

    return CslDailyIntakeReport(
        batch_dir=batch_path,
        batch_id=batch_id,
        payload_root=payload,
        validated_at=_utc_now(),
        ok=all(check.ok for check in checks),
        checks=tuple(checks),
        metadata=metadata,
        manifest_entries=manifest_entries,
        sha256sums=sha256sums,
    )


def _inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            relative = file_path.relative_to(root).as_posix()
            files.append(
                {
                    "relative_path": relative,
                    "size_bytes": file_path.stat().st_size,
                    "sha256": _sha256(file_path),
                }
            )
    return files


def promote_csl_daily_batch(
    batch_dir: str | Path,
    data_root: str | Path,
    *,
    payload_root: str | PurePosixPath = DEFAULT_PAYLOAD_ROOT,
    report: CslDailyIntakeReport | None = None,
) -> dict[str, Any]:
    """Promote a validated batch payload to ``<data_root>/raw/csl_daily/``.

    The payload is copied (never moved) through a staging directory, every
    destination checksum is re-verified against the upload manifest, and an
    ``INTAKE_RECORD.json`` receipt is written. Promotion is no-clobber: an
    existing destination raises :class:`CslDailyIntakeDestinationExistsError`
    before anything is copied. The incoming batch stays untouched.
    """

    report = report if report is not None else validate_csl_daily_batch(
        batch_dir, payload_root=payload_root
    )
    if not report.ok:
        failure_summary = "; ".join(
            f"{check.name}: {check.detail}" for check in report.failed_checks()
        )
        raise CslDailyIntakeError(
            f"batch {report.batch_id} failed validation, promotion refused: {failure_summary}"
        )
    if report.metadata is None:
        raise CslDailyIntakeError("validated report is missing source metadata")

    root = Path(data_root).expanduser().resolve()
    destination = root / "raw" / "csl_daily"
    if destination.exists():
        raise CslDailyIntakeDestinationExistsError(
            f"raw CSL-Daily destination already exists, refusing to clobber: {destination}"
        )
    payload_dir = report.batch_dir / Path(report.payload_root.as_posix())

    raw_root = root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    staging = raw_root / f".csl_daily.staging-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        shutil.copytree(payload_dir, staging, copy_function=shutil.copy2)

        # Re-verify every manifest-listed payload file at the destination.
        prefix = f"{report.payload_root.as_posix()}/"
        manifest_payload = {
            PurePosixPath(entry.relative_path.as_posix()[len(prefix):]): entry.sha256
            for entry in report.manifest_entries
            if entry.relative_path.as_posix().startswith(prefix)
        }
        failures = []
        for relative, digest in sorted(manifest_payload.items()):
            staged_file = staging / Path(relative.as_posix())
            if not staged_file.is_file() or _sha256(staged_file) != digest:
                failures.append(str(relative))
        if failures:
            raise CslDailyIntakeError(
                "destination checksum re-verification failed for: " + ", ".join(failures)
            )

        files = _inventory(staging)
        if destination.exists():
            raise CslDailyIntakeDestinationExistsError(
                f"raw CSL-Daily destination appeared during promotion: {destination}"
            )
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    record: dict[str, Any] = {
        "schema_version": INTAKE_RECORD_SCHEMA,
        "validator": {
            "module": "mmprism.data.csl_daily_intake",
            "version": INTAKE_VALIDATOR_VERSION,
        },
        "batch": {
            "batch_id": report.batch_id,
            "batch_dir": str(report.batch_dir),
            "payload_root": report.payload_root.as_posix(),
        },
        "source_metadata": {
            "dataset_name": report.metadata.dataset_name,
            "dataset_version": report.metadata.dataset_version,
            "download_date": report.metadata.download_date,
            "source_url": report.metadata.source_url,
            "license": report.metadata.license,
        },
        "files": files,
        "timestamps": {
            "validated_at": report.validated_at,
            "promoted_at": _utc_now(),
        },
        "verification": {
            "manifest_entry_count": len(report.manifest_entries),
            "sha256sums_entry_count": len(report.sha256sums),
            "destination_file_count": len(files),
            "destination_checksums_verified": True,
        },
        "destination": str(destination),
    }
    _write_json_atomic(record, destination / INTAKE_RECORD_FILENAME)
    return record
