"""Read-only receipt for a direct-preservation CSL-Daily source upload."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

CSL_DAILY_SOURCE_RECEIPT_SCHEMA = "mmprism.csl_daily_source_receipt.v1"
CSL_DAILY_SOURCE_INVENTORY_SCHEMA = "mmprism.csl_daily_source_inventory.v1"
_RECEIPT_ID_PATTERN = re.compile(r"receipt-[0-9a-f]{24}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class CslDailySourceReceiptError(RuntimeError):
    """Raised when a direct-preservation source receipt is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class CslDailySourceInventoryEntry:
    """One source file fingerprint without any machine-local path."""

    relative_path: PurePosixPath
    size_bytes: int
    mtime_ns: int
    sha256: str | None = None

    def stat_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }

    def inventory_payload(self) -> dict[str, object]:
        if self.sha256 is None:
            raise CslDailySourceReceiptError(
                f"source inventory entry is missing a checksum: {self.relative_path}"
            )
        return {
            "schema_version": CSL_DAILY_SOURCE_INVENTORY_SCHEMA,
            **self.stat_payload(),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _SourceInventory:
    entries: tuple[CslDailySourceInventoryEntry, ...]

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def stat_digest(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes([entry.stat_payload() for entry in self.entries])
        )

    @property
    def content_digest(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes([entry.inventory_payload() for entry in self.entries])
        )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_relative(path: Path, root: Path) -> PurePosixPath:
    relative = PurePosixPath(path.relative_to(root).as_posix())
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CslDailySourceReceiptError(f"unsafe source-relative path: {relative}")
    return relative


def _require_source_directory(path: str | Path, location: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise CslDailySourceReceiptError(f"{location} must not be a symbolic link")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise CslDailySourceReceiptError(
            f"{location} does not exist or is not a directory: {resolved}"
        )
    return resolved


def _inventory_source(root: Path, *, include_checksums: bool) -> _SourceInventory:
    entries: list[CslDailySourceInventoryEntry] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        directory_names.sort()
        for name in directory_names:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise CslDailySourceReceiptError(
                    f"source tree contains a non-directory or symbolic-link directory: {child}"
                )
        for name in sorted(file_names):
            path = current / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise CslDailySourceReceiptError(
                    f"source tree contains a non-regular or symbolic-link file: {path}"
                )
            checksum = _sha256_file(path) if include_checksums else None
            final_metadata = path.lstat()
            if (
                final_metadata.st_size != metadata.st_size
                or final_metadata.st_mtime_ns != metadata.st_mtime_ns
            ):
                raise CslDailySourceReceiptError(
                    f"source file changed while inventorying: {_safe_relative(path, root)}"
                )
            entries.append(
                CslDailySourceInventoryEntry(
                    relative_path=_safe_relative(path, root),
                    size_bytes=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                    sha256=checksum,
                )
            )
    entries.sort(key=lambda entry: entry.relative_path.as_posix().encode("utf-8"))
    if not entries:
        raise CslDailySourceReceiptError("source tree contains no regular files")
    return _SourceInventory(entries=tuple(entries))


def _same_stat_inventory(left: _SourceInventory, right: _SourceInventory) -> bool:
    return (
        left.stat_digest == right.stat_digest
        and len(left.entries) == len(right.entries)
        and left.total_bytes == right.total_bytes
    )


def _legacy_split_receipt(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {"status": "not_supplied"}
    root = _require_source_directory(path, "legacy split root")
    records: dict[str, dict[str, object]] = {}
    for name in ("all", "train", "val", "test"):
        split = root / f"{name}.json"
        if split.is_symlink():
            raise CslDailySourceReceiptError(
                f"legacy split file must not be a symbolic link: {split}"
            )
        records[name] = (
            {
                "status": "present",
                "size_bytes": split.stat().st_size,
                "sha256": _sha256_file(split),
            }
            if split.is_file()
            else {"status": "missing"}
        )
    val = records["val"]
    test = records["test"]
    byte_identical = (
        val.get("status") == "present"
        and test.get("status") == "present"
        and val.get("sha256") == test.get("sha256")
    )
    return {
        "status": "receipted",
        "files": records,
        "val_test_byte_identical": byte_identical,
        "use_boundary": (
            "historical_replay_only_legacy_validation_as_test"
            if byte_identical
            else "requires_independent_split_audit"
        ),
    }


def _source_candidates(inventory: _SourceInventory) -> dict[str, list[str]]:
    paths = [entry.relative_path.as_posix() for entry in inventory.entries]
    archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
    return {
        "annotation_pickles": [
            path for path in paths if path.rsplit("/", 1)[-1] == "csl2020ct_v2.pkl"
        ],
        "metadata_files": [
            path for path in paths if path.rsplit("/", 1)[-1] == "SOURCE_METADATA.json"
        ],
        "archive_files": [path for path in paths if path.lower().endswith(archive_suffixes)],
    }


def _receipt_id(
    source_id: str,
    content_inventory_sha256: str,
    legacy_splits: Mapping[str, object],
) -> str:
    return "receipt-" + _sha256_bytes(
        _canonical_json_bytes(
            {
                "source_id": source_id,
                "source_inventory_sha256": content_inventory_sha256,
                "legacy_splits_sha256": _sha256_bytes(_canonical_json_bytes(legacy_splits)),
            }
        )
    )[:24]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_bytes(_canonical_json_bytes(payload))


def _write_sha256sums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files
    ]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def create_csl_daily_source_receipt(
    source_root: str | Path,
    receipt_root: str | Path,
    *,
    source_id: str,
    legacy_split_root: str | Path | None = None,
    stability_wait_seconds: float = 60.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Publish a no-clobber receipt after stable inventories and full hashing.

    The source tree is only read. The caller must invoke this after the source
    transfer has stopped; the two stat inventories and post-hash stat inventory
    reject ordinary in-flight transfer races.
    """

    if not source_id.strip():
        raise CslDailySourceReceiptError("source_id must be non-empty")
    if isinstance(stability_wait_seconds, bool) or stability_wait_seconds < 0:
        raise CslDailySourceReceiptError("stability_wait_seconds must be non-negative")
    source = _require_source_directory(source_root, "source root")
    first = _inventory_source(source, include_checksums=False)
    sleeper(float(stability_wait_seconds))
    second = _inventory_source(source, include_checksums=False)
    if not _same_stat_inventory(first, second):
        raise CslDailySourceReceiptError(
            "source tree changed during stability window; wait for transfer completion"
        )
    hashed = _inventory_source(source, include_checksums=True)
    if not _same_stat_inventory(second, hashed):
        raise CslDailySourceReceiptError(
            "source tree changed while checksumming; receipt not published"
        )
    post_hash = _inventory_source(source, include_checksums=False)
    if not _same_stat_inventory(hashed, post_hash):
        raise CslDailySourceReceiptError(
            "source tree changed after checksumming; receipt not published"
        )
    legacy_splits = _legacy_split_receipt(legacy_split_root)
    receipt_id = _receipt_id(source_id.strip(), hashed.content_digest, legacy_splits)
    root = Path(receipt_root).expanduser().resolve()
    if root == source or source in root.parents:
        raise CslDailySourceReceiptError(
            "receipt root must be outside the read-only source tree"
        )
    target = root / receipt_id
    if target.exists():
        raise CslDailySourceReceiptError(
            f"source receipt already exists, refusing to clobber: {target}"
        )
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{receipt_id}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        inventory_path = staging / "inventory.jsonl"
        inventory_path.write_bytes(
            b"".join(
                _canonical_json_bytes(entry.inventory_payload()) for entry in hashed.entries
            )
        )
        receipt = {
            "schema_version": CSL_DAILY_SOURCE_RECEIPT_SCHEMA,
            "status": "stable_receipted",
            "receipt_id": receipt_id,
            "created_at": _utc_now(),
            "source": {
                "source_id": source_id.strip(),
                "storage_boundary": "direct_preservation_external_root",
                "inventory_path": "inventory.jsonl",
                "file_count": len(hashed.entries),
                "total_bytes": hashed.total_bytes,
                "stat_inventory_sha256": hashed.stat_digest,
                "content_inventory_sha256": hashed.content_digest,
                "candidates": _source_candidates(hashed),
            },
            "stability": {
                "inventory_count": 3,
                "wait_seconds": float(stability_wait_seconds),
                "first_stat_inventory_sha256": first.stat_digest,
                "second_stat_inventory_sha256": second.stat_digest,
                "post_hash_stat_inventory_sha256": post_hash.stat_digest,
                "status": "unchanged_before_during_and_after_hashing",
            },
            "legacy_splits": legacy_splits,
            "validation": {
                "source_tree_write_policy": "read_only",
                "source_tree_stable": True,
                "source_tree_checksum_coverage": "all_regular_files",
            },
        }
        _write_json(staging / "receipt.json", receipt)
        _write_sha256sums(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "schema_version": CSL_DAILY_SOURCE_RECEIPT_SCHEMA,
        "status": "stable_receipted",
        "receipt_id": receipt_id,
        "receipt_root": str(target),
        "file_count": len(hashed.entries),
        "total_bytes": hashed.total_bytes,
        "content_inventory_sha256": hashed.content_digest,
    }


def validate_csl_daily_source_receipt(receipt_root: str | Path) -> dict[str, object]:
    """Validate one immutable receipt artifact without reading the source tree."""

    root = _require_source_directory(receipt_root, "source receipt root")
    payload_path = root / "receipt.json"
    try:
        payload: object = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CslDailySourceReceiptError(f"unable to read receipt metadata: {error}") from error
    if not isinstance(payload, Mapping):
        raise CslDailySourceReceiptError("receipt metadata must be an object")
    if payload.get("schema_version") != CSL_DAILY_SOURCE_RECEIPT_SCHEMA:
        raise CslDailySourceReceiptError("unsupported source receipt schema")
    if payload.get("status") != "stable_receipted":
        raise CslDailySourceReceiptError("source receipt is not stable")
    receipt_id = payload.get("receipt_id")
    if not isinstance(receipt_id, str) or not _RECEIPT_ID_PATTERN.fullmatch(receipt_id):
        raise CslDailySourceReceiptError("source receipt ID is invalid")
    source = payload.get("source")
    stability = payload.get("stability")
    if not isinstance(source, Mapping) or not isinstance(stability, Mapping):
        raise CslDailySourceReceiptError("source receipt is missing source or stability metadata")
    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise CslDailySourceReceiptError("source receipt source ID is invalid")
    legacy_splits = payload.get("legacy_splits")
    if not isinstance(legacy_splits, Mapping):
        raise CslDailySourceReceiptError("source receipt legacy split metadata is invalid")
    inventory_name = source.get("inventory_path")
    if inventory_name != "inventory.jsonl":
        raise CslDailySourceReceiptError("source receipt inventory path is invalid")
    entries: list[dict[str, str | int]] = []
    try:
        with (root / inventory_name).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                item: object = json.loads(line)
                if not isinstance(item, dict):
                    raise CslDailySourceReceiptError(
                        f"inventory line {line_number} must be an object"
                    )
                if item.get("schema_version") != CSL_DAILY_SOURCE_INVENTORY_SCHEMA:
                    raise CslDailySourceReceiptError(
                        f"inventory line {line_number} has an unsupported schema"
                    )
                relative = item.get("relative_path")
                size = item.get("size_bytes")
                mtime = item.get("mtime_ns")
                checksum = item.get("sha256")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or PurePosixPath(relative).is_absolute()
                    or ".." in PurePosixPath(relative).parts
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                    or isinstance(mtime, bool)
                    or not isinstance(mtime, int)
                    or mtime < 0
                    or not isinstance(checksum, str)
                    or not _SHA256_PATTERN.fullmatch(checksum)
                ):
                    raise CslDailySourceReceiptError(
                        f"inventory line {line_number} is malformed"
                    )
                entries.append(
                    {
                        "relative_path": relative,
                        "size_bytes": size,
                        "mtime_ns": mtime,
                        "sha256": checksum,
                    }
                )
    except (OSError, json.JSONDecodeError) as error:
        raise CslDailySourceReceiptError(f"unable to read receipt inventory: {error}") from error
    if not entries:
        raise CslDailySourceReceiptError("source receipt inventory is empty")
    ordered = sorted(entries, key=lambda entry: str(entry["relative_path"]).encode("utf-8"))
    if (
        entries != ordered
        or len({str(entry["relative_path"]) for entry in entries}) != len(entries)
    ):
        raise CslDailySourceReceiptError("source receipt inventory is not canonical")
    stat_digest = _sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "relative_path": entry["relative_path"],
                    "size_bytes": entry["size_bytes"],
                    "mtime_ns": entry["mtime_ns"],
                }
                for entry in entries
            ]
        )
    )
    content_digest = _sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "schema_version": CSL_DAILY_SOURCE_INVENTORY_SCHEMA,
                    **entry,
                }
                for entry in entries
            ]
        )
    )
    total_bytes = sum(cast(int, entry["size_bytes"]) for entry in entries)
    if source.get("file_count") != len(entries) or source.get("total_bytes") != total_bytes:
        raise CslDailySourceReceiptError("source receipt inventory count or bytes mismatch")
    if (
        source.get("stat_inventory_sha256") != stat_digest
        or source.get("content_inventory_sha256") != content_digest
    ):
        raise CslDailySourceReceiptError("source receipt inventory digest mismatch")
    if receipt_id != _receipt_id(source_id.strip(), content_digest, legacy_splits):
        raise CslDailySourceReceiptError("source receipt ID does not bind its inputs")
    if (
        stability.get("inventory_count") != 3
        or stability.get("status") != "unchanged_before_during_and_after_hashing"
        or stability.get("first_stat_inventory_sha256") != stat_digest
        or stability.get("second_stat_inventory_sha256") != stat_digest
        or stability.get("post_hash_stat_inventory_sha256") != stat_digest
    ):
        raise CslDailySourceReceiptError("source receipt stability inventory mismatch")
    sums_path = root / "SHA256SUMS"
    try:
        sums = sums_path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise CslDailySourceReceiptError(
            f"source receipt checksum list is missing: {error}"
        ) from error
    expected_sums = {
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(sums) != expected_sums or len(sums) != len(expected_sums):
        raise CslDailySourceReceiptError("source receipt checksum coverage mismatch")
    return {
        "schema_version": CSL_DAILY_SOURCE_RECEIPT_SCHEMA,
        "status": "passed",
        "receipt_id": receipt_id,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "content_inventory_sha256": content_digest,
    }
