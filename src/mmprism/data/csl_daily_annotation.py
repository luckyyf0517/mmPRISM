"""Read-only parser for the CSL-Daily ``csl2020ct_v2.pkl`` annotation file.

The legacy training stack (forensic reference ``src/data/dataset.py``,
``CslDailyDataset._load_captions``) reads the pickle as a mapping whose
``info`` key holds a list of per-sequence items. Each item carries at least
``name`` (sequence id such as ``S000000_P0004_T00``) and ``label_char``
(character-level caption segments). Some releases also carry gloss labels
under ``label_gloss``; this parser treats gloss labels as optional and
normalizes both plain strings and ``{"gloss": ...}`` mappings.

This module never writes pickles. Serialization for downstream consumers is
JSONL via :func:`build_csl_daily_annotation_jsonl`.
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANNOTATION_JSONL_SCHEMA = "mmprism.csl_daily_annotation.v1"


class CslDailyAnnotationError(ValueError):
    """Raised when a CSL-Daily annotation pickle cannot be parsed safely."""


@dataclass(frozen=True)
class CslDailyAnnotationRecord:
    """One CSL-Daily sequence annotation."""

    name: str
    caption: str
    label_char: tuple[str, ...]
    label_gloss: tuple[str, ...]


def _require_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CslDailyAnnotationError(f"{location} must be a non-empty string")
    return value


def _parse_label_char(value: object, location: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CslDailyAnnotationError(f"{location} must be a sequence of strings")
    characters: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise CslDailyAnnotationError(
                f"{location}[{index}] must be a string, got {type(item).__name__}"
            )
        characters.append(item)
    if not characters:
        raise CslDailyAnnotationError(f"{location} must not be empty")
    return tuple(characters)


def _parse_label_gloss(value: object, location: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CslDailyAnnotationError(
            f"{location} must be a sequence of gloss labels when present"
        )
    glosses: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            gloss = item
        elif isinstance(item, Mapping):
            candidate = item.get("gloss")
            if not isinstance(candidate, str):
                raise CslDailyAnnotationError(
                    f"{location}[{index}] mapping must carry a string 'gloss' value"
                )
            gloss = candidate
        else:
            raise CslDailyAnnotationError(
                f"{location}[{index}] must be a string or mapping, "
                f"got {type(item).__name__}"
            )
        if not gloss.strip():
            raise CslDailyAnnotationError(f"{location}[{index}] must be non-empty")
        glosses.append(gloss)
    return tuple(glosses)


def _parse_record(item: object, index: int) -> CslDailyAnnotationRecord:
    location = f"info[{index}]"
    if not isinstance(item, Mapping):
        raise CslDailyAnnotationError(f"{location} must be a mapping")
    name = _require_string(item.get("name"), f"{location}.name")
    if "label_char" not in item:
        raise CslDailyAnnotationError(f"{location}.label_char is required")
    label_char = _parse_label_char(item.get("label_char"), f"{location}.label_char")
    label_gloss = _parse_label_gloss(item.get("label_gloss"), f"{location}.label_gloss")
    return CslDailyAnnotationRecord(
        name=name,
        caption="".join(label_char),
        label_char=label_char,
        label_gloss=label_gloss,
    )


def load_csl_daily_annotations(
    path: str | Path,
) -> tuple[CslDailyAnnotationRecord, ...]:
    """Parse ``csl2020ct_v2.pkl`` into typed records.

    The file is only read; pickles are never written. ``name`` and
    ``label_char`` are required per item (matching the legacy consumer);
    ``label_gloss`` is optional and normalized to a tuple of strings.
    """

    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise CslDailyAnnotationError(
            f"CSL-Daily annotation file does not exist: {annotation_path}"
        )
    try:
        with annotation_path.open("rb") as stream:
            payload: object = pickle.load(stream)
    except Exception as error:
        raise CslDailyAnnotationError(
            f"Unable to load CSL-Daily annotation pickle {annotation_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CslDailyAnnotationError("CSL-Daily annotation pickle must be a mapping")
    info = payload.get("info")
    if isinstance(info, str) or not isinstance(info, Sequence):
        raise CslDailyAnnotationError("CSL-Daily annotation 'info' must be a sequence")

    records: list[CslDailyAnnotationRecord] = []
    seen: set[str] = set()
    for index, item in enumerate(info):
        record = _parse_record(item, index)
        if record.name in seen:
            raise CslDailyAnnotationError(
                f"Duplicate CSL-Daily sequence name: {record.name}"
            )
        seen.add(record.name)
        records.append(record)
    return tuple(records)


def _record_payload(record: CslDailyAnnotationRecord) -> dict[str, Any]:
    return {
        "schema_version": ANNOTATION_JSONL_SCHEMA,
        "name": record.name,
        "caption": record.caption,
        "label_char": list(record.label_char),
        "label_gloss": list(record.label_gloss),
    }


def build_csl_daily_annotation_jsonl(
    records: Sequence[CslDailyAnnotationRecord],
) -> str:
    """Serialize annotation records to JSONL text for downstream use."""

    lines = [
        json.dumps(_record_payload(record), ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_csl_daily_annotation_jsonl(
    records: Sequence[CslDailyAnnotationRecord],
    path: str | Path,
) -> Path:
    """Atomically write annotation records as JSONL to ``path``."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(build_csl_daily_annotation_jsonl(records))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
