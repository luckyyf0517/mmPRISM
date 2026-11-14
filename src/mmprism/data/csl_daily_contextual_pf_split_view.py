"""Build semantic membership views reusable across canonical CSL-Daily stores."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmprism.contracts import SplitAssignment

CONTEXTUAL_PF_SPLIT_VIEW_SCHEMA = "mmprism.csl_daily_semantic_split_view.v1"

CONTEXTUAL_PF_SPLIT_VIEW_TENSOR_POLICY = "reuse_canonical_sample_payloads_without_modification"

_ACTIVE_SPLITS = frozenset({"train", "validation", "test"})


class CslDailyContextualPFSplitViewError(RuntimeError):
    """Raised when an alternate contextual P/F semantic view is inconsistent."""


@dataclass(frozen=True, slots=True)
class CslDailyContextualPFExternalSplitView:
    """Lightweight semantic membership overlay for a canonical Parquet store."""

    root: Path
    split_assignment_sha256: str
    sample_ids_by_role: Mapping[str, tuple[str, ...]]

    @property
    def sample_count(self) -> int:
        return sum(len(sample_ids) for sample_ids in self.sample_ids_by_role.values())

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            role: len(self.sample_ids_by_role[role]) for role in sorted(self.sample_ids_by_role)
        }

    def sample_ids_for_role(self, role: str) -> tuple[str, ...]:
        if role not in _ACTIVE_SPLITS:
            raise CslDailyContextualPFSplitViewError(
                f"external split-view role is unsupported: {role!r}"
            )
        try:
            return self.sample_ids_by_role[role]
        except KeyError as error:
            raise CslDailyContextualPFSplitViewError(
                f"external split view has no samples for role {role!r}"
            ) from error

    def checkpoint_identity(self) -> dict[str, object]:
        """Stable membership identity paired with the canonical build in consumers."""

        return {
            "schema_version": CONTEXTUAL_PF_SPLIT_VIEW_SCHEMA,
            "root": str(self.root),
            "split_assignment_sha256": self.split_assignment_sha256,
            "sample_count": self.sample_count,
            "split_counts": self.split_counts,
        }


@dataclass(frozen=True, slots=True)
class CslDailyContextualPFParquetSampleReference:
    """A canonical Parquet location selected through an external semantic view."""

    sample_id: str
    part_path: str
    row_index: int
    row_group_index: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl_by_id(path: Path, *, location: str) -> dict[str, Mapping[str, Any]]:
    if not path.is_file():
        raise CslDailyContextualPFSplitViewError(f"{location} is missing: {path}")
    rows: dict[str, Mapping[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    raise CslDailyContextualPFSplitViewError(
                        f"{location} line {line_number} is not an object"
                    )
                sample_id = value.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise CslDailyContextualPFSplitViewError(
                        f"{location} line {line_number} has no sample_id"
                    )
                if sample_id in rows:
                    raise CslDailyContextualPFSplitViewError(
                        f"{location} duplicates sample_id {sample_id!r}"
                    )
                rows[sample_id] = value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslDailyContextualPFSplitViewError(f"unable to read {location}: {error}") from error
    if not rows:
        raise CslDailyContextualPFSplitViewError(f"{location} is empty")
    return rows


def _assignments(path: Path, *, location: str) -> tuple[dict[str, SplitAssignment], str]:
    raw_rows = _read_jsonl_by_id(path, location=location)
    parsed: dict[str, SplitAssignment] = {}
    for sample_id, row in raw_rows.items():
        try:
            assignment = SplitAssignment.from_mapping(row, f"{location} {sample_id}")
        except ValueError as error:
            raise CslDailyContextualPFSplitViewError(str(error)) from error
        parsed[sample_id] = assignment
    return parsed, _sha256_file(path)


def _view_mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CslDailyContextualPFSplitViewError(f"{location} must be an object")
    return value


def _view_count(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CslDailyContextualPFSplitViewError(f"{location} must be a positive integer")
    return value


def _view_sha256(value: object, *, location: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CslDailyContextualPFSplitViewError(f"{location} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise CslDailyContextualPFSplitViewError(f"{location} must be a SHA-256 digest") from error
    return value


def load_csl_daily_contextual_pf_external_split_view(
    root: str | Path,
) -> CslDailyContextualPFExternalSplitView:
    """Load a membership-only semantic view without opening or copying Parquet payloads."""

    view_root = Path(root).expanduser().resolve()
    view_path = view_root / "view.json"
    assignments_path = view_root / "split_assignments.jsonl"
    if not view_root.is_dir():
        raise CslDailyContextualPFSplitViewError(
            f"external split view root is not a directory: {view_root}"
        )
    try:
        raw = json.loads(view_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslDailyContextualPFSplitViewError(
            f"unable to read external split view metadata: {error}"
        ) from error
    payload = _view_mapping(raw, location="external split view")
    if payload.get("schema_version") != CONTEXTUAL_PF_SPLIT_VIEW_SCHEMA:
        raise CslDailyContextualPFSplitViewError("external split view schema is unsupported")
    if payload.get("status") != "complete":
        raise CslDailyContextualPFSplitViewError("external split view is not complete")
    if payload.get("tensor_policy") != CONTEXTUAL_PF_SPLIT_VIEW_TENSOR_POLICY:
        raise CslDailyContextualPFSplitViewError(
            "external split view must reuse canonical sample payloads"
        )
    declared_sample_count = _view_count(payload.get("sample_count"), location="view.sample_count")
    declared_counts = _view_mapping(payload.get("split_counts"), location="view.split_counts")
    if set(declared_counts) != _ACTIVE_SPLITS:
        raise CslDailyContextualPFSplitViewError(
            "external split view must declare train, validation, and test counts"
        )
    expected_counts = {
        role: _view_count(declared_counts[role], location=f"view.split_counts.{role}")
        for role in sorted(_ACTIVE_SPLITS)
    }
    outputs = _view_mapping(payload.get("outputs"), location="view.outputs")
    declared_assignment_sha = _view_sha256(
        outputs.get("split_assignment_sha256"),
        location="view.outputs.split_assignment_sha256",
    )
    assignments, observed_assignment_sha = _assignments(
        assignments_path,
        location="external split-view assignments",
    )
    if observed_assignment_sha != declared_assignment_sha:
        raise CslDailyContextualPFSplitViewError(
            "external split-view assignment identity does not match view metadata"
        )
    sample_ids_by_role: dict[str, tuple[str, ...]] = {
        role: tuple(
            sorted(
                sample_id
                for sample_id, assignment in assignments.items()
                if assignment.split == role
            )
        )
        for role in sorted(_ACTIVE_SPLITS)
    }
    observed_counts = {role: len(sample_ids) for role, sample_ids in sample_ids_by_role.items()}
    if observed_counts != expected_counts:
        raise CslDailyContextualPFSplitViewError(
            "external split-view counts drift: "
            f"expected {expected_counts}, observed {observed_counts}"
        )
    if len(assignments) != declared_sample_count:
        raise CslDailyContextualPFSplitViewError(
            "external split-view sample count does not match assignments"
        )
    return CslDailyContextualPFExternalSplitView(
        root=view_root,
        split_assignment_sha256=observed_assignment_sha,
        sample_ids_by_role=sample_ids_by_role,
    )


def resolve_csl_daily_contextual_pf_parquet_references(
    canonical_root: str | Path,
    split_view: CslDailyContextualPFExternalSplitView,
    *,
    role: str,
) -> tuple[CslDailyContextualPFParquetSampleReference, ...]:
    """Resolve one semantic role to canonical one-row-group Parquet locations."""

    root = Path(canonical_root).expanduser().resolve()
    index_rows = _read_jsonl_by_id(
        root / "indices" / "sample_index.jsonl",
        location="canonical Parquet sample index",
    )
    references: list[CslDailyContextualPFParquetSampleReference] = []
    missing: list[str] = []
    for sample_id in split_view.sample_ids_for_role(role):
        row = index_rows.get(sample_id)
        if row is None:
            missing.append(sample_id)
            continue
        part_path = row.get("path")
        row_index = row.get("row_index")
        row_group_index = row.get("row_group_index")
        if not isinstance(part_path, str) or not part_path:
            raise CslDailyContextualPFSplitViewError(
                f"canonical index path is invalid for {sample_id!r}"
            )
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
            raise CslDailyContextualPFSplitViewError(
                f"canonical index row_index is invalid for {sample_id!r}"
            )
        if (
            isinstance(row_group_index, bool)
            or not isinstance(row_group_index, int)
            or row_group_index < 0
        ):
            raise CslDailyContextualPFSplitViewError(
                f"canonical index row_group_index is invalid for {sample_id!r}"
            )
        references.append(
            CslDailyContextualPFParquetSampleReference(
                sample_id=sample_id,
                part_path=part_path,
                row_index=row_index,
                row_group_index=row_group_index,
            )
        )
    if missing:
        raise CslDailyContextualPFSplitViewError(
            "external split view references sample IDs absent from canonical delivery: "
            f"count={len(missing)}, first={missing[:3]}"
        )
    return tuple(references)
