from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLIT_ASSIGNMENT_SCHEMA = "mmprism.split_assignment.v1"
SPLIT_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class SplitContractError(ValueError):
    """Raised when split assignments violate the canonical contract."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SplitContractError(f"{location} must be a mapping")
    return value


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SplitContractError(f"{location}.{key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class SplitAssignment:
    sample_id: str
    group_id: str
    split: str

    @classmethod
    def from_mapping(cls, value: object, location: str) -> SplitAssignment:
        payload = _mapping(value, location)
        unknown = sorted(
            set(payload) - {"schema_version", "sample_id", "group_id", "split"}
        )
        if unknown:
            raise SplitContractError(
                f"Unknown keys in {location}: {', '.join(unknown)}"
            )
        if payload.get("schema_version") != SPLIT_ASSIGNMENT_SCHEMA:
            raise SplitContractError(
                f"Unsupported split schema at {location}: "
                f"{payload.get('schema_version')}"
            )
        group_id = _text(payload, "group_id", location)
        if not SHA256_PATTERN.fullmatch(group_id):
            raise SplitContractError(
                f"{location}.group_id must be a lowercase SHA-256 digest"
            )
        split = _text(payload, "split", location)
        if not SPLIT_NAME_PATTERN.fullmatch(split):
            raise SplitContractError(f"{location}.split has an invalid name")
        return cls(
            sample_id=_text(payload, "sample_id", location),
            group_id=group_id,
            split=split,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": SPLIT_ASSIGNMENT_SCHEMA,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "split": self.split,
        }


@dataclass(frozen=True)
class SplitValidationSummary:
    path: Path
    assignment_count: int
    group_count: int
    splits: tuple[str, ...]
    sample_counts: dict[str, int]
    group_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "assignment_count": self.assignment_count,
            "group_count": self.group_count,
            "splits": list(self.splits),
            "sample_counts": self.sample_counts,
            "group_counts": self.group_counts,
        }


def _parse_split_assignments(path: Path) -> list[SplitAssignment]:
    assignments: list[SplitAssignment] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise SplitContractError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            assignments.append(
                SplitAssignment.from_mapping(payload, f"line {line_number}")
            )
    if not assignments:
        raise SplitContractError(f"Split assignment file contains no records: {path}")
    return assignments


def validate_split_assignments(
    path: str | Path,
    *,
    expected_sample_ids: set[str] | None = None,
    allowed_splits: set[str] | None = None,
) -> SplitValidationSummary:
    """Validate coverage, uniqueness, and group disjointness for split assignments."""

    assignment_path = Path(path).expanduser().resolve()
    if not assignment_path.is_file():
        raise SplitContractError(
            f"Split assignment file does not exist: {assignment_path}"
        )
    assignments = _parse_split_assignments(assignment_path)
    sample_ids: set[str] = set()
    group_to_split: dict[str, str] = {}
    sample_counts: Counter[str] = Counter()
    groups_by_split: defaultdict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        if assignment.sample_id in sample_ids:
            raise SplitContractError(
                f"Duplicate split sample_id: {assignment.sample_id}"
            )
        sample_ids.add(assignment.sample_id)
        if allowed_splits is not None and assignment.split not in allowed_splits:
            raise SplitContractError(
                f"Unknown split {assignment.split!r} for {assignment.sample_id}"
            )
        previous = group_to_split.setdefault(assignment.group_id, assignment.split)
        if previous != assignment.split:
            raise SplitContractError(
                f"Group leakage: {assignment.group_id} occurs in {previous} and "
                f"{assignment.split}"
            )
        sample_counts[assignment.split] += 1
        groups_by_split[assignment.split].add(assignment.group_id)

    if expected_sample_ids is not None and sample_ids != expected_sample_ids:
        missing = sorted(expected_sample_ids - sample_ids)
        extra = sorted(sample_ids - expected_sample_ids)
        raise SplitContractError(
            "Split sample coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"missing_examples={missing[:5]}, extra_examples={extra[:5]}"
        )
    splits = tuple(sorted(sample_counts))
    return SplitValidationSummary(
        path=assignment_path,
        assignment_count=len(assignments),
        group_count=len(group_to_split),
        splits=splits,
        sample_counts={name: sample_counts[name] for name in splits},
        group_counts={name: len(groups_by_split[name]) for name in splits},
    )


class SplitIndex:
    """Dependency-light sample-to-split index backed by canonical assignments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        validate_split_assignments(self.path)
        assignments = _parse_split_assignments(self.path)
        self._assignments = {
            assignment.sample_id: assignment for assignment in assignments
        }
        by_split: defaultdict[str, list[str]] = defaultdict(list)
        for assignment in assignments:
            by_split[assignment.split].append(assignment.sample_id)
        self._by_split = {
            name: tuple(sorted(sample_ids)) for name, sample_ids in by_split.items()
        }

    def __len__(self) -> int:
        return len(self._assignments)

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_split))

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._assignments

    def __getitem__(self, sample_id: str) -> SplitAssignment:
        return self._assignments[sample_id]

    def sample_ids(self, split: str) -> tuple[str, ...]:
        try:
            return self._by_split[split]
        except KeyError as error:
            raise KeyError(f"Unknown split: {split}") from error
