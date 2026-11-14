"""Reconstruction membership and deterministic radar-window cycles."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

Role = Literal["train", "validation", "test"]

ROLES: tuple[Role, ...] = ("train", "validation", "test")


class MembershipError(ValueError):
    """Raised when the trainer-facing reconstruction membership is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MembershipError(message)


@dataclass(frozen=True, slots=True)
class MembershipRecord:
    """One raw-radar segment selected by the reconstruction membership."""

    corpus: Literal["old", "new"]
    source_id: str
    sample_id: str
    reference_id: str
    role: Role

    @property
    def source_segment_id(self) -> str:
        """Return the source-local segment ID accepted by CompactSegmentResolver."""

        prefix = f"{self.source_id}:"
        _require(
            self.sample_id.startswith(prefix) and len(self.sample_id) > len(prefix),
            f"sample {self.sample_id!r} does not match source {self.source_id!r}",
        )
        return self.sample_id[len(prefix) :]


def parse_membership(payload: Mapping[str, Any]) -> tuple[MembershipRecord, ...]:
    """Parse versioned membership records into the training roster.

    The schema identifier and role field follow the source membership format;
    binding validation checks the associated label stores separately.
    """

    schema = payload.get("schema_version")
    _require(
        schema == "mmprism.corpus_membership_v3_draft.20260915",
        "reconstruction membership schema differs",
    )
    raw_records = payload.get("records")
    _require(
        isinstance(raw_records, list) and raw_records,
        "reconstruction membership records are absent",
    )
    records: list[MembershipRecord] = []
    sample_ids: set[str] = set()
    for row_number, value in enumerate(raw_records):
        _require(isinstance(value, Mapping), f"V3 record {row_number} is not an object")
        corpus = value.get("corpus")
        source_id = value.get("source_id")
        sample_id = value.get("sample_id")
        reference_id = value.get("reference_id")
        role = value.get("role_v3")
        _require(corpus in {"old", "new"}, f"V3 record {row_number} corpus differs")
        _require(isinstance(source_id, str) and source_id, f"V3 record {row_number} source differs")
        _require(isinstance(sample_id, str) and sample_id, f"V3 record {row_number} sample differs")
        _require(
            isinstance(reference_id, str) and reference_id,
            f"V3 record {row_number} reference differs",
        )
        _require(role in ROLES, f"V3 record {row_number} role differs")
        _require(
            sample_id not in sample_ids,
            f"reconstruction membership has duplicate sample {sample_id!r}",
        )
        record = MembershipRecord(
            corpus=cast(Literal["old", "new"], corpus),
            source_id=source_id,
            sample_id=sample_id,
            reference_id=reference_id,
            role=cast(Role, role),
        )
        _ = record.source_segment_id
        records.append(record)
        sample_ids.add(sample_id)
    return tuple(records)


def deterministic_window_start(
    *,
    frame_count: int,
    window_frames: int,
    base_seed: int,
    sampler_cycle: int,
    sample_id: str,
) -> int:
    """Choose one continuous window using the issued ``(seed, cycle, sample)`` rule."""

    _require(frame_count > 0, "segment frame count must be positive")
    _require(window_frames > 0, "window frame count must be positive")
    _require(sampler_cycle >= 0, "sampler cycle must be non-negative")
    _require(sample_id, "sample ID must be non-empty")
    if frame_count <= window_frames:
        return 0
    seed_text = f"dmhr1-window|{base_seed}|{sampler_cycle}|{sample_id}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], byteorder="little")
    return random.Random(seed).randrange(frame_count - window_frames + 1)


def deterministic_cycle_batches(
    *,
    segment_count: int,
    world_size: int,
    per_device_batch: int,
    base_seed: int,
    sampler_cycle: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return rank-local batches for one DDP sampler cycle.

    The final global batch is deterministically padded from the same shuffled
    roster so every rank executes an equal number of synchronized updates.
    Every segment appears before any padding repeat.
    """

    _require(segment_count > 0, "cycle requires at least one segment")
    _require(world_size > 0, "world size must be positive")
    _require(per_device_batch > 0, "per-device batch must be positive")
    _require(sampler_cycle >= 0, "sampler cycle must be non-negative")
    global_batch = world_size * per_device_batch
    order = list(range(segment_count))
    seed_text = f"dmhr1-cycle|{base_seed}|{sampler_cycle}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], byteorder="little")
    random.Random(seed).shuffle(order)
    batch_count = (segment_count + global_batch - 1) // global_batch
    padded_count = batch_count * global_batch
    padding = padded_count - segment_count
    order.extend((order * ((padding + segment_count - 1) // segment_count))[:padding])
    by_rank: list[list[tuple[int, ...]]] = [[] for _ in range(world_size)]
    for start in range(0, padded_count, global_batch):
        global_rows = order[start : start + global_batch]
        for rank in range(world_size):
            local_start = rank * per_device_batch
            by_rank[rank].append(tuple(global_rows[local_start : local_start + per_device_batch]))
    return tuple(tuple(batches) for batches in by_rank)


def role_counts(records: Sequence[MembershipRecord]) -> dict[str, int]:
    """Return the declared segment count for each training or evaluation role."""

    return {role: sum(record.role == role for record in records) for role in ROLES}
