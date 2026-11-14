"""Strict CSL-Daily raw-MHR membership and 10-Hz window selection.

This module owns the data-side contract for online EarlyFusion simulation.
It deliberately stays independent of PyTorch and the radar simulator: callers
receive one contiguous 10-Hz motion window plus its physical successor frame,
then decide how to turn that motion into a surface and complex radar input.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

CSL_DAILY_STRICT_ROLES: Final[tuple[str, str, str]] = ("train", "validation", "test")


class CslDailyMhrWindowError(ValueError):
    """Raised when the accepted Daily raw-MHR window contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CslDailyMhrWindowError(message)


@dataclass(frozen=True)
class CslDailyMhrEpisodeSpec:
    """One source episode assigned by the official strict split."""

    sample_id: str
    role: str
    path: Path
    ordinal: int


@dataclass(frozen=True)
class CslDailyMhrMembership:
    """The complete raw-MHR support of the declared strict split."""

    by_role: Mapping[str, tuple[CslDailyMhrEpisodeSpec, ...]]
    unassigned_sample_ids: tuple[str, ...]

    def role(self, name: str) -> tuple[CslDailyMhrEpisodeSpec, ...]:
        _require(name in CSL_DAILY_STRICT_ROLES, f"unsupported strict role: {name}")
        return self.by_role[name]

    @property
    def counts(self) -> dict[str, int]:
        return {role: len(self.by_role[role]) for role in CSL_DAILY_STRICT_ROLES}


@dataclass(frozen=True)
class CslDailyMhrTrajectory:
    """One complete native-30-Hz Daily trajectory with its 10-Hz target grid."""

    spec: CslDailyMhrEpisodeSpec
    source_frame_count: int
    target_source_frame_indices: np.ndarray
    body_pose133: np.ndarray
    hand_pose108: np.ndarray
    global_rotation3: np.ndarray

    @property
    def target_frame_count(self) -> int:
        return int(self.target_source_frame_indices.size)


@dataclass(frozen=True)
class CslDailyMhrWindow:
    """A valid 10-Hz window and the one successor needed for Doppler velocity."""

    spec: CslDailyMhrEpisodeSpec
    target_start: int
    valid_frames: int
    source_frame_indices: np.ndarray
    successor_source_frame_index: int
    body_target133: np.ndarray
    hand_target108: np.ndarray
    global_target3: np.ndarray
    body_surface133: np.ndarray
    hand_surface108: np.ndarray
    global_surface3: np.ndarray


def build_csl_daily_strict_mhr_membership(
    *,
    source_root: str | Path,
    split_assignments_path: str | Path,
    expected_counts: Mapping[str, int],
    allowed_unassigned_sample_ids: Sequence[str],
) -> CslDailyMhrMembership:
    """Bind raw MHR NPZ files to the official strict Daily split.

    The one known source that is deliberately absent from the strict split must
    be declared by the caller.  This prevents a storage-level file listing from
    silently turning into a training membership.
    """

    root = Path(source_root).resolve()
    assignments_path = Path(split_assignments_path).resolve()
    _require(root.is_dir(), f"Daily raw-MHR source root is absent: {root}")
    _require(assignments_path.is_file(), f"Daily strict split is absent: {assignments_path}")
    _require(
        set(expected_counts) == set(CSL_DAILY_STRICT_ROLES),
        "Daily expected role counts must name train, validation, and test",
    )
    _require(
        all(isinstance(value, int) and value > 0 for value in expected_counts.values()),
        "Daily expected role counts must be positive integers",
    )

    assignments: dict[str, str] = {}
    try:
        lines = assignments_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CslDailyMhrWindowError(f"cannot read Daily strict split: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CslDailyMhrWindowError(
                f"Daily strict split row {line_number} is not JSON: {error}"
            ) from error
        _require(isinstance(row, dict), f"Daily strict split row {line_number} is not an object")
        sample_id = row.get("sample_id")
        role = row.get("split")
        _require(
            isinstance(sample_id, str) and sample_id,
            f"Daily strict split row {line_number} has no sample_id",
        )
        _require(
            role in CSL_DAILY_STRICT_ROLES,
            f"Daily strict split row {line_number} has an unsupported role: {role!r}",
        )
        _require(sample_id not in assignments, f"Daily strict split repeats {sample_id}")
        assignments[sample_id] = str(role)

    paths = {path.stem: path.resolve() for path in root.glob("*.npz")}
    _require(paths, "Daily raw-MHR source root has no NPZ files")
    missing_sources = sorted(set(assignments) - set(paths))
    _require(
        not missing_sources,
        "Daily strict split names missing source NPZ files: " + ", ".join(missing_sources[:8]),
    )
    unassigned = tuple(sorted(set(paths) - set(assignments)))
    _require(
        unassigned == tuple(sorted(set(allowed_unassigned_sample_ids))),
        "Daily raw-MHR unassigned sources differ from the declared strict-split exception: "
        + ", ".join(unassigned),
    )

    grouped: dict[str, list[CslDailyMhrEpisodeSpec]] = {role: [] for role in CSL_DAILY_STRICT_ROLES}
    for sample_id, role in assignments.items():
        grouped[role].append(
            CslDailyMhrEpisodeSpec(
                sample_id=sample_id,
                role=role,
                path=paths[sample_id],
                ordinal=0,
            )
        )
    result: dict[str, tuple[CslDailyMhrEpisodeSpec, ...]] = {}
    for role in CSL_DAILY_STRICT_ROLES:
        ordered = sorted(grouped[role], key=lambda value: value.sample_id)
        _require(
            len(ordered) == expected_counts[role],
            f"Daily strict {role} count differs: {len(ordered)} != {expected_counts[role]}",
        )
        result[role] = tuple(
            CslDailyMhrEpisodeSpec(
                sample_id=value.sample_id,
                role=value.role,
                path=value.path,
                ordinal=ordinal,
            )
            for ordinal, value in enumerate(ordered)
        )
    return CslDailyMhrMembership(by_role=result, unassigned_sample_ids=unassigned)


def load_csl_daily_mhr_trajectory(
    spec: CslDailyMhrEpisodeSpec,
    *,
    source_frame_stride: int,
) -> CslDailyMhrTrajectory:
    """Read one native Daily NPZ and establish its 10-Hz successor-safe grid."""

    _require(source_frame_stride > 0, "Daily source-frame stride must be positive")
    try:
        with np.load(spec.path, allow_pickle=False) as values:
            required = (
                "episode_id",
                "frame_count",
                "all_finite",
                "frame_indices",
                "body_pose_params",
                "hand_pose_params",
                "global_rot",
            )
            _require(
                all(name in values.files for name in required),
                f"Daily MHR fields are missing: {spec.path}",
            )
            episode_id = str(values["episode_id"].item())
            frame_count = int(values["frame_count"])
            all_finite = bool(values["all_finite"])
            frame_indices = np.asarray(values["frame_indices"], dtype=np.float32)
            body = np.asarray(values["body_pose_params"], dtype=np.float32)
            hand = np.asarray(values["hand_pose_params"], dtype=np.float32)
            global_rotation = np.asarray(values["global_rot"], dtype=np.float32)
    except (OSError, ValueError) as error:
        raise CslDailyMhrWindowError(
            f"cannot load Daily MHR source {spec.path}: {error}"
        ) from error
    _require(episode_id == spec.sample_id, f"Daily NPZ episode ID differs: {spec.path}")
    _require(all_finite, f"Daily NPZ is marked non-finite: {spec.sample_id}")
    _require(frame_count >= source_frame_stride + 3, f"Daily NPZ is too short: {spec.sample_id}")
    _require(
        frame_indices.shape == (frame_count,)
        and np.array_equal(frame_indices, np.arange(frame_count, dtype=np.float32)),
        f"Daily source time axis is not contiguous: {spec.sample_id}",
    )
    _require(
        body.shape == (frame_count, 133)
        and hand.shape == (frame_count, 108)
        and global_rotation.shape == (frame_count, 3),
        f"Daily MHR tensor shape differs: {spec.sample_id}",
    )
    _require(
        bool(np.isfinite(body).all())
        and bool(np.isfinite(hand).all())
        and bool(np.isfinite(global_rotation).all()),
        f"Daily MHR tensors are non-finite: {spec.sample_id}",
    )
    target_indices = np.arange(
        0, frame_count - source_frame_stride, source_frame_stride, dtype=np.int64
    )
    _require(
        target_indices.size >= 2,
        f"Daily source has fewer than two successor-safe 10-Hz frames: {spec.sample_id}",
    )
    return CslDailyMhrTrajectory(
        spec=spec,
        source_frame_count=frame_count,
        target_source_frame_indices=target_indices,
        body_pose133=body,
        hand_pose108=hand,
        global_rotation3=global_rotation,
    )


def csl_daily_mhr_window_from_start(
    trajectory: CslDailyMhrTrajectory,
    *,
    target_start: int,
    window_frames: int,
    source_frame_stride: int,
) -> CslDailyMhrWindow:
    """Select one contiguous 10-Hz window without crossing its episode boundary."""

    _require(window_frames > 0, "Daily window length must be positive")
    _require(source_frame_stride > 0, "Daily source-frame stride must be positive")
    count = trajectory.target_frame_count
    _require(0 <= target_start < count, "Daily target window start is outside its episode")
    valid = min(window_frames, count - target_start)
    source_indices = trajectory.target_source_frame_indices[target_start : target_start + valid]
    _require(source_indices.size == valid and valid >= 1, "Daily selected window is empty")
    successor = int(source_indices[-1] + source_frame_stride)
    _require(
        successor < trajectory.source_frame_count,
        "Daily selected window lacks its successor frame",
    )
    surface_indices = np.concatenate((source_indices, np.asarray((successor,), dtype=np.int64)))
    return CslDailyMhrWindow(
        spec=trajectory.spec,
        target_start=target_start,
        valid_frames=valid,
        source_frame_indices=source_indices.copy(),
        successor_source_frame_index=successor,
        body_target133=trajectory.body_pose133[source_indices].copy(),
        hand_target108=trajectory.hand_pose108[source_indices].copy(),
        global_target3=trajectory.global_rotation3[source_indices].copy(),
        body_surface133=trajectory.body_pose133[surface_indices].copy(),
        hand_surface108=trajectory.hand_pose108[surface_indices].copy(),
        global_surface3=trajectory.global_rotation3[surface_indices].copy(),
    )


def sample_csl_daily_mhr_window(
    trajectory: CslDailyMhrTrajectory,
    *,
    window_frames: int,
    random_seed: int,
    source_frame_stride: int,
) -> CslDailyMhrWindow:
    """Deterministically sample a valid contiguous window from one episode."""

    _require(isinstance(random_seed, int), "Daily window random seed must be an integer")
    choices = max(1, trajectory.target_frame_count - window_frames + 1)
    rng = np.random.default_rng(random_seed)
    return csl_daily_mhr_window_from_start(
        trajectory,
        target_start=int(rng.integers(0, choices)),
        window_frames=window_frames,
        source_frame_stride=source_frame_stride,
    )
