"""Predicted-pose delivery and normalization for public-corpus reader training.

The delivery stores decoded pose heads and sequence metadata. The reader
normalizes these poses using shoulder-centred geometry and fixed affine
statistics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from mmprism.data.sam3d_full24_yoke import POSE_SHAPE, yoke_normalize
from mmprism.data.sign_language_translation import SignLanguageTranslationSample


def normalize_ce_daily_pose(
    raw_pose: np.ndarray, frame_mask: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply own-shoulder Yoke, per-side roots, and a separately fixed affine once."""
    yoke_pose, _midpoint, _yaw = yoke_normalize(raw_pose)
    root_centered = yoke_pose - yoke_pose[:, :, 0:1, :]
    normalized: np.ndarray = (
        ((root_centered.reshape(len(raw_pose), -1) - mean) / std)
        .reshape(root_centered.shape)
        .astype(np.float32)
    )
    normalized[~frame_mask] = 0.0
    return normalized


CE_FROZEN_DAILY_PREDPOSE_SCHEMA: Final[str] = "mmprism.ce_frozen_daily_predpose.v1"

CE_FROZEN_DAILY_PREDPOSE_NORMALIZATION_SCHEMA: Final[str] = (
    "mmprism.ce_frozen_daily_predpose_yoke_normalization.v1"
)

CE_FROZEN_DAILY_PREDPOSE_SELECTED_HEAD: Final[str] = "mhr_fk"

CE_FROZEN_DAILY_PREDPOSE_HEADS: Final[frozenset[str]] = frozenset({"mhr_fk", "direct"})

_CE_FROZEN_DAILY_PREDPOSE_COORDINATE_FRAMES: Final[dict[str, str]] = {
    "mhr_fk": "ce_frozen_daily_predpose_mhr_fk_yoke_per_hand_shoulder_root_v1",
    "direct": "ce_frozen_daily_predpose_direct_yoke_per_hand_shoulder_root_v1",
}

_CE_FROZEN_DAILY_PREDPOSE_NORMALIZATION_PROTOCOLS: Final[dict[str, str]] = {
    "mhr_fk": (
        "mmprism.ce_frozen_daily_predpose_mhr_fk_yoke_per_hand_root_global_affine.train_only_v1"
    ),
    "direct": (
        "mmprism.ce_frozen_daily_predpose_direct_yoke_per_hand_root_global_affine.train_only_v1"
    ),
}

CE_FROZEN_DAILY_PREDPOSE_COORDINATE_FRAME: Final[str] = _CE_FROZEN_DAILY_PREDPOSE_COORDINATE_FRAMES[
    CE_FROZEN_DAILY_PREDPOSE_SELECTED_HEAD
]

CE_FROZEN_DAILY_PREDPOSE_POSE_UNITS: Final[str] = "model_camera_pseudo_units"

CE_FROZEN_DAILY_PREDPOSE_SPLITS: Final[tuple[str, str, str]] = (
    "train",
    "validation",
    "test",
)

_POSE_FLAT: Final[int] = int(np.prod(POSE_SHAPE))


class CeFrozenDailyPredPoseError(RuntimeError):
    """Raised when the CE-frozen PredPose consumer contract is invalid."""


def _validate_head(selected_head: str) -> str:
    if selected_head not in CE_FROZEN_DAILY_PREDPOSE_HEADS:
        raise CeFrozenDailyPredPoseError(
            "PredPose head must be one of "
            f"{sorted(CE_FROZEN_DAILY_PREDPOSE_HEADS)}, got {selected_head!r}"
        )
    return selected_head


def ce_frozen_daily_predpose_coordinate_frame(selected_head: str) -> str:
    """Return the declared normalized Yoke frame for one decoded pose head."""

    return _CE_FROZEN_DAILY_PREDPOSE_COORDINATE_FRAMES[_validate_head(selected_head)]


def ce_frozen_daily_predpose_normalization_protocol(selected_head: str) -> str:
    """Return the train-only affine-normalization protocol for one pose head."""

    return _CE_FROZEN_DAILY_PREDPOSE_NORMALIZATION_PROTOCOLS[_validate_head(selected_head)]


@dataclass(frozen=True, slots=True)
class CeFrozenDailyPredPoseEpisode:
    """One explicit sample/range record in the compact predicted-pose source."""

    sample_id: str
    split: str
    group_id: str
    caption: str
    start: int
    frame_count: int


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CeFrozenDailyPredPoseError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CeFrozenDailyPredPoseError(f"{label} must be one JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row: object = json.loads(line)
                if not isinstance(row, dict):
                    raise CeFrozenDailyPredPoseError(
                        f"{label} line {line_number} must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CeFrozenDailyPredPoseError(f"cannot read {label}: {error}") from error
    if not rows:
        raise CeFrozenDailyPredPoseError(f"{label} has no records")
    return rows


def _text(row: Mapping[str, object], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CeFrozenDailyPredPoseError(f"{label}.{key} must be non-empty text")
    return value.strip()


def _integer(row: Mapping[str, object], key: str, label: str, *, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CeFrozenDailyPredPoseError(f"{label}.{key} must be an integer >= {minimum}")
    return value


def _relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CeFrozenDailyPredPoseError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative == PurePosixPath("."):
        raise CeFrozenDailyPredPoseError(f"{label} must be a safe relative path")
    path = (root / Path(relative)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise CeFrozenDailyPredPoseError(f"{label} is not a readable delivery file")
    return path


def _episode_from_row(row: Mapping[str, object], index: int) -> CeFrozenDailyPredPoseEpisode:
    label = f"episodes[{index}]"
    split = _text(row, "split", label)
    if split not in CE_FROZEN_DAILY_PREDPOSE_SPLITS:
        raise CeFrozenDailyPredPoseError(f"{label}.split is unsupported: {split!r}")
    range_value = row.get("range")
    if not isinstance(range_value, Mapping):
        raise CeFrozenDailyPredPoseError(f"{label}.range must be an object")
    return CeFrozenDailyPredPoseEpisode(
        sample_id=_text(row, "sample_id", label),
        split=split,
        group_id=_text(row, "group_id", label),
        caption=_text(row, "caption", label),
        start=_integer(range_value, "start", f"{label}.range"),
        frame_count=_integer(range_value, "frame_count", f"{label}.range", minimum=1),
    )


def load_ce_frozen_daily_predpose_delivery(
    delivery_root: str | Path,
) -> tuple[Path, dict[str, Any], tuple[CeFrozenDailyPredPoseEpisode, ...]]:
    """Load the compact delivery declaration and explicit episode ranges."""

    root = Path(delivery_root).expanduser().resolve()
    if not root.is_dir():
        raise CeFrozenDailyPredPoseError(f"PredPose delivery root is not a directory: {root}")
    metadata = _read_json(root / "delivery.json", "PredPose delivery metadata")
    if metadata.get("schema_version") != CE_FROZEN_DAILY_PREDPOSE_SCHEMA:
        raise CeFrozenDailyPredPoseError("PredPose delivery schema is unsupported")
    if metadata.get("status") != "completed":
        raise CeFrozenDailyPredPoseError("PredPose delivery is not completed")
    if metadata.get("selected_semantic_head") != CE_FROZEN_DAILY_PREDPOSE_SELECTED_HEAD:
        raise CeFrozenDailyPredPoseError("PredPose delivery selected semantic head is unsupported")
    if metadata.get("coordinate_frame") != CE_FROZEN_DAILY_PREDPOSE_COORDINATE_FRAME:
        raise CeFrozenDailyPredPoseError("PredPose delivery coordinate frame is unsupported")
    if metadata.get("normalized_pose_units") != CE_FROZEN_DAILY_PREDPOSE_POSE_UNITS:
        raise CeFrozenDailyPredPoseError("PredPose delivery normalized pose units are unsupported")
    episode_path = _relative_file(root, metadata.get("episodes_path"), "episodes_path")
    rows = _read_jsonl(episode_path, "PredPose episode index")
    episodes = tuple(_episode_from_row(row, index) for index, row in enumerate(rows))
    sample_ids = [episode.sample_id for episode in episodes]
    if len(set(sample_ids)) != len(sample_ids):
        raise CeFrozenDailyPredPoseError("PredPose episode index has duplicate sample IDs")
    return root, metadata, episodes


def _load_normalization(
    root: Path,
    metadata: Mapping[str, object],
    *,
    selected_head: str,
    normalization_profile_path: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    head = _validate_head(selected_head)
    if normalization_profile_path is None:
        if head != CE_FROZEN_DAILY_PREDPOSE_SELECTED_HEAD:
            raise CeFrozenDailyPredPoseError(
                "the Direct PredPose head requires its own explicit train-only normalization"
            )
        path = _relative_file(root, metadata.get("normalization_path"), "normalization_path")
    else:
        path = Path(normalization_profile_path).expanduser().resolve()
        if not path.is_file():
            raise CeFrozenDailyPredPoseError(
                f"PredPose normalization profile is not readable: {path}"
            )
    profile = _read_json(path, "PredPose normalization profile")
    if profile.get("schema_version") != CE_FROZEN_DAILY_PREDPOSE_NORMALIZATION_SCHEMA:
        raise CeFrozenDailyPredPoseError("PredPose normalization schema is unsupported")
    if profile.get("protocol_id") != ce_frozen_daily_predpose_normalization_protocol(head):
        raise CeFrozenDailyPredPoseError("PredPose normalization protocol is unsupported")
    if profile.get("selected_semantic_head") != head:
        raise CeFrozenDailyPredPoseError("PredPose normalization head differs")
    if profile.get("coordinate_frame") != ce_frozen_daily_predpose_coordinate_frame(head):
        raise CeFrozenDailyPredPoseError("PredPose normalization coordinate frame differs")
    if profile.get("normalized_pose_units") != CE_FROZEN_DAILY_PREDPOSE_POSE_UNITS:
        raise CeFrozenDailyPredPoseError("PredPose normalization units differ")
    if profile.get("split") != "train":
        raise CeFrozenDailyPredPoseError("PredPose normalization must be train-only")
    if profile.get("delivery_id") != metadata.get("delivery_id"):
        raise CeFrozenDailyPredPoseError("PredPose normalization delivery identity differs")
    mean = np.asarray(profile.get("mean"), dtype=np.float32)
    std = np.asarray(profile.get("std"), dtype=np.float32)
    if mean.shape != (_POSE_FLAT,) or std.shape != (_POSE_FLAT,):
        raise CeFrozenDailyPredPoseError("PredPose normalization statistics have invalid shape")
    if (
        not bool(np.isfinite(mean).all())
        or not bool(np.isfinite(std).all())
        or bool(np.any(std <= 0))
    ):
        raise CeFrozenDailyPredPoseError(
            "PredPose normalization statistics are not finite positive"
        )
    return profile, path


class CeFrozenDailyPredPoseDataset:
    """Explicit-range predicted-pose source that supplies normalized Yoke full24 input."""

    input_mode: Final[str] = "pose_only"
    joint_count: Final[int] = 24
    coordinate_dim: Final[int] = 3
    radar_feature_dim: Final[None] = None
    pose_units: Final[str] = CE_FROZEN_DAILY_PREDPOSE_POSE_UNITS

    def __init__(
        self,
        delivery_root: str | Path,
        *,
        split: str,
        selected_head: str = CE_FROZEN_DAILY_PREDPOSE_SELECTED_HEAD,
        normalization_profile_path: str | Path | None = None,
    ) -> None:
        if split not in CE_FROZEN_DAILY_PREDPOSE_SPLITS:
            raise CeFrozenDailyPredPoseError(f"unsupported PredPose split: {split!r}")
        head = _validate_head(selected_head)
        if (
            head == CE_FROZEN_DAILY_PREDPOSE_SELECTED_HEAD
            and normalization_profile_path is not None
        ):
            raise CeFrozenDailyPredPoseError(
                "the accepted MHR/FK delivery owns its train-only normalization profile"
            )
        root, metadata, episodes = load_ce_frozen_daily_predpose_delivery(delivery_root)
        arrays = metadata.get("arrays")
        if not isinstance(arrays, Mapping):
            raise CeFrozenDailyPredPoseError("PredPose delivery arrays must be an object")
        required = {
            "direct_full24_m",
            "mhr_fk_full24_m",
            "frame_mask",
            "target_source_frame_indices",
            "target_timestamps_s",
        }
        if set(arrays) != required:
            raise CeFrozenDailyPredPoseError(
                "PredPose delivery arrays differ from the declared payload"
            )
        try:
            direct = np.load(
                _relative_file(root, arrays["direct_full24_m"], "arrays.direct_full24_m"),
                mmap_mode="r",
                allow_pickle=False,
            )
            mhr = np.load(
                _relative_file(root, arrays["mhr_fk_full24_m"], "arrays.mhr_fk_full24_m"),
                mmap_mode="r",
                allow_pickle=False,
            )
            frame_mask = np.load(
                _relative_file(root, arrays["frame_mask"], "arrays.frame_mask"),
                mmap_mode="r",
                allow_pickle=False,
            )
            source_indices = np.load(
                _relative_file(
                    root,
                    arrays["target_source_frame_indices"],
                    "arrays.target_source_frame_indices",
                ),
                mmap_mode="r",
                allow_pickle=False,
            )
            timestamps = np.load(
                _relative_file(
                    root,
                    arrays["target_timestamps_s"],
                    "arrays.target_timestamps_s",
                ),
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise CeFrozenDailyPredPoseError(f"cannot open PredPose arrays: {error}") from error
        total_frames = int(sum(episode.frame_count for episode in episodes))
        expected_pose_shape = (total_frames, *POSE_SHAPE)
        if direct.shape != expected_pose_shape or mhr.shape != expected_pose_shape:
            raise CeFrozenDailyPredPoseError("PredPose head arrays have invalid shape")
        if frame_mask.shape != (total_frames,):
            raise CeFrozenDailyPredPoseError("PredPose frame mask has invalid shape")
        if source_indices.shape != (total_frames,) or timestamps.shape != (total_frames,):
            raise CeFrozenDailyPredPoseError("PredPose source-time arrays have invalid shape")
        if direct.dtype != np.float32 or mhr.dtype != np.float32:
            raise CeFrozenDailyPredPoseError("PredPose head arrays must be float32")
        if frame_mask.dtype != np.bool_ or source_indices.dtype != np.int64:
            raise CeFrozenDailyPredPoseError("PredPose mask/source-index array types differ")
        if timestamps.dtype != np.float64:
            raise CeFrozenDailyPredPoseError("PredPose timestamp array must be float64")
        # Export validates array finiteness once.  A normal consumer validates
        # the selected episode it opens rather than rescanning the complete
        # multi-gigabyte pose delivery on every rank.
        expected_start = 0
        for episode in episodes:
            if episode.start != expected_start:
                raise CeFrozenDailyPredPoseError("PredPose episode ranges are not contiguous")
            expected_start += episode.frame_count
        if expected_start != total_frames:
            raise CeFrozenDailyPredPoseError("PredPose episode ranges do not cover arrays")
        counts = metadata.get("counts")
        if not isinstance(counts, Mapping):
            raise CeFrozenDailyPredPoseError("PredPose delivery counts are missing")
        declared_by_split = counts.get("episodes_by_split")
        if not isinstance(declared_by_split, Mapping):
            raise CeFrozenDailyPredPoseError("PredPose delivery split counts are missing")
        observed_by_split = {
            role: sum(episode.split == role for episode in episodes)
            for role in CE_FROZEN_DAILY_PREDPOSE_SPLITS
        }
        if dict(declared_by_split) != observed_by_split:
            raise CeFrozenDailyPredPoseError("PredPose delivery episode counts differ")
        declared_frames = counts.get("frames")
        if declared_frames != total_frames:
            raise CeFrozenDailyPredPoseError("PredPose delivery frame count differs")
        self._root = root
        self._metadata = metadata
        self._episodes = tuple(episode for episode in episodes if episode.split == split)
        if not self._episodes:
            raise CeFrozenDailyPredPoseError(f"PredPose split {split!r} has no episodes")
        self._pose = direct if head == "direct" else mhr
        self._selected_head = head
        self._frame_mask = frame_mask
        self._source_indices = source_indices
        self._timestamps = timestamps
        self._profile, self._normalization_path = _load_normalization(
            root,
            metadata,
            selected_head=head,
            normalization_profile_path=normalization_profile_path,
        )
        self._mean = np.asarray(self._profile["mean"], dtype=np.float32)
        self._std = np.asarray(self._profile["std"], dtype=np.float32)
        self.coordinate_frame = ce_frozen_daily_predpose_coordinate_frame(head)
        self.sample_ids = tuple(episode.sample_id for episode in self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def close(self) -> None:
        """Match the evaluation source protocol; NumPy memmaps need no explicit close."""

        return

    def checkpoint_identity(self) -> dict[str, object]:
        counts = self._metadata["counts"]
        assert isinstance(counts, Mapping)  # Validated in __init__.
        return {
            "backend": "ce_frozen_daily_predpose",
            "delivery_root": str(self._root),
            "delivery_id": self._metadata.get("delivery_id"),
            "schema_version": CE_FROZEN_DAILY_PREDPOSE_SCHEMA,
            "selected_semantic_head": self._selected_head,
            "coordinate_frame": self.coordinate_frame,
            "pose_units": self.pose_units,
            "episode_counts": dict(counts["episodes_by_split"]),
            "frame_count": counts["frames"],
            "normalization_protocol": ce_frozen_daily_predpose_normalization_protocol(
                self._selected_head
            ),
            "normalization_profile": str(self._normalization_path),
        }

    def load_sample(self, index: int) -> SignLanguageTranslationSample:
        if not 0 <= index < len(self._episodes):
            raise IndexError(index)
        episode = self._episodes[index]
        stop = episode.start + episode.frame_count
        raw_pose = np.asarray(self._pose[episode.start : stop], dtype=np.float32)
        frame_mask = np.asarray(self._frame_mask[episode.start : stop], dtype=np.bool_).copy()
        if raw_pose.shape != (episode.frame_count, *POSE_SHAPE):
            raise CeFrozenDailyPredPoseError(
                f"{episode.sample_id}: {self._selected_head} range shape differs from its index"
            )
        if not bool(np.any(frame_mask)):
            raise CeFrozenDailyPredPoseError(f"{episode.sample_id}: no valid predicted frames")
        if not bool(np.isfinite(raw_pose).all()):
            raise CeFrozenDailyPredPoseError(
                f"{episode.sample_id}: {self._selected_head} pose is non-finite"
            )
        # This is the only geometric consumer transform: its shoulders are the
        # selected prediction's shoulders, never target geometry.
        normalized = normalize_ce_daily_pose(raw_pose, frame_mask, self._mean, self._std)
        confidence = np.ones((episode.frame_count, 2, 24), dtype=np.float32)
        confidence[~frame_mask] = 0.0
        return SignLanguageTranslationSample(
            sample_id=episode.sample_id,
            pose=normalized,
            pose_confidence=confidence,
            radar_feature=None,
            frame_mask=frame_mask,
            caption=episode.caption,
            coordinate_frame=self.coordinate_frame,
            pose_units=self.pose_units,
        )
