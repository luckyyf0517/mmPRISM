"""GT paired-control delivery for the frozen CE Daily semantic route.

The source is deliberately distinct from the frozen reconstruction-prediction
delivery.  It carries only the standard-body annotation target recovered on
the accepted sample/range/time grid, then applies the fixed Direct PredPose
affine profile after its own Yoke geometry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from mmprism.data.ce_frozen_daily_predpose import (
    CE_FROZEN_DAILY_PREDPOSE_NORMALIZATION_SCHEMA,
    CE_FROZEN_DAILY_PREDPOSE_POSE_UNITS,
    ce_frozen_daily_predpose_coordinate_frame,
    ce_frozen_daily_predpose_normalization_protocol,
    normalize_ce_daily_pose,
)
from mmprism.data.sam3d_full24_yoke import POSE_SHAPE, yoke_normalize
from mmprism.data.sign_language_translation import SignLanguageTranslationSample

CE_FROZEN_DAILY_GTPOSE_SCHEMA: Final[str] = "mmprism.ce_frozen_daily_gtpose.v1"

CE_FROZEN_DAILY_GTPOSE_POSE_UNITS: Final[str] = CE_FROZEN_DAILY_PREDPOSE_POSE_UNITS

CE_FROZEN_DAILY_GTPOSE_COORDINATE_FRAME: Final[str] = (
    "ce_frozen_daily_gtpose_yoke_per_hand_shoulder_root_fixed_predpose_direct_affine_v1"
)

CE_FROZEN_DAILY_GTPOSE_NORMALIZATION_REFERENCE_KIND: Final[str] = (
    "fixed_predpose_direct_train_only_affine"
)

CE_FROZEN_DAILY_GTPOSE_INPUT_KIND: Final[str] = "standard_body_annotation_target_full24"

CE_FROZEN_DAILY_GTPOSE_SPLITS: Final[tuple[str, str, str]] = (
    "train",
    "validation",
    "test",
)

_POSE_FLAT: Final[int] = int(np.prod(POSE_SHAPE))

CE_DAILY_GTPOSE_STYLE_JOINT_AUGMENTATION_PROFILE_ID: Final[str] = "ce_daily_gtpose_style_joint_v1"


class CeFrozenDailyGtPoseError(RuntimeError):
    """Raised when the paired GT pose consumer contract is invalid."""


@dataclass(frozen=True, slots=True)
class CeDailyGtPoseStyleJointAugmentationConfig:
    """The fixed train-only joint style package used by the generation arm."""

    profile_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.profile_id != CE_DAILY_GTPOSE_STYLE_JOINT_AUGMENTATION_PROFILE_ID:
            raise CeFrozenDailyGtPoseError(
                f"unsupported GT pose style augmentation profile {self.profile_id!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise CeFrozenDailyGtPoseError("GT pose style augmentation seed must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {"profile_id": self.profile_id, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class CeFrozenDailyGtPoseEpisode:
    """One explicit sample/range record in the paired GT input delivery."""

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
        raise CeFrozenDailyGtPoseError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CeFrozenDailyGtPoseError(f"{label} must be one JSON object")
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
                    raise CeFrozenDailyGtPoseError(f"{label} line {line_number} must be an object")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CeFrozenDailyGtPoseError(f"cannot read {label}: {error}") from error
    if not rows:
        raise CeFrozenDailyGtPoseError(f"{label} has no records")
    return rows


def _text(row: Mapping[str, object], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CeFrozenDailyGtPoseError(f"{label}.{key} must be non-empty text")
    return value.strip()


def _integer(row: Mapping[str, object], key: str, label: str, *, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CeFrozenDailyGtPoseError(f"{label}.{key} must be an integer >= {minimum}")
    return value


def _relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CeFrozenDailyGtPoseError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative == PurePosixPath("."):
        raise CeFrozenDailyGtPoseError(f"{label} must be a safe relative path")
    path = (root / Path(relative)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise CeFrozenDailyGtPoseError(f"{label} is not a readable delivery file")
    return path


def _episode_from_row(row: Mapping[str, object], index: int) -> CeFrozenDailyGtPoseEpisode:
    label = f"episodes[{index}]"
    split = _text(row, "split", label)
    if split not in CE_FROZEN_DAILY_GTPOSE_SPLITS:
        raise CeFrozenDailyGtPoseError(f"{label}.split is unsupported: {split!r}")
    range_value = row.get("range")
    if not isinstance(range_value, Mapping):
        raise CeFrozenDailyGtPoseError(f"{label}.range must be an object")
    return CeFrozenDailyGtPoseEpisode(
        sample_id=_text(row, "sample_id", label),
        split=split,
        group_id=_text(row, "group_id", label),
        caption=_text(row, "caption", label),
        start=_integer(range_value, "start", f"{label}.range"),
        frame_count=_integer(range_value, "frame_count", f"{label}.range", minimum=1),
    )


def _normalization_reference(metadata: Mapping[str, object]) -> dict[str, object]:
    reference = metadata.get("normalization_reference")
    if not isinstance(reference, Mapping):
        raise CeFrozenDailyGtPoseError("GT delivery normalization_reference is missing")
    if reference.get("kind") != CE_FROZEN_DAILY_GTPOSE_NORMALIZATION_REFERENCE_KIND:
        raise CeFrozenDailyGtPoseError("GT delivery normalization reference kind differs")
    if reference.get("selected_predpose_head") != "direct":
        raise CeFrozenDailyGtPoseError("GT delivery requires the Direct baseline profile")
    if reference.get("profile_protocol_id") != ce_frozen_daily_predpose_normalization_protocol(
        "direct"
    ):
        raise CeFrozenDailyGtPoseError("GT delivery normalization protocol differs")
    delivery_id = reference.get("profile_delivery_id")
    if not isinstance(delivery_id, str) or not delivery_id.strip():
        raise CeFrozenDailyGtPoseError("GT delivery normalization profile delivery ID is missing")
    return dict(reference)


def load_ce_frozen_daily_gtpose_delivery(
    delivery_root: str | Path,
) -> tuple[Path, dict[str, Any], tuple[CeFrozenDailyGtPoseEpisode, ...]]:
    """Load an explicit-range GT paired-control delivery declaration."""

    root = Path(delivery_root).expanduser().resolve()
    if not root.is_dir():
        raise CeFrozenDailyGtPoseError(f"GT delivery root is not a directory: {root}")
    metadata = _read_json(root / "delivery.json", "GT delivery metadata")
    if metadata.get("schema_version") != CE_FROZEN_DAILY_GTPOSE_SCHEMA:
        raise CeFrozenDailyGtPoseError("GT delivery schema is unsupported")
    if metadata.get("status") != "completed":
        raise CeFrozenDailyGtPoseError("GT delivery is not completed")
    if metadata.get("coordinate_frame") != CE_FROZEN_DAILY_GTPOSE_COORDINATE_FRAME:
        raise CeFrozenDailyGtPoseError("GT delivery coordinate frame is unsupported")
    if metadata.get("normalized_pose_units") != CE_FROZEN_DAILY_GTPOSE_POSE_UNITS:
        raise CeFrozenDailyGtPoseError("GT delivery pose units are unsupported")
    input_source = metadata.get("input_pose_source")
    if (
        not isinstance(input_source, Mapping)
        or input_source.get("kind") != CE_FROZEN_DAILY_GTPOSE_INPUT_KIND
    ):
        raise CeFrozenDailyGtPoseError("GT delivery input pose source differs")
    source_predpose = metadata.get("source_predpose_delivery")
    if not isinstance(source_predpose, Mapping):
        raise CeFrozenDailyGtPoseError("GT delivery source PredPose declaration is missing")
    source_delivery_id = source_predpose.get("delivery_id")
    reference = _normalization_reference(metadata)
    if source_delivery_id != reference["profile_delivery_id"]:
        raise CeFrozenDailyGtPoseError(
            "GT delivery source PredPose and normalization profile identities differ"
        )
    episode_path = _relative_file(root, metadata.get("episodes_path"), "episodes_path")
    episodes = tuple(
        _episode_from_row(row, index)
        for index, row in enumerate(_read_jsonl(episode_path, "GT episode index"))
    )
    sample_ids = [episode.sample_id for episode in episodes]
    if len(set(sample_ids)) != len(sample_ids):
        raise CeFrozenDailyGtPoseError("GT episode index has duplicate sample IDs")
    return root, metadata, episodes


def _load_fixed_direct_normalization(
    metadata: Mapping[str, object], *, normalization_profile_path: str | Path | None
) -> tuple[dict[str, Any], Path]:
    if normalization_profile_path is None:
        raise CeFrozenDailyGtPoseError(
            "GT paired control requires the explicit frozen Direct normalization profile"
        )
    path = Path(normalization_profile_path).expanduser().resolve()
    if not path.is_file():
        raise CeFrozenDailyGtPoseError(f"GT normalization profile is not readable: {path}")
    profile = _read_json(path, "GT normalization profile")
    reference = _normalization_reference(metadata)
    if profile.get("schema_version") != CE_FROZEN_DAILY_PREDPOSE_NORMALIZATION_SCHEMA:
        raise CeFrozenDailyGtPoseError("GT normalization profile schema is unsupported")
    if profile.get("protocol_id") != reference["profile_protocol_id"]:
        raise CeFrozenDailyGtPoseError("GT normalization protocol differs from its declaration")
    if profile.get("delivery_id") != reference["profile_delivery_id"]:
        raise CeFrozenDailyGtPoseError("GT normalization delivery identity differs")
    if profile.get("selected_semantic_head") != "direct":
        raise CeFrozenDailyGtPoseError("GT normalization profile is not Direct")
    if profile.get("coordinate_frame") != ce_frozen_daily_predpose_coordinate_frame("direct"):
        raise CeFrozenDailyGtPoseError("GT normalization coordinate frame differs")
    if profile.get("normalized_pose_units") != CE_FROZEN_DAILY_GTPOSE_POSE_UNITS:
        raise CeFrozenDailyGtPoseError("GT normalization profile pose units differ")
    if profile.get("split") != "train":
        raise CeFrozenDailyGtPoseError("GT normalization profile must be train-only")
    mean = np.asarray(profile.get("mean"), dtype=np.float32)
    std = np.asarray(profile.get("std"), dtype=np.float32)
    if mean.shape != (_POSE_FLAT,) or std.shape != (_POSE_FLAT,):
        raise CeFrozenDailyGtPoseError("GT normalization statistics have invalid shape")
    if (
        not bool(np.isfinite(mean).all())
        or not bool(np.isfinite(std).all())
        or bool(np.any(std <= 0.0))
    ):
        raise CeFrozenDailyGtPoseError("GT normalization statistics are not finite positive")
    return profile, path


class CeFrozenDailyGtPoseDataset:
    """Explicit-range GT source with GT Yoke geometry and fixed Direct affine stats."""

    input_mode: Final[str] = "pose_only"
    joint_count: Final[int] = 24
    coordinate_dim: Final[int] = 3
    radar_feature_dim: Final[None] = None
    pose_units: Final[str] = CE_FROZEN_DAILY_GTPOSE_POSE_UNITS

    def __init__(
        self,
        delivery_root: str | Path,
        *,
        split: str,
        normalization_profile_path: str | Path | None,
    ) -> None:
        if split not in CE_FROZEN_DAILY_GTPOSE_SPLITS:
            raise CeFrozenDailyGtPoseError(f"unsupported GT split: {split!r}")
        root, metadata, episodes = load_ce_frozen_daily_gtpose_delivery(delivery_root)
        arrays = metadata.get("arrays")
        if not isinstance(arrays, Mapping):
            raise CeFrozenDailyGtPoseError("GT delivery arrays must be an object")
        required = {
            "gt_full24_m",
            "frame_mask",
            "target_source_frame_indices",
            "target_timestamps_s",
        }
        if set(arrays) != required:
            raise CeFrozenDailyGtPoseError("GT delivery arrays differ from the declared payload")
        try:
            pose = np.load(
                _relative_file(root, arrays["gt_full24_m"], "arrays.gt_full24_m"),
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
            raise CeFrozenDailyGtPoseError(f"cannot open GT delivery arrays: {error}") from error
        total_frames = int(sum(episode.frame_count for episode in episodes))
        if pose.shape != (total_frames, *POSE_SHAPE) or pose.dtype != np.float32:
            raise CeFrozenDailyGtPoseError("GT pose array has invalid shape or type")
        if frame_mask.shape != (total_frames,) or frame_mask.dtype != np.bool_:
            raise CeFrozenDailyGtPoseError("GT frame mask has invalid shape or type")
        if source_indices.shape != (total_frames,) or source_indices.dtype != np.int64:
            raise CeFrozenDailyGtPoseError("GT source-index array has invalid shape or type")
        if timestamps.shape != (total_frames,) or timestamps.dtype != np.float64:
            raise CeFrozenDailyGtPoseError("GT timestamp array has invalid shape or type")
        expected_start = 0
        for episode in episodes:
            if episode.start != expected_start:
                raise CeFrozenDailyGtPoseError("GT episode ranges are not contiguous")
            expected_start += episode.frame_count
        if expected_start != total_frames:
            raise CeFrozenDailyGtPoseError("GT episode ranges do not cover arrays")
        counts = metadata.get("counts")
        if not isinstance(counts, Mapping):
            raise CeFrozenDailyGtPoseError("GT delivery counts are missing")
        declared_by_split = counts.get("episodes_by_split")
        if not isinstance(declared_by_split, Mapping):
            raise CeFrozenDailyGtPoseError("GT delivery split counts are missing")
        observed_by_split = {
            role: sum(episode.split == role for episode in episodes)
            for role in CE_FROZEN_DAILY_GTPOSE_SPLITS
        }
        if dict(declared_by_split) != observed_by_split or counts.get("frames") != total_frames:
            raise CeFrozenDailyGtPoseError("GT delivery declared counts differ")
        self._root = root
        self._split = split
        self._metadata = metadata
        self._episodes = tuple(episode for episode in episodes if episode.split == split)
        if not self._episodes:
            raise CeFrozenDailyGtPoseError(f"GT split {split!r} has no episodes")
        self._pose = pose
        self._frame_mask = frame_mask
        self._source_indices = source_indices
        self._timestamps = timestamps
        self._profile, self._normalization_path = _load_fixed_direct_normalization(
            metadata,
            normalization_profile_path=normalization_profile_path,
        )
        self._mean = np.asarray(self._profile["mean"], dtype=np.float32)
        self._std = np.asarray(self._profile["std"], dtype=np.float32)
        self.coordinate_frame = CE_FROZEN_DAILY_GTPOSE_COORDINATE_FRAME
        self.sample_ids = tuple(episode.sample_id for episode in self._episodes)

    @property
    def sample_group_ids(self) -> dict[str, str]:
        return {episode.sample_id: episode.group_id for episode in self._episodes}

    @property
    def sample_frame_counts(self) -> tuple[int, ...]:
        """Return pre-augmentation lengths for batching without loading poses."""

        return tuple(episode.frame_count for episode in self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def close(self) -> None:
        """Match the translation-source protocol; NumPy memmaps need no close."""

        return

    def checkpoint_identity(self) -> dict[str, object]:
        counts = self._metadata["counts"]
        assert isinstance(counts, Mapping)
        input_source = self._metadata["input_pose_source"]
        reference = _normalization_reference(self._metadata)
        return {
            "backend": "ce_frozen_daily_gtpose",
            "delivery_root": str(self._root),
            "delivery_id": self._metadata.get("delivery_id"),
            "schema_version": CE_FROZEN_DAILY_GTPOSE_SCHEMA,
            "input_pose_source": dict(input_source)
            if isinstance(input_source, Mapping)
            else input_source,
            "coordinate_frame": self.coordinate_frame,
            "pose_units": self.pose_units,
            "episode_counts": dict(counts["episodes_by_split"]),
            "frame_count": counts["frames"],
            "normalization_protocol": reference["profile_protocol_id"],
            "normalization_profile": str(self._normalization_path),
            "normalization_reference": reference,
        }

    def load_sample(self, index: int) -> SignLanguageTranslationSample:
        if not 0 <= index < len(self._episodes):
            raise IndexError(index)
        episode = self._episodes[index]
        stop = episode.start + episode.frame_count
        raw_pose = np.asarray(self._pose[episode.start : stop], dtype=np.float32)
        frame_mask = np.asarray(self._frame_mask[episode.start : stop], dtype=np.bool_).copy()
        if raw_pose.shape != (episode.frame_count, *POSE_SHAPE):
            raise CeFrozenDailyGtPoseError(
                f"{episode.sample_id}: GT range shape differs from its index"
            )
        if not bool(np.any(frame_mask)):
            raise CeFrozenDailyGtPoseError(f"{episode.sample_id}: no valid GT frames")
        if not bool(np.isfinite(raw_pose).all()):
            raise CeFrozenDailyGtPoseError(f"{episode.sample_id}: GT pose is non-finite")
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


class CeDailyGtPoseStyleJointTrainingDataset:
    """Train-only joint style augmentation over the paired GT source.

    A shared epoch scalar lets persistent DataLoader workers observe the
    Lightning epoch without consuming the sampler/model random streams.
    """

    input_mode: Final[str] = CeFrozenDailyGtPoseDataset.input_mode
    joint_count: Final[int] = CeFrozenDailyGtPoseDataset.joint_count
    coordinate_dim: Final[int] = CeFrozenDailyGtPoseDataset.coordinate_dim
    radar_feature_dim: Final[None] = CeFrozenDailyGtPoseDataset.radar_feature_dim
    pose_units: Final[str] = CeFrozenDailyGtPoseDataset.pose_units

    def __init__(
        self,
        source: CeFrozenDailyGtPoseDataset,
        config: CeDailyGtPoseStyleJointAugmentationConfig,
    ) -> None:
        if source._split != "train":
            raise CeFrozenDailyGtPoseError("joint style augmentation is restricted to train")
        self.source = source
        self.config = config
        self._epoch = Value("q", 0, lock=False)

    def __len__(self) -> int:
        return len(self.source)

    @property
    def sample_group_ids(self) -> dict[str, str]:
        return self.source.sample_group_ids

    @property
    def sample_frame_counts(self) -> tuple[int, ...]:
        return self.source.sample_frame_counts

    def checkpoint_identity(self) -> dict[str, object]:
        identity = self.source.checkpoint_identity()
        identity["train_augmentation"] = self.config.to_dict()
        return identity

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise CeFrozenDailyGtPoseError("augmentation epoch must be non-negative")
        self._epoch.value = epoch

    @staticmethod
    def _stable_sample_key(sample_id: str) -> int:
        digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little")

    def load_sample(self, index: int) -> SignLanguageTranslationSample:
        from dataclasses import replace

        from mmprism.data.daily_pose_rotation import sample_rotation
        from mmprism.data.daily_style_packages import transform

        clean = self.source.load_sample(index)
        episode = self.source._episodes[index]
        epoch = int(self._epoch.value)
        key = (self.config.seed, epoch, self._stable_sample_key(episode.sample_id))
        if not sample_rotation(*key)["enabled"]:
            return clean

        stop = episode.start + episode.frame_count
        raw_pose = np.asarray(self.source._pose[episode.start : stop], dtype=np.float32)
        timestamps = np.asarray(self.source._timestamps[episode.start : stop], dtype=np.float64)
        common_pose, _, _ = yoke_normalize(raw_pose)
        pose, mapping, _ = transform(common_pose, timestamps, key, "joint")
        frame_mask = clean.frame_mask[mapping].copy()
        normalized = (
            ((pose.reshape(len(pose), -1) - self.source._mean) / self.source._std)
            .reshape(pose.shape)
            .astype(np.float32)
        )
        normalized[~frame_mask] = 0.0
        confidence = clean.pose_confidence[mapping].copy()
        confidence[~frame_mask] = 0.0
        return replace(
            clean,
            pose=normalized,
            pose_confidence=confidence,
            frame_mask=frame_mask,
        )
