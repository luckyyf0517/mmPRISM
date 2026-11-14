from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mmprism.contracts import POSE_ONLY_INPUT_MODE, POSE_PLUS_RADAR_FEATURE_INPUT_MODE

SIGN_LANGUAGE_TRANSLATION_SAMPLE_PROTOCOL = "mmprism.sign_language_translation.sample_v2"

TRANSLATION_POSE_UNITS = frozenset(("m", "model_camera_pseudo_units"))


class SignLanguageTranslationDataError(RuntimeError):
    """Raised when model-ready translation data violates its contract."""


@dataclass(frozen=True, slots=True)
class SignLanguageTranslationSample:
    sample_id: str
    pose: NDArray[np.float32]
    pose_confidence: NDArray[np.float32]
    radar_feature: NDArray[np.float32] | None
    frame_mask: NDArray[np.bool_]
    caption: str
    coordinate_frame: str
    pose_units: str


@dataclass(frozen=True, slots=True)
class SignLanguageTranslationBatch:
    sample_ids: tuple[str, ...]
    pose: NDArray[np.float32]
    pose_confidence: NDArray[np.float32]
    radar_feature: NDArray[np.float32] | None
    frame_mask: NDArray[np.bool_]
    captions: tuple[str, ...]
    coordinate_frame: str
    pose_units: str


def collate_sign_language_translation_samples(
    samples: Sequence[SignLanguageTranslationSample],
    *,
    max_frames: int,
) -> SignLanguageTranslationBatch:
    if not samples:
        raise SignLanguageTranslationDataError("cannot collate an empty translation batch")
    if max_frames < 1:
        raise SignLanguageTranslationDataError("max_frames must be positive")
    pose_shapes = {sample.pose.shape[1:] for sample in samples}
    input_modes = {
        POSE_PLUS_RADAR_FEATURE_INPUT_MODE
        if sample.radar_feature is not None
        else POSE_ONLY_INPUT_MODE
        for sample in samples
    }
    feature_dims = {
        sample.radar_feature.shape[1] for sample in samples if sample.radar_feature is not None
    }
    coordinate_frames = {sample.coordinate_frame for sample in samples}
    pose_units = {sample.pose_units for sample in samples}
    if (
        len(pose_shapes) != 1
        or len(input_modes) != 1
        or len(feature_dims) > 1
        or len(coordinate_frames) != 1
        or len(pose_units) != 1
    ):
        raise SignLanguageTranslationDataError(
            "one translation batch requires aligned pose, feature, coordinate, and unit contracts"
        )
    batch_frames = max(sample.pose.shape[0] for sample in samples)
    if batch_frames > max_frames:
        raise SignLanguageTranslationDataError(
            f"translation batch contains {batch_frames} frames, model maximum is {max_frames}"
        )
    pose_shape = next(iter(pose_shapes))
    feature_dim = next(iter(feature_dims)) if feature_dims else None
    pose = np.zeros((len(samples), batch_frames, *pose_shape), dtype=np.float32)
    confidence = np.zeros(
        (len(samples), batch_frames, pose_shape[0], pose_shape[1]), dtype=np.float32
    )
    radar_feature = (
        np.zeros((len(samples), batch_frames, feature_dim), dtype=np.float32)
        if feature_dim is not None
        else None
    )
    frame_mask = np.zeros((len(samples), batch_frames), dtype=np.bool_)
    for index, sample in enumerate(samples):
        frames = sample.pose.shape[0]
        pose[index, :frames] = sample.pose
        confidence[index, :frames] = sample.pose_confidence
        if radar_feature is not None and sample.radar_feature is not None:
            radar_feature[index, :frames] = sample.radar_feature
        frame_mask[index, :frames] = sample.frame_mask
        pose[index, :frames][~sample.frame_mask] = 0
        confidence[index, :frames][~sample.frame_mask] = 0
        if radar_feature is not None:
            radar_feature[index, :frames][~sample.frame_mask] = 0
    return SignLanguageTranslationBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        pose=pose,
        pose_confidence=confidence,
        radar_feature=radar_feature,
        frame_mask=frame_mask,
        captions=tuple(sample.caption for sample in samples),
        coordinate_frame=next(iter(coordinate_frames)),
        pose_units=next(iter(pose_units)),
    )
