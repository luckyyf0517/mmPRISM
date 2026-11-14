from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from mmprism.config import (
    ConfigError,
    ProfileRef,
    ResolvedProfileBinding,
    resolve_profile_reference,
)
from mmprism.contracts import (
    POSE_ONLY_INPUT_MODE,
    POSE_PLUS_RADAR_FEATURE_INPUT_MODE,
    RADAR_FEATURE_ONLY_INPUT_MODE,
    WAVELLM_INPUT_MODES,
)
from mmprism.data.ce_frozen_daily_gtpose import (
    CE_DAILY_GTPOSE_STYLE_JOINT_AUGMENTATION_PROFILE_ID,
    CeDailyGtPoseStyleJointAugmentationConfig,
    CeFrozenDailyGtPoseError,
)
from mmprism.data.wavellm_mesh_fk_l2_dataset import (
    FULL24_TEMPORAL_SAMPLING_PROFILES,
    HYBRID24_POSTNORM_GAUSSIAN_NOISE_PROFILE_ID,
    MESH_POSE_TRAIN_AUGMENTATION_PROFILE_ID,
    PURE_HAND_COORDINATE_FRAMES,
    PURE_HAND_POSE_FRAMES,
    Full24TemporalSamplingProfile,
    Hybrid24PostnormGaussianNoiseConfig,
    MeshFkL2DatasetError,
    MeshPoseTrainAugmentationConfig,
    PureHandCoordinateFrame,
    PureHandPoseFrame,
    WaveLLMTrainAugmentationConfig,
)
from mmprism.models.wavellm_fusion_profiles import (
    WAVELLM_FUSION_COMPACT_CONFIDENCE_V1,
    WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1,
    WAVELLM_FUSION_VARIANTS,
)
from mmprism.models.wavellm_profiles import (
    WAVELLM_POSE_ENCODER_PROFILE_KIND,
    WAVELLM_POSE_LEGACY_STGCN_UNISIGN_HAND_TRANSPLANT_ROOTCENTER_VARIANT,
    WAVELLM_UNISIGN_HAND_TRANSPLANT_KIND,
    WaveLLMPoseEncoderProfile,
    wavellm_pose_encoder_frame,
)
from mmprism.training.translation.mt5_config import MT5ModelConfig, ReaderConfigurationError

WAVELLM_RUN_CONFIG_SCHEMA = "mmprism.wavellm_run.v2"

WAVELLM_CONTEXTUAL_FEATURE_IDENTITY = "post_temporal_pre_pose_head_v1"

WAVELLM_PURE_PREDICTED_RECON_V2_POSE_CONTRACT = "P_recon_v2_predicted_root_derived_P_semantic"

_WAVELLM_CONTEXTUAL_POSE_CONTRACTS = frozenset(
    {
        WAVELLM_PURE_PREDICTED_RECON_V2_POSE_CONTRACT,
    }
)


class WaveLLMRunError(RuntimeError):
    """Raised when a formal WaveLLM run contract is invalid or fails."""


def _resolve_pose_encoder_profile(
    value: object,
    *,
    project_root: str | Path,
) -> ResolvedProfileBinding[WaveLLMPoseEncoderProfile]:
    try:
        reference = ProfileRef.from_mapping(value, "pose_encoder_profile")
        if reference.expected_kind != WAVELLM_POSE_ENCODER_PROFILE_KIND:
            raise WaveLLMRunError(
                f"pose_encoder_profile.expected_kind must be {WAVELLM_POSE_ENCODER_PROFILE_KIND}"
            )
        return resolve_profile_reference(
            reference,
            project_root=project_root,
            domain_parser=WaveLLMPoseEncoderProfile.from_parameters,
        )
    except ConfigError as error:
        raise WaveLLMRunError(str(error)) from error


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WaveLLMRunError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise WaveLLMRunError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WaveLLMRunError(f"{location}.{key} must be non-empty text")
    return value.strip()


def _integer(payload: Mapping[str, Any], key: str, location: str, *, minimum: int = 1) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise WaveLLMRunError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(payload: Mapping[str, Any], key: str, location: str, *, minimum: float = 0.0) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WaveLLMRunError(f"{location}.{key} must be a number")
    result = float(value)
    if result < minimum:
        raise WaveLLMRunError(f"{location}.{key} must be >= {minimum}")
    return result


def _positive_pair(
    payload: Mapping[str, Any],
    key: str,
    location: str,
) -> tuple[float, float]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise WaveLLMRunError(f"{location}.{key} must be a two-item list")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise WaveLLMRunError(f"{location}.{key} must contain numbers")
    lower, upper = float(value[0]), float(value[1])
    if lower <= 0.0 or lower > upper:
        raise WaveLLMRunError(f"{location}.{key} must be ordered and positive")
    return lower, upper


def _train_augmentation(
    value: object | None,
) -> WaveLLMTrainAugmentationConfig | CeDailyGtPoseStyleJointAugmentationConfig | None:
    if value is None:
        return None
    location = "data.train_augmentation"
    payload = _mapping(value, location)
    profile_id = _text(payload, "profile_id", location)
    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise WaveLLMRunError(f"{location}.seed must be a non-negative integer")
    try:
        if profile_id == MESH_POSE_TRAIN_AUGMENTATION_PROFILE_ID:
            _reject_unknown(
                payload,
                {"profile_id", "seed", "rotation_deg", "uniform_scale", "speed"},
                location,
            )
            return MeshPoseTrainAugmentationConfig(
                profile_id=profile_id,
                seed=seed,
                rotation_deg=_number(payload, "rotation_deg", location),
                uniform_scale=_positive_pair(payload, "uniform_scale", location),
                speed=_positive_pair(payload, "speed", location),
            )
        if profile_id == HYBRID24_POSTNORM_GAUSSIAN_NOISE_PROFILE_ID:
            _reject_unknown(
                payload,
                {"profile_id", "seed", "coordinate_noise_std"},
                location,
            )
            return Hybrid24PostnormGaussianNoiseConfig(
                profile_id=profile_id,
                seed=seed,
                coordinate_noise_std=_number(payload, "coordinate_noise_std", location),
            )
        if profile_id == CE_DAILY_GTPOSE_STYLE_JOINT_AUGMENTATION_PROFILE_ID:
            _reject_unknown(payload, {"profile_id", "seed"}, location)
            return CeDailyGtPoseStyleJointAugmentationConfig(
                profile_id=profile_id,
                seed=seed,
            )
        raise WaveLLMRunError(f"unsupported data.train_augmentation profile {profile_id!r}")
    except (MeshFkL2DatasetError, CeFrozenDailyGtPoseError) as error:
        raise WaveLLMRunError(str(error)) from error


@dataclass(frozen=True, slots=True)
class WaveLLMDataConfig:
    batch_size: int
    num_workers: int
    verify_checksums: bool
    shuffle: bool
    max_frames: int
    prompt: str
    max_prompt_length: int
    max_target_length: int
    train_augmentation: (
        WaveLLMTrainAugmentationConfig | CeDailyGtPoseStyleJointAugmentationConfig | None
    ) = None
    # Pretrained token format: sequences longer than
    # max_frames are RANDOM-subsampled per load (legacy pad_sequence), and
    # every sequence is zero-padded to a FIXED max_frames with an all-true
    # frame mask — the warm-started encoder's native input distribution.
    legacy_sequence_format: bool = False
    pose_frame: PureHandPoseFrame | None = None
    pose_coordinate_frame: PureHandCoordinateFrame | None = None
    pose_normalization_profile: str | None = None
    full24_normalization_profile: str | None = None
    hybrid24_normalization_profile: str | None = None
    hybrid24v2_normalization_profile: str | None = None
    unisign_hand_bridge_normalization_profile: str | None = None
    ce_frozen_predpose_head: str | None = None
    ce_frozen_predpose_normalization_profile: str | None = None
    full24_temporal_sampling_profile: Full24TemporalSamplingProfile | None = None
    # Training-only random stride-3 phase (0/1/2), drawn per
    # (sample, epoch); evaluation splits keep the canonical phase-0 grid.
    full24_temporal_train_phase: Literal["random"] | None = None
    split_assignments_path: str | None = None

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMDataConfig:
        location = "data"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {
                "batch_size",
                "num_workers",
                "verify_checksums",
                "shuffle",
                "max_frames",
                "prompt",
                "max_prompt_length",
                "max_target_length",
                "train_augmentation",
                "legacy_sequence_format",
                "pose_frame",
                "pose_coordinate_frame",
                "pose_normalization_profile",
                "full24_normalization_profile",
                "hybrid24_normalization_profile",
                "hybrid24v2_normalization_profile",
                "unisign_hand_bridge_normalization_profile",
                "ce_frozen_predpose_head",
                "ce_frozen_predpose_normalization_profile",
                "full24_temporal_sampling_profile",
                "full24_temporal_train_phase",
                "split_assignments_path",
            },
            location,
        )
        num_workers = payload.get("num_workers")
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise WaveLLMRunError("data.num_workers must be a non-negative integer")
        verify_checksums = payload.get("verify_checksums")
        shuffle = payload.get("shuffle")
        if not isinstance(verify_checksums, bool):
            raise WaveLLMRunError("data.verify_checksums must be a boolean")
        if not isinstance(shuffle, bool):
            raise WaveLLMRunError("data.shuffle must be a boolean")
        legacy_sequence_format = payload.get("legacy_sequence_format", False)
        if not isinstance(legacy_sequence_format, bool):
            raise WaveLLMRunError("data.legacy_sequence_format must be a boolean")
        pose_frame = payload.get("pose_frame")
        if pose_frame is not None and (
            not isinstance(pose_frame, str) or pose_frame not in PURE_HAND_POSE_FRAMES
        ):
            raise WaveLLMRunError("data.pose_frame must be omitted, root_relative, or camera")
        resolved_pose_frame = (
            cast(PureHandPoseFrame, pose_frame) if pose_frame is not None else None
        )
        pose_coordinate_frame = payload.get("pose_coordinate_frame")
        if pose_coordinate_frame is not None and (
            not isinstance(pose_coordinate_frame, str)
            or pose_coordinate_frame not in PURE_HAND_COORDINATE_FRAMES
        ):
            raise WaveLLMRunError(
                "data.pose_coordinate_frame must be omitted or one of "
                f"{sorted(PURE_HAND_COORDINATE_FRAMES)}"
            )
        resolved_pose_coordinate_frame = (
            cast(PureHandCoordinateFrame, pose_coordinate_frame)
            if pose_coordinate_frame is not None
            else None
        )
        pose_normalization_profile = payload.get("pose_normalization_profile")
        if pose_normalization_profile is not None and (
            not isinstance(pose_normalization_profile, str)
            or not pose_normalization_profile.strip()
        ):
            raise WaveLLMRunError("data.pose_normalization_profile must be non-empty text when set")
        if resolved_pose_frame is None and pose_normalization_profile is not None:
            raise WaveLLMRunError("data.pose_normalization_profile requires data.pose_frame")
        if resolved_pose_frame is not None and pose_normalization_profile is None:
            raise WaveLLMRunError("data.pose_frame requires data.pose_normalization_profile")
        if resolved_pose_frame is None and resolved_pose_coordinate_frame is not None:
            raise WaveLLMRunError("data.pose_coordinate_frame requires data.pose_frame")
        if resolved_pose_frame is not None and resolved_pose_coordinate_frame is None:
            raise WaveLLMRunError("data.pose_frame requires data.pose_coordinate_frame")
        full24_normalization_profile = payload.get("full24_normalization_profile")
        if full24_normalization_profile is not None and (
            not isinstance(full24_normalization_profile, str)
            or not full24_normalization_profile.strip()
        ):
            raise WaveLLMRunError(
                "data.full24_normalization_profile must be non-empty text when set"
            )
        hybrid24_normalization_profile = payload.get("hybrid24_normalization_profile")
        if hybrid24_normalization_profile is not None and (
            not isinstance(hybrid24_normalization_profile, str)
            or not hybrid24_normalization_profile.strip()
        ):
            raise WaveLLMRunError(
                "data.hybrid24_normalization_profile must be non-empty text when set"
            )
        hybrid24v2_normalization_profile = payload.get("hybrid24v2_normalization_profile")
        if hybrid24v2_normalization_profile is not None and (
            not isinstance(hybrid24v2_normalization_profile, str)
            or not hybrid24v2_normalization_profile.strip()
        ):
            raise WaveLLMRunError(
                "data.hybrid24v2_normalization_profile must be non-empty text when set"
            )
        unisign_hand_bridge_normalization_profile = payload.get(
            "unisign_hand_bridge_normalization_profile"
        )
        if unisign_hand_bridge_normalization_profile is not None and (
            not isinstance(unisign_hand_bridge_normalization_profile, str)
            or not unisign_hand_bridge_normalization_profile.strip()
        ):
            raise WaveLLMRunError(
                "data.unisign_hand_bridge_normalization_profile must be non-empty text when set"
            )
        ce_frozen_predpose_head = payload.get("ce_frozen_predpose_head")
        if ce_frozen_predpose_head is not None and (
            not isinstance(ce_frozen_predpose_head, str)
            or ce_frozen_predpose_head not in {"mhr_fk", "direct"}
        ):
            raise WaveLLMRunError("data.ce_frozen_predpose_head must be omitted, mhr_fk, or direct")
        ce_frozen_predpose_normalization_profile = payload.get(
            "ce_frozen_predpose_normalization_profile"
        )
        if ce_frozen_predpose_normalization_profile is not None and (
            not isinstance(ce_frozen_predpose_normalization_profile, str)
            or not ce_frozen_predpose_normalization_profile.strip()
        ):
            raise WaveLLMRunError(
                "data.ce_frozen_predpose_normalization_profile must be non-empty text when set"
            )
        if ce_frozen_predpose_normalization_profile is not None and ce_frozen_predpose_head is None:
            raise WaveLLMRunError(
                "data.ce_frozen_predpose_normalization_profile requires "
                "data.ce_frozen_predpose_head"
            )
        external_profile_count = sum(
            profile is not None
            for profile in (
                full24_normalization_profile,
                hybrid24_normalization_profile,
                hybrid24v2_normalization_profile,
                unisign_hand_bridge_normalization_profile,
            )
        )
        if external_profile_count > 1:
            raise WaveLLMRunError(
                "data.full24_normalization_profile, data.hybrid24_normalization_profile, "
                "data.hybrid24v2_normalization_profile, and "
                "data.unisign_hand_bridge_normalization_profile are mutually exclusive"
            )
        full24_temporal_sampling_profile = payload.get("full24_temporal_sampling_profile")
        if full24_temporal_sampling_profile is not None and (
            not isinstance(full24_temporal_sampling_profile, str)
            or full24_temporal_sampling_profile not in FULL24_TEMPORAL_SAMPLING_PROFILES
        ):
            raise WaveLLMRunError(
                "data.full24_temporal_sampling_profile must be omitted or one of "
                f"{sorted(FULL24_TEMPORAL_SAMPLING_PROFILES)}"
            )
        resolved_full24_temporal_sampling_profile = (
            cast(Full24TemporalSamplingProfile, full24_temporal_sampling_profile)
            if full24_temporal_sampling_profile is not None
            else None
        )
        full24_temporal_train_phase = payload.get("full24_temporal_train_phase")
        if full24_temporal_train_phase is not None and (
            not isinstance(full24_temporal_train_phase, str)
            or full24_temporal_train_phase != "random"
        ):
            raise WaveLLMRunError("data.full24_temporal_train_phase must be omitted or 'random'")
        split_assignments_path = payload.get("split_assignments_path")
        if split_assignments_path is not None and (
            not isinstance(split_assignments_path, str) or not split_assignments_path.strip()
        ):
            raise WaveLLMRunError("data.split_assignments_path must be non-empty text when set")
        if full24_normalization_profile is not None and resolved_pose_frame is not None:
            raise WaveLLMRunError(
                "data.full24_normalization_profile cannot be used with data.pose_frame"
            )
        if hybrid24_normalization_profile is not None and resolved_pose_frame is not None:
            raise WaveLLMRunError(
                "data.hybrid24_normalization_profile cannot be used with data.pose_frame"
            )
        if hybrid24v2_normalization_profile is not None and resolved_pose_frame is not None:
            raise WaveLLMRunError(
                "data.hybrid24v2_normalization_profile cannot be used with data.pose_frame"
            )
        if (
            unisign_hand_bridge_normalization_profile is not None
            and resolved_pose_frame is not None
        ):
            raise WaveLLMRunError(
                "data.unisign_hand_bridge_normalization_profile cannot be used with data.pose_frame"
            )
        if (
            resolved_full24_temporal_sampling_profile is not None
            and resolved_pose_frame is not None
        ):
            raise WaveLLMRunError(
                "data.full24_temporal_sampling_profile cannot be used with data.pose_frame"
            )
        if (
            full24_temporal_train_phase == "random"
            and resolved_full24_temporal_sampling_profile is None
        ):
            raise WaveLLMRunError(
                "data.full24_temporal_train_phase=random requires "
                "data.full24_temporal_sampling_profile"
            )
        if (
            resolved_full24_temporal_sampling_profile is not None
            and full24_normalization_profile is None
            and hybrid24_normalization_profile is None
            and hybrid24v2_normalization_profile is None
            and unisign_hand_bridge_normalization_profile is None
        ):
            raise WaveLLMRunError(
                "data.full24_temporal_sampling_profile requires "
                "data.full24_normalization_profile, data.hybrid24_normalization_profile, or "
                "data.hybrid24v2_normalization_profile, or "
                "data.unisign_hand_bridge_normalization_profile"
            )
        if resolved_pose_frame is None:
            if (
                full24_normalization_profile is not None
                or hybrid24_normalization_profile is not None
                or hybrid24v2_normalization_profile is not None
                or unisign_hand_bridge_normalization_profile is not None
            ) and split_assignments_path is None:
                raise WaveLLMRunError(
                    "external 24-joint normalization profile requires data.split_assignments_path"
                )
            if (
                split_assignments_path is not None
                and full24_normalization_profile is None
                and hybrid24_normalization_profile is None
                and hybrid24v2_normalization_profile is None
                and unisign_hand_bridge_normalization_profile is None
            ):
                raise WaveLLMRunError(
                    "data.split_assignments_path requires data.pose_frame or "
                    "data.full24_normalization_profile, data.hybrid24_normalization_profile, or "
                    "data.hybrid24v2_normalization_profile, or "
                    "data.unisign_hand_bridge_normalization_profile"
                )
        train_augmentation = _train_augmentation(payload.get("train_augmentation"))
        if (
            isinstance(train_augmentation, Hybrid24PostnormGaussianNoiseConfig)
            and hybrid24_normalization_profile is None
        ):
            raise WaveLLMRunError(
                "hybrid24_postnorm_gaussian_noise_v1 requires data.hybrid24_normalization_profile"
            )
        if isinstance(train_augmentation, MeshPoseTrainAugmentationConfig) and (
            hybrid24_normalization_profile is not None
            or hybrid24v2_normalization_profile is not None
            or unisign_hand_bridge_normalization_profile is not None
        ):
            raise WaveLLMRunError(
                "mesh_pose_train_v1 cannot be used with hybrid24 or the Uni-Sign bridge"
            )
        if isinstance(train_augmentation, CeDailyGtPoseStyleJointAugmentationConfig) and (
            ce_frozen_predpose_head != "direct" or ce_frozen_predpose_normalization_profile is None
        ):
            raise WaveLLMRunError(
                "ce_daily_gtpose_style_joint_v1 requires the paired Direct CE pose view"
            )
        return cls(
            batch_size=_integer(payload, "batch_size", location),
            num_workers=num_workers,
            verify_checksums=verify_checksums,
            shuffle=shuffle,
            max_frames=_integer(payload, "max_frames", location),
            prompt=_text(payload, "prompt", location),
            max_prompt_length=_integer(payload, "max_prompt_length", location),
            max_target_length=_integer(payload, "max_target_length", location),
            train_augmentation=train_augmentation,
            legacy_sequence_format=legacy_sequence_format,
            pose_frame=resolved_pose_frame,
            pose_coordinate_frame=resolved_pose_coordinate_frame,
            pose_normalization_profile=pose_normalization_profile,
            full24_normalization_profile=full24_normalization_profile,
            hybrid24_normalization_profile=hybrid24_normalization_profile,
            hybrid24v2_normalization_profile=hybrid24v2_normalization_profile,
            unisign_hand_bridge_normalization_profile=unisign_hand_bridge_normalization_profile,
            ce_frozen_predpose_head=ce_frozen_predpose_head,
            ce_frozen_predpose_normalization_profile=ce_frozen_predpose_normalization_profile,
            full24_temporal_sampling_profile=resolved_full24_temporal_sampling_profile,
            full24_temporal_train_phase=(
                cast(Literal["random"], full24_temporal_train_phase)
                if full24_temporal_train_phase is not None
                else None
            ),
            split_assignments_path=split_assignments_path,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "verify_checksums": self.verify_checksums,
            "shuffle": self.shuffle,
            "max_frames": self.max_frames,
            "prompt": self.prompt,
            "max_prompt_length": self.max_prompt_length,
            "max_target_length": self.max_target_length,
        }
        if self.train_augmentation is not None:
            result["train_augmentation"] = self.train_augmentation.to_dict()
        result["legacy_sequence_format"] = self.legacy_sequence_format
        if self.pose_frame is not None:
            result["pose_frame"] = self.pose_frame
            result["pose_coordinate_frame"] = self.pose_coordinate_frame
            result["pose_normalization_profile"] = self.pose_normalization_profile
        if self.full24_normalization_profile is not None:
            result["full24_normalization_profile"] = self.full24_normalization_profile
        if self.hybrid24_normalization_profile is not None:
            result["hybrid24_normalization_profile"] = self.hybrid24_normalization_profile
        if self.hybrid24v2_normalization_profile is not None:
            result["hybrid24v2_normalization_profile"] = self.hybrid24v2_normalization_profile
        if self.unisign_hand_bridge_normalization_profile is not None:
            result["unisign_hand_bridge_normalization_profile"] = (
                self.unisign_hand_bridge_normalization_profile
            )
        if self.ce_frozen_predpose_head is not None:
            result["ce_frozen_predpose_head"] = self.ce_frozen_predpose_head
        if self.ce_frozen_predpose_normalization_profile is not None:
            result["ce_frozen_predpose_normalization_profile"] = (
                self.ce_frozen_predpose_normalization_profile
            )
        if self.full24_temporal_sampling_profile is not None:
            result["full24_temporal_sampling_profile"] = self.full24_temporal_sampling_profile
        if self.split_assignments_path is not None:
            result["split_assignments_path"] = self.split_assignments_path
        return result


@dataclass(frozen=True, slots=True)
class WaveLLMRunOptimizationConfig:
    epochs: int
    max_steps: int | None
    accumulate_grad_batches: int
    learning_rate: float
    language_model_learning_rate: float | None
    pose_branch_learning_rate: float | None
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip_norm: float

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMRunOptimizationConfig:
        location = "optimization"
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {
                "epochs",
                "max_steps",
                "accumulate_grad_batches",
                "learning_rate",
                "language_model_learning_rate",
                "pose_branch_learning_rate",
                "weight_decay",
                "beta1",
                "beta2",
                "gradient_clip_norm",
            },
            location,
        )
        max_steps_value = payload.get("max_steps")
        if max_steps_value is not None and (
            not isinstance(max_steps_value, int)
            or isinstance(max_steps_value, bool)
            or max_steps_value < 1
        ):
            raise WaveLLMRunError("optimization.max_steps must be null or a positive integer")
        learning_rate = _number(payload, "learning_rate", location)
        language_model_learning_rate = payload.get("language_model_learning_rate")
        pose_branch_learning_rate = payload.get("pose_branch_learning_rate")
        dual_rates = (
            language_model_learning_rate is not None or pose_branch_learning_rate is not None
        )
        if dual_rates and (
            language_model_learning_rate is None or pose_branch_learning_rate is None
        ):
            raise WaveLLMRunError(
                "optimization.language_model_learning_rate and "
                "optimization.pose_branch_learning_rate must be configured together"
            )
        if language_model_learning_rate is not None:
            if isinstance(language_model_learning_rate, bool) or not isinstance(
                language_model_learning_rate, (int, float)
            ):
                raise WaveLLMRunError("optimization.language_model_learning_rate must be numeric")
            language_model_learning_rate = float(language_model_learning_rate)
        if pose_branch_learning_rate is not None:
            if isinstance(pose_branch_learning_rate, bool) or not isinstance(
                pose_branch_learning_rate, (int, float)
            ):
                raise WaveLLMRunError("optimization.pose_branch_learning_rate must be numeric")
            pose_branch_learning_rate = float(pose_branch_learning_rate)
        gradient_clip_norm = _number(payload, "gradient_clip_norm", location)
        beta1 = _number(payload, "beta1", location)
        beta2 = _number(payload, "beta2", location)
        if learning_rate <= 0:
            raise WaveLLMRunError("optimization.learning_rate must be > 0")
        if language_model_learning_rate is not None and language_model_learning_rate <= 0:
            raise WaveLLMRunError("optimization.language_model_learning_rate must be > 0")
        if pose_branch_learning_rate is not None and pose_branch_learning_rate <= 0:
            raise WaveLLMRunError("optimization.pose_branch_learning_rate must be > 0")
        if gradient_clip_norm <= 0:
            raise WaveLLMRunError("optimization.gradient_clip_norm must be > 0")
        if beta1 >= 1 or beta2 >= 1:
            raise WaveLLMRunError("optimization beta values must be < 1")
        return cls(
            epochs=_integer(payload, "epochs", location),
            max_steps=max_steps_value,
            accumulate_grad_batches=(
                _integer(payload, "accumulate_grad_batches", location)
                if "accumulate_grad_batches" in payload
                else 1
            ),
            learning_rate=learning_rate,
            language_model_learning_rate=language_model_learning_rate,
            pose_branch_learning_rate=pose_branch_learning_rate,
            weight_decay=_number(payload, "weight_decay", location),
            beta1=beta1,
            beta2=beta2,
            gradient_clip_norm=gradient_clip_norm,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        if self.language_model_learning_rate is not None:
            result["language_model_learning_rate"] = self.language_model_learning_rate
            assert self.pose_branch_learning_rate is not None
            result["pose_branch_learning_rate"] = self.pose_branch_learning_rate
        return result

    @property
    def uses_dual_learning_rates(self) -> bool:
        return self.language_model_learning_rate is not None


@dataclass(frozen=True, slots=True)
class WaveLLMGenerationConfig:
    max_new_tokens: int
    num_beams: int

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMGenerationConfig:
        location = "generation"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"max_new_tokens", "num_beams"}, location)
        return cls(
            max_new_tokens=_integer(payload, "max_new_tokens", location),
            num_beams=_integer(payload, "num_beams", location),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
        }


@dataclass(frozen=True, slots=True)
class WaveLLMEvaluationConfig:
    save_references: bool

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMEvaluationConfig:
        location = "evaluation"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"save_references"}, location)
        save_references = payload.get("save_references")
        if not isinstance(save_references, bool):
            raise WaveLLMRunError("evaluation.save_references must be a boolean")
        return cls(save_references=save_references)

    def to_dict(self) -> dict[str, bool]:
        return {"save_references": self.save_references}


WAVELLM_POSE_ANCHORED_RESIDUAL_INITIALIZATION_KIND = (
    "official_pose_only_terminal_to_pose_anchored_residual_pf_v1"
)

WAVELLM_CONTEXTUAL_INITIALIZATION_KINDS = frozenset(
    (WAVELLM_POSE_ANCHORED_RESIDUAL_INITIALIZATION_KIND,)
)


def _resolved_project_path(
    value: object,
    *,
    field: str,
    project_root: str | Path | None,
    expect: Literal["file", "directory"],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WaveLLMRunError(f"{field} must be a non-empty path")
    if project_root is None:
        raise WaveLLMRunError(f"project_root is required to resolve {field}")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    path = path.resolve()
    valid = path.is_file() if expect == "file" else path.is_dir()
    if not valid:
        raise WaveLLMRunError(f"{field} is not a {expect}: {path}")
    return str(path)


@dataclass(frozen=True, slots=True)
class WaveLLMCheckpointInitialization:
    """Explicit partial initialization from a completed P-only terminal."""

    kind: str
    checkpoint_path: str
    expected_split_view_root: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        project_root: str | Path | None,
    ) -> WaveLLMCheckpointInitialization:
        location = "initialization"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"kind", "checkpoint", "expected_split_view"}, location)
        kind = _text(payload, "kind", location)
        if kind not in WAVELLM_CONTEXTUAL_INITIALIZATION_KINDS:
            raise WaveLLMRunError(f"{location}.kind is not a supported initialization route")
        return cls(
            kind=kind,
            checkpoint_path=_resolved_project_path(
                payload.get("checkpoint"),
                field=f"{location}.checkpoint",
                project_root=project_root,
                expect="file",
            ),
            expected_split_view_root=_resolved_project_path(
                payload.get("expected_split_view"),
                field=f"{location}.expected_split_view",
                project_root=project_root,
                expect="directory",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "checkpoint": self.checkpoint_path,
            "expected_split_view": self.expected_split_view_root,
        }


@dataclass(frozen=True, slots=True)
class WaveLLMContextualInputContract:
    """Named tensor contracts required by a contextual P+F consumer."""

    pose_contract: str
    feature_identity: str

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMContextualInputContract:
        location = "contextual_input_contract"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"pose_contract", "feature_identity"}, location)
        return cls(
            pose_contract=_text(payload, "pose_contract", location),
            feature_identity=_text(payload, "feature_identity", location),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "pose_contract": self.pose_contract,
            "feature_identity": self.feature_identity,
        }


@dataclass(frozen=True, slots=True)
class WaveLLMFusionTrainingPolicy:
    """Training boundary declared only by a non-default fusion profile."""

    freeze_inherited_backbone: bool

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMFusionTrainingPolicy:
        location = "fusion_training"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"freeze_inherited_backbone"}, location)
        freeze_inherited_backbone = payload.get("freeze_inherited_backbone")
        if not isinstance(freeze_inherited_backbone, bool):
            raise WaveLLMRunError("fusion_training.freeze_inherited_backbone must be a boolean")
        return cls(freeze_inherited_backbone=freeze_inherited_backbone)

    def to_dict(self) -> dict[str, bool]:
        return {"freeze_inherited_backbone": self.freeze_inherited_backbone}


WAVELLM_CONDITIONAL_NEGATIVE_STRATEGY = "local_closest_length_excluding_same_text_and_group_v1"


@dataclass(frozen=True, slots=True)
class WaveLLMConditionalDiscriminationConfig:
    coefficient: float
    tie_seed: int
    negative_strategy: str

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMConditionalDiscriminationConfig:
        location = "conditional_discrimination"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"coefficient", "tie_seed", "negative_strategy"}, location)
        coefficient = _number(payload, "coefficient", location)
        if not math.isfinite(coefficient):
            raise WaveLLMRunError(f"{location}.coefficient must be finite")
        tie_seed = _integer(payload, "tie_seed", location, minimum=0)
        negative_strategy = _text(payload, "negative_strategy", location)
        if negative_strategy != WAVELLM_CONDITIONAL_NEGATIVE_STRATEGY:
            raise WaveLLMRunError(
                f"{location}.negative_strategy must be {WAVELLM_CONDITIONAL_NEGATIVE_STRATEGY}"
            )
        return cls(
            coefficient=coefficient,
            tie_seed=tie_seed,
            negative_strategy=negative_strategy,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficient": self.coefficient,
            "tie_seed": self.tie_seed,
            "negative_strategy": self.negative_strategy,
        }


@dataclass(frozen=True, slots=True)
class WaveLLMPoseEncoderTransplant:
    """Resolved D1 two-source initializer for one explicit encoder profile."""

    kind: str
    base_state_path: str
    hand_checkpoint_path: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        project_root: str | Path,
    ) -> WaveLLMPoseEncoderTransplant:
        location = "pose_encoder_transplant"
        payload = _mapping(value, location)
        _reject_unknown(payload, {"kind", "base_state", "hand_checkpoint"}, location)
        kind = _text(payload, "kind", location)
        if kind != WAVELLM_UNISIGN_HAND_TRANSPLANT_KIND:
            raise WaveLLMRunError(f"{location}.kind is not supported")

        def resolve_path(key: str) -> str:
            raw = _text(payload, key, location)
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path(project_root) / path
            path = path.resolve()
            if not path.is_file():
                raise WaveLLMRunError(f"{location}.{key} is not a file: {path}")
            return str(path)

        return cls(
            kind=kind,
            base_state_path=resolve_path("base_state"),
            hand_checkpoint_path=resolve_path("hand_checkpoint"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "base_state": self.base_state_path,
            "hand_checkpoint": self.hand_checkpoint_path,
        }


@dataclass(frozen=True, slots=True)
class WaveLLMRunConfig:
    input_mode: str
    model: MT5ModelConfig
    pose_encoder_profile: ResolvedProfileBinding[WaveLLMPoseEncoderProfile] | None
    pose_encoder_initial_state: str | None
    pose_encoder_transplant: WaveLLMPoseEncoderTransplant | None
    fusion_variant: str
    initialization: WaveLLMCheckpointInitialization | None
    contextual_input_contract: WaveLLMContextualInputContract | None
    fusion_training: WaveLLMFusionTrainingPolicy | None
    conditional_discrimination: WaveLLMConditionalDiscriminationConfig | None
    data: WaveLLMDataConfig
    optimization: WaveLLMRunOptimizationConfig
    generation: WaveLLMGenerationConfig
    evaluation: WaveLLMEvaluationConfig

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        project_root: str | Path | None = None,
    ) -> WaveLLMRunConfig:
        payload = _mapping(value, "WaveLLM run config")
        _reject_unknown(
            payload,
            {
                "schema_version",
                "input_mode",
                "model",
                "pose_encoder_profile",
                "pose_encoder_initial_state",
                "pose_encoder_transplant",
                "fusion_variant",
                "initialization",
                "contextual_input_contract",
                "fusion_training",
                "conditional_discrimination",
                "data",
                "optimization",
                "generation",
                "evaluation",
            },
            "WaveLLM run config",
        )
        if payload.get("schema_version") != WAVELLM_RUN_CONFIG_SCHEMA:
            raise WaveLLMRunError(f"schema_version must be {WAVELLM_RUN_CONFIG_SCHEMA}")
        input_mode = payload.get("input_mode")
        if not isinstance(input_mode, str) or input_mode not in WAVELLM_INPUT_MODES:
            raise WaveLLMRunError(
                "input_mode must be pose_only, radar_feature_only, or pose_plus_radar_feature"
            )
        try:
            model = MT5ModelConfig.from_mapping(payload.get("model"))
        except ReaderConfigurationError as error:
            raise WaveLLMRunError(str(error)) from error
        if input_mode == POSE_ONLY_INPUT_MODE and model.radar_feature_dim is not None:
            raise WaveLLMRunError("pose_only config must not declare model.radar_feature_dim")
        if input_mode != POSE_ONLY_INPUT_MODE and model.radar_feature_dim is None:
            raise WaveLLMRunError("feature-consuming config requires model.radar_feature_dim")
        data = WaveLLMDataConfig.from_mapping(payload.get("data"))
        optimization = WaveLLMRunOptimizationConfig.from_mapping(payload.get("optimization"))
        if optimization.uses_dual_learning_rates and model.freeze_language_model:
            raise WaveLLMRunError("dual learning rates require model.freeze_language_model=false")
        if input_mode == RADAR_FEATURE_ONLY_INPUT_MODE and (
            "pose_encoder_profile" in payload
            or "pose_encoder_initial_state" in payload
            or "pose_encoder_transplant" in payload
        ):
            raise WaveLLMRunError(
                "radar_feature_only config must not declare a pose encoder profile, state, "
                "or transplant"
            )
        fusion_variant = payload.get("fusion_variant", WAVELLM_FUSION_COMPACT_CONFIDENCE_V1)
        if not isinstance(fusion_variant, str) or fusion_variant not in WAVELLM_FUSION_VARIANTS:
            raise WaveLLMRunError("fusion_variant is not a supported WaveLLM fusion profile")
        if (
            fusion_variant != WAVELLM_FUSION_COMPACT_CONFIDENCE_V1
            and input_mode != POSE_PLUS_RADAR_FEATURE_INPUT_MODE
        ):
            raise WaveLLMRunError(
                "non-default fusion_variant requires input_mode=pose_plus_radar_feature"
            )
        pose_encoder_profile = None
        if "pose_encoder_profile" in payload:
            if project_root is None:
                raise WaveLLMRunError("project_root is required to resolve pose_encoder_profile")
            pose_encoder_profile = _resolve_pose_encoder_profile(
                payload.get("pose_encoder_profile"), project_root=project_root
            )
            profile = pose_encoder_profile.domain_profile
            if profile.input_shape != (
                2,
                model.joint_count,
                model.coordinate_dim,
            ):
                raise WaveLLMRunError(
                    "pose encoder profile input shape does not match model joints"
                )
            if profile.output_dim != model.hidden_size:
                raise WaveLLMRunError(
                    "pose encoder profile output dimension does not match model hidden size"
                )
        pose_encoder_initial_state = None
        if "pose_encoder_initial_state" in payload:
            if pose_encoder_profile is None:
                raise WaveLLMRunError("pose_encoder_initial_state requires pose_encoder_profile")
            raw_state = payload.get("pose_encoder_initial_state")
            if not isinstance(raw_state, str) or not raw_state.strip():
                raise WaveLLMRunError("pose_encoder_initial_state must be a non-empty path")
            state_path = Path(raw_state.strip()).expanduser()
            if not state_path.is_absolute():
                if project_root is None:
                    raise WaveLLMRunError(
                        "project_root is required to resolve pose_encoder_initial_state"
                    )
                state_path = Path(project_root) / state_path
            state_path = state_path.resolve()
            if not state_path.is_file():
                raise WaveLLMRunError(f"pose_encoder_initial_state is not a file: {state_path}")
            pose_encoder_initial_state = str(state_path)
        pose_encoder_transplant = None
        if "pose_encoder_transplant" in payload:
            if project_root is None:
                raise WaveLLMRunError("project_root is required to resolve pose_encoder_transplant")
            pose_encoder_transplant = WaveLLMPoseEncoderTransplant.from_mapping(
                payload.get("pose_encoder_transplant"),
                project_root=project_root,
            )
        if pose_encoder_transplant is not None:
            if pose_encoder_profile is None:
                raise WaveLLMRunError("pose_encoder_transplant requires pose_encoder_profile")
            if pose_encoder_initial_state is not None:
                raise WaveLLMRunError(
                    "pose_encoder_transplant cannot be combined with pose_encoder_initial_state"
                )
        initialization = (
            WaveLLMCheckpointInitialization.from_mapping(
                payload.get("initialization"), project_root=project_root
            )
            if "initialization" in payload
            else None
        )
        contextual_input_contract = (
            WaveLLMContextualInputContract.from_mapping(payload.get("contextual_input_contract"))
            if "contextual_input_contract" in payload
            else None
        )
        fusion_training = (
            WaveLLMFusionTrainingPolicy.from_mapping(payload.get("fusion_training"))
            if "fusion_training" in payload
            else None
        )
        conditional_discrimination = (
            WaveLLMConditionalDiscriminationConfig.from_mapping(
                payload.get("conditional_discrimination")
            )
            if "conditional_discrimination" in payload
            else None
        )
        if initialization is not None:
            if fusion_variant != WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1:
                raise WaveLLMRunError("initialization is only defined for contextual P+F profiles")
            if pose_encoder_initial_state is not None:
                raise WaveLLMRunError(
                    "initialization cannot be combined with pose_encoder_initial_state"
                )
            if pose_encoder_transplant is not None:
                raise WaveLLMRunError(
                    "initialization cannot be combined with pose_encoder_transplant"
                )
        if fusion_variant == WAVELLM_FUSION_POSE_ANCHORED_RESIDUAL_V1:
            if model.hidden_size != 768 or model.radar_feature_dim != 1024:
                raise WaveLLMRunError(
                    "pose_anchored_residual_pf_v1 requires model.hidden_size=768 and "
                    "model.radar_feature_dim=1024"
                )
            if initialization is None:
                raise WaveLLMRunError(
                    "pose_anchored_residual_pf_v1 requires explicit P-only initialization"
                )
            if initialization.kind != WAVELLM_POSE_ANCHORED_RESIDUAL_INITIALIZATION_KIND:
                raise WaveLLMRunError(
                    "pose_anchored_residual_pf_v1 requires its declared P-only initialization route"
                )
            if contextual_input_contract is None:
                raise WaveLLMRunError(
                    "pose_anchored_residual_pf_v1 requires contextual_input_contract"
                )
            if fusion_training is None:
                raise WaveLLMRunError("pose_anchored_residual_pf_v1 requires fusion_training")
        elif contextual_input_contract is not None:
            if input_mode not in {POSE_ONLY_INPUT_MODE, RADAR_FEATURE_ONLY_INPUT_MODE}:
                raise WaveLLMRunError(
                    "contextual_input_contract without a contextual fusion requires "
                    "input_mode=pose_only or radar_feature_only"
                )
            if contextual_input_contract.pose_contract not in _WAVELLM_CONTEXTUAL_POSE_CONTRACTS:
                raise WaveLLMRunError("contextual input contract has an unsupported pose contract")
            if contextual_input_contract.feature_identity != WAVELLM_CONTEXTUAL_FEATURE_IDENTITY:
                raise WaveLLMRunError(
                    "contextual input contract requires the current contextual feature identity"
                )
            if (
                input_mode == RADAR_FEATURE_ONLY_INPUT_MODE
                and contextual_input_contract.pose_contract
                != WAVELLM_PURE_PREDICTED_RECON_V2_POSE_CONTRACT
            ):
                raise WaveLLMRunError(
                    "radar_feature_only contextual control requires the pure-predicted "
                    "P_recon_v2 pose contract"
                )
        elif fusion_training is not None:
            raise WaveLLMRunError(
                "fusion_training is only defined for pose_anchored_residual_pf_v1"
            )
        profile_frame = (
            wavellm_pose_encoder_frame(pose_encoder_profile.domain_profile.variant_id)
            if pose_encoder_profile is not None
            else None
        )
        if data.pose_frame is None:
            if profile_frame is not None:
                raise WaveLLMRunError("a pure-hand pose encoder profile requires data.pose_frame")
        else:
            if pose_encoder_profile is None:
                raise WaveLLMRunError("data.pose_frame requires a pure-hand pose encoder profile")
            if profile_frame != data.pose_frame:
                raise WaveLLMRunError(
                    "data.pose_frame does not match the pose encoder profile frame"
                )
            if pose_encoder_initial_state is not None:
                raise WaveLLMRunError(
                    "pure-hand pose encoders must be freshly initialized without a state file"
                )
        if pose_encoder_profile is not None:
            is_unisign_transplant_profile = (
                pose_encoder_profile.domain_profile.variant_id
                == WAVELLM_POSE_LEGACY_STGCN_UNISIGN_HAND_TRANSPLANT_ROOTCENTER_VARIANT
            )
            if is_unisign_transplant_profile and pose_encoder_transplant is None:
                raise WaveLLMRunError(
                    "Uni-Sign hand transplant profile requires pose_encoder_transplant"
                )
            if pose_encoder_transplant is not None and not is_unisign_transplant_profile:
                raise WaveLLMRunError(
                    "pose_encoder_transplant requires the Uni-Sign hand transplant profile"
                )
        return cls(
            input_mode=input_mode,
            model=model,
            pose_encoder_profile=pose_encoder_profile,
            pose_encoder_initial_state=pose_encoder_initial_state,
            pose_encoder_transplant=pose_encoder_transplant,
            fusion_variant=fusion_variant,
            initialization=initialization,
            contextual_input_contract=contextual_input_contract,
            fusion_training=fusion_training,
            conditional_discrimination=conditional_discrimination,
            data=data,
            optimization=optimization,
            generation=WaveLLMGenerationConfig.from_mapping(payload.get("generation")),
            evaluation=WaveLLMEvaluationConfig.from_mapping(payload.get("evaluation")),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": WAVELLM_RUN_CONFIG_SCHEMA,
            "input_mode": self.input_mode,
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "optimization": self.optimization.to_dict(),
            "generation": self.generation.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }
        if self.pose_encoder_profile is not None:
            payload["pose_encoder_profile"] = self.pose_encoder_profile.to_dict()
        if self.pose_encoder_initial_state is not None:
            payload["pose_encoder_initial_state"] = self.pose_encoder_initial_state
        if self.pose_encoder_transplant is not None:
            payload["pose_encoder_transplant"] = self.pose_encoder_transplant.to_dict()
        if self.fusion_variant != WAVELLM_FUSION_COMPACT_CONFIDENCE_V1:
            payload["fusion_variant"] = self.fusion_variant
        if self.initialization is not None:
            payload["initialization"] = self.initialization.to_dict()
        if self.contextual_input_contract is not None:
            payload["contextual_input_contract"] = self.contextual_input_contract.to_dict()
        if self.fusion_training is not None:
            payload["fusion_training"] = self.fusion_training.to_dict()
        if self.conditional_discrimination is not None:
            payload["conditional_discrimination"] = self.conditional_discrimination.to_dict()
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def model_fingerprint(self) -> str:
        payload: dict[str, object] = {
            "input_mode": self.input_mode,
            "model": self.model.to_dict(),
        }
        if self.pose_encoder_profile is not None:
            payload["pose_encoder_profile"] = self.pose_encoder_profile.to_dict()
        if self.fusion_variant != WAVELLM_FUSION_COMPACT_CONFIDENCE_V1:
            payload["fusion_variant"] = self.fusion_variant
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def training_fingerprint(self) -> str:
        payload = self.to_dict()
        optimization = dict(self.optimization.to_dict())
        optimization.pop("epochs")
        optimization.pop("max_steps")
        payload["optimization"] = optimization
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
