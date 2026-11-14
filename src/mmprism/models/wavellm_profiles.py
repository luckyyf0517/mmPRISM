from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from mmprism.config import (
    ConfigError,
    reject_unknown,
    require_integer,
    require_mapping,
    require_schema_version,
    require_stable_id,
)

WAVELLM_POSE_ENCODER_PROFILE_SCHEMA = "mmprism.wavellm_pose_encoder_profile.v1"

WAVELLM_POSE_ENCODER_PROFILE_KIND = "models"

WAVELLM_POSE_LIGHTWEIGHT_VARIANT = "wavellm_pose_lightweight_v1"

WAVELLM_POSE_LEGACY_STGCN_VARIANT = "wavellm_pose_legacy_stgcn_v1"

WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_VARIANT = "wavellm_pose_legacy_stgcn_rootcenter_v1"

WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_CONFBRANCH_VARIANT = (
    "wavellm_pose_legacy_stgcn_rootcenter_confbranch_v1"
)

WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4_VARIANT = "wavellm_pose_legacy_stgcn_rootcenter_c4_v1"

WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4B_VARIANT = "wavellm_pose_legacy_stgcn_rootcenter_c4b_v1"

WAVELLM_POSE_LEGACY_STGCN_FINGERWISE_ROOTCENTER_VARIANT = (
    "wavellm_pose_legacy_stgcn_fingerwise_rootcenter_v1"
)

WAVELLM_POSE_LEGACY_STGCN_UNISIGN_HAND_TRANSPLANT_ROOTCENTER_VARIANT = (
    "wavellm_pose_legacy_stgcn_unisign_hand_transplant_rootcenter_v1"
)

WAVELLM_UNISIGN_HAND_TRANSPLANT_KIND = "unisign_hand_stage3_pose_only_v1"

WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_VARIANT = "wavellm_pose_root_abs_hand_rel21_camera_stgcn_v1"

WAVELLM_POSE_ROOT_ABS_HAND_REL21_ROOT_RELATIVE_VARIANT = (
    "wavellm_pose_root_abs_hand_rel21_root_relative_stgcn_v1"
)

WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_VELOCITY_VARIANT = (
    "wavellm_pose_root_abs_hand_rel21_camera_position_velocity_stgcn_v1"
)

WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VARIANT = (
    "wavellm_pose_root_abs_hand_rel21_camera_position_bone_stgcn_v1"
)

WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VELOCITY_VARIANT = (
    "wavellm_pose_root_abs_hand_rel21_camera_position_bone_velocity_stgcn_v1"
)

type WaveLLMPoseEncoderVariant = Literal[
    "wavellm_pose_lightweight_v1",
    "wavellm_pose_legacy_stgcn_v1",
    "wavellm_pose_legacy_stgcn_rootcenter_v1",
    "wavellm_pose_legacy_stgcn_rootcenter_confbranch_v1",
    "wavellm_pose_legacy_stgcn_rootcenter_c4_v1",
    "wavellm_pose_legacy_stgcn_rootcenter_c4b_v1",
    "wavellm_pose_legacy_stgcn_fingerwise_rootcenter_v1",
    "wavellm_pose_legacy_stgcn_unisign_hand_transplant_rootcenter_v1",
    "wavellm_pose_root_abs_hand_rel21_camera_stgcn_v1",
    "wavellm_pose_root_abs_hand_rel21_root_relative_stgcn_v1",
    "wavellm_pose_root_abs_hand_rel21_camera_position_velocity_stgcn_v1",
    "wavellm_pose_root_abs_hand_rel21_camera_position_bone_stgcn_v1",
    "wavellm_pose_root_abs_hand_rel21_camera_position_bone_velocity_stgcn_v1",
]

_VARIANT_IDS = frozenset(
    {
        WAVELLM_POSE_LIGHTWEIGHT_VARIANT,
        WAVELLM_POSE_LEGACY_STGCN_VARIANT,
        WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_VARIANT,
        WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_CONFBRANCH_VARIANT,
        WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4_VARIANT,
        WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4B_VARIANT,
        WAVELLM_POSE_LEGACY_STGCN_FINGERWISE_ROOTCENTER_VARIANT,
        WAVELLM_POSE_LEGACY_STGCN_UNISIGN_HAND_TRANSPLANT_ROOTCENTER_VARIANT,
        WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_VARIANT,
        WAVELLM_POSE_ROOT_ABS_HAND_REL21_ROOT_RELATIVE_VARIANT,
        WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_VELOCITY_VARIANT,
        WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VARIANT,
        WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VELOCITY_VARIANT,
    }
)


class WaveLLMPoseEncoderProfileError(ConfigError):
    """Raised when a WaveLLM pose-encoder profile is not an exact variant."""


def _input_shape(value: object, location: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WaveLLMPoseEncoderProfileError(f"{location} must be a sequence")
    shape = tuple(
        require_integer(item, f"{location}[{index}]", minimum=1) for index, item in enumerate(value)
    )
    if len(shape) != 3:
        raise WaveLLMPoseEncoderProfileError(
            f"{location} must contain hand, joint, and coordinate dimensions"
        )
    return shape[0], shape[1], shape[2]


@dataclass(frozen=True, slots=True)
class WaveLLMPoseEncoderProfile:
    variant_id: WaveLLMPoseEncoderVariant
    state_schema_version: str
    input_shape: tuple[int, int, int]
    output_dim: int

    @classmethod
    def from_parameters(
        cls, schema_version: str, parameters: Mapping[str, object]
    ) -> WaveLLMPoseEncoderProfile:
        require_schema_version(
            schema_version,
            WAVELLM_POSE_ENCODER_PROFILE_SCHEMA,
            "WaveLLM pose-encoder profile.schema_version",
        )
        location = "WaveLLM pose-encoder profile.parameters"
        payload = require_mapping(parameters, location)
        reject_unknown(
            payload,
            {
                "variant_id",
                "state_schema_version",
                "input_shape",
                "output_dim",
            },
            location,
        )
        variant_id = require_stable_id(payload.get("variant_id"), f"{location}.variant_id")
        if variant_id not in _VARIANT_IDS:
            raise WaveLLMPoseEncoderProfileError(
                f"{location}.variant_id is not a registered stable model ID"
            )
        result = cls(
            variant_id=cast(WaveLLMPoseEncoderVariant, variant_id),
            state_schema_version=require_stable_id(
                payload.get("state_schema_version"),
                f"{location}.state_schema_version",
            ),
            input_shape=_input_shape(payload.get("input_shape"), f"{location}.input_shape"),
            output_dim=require_integer(
                payload.get("output_dim"), f"{location}.output_dim", minimum=1
            ),
        )
        result.validate_fixed_contract()
        return result

    def validate_fixed_contract(self) -> None:
        observed = {
            "state_schema_version": self.state_schema_version,
            "input_shape": self.input_shape,
            "output_dim": self.output_dim,
        }
        if observed != _FIXED_VARIANT_CONTRACTS[self.variant_id]:
            raise WaveLLMPoseEncoderProfileError(
                f"{self.variant_id} does not match its fixed architecture contract"
            )

    def to_parameters(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "state_schema_version": self.state_schema_version,
            "input_shape": list(self.input_shape),
            "output_dim": self.output_dim,
        }


_FIXED_VARIANT_CONTRACTS: dict[str, dict[str, object]] = {
    WAVELLM_POSE_LIGHTWEIGHT_VARIANT: {
        "state_schema_version": "wavellm_pose_lightweight_state_v1",
        "input_shape": (2, 24, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_LEGACY_STGCN_VARIANT: {
        "state_schema_version": "wavellm_pose_legacy_stgcn_state_v1",
        "input_shape": (2, 24, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_VARIANT: {
        "state_schema_version": "wavellm_pose_legacy_stgcn_rootcenter_state_v1",
        "input_shape": (2, 24, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_CONFBRANCH_VARIANT: {
        "state_schema_version": ("wavellm_pose_legacy_stgcn_rootcenter_confbranch_state_v1"),
        "input_shape": (2, 24, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4_VARIANT: {
        "state_schema_version": ("wavellm_pose_legacy_stgcn_rootcenter_c4_state_v1"),
        "input_shape": (2, 24, 4),
        "output_dim": 768,
    },
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4B_VARIANT: {
        "state_schema_version": ("wavellm_pose_legacy_stgcn_rootcenter_c4b_state_v1"),
        "input_shape": (2, 24, 4),
        "output_dim": 768,
    },
    WAVELLM_POSE_LEGACY_STGCN_FINGERWISE_ROOTCENTER_VARIANT: {
        "state_schema_version": "wavellm_pose_legacy_stgcn_fingerwise_rootcenter_state_v1",
        "input_shape": (2, 24, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_LEGACY_STGCN_UNISIGN_HAND_TRANSPLANT_ROOTCENTER_VARIANT: {
        "state_schema_version": (
            "wavellm_pose_legacy_stgcn_unisign_hand_transplant_rootcenter_state_v1"
        ),
        "input_shape": (2, 24, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_VARIANT: {
        "state_schema_version": "wavellm_pose_root_abs_hand_rel21_camera_stgcn_state_v1",
        "input_shape": (2, 22, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_ROOT_RELATIVE_VARIANT: {
        "state_schema_version": ("wavellm_pose_root_abs_hand_rel21_root_relative_stgcn_state_v1"),
        "input_shape": (2, 22, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_VELOCITY_VARIANT: {
        "state_schema_version": (
            "wavellm_pose_root_abs_hand_rel21_camera_position_velocity_stgcn_state_v1"
        ),
        "input_shape": (2, 22, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VARIANT: {
        "state_schema_version": (
            "wavellm_pose_root_abs_hand_rel21_camera_position_bone_stgcn_state_v1"
        ),
        "input_shape": (2, 22, 3),
        "output_dim": 768,
    },
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VELOCITY_VARIANT: {
        "state_schema_version": (
            "wavellm_pose_root_abs_hand_rel21_camera_position_bone_velocity_stgcn_state_v1"
        ),
        "input_shape": (2, 22, 3),
        "output_dim": 768,
    },
}

_POSE_FRAME_BY_VARIANT = {
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_VARIANT: "camera",
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_ROOT_RELATIVE_VARIANT: "root_relative",
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_VELOCITY_VARIANT: "camera",
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VARIANT: "camera",
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VELOCITY_VARIANT: "camera",
}


def wavellm_pose_encoder_frame(variant_id: str) -> str | None:
    """Return the pure-hand frame bound by a variant, if it has one."""

    return _POSE_FRAME_BY_VARIANT.get(variant_id)
