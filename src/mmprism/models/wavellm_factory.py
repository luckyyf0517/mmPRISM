from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from mmprism.config import (
    ResolvedProfileBinding,
    payload_fingerprint,
    reject_unknown,
    require_integer,
    require_mapping,
    require_sha256,
    require_stable_id,
)
from mmprism.models.wavellm_profiles import (
    WAVELLM_POSE_LEGACY_STGCN_FINGERWISE_ROOTCENTER_VARIANT,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4_VARIANT,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4B_VARIANT,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_CONFBRANCH_VARIANT,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_VARIANT,
    WAVELLM_POSE_LEGACY_STGCN_UNISIGN_HAND_TRANSPLANT_ROOTCENTER_VARIANT,
    WAVELLM_POSE_LEGACY_STGCN_VARIANT,
    WAVELLM_POSE_LIGHTWEIGHT_VARIANT,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VARIANT,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VELOCITY_VARIANT,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_VELOCITY_VARIANT,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_VARIANT,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_ROOT_RELATIVE_VARIANT,
    WaveLLMPoseEncoderProfile,
)


class WaveLLMModelFactoryError(RuntimeError):
    """Raised before an incompatible WaveLLM model can enter a run."""


@dataclass(frozen=True, slots=True)
class WaveLLMPoseEncoderDescriptor:
    variant_id: str
    profile_id: str
    profile_fingerprint: str
    input_shape: tuple[int, int, int]
    output_dim: int
    parameter_count: int
    state_schema_version: str
    state_key_shape_fingerprint: str

    @classmethod
    def from_mapping(cls, value: object) -> WaveLLMPoseEncoderDescriptor:
        location = "WaveLLM pose-encoder descriptor"
        payload = require_mapping(value, location)
        reject_unknown(
            payload,
            {
                "variant_id",
                "profile_id",
                "profile_fingerprint",
                "input_shape",
                "output_dim",
                "parameter_count",
                "state_schema_version",
                "state_key_shape_fingerprint",
            },
            location,
        )
        raw_shape = payload.get("input_shape")
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes, bytearray)):
            raise WaveLLMModelFactoryError(f"{location}.input_shape must be a sequence")
        shape = tuple(
            require_integer(item, f"{location}.input_shape[{index}]", minimum=1)
            for index, item in enumerate(raw_shape)
        )
        if len(shape) != 3:
            raise WaveLLMModelFactoryError(f"{location}.input_shape must contain three integers")
        return cls(
            variant_id=require_stable_id(payload.get("variant_id"), f"{location}.variant_id"),
            profile_id=require_stable_id(payload.get("profile_id"), f"{location}.profile_id"),
            profile_fingerprint=require_sha256(
                payload.get("profile_fingerprint"), f"{location}.profile_fingerprint"
            ),
            input_shape=(shape[0], shape[1], shape[2]),
            output_dim=require_integer(
                payload.get("output_dim"), f"{location}.output_dim", minimum=1
            ),
            parameter_count=require_integer(
                payload.get("parameter_count"),
                f"{location}.parameter_count",
                minimum=1,
            ),
            state_schema_version=require_stable_id(
                payload.get("state_schema_version"),
                f"{location}.state_schema_version",
            ),
            state_key_shape_fingerprint=require_sha256(
                payload.get("state_key_shape_fingerprint"),
                f"{location}.state_key_shape_fingerprint",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "profile_id": self.profile_id,
            "profile_fingerprint": self.profile_fingerprint,
            "input_shape": list(self.input_shape),
            "output_dim": self.output_dim,
            "parameter_count": self.parameter_count,
            "state_schema_version": self.state_schema_version,
            "state_key_shape_fingerprint": self.state_key_shape_fingerprint,
        }


def _state_key_shape_fingerprint(state: Mapping[str, Tensor]) -> str:
    inventory = [
        {"name": name, "shape": list(tensor.shape)} for name, tensor in sorted(state.items())
    ]
    return payload_fingerprint(inventory)


_EXPECTED_PARAMETER_COUNTS = {
    WAVELLM_POSE_LIGHTWEIGHT_VARIANT: 161_344,
    WAVELLM_POSE_LEGACY_STGCN_VARIANT: 3_628_792,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_VARIANT: 3_628_792,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_CONFBRANCH_VARIANT: 3_633_080,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4_VARIANT: 3_628_856,
    WAVELLM_POSE_LEGACY_STGCN_ROOTCENTER_C4B_VARIANT: 3_633_144,
    WAVELLM_POSE_LEGACY_STGCN_FINGERWISE_ROOTCENTER_VARIANT: 3_712_760,
    WAVELLM_POSE_LEGACY_STGCN_UNISIGN_HAND_TRANSPLANT_ROOTCENTER_VARIANT: 3_629_048,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_VARIANT: 2_720_184,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_ROOT_RELATIVE_VARIANT: 2_720_184,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_VELOCITY_VARIANT: 2_720_772,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VARIANT: 2_720_382,
    WAVELLM_POSE_ROOT_ABS_HAND_REL21_CAMERA_POSITION_BONE_VELOCITY_VARIANT: 2_720_970,
}


def describe_wavellm_pose_encoder(
    binding: ResolvedProfileBinding[WaveLLMPoseEncoderProfile],
    model: nn.Module,
) -> WaveLLMPoseEncoderDescriptor:
    profile = binding.domain_profile
    profile.validate_fixed_contract()
    if binding.envelope.profile_id != profile.variant_id:
        raise WaveLLMModelFactoryError("pose-encoder profile ID must equal its stable variant ID")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_count = _EXPECTED_PARAMETER_COUNTS[profile.variant_id]
    if parameter_count != expected_count:
        raise WaveLLMModelFactoryError(
            f"{profile.variant_id} parameter count differs from its fixed profile"
        )
    return WaveLLMPoseEncoderDescriptor(
        variant_id=profile.variant_id,
        profile_id=binding.envelope.profile_id,
        profile_fingerprint=binding.fingerprint,
        input_shape=profile.input_shape,
        output_dim=profile.output_dim,
        parameter_count=parameter_count,
        state_schema_version=profile.state_schema_version,
        state_key_shape_fingerprint=_state_key_shape_fingerprint(model.state_dict()),
    )
