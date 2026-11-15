"""WaveLLM mesh-FK L2 episode dataset: handoff-bound, episode-index reader.

One dataset instance serves one split of one validated
``mmprism.csl_daily_mesh_fk_wavellm_handoff.v1``:

1. construction validates the handoff hash set against the declared L2
   build (delivery build id, episode-index sha256, sidecar sha256, member
   digest) and resolves the split's members from
   ``eligible_episodes.jsonl`` — never from a sibling-path convention;
2. ``__getitem__`` opens only the member's declared part, reads only the
   episode's row range with a projection of the core columns
   (``sample_id/frame_index/caption/pose/pose_confidence/frame_mask``),
   and never touches SMPL-X/MANO parameter or QC columns;
3. every decoded episode is validated (identity, contiguous frames,
   finite pose, confidence bounds, mask semantics) before the train-only
   normalization is applied exactly once — root-center, then the bound
   mean/std — exposing the existing ``SignLanguageTranslationSample``
   contract in ``model_camera_pseudo_units``.

Part file handles are cached per instance with a small bounded LRU; no
corpus scan or legacy sidecar path is used at sample time.  PyArrow is
imported lazily.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from mmprism.contracts import SplitAssignment, SplitContractError, validate_split_assignments
from mmprism.data.csl_daily_mesh_fk_handoff import HANDOFF_SCHEMA_NAME
from mmprism.data.csl_daily_mesh_l2 import (
    MESH_L2_SCHEMA_NAME,
    MESH_NORMALIZATION_PROTOCOL_ID,
    POSE_FLAT,
    ROOT_JOINT_INDEX,
    _member_digest,
    _sha256_file,
)
from mmprism.data.csl_daily_mesh_l2 import MESH_POSE_SOURCE as CANONICAL_MESH_POSE_SOURCE
from mmprism.data.sign_language_translation import SignLanguageTranslationSample

MESH_POSE_SOURCE = CANONICAL_MESH_POSE_SOURCE

SAM3D_YOKE_POSE_SOURCE = "sam3d_body_vith_upperfree_mannequin_v3_yoke_full24"

SAM3D_YOKE_L2_SCHEMA_NAME = "mmprism.csl_daily_sam3d_yoke_annotation.v1"

SAM3D_YOKE_HANDOFF_SCHEMA_NAME = "mmprism.csl_daily_sam3d_yoke_wavellm_handoff.v1"

SAM3D_YOKE_SMOOTHW5_POSE_SOURCE = "sam3d_body_vith_upperfree_mannequin_v3_yoke_full24_smoothw5"

SAM3D_YOKE_SMOOTHW5_L2_SCHEMA_NAME = "mmprism.csl_daily_sam3d_yoke_annotation.smoothw5.v1"

SAM3D_YOKE_SMOOTHW5_HANDOFF_SCHEMA_NAME = "mmprism.csl_daily_sam3d_yoke_wavellm_handoff.smoothw5.v1"

SAM3D_CAMERA_POSE_SOURCE = "sam3d_body_vith_upperfree_mannequin_v3_camera_full24"

SAM3D_HYBRID24_SOURCE_L2_SCHEMA_NAME = "mmprism.csl_daily_sam3d_hybrid24_source_annotation.v1"

SAM3D_HYBRID24_SOURCE_HANDOFF_SCHEMA_NAME = (
    "mmprism.csl_daily_sam3d_hybrid24_source_wavellm_handoff.v1"
)

ALLOWED_POSE_SOURCES = (
    MESH_POSE_SOURCE,
    SAM3D_YOKE_POSE_SOURCE,
    SAM3D_YOKE_SMOOTHW5_POSE_SOURCE,
    SAM3D_CAMERA_POSE_SOURCE,
)

ALLOWED_L2_SCHEMA_NAMES = (
    MESH_L2_SCHEMA_NAME,
    SAM3D_YOKE_L2_SCHEMA_NAME,
    SAM3D_YOKE_SMOOTHW5_L2_SCHEMA_NAME,
    SAM3D_HYBRID24_SOURCE_L2_SCHEMA_NAME,
)

ALLOWED_HANDOFF_SCHEMA_NAMES = (
    HANDOFF_SCHEMA_NAME,
    SAM3D_YOKE_HANDOFF_SCHEMA_NAME,
    SAM3D_YOKE_SMOOTHW5_HANDOFF_SCHEMA_NAME,
    SAM3D_HYBRID24_SOURCE_HANDOFF_SCHEMA_NAME,
)

PROJECTION_COLUMNS = (
    "sample_id",
    "frame_index",
    "caption",
    "pose",
    "pose_confidence",
    "frame_mask",
)

NORMALIZED_POSE_UNITS = "model_camera_pseudo_units"

DEFAULT_PART_CACHE_SIZE = 8

PURE_HAND_POSE_FRAME_ROOT_RELATIVE = "root_relative"

PURE_HAND_POSE_FRAME_CAMERA = "camera"

type PureHandPoseFrame = Literal["root_relative", "camera"]

PURE_HAND_POSE_FRAMES = frozenset({PURE_HAND_POSE_FRAME_ROOT_RELATIVE, PURE_HAND_POSE_FRAME_CAMERA})

PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08 = "radar_camera_y_up_shoulder_standoff_0p8m"

type PureHandCoordinateFrame = Literal["radar_camera_y_up_shoulder_standoff_0p8m"]

PURE_HAND_COORDINATE_FRAMES = frozenset({PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08})

PURE_HAND_SOURCE_SHOULDER_JOINT_INDEX = 0

PURE_HAND_RADAR_CAMERA_AXIS_FLIP = (False, True, False)

PURE_HAND_RADAR_CAMERA_SHOULDER_MIDPOINT_M = (0.0, 0.0, 0.8)

PURE_HAND_ROOT_SOURCE_JOINT_INDEX = 2

PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES = tuple(range(3, 24))

PURE_HAND_SOURCE_JOINT_INDICES = (
    PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
    *PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES,
)

PURE_HAND_GEOMETRY_JOINT_COUNT = len(PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES)

PURE_HAND_JOINT_COUNT = len(PURE_HAND_SOURCE_JOINT_INDICES)

PURE_HAND_POSE_FLAT = 2 * PURE_HAND_JOINT_COUNT * 3

PURE_HAND_NORMALIZATION_SCHEMA = "mmprism.wavellm_pure_hand_normalization.v3"

PURE_HAND_NORMALIZATION_PROTOCOL = (
    "mmprism.wavellm_root_abs_hand_rel21_radar_camera_global_affine.train_only_v2"
)

PURE_HAND_POSE_VIEW = "wrist_abs_plus_palm_fingers_relative_radar_camera_v2"

PURE_HAND_NORMALIZED_POSE_UNITS = "model_radar_camera_scene_units"

FULL24_EXTERNAL_NORMALIZATION_SCHEMA = "mmprism.wavellm_mesh_fk_full24_normalization.v1"

FULL24_EXTERNAL_NORMALIZATION_PROTOCOL = (
    "mmprism.wavellm_mesh_fk_full24_root_center_global_affine.train_only_v1"
)

FULL24_POSE_VIEW = "legacy_full24_per_hand_shoulder_root_centered_v1"

HYBRID24_EXTERNAL_NORMALIZATION_SCHEMA = "mmprism.wavellm_hybrid24_normalization.v1"

HYBRID24_EXTERNAL_NORMALIZATION_PROTOCOL = (
    "mmprism.wavellm_hybrid24_root_abs_hand_rel21_radar_camera_global_affine.train_only_v1"
)

HYBRID24_POSE_VIEW = "hybrid24_wrist_abs_plus_palm_fingers_relative_radar_camera_v1"

HYBRID24_NORMALIZED_POSE_UNITS = "model_radar_camera_scene_units"

HYBRID24_NORMALIZED_COORDINATE_FRAME = (
    f"{PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08}_hybrid24_rootabs_handrel21"
)

HYBRID24_ZERO_VARIANCE_AXES = (*range(6), *range(72, 78))

HYBRID24_V2_EXTERNAL_NORMALIZATION_SCHEMA = "mmprism.wavellm_hybrid24v2_normalization.v1"

HYBRID24_V2_EXTERNAL_NORMALIZATION_PROTOCOL = (
    "mmprism.wavellm_hybrid24_root_abs_hand_global_radar_camera_global_affine.train_only_v1"
)

HYBRID24_V2_POSE_VIEW = "hybrid24v2_wrist_abs_plus_palm_fingers_global_radar_camera_v1"

HYBRID24_V2_NORMALIZED_POSE_UNITS = "model_radar_camera_scene_units"

HYBRID24_V2_ZERO_VARIANCE_AXES = (*range(6), *range(72, 78))

UNISIGN_HAND_BRIDGE_SIDECAR_SCHEMA = "mmprism.wavellm_unisign_hand_bridge_sidecar.v1"

UNISIGN_HAND_BRIDGE_PROTOCOL = (
    "mmprism.wavellm_unisign_hand_bridge_bodyproxy_crop_scale.train_only_v1"
)

UNISIGN_HAND_BRIDGE_POSE_VIEW = "unisign_hand_bridge_wrist_global_hand_relative_xy_v1"

UNISIGN_HAND_BRIDGE_POSE_UNITS = "unisign_bodyproxy_crop_xy_score_units"

UNISIGN_HAND_BRIDGE_BODY_PROXY_SOURCE_JOINT_INDICES = (0, 1, 2)

UNISIGN_HAND_BRIDGE_FINGER_SOURCE_JOINT_INDICES = tuple(range(4, 24))

UNISIGN_HAND_BRIDGE_SCORE = 1.0

FULL24_TEMPORAL_SAMPLING_LEGACY_CONTEXTUAL_10HZ = "legacy_contextual_10hz_stride3_successor_v1"

type Full24TemporalSamplingProfile = Literal["legacy_contextual_10hz_stride3_successor_v1"]

FULL24_TEMPORAL_SAMPLING_PROFILES = frozenset({FULL24_TEMPORAL_SAMPLING_LEGACY_CONTEXTUAL_10HZ})

EXTERNAL_SPLIT_ROLES = frozenset({"train", "validation", "test"})


class MeshFkL2DatasetError(RuntimeError):
    """Raised when the closed L2 source fails a binding or decode gate."""


def _require_pure_hand_pose_frame(value: str) -> PureHandPoseFrame:
    if value == PURE_HAND_POSE_FRAME_ROOT_RELATIVE:
        return cast(PureHandPoseFrame, value)
    if value == PURE_HAND_POSE_FRAME_CAMERA:
        return cast(PureHandPoseFrame, value)
    raise MeshFkL2DatasetError(
        f"unsupported pure-hand pose_frame {value!r}; expected one of "
        f"{sorted(PURE_HAND_POSE_FRAMES)}"
    )


def _require_pure_hand_coordinate_frame(value: str) -> PureHandCoordinateFrame:
    if value == PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08:
        return cast(PureHandCoordinateFrame, value)
    raise MeshFkL2DatasetError(
        f"unsupported pure-hand coordinate frame {value!r}; expected one of "
        f"{sorted(PURE_HAND_COORDINATE_FRAMES)}"
    )


def _require_full24_temporal_sampling_profile(value: str) -> Full24TemporalSamplingProfile:
    if value == FULL24_TEMPORAL_SAMPLING_LEGACY_CONTEXTUAL_10HZ:
        return cast(Full24TemporalSamplingProfile, value)
    raise MeshFkL2DatasetError(
        f"unsupported full-24 temporal sampling profile {value!r}; expected one of "
        f"{sorted(FULL24_TEMPORAL_SAMPLING_PROFILES)}"
    )


def full24_temporal_sampling_identity(
    profile: Full24TemporalSamplingProfile | None,
) -> dict[str, object] | None:
    """Describe an explicit full-24 loader sampling view."""

    if profile is None:
        return None
    resolved = _require_full24_temporal_sampling_profile(profile)
    if resolved == FULL24_TEMPORAL_SAMPLING_LEGACY_CONTEXTUAL_10HZ:
        return {
            "id": resolved,
            "source_nominal_hz": 30.0,
            "output_hz": 10.0,
            "source_frame_stride": 3,
            "target_indices": "range(0, frame_count - 3, 3)",
            "terminal_successor_frames_excluded": 3,
        }
    raise AssertionError(f"unsupported full-24 temporal sampling profile {resolved!r}")


def full24_temporal_indices(
    frame_count: int,
    profile: Full24TemporalSamplingProfile | None,
    phase: int = 0,
) -> np.ndarray:
    """Return the exact historical contextual 10 Hz target grid when requested.

    ``phase`` selects one of the three stride-3 grids (0, 1, or 2). The
    default 0 is the canonical historical grid; the train-only random-phase
    policy draws the phase per (sample, epoch) elsewhere.
    """

    if frame_count < 1:
        raise MeshFkL2DatasetError("full-24 temporal sampling requires at least one frame")
    if not 0 <= phase <= 2:
        raise MeshFkL2DatasetError(f"temporal phase must be 0..2, got {phase}")
    if profile is None:
        if phase != 0:
            raise MeshFkL2DatasetError("temporal phase requires a sampling profile")
        return np.arange(frame_count, dtype=np.int64)
    resolved = _require_full24_temporal_sampling_profile(profile)
    if resolved == FULL24_TEMPORAL_SAMPLING_LEGACY_CONTEXTUAL_10HZ:
        indices = np.arange(phase, frame_count - 3, 3, dtype=np.int64)
        if indices.size < 1:
            raise MeshFkL2DatasetError(
                "legacy contextual 10 Hz sampling requires at least four source frames"
            )
        return indices
    raise AssertionError(f"unsupported full-24 temporal sampling profile {resolved!r}")


def pure_hand_coordinate_transform(
    coordinate_frame: PureHandCoordinateFrame,
) -> dict[str, object]:
    """Describe the fixed source-to-radar-camera P1 placement."""

    frame = _require_pure_hand_coordinate_frame(coordinate_frame)
    if frame != PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08:  # pragma: no cover
        raise AssertionError(f"unsupported pure-hand coordinate frame {frame!r}")
    return {
        "id": frame,
        "source_coordinate_semantics": (
            "bilateral_shoulder_midpoint_centered_smplx_mano_fk_metres"
        ),
        "source_anchor": {
            "kind": "per_frame_bilateral_hand_shoulder_midpoint",
            "joint_index": PURE_HAND_SOURCE_SHOULDER_JOINT_INDEX,
        },
        "axis_flip": list(PURE_HAND_RADAR_CAMERA_AXIS_FLIP),
        "target_coordinate_semantics": "radar_camera_origin_y_up_scene_metres",
        "shoulder_midpoint_position_m": list(PURE_HAND_RADAR_CAMERA_SHOULDER_MIDPOINT_M),
    }


def pure_hand_source_to_radar_camera_pose(
    pose: np.ndarray,
    coordinate_frame: PureHandCoordinateFrame,
) -> np.ndarray:
    """Place shoulder-centered source FK in the fixed radar-camera scene.

    The mesh L2 stores its bilateral shoulder midpoint as the per-frame
    origin.  P1 instead uses the co-located camera/radar origin: source
    image-style ``+y`` becomes physical ``+y`` up and the shoulder midpoint
    is placed at ``[0, 0, 0.8]`` m.  This is a loader view only; L2 bytes are
    never rewritten.
    """

    _require_pure_hand_coordinate_frame(coordinate_frame)
    if pose.ndim < 4 or pose.shape[-3:] != (2, 24, 3):
        raise MeshFkL2DatasetError(
            f"pure-hand source pose must end in [2,24,3], got {tuple(pose.shape)}"
        )
    source = np.asarray(pose, dtype=np.float32)
    shoulders = source[..., :, PURE_HAND_SOURCE_SHOULDER_JOINT_INDEX, :]
    shoulder_midpoint = shoulders.mean(axis=-2)
    centered = source - shoulder_midpoint[..., np.newaxis, np.newaxis, :]
    axis_sign = _pure_hand_radar_camera_axis_sign()
    anchor = np.asarray(PURE_HAND_RADAR_CAMERA_SHOULDER_MIDPOINT_M, dtype=np.float32)
    return np.asarray(centered * axis_sign + anchor, dtype=np.float32)


def _pure_hand_radar_camera_axis_sign() -> np.ndarray:
    return np.asarray(
        [-1.0 if flipped else 1.0 for flipped in PURE_HAND_RADAR_CAMERA_AXIS_FLIP],
        dtype=np.float32,
    )


def extract_pure_hand_root_and_geometry(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one frame-local wrist root and 21 palm/finger relative points."""

    if pose.ndim < 4 or pose.shape[-3:] != (2, 24, 3):
        raise MeshFkL2DatasetError(
            f"pure-hand source pose must end in [2,24,3], got {tuple(pose.shape)}"
        )
    root = np.asarray(
        pose[
            ...,
            PURE_HAND_ROOT_SOURCE_JOINT_INDEX : PURE_HAND_ROOT_SOURCE_JOINT_INDEX + 1,
            :,
        ],
        dtype=np.float32,
    )
    geometry_camera = np.asarray(
        pose[..., PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES, :],
        dtype=np.float32,
    )
    return root, np.asarray(geometry_camera - root, dtype=np.float32)


def _pure_hand_root_slot(pose_frame: PureHandPoseFrame) -> dict[str, object]:
    if pose_frame == PURE_HAND_POSE_FRAME_CAMERA:
        kind = "absolute_radar_camera_wrist"
    else:
        kind = "zeroed_radar_camera_wrist"
    return {
        "kind": kind,
        "source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
        "output_joint_index": 0,
    }


def pure_hand_pose_view(
    pose: np.ndarray,
    pose_frame: PureHandPoseFrame,
    coordinate_frame: PureHandCoordinateFrame,
) -> np.ndarray:
    """Return P1's placed wrist-absolute plus hand-relative hands22 view."""

    frame = _require_pure_hand_pose_frame(pose_frame)
    camera_pose = pure_hand_source_to_radar_camera_pose(pose, coordinate_frame)
    root, _ = extract_pure_hand_root_and_geometry(camera_pose)
    _, source_geometry = extract_pure_hand_root_and_geometry(pose)
    # Translation cancels from hand geometry. Form it before the fixed scene
    # anchor is added so slot 1..21 do not lose precision through cancellation.
    geometry = np.asarray(source_geometry * _pure_hand_radar_camera_axis_sign(), dtype=np.float32)
    root_slot = root if frame == PURE_HAND_POSE_FRAME_CAMERA else np.zeros_like(root)
    return np.asarray(np.concatenate((root_slot, geometry), axis=-2), dtype=np.float32)


def pure_hand_confidence_view(confidence: np.ndarray) -> np.ndarray:
    """Select wrist, palm, and finger confidence in the P1 output order."""

    if confidence.ndim < 3 or confidence.shape[-2:] != (2, 24):
        raise MeshFkL2DatasetError(
            f"pure-hand source confidence must end in [2,24], got {tuple(confidence.shape)}"
        )
    return np.asarray(confidence[..., PURE_HAND_SOURCE_JOINT_INDICES], dtype=np.float32)


def _hybrid24_slot_layout() -> dict[str, object]:
    return {
        "zero_slots": [0, 1],
        "absolute_wrist": {
            "output_joint_index": 2,
            "source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
            "coordinate_frame": PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        },
        "relative_hand_geometry": {
            "output_joint_indices": [3, 23],
            "source_joint_indices": list(PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES),
            "root_source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
        },
    }


def hybrid24_pose_view(pose: np.ndarray) -> np.ndarray:
    """Return the warm-compatible 24-slot pure-hand loader view.

    Slots 0 and 1 intentionally carry no arm data.  The remaining slots are
    copied from the camera pure-hand view so their values are byte-for-byte
    equivalent before normalization.
    """

    camera = pure_hand_pose_view(
        pose,
        cast(PureHandPoseFrame, PURE_HAND_POSE_FRAME_CAMERA),
        cast(
            PureHandCoordinateFrame,
            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        ),
    )
    result = np.zeros((*camera.shape[:-2], 24, 3), dtype=np.float32)
    result[..., 2:3, :] = camera[..., :1, :]
    result[..., 3:, :] = camera[..., 1:, :]
    return result


def hybrid24_confidence_view(confidence: np.ndarray) -> np.ndarray:
    """Lay out confidence alongside :func:`hybrid24_pose_view`."""

    hand_confidence = pure_hand_confidence_view(confidence)
    result = np.zeros((*hand_confidence.shape[:-1], 24), dtype=np.float32)
    result[..., 2] = hand_confidence[..., 0]
    result[..., 3:] = hand_confidence[..., 1:]
    return result


def _hybrid24v2_slot_layout() -> dict[str, object]:
    return {
        "zero_slots": [0, 1],
        "absolute_wrist": {
            "output_joint_index": 2,
            "source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
            "coordinate_frame": PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        },
        "global_hand_geometry": {
            "output_joint_indices": [3, 23],
            "source_joint_indices": list(PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES),
            "root_source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
            "value": "absolute_wrist_plus_hybrid24_relative_hand_geometry",
        },
    }


def hybrid24v2_pose_view(pose: np.ndarray) -> np.ndarray:
    """Return the information-equivalent global-hand hybrid24 v2 view."""

    hybrid = hybrid24_pose_view(pose)
    result = np.zeros_like(hybrid)
    result[..., 2:3, :] = hybrid[..., 2:3, :]
    result[..., 3:, :] = hybrid[..., 2:3, :] + hybrid[..., 3:, :]
    return result


def hybrid24v2_confidence_view(confidence: np.ndarray) -> np.ndarray:
    """Keep the v1 hybrid confidence layout for the v2 coordinate view."""

    return hybrid24_confidence_view(confidence)


def _unisign_hand_bridge_slot_layout() -> dict[str, object]:
    return {
        "channels": ["x", "y", "score"],
        "zero_slots": [0, 1],
        "global_wrist": {
            "output_joint_index": 2,
            "source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
            "coordinate_transform": "body_proxy_crop_scale_xy",
            "score": UNISIGN_HAND_BRIDGE_SCORE,
        },
        "hand_wrist_root": {
            "output_joint_index": 3,
            "source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
            "value": [0.0, 0.0, UNISIGN_HAND_BRIDGE_SCORE],
        },
        "relative_fingers": {
            "output_joint_indices": [4, 23],
            "source_joint_indices": list(UNISIGN_HAND_BRIDGE_FINGER_SOURCE_JOINT_INDICES),
            "root_source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
            "coordinate_transform": "relative_xy_divided_by_body_proxy_scale",
            "score": UNISIGN_HAND_BRIDGE_SCORE,
        },
        "skipped_source_joint_indices": [3],
    }


def _unisign_hand_bridge_body_proxy() -> dict[str, object]:
    return {
        "source_joint_indices_per_hand": list(UNISIGN_HAND_BRIDGE_BODY_PROXY_SOURCE_JOINT_INDICES),
        "source_coordinate_frame": PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        "coordinate_axes": ["x", "y"],
        "scale": "max(xmax-xmin,ymax-ymin)",
        "center": "bbox_midpoint",
        "global_wrist_transform": "(xy-center)/(scale/2), clipped_to[-1,1]",
        "relative_hand_transform": "relative_xy/scale, clipped_to[-1,1]",
        "zero_scale_result": "all_zero_pose",
    }


def _unisign_hand_bridge_crop_parameters(
    radar_camera_pose: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Return Uni-Sign's body-proxy bbox center and scale for one clip."""

    if radar_camera_pose.ndim != 4 or radar_camera_pose.shape[1:] != (2, 24, 3):
        raise MeshFkL2DatasetError(
            "Uni-Sign hand bridge pose must have shape [time,2,24,3], got "
            f"{tuple(radar_camera_pose.shape)}"
        )
    anchors = radar_camera_pose[
        :, :, UNISIGN_HAND_BRIDGE_BODY_PROXY_SOURCE_JOINT_INDICES, :2
    ].reshape(-1, 2)
    if not bool(np.isfinite(anchors).all()):
        raise MeshFkL2DatasetError("Uni-Sign hand bridge body proxy must be finite")
    lower = anchors.min(axis=0)
    upper = anchors.max(axis=0)
    scale = float(np.max(upper - lower))
    if scale == 0.0:
        return None
    return np.asarray((lower + upper) / 2.0, dtype=np.float32), scale


def unisign_hand_bridge_pose_view(pose: np.ndarray) -> np.ndarray:
    """Project one selected clip into Uni-Sign's hand-input coordinate contract.

    This view deliberately accepts geometry only.  Its third channel is the
    fixed external pose-projection score, not a confidence value or gate.
    """

    if pose.ndim != 4 or pose.shape[1:] != (2, 24, 3):
        raise MeshFkL2DatasetError(
            "Uni-Sign hand bridge source pose must have shape [time,2,24,3], got "
            f"{tuple(pose.shape)}"
        )
    radar_camera_pose = pure_hand_source_to_radar_camera_pose(
        pose,
        cast(
            PureHandCoordinateFrame,
            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        ),
    )
    crop = _unisign_hand_bridge_crop_parameters(radar_camera_pose)
    result = np.zeros((*pose.shape[:-2], 24, 3), dtype=np.float32)
    if crop is None:
        return result
    center, scale = crop
    wrist = radar_camera_pose[..., PURE_HAND_ROOT_SOURCE_JOINT_INDEX, :2]
    result[..., 2, :2] = np.clip((wrist - center) * (2.0 / scale), -1.0, 1.0)
    result[..., 2, 2] = UNISIGN_HAND_BRIDGE_SCORE
    result[..., 3, 2] = UNISIGN_HAND_BRIDGE_SCORE
    fingers = radar_camera_pose[..., UNISIGN_HAND_BRIDGE_FINGER_SOURCE_JOINT_INDICES, :2]
    result[..., 4:, :2] = np.clip((fingers - wrist[..., np.newaxis, :]) / scale, -1.0, 1.0)
    result[..., 4:, 2] = UNISIGN_HAND_BRIDGE_SCORE
    return result


MESH_POSE_TRAIN_AUGMENTATION_PROFILE_ID = "mesh_pose_train_v1"

HYBRID24_POSTNORM_GAUSSIAN_NOISE_PROFILE_ID = "hybrid24_postnorm_gaussian_noise_v1"


@dataclass(frozen=True, slots=True)
class MeshPoseTrainAugmentationConfig:
    """Resolved train-only augmentation parameters for mesh-FK pose."""

    profile_id: str
    seed: int
    rotation_deg: float
    uniform_scale: tuple[float, float]
    speed: tuple[float, float]

    def __post_init__(self) -> None:
        if self.profile_id != MESH_POSE_TRAIN_AUGMENTATION_PROFILE_ID:
            raise MeshFkL2DatasetError(
                f"unsupported mesh pose augmentation profile {self.profile_id!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise MeshFkL2DatasetError("mesh pose augmentation seed must be non-negative")
        if (
            not isinstance(self.rotation_deg, (int, float))
            or isinstance(self.rotation_deg, bool)
            or not 0.0 <= float(self.rotation_deg) <= 15.0
        ):
            raise MeshFkL2DatasetError("mesh pose augmentation rotation_deg must be within [0, 15]")
        for name, bounds in (
            ("uniform_scale", self.uniform_scale),
            ("speed", self.speed),
        ):
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in bounds
                )
            ):
                raise MeshFkL2DatasetError(f"mesh pose augmentation {name} must be a numeric pair")
            lower, upper = float(bounds[0]), float(bounds[1])
            if lower <= 0.0 or lower > upper:
                raise MeshFkL2DatasetError(
                    f"mesh pose augmentation {name} must be ordered and positive"
                )
        if not 0.9 <= float(self.uniform_scale[0]) <= float(self.uniform_scale[1]) <= 1.1:
            raise MeshFkL2DatasetError(
                "mesh pose augmentation uniform_scale must stay within [0.9, 1.1]"
            )
        if not 0.9 <= float(self.speed[0]) <= float(self.speed[1]) <= 1.1:
            raise MeshFkL2DatasetError("mesh pose augmentation speed must stay within [0.9, 1.1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "seed": self.seed,
            "rotation_deg": float(self.rotation_deg),
            "uniform_scale": [float(value) for value in self.uniform_scale],
            "speed": [float(value) for value in self.speed],
        }


@dataclass(frozen=True, slots=True)
class Hybrid24PostnormGaussianNoiseConfig:
    """Resolved train-only Gaussian noise for the normalized hybrid24-v1 view."""

    profile_id: str
    seed: int
    coordinate_noise_std: float

    def __post_init__(self) -> None:
        if self.profile_id != HYBRID24_POSTNORM_GAUSSIAN_NOISE_PROFILE_ID:
            raise MeshFkL2DatasetError(
                f"unsupported hybrid24 Gaussian-noise profile {self.profile_id!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise MeshFkL2DatasetError("hybrid24 Gaussian-noise seed must be non-negative")
        if (
            not isinstance(self.coordinate_noise_std, (int, float))
            or isinstance(self.coordinate_noise_std, bool)
            or not np.isfinite(float(self.coordinate_noise_std))
            or float(self.coordinate_noise_std) <= 0.0
        ):
            raise MeshFkL2DatasetError(
                "hybrid24 Gaussian coordinate_noise_std must be finite and positive"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "seed": self.seed,
            "coordinate_noise_std": float(self.coordinate_noise_std),
        }


type WaveLLMTrainAugmentationConfig = (
    MeshPoseTrainAugmentationConfig | Hybrid24PostnormGaussianNoiseConfig
)


@dataclass(frozen=True, slots=True)
class MeshFkL2DatasetConfig:
    """Closed source configuration: one handoff root plus its L2 build."""

    handoff_root: Path
    l2_root: Path
    split: str
    part_cache_size: int = DEFAULT_PART_CACHE_SIZE
    # Sequence cap: longer episodes are uniformly
    # SAMPLED to max_frames frames via np.linspace(0, T-1, max_frames) —
    # the legacy ``load_and_normalize_pose`` semantics (whole-action
    # coverage, not a prefix crop); None keeps full episodes.
    max_frames: int | None = None
    # Pretrained token format: longer episodes use the pad_sequence rule —
    # random.sample per load — and every episode is zero-padded to a FIXED
    # max_frames with an all-true frame mask (no attention masking of pad
    # frames), matching the warm-start encoder's native distribution.
    legacy_sequence_format: bool = False
    train_augmentation: WaveLLMTrainAugmentationConfig | None = None
    # An explicit hand frame selects a 22-slot root-absolute plus 21-node
    # hand-relative view. Omission preserves the historical full-24 reader.
    pose_frame: PureHandPoseFrame | None = None
    pose_coordinate_frame: PureHandCoordinateFrame | None = None
    pose_normalization_profile: Path | None = None
    # A full-24 external role view must bring an independently fitted profile:
    # the L2-owned sidecar is fitted against the handoff's embedded split.
    full24_normalization_profile: Path | None = None
    # The warm-compatible hybrid24 view is a separate external role view.  It
    # has the legacy tensor shape but its own radar-camera sidecar contract.
    hybrid24_normalization_profile: Path | None = None
    # The global-hand hybrid24 v2 view shares the legacy tensor shape but has
    # its own coordinate contract and must never reuse a v1 sidecar.
    hybrid24v2_normalization_profile: Path | None = None
    # The Uni-Sign bridge is a clip-normalized 2D view. Its sidecar binds
    # that exact transform; it intentionally has no global affine statistics.
    unisign_hand_bridge_normalization_profile: Path | None = None
    # This control replays the historical contextual 10 Hz target grid before
    # any length cap; the selected profile is bound into its statistics.
    full24_temporal_sampling_profile: Full24TemporalSamplingProfile | None = None
    # Training-only random phase (0/1/2) for the strided
    # 10 Hz grid — each epoch each TRAIN episode draws one of the three
    # stride-3 phases (deterministic per sample+epoch via the trainer's
    # set_train_epoch hook), so the 2/3 of source frames the canonical
    # phase-0 grid never reads become training signal. Evaluation and
    # validation splits always keep the canonical phase-0 grid.
    full24_temporal_train_phase: Literal["random"] | None = None
    # The optional official semantic role view leaves L2 payloads untouched.
    split_assignments_path: Path | None = None

    def __post_init__(self) -> None:
        external_profile_count = sum(
            profile is not None
            for profile in (
                self.full24_normalization_profile,
                self.hybrid24_normalization_profile,
                self.hybrid24v2_normalization_profile,
                self.unisign_hand_bridge_normalization_profile,
            )
        )
        if external_profile_count > 1:
            raise MeshFkL2DatasetError(
                "full24_normalization_profile, hybrid24_normalization_profile, "
                "hybrid24v2_normalization_profile, and "
                "unisign_hand_bridge_normalization_profile are mutually exclusive"
            )
        if isinstance(self.train_augmentation, Hybrid24PostnormGaussianNoiseConfig) and (
            self.pose_frame is not None or self.hybrid24_normalization_profile is None
        ):
            raise MeshFkL2DatasetError(
                "hybrid24_postnorm_gaussian_noise_v1 requires the hybrid24-v1 view"
            )
        if isinstance(self.train_augmentation, MeshPoseTrainAugmentationConfig) and (
            self.hybrid24_normalization_profile is not None
            or self.hybrid24v2_normalization_profile is not None
            or self.unisign_hand_bridge_normalization_profile is not None
        ):
            raise MeshFkL2DatasetError(
                "mesh_pose_train_v1 cannot be used with hybrid24 or the Uni-Sign bridge"
            )
        if self.pose_frame is None:
            if self.pose_coordinate_frame is not None:
                raise MeshFkL2DatasetError(
                    "pose_coordinate_frame requires an explicit pure-hand pose_frame"
                )
            if self.pose_normalization_profile is not None:
                raise MeshFkL2DatasetError(
                    "pose_normalization_profile requires an explicit pure-hand pose_frame"
                )
            external_profile = (
                self.full24_normalization_profile
                or self.hybrid24_normalization_profile
                or self.hybrid24v2_normalization_profile
                or self.unisign_hand_bridge_normalization_profile
            )
            if external_profile is not None and self.split_assignments_path is None:
                raise MeshFkL2DatasetError(
                    "external 24-joint normalization profile requires split_assignments_path"
                )
            if self.split_assignments_path is not None and external_profile is None:
                raise MeshFkL2DatasetError(
                    "24-joint split_assignments_path requires an external normalization profile"
                )
            if self.full24_temporal_sampling_profile is not None:
                _require_full24_temporal_sampling_profile(self.full24_temporal_sampling_profile)
                if external_profile is None:
                    raise MeshFkL2DatasetError(
                        "full24_temporal_sampling_profile requires an external 24-joint "
                        "normalization profile"
                    )
                if isinstance(self.train_augmentation, MeshPoseTrainAugmentationConfig):
                    raise MeshFkL2DatasetError(
                        "full24_temporal_sampling_profile cannot be combined with "
                        "mesh_pose_train_v1"
                    )
            if (
                self.full24_temporal_train_phase == "random"
                and self.full24_temporal_sampling_profile is None
            ):
                raise MeshFkL2DatasetError(
                    "full24_temporal_train_phase=random requires a full-24 temporal "
                    "sampling profile"
                )
            return
        _require_pure_hand_pose_frame(self.pose_frame)
        if self.pose_coordinate_frame is None:
            raise MeshFkL2DatasetError(
                "explicit pure-hand pose_frame requires pose_coordinate_frame"
            )
        _require_pure_hand_coordinate_frame(self.pose_coordinate_frame)
        if self.pose_normalization_profile is None:
            raise MeshFkL2DatasetError(
                "explicit pure-hand pose_frame requires pose_normalization_profile"
            )
        if self.train_augmentation is not None:
            raise MeshFkL2DatasetError("train augmentation is unavailable to the pure-hand P1 view")
        if self.full24_normalization_profile is not None:
            raise MeshFkL2DatasetError(
                "full24_normalization_profile cannot be used with a pure-hand pose_frame"
            )
        if self.hybrid24_normalization_profile is not None:
            raise MeshFkL2DatasetError(
                "hybrid24_normalization_profile cannot be used with a pure-hand pose_frame"
            )
        if self.hybrid24v2_normalization_profile is not None:
            raise MeshFkL2DatasetError(
                "hybrid24v2_normalization_profile cannot be used with a pure-hand pose_frame"
            )
        if self.unisign_hand_bridge_normalization_profile is not None:
            raise MeshFkL2DatasetError(
                "unisign_hand_bridge_normalization_profile cannot be used with a "
                "pure-hand pose_frame"
            )
        if self.full24_temporal_sampling_profile is not None:
            raise MeshFkL2DatasetError(
                "full24_temporal_sampling_profile cannot be used with a pure-hand pose_frame"
            )


@dataclass(frozen=True, slots=True)
class _Member:
    sample_id: str
    part_path: str
    row_group: int  # episode ordinal inside its part (one row group per episode)
    frame_count: int


@dataclass(frozen=True, slots=True)
class MeshFkL2ExternalSplitView:
    """Membership-only role overlay for one handoff-bound L2 source."""

    path: Path
    sha256: str
    roles_by_sample: dict[str, str]
    split_counts: dict[str, int]
    unassigned_eligible_ids: tuple[str, ...]

    def checkpoint_identity(self) -> dict[str, object]:
        return {
            "kind": "external_split_assignments_jsonl_v1",
            "path": str(self.path),
            "sha256": self.sha256,
            "sample_count": sum(self.split_counts.values()),
            "split_counts": dict(self.split_counts),
            "unassigned_eligible_ids": list(self.unassigned_eligible_ids),
        }


def load_mesh_fk_external_split_view(
    split_assignments_path: str | Path,
    *,
    eligible_sample_ids: set[str],
) -> MeshFkL2ExternalSplitView:
    """Resolve an assignment-only semantic view over the accepted handoff.

    The assignment file may include source IDs not present in this stabilized
    handoff.  Those are irrelevant to the L2 reader; eligible IDs absent from
    the assignment are deliberately excluded rather than falling back to the
    handoff's embedded signer-disjoint role.
    """

    path = Path(split_assignments_path).expanduser().resolve()
    if not path.is_file():
        raise MeshFkL2DatasetError(f"external split assignment is missing: {path}")
    try:
        validate_split_assignments(path, allowed_splits=set(EXTERNAL_SPLIT_ROLES))
    except SplitContractError as error:
        raise MeshFkL2DatasetError(f"external split assignment is invalid: {error}") from error

    roles_by_sample: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                assignment = SplitAssignment.from_mapping(
                    json.loads(raw_line),
                    f"external split assignment line {line_number}",
                )
                if assignment.sample_id in eligible_sample_ids:
                    roles_by_sample[assignment.sample_id] = assignment.split
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SplitContractError) as error:
        raise MeshFkL2DatasetError(
            f"unable to read external split assignment {path}: {error}"
        ) from error

    split_counts = {
        role: sum(role_value == role for role_value in roles_by_sample.values())
        for role in sorted(EXTERNAL_SPLIT_ROLES)
    }
    if any(count < 1 for count in split_counts.values()):
        raise MeshFkL2DatasetError(
            "external split assignment has no eligible samples for one or more roles: "
            f"{split_counts}"
        )
    return MeshFkL2ExternalSplitView(
        path=path,
        sha256=_sha256_file(path),
        roles_by_sample=roles_by_sample,
        split_counts=split_counts,
        unassigned_eligible_ids=tuple(sorted(eligible_sample_ids - set(roles_by_sample))),
    )


def _require_pyarrow_parquet() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as error:  # pragma: no cover - environment dependent
        raise MeshFkL2DatasetError("the mesh-FK L2 reader requires pyarrow") from error
    return pa, pq


def load_mesh_fk_normalization_profile(l2_root: Path, handoff: dict[str, Any]) -> dict[str, Any]:
    """Load and bind-check the train-only normalization sidecar.

    The sidecar's sha256 must equal the hash recorded in BOTH the L2
    ``schema.json`` and the handoff; anything else is drift.
    """

    sidecar_path = l2_root / "normalization_stats.json"
    if not sidecar_path.is_file():
        raise MeshFkL2DatasetError(f"missing normalization sidecar {sidecar_path}")
    sidecar_sha = _sha256_file(sidecar_path)
    schema_doc = json.loads((l2_root / "schema.json").read_text())
    if schema_doc.get("normalization_sidecar_sha256") != sidecar_sha:
        raise MeshFkL2DatasetError("normalization sidecar drifted from schema.json")
    bound = handoff.get("normalization_binding", {})
    if bound.get("sha256") != sidecar_sha:
        raise MeshFkL2DatasetError("normalization sidecar drifted from the handoff")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("protocol_id") != MESH_NORMALIZATION_PROTOCOL_ID:
        raise MeshFkL2DatasetError(
            f"unsupported normalization protocol {sidecar.get('protocol_id')!r}"
        )
    if len(sidecar.get("mean", [])) != POSE_FLAT or len(sidecar.get("std", [])) != POSE_FLAT:
        raise MeshFkL2DatasetError("normalization statistics have the wrong shape")
    if any(value <= 0 for value in sidecar["std"]):
        raise MeshFkL2DatasetError("normalization std must be strictly positive")
    profile: dict[str, Any] = sidecar
    return profile


def _campaign_normalization_source_identity(source_identity: dict[str, str]) -> dict[str, str]:
    """Keep the L2 fields that bind a campaign-owned sidecar to one source."""

    return {
        key: source_identity[key]
        for key in (
            "handoff_schema",
            "pose_source",
            "l2_build_id",
            "l2_schema_version",
            "index_sha256",
            "member_digest",
        )
    }


def _read_handoff_members(handoff_root: Path, handoff: dict[str, Any]) -> list[dict[str, Any]]:
    member_path = handoff_root / "eligible_episodes.jsonl"
    members = [json.loads(line) for line in member_path.read_text().splitlines() if line.strip()]
    if not members:
        raise MeshFkL2DatasetError("handoff member file is empty")
    sample_ids = [str(row.get("sample_id", "")) for row in members]
    if any(not sample_id for sample_id in sample_ids):
        raise MeshFkL2DatasetError("handoff member is missing sample_id")
    if _member_digest(sorted(sample_ids)) != handoff.get("member_digest"):
        raise MeshFkL2DatasetError("handoff member digest mismatch")
    return members


def load_pure_hand_normalization_profile(
    *,
    handoff_root: Path,
    l2_root: Path,
    pose_frame: PureHandPoseFrame,
    pose_coordinate_frame: PureHandCoordinateFrame,
    profile_path: Path,
    external_split_view: MeshFkL2ExternalSplitView | None = None,
) -> dict[str, Any]:
    """Load a mode-bound pure-hand sidecar and reject any cross-mode reuse."""

    frame = _require_pure_hand_pose_frame(pose_frame)
    coordinate_frame = _require_pure_hand_coordinate_frame(pose_coordinate_frame)
    path = profile_path.expanduser().resolve()
    if not path.is_file():
        raise MeshFkL2DatasetError(f"missing pure-hand normalization sidecar {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MeshFkL2DatasetError(
            f"unreadable pure-hand normalization sidecar: {error}"
        ) from error
    if payload.get("schema_version") != PURE_HAND_NORMALIZATION_SCHEMA:
        raise MeshFkL2DatasetError("unsupported pure-hand normalization schema")
    if payload.get("protocol_id") != PURE_HAND_NORMALIZATION_PROTOCOL:
        raise MeshFkL2DatasetError("unsupported pure-hand normalization protocol")
    if payload.get("pose_view") != PURE_HAND_POSE_VIEW:
        raise MeshFkL2DatasetError("unsupported pure-hand pose-view contract")
    if payload.get("pose_frame") != frame:
        raise MeshFkL2DatasetError(
            "pure-hand normalization sidecar pose_frame does not match the dataset"
        )
    if payload.get("coordinate_frame") != coordinate_frame:
        raise MeshFkL2DatasetError(
            "pure-hand normalization sidecar coordinate_frame does not match the dataset"
        )
    if payload.get("coordinate_transform") != pure_hand_coordinate_transform(coordinate_frame):
        raise MeshFkL2DatasetError(
            "pure-hand normalization sidecar coordinate transform does not match the dataset"
        )
    if payload.get("root_slot") != _pure_hand_root_slot(frame):
        raise MeshFkL2DatasetError(
            "pure-hand normalization sidecar root-slot convention does not match the dataset frame"
        )
    if payload.get("geometry_frame") != {
        "kind": "relative_to_radar_camera_wrist",
        "source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
    }:
        raise MeshFkL2DatasetError(
            "pure-hand normalization sidecar geometry frame does not match the dataset"
        )
    if payload.get("source_joint_indices") != list(PURE_HAND_SOURCE_JOINT_INDICES):
        raise MeshFkL2DatasetError("pure-hand normalization sidecar joint layout mismatch")
    if payload.get("root_source_joint_index") != PURE_HAND_ROOT_SOURCE_JOINT_INDEX:
        raise MeshFkL2DatasetError("pure-hand normalization sidecar root layout mismatch")
    if payload.get("geometry_source_joint_indices") != list(
        PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES
    ):
        raise MeshFkL2DatasetError("pure-hand normalization sidecar geometry layout mismatch")
    if payload.get("palm_included") is not True:
        raise MeshFkL2DatasetError("pure-hand normalization sidecar must include the palm")
    if payload.get("pose_shape") != [2, PURE_HAND_JOINT_COUNT, 3]:
        raise MeshFkL2DatasetError("pure-hand normalization sidecar pose shape mismatch")
    if payload.get("geometry_shape") != [2, PURE_HAND_GEOMETRY_JOINT_COUNT, 3]:
        raise MeshFkL2DatasetError("pure-hand normalization sidecar geometry shape mismatch")
    if payload.get("pose_flat_values") != PURE_HAND_POSE_FLAT:
        raise MeshFkL2DatasetError("pure-hand normalization sidecar flat shape mismatch")
    expected_identity = _campaign_normalization_source_identity(
        mesh_fk_source_identity(handoff_root, l2_root)
    )
    if payload.get("source_identity") != expected_identity:
        raise MeshFkL2DatasetError("pure-hand normalization sidecar source identity drift")
    expected_external_view = (
        external_split_view.checkpoint_identity() if external_split_view is not None else None
    )
    if payload.get("external_split_view") != expected_external_view:
        raise MeshFkL2DatasetError(
            "pure-hand normalization sidecar external membership binding mismatch"
        )
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise MeshFkL2DatasetError("pure-hand normalization statistics are missing")
    if len(mean) != PURE_HAND_POSE_FLAT or len(std) != PURE_HAND_POSE_FLAT:
        raise MeshFkL2DatasetError("pure-hand normalization statistics have the wrong shape")
    if any(not isinstance(value, (int, float)) for value in mean):
        raise MeshFkL2DatasetError("pure-hand normalization mean must be numeric")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in std):
        raise MeshFkL2DatasetError("pure-hand normalization std must be positive")
    return dict(payload)


def load_full24_external_normalization_profile(
    *,
    handoff_root: Path,
    l2_root: Path,
    profile_path: Path,
    external_split_view: MeshFkL2ExternalSplitView,
    temporal_sampling_profile: Full24TemporalSamplingProfile | None = None,
) -> dict[str, Any]:
    """Load an official-view full-24 sidecar and reject membership drift."""

    path = profile_path.expanduser().resolve()
    if not path.is_file():
        raise MeshFkL2DatasetError(f"missing full-24 normalization sidecar {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MeshFkL2DatasetError(f"unreadable full-24 normalization sidecar: {error}") from error
    if payload.get("schema_version") != FULL24_EXTERNAL_NORMALIZATION_SCHEMA:
        raise MeshFkL2DatasetError("unsupported full-24 normalization schema")
    if payload.get("protocol_id") != FULL24_EXTERNAL_NORMALIZATION_PROTOCOL:
        raise MeshFkL2DatasetError("unsupported full-24 normalization protocol")
    if payload.get("pose_view") != FULL24_POSE_VIEW:
        raise MeshFkL2DatasetError("unsupported full-24 pose-view contract")
    if payload.get("temporal_sampling") != full24_temporal_sampling_identity(
        temporal_sampling_profile
    ):
        raise MeshFkL2DatasetError("full-24 normalization sidecar temporal sampling mismatch")
    if payload.get("pose_shape") != [2, 24, 3] or payload.get("pose_flat_values") != POSE_FLAT:
        raise MeshFkL2DatasetError("full-24 normalization sidecar pose shape mismatch")
    if payload.get("root_center") != {
        "kind": "per_frame_per_person_joint",
        "joint_index": ROOT_JOINT_INDEX,
    }:
        raise MeshFkL2DatasetError("full-24 normalization sidecar root-center mismatch")
    source = mesh_fk_source_identity(handoff_root, l2_root)
    expected_identity = _campaign_normalization_source_identity(source)
    if payload.get("source_identity") != expected_identity:
        raise MeshFkL2DatasetError("full-24 normalization sidecar source identity drift")
    if payload.get("raw_coordinate_frame") != source["coordinate_frame"]:
        raise MeshFkL2DatasetError("full-24 normalization sidecar coordinate frame mismatch")
    if payload.get("normalized_pose_units") != NORMALIZED_POSE_UNITS:
        raise MeshFkL2DatasetError("full-24 normalization sidecar pose units mismatch")
    if payload.get("split") != "train":
        raise MeshFkL2DatasetError("full-24 normalization sidecar must be fit on train")
    if payload.get("external_split_view") != external_split_view.checkpoint_identity():
        raise MeshFkL2DatasetError(
            "full-24 normalization sidecar external membership binding mismatch"
        )
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise MeshFkL2DatasetError("full-24 normalization statistics are missing")
    if len(mean) != POSE_FLAT or len(std) != POSE_FLAT:
        raise MeshFkL2DatasetError("full-24 normalization statistics have the wrong shape")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in mean):
        raise MeshFkL2DatasetError("full-24 normalization mean must be finite numeric")
    if any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in std
    ):
        raise MeshFkL2DatasetError("full-24 normalization std must be finite and positive")
    return dict(payload)


def load_hybrid24_external_normalization_profile(
    *,
    handoff_root: Path,
    l2_root: Path,
    profile_path: Path,
    external_split_view: MeshFkL2ExternalSplitView,
    temporal_sampling_profile: Full24TemporalSamplingProfile | None = None,
) -> dict[str, Any]:
    """Load a hybrid24 sidecar and reject any view or membership drift."""

    path = profile_path.expanduser().resolve()
    if not path.is_file():
        raise MeshFkL2DatasetError(f"missing hybrid24 normalization sidecar {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MeshFkL2DatasetError(f"unreadable hybrid24 normalization sidecar: {error}") from error
    if payload.get("schema_version") != HYBRID24_EXTERNAL_NORMALIZATION_SCHEMA:
        raise MeshFkL2DatasetError("unsupported hybrid24 normalization schema")
    if payload.get("protocol_id") != HYBRID24_EXTERNAL_NORMALIZATION_PROTOCOL:
        raise MeshFkL2DatasetError("unsupported hybrid24 normalization protocol")
    if payload.get("pose_view") != HYBRID24_POSE_VIEW:
        raise MeshFkL2DatasetError("unsupported hybrid24 pose-view contract")
    if payload.get("slot_layout") != _hybrid24_slot_layout():
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar slot-layout mismatch")
    if payload.get("coordinate_frame") != PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08:
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar coordinate frame mismatch")
    if payload.get("coordinate_transform") != pure_hand_coordinate_transform(
        cast(
            PureHandCoordinateFrame,
            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        )
    ):
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar coordinate transform mismatch")
    if payload.get("temporal_sampling") != full24_temporal_sampling_identity(
        temporal_sampling_profile
    ):
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar temporal sampling mismatch")
    if payload.get("pose_shape") != [2, 24, 3] or payload.get("pose_flat_values") != POSE_FLAT:
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar pose shape mismatch")
    if payload.get("root_center") != {
        "kind": "per_frame_per_person_joint",
        "joint_index": ROOT_JOINT_INDEX,
    }:
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar root-center mismatch")
    source = mesh_fk_source_identity(handoff_root, l2_root)
    expected_identity = _campaign_normalization_source_identity(source)
    if payload.get("source_identity") != expected_identity:
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar source identity drift")
    if payload.get("normalized_pose_units") != HYBRID24_NORMALIZED_POSE_UNITS:
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar pose units mismatch")
    if payload.get("split") != "train":
        raise MeshFkL2DatasetError("hybrid24 normalization sidecar must be fit on train")
    if payload.get("external_split_view") != external_split_view.checkpoint_identity():
        raise MeshFkL2DatasetError(
            "hybrid24 normalization sidecar external membership binding mismatch"
        )
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise MeshFkL2DatasetError("hybrid24 normalization statistics are missing")
    if len(mean) != POSE_FLAT or len(std) != POSE_FLAT:
        raise MeshFkL2DatasetError("hybrid24 normalization statistics have the wrong shape")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in mean):
        raise MeshFkL2DatasetError("hybrid24 normalization mean must be finite numeric")
    if any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in std
    ):
        raise MeshFkL2DatasetError("hybrid24 normalization std must be finite and positive")
    zero_variance_axes = payload.get("zero_variance_axes")
    if (
        not isinstance(zero_variance_axes, list)
        or any(not isinstance(axis, int) or isinstance(axis, bool) for axis in zero_variance_axes)
        or len(set(zero_variance_axes)) != len(zero_variance_axes)
    ):
        raise MeshFkL2DatasetError("hybrid24 normalization zero-variance axes are invalid")
    if not set(HYBRID24_ZERO_VARIANCE_AXES).issubset(zero_variance_axes):
        raise MeshFkL2DatasetError(
            "hybrid24 normalization sidecar omits required zero-variance arm axes"
        )
    if any(mean[axis] != 0.0 or std[axis] != 1.0 for axis in HYBRID24_ZERO_VARIANCE_AXES):
        raise MeshFkL2DatasetError(
            "hybrid24 normalization sidecar zero-variance arm axes lack the affine floor"
        )
    return dict(payload)


def apply_pure_hand_normalization(pose: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Apply only the profile's global affine transform to a formed hands22 view."""

    if pose.ndim < 4 or pose.shape[-3:] != (2, PURE_HAND_JOINT_COUNT, 3):
        raise MeshFkL2DatasetError(f"pure-hand pose must end in [2,22,3], got {tuple(pose.shape)}")
    flat = pose.reshape(-1, PURE_HAND_POSE_FLAT).astype(np.float32)
    mean = np.asarray(profile["mean"], dtype=np.float32)
    std = np.asarray(profile["std"], dtype=np.float32)
    return ((flat - mean) / std).reshape(pose.shape).astype(np.float32)


def apply_hybrid24_normalization(pose: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Apply the hybrid sidecar after asserting its no-arm input contract."""

    if pose.ndim < 4 or pose.shape[-3:] != (2, 24, 3):
        raise MeshFkL2DatasetError(f"hybrid24 pose must end in [2,24,3], got {tuple(pose.shape)}")
    if not bool(np.array_equal(pose[..., :2, :], np.zeros_like(pose[..., :2, :]))):
        raise MeshFkL2DatasetError("hybrid24 pose slots 0 and 1 must be exact zero")
    return apply_mesh_fk_normalization(pose, profile)


def apply_hybrid24_postnorm_gaussian_noise(
    normalized_pose: np.ndarray,
    profile: Hybrid24PostnormGaussianNoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent normalized-space noise without perturbing structural arm slots."""

    if normalized_pose.ndim < 4 or normalized_pose.shape[-3:] != (2, 24, 3):
        raise MeshFkL2DatasetError(
            f"normalized hybrid24 pose must end in [2,24,3], got {tuple(normalized_pose.shape)}"
        )
    if not bool(
        np.array_equal(
            normalized_pose[..., :2, :],
            np.zeros_like(normalized_pose[..., :2, :]),
        )
    ):
        raise MeshFkL2DatasetError(
            "hybrid24 Gaussian noise requires exact-zero shoulder/elbow slots"
        )
    result = np.array(normalized_pose, dtype=np.float32, copy=True)
    result[..., 2:, :] += rng.normal(
        loc=0.0,
        scale=float(profile.coordinate_noise_std),
        size=result[..., 2:, :].shape,
    ).astype(np.float32)
    return result


def load_hybrid24v2_external_normalization_profile(
    *,
    handoff_root: Path,
    l2_root: Path,
    profile_path: Path,
    external_split_view: MeshFkL2ExternalSplitView,
    temporal_sampling_profile: Full24TemporalSamplingProfile | None = None,
) -> dict[str, Any]:
    """Load a v2 sidecar and reject any view or membership drift."""

    path = profile_path.expanduser().resolve()
    if not path.is_file():
        raise MeshFkL2DatasetError(f"missing hybrid24v2 normalization sidecar {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MeshFkL2DatasetError(
            f"unreadable hybrid24v2 normalization sidecar: {error}"
        ) from error
    if payload.get("schema_version") != HYBRID24_V2_EXTERNAL_NORMALIZATION_SCHEMA:
        raise MeshFkL2DatasetError("unsupported hybrid24v2 normalization schema")
    if payload.get("protocol_id") != HYBRID24_V2_EXTERNAL_NORMALIZATION_PROTOCOL:
        raise MeshFkL2DatasetError("unsupported hybrid24v2 normalization protocol")
    if payload.get("pose_view") != HYBRID24_V2_POSE_VIEW:
        raise MeshFkL2DatasetError("unsupported hybrid24v2 pose-view contract")
    if payload.get("slot_layout") != _hybrid24v2_slot_layout():
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar slot-layout mismatch")
    if payload.get("coordinate_frame") != PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08:
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar coordinate frame mismatch")
    if payload.get("coordinate_transform") != pure_hand_coordinate_transform(
        cast(
            PureHandCoordinateFrame,
            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        )
    ):
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar coordinate transform mismatch")
    if payload.get("temporal_sampling") != full24_temporal_sampling_identity(
        temporal_sampling_profile
    ):
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar temporal sampling mismatch")
    if payload.get("pose_shape") != [2, 24, 3] or payload.get("pose_flat_values") != POSE_FLAT:
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar pose shape mismatch")
    if payload.get("root_center") != {
        "kind": "per_frame_per_person_joint",
        "joint_index": ROOT_JOINT_INDEX,
    }:
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar root-center mismatch")
    source = mesh_fk_source_identity(handoff_root, l2_root)
    expected_identity = _campaign_normalization_source_identity(source)
    if payload.get("source_identity") != expected_identity:
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar source identity drift")
    if payload.get("normalized_pose_units") != HYBRID24_V2_NORMALIZED_POSE_UNITS:
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar pose units mismatch")
    if payload.get("split") != "train":
        raise MeshFkL2DatasetError("hybrid24v2 normalization sidecar must be fit on train")
    if payload.get("external_split_view") != external_split_view.checkpoint_identity():
        raise MeshFkL2DatasetError(
            "hybrid24v2 normalization sidecar external membership binding mismatch"
        )
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise MeshFkL2DatasetError("hybrid24v2 normalization statistics are missing")
    if len(mean) != POSE_FLAT or len(std) != POSE_FLAT:
        raise MeshFkL2DatasetError("hybrid24v2 normalization statistics have the wrong shape")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in mean):
        raise MeshFkL2DatasetError("hybrid24v2 normalization mean must be finite numeric")
    if any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in std
    ):
        raise MeshFkL2DatasetError("hybrid24v2 normalization std must be finite and positive")
    zero_variance_axes = payload.get("zero_variance_axes")
    if (
        not isinstance(zero_variance_axes, list)
        or any(not isinstance(axis, int) or isinstance(axis, bool) for axis in zero_variance_axes)
        or len(set(zero_variance_axes)) != len(zero_variance_axes)
    ):
        raise MeshFkL2DatasetError("hybrid24v2 normalization zero-variance axes are invalid")
    if not set(HYBRID24_V2_ZERO_VARIANCE_AXES).issubset(zero_variance_axes):
        raise MeshFkL2DatasetError(
            "hybrid24v2 normalization sidecar omits required zero-variance arm axes"
        )
    if any(mean[axis] != 0.0 or std[axis] != 1.0 for axis in HYBRID24_V2_ZERO_VARIANCE_AXES):
        raise MeshFkL2DatasetError(
            "hybrid24v2 normalization sidecar zero-variance arm axes lack the affine floor"
        )
    return dict(payload)


def apply_hybrid24v2_normalization(pose: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Apply the v2 sidecar after asserting its no-arm input contract."""

    if pose.ndim < 4 or pose.shape[-3:] != (2, 24, 3):
        raise MeshFkL2DatasetError(f"hybrid24v2 pose must end in [2,24,3], got {tuple(pose.shape)}")
    if not bool(np.array_equal(pose[..., :2, :], np.zeros_like(pose[..., :2, :]))):
        raise MeshFkL2DatasetError("hybrid24v2 pose slots 0 and 1 must be exact zero")
    return apply_mesh_fk_normalization(pose, profile)


def load_unisign_hand_bridge_sidecar(
    *,
    handoff_root: Path,
    l2_root: Path,
    profile_path: Path,
    external_split_view: MeshFkL2ExternalSplitView,
    temporal_sampling_profile: Full24TemporalSamplingProfile | None = None,
) -> dict[str, Any]:
    """Load D1's sidecar and reject any bridge-view or split drift."""

    path = profile_path.expanduser().resolve()
    if not path.is_file():
        raise MeshFkL2DatasetError(f"missing Uni-Sign hand bridge sidecar {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MeshFkL2DatasetError(f"unreadable Uni-Sign hand bridge sidecar: {error}") from error
    if payload.get("schema_version") != UNISIGN_HAND_BRIDGE_SIDECAR_SCHEMA:
        raise MeshFkL2DatasetError("unsupported Uni-Sign hand bridge sidecar schema")
    if payload.get("protocol_id") != UNISIGN_HAND_BRIDGE_PROTOCOL:
        raise MeshFkL2DatasetError("unsupported Uni-Sign hand bridge sidecar protocol")
    if payload.get("pose_view") != UNISIGN_HAND_BRIDGE_POSE_VIEW:
        raise MeshFkL2DatasetError("unsupported Uni-Sign hand bridge pose-view contract")
    if payload.get("slot_layout") != _unisign_hand_bridge_slot_layout():
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar slot-layout mismatch")
    if payload.get("body_proxy") != _unisign_hand_bridge_body_proxy():
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar body-proxy mismatch")
    if payload.get("coordinate_frame") != PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08:
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar coordinate frame mismatch")
    if payload.get("coordinate_transform") != pure_hand_coordinate_transform(
        cast(
            PureHandCoordinateFrame,
            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
        )
    ):
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar coordinate transform mismatch")
    if temporal_sampling_profile != FULL24_TEMPORAL_SAMPLING_LEGACY_CONTEXTUAL_10HZ:
        raise MeshFkL2DatasetError(
            "Uni-Sign hand bridge requires legacy_contextual_10hz_stride3_successor_v1"
        )
    if payload.get("temporal_sampling") != full24_temporal_sampling_identity(
        temporal_sampling_profile
    ):
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar temporal sampling mismatch")
    if payload.get("pose_shape") != [2, 24, 3] or payload.get("pose_flat_values") != POSE_FLAT:
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar pose shape mismatch")
    if payload.get("legacy_root_centering") != {
        "input_root_slots": [0, 1],
        "effect": "no_op_when_zero",
    }:
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar root-centering mismatch")
    if payload.get("fixed_score") != UNISIGN_HAND_BRIDGE_SCORE:
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar fixed-score mismatch")
    if payload.get("global_affine") != "none":
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar must not declare a global affine")
    source = mesh_fk_source_identity(handoff_root, l2_root)
    expected_identity = _campaign_normalization_source_identity(source)
    if payload.get("source_identity") != expected_identity:
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar source identity drift")
    if payload.get("normalized_pose_units") != UNISIGN_HAND_BRIDGE_POSE_UNITS:
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar pose units mismatch")
    if payload.get("split") != "train":
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar must be fit on train")
    if payload.get("external_split_view") != external_split_view.checkpoint_identity():
        raise MeshFkL2DatasetError(
            "Uni-Sign hand bridge sidecar external membership binding mismatch"
        )
    for key in (
        "episode_count",
        "frame_count",
        "zero_scale_episode_count",
        "zero_scale_frame_count",
    ):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise MeshFkL2DatasetError(f"Uni-Sign hand bridge sidecar {key} is invalid")
    if payload["zero_scale_episode_count"] > payload["episode_count"]:
        raise MeshFkL2DatasetError(
            "Uni-Sign hand bridge sidecar zero-scale episode count is invalid"
        )
    if payload["zero_scale_frame_count"] > payload["frame_count"]:
        raise MeshFkL2DatasetError("Uni-Sign hand bridge sidecar zero-scale frame count is invalid")
    return dict(payload)


def _normalization_root_joint(profile: dict[str, Any]) -> int:
    root = profile.get("root_center", {})
    if root.get("kind") != "per_frame_per_person_joint":
        raise MeshFkL2DatasetError(f"unsupported root-center convention {root.get('kind')!r}")
    return int(root.get("joint_index", ROOT_JOINT_INDEX))


def root_center_mesh_fk_pose(pose: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Apply the bound geometry-space root center without standardizing."""

    joint = _normalization_root_joint(profile)
    flat = pose.reshape(-1, POSE_FLAT).astype(np.float32)
    shaped = flat.reshape(-1, 2, 24, 3)
    return shaped - shaped[:, :, joint : joint + 1, :]


def standardize_mesh_fk_pose(root_centered_pose: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Apply the bound per-coordinate mean/std to root-centered pose."""

    flat = root_centered_pose.reshape(-1, POSE_FLAT).astype(np.float32)
    mean = np.asarray(profile["mean"], dtype=np.float32)
    std = np.asarray(profile["std"], dtype=np.float32)
    return ((flat - mean) / std).reshape(root_centered_pose.shape).astype(np.float32)


def apply_mesh_fk_normalization(pose: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Apply the bound train-only profile exactly once.

    Reference expression (proved by regression test):
    ``((pose - pose[:, :, joint0:joint0+1, :]) - mean) / std`` on the
    flat ``[T, 144]`` view of a ``[T, 2, 24, 3]`` episode.
    """

    return standardize_mesh_fk_pose(root_center_mesh_fk_pose(pose, profile), profile)


def full_action_temporal_indices(
    source_frames: int,
    speed: float,
    max_frames: int | None,
) -> np.ndarray:
    """Select an endpoint-covering, monotonic index grid for one episode."""

    if source_frames < 1:
        raise MeshFkL2DatasetError("temporal augmentation requires at least one source frame")
    if not isinstance(speed, (int, float)) or isinstance(speed, bool) or speed <= 0:
        raise MeshFkL2DatasetError("temporal augmentation speed must be positive")
    output_frames = max(1, round(source_frames / float(speed)))
    if max_frames is not None:
        if max_frames < 1:
            raise MeshFkL2DatasetError("temporal augmentation max_frames must be positive")
        output_frames = min(output_frames, max_frames)
    return np.linspace(0, source_frames - 1, output_frames, dtype=np.int64)


def _axis_angle_rotation_matrix(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(unit_axis))
    if norm == 0.0:
        raise MeshFkL2DatasetError("rotation axis must be non-zero")
    x, y, z = unit_axis / norm
    skew = np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float32,
    )
    identity = np.eye(3, dtype=np.float32)
    return (
        identity
        + np.float32(np.sin(angle_radians)) * skew
        + np.float32(1.0 - np.cos(angle_radians)) * (skew @ skew)
    )


def apply_mesh_pose_spatial_transform(
    root_centered_pose: np.ndarray,
    rotation: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Apply one rotation and isotropic scale to all episode frames."""

    if rotation.shape != (3, 3):
        raise MeshFkL2DatasetError("mesh pose rotation must have shape (3, 3)")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise MeshFkL2DatasetError("mesh pose scale must be positive")
    transformed = np.matmul(root_centered_pose, rotation.T)
    return np.asarray(transformed * np.float32(scale), dtype=np.float32)


def sample_mesh_pose_train_parameters(
    profile: MeshPoseTrainAugmentationConfig,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, float]:
    """Draw speed, rotation, and scale for one train episode."""

    speed = float(rng.uniform(*profile.speed))
    scale = float(rng.uniform(*profile.uniform_scale))
    axis = rng.normal(size=3).astype(np.float32)
    angle = float(
        np.deg2rad(rng.uniform(-float(profile.rotation_deg), float(profile.rotation_deg)))
    )
    return speed, _axis_angle_rotation_matrix(axis, angle), scale


def validate_decoded_episode(
    sample_id: str,
    frame_index: np.ndarray,
    pose: np.ndarray,
    confidence: np.ndarray,
    frame_mask: np.ndarray,
    captions: set[str],
    expected_frames: int,
    *,
    require_all_true_mask: bool = True,
    require_first_frame_zero: bool = True,
) -> None:
    """Validate one decoded episode before it becomes a sample."""

    if pose.shape != (expected_frames, 2, 24, 3):
        raise MeshFkL2DatasetError(
            f"{sample_id}: pose shape {pose.shape} != ({expected_frames}, 2, 24, 3)"
        )
    if confidence.shape != (expected_frames, 2, 24):
        raise MeshFkL2DatasetError(f"{sample_id}: confidence shape mismatch")
    if frame_mask.shape != (expected_frames,):
        raise MeshFkL2DatasetError(f"{sample_id}: frame_mask shape mismatch")
    if (
        frame_index.shape[0] != expected_frames
        or (require_first_frame_zero and int(frame_index[0]) != 0)
        or not bool(np.all(np.diff(frame_index) > 0))
    ):
        raise MeshFkL2DatasetError(
            f"{sample_id}: frame indices are not strictly increasing from "
            f"zero (duplicate/reordered frames)"
        )
    # Unsampled rows are the special case where the strictly-increasing
    # integers from zero with count == last+1 ARE contiguous arange; sampled
    # rows keep their original (gapped) source indices.
    if not bool(np.isfinite(pose).all()):
        raise MeshFkL2DatasetError(f"{sample_id}: non-finite pose values")
    if bool(confidence.min() < 0.0) or bool(confidence.max() > 1.0):
        raise MeshFkL2DatasetError(f"{sample_id}: confidence outside [0, 1]")
    if not bool(frame_mask.any()):
        raise MeshFkL2DatasetError(f"{sample_id}: frame_mask has no eligible frame")
    if require_all_true_mask and not bool(frame_mask.all()):
        raise MeshFkL2DatasetError(
            f"{sample_id}: mesh L2 frame_mask must be all true (tri-state carries absence)"
        )
    if len(captions) != 1:
        raise MeshFkL2DatasetError(f"{sample_id}: caption drift inside the episode")


def mesh_fk_source_identity(handoff_root: str | Path, l2_root: str | Path) -> dict[str, str]:
    """Canonical, split-agnostic run/checkpoint identity for one L2 source.

    Training and evaluation MUST derive their binding from this function so
    both sides produce byte-identical identity dicts (the checkpoint binding
    comparison is exact).  It re-validates the handoff->build binding on
    every call.
    """

    handoff = json.loads((Path(handoff_root) / "handoff.json").read_text())
    if handoff.get("schema_version") not in ALLOWED_HANDOFF_SCHEMA_NAMES:
        raise MeshFkL2DatasetError("handoff schema unsupported")
    if handoff.get("pose_source") not in ALLOWED_POSE_SOURCES:
        raise MeshFkL2DatasetError("handoff pose lineage unsupported")
    root = Path(l2_root)
    delivery = json.loads((root / "delivery.json").read_text())
    identities = handoff.get("source_identities", {})
    if delivery.get("build_id") != identities.get("l2_build_id"):
        raise MeshFkL2DatasetError("L2 build id drifted from the handoff")
    if _sha256_file(root / "episode_index.parquet") != identities.get("index_sha256"):
        raise MeshFkL2DatasetError("episode index drifted from the handoff")
    return {
        "backend": "mesh_fk_l2_handoff",
        "handoff_schema": str(handoff.get("schema_version", HANDOFF_SCHEMA_NAME)),
        "pose_source": str(handoff.get("pose_source", "")),
        "l2_build_id": str(identities.get("l2_build_id", "")),
        "l2_schema_version": str(delivery.get("schema_version", "")),
        "index_sha256": str(identities.get("index_sha256", "")),
        "member_digest": str(handoff.get("member_digest", "")),
        "normalization_sha256": str(handoff.get("normalization_binding", {}).get("sha256", "")),
        "normalization_protocol": str(
            handoff.get("normalization_binding", {}).get("protocol_id", "")
        ),
        "pose_units": NORMALIZED_POSE_UNITS,
        "coordinate_frame": str(delivery.get("protocols", {}).get("coordinate_frame", "")),
    }


class MeshFkL2EpisodeDataset:
    """Map-style dataset over one split of one validated mesh-FK handoff."""

    def __init__(self, config: MeshFkL2DatasetConfig) -> None:
        _, pq = _require_pyarrow_parquet()
        self._pq = pq
        handoff_root = Path(config.handoff_root).expanduser().resolve()
        handoff_path = handoff_root / "handoff.json"
        for required in ("handoff.json", "eligible_episodes.jsonl"):
            if not (handoff_root / required).is_file():
                raise MeshFkL2DatasetError(f"handoff root missing {required}")
        handoff = json.loads(handoff_path.read_text())
        if handoff.get("schema_version") not in ALLOWED_HANDOFF_SCHEMA_NAMES:
            raise MeshFkL2DatasetError(
                f"handoff schema {handoff.get('schema_version')!r} unsupported"
            )
        if handoff.get("pose_source") not in ALLOWED_POSE_SOURCES:
            raise MeshFkL2DatasetError(
                f"handoff pose lineage {handoff.get('pose_source')!r} is unsupported; "
                f"expected one of {list(ALLOWED_POSE_SOURCES)}"
            )

        l2_root = Path(config.l2_root).expanduser().resolve()
        delivery = json.loads((l2_root / "delivery.json").read_text())
        if delivery.get("schema_version") not in ALLOWED_L2_SCHEMA_NAMES:
            raise MeshFkL2DatasetError("L2 delivery declares a different product")
        identities = handoff.get("source_identities", {})
        if delivery.get("build_id") != identities.get("l2_build_id"):
            raise MeshFkL2DatasetError("L2 build id drifted from the handoff")
        if _sha256_file(l2_root / "episode_index.parquet") != identities.get("index_sha256"):
            raise MeshFkL2DatasetError("episode index drifted from the handoff")

        members_raw = _read_handoff_members(handoff_root, handoff)
        all_ids = sorted(str(row["sample_id"]) for row in members_raw)
        self._external_split_view = (
            load_mesh_fk_external_split_view(
                config.split_assignments_path,
                eligible_sample_ids=set(all_ids),
            )
            if config.split_assignments_path is not None
            else None
        )
        index_rows = pq.read_table(
            l2_root / "episode_index.parquet",
            columns=["sample_id", "split", "part_uri", "row_start"],
        ).to_pylist()
        index_by_sample = {row["sample_id"]: row for row in index_rows}
        if set(index_by_sample) != set(all_ids):
            raise MeshFkL2DatasetError("episode index does not cover the membership")
        # Episode ordinal inside each part: row groups align with episodes,
        # so position by row_start order gives the exact row-group index.
        ordinal_by_part: dict[str, dict[str, int]] = {}
        for row in sorted(index_rows, key=lambda item: (item["part_uri"], item["row_start"])):
            ordinal_by_part.setdefault(row["part_uri"], {})[row["sample_id"]] = len(
                ordinal_by_part.get(row["part_uri"], {})
            )
        self._members: list[_Member] = []
        for row in members_raw:
            index_row = index_by_sample[row["sample_id"]]
            if index_row["split"] != row["split"]:
                raise MeshFkL2DatasetError(
                    f"{row['sample_id']}: handoff split disagrees with the episode index"
                )
            if index_row["part_uri"] != row["part_path"]:
                raise MeshFkL2DatasetError(
                    f"{row['sample_id']}: handoff locator disagrees with the episode index"
                )
            member_role = (
                self._external_split_view.roles_by_sample.get(row["sample_id"])
                if self._external_split_view is not None
                else row["split"]
            )
            if member_role == config.split:
                self._members.append(
                    _Member(
                        sample_id=row["sample_id"],
                        part_path=row["part_path"],
                        row_group=ordinal_by_part[row["part_path"]][row["sample_id"]],
                        frame_count=int(row["frame_count"]),
                    )
                )
        if not self._members:
            raise MeshFkL2DatasetError(f"split {config.split!r} has no eligible episodes")
        self._by_sample = {member.sample_id: member for member in self._members}

        self._handoff = handoff
        self._handoff_root = handoff_root
        self._l2_root = l2_root
        # Mesh rows carry all-true masks by contract.
        self._require_all_true_mask = True
        if config.max_frames is not None and config.max_frames < 1:
            raise MeshFkL2DatasetError("max_frames must be positive when set")
        self._max_frames = config.max_frames
        self._split = config.split
        self._train_augmentation = config.train_augmentation if config.split == "train" else None
        self._legacy_sequence_format = config.legacy_sequence_format
        self._full24_temporal_train_phase = config.full24_temporal_train_phase
        self._augmentation_epoch = (
            Value("q", 0, lock=False)
            if (
                self._train_augmentation is not None
                or self._legacy_sequence_format
                or self._full24_temporal_train_phase == "random"
            )
            else None
        )
        self._pose_frame = config.pose_frame
        self._pose_coordinate_frame = config.pose_coordinate_frame
        self._pure_hand_profile_path = (
            config.pose_normalization_profile.expanduser().resolve()
            if config.pose_normalization_profile is not None
            else None
        )
        self._full24_profile_path = (
            config.full24_normalization_profile.expanduser().resolve()
            if config.full24_normalization_profile is not None
            else None
        )
        self._hybrid24_profile_path = (
            config.hybrid24_normalization_profile.expanduser().resolve()
            if config.hybrid24_normalization_profile is not None
            else None
        )
        self._hybrid24v2_profile_path = (
            config.hybrid24v2_normalization_profile.expanduser().resolve()
            if config.hybrid24v2_normalization_profile is not None
            else None
        )
        self._unisign_hand_bridge_profile_path = (
            config.unisign_hand_bridge_normalization_profile.expanduser().resolve()
            if config.unisign_hand_bridge_normalization_profile is not None
            else None
        )
        self._full24_temporal_sampling_profile = config.full24_temporal_sampling_profile
        if self._pose_frame is None:
            if self._unisign_hand_bridge_profile_path is not None:
                if self._external_split_view is None:  # pragma: no cover - config guards it
                    raise MeshFkL2DatasetError(
                        "Uni-Sign hand bridge external split view is unavailable"
                    )
                self._profile = load_unisign_hand_bridge_sidecar(
                    handoff_root=handoff_root,
                    l2_root=l2_root,
                    profile_path=self._unisign_hand_bridge_profile_path,
                    external_split_view=self._external_split_view,
                    temporal_sampling_profile=self._full24_temporal_sampling_profile,
                )
            elif self._hybrid24v2_profile_path is not None:
                if self._external_split_view is None:  # pragma: no cover - config guards it
                    raise MeshFkL2DatasetError("hybrid24v2 external split view is unavailable")
                self._profile = load_hybrid24v2_external_normalization_profile(
                    handoff_root=handoff_root,
                    l2_root=l2_root,
                    profile_path=self._hybrid24v2_profile_path,
                    external_split_view=self._external_split_view,
                    temporal_sampling_profile=self._full24_temporal_sampling_profile,
                )
            elif self._hybrid24_profile_path is not None:
                if self._external_split_view is None:  # pragma: no cover - config guards it
                    raise MeshFkL2DatasetError("hybrid24 external split view is unavailable")
                self._profile = load_hybrid24_external_normalization_profile(
                    handoff_root=handoff_root,
                    l2_root=l2_root,
                    profile_path=self._hybrid24_profile_path,
                    external_split_view=self._external_split_view,
                    temporal_sampling_profile=self._full24_temporal_sampling_profile,
                )
            elif self._full24_profile_path is None:
                self._profile = load_mesh_fk_normalization_profile(l2_root, handoff)
            else:
                if self._external_split_view is None:  # pragma: no cover - config guards it
                    raise MeshFkL2DatasetError("full-24 external split view is unavailable")
                self._profile = load_full24_external_normalization_profile(
                    handoff_root=handoff_root,
                    l2_root=l2_root,
                    profile_path=self._full24_profile_path,
                    external_split_view=self._external_split_view,
                    temporal_sampling_profile=self._full24_temporal_sampling_profile,
                )
        else:
            if self._pure_hand_profile_path is None:  # pragma: no cover - config guards it
                raise MeshFkL2DatasetError("pure-hand normalization profile is unavailable")
            self._profile = load_pure_hand_normalization_profile(
                handoff_root=handoff_root,
                l2_root=l2_root,
                pose_frame=self._pose_frame,
                pose_coordinate_frame=cast(PureHandCoordinateFrame, self._pose_coordinate_frame),
                profile_path=self._pure_hand_profile_path,
                external_split_view=self._external_split_view,
            )
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = max(1, config.part_cache_size)
        raw_coordinate_frame = str(delivery["protocols"]["coordinate_frame"])
        if self._unisign_hand_bridge_profile_path is not None:
            self.coordinate_frame = (
                f"{PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08}_"
                "unisign_hand_bridge_bodyproxy_crop_xy"
            )
            self.pose_units = UNISIGN_HAND_BRIDGE_POSE_UNITS
        elif self._hybrid24v2_profile_path is not None:
            self.coordinate_frame = (
                f"{PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08}_"
                "hybrid24v2_rootabs_handglobal"
            )
            self.pose_units = HYBRID24_V2_NORMALIZED_POSE_UNITS
        elif self._hybrid24_profile_path is not None:
            self.coordinate_frame = HYBRID24_NORMALIZED_COORDINATE_FRAME
            self.pose_units = HYBRID24_NORMALIZED_POSE_UNITS
        elif self._pose_frame is None:
            self.coordinate_frame = raw_coordinate_frame
            self.pose_units = NORMALIZED_POSE_UNITS
        else:
            self.coordinate_frame = (
                f"{self._pose_coordinate_frame}_{self._pose_frame}_rootabs_handrel21"
            )
            self.pose_units = PURE_HAND_NORMALIZED_POSE_UNITS
        self.sample_ids = tuple(member.sample_id for member in self._members)
        self.joint_count = 24 if self._pose_frame is None else PURE_HAND_JOINT_COUNT
        self.coordinate_dim = 3
        self.input_mode = "pose_only"
        self.radar_feature_dim = None

    # ---- dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._members)

    def __getitem__(self, index: int) -> SignLanguageTranslationSample:
        return self.load_sample(index)

    def load_sample(self, index: int) -> SignLanguageTranslationSample:
        member = self._members[index]
        return self.load_episode(member.sample_id)

    def load_episode(self, sample_id: str) -> SignLanguageTranslationSample:
        member = self._by_sample.get(sample_id)
        if member is None:
            raise MeshFkL2DatasetError(f"{sample_id!r} is not in split {self._split!r}")
        table = self._part_reader(member.part_path).read_row_group(
            member.row_group, columns=list(PROJECTION_COLUMNS)
        )
        if table.num_rows != member.frame_count:
            raise MeshFkL2DatasetError(
                f"{sample_id}: row group holds {table.num_rows} rows, expected {member.frame_count}"
            )
        if isinstance(self._train_augmentation, MeshPoseTrainAugmentationConfig):
            return self._load_augmented_episode(sample_id, table)
        temporal_phase = 0
        if self._full24_temporal_sampling_profile is not None:
            if self._full24_temporal_train_phase == "random" and self._split == "train":
                temporal_phase = int(self._temporal_phase_rng(member.sample_id).integers(0, 3))
            source_indices = full24_temporal_indices(
                member.frame_count,
                self._full24_temporal_sampling_profile,
                temporal_phase,
            )
            source_frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
            if not np.array_equal(
                source_frame_index, np.arange(member.frame_count, dtype=np.int64)
            ):
                raise MeshFkL2DatasetError(
                    f"{sample_id}: full-24 temporal source indices are not contiguous"
                )
            table = table.take(source_indices)
        available_frames = table.num_rows
        limit = (
            available_frames
            if self._max_frames is None
            else min(available_frames, self._max_frames)
        )
        if limit < available_frames:
            if self._legacy_sequence_format:
                # The trainer's pad_sequence rule: a RANDOM subset per load
                # (sorted to keep order) — not a deterministic grid.
                epoch_value = (
                    self._augmentation_epoch.value if self._augmentation_epoch is not None else 0
                )
                rng = np.random.default_rng(
                    (hash(sample_id) & 0xFFFFFFFF) ^ (epoch_value * 2654435761)
                )
                indices = np.sort(rng.choice(available_frames, size=limit, replace=False))
                table = table.take(indices)
            else:
                # Legacy uniform sampling: keep the whole action, drop frames
                # on a uniform grid (linspace includes both endpoints).
                indices = np.linspace(0, available_frames - 1, limit, dtype=np.int64)
                table = table.take(indices)
        ids = set(table.column("sample_id").to_pylist())
        if ids != {sample_id}:
            raise MeshFkL2DatasetError(f"{sample_id}: row range covers samples {sorted(ids)[:3]}")
        captions = set(table.column("caption").to_pylist())
        frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        frames = table.num_rows
        pose = np.asarray(
            table.column("pose").combine_chunks().flatten().to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(frames, 2, 24, 3)
        confidence = np.asarray(
            table.column("pose_confidence")
            .combine_chunks()
            .flatten()
            .to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(frames, 2, 24)
        frame_mask = np.asarray(table.column("frame_mask").to_pylist(), dtype=np.bool_)
        validate_decoded_episode(
            sample_id,
            frame_index,
            pose,
            confidence,
            frame_mask,
            captions,
            frames,
            require_all_true_mask=self._require_all_true_mask,
            require_first_frame_zero=(not self._legacy_sequence_format and temporal_phase == 0),
        )
        caption = next(iter(captions))
        if self._unisign_hand_bridge_profile_path is not None:
            normalized = unisign_hand_bridge_pose_view(pose)
        elif self._hybrid24v2_profile_path is not None:
            normalized = apply_hybrid24v2_normalization(hybrid24v2_pose_view(pose), self._profile)
            confidence = hybrid24v2_confidence_view(confidence)
        elif self._hybrid24_profile_path is not None:
            normalized = apply_hybrid24_normalization(hybrid24_pose_view(pose), self._profile)
            confidence = hybrid24_confidence_view(confidence)
        elif self._pose_frame is None:
            normalized = apply_mesh_fk_normalization(pose, self._profile)
        else:
            normalized = apply_pure_hand_normalization(
                pure_hand_pose_view(
                    pose,
                    self._pose_frame,
                    cast(PureHandCoordinateFrame, self._pose_coordinate_frame),
                ),
                self._profile,
            )
            confidence = pure_hand_confidence_view(confidence)
        if isinstance(self._train_augmentation, Hybrid24PostnormGaussianNoiseConfig):
            normalized = apply_hybrid24_postnorm_gaussian_noise(
                normalized,
                self._train_augmentation,
                self._augmentation_rng(sample_id),
            )
        if self._legacy_sequence_format and self._max_frames is not None:
            # Trainer pad_sequence semantics: zero-pad pose to a FIXED
            # max_frames and let pad frames attend (all-true mask) — the
            # warm-start encoder's native token distribution.  Validation
            # above ran on the real frames only.
            pad = self._max_frames - frames
            if pad > 0:
                normalized = np.concatenate(
                    [
                        normalized,
                        np.zeros((pad, *normalized.shape[1:]), dtype=normalized.dtype),
                    ]
                )
                confidence = np.concatenate(
                    [
                        confidence,
                        np.zeros((pad, *confidence.shape[1:]), dtype=confidence.dtype),
                    ]
                )
                frame_mask = np.ones(self._max_frames, dtype=np.bool_)
        return SignLanguageTranslationSample(
            sample_id=sample_id,
            pose=normalized,
            pose_confidence=confidence,
            radar_feature=None,
            frame_mask=frame_mask,
            caption=caption,
            coordinate_frame=self.coordinate_frame,
            pose_units=self.pose_units,
        )

    def _load_augmented_episode(
        self,
        sample_id: str,
        table: Any,
    ) -> SignLanguageTranslationSample:
        profile = self._train_augmentation
        if not isinstance(profile, MeshPoseTrainAugmentationConfig):  # pragma: no cover
            raise MeshFkL2DatasetError("train augmentation profile is unavailable")
        ids = set(table.column("sample_id").to_pylist())
        if ids != {sample_id}:
            raise MeshFkL2DatasetError(f"{sample_id}: row range covers samples {sorted(ids)[:3]}")
        captions = set(table.column("caption").to_pylist())
        frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        source_frames = table.num_rows
        pose = np.asarray(
            table.column("pose").combine_chunks().flatten().to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(source_frames, 2, 24, 3)
        confidence = np.asarray(
            table.column("pose_confidence")
            .combine_chunks()
            .flatten()
            .to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(source_frames, 2, 24)
        frame_mask = np.asarray(table.column("frame_mask").to_pylist(), dtype=np.bool_)
        validate_decoded_episode(
            sample_id,
            frame_index,
            pose,
            confidence,
            frame_mask,
            captions,
            source_frames,
            require_all_true_mask=self._require_all_true_mask,
        )
        speed, rotation, scale = sample_mesh_pose_train_parameters(
            profile,
            self._augmentation_rng(sample_id),
        )
        indices = full_action_temporal_indices(source_frames, speed, self._max_frames)
        if (
            int(indices[0]) != 0
            or int(indices[-1]) != source_frames - 1
            or bool(np.any(np.diff(indices) < 0))
        ):
            raise MeshFkL2DatasetError("temporal augmentation grid is invalid")
        selected_pose = pose[indices]
        selected_confidence = confidence[indices]
        selected_mask = frame_mask[indices]
        normalized = standardize_mesh_fk_pose(
            apply_mesh_pose_spatial_transform(
                root_center_mesh_fk_pose(selected_pose, self._profile),
                rotation,
                scale,
            ),
            self._profile,
        )
        return SignLanguageTranslationSample(
            sample_id=sample_id,
            pose=normalized,
            pose_confidence=selected_confidence,
            radar_feature=None,
            frame_mask=selected_mask,
            caption=next(iter(captions)),
            coordinate_frame=self.coordinate_frame,
            pose_units=self.pose_units,
        )

    # ---- binding metadata -------------------------------------------------

    def checkpoint_identity(self) -> dict[str, object]:
        """Canonical run/checkpoint identity for this exact source.

        Split-agnostic by design: training (train split) and evaluation
        (validation/test split) must produce identical dicts for the
        checkpoint binding comparison.  Use :attr:`split` for per-split
        diagnostics.
        """

        identity: dict[str, object] = {
            key: value
            for key, value in mesh_fk_source_identity(
                self._handoff_root,
                self._l2_root,
            ).items()
        }
        if self._pose_frame is not None:
            if self._pure_hand_profile_path is None:  # pragma: no cover - config guards it
                raise MeshFkL2DatasetError("pure-hand normalization profile is unavailable")
            identity.update(
                {
                    "pose_view": PURE_HAND_POSE_VIEW,
                    "pose_frame": self._pose_frame,
                    "pose_coordinate_frame": self._pose_coordinate_frame,
                    "coordinate_transform": pure_hand_coordinate_transform(
                        cast(PureHandCoordinateFrame, self._pose_coordinate_frame)
                    ),
                    "source_joint_indices": ",".join(
                        str(index) for index in PURE_HAND_SOURCE_JOINT_INDICES
                    ),
                    "root_source_joint_index": PURE_HAND_ROOT_SOURCE_JOINT_INDEX,
                    "geometry_source_joint_indices": ",".join(
                        str(index) for index in PURE_HAND_GEOMETRY_SOURCE_JOINT_INDICES
                    ),
                    "normalization_profile_sha256": _sha256_file(self._pure_hand_profile_path),
                    "normalization_protocol": str(self._profile["protocol_id"]),
                }
            )
        elif self._unisign_hand_bridge_profile_path is not None:
            identity.update(
                {
                    "pose_view": UNISIGN_HAND_BRIDGE_POSE_VIEW,
                    "pose_coordinate_frame": PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
                    "coordinate_transform": pure_hand_coordinate_transform(
                        cast(
                            PureHandCoordinateFrame,
                            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
                        )
                    ),
                    "slot_layout": _unisign_hand_bridge_slot_layout(),
                    "body_proxy": _unisign_hand_bridge_body_proxy(),
                    "normalization_profile_sha256": _sha256_file(
                        self._unisign_hand_bridge_profile_path
                    ),
                    "normalization_protocol": str(self._profile["protocol_id"]),
                    "temporal_sampling": self._temporal_sampling_identity(),
                }
            )
        elif self._hybrid24v2_profile_path is not None:
            identity.update(
                {
                    "pose_view": HYBRID24_V2_POSE_VIEW,
                    "pose_coordinate_frame": PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
                    "coordinate_transform": pure_hand_coordinate_transform(
                        cast(
                            PureHandCoordinateFrame,
                            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
                        )
                    ),
                    "slot_layout": _hybrid24v2_slot_layout(),
                    "normalization_profile_sha256": _sha256_file(self._hybrid24v2_profile_path),
                    "normalization_protocol": str(self._profile["protocol_id"]),
                    "temporal_sampling": self._temporal_sampling_identity(),
                }
            )
        elif self._hybrid24_profile_path is not None:
            identity.update(
                {
                    "pose_view": HYBRID24_POSE_VIEW,
                    "pose_coordinate_frame": PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
                    "coordinate_transform": pure_hand_coordinate_transform(
                        cast(
                            PureHandCoordinateFrame,
                            PURE_HAND_COORDINATE_FRAME_RADAR_CAMERA_STANDOFF08,
                        )
                    ),
                    "slot_layout": _hybrid24_slot_layout(),
                    "normalization_profile_sha256": _sha256_file(self._hybrid24_profile_path),
                    "normalization_protocol": str(self._profile["protocol_id"]),
                    "temporal_sampling": self._temporal_sampling_identity(),
                }
            )
        elif self._full24_profile_path is not None:
            identity.update(
                {
                    "pose_view": FULL24_POSE_VIEW,
                    "normalization_profile_sha256": _sha256_file(self._full24_profile_path),
                    "normalization_protocol": str(self._profile["protocol_id"]),
                    "temporal_sampling": self._temporal_sampling_identity(),
                }
            )
        if self._external_split_view is not None:
            identity["external_split_view"] = self._external_split_view.checkpoint_identity()
        return identity

    @property
    def split(self) -> str:
        return self._split

    @property
    def train_augmentation(self) -> WaveLLMTrainAugmentationConfig | None:
        return self._train_augmentation

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise MeshFkL2DatasetError("augmentation epoch must be non-negative")
        if self._augmentation_epoch is not None:
            self._augmentation_epoch.value = epoch

    def _augmentation_rng(self, sample_id: str) -> np.random.Generator:
        profile = self._train_augmentation
        epoch_state = self._augmentation_epoch
        if profile is None or epoch_state is None:
            raise MeshFkL2DatasetError("train augmentation profile is unavailable")
        encoded = f"{profile.seed}\0{sample_id}\0{epoch_state.value}".encode()
        seed = int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")
        return np.random.default_rng(seed)

    def _temporal_phase_rng(self, sample_id: str) -> np.random.Generator:
        epoch_value = self._augmentation_epoch.value if self._augmentation_epoch is not None else 0
        encoded = f"temporal-phase\0{sample_id}\0{epoch_value}".encode()
        seed = int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")
        return np.random.default_rng(seed)

    def _temporal_sampling_identity(self) -> dict[str, object] | None:
        identity = full24_temporal_sampling_identity(self._full24_temporal_sampling_profile)
        if identity is not None and self._full24_temporal_train_phase == "random":
            identity = {**identity, "train_phase": "random_per_epoch"}
        return identity

    def normalization_profile_digest(self) -> str:
        payload = json.dumps(
            {
                "protocol_id": self._profile.get("protocol_id"),
                "root_center": self._profile.get("root_center"),
                "root_slot": self._profile.get("root_slot"),
                "geometry_frame": self._profile.get("geometry_frame"),
                "mean": self._profile.get("mean"),
                "std": self._profile.get("std"),
            },
            sort_keys=True,
        )
        if self._pose_frame is not None:
            payload = json.dumps(
                {
                    **json.loads(payload),
                    "pose_frame": self._profile.get("pose_frame"),
                    "coordinate_frame": self._profile.get("coordinate_frame"),
                    "coordinate_transform": self._profile.get("coordinate_transform"),
                    "pose_view": self._profile.get("pose_view"),
                    "source_joint_indices": self._profile.get("source_joint_indices"),
                    "geometry_source_joint_indices": self._profile.get(
                        "geometry_source_joint_indices"
                    ),
                },
                sort_keys=True,
            )
        elif self._unisign_hand_bridge_profile_path is not None:
            payload = json.dumps(
                {
                    **json.loads(payload),
                    "pose_view": self._profile.get("pose_view"),
                    "slot_layout": self._profile.get("slot_layout"),
                    "body_proxy": self._profile.get("body_proxy"),
                    "temporal_sampling": self._profile.get("temporal_sampling"),
                    "external_split_view": self._profile.get("external_split_view"),
                },
                sort_keys=True,
            )
        elif self._full24_profile_path is not None:
            payload = json.dumps(
                {
                    **json.loads(payload),
                    "pose_view": self._profile.get("pose_view"),
                    "temporal_sampling": self._profile.get("temporal_sampling"),
                    "external_split_view": self._profile.get("external_split_view"),
                },
                sort_keys=True,
            )
        return hashlib.sha256(payload.encode()).hexdigest()

    def close(self) -> None:
        """Release cached part readers (parquet handles)."""

        for reader in self._cache.values():
            closer = getattr(reader, "close", None)
            if closer is not None:
                closer()
        self._cache.clear()

    # ---- bounded part cache (used by batch tooling) ------------------------

    def _part_reader(self, part_path: str) -> Any:
        cached = self._cache.get(part_path)
        if cached is not None:
            self._cache.move_to_end(part_path)
            return cached
        reader = self._pq.ParquetFile(self._l2_root / part_path)
        self._cache[part_path] = reader
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return reader
