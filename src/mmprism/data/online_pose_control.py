"""CPU-only source contract for pose-driven online simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ONLINE_POSE_CONTROL_SAMPLE_PROTOCOL = "mmprism.online_pose_control.sample_v1"

ONLINE_POSE_MODALITY = "cam_pose"

ONLINE_POSE_AUDIT_MODALITY = "cam_pose_audit"

ONLINE_POSE_CAPTION_MODALITY = "caption"

ONLINE_POSE_ELIGIBILITY_RELIABILITY_FLOOR = np.float32(0.25)

ONLINE_POSE_MINIMUM_VALID_JOINTS = 12

ONLINE_POSE_MINIMUM_VALID_JOINTS_PER_HAND = 3


class OnlinePoseControlDataError(ValueError):
    """Raised when an online-pose source violates its accepted contract."""


@dataclass(frozen=True, slots=True)
class OnlinePoseControlSample:
    """One temporally aligned pose sequence before online radar simulation."""

    sample_id: str
    pose: np.ndarray
    reliability: np.ndarray
    xy_valid: np.ndarray
    z_valid: np.ndarray
    observed: np.ndarray
    valid: np.ndarray
    fitted: np.ndarray
    imputed: np.ndarray
    gap_length: np.ndarray
    frame_mask: np.ndarray
    caption: str
    coordinate_frame: str
    scale_mode: str
    scene_placement: str
    timing_source: str
    evidence_role: str
    annotation_build_id: str
    annotation_producer_commit: str
    acceptance_report_sha256: str
    eligibility_manifest_sha256: str
    pose_sha256: str
    audit_sha256: str


def validate_online_pose_arrays(
    *,
    sample_id: str,
    pose: np.ndarray,
    reliability: np.ndarray,
    xy_valid: np.ndarray,
    z_valid: np.ndarray,
    observed: np.ndarray,
    valid: np.ndarray,
    fitted: np.ndarray,
    imputed: np.ndarray,
    gap_length: np.ndarray,
    frame_mask: np.ndarray,
) -> None:
    """Validate annotation-v2 relationships without importing training dependencies."""

    frames = pose.shape[0] if pose.ndim == 4 else 0
    joint_shape = (frames, 2, 24)
    if frames < 1 or pose.shape != (*joint_shape, 3) or pose.dtype != np.float32:
        raise OnlinePoseControlDataError(f"sample {sample_id} pose must be float32 [T,2,24,3]")
    for name, value in (
        ("xy_valid", xy_valid),
        ("z_valid", z_valid),
        ("observed", observed),
        ("valid", valid),
        ("fitted", fitted),
        ("imputed", imputed),
    ):
        if value.shape != joint_shape or value.dtype != np.bool_:
            raise OnlinePoseControlDataError(f"sample {sample_id} {name} must be bool [T,2,24]")
    if reliability.shape != joint_shape or reliability.dtype != np.float32:
        raise OnlinePoseControlDataError(f"sample {sample_id} reliability must be float32 [T,2,24]")
    if gap_length.shape != joint_shape or gap_length.dtype != np.uint16:
        raise OnlinePoseControlDataError(f"sample {sample_id} gap_length must be uint16 [T,2,24]")
    if frame_mask.shape != (frames,) or frame_mask.dtype != np.bool_:
        raise OnlinePoseControlDataError(f"sample {sample_id} frame_mask must be bool [T]")
    if not bool(np.isfinite(pose).all() and np.isfinite(reliability).all()):
        raise OnlinePoseControlDataError(f"sample {sample_id} pose/reliability must be finite")
    if bool(((reliability < 0.0) | (reliability > 1.0)).any()):
        raise OnlinePoseControlDataError(f"sample {sample_id} reliability must be within [0,1]")
    if not np.array_equal(valid, observed & xy_valid & z_valid):
        raise OnlinePoseControlDataError(
            f"sample {sample_id} valid mask is inconsistent with observation/x/y/z"
        )
    if not np.array_equal(imputed, fitted):
        raise OnlinePoseControlDataError(f"sample {sample_id} imputed mask must equal fitted mask")
    if bool((fitted & valid).any()) or bool((fitted & (gap_length == 0)).any()):
        raise OnlinePoseControlDataError(
            f"sample {sample_id} fitted/gap/valid relationships are inconsistent"
        )
    if bool((valid & (gap_length != 0)).any()):
        raise OnlinePoseControlDataError(
            f"sample {sample_id} valid joints must have zero invalid-gap length"
        )
    if bool(((reliability > 0.0) & ~(observed | fitted)).any()):
        raise OnlinePoseControlDataError(
            f"sample {sample_id} reliability has no observed or fitted evidence"
        )
    eligible_joint = reliability >= ONLINE_POSE_ELIGIBILITY_RELIABILITY_FLOOR
    expected_frame_mask = (eligible_joint.sum(axis=(1, 2)) >= ONLINE_POSE_MINIMUM_VALID_JOINTS) & (
        eligible_joint.sum(axis=2) >= ONLINE_POSE_MINIMUM_VALID_JOINTS_PER_HAND
    ).all(axis=1)
    if not np.array_equal(frame_mask, expected_frame_mask):
        raise OnlinePoseControlDataError(
            f"sample {sample_id} frame mask is inconsistent with accepted reliability policy"
        )
