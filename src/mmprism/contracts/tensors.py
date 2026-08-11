from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

LeadingAxis = Literal["batch", "time"]

RAW_RADAR_TRAILING_AXES = ("chirp", "antenna", "sample")
RANGE_DOPPLER_TRAILING_AXES = ("doppler", "antenna", "range")
RADAR_CUBE_TRAILING_AXES = ("doppler", "range", "azimuth", "elevation")
DUAL_HAND_POSE_TRAILING_AXES = ("hand", "joint", "coordinate")
FEATURE_SEQUENCE_TRAILING_AXES = ("time", "feature")
DUAL_HAND_ORDER = ("left", "right")
DUAL_HAND_JOINT_ORDER = (
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
    "hand_wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_tip",
    "index_1",
    "index_2",
    "index_3",
    "index_tip",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_tip",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_tip",
    "little_1",
    "little_2",
    "little_3",
    "little_tip",
)
CARTESIAN_COORDINATE_ORDER = ("x", "y", "z")

_ALLOWED_LEADING_AXES: set[tuple[LeadingAxis, ...]] = {
    (),
    ("batch",),
    ("time",),
    ("batch", "time"),
}
_LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")


class TensorContractError(ValueError):
    """Raised when an array violates a canonical tensor contract."""


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Validated structural metadata without retaining the source array."""

    axes: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    units: str | None = None
    coordinate_frame: str | None = None


def _leading_axes(value: tuple[LeadingAxis, ...]) -> tuple[LeadingAxis, ...]:
    if value not in _ALLOWED_LEADING_AXES:
        raise TensorContractError(
            "leading_axes must be one of (), ('batch',), ('time',), or "
            "('batch', 'time')"
        )
    return value


def _array(
    value: object,
    *,
    name: str,
    leading_axes: tuple[LeadingAxis, ...],
    trailing_axes: tuple[str, ...],
    dtype_kind: Literal["complex", "floating"],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TensorContractError(f"{name} must be a numpy.ndarray")
    axes = _leading_axes(leading_axes) + trailing_axes
    if value.ndim != len(axes):
        raise TensorContractError(
            f"{name} must have axes {axes}; expected {len(axes)} dimensions, "
            f"got shape {value.shape}"
        )
    if any(size <= 0 for size in value.shape):
        raise TensorContractError(f"{name} dimensions must all be positive: {value.shape}")

    dtype = np.dtype(value.dtype)
    if dtype_kind == "complex" and not np.issubdtype(dtype, np.complexfloating):
        raise TensorContractError(f"{name} must have a complex dtype, got {dtype.name}")
    if dtype_kind == "floating" and not np.issubdtype(dtype, np.floating):
        raise TensorContractError(f"{name} must have a floating dtype, got {dtype.name}")
    if not bool(np.all(np.isfinite(value))):
        raise TensorContractError(f"{name} must contain only finite values")
    return value


def _metadata(
    value: np.ndarray,
    *,
    leading_axes: tuple[LeadingAxis, ...],
    trailing_axes: tuple[str, ...],
    units: str | None = None,
    coordinate_frame: str | None = None,
) -> TensorMetadata:
    return TensorMetadata(
        axes=_leading_axes(leading_axes) + trailing_axes,
        shape=tuple(int(size) for size in value.shape),
        dtype=np.dtype(value.dtype).name,
        units=units,
        coordinate_frame=coordinate_frame,
    )


def validate_raw_radar(
    value: object,
    *,
    leading_axes: tuple[LeadingAxis, ...] = (),
) -> TensorMetadata:
    """Validate complex ADC data stored as ``[..., chirp, antenna, sample]``."""

    array = _array(
        value,
        name="raw radar",
        leading_axes=leading_axes,
        trailing_axes=RAW_RADAR_TRAILING_AXES,
        dtype_kind="complex",
    )
    return _metadata(
        array,
        leading_axes=leading_axes,
        trailing_axes=RAW_RADAR_TRAILING_AXES,
        units="complex_adc",
    )


def validate_range_doppler(
    value: object,
    *,
    leading_axes: tuple[LeadingAxis, ...] = (),
) -> TensorMetadata:
    """Validate a complex ``[..., doppler, antenna, range]`` spectrum."""

    array = _array(
        value,
        name="range-Doppler spectrum",
        leading_axes=leading_axes,
        trailing_axes=RANGE_DOPPLER_TRAILING_AXES,
        dtype_kind="complex",
    )
    return _metadata(
        array,
        leading_axes=leading_axes,
        trailing_axes=RANGE_DOPPLER_TRAILING_AXES,
        units="complex_spectrum",
    )


def validate_radar_cube(
    value: object,
    *,
    leading_axes: tuple[LeadingAxis, ...] = (),
) -> TensorMetadata:
    """Validate non-negative power with ``[..., D, R, azimuth, elevation]`` axes."""

    array = _array(
        value,
        name="radar cube",
        leading_axes=leading_axes,
        trailing_axes=RADAR_CUBE_TRAILING_AXES,
        dtype_kind="floating",
    )
    if bool(np.any(array < 0)):
        raise TensorContractError("radar cube power must be non-negative")
    return _metadata(
        array,
        leading_axes=leading_axes,
        trailing_axes=RADAR_CUBE_TRAILING_AXES,
        units="power",
    )


def validate_dual_hand_pose(
    value: object,
    *,
    coordinate_frame: str,
    units: str = "m",
    leading_axes: tuple[LeadingAxis, ...] = (),
) -> TensorMetadata:
    """Validate metric dual-hand joints with trailing shape ``[2, 24, 3]``."""

    if units != "m":
        raise TensorContractError("dual-hand pose units must be 'm'")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise TensorContractError("dual-hand pose coordinate_frame must be explicit")
    array = _array(
        value,
        name="dual-hand pose",
        leading_axes=leading_axes,
        trailing_axes=DUAL_HAND_POSE_TRAILING_AXES,
        dtype_kind="floating",
    )
    if array.shape[-3:] != (2, 24, 3):
        raise TensorContractError(
            "dual-hand pose trailing shape must be (2, 24, 3), "
            f"got {array.shape[-3:]}"
        )
    return _metadata(
        array,
        leading_axes=leading_axes,
        trailing_axes=DUAL_HAND_POSE_TRAILING_AXES,
        units=units,
        coordinate_frame=coordinate_frame.strip(),
    )


def validate_feature_sequence(
    value: object,
    *,
    leading_axes: tuple[LeadingAxis, ...] = (),
) -> TensorMetadata:
    """Validate finite floating features stored as ``[..., time, feature]``."""

    if "time" in leading_axes:
        raise TensorContractError(
            "feature sequence already has a trailing time axis; leading_axes may only be () "
            "or ('batch',)"
        )
    array = _array(
        value,
        name="feature sequence",
        leading_axes=leading_axes,
        trailing_axes=FEATURE_SEQUENCE_TRAILING_AXES,
        dtype_kind="floating",
    )
    return _metadata(
        array,
        leading_axes=leading_axes,
        trailing_axes=FEATURE_SEQUENCE_TRAILING_AXES,
        units="feature",
    )


def validate_caption(value: object, *, language: str) -> str:
    """Validate one non-empty Unicode caption and its explicit language tag."""

    if not isinstance(value, str) or not value.strip():
        raise TensorContractError("caption must be a non-empty string")
    if "\x00" in value:
        raise TensorContractError("caption must not contain NUL characters")
    if not isinstance(language, str) or not _LANGUAGE_TAG.fullmatch(language):
        raise TensorContractError(f"caption language must be a BCP-47-like tag: {language!r}")
    return value
