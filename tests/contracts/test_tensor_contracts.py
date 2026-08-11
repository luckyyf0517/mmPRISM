from __future__ import annotations

import numpy as np
import pytest

from mmprism.contracts.tensors import (
    CARTESIAN_COORDINATE_ORDER,
    DUAL_HAND_JOINT_ORDER,
    DUAL_HAND_ORDER,
    TensorContractError,
    validate_caption,
    validate_dual_hand_pose,
    validate_feature_sequence,
    validate_radar_cube,
    validate_range_doppler,
    validate_raw_radar,
)


def test_raw_radar_contract_records_explicit_leading_axes() -> None:
    raw = np.zeros((2, 3, 8, 4, 16), dtype=np.complex64)

    metadata = validate_raw_radar(raw, leading_axes=("batch", "time"))

    assert metadata.axes == ("batch", "time", "chirp", "antenna", "sample")
    assert metadata.shape == raw.shape
    assert metadata.dtype == "complex64"
    assert metadata.units == "complex_adc"


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.zeros((8, 4, 16), dtype=np.float32), "complex dtype"),
        (np.zeros((8, 16), dtype=np.complex64), "expected 3 dimensions"),
        (np.zeros((8, 0, 16), dtype=np.complex64), "positive"),
    ],
)
def test_raw_radar_contract_rejects_invalid_arrays(value: np.ndarray, match: str) -> None:
    with pytest.raises(TensorContractError, match=match):
        validate_raw_radar(value)


def test_complex_contracts_reject_nonfinite_values() -> None:
    spectrum = np.zeros((8, 4, 16), dtype=np.complex64)
    spectrum[0, 0, 0] = complex(np.nan, 0.0)

    with pytest.raises(TensorContractError, match="finite"):
        validate_range_doppler(spectrum)


def test_radar_cube_requires_real_nonnegative_power() -> None:
    cube = np.ones((8, 16, 4, 3), dtype=np.float32)
    metadata = validate_radar_cube(cube)
    assert metadata.axes == ("doppler", "range", "azimuth", "elevation")
    assert metadata.units == "power"

    cube[0, 0, 0, 0] = -0.1
    with pytest.raises(TensorContractError, match="non-negative"):
        validate_radar_cube(cube)
    with pytest.raises(TensorContractError, match="floating dtype"):
        validate_radar_cube(np.ones((8, 16, 4, 3), dtype=np.complex64))


def test_dual_hand_pose_requires_shape_metres_and_coordinate_frame() -> None:
    pose = np.zeros((5, 2, 24, 3), dtype=np.float32)
    metadata = validate_dual_hand_pose(
        pose,
        leading_axes=("time",),
        units="m",
        coordinate_frame="radar_cartesian_v1",
    )
    assert metadata.axes == ("time", "hand", "joint", "coordinate")
    assert metadata.units == "m"
    assert metadata.coordinate_frame == "radar_cartesian_v1"
    assert DUAL_HAND_ORDER == ("left", "right")
    assert len(DUAL_HAND_JOINT_ORDER) == 24
    assert CARTESIAN_COORDINATE_ORDER == ("x", "y", "z")

    with pytest.raises(TensorContractError, match="trailing shape"):
        validate_dual_hand_pose(
            np.zeros((2, 21, 3), dtype=np.float32),
            coordinate_frame="radar_cartesian_v1",
        )
    with pytest.raises(TensorContractError, match="units"):
        validate_dual_hand_pose(pose, units="mm", coordinate_frame="radar_cartesian_v1")
    with pytest.raises(TensorContractError, match="coordinate_frame"):
        validate_dual_hand_pose(pose, coordinate_frame="")


def test_feature_and_caption_contracts() -> None:
    features = np.zeros((2, 10, 64), dtype=np.float32)
    metadata = validate_feature_sequence(features, leading_axes=("batch",))
    assert metadata.axes == ("batch", "time", "feature")
    assert validate_caption("早上好", language="zh-CN") == "早上好"

    with pytest.raises(TensorContractError, match="already has"):
        validate_feature_sequence(features, leading_axes=("time",))
    with pytest.raises(TensorContractError, match="non-empty"):
        validate_caption("  ", language="zh-CN")
    with pytest.raises(TensorContractError, match="language"):
        validate_caption("text", language="Chinese (simplified)")


def test_contract_rejects_implicit_or_misordered_leading_axes() -> None:
    raw = np.zeros((2, 3, 8, 4, 16), dtype=np.complex64)
    with pytest.raises(TensorContractError, match="leading_axes"):
        validate_raw_radar(raw, leading_axes=("time", "batch"))  # type: ignore[arg-type]
    with pytest.raises(TensorContractError, match="expected 3 dimensions"):
        validate_raw_radar(raw)
