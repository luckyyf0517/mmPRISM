from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmprism.evaluation.pose import (  # noqa: E402
    HAND_WRIST_INDEX,
    POSE_METRIC_PROTOCOL,
    PoseMetricAccumulator,
    masked_pose_l1_metres,
    pose_metric_tensors,
)


def test_pose_metrics_use_hand_wrist_and_exclude_it_from_relative_pck() -> None:
    target = torch.zeros(1, 2, 24, 3)
    prediction = torch.zeros_like(target)
    valid = torch.zeros(1, 2, 24, dtype=torch.bool)
    valid[0, 0, HAND_WRIST_INDEX:] = True

    prediction[0, 0, HAND_WRIST_INDEX:, 0] = 0.1
    prediction[0, 0, HAND_WRIST_INDEX + 1, 0] = 0.15

    metrics = pose_metric_tensors(prediction, target, valid, pck_threshold_mm=40.0)

    assert POSE_METRIC_PROTOCOL.endswith("dual_hand_metric_v1")
    torch.testing.assert_close(metrics["root_relative_mpjpe_mm"], torch.tensor(2.5))
    torch.testing.assert_close(metrics["root_relative_pck"], torch.tensor(0.95))
    assert metrics["absolute_mpjpe_mm"] > 100
    assert torch.isfinite(masked_pose_l1_metres(prediction, target, valid))


def test_pose_metrics_require_a_valid_sample_and_wrist_finger_pair() -> None:
    pose = torch.zeros(2, 2, 24, 3)
    valid = torch.ones(2, 2, 24, dtype=torch.bool)
    valid[1] = False

    with pytest.raises(ValueError, match="every pose sample"):
        pose_metric_tensors(pose, pose, valid)

    valid[:] = False
    valid[:, :, 0] = True
    with pytest.raises(ValueError, match="valid wrist and finger"):
        pose_metric_tensors(pose, pose, valid)


def test_streaming_pose_metrics_match_one_shot_weighting() -> None:
    target = torch.zeros(3, 2, 24, 3)
    prediction = torch.full_like(target, 0.001)
    prediction[2] = 0.004
    valid = torch.ones(3, 2, 24, dtype=torch.bool)
    valid[1, 1, 10:] = False
    expected = pose_metric_tensors(prediction, target, valid, pck_threshold_mm=2.0)
    expected_l1 = masked_pose_l1_metres(prediction, target, valid)
    accumulator = PoseMetricAccumulator(pck_threshold_mm=2.0)

    first_samples = accumulator.update(prediction[:1], target[:1], valid[:1])
    remaining_samples = accumulator.update(prediction[1:], target[1:], valid[1:])
    values = accumulator.values()

    assert accumulator.sample_count == 3
    assert torch.cat((first_samples, remaining_samples)).shape == (3,)
    assert values["masked_pose_l1_metres"] == pytest.approx(float(expected_l1), rel=1e-6)
    assert values["absolute_mpjpe_mm"] == pytest.approx(
        float(expected["absolute_mpjpe_mm"]), rel=1e-6
    )
    assert values["root_relative_mpjpe_mm"] == pytest.approx(
        float(expected["root_relative_mpjpe_mm"]), abs=1e-6
    )
    assert values["root_relative_pck"] == pytest.approx(
        float(expected["root_relative_pck"]), abs=1e-6
    )
