"""Versioned pose and language evaluation protocols with lazy optional imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mmprism.evaluation.language import (
        LANGUAGE_METRIC_PROTOCOL as LANGUAGE_METRIC_PROTOCOL,
    )
    from mmprism.evaluation.language import (
        LanguageMetricAccumulator as LanguageMetricAccumulator,
    )
    from mmprism.evaluation.language import (
        LanguageMetricError as LanguageMetricError,
    )
    from mmprism.evaluation.language import (
        LanguageSampleMetric as LanguageSampleMetric,
    )
    from mmprism.evaluation.language import (
        character_edit_distance as character_edit_distance,
    )
    from mmprism.evaluation.pose import (
        FINGER_JOINT_START_INDEX as FINGER_JOINT_START_INDEX,
    )
    from mmprism.evaluation.pose import (
        HAND_WRIST_INDEX as HAND_WRIST_INDEX,
    )
    from mmprism.evaluation.pose import (
        POSE_METRIC_PROTOCOL as POSE_METRIC_PROTOCOL,
    )
    from mmprism.evaluation.pose import (
        PoseMetricAccumulator as PoseMetricAccumulator,
    )
    from mmprism.evaluation.pose import (
        masked_pose_l1_metres as masked_pose_l1_metres,
    )
    from mmprism.evaluation.pose import (
        pose_metric_tensors as pose_metric_tensors,
    )

_LANGUAGE_EXPORTS = frozenset(
    {
        "LANGUAGE_METRIC_PROTOCOL",
        "LanguageMetricAccumulator",
        "LanguageMetricError",
        "LanguageSampleMetric",
        "character_edit_distance",
    }
)
_POSE_EXPORTS = frozenset(
    {
        "FINGER_JOINT_START_INDEX",
        "HAND_WRIST_INDEX",
        "POSE_METRIC_PROTOCOL",
        "PoseMetricAccumulator",
        "masked_pose_l1_metres",
        "pose_metric_tensors",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LANGUAGE_EXPORTS:
        language = import_module("mmprism.evaluation.language")
        return getattr(language, name)
    if name in _POSE_EXPORTS:
        pose = import_module("mmprism.evaluation.pose")
        return getattr(pose, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_LANGUAGE_EXPORTS | _POSE_EXPORTS)
