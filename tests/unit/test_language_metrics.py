from __future__ import annotations

import subprocess
import sys

import pytest

from mmprism.evaluation import (
    LANGUAGE_METRIC_PROTOCOL,
    LanguageMetricAccumulator,
    LanguageMetricError,
    character_edit_distance,
)


def test_character_metric_protocol_and_count_weighting() -> None:
    assert LANGUAGE_METRIC_PROTOCOL == "mmprism.language_metric.character_v1"
    assert character_edit_distance("abc", "adc") == 1
    assert character_edit_distance("abc", "") == 3

    accumulator = LanguageMetricAccumulator()
    samples = accumulator.update(("abc", "de"), ("adc", "de"))

    assert [sample.character_edit_distance for sample in samples] == [1, 0]
    assert [sample.exact_match for sample in samples] == [False, True]
    assert accumulator.values() == {
        "exact_match": 0.5,
        "character_error_rate": 0.2,
        "exact_match_count": 1,
        "character_edit_distance_sum": 1,
        "reference_character_count": 5,
        "prediction_character_count": 5,
    }


def test_language_metric_import_does_not_load_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from mmprism.evaluation import LanguageMetricAccumulator; "
                "assert LanguageMetricAccumulator is not None; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_character_metrics_reject_empty_reference_and_misaligned_batches() -> None:
    accumulator = LanguageMetricAccumulator()
    with pytest.raises(LanguageMetricError, match="same non-zero"):
        accumulator.update(("reference",), ())
    with pytest.raises(LanguageMetricError, match="reference must not be empty"):
        accumulator.update(("  ",), ("prediction",))
    with pytest.raises(LanguageMetricError, match="at least one"):
        accumulator.values()
