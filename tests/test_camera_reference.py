"""The computed camera-pose reference must match the constants the paper prints."""

from __future__ import annotations

import json

import pytest

from mmprism.release.camera_reference import compare_with_printed
from mmprism.release.data import SIGNERS
from mmprism.release.paths import PathConfigurationError, resolve

# The printed values come from the delivered decode; mmprism-reference re-derives them, so the
# recomputation is allowed the rounding of a single decode.
TOLERANCE = {"bleu4": 0.2, "rouge_l": 0.2, "sbert": 0.005, "simcse": 0.2}


def _printed(tmp_path, metrics):
    path = tmp_path / "printed.json"
    path.write_text(json.dumps({"metrics": metrics}))
    return path


def test_comparison_reports_drift(tmp_path) -> None:
    printed = {"bleu4": 24.0, "rouge_l": 50.0, "sbert": 0.66, "simcse": 87.0}
    computed = {"metrics": {"bleu4": 25.5, "rouge_l": 49.0, "sbert": 0.68, "simcse": 87.0}}
    rows = compare_with_printed(computed, _printed(tmp_path, printed))
    assert rows["bleu4"]["delta"] == pytest.approx(1.5)
    assert rows["rouge_l"]["delta"] == pytest.approx(-1.0)
    assert rows["simcse"]["delta"] == 0.0


def test_computed_reference_matches_printed_constants() -> None:
    try:
        computed_path = resolve("outputs.translation_reference")
    except PathConfigurationError:
        pytest.skip("set output_root, then run mmprism-reference for all nine signers")
    if not computed_path.is_file():
        pytest.skip("run mmprism-reference for all nine signers first")
    computed = json.loads(computed_path.read_text())
    assert computed["signers"] == list(SIGNERS), "the reference must be computed on all nine signers"
    rows = compare_with_printed(computed)
    assert set(rows) == set(TOLERANCE)
    for key, row in rows.items():
        assert abs(row["delta"]) <= TOLERANCE[key], f"{key}: {row}"
