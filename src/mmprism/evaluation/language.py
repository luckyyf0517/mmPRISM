from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

LANGUAGE_METRIC_PROTOCOL = "mmprism.language_metric.character_v1"


class LanguageMetricError(ValueError):
    """Raised when language metric inputs violate the protocol."""


def _normalized_text(value: str, *, role: str) -> str:
    if not isinstance(value, str):
        raise LanguageMetricError(f"{role} must be text")
    normalized = value.strip()
    if role == "reference" and not normalized:
        raise LanguageMetricError("reference must not be empty after stripping")
    return normalized


def character_edit_distance(reference: str, prediction: str) -> int:
    """Compute Unicode code-point Levenshtein distance using linear memory."""

    source = _normalized_text(reference, role="reference")
    target = _normalized_text(prediction, role="prediction")
    previous = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_character in enumerate(target, start=1):
            substitution = previous[target_index - 1] + (
                source_character != target_character
            )
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class LanguageSampleMetric:
    exact_match: bool
    character_edit_distance: int
    reference_character_count: int
    prediction_character_count: int


class LanguageMetricAccumulator:
    """Count-weighted exact match and character error rate accumulator."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.exact_match_count = 0
        self.character_edit_distance_sum = 0
        self.reference_character_count = 0
        self.prediction_character_count = 0

    def update(
        self, references: tuple[str, ...], predictions: tuple[str, ...]
    ) -> tuple[LanguageSampleMetric, ...]:
        if not references or len(references) != len(predictions):
            raise LanguageMetricError(
                "references and predictions must contain the same non-zero sample count"
            )
        metrics: list[LanguageSampleMetric] = []
        for reference, prediction in zip(references, predictions, strict=True):
            normalized_reference = _normalized_text(reference, role="reference")
            normalized_prediction = _normalized_text(prediction, role="prediction")
            distance = character_edit_distance(normalized_reference, normalized_prediction)
            exact = normalized_reference == normalized_prediction
            metric = LanguageSampleMetric(
                exact_match=exact,
                character_edit_distance=distance,
                reference_character_count=len(normalized_reference),
                prediction_character_count=len(normalized_prediction),
            )
            metrics.append(metric)
            self.sample_count += 1
            self.exact_match_count += int(exact)
            self.character_edit_distance_sum += distance
            self.reference_character_count += len(normalized_reference)
            self.prediction_character_count += len(normalized_prediction)
        return tuple(metrics)

    def values(self) -> dict[str, int | float]:
        if self.sample_count == 0 or self.reference_character_count == 0:
            raise LanguageMetricError("language metrics require at least one non-empty reference")
        return {
            "exact_match": self.exact_match_count / self.sample_count,
            "character_error_rate": (
                self.character_edit_distance_sum / self.reference_character_count
            ),
            "exact_match_count": self.exact_match_count,
            "character_edit_distance_sum": self.character_edit_distance_sum,
            "reference_character_count": self.reference_character_count,
            "prediction_character_count": self.prediction_character_count,
        }

    def state_dict(self) -> dict[str, int]:
        return {
            "sample_count": self.sample_count,
            "exact_match_count": self.exact_match_count,
            "character_edit_distance_sum": self.character_edit_distance_sum,
            "reference_character_count": self.reference_character_count,
            "prediction_character_count": self.prediction_character_count,
        }

    def merge_state(self, state: Mapping[str, int]) -> None:
        expected = {
            "sample_count",
            "exact_match_count",
            "character_edit_distance_sum",
            "reference_character_count",
            "prediction_character_count",
        }
        if set(state) != expected or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in state.values()
        ):
            raise LanguageMetricError("language metric state violates the protocol")
        if state["exact_match_count"] > state["sample_count"]:
            raise LanguageMetricError("language metric state has too many exact matches")
        self.sample_count += state["sample_count"]
        self.exact_match_count += state["exact_match_count"]
        self.character_edit_distance_sum += state["character_edit_distance_sum"]
        self.reference_character_count += state["reference_character_count"]
        self.prediction_character_count += state["prediction_character_count"]
