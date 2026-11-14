from __future__ import annotations

TRANSLATION_BLEU_TOKENIZER = "13a"


class LanguageMetricError(ValueError):
    """Raised when language metric inputs violate the protocol."""


def _official_character_text(value: str, *, role: str) -> str:
    """UniSign-official lexical normalization (fine_tuning.py:283-285).

    References additionally normalize the full-width comma and question mark
    to their ASCII forms before character segmentation; predictions are only
    stripped of spaces/newlines. Both sides are then space-joined per
    Unicode character.
    """

    if not isinstance(value, str):
        raise LanguageMetricError(f"{role} must be text")
    if role == "reference":
        value = value.replace("，", ",").replace("？", "?")
    character_text = value.replace(" ", "").replace("\n", "")
    if not character_text and role == "reference":
        raise LanguageMetricError(f"{role} is empty after official character metric preprocessing")
    return " ".join(character_text)


def _summary_sentences(character_text: str) -> tuple[tuple[str, ...], ...]:
    """Match pltrdy Rouge's period split and normalized space tokenization."""

    sentences = tuple(
        tuple(" ".join(segment.split()).split(" "))
        for segment in character_text.split(".")
        if len(segment) > 0
    )
    if not sentences:
        raise LanguageMetricError("official ROUGE-L requires at least one sentence")
    return sentences


def _lcs_token_set(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> set[str]:
    """Return the exclusive-token LCS used by pltrdy Rouge."""

    rows = len(reference)
    columns = len(hypothesis)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for reference_index in range(1, rows + 1):
        for hypothesis_index in range(1, columns + 1):
            if reference[reference_index - 1] == hypothesis[hypothesis_index - 1]:
                table[reference_index][hypothesis_index] = (
                    table[reference_index - 1][hypothesis_index - 1] + 1
                )
            else:
                table[reference_index][hypothesis_index] = max(
                    table[reference_index - 1][hypothesis_index],
                    table[reference_index][hypothesis_index - 1],
                )

    tokens: list[str] = []
    reference_index = rows
    hypothesis_index = columns
    while reference_index > 0 and hypothesis_index > 0:
        if reference[reference_index - 1] == hypothesis[hypothesis_index - 1]:
            tokens.append(reference[reference_index - 1])
            reference_index -= 1
            hypothesis_index -= 1
        elif (
            table[reference_index - 1][hypothesis_index]
            > table[reference_index][hypothesis_index - 1]
        ):
            reference_index -= 1
        else:
            hypothesis_index -= 1
    return set(tokens)


def _official_rouge_l(reference: str, prediction: str) -> float:
    """Port the summary-level exclusive ROUGE-L behavior of pltrdy Rouge."""

    reference_sentences = _summary_sentences(reference)
    # Blank or period-only predictions have no summary sentence; references still validate above.
    if not prediction.replace(".", "").strip():
        return 0.0
    prediction_sentences = _summary_sentences(prediction)
    reference_tokens = {token for sentence in reference_sentences for token in sentence}
    prediction_tokens = {token for sentence in prediction_sentences for token in sentence}
    if not reference_tokens or not prediction_tokens:
        raise LanguageMetricError("official ROUGE-L requires non-empty token sets")

    union: set[str] = set()
    lcs_count = 0
    for reference_sentence in reference_sentences:
        next_union = set(union)
        for prediction_sentence in prediction_sentences:
            next_union.update(_lcs_token_set(reference_sentence, prediction_sentence))
        lcs_count += len(next_union) - len(union)
        union = next_union

    recall = lcs_count / len(reference_tokens)
    precision = lcs_count / len(prediction_tokens)
    return 2.0 * ((precision * recall) / (precision + recall + 1e-8))
