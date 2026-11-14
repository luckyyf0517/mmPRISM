from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class ConditionalNegativePairs:
    target_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]
    length_differences: tuple[int, ...]


def select_conditional_negative_pairs(
    *,
    sample_ids: Sequence[str],
    captions: Sequence[str],
    group_ids: Sequence[str],
    valid_frame_counts: Sequence[int],
    tie_seed: int,
) -> ConditionalNegativePairs:
    """Choose one closest-length legal in-batch pose negative per target."""

    count = len(sample_ids)
    if not (
        len(captions) == count and len(group_ids) == count and len(valid_frame_counts) == count
    ):
        raise ValueError("conditional-negative fields must have identical lengths")
    if tie_seed < 0:
        raise ValueError("tie_seed must be non-negative")
    normalized = tuple(text.strip() for text in captions)
    targets: list[int] = []
    negatives: list[int] = []
    differences: list[int] = []
    for target in range(count):
        candidates = [
            candidate
            for candidate in range(count)
            if candidate != target
            and normalized[candidate] != normalized[target]
            and group_ids[candidate] != group_ids[target]
        ]
        if not candidates:
            continue
        difference = min(
            abs(valid_frame_counts[candidate] - valid_frame_counts[target])
            for candidate in candidates
        )
        tied = sorted(
            (
                candidate
                for candidate in candidates
                if abs(valid_frame_counts[candidate] - valid_frame_counts[target]) == difference
            ),
            key=lambda candidate: (sample_ids[candidate], candidate),
        )
        digest = hashlib.blake2b(
            f"{tie_seed}\0{sample_ids[target]}".encode(), digest_size=8
        ).digest()
        selected = tied[int.from_bytes(digest, "little") % len(tied)]
        targets.append(target)
        negatives.append(selected)
        differences.append(difference)
    return ConditionalNegativePairs(tuple(targets), tuple(negatives), tuple(differences))


def sequence_log_probability(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean unsmoothed log probability over non-padding target tokens."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must align as [batch,target,vocabulary]")
    valid = labels.ne(-100)
    if not bool(torch.all(valid.any(dim=1))):
        raise ValueError("every conditional target must contain a valid token")
    token_log_probability = -functional.cross_entropy(
        logits.float().transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )
    return (token_log_probability * valid).sum(dim=1) / valid.sum(dim=1)


def conditional_pair_loss(positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
    if positive_scores.shape != negative_scores.shape or positive_scores.ndim != 1:
        raise ValueError("positive and negative scores must be matching vectors")
    if positive_scores.numel() < 1:
        raise ValueError("conditional pair loss requires at least one pair")
    return functional.softplus(negative_scores - positive_scores).mean()
