"""Materialize a contextual Daily frame export for the WaveLLM Parquet reader.

The OmniHand producer owns frame-level ``P_local_v1`` and contextual feature
rows.  This module groups those already-computed rows into ordinary WaveLLM
episode samples; it never reruns MeshSim, CubeNet, or Temporal inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


class CslDailyContextualPFWaveLLMDeliveryError(RuntimeError):
    """Raised when exported contextual frames cannot form WaveLLM episodes."""


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CslDailyContextualPFWaveLLMDeliveryError(f"{location} must be a mapping")
    return value


def validate_csl_daily_contextual_pf_wavellm_payload_contract(
    delivery_root: str | Path,
    *,
    pose_contract: str,
    feature_identity: str,
) -> dict[str, object]:
    """Validate one contextual canonical payload before a named consumer uses it."""

    root = Path(delivery_root).expanduser().resolve()
    manifest_path = root / "source_manifest.jsonl"
    if not manifest_path.is_file():
        raise CslDailyContextualPFWaveLLMDeliveryError(
            f"contextual canonical source manifest is missing: {manifest_path}"
        )
    sample_count = 0
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    continue
                row = _mapping(json.loads(raw), f"source manifest line {line_number}")
                provenance = _mapping(
                    row.get("provenance"), f"source manifest line {line_number}.provenance"
                )
                contextual = _mapping(
                    provenance.get("contextual_source"),
                    f"source manifest line {line_number}.contextual_source",
                )
                if contextual.get("pose_contract") != pose_contract:
                    raise CslDailyContextualPFWaveLLMDeliveryError(
                        f"source manifest line {line_number} pose contract does not match "
                        f"{pose_contract}"
                    )
                if contextual.get("feature_identity") != feature_identity:
                    raise CslDailyContextualPFWaveLLMDeliveryError(
                        f"source manifest line {line_number} feature identity does not match "
                        f"{feature_identity}"
                    )
                sample_count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslDailyContextualPFWaveLLMDeliveryError(
            f"unable to read contextual canonical source manifest: {error}"
        ) from error
    if sample_count < 1:
        raise CslDailyContextualPFWaveLLMDeliveryError(
            "contextual canonical source manifest is empty"
        )
    return {
        "pose_contract": pose_contract,
        "feature_identity": feature_identity,
        "sample_count": sample_count,
    }
