from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from mmprism.training import (
    OmniHandSmokeConfig,
    OmniHandSmokeError,
    load_omnihand_smoke_config,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "mmprism.omnihand_smoke.v1",
        "smoke_id": "fixture_omnihand_v1",
        "model": {
            "spatial": {
                "in_channels": 4,
                "stem_channels": 8,
                "stage_channels": [8, 16],
                "stage_depths": [1, 1],
                "channel_attention": True,
                "spatial_attention": True,
                "se_attention": True,
                "use_pafpn": True,
                "fpn_channels": 16,
            },
            "temporal": {
                "max_frames": 3,
                "layers": 2,
                "heads": 4,
                "feedforward_dim": 32,
                "dropout": 0.0,
            },
            "joint_count": 24,
            "coordinate_dim": 3,
        },
        "batch": {
            "batch_size": 2,
            "frame_count": 3,
            "spatial_shape": [8, 6, 4],
        },
        "optimization": {
            "steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "beta1": 0.9,
            "beta2": 0.98,
        },
        "metrics": {"pck_threshold_mm": 40.0},
        "runtime": {"seed": 17, "dtype": "float32", "deterministic": True},
    }


def test_omnihand_smoke_config_is_strict_and_stable() -> None:
    config = OmniHandSmokeConfig.from_mapping(_payload())

    assert config.model.spatial.stage_channels == (8, 16)
    assert config.model.temporal.heads == 4
    assert config.batch.spatial_shape == (8, 6, 4)
    assert len(config.fingerprint) == 64
    assert config.fingerprint == OmniHandSmokeConfig.from_mapping(config.to_dict()).fingerprint


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("top", "surprise", True), "Unknown keys"),
        (("spatial", "stage_depths", [1]), "must have equal length"),
        (("temporal", "heads", 3), "must divide"),
        (("batch", "batch_size", 1), "requires batch_size"),
        (("optimization", "learning_rate", float("nan")), "must be finite"),
        (("runtime", "dtype", "float16"), "float32 or bfloat16"),
    ],
)
def test_omnihand_smoke_config_rejects_invalid_values(
    mutation: tuple[str, str, object], message: str
) -> None:
    payload = deepcopy(_payload())
    section, key, value = mutation
    if section == "top":
        payload[key] = value
    elif section in {"spatial", "temporal"}:
        model = payload["model"]
        assert isinstance(model, dict)
        nested = model[section]
        assert isinstance(nested, dict)
        nested[key] = value
    else:
        nested = payload[section]
        assert isinstance(nested, dict)
        nested[key] = value

    with pytest.raises(OmniHandSmokeError, match=message):
        OmniHandSmokeConfig.from_mapping(payload)


def test_versioned_omnihand_smoke_config_matches_manuscript_temporal_contract() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_omnihand_smoke_config(project_root / "configs/examples/omnihand_smoke.yaml")

    assert config.model.temporal.max_frames == 10
    assert config.model.temporal.layers == 8
    assert config.model.temporal.heads == 16
    assert config.model.spatial.channel_attention is True
    assert config.model.spatial.spatial_attention is True
    assert config.model.spatial.se_attention is True
    assert config.optimization.steps == 2
