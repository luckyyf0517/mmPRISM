from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from mmprism.training import OmniHandRunConfig, OmniHandRunError, load_omnihand_run_config


def _payload() -> dict[str, object]:
    return {
        "schema_version": "mmprism.omnihand_run.v1",
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
        "data": {
            "batch_size": 2,
            "num_workers": 0,
            "verify_checksums": True,
            "shuffle": True,
        },
        "optimization": {
            "epochs": 2,
            "max_steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "beta1": 0.9,
            "beta2": 0.98,
            "gradient_clip_norm": 1.0,
        },
        "evaluation": {"pck_threshold_mm": 40.0, "save_targets": True},
    }


def test_omnihand_run_config_is_strict_and_stable() -> None:
    config = OmniHandRunConfig.from_mapping(_payload())

    assert config.model.temporal.max_frames == 3
    assert config.data.num_workers == 0
    assert config.optimization.max_steps == 2
    assert config.evaluation.save_targets is True
    assert len(config.fingerprint) == 64
    assert len(config.model_fingerprint) == 64
    assert config.fingerprint == OmniHandRunConfig.from_mapping(config.to_dict()).fingerprint


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("top", "unknown", True, "Unknown keys"),
        ("data", "num_workers", -1, "integer >= 0"),
        ("optimization", "max_steps", 0, "null or a positive integer"),
        ("optimization", "gradient_clip_norm", 0.0, "must be > 0"),
        ("evaluation", "pck_threshold_mm", float("nan"), "must be finite"),
        ("model", "coordinate_dim", 2, "canonical"),
    ],
)
def test_omnihand_run_config_rejects_invalid_values(
    section: str, key: str, value: object, message: str
) -> None:
    payload = deepcopy(_payload())
    if section == "top":
        payload[key] = value
    else:
        nested = payload[section]
        assert isinstance(nested, dict)
        nested[key] = value

    with pytest.raises(OmniHandRunError, match=message):
        OmniHandRunConfig.from_mapping(payload)


def test_omnihand_run_allows_unbounded_training_steps() -> None:
    payload = _payload()
    optimization = payload["optimization"]
    assert isinstance(optimization, dict)
    optimization["max_steps"] = None

    assert OmniHandRunConfig.from_mapping(payload).optimization.max_steps is None


def test_versioned_omnihand_run_config_matches_formal_smoke_contract() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_omnihand_run_config(project_root / "configs/examples/omnihand_train_smoke.yaml")

    assert config.model.temporal.max_frames == 10
    assert config.model.temporal.layers == 8
    assert config.optimization.max_steps == 2
    assert config.data.verify_checksums is True
    assert config.evaluation.save_targets is True
