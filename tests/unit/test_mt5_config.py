from __future__ import annotations

from pathlib import Path

import pytest

from mmprism.training import MT5SmokeConfig, MT5SmokeError, load_mt5_smoke_config


def _payload() -> dict[str, object]:
    return {
        "schema_version": "mmprism.mt5_smoke.v1",
        "smoke_id": "fixture_mt5_v1",
        "model": {
            "asset_id": "mt5_base",
            "hidden_size": 32,
            "radar_feature_dim": 16,
            "joint_count": 24,
            "coordinate_dim": 3,
            "pose_channels": [8, 16],
            "temporal_kernel_size": 3,
            "dropout": 0.0,
            "label_smoothing": 0.0,
            "freeze_language_model": True,
        },
        "batch": {
            "batch_size": 2,
            "frame_count": 3,
            "prompt": "translate:",
            "targets": ["one", "two"],
            "max_target_length": 8,
        },
        "optimization": {
            "steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "beta1": 0.9,
            "beta2": 0.98,
        },
        "generation": {"max_new_tokens": 4, "num_beams": 2},
        "runtime": {"seed": 17, "dtype": "float32", "deterministic": True},
    }


def test_mt5_smoke_config_is_strict_and_stable() -> None:
    config = MT5SmokeConfig.from_mapping(_payload())

    assert config.model.pose_channels == (8, 16)
    assert config.batch.targets == ("one", "two")
    assert config.optimization.steps == 2
    assert config.model.label_smoothing == 0
    assert len(config.fingerprint) == 64
    assert config.fingerprint == MT5SmokeConfig.from_mapping(config.to_dict()).fingerprint


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("model", "surprise", True, "Unknown keys"),
        ("model", "temporal_kernel_size", 4, "must be odd"),
        ("model", "label_smoothing", 1.0, "must be < 1"),
        ("batch", "targets", ["one"], "one non-empty string per sample"),
        ("runtime", "dtype", "float16", "float32 or bfloat16"),
    ],
)
def test_mt5_smoke_config_rejects_invalid_values(
    section: str, key: str, value: object, message: str
) -> None:
    payload = _payload()
    section_payload = dict(payload[section])  # type: ignore[arg-type]
    section_payload[key] = value
    payload[section] = section_payload

    with pytest.raises(MT5SmokeError, match=message):
        MT5SmokeConfig.from_mapping(payload)


def test_versioned_mt5_configs_load_without_environment_paths() -> None:
    project_root = Path(__file__).resolve().parents[2]
    smoke = load_mt5_smoke_config(project_root / "configs/examples/mt5_smoke.yaml")

    assert smoke.model.asset_id == "mt5_base"
    assert smoke.model.hidden_size == 768
    assert smoke.model.radar_feature_dim == 1024
    assert smoke.model.label_smoothing == 0
