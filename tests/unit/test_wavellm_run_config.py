from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mmprism.training import WaveLLMRunError, load_wavellm_run_config


def _payload() -> dict[str, object]:
    return {
        "schema_version": "mmprism.wavellm_run.v1",
        "model": {
            "asset_id": "tiny_mt5",
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
        "data": {
            "batch_size": 2,
            "num_workers": 0,
            "verify_checksums": True,
            "shuffle": True,
            "max_frames": 4,
            "prompt": "Translate signs:",
            "max_prompt_length": 16,
            "max_target_length": 16,
        },
        "optimization": {
            "epochs": 2,
            "max_steps": 3,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "beta1": 0.9,
            "beta2": 0.98,
            "gradient_clip_norm": 1.0,
        },
        "generation": {"max_new_tokens": 8, "num_beams": 2},
        "evaluation": {"save_references": True},
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_wavellm_run_config_is_strict_and_stable(tmp_path: Path) -> None:
    config = load_wavellm_run_config(_write(tmp_path / "run.yaml", _payload()))

    assert config.model.asset_id == "tiny_mt5"
    assert config.data.max_frames == 4
    assert config.optimization.max_steps == 3
    assert config.generation.num_beams == 2
    assert config.evaluation.save_references is True
    assert len(config.fingerprint) == 64
    assert len(config.model_fingerprint) == 64
    assert len(config.training_fingerprint) == 64
    assert config.to_dict()["schema_version"] == "mmprism.wavellm_run.v1"

    extended_payload = _payload()
    extended_optimization = extended_payload["optimization"]
    assert isinstance(extended_optimization, dict)
    extended_optimization["epochs"] = 5
    extended_optimization["max_steps"] = None
    extended = load_wavellm_run_config(
        _write(tmp_path / "extended.yaml", extended_payload)
    )
    assert extended.fingerprint != config.fingerprint
    assert extended.training_fingerprint == config.training_fingerprint

    changed_payload = _payload()
    changed_optimization = changed_payload["optimization"]
    assert isinstance(changed_optimization, dict)
    changed_optimization["learning_rate"] = 0.002
    changed = load_wavellm_run_config(_write(tmp_path / "changed.yaml", changed_payload))
    assert changed.training_fingerprint != config.training_fingerprint


@pytest.mark.parametrize(
    ("location", "key", "value", "message"),
    [
        ("root", "unexpected", True, "Unknown keys"),
        ("data", "max_frames", 0, "max_frames"),
        ("optimization", "max_steps", 0, "max_steps"),
        ("generation", "num_beams", 0, "num_beams"),
        ("model", "freeze_language_model", "yes", "boolean"),
    ],
)
def test_wavellm_run_config_rejects_invalid_values(
    tmp_path: Path, location: str, key: str, value: object, message: str
) -> None:
    payload = _payload()
    target = payload if location == "root" else payload[location]
    assert isinstance(target, dict)
    target[key] = value
    with pytest.raises(WaveLLMRunError, match=message):
        load_wavellm_run_config(_write(tmp_path / f"{location}-{key}.yaml", payload))
