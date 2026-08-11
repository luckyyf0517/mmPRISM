from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mmprism.assets import (
    MODEL_ASSET_COLLECTION_SCHEMA,
    ModelAssetError,
    ModelAssetSetConfig,
    download_model_assets,
    load_model_asset_config,
    plan_model_assets,
    run_model_asset_smoke,
    verify_model_assets,
)
from mmprism.cli import main

REVISION_A = "a" * 40
REVISION_B = "b" * 40


class FakeHubClient:
    def __init__(self, snapshot_root: Path, resolved: dict[str, str]) -> None:
        self.snapshot_root = snapshot_root
        self.resolved = resolved
        self.resolve_calls: list[tuple[str, str]] = []
        self.download_calls: list[tuple[str, str, tuple[str, ...], Path]] = []

    def resolve_revision(self, repo_id: str, revision: str) -> str:
        self.resolve_calls.append((repo_id, revision))
        return self.resolved[repo_id]

    def snapshot_download(
        self,
        repo_id: str,
        revision: str,
        required_files: list[str],
        cache_dir: Path,
    ) -> Path:
        self.download_calls.append(
            (repo_id, revision, tuple(required_files), cache_dir)
        )
        return self.snapshot_root / repo_id.replace("/", "--")


def _payload() -> dict[str, object]:
    return {
        "schema_version": "mmprism.model_asset_config.v1",
        "asset_set_id": "fixture_models_v1",
        "models": [
            {
                "asset_id": "simcse",
                "repo_id": "owner/simcse",
                "revision": REVISION_A,
                "destination": "semantic/simcse",
                "loader": "transformers_auto",
                "required_files": ["config.json", "weights/model.bin"],
            },
            {
                "asset_id": "sbert",
                "repo_id": "owner/sbert",
                "revision": REVISION_B,
                "destination": "semantic/sbert",
                "loader": "sentence_transformers",
                "required_files": ["1_Pooling/config.json", "model.safetensors"],
            },
        ],
    }


def _snapshots(root: Path) -> None:
    contents = {
        "owner--simcse/config.json": b'{"model_type":"bert"}\n',
        "owner--simcse/weights/model.bin": b"simcse-weights",
        "owner--sbert/1_Pooling/config.json": b'{"pooling_mode_mean_tokens":true}\n',
        "owner--sbert/model.safetensors": b"sbert-weights",
    }
    for relative, content in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _client(root: Path) -> FakeHubClient:
    _snapshots(root)
    return FakeHubClient(
        root,
        {"owner/simcse": REVISION_A, "owner/sbert": REVISION_B},
    )


def test_model_asset_config_is_strict_and_portable(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """\
schema_version: mmprism.model_asset_config.v1
asset_set_id: one_model_v1
models:
  - asset_id: encoder
    repo_id: owner/model
    revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    destination: encoders/model
    loader: transformers_auto
    required_files: [config.json, weights/model.bin]
""",
        encoding="utf-8",
    )

    config = load_model_asset_config(config_path)

    assert config.asset_set_id == "one_model_v1"
    assert config.models[0].destination.as_posix() == "encoders/model"
    assert config.models[0].required_files[1].as_posix() == "weights/model.bin"
    assert len(config.fingerprint) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        {"surprise": True},
        {"models": []},
        {
            "models": [
                {
                    "asset_id": "bad",
                    "repo_id": "owner/model",
                    "revision": "main",
                    "destination": "model",
                    "loader": "transformers_auto",
                    "required_files": ["config.json"],
                }
            ]
        },
        {
            "models": [
                {
                    "asset_id": "bad",
                    "repo_id": "owner/model",
                    "revision": REVISION_A,
                    "destination": "../outside",
                    "loader": "transformers_auto",
                    "required_files": ["config.json"],
                }
            ]
        },
    ],
)
def test_model_asset_config_rejects_unknown_or_unsafe_values(
    mutation: dict[str, object],
) -> None:
    payload = _payload()
    payload.update(mutation)
    with pytest.raises(ModelAssetError):
        ModelAssetSetConfig.from_mapping(payload)


def test_downloads_atomically_hashes_and_reuses_assets(tmp_path: Path) -> None:
    config = ModelAssetSetConfig.from_mapping(_payload())
    client = _client(tmp_path / "snapshots")
    output_root = tmp_path / "models"
    runtime = {
        "python": "3.12.11",
        "git": {"commit": "c" * 40, "dirty": False},
    }

    first = download_model_assets(
        config,
        output_root,
        hub_client=client,
        runtime_report=runtime,
        downloaded_at_utc="2026-08-11T18:00:00+00:00",
    )

    assert first["status"] == "ready"
    assert first["actions"] == {"simcse": "downloaded", "sbert": "downloaded"}
    assert len(client.resolve_calls) == 2
    assert len(client.download_calls) == 2
    collection_path = Path(first["collection_manifest"])
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    assert collection["schema_version"] == MODEL_ASSET_COLLECTION_SCHEMA
    assert collection["builder"]["git_commit"] == "c" * 40
    assert str(tmp_path) not in collection_path.read_text(encoding="utf-8")

    simcse_manifest_path = output_root / "semantic/simcse/mmprism_model_asset.json"
    simcse_manifest = json.loads(simcse_manifest_path.read_text(encoding="utf-8"))
    expected_digest = hashlib.sha256(b"simcse-weights").hexdigest()
    assert simcse_manifest["revision"] == REVISION_A
    assert simcse_manifest["files"][1]["sha256"] == expected_digest
    assert not list(output_root.rglob("*.staging.*"))

    second_client = FakeHubClient(tmp_path / "unused", {})
    second = download_model_assets(
        config,
        output_root,
        hub_client=second_client,
        runtime_report=runtime,
        downloaded_at_utc="2026-08-11T18:01:00+00:00",
    )

    assert second["actions"] == {"simcse": "reused", "sbert": "reused"}
    assert second_client.resolve_calls == []
    assert verify_model_assets(config, output_root)["status"] == "ready"


def test_refuses_corrupt_existing_asset_without_overwrite(tmp_path: Path) -> None:
    config = ModelAssetSetConfig.from_mapping(_payload())
    output_root = tmp_path / "models"
    download_model_assets(
        config,
        output_root,
        hub_client=_client(tmp_path / "snapshots"),
        downloaded_at_utc="2026-08-11T18:00:00+00:00",
    )
    corrupt_path = output_root / "semantic/sbert/model.safetensors"
    corrupt_path.write_bytes(b"corrupt")

    plan = plan_model_assets(config, output_root)

    assert plan["status"] == "incomplete"
    assert [model["state"] for model in plan["models"]] == ["ready", "invalid"]
    with pytest.raises(ModelAssetError, match="checksum mismatch"):
        download_model_assets(
            config,
            output_root,
            hub_client=FakeHubClient(tmp_path / "unused", {}),
        )
    assert corrupt_path.read_bytes() == b"corrupt"


def test_smoke_rejects_collection_manifest_drift_before_model_imports(
    tmp_path: Path,
) -> None:
    config = ModelAssetSetConfig.from_mapping(_payload())
    output_root = tmp_path / "models"
    result = download_model_assets(
        config,
        output_root,
        hub_client=_client(tmp_path / "snapshots"),
        downloaded_at_utc="2026-08-11T18:00:00+00:00",
    )
    collection_path = Path(result["collection_manifest"])
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["models"][0]["revision"] = "f" * 40
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(ModelAssetError, match="collection mismatch"):
        run_model_asset_smoke(config, output_root)


def test_rejects_remote_revision_mismatch_before_download(tmp_path: Path) -> None:
    config = ModelAssetSetConfig.from_mapping(_payload())
    client = _client(tmp_path / "snapshots")
    client.resolved["owner/simcse"] = "f" * 40

    with pytest.raises(ModelAssetError, match="Resolved revision mismatch"):
        download_model_assets(config, tmp_path / "models", hub_client=client)

    assert client.download_calls == []
    assert not (tmp_path / "models/semantic/simcse").exists()


def test_models_plan_cli_is_dependency_light(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(json.dumps(_payload()), encoding="utf-8")

    exit_code = main(
        [
            "models-plan",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "models"),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "incomplete"
    assert {model["state"] for model in payload["models"]} == {"missing"}
