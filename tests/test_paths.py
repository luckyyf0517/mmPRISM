"""Fresh release path-contract checks; no dataset or checkpoint is needed."""

from pathlib import Path

import pytest

from mmprism.release.paths import PathConfigurationError, Paths


def configuration(tmp_path: Path, layout: str = '[pack]\nsplits = "splits/list.json"\n') -> Path:
    config = tmp_path / "configs"
    config.mkdir()
    (config / "data_layout.toml").write_text(layout)
    (config / "paths.toml").write_text(
        f'data_root = "{tmp_path}/file-data"\n'
        f'weights_root = "{tmp_path}/file-weights"\n'
        f'output_root = "{tmp_path}/file-output"\n'
    )
    return config


def test_precedence(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    roots = {
        "MMPRISM_DATA_ROOT": str(tmp_path / "env-data"),
        "MMPRISM_WEIGHTS_ROOT": str(tmp_path / "env-weights"),
        "MMPRISM_OUTPUT_ROOT": str(tmp_path / "env-output"),
    }
    paths = Paths(config_dir=config, data_root=tmp_path / "cli-data", environ=roots)
    assert paths.resolve("pack.splits") == tmp_path / "cli-data/splits/list.json"
    assert paths.resolve("weights_root") == tmp_path / "env-weights"
    assert paths.resolve("output_root") == tmp_path / "env-output"
    assert Paths(config_dir=config, environ=roots).root("data_root") == tmp_path / "env-data"
    assert Paths(config_dir=config, environ={}).root("data_root") == tmp_path / "file-data"


@pytest.mark.parametrize(
    "value", ["../escape", "safe/../escape", "/absolute", "C:/absolute", "safe\\\\..\\\\escape"]
)
def test_invalid_layout(tmp_path: Path, value: str) -> None:
    config = configuration(tmp_path, f"[pack]\nsplits = '{value}'\n")
    with pytest.raises(PathConfigurationError, match="relative|must not contain"):
        Paths(config_dir=config, environ={})


def test_missing_root_and_key(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    (config / "paths.toml").unlink()
    paths = Paths(config_dir=config, environ={})
    with pytest.raises(
        PathConfigurationError, match="Missing data_root: set --data-root, MMPRISM_DATA_ROOT"
    ):
        paths.resolve("pack.splits")
    with pytest.raises(PathConfigurationError, match="Unknown resource 'pack.missing'"):
        paths.resolve("pack.missing")


def test_formatted_layout_rejects_escape(tmp_path: Path) -> None:
    config = configuration(tmp_path, '[pack]\npool = "pool/{signer}"\n')
    with pytest.raises(PathConfigurationError, match="must not contain"):
        Paths(config_dir=config, environ={}).resolve("pack.pool", signer="../escape")


@pytest.mark.parametrize("value", ["123", "true", '["data"]'])
def test_invalid_root_type_has_configuration_error(tmp_path: Path, value: str) -> None:
    config = configuration(tmp_path)
    (config / "paths.toml").write_text(f"data_root = {value}\n")
    with pytest.raises(PathConfigurationError, match="data_root must be a path string"):
        Paths(config_dir=config, environ={}).root("data_root")
