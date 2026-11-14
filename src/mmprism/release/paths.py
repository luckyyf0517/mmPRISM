"""Resolve release resources from explicit roots and validated relative layouts."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath


class PathConfigurationError(ValueError):
    """A root or logical resource does not satisfy the release path contract."""


_SOURCE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = (
    _SOURCE_ROOT
    if (_SOURCE_ROOT / "configs/data_layout.toml").is_file()
    else Path(__file__).resolve().parent / "resources"
)


class Paths:
    """CLI data root > environment roots > configs/paths.toml."""

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        config_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config_dir = config_dir or PACKAGE_ROOT / "configs"
        self.environ = os.environ if environ is None else environ
        self.data_root = data_root
        config_file = self.config_dir / "paths.toml"
        self.roots = self._read(config_file) if config_file.is_file() else {}
        self.layout = self._read(self.config_dir / "data_layout.toml")
        for group, entries in self.layout.items():
            if not isinstance(entries, dict):
                raise PathConfigurationError(f"data_layout.toml: {group} must be a table")
            for key, value in entries.items():
                self._relative(value, f"{group}.{key}")

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            with path.open("rb") as stream:
                return tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise PathConfigurationError(
                f"Cannot read path configuration {path}: {error}"
            ) from error

    @staticmethod
    def _relative(value: object, key: str) -> Path:
        if not isinstance(value, str) or not value:
            raise PathConfigurationError(f"Layout {key} must be a nonempty relative path")
        windows = PureWindowsPath(value)
        path = Path(value)
        if path.is_absolute() or windows.is_absolute() or windows.drive:
            raise PathConfigurationError(f"Layout {key} must be relative; got {value!r}")
        if ".." in path.parts or ".." in windows.parts:
            raise PathConfigurationError(f"Layout {key} must not contain '..'; got {value!r}")
        return path

    def root(self, key: str) -> Path:
        if key not in {"data_root", "weights_root", "assets_root", "output_root"}:
            raise PathConfigurationError(f"Unknown root {key!r}")
        variable = f"MMPRISM_{key.upper()}"
        cli = self.data_root if key == "data_root" else None
        value = cli or self.environ.get(variable) or self.roots.get(key)
        if value is not None and not isinstance(value, (str, Path)):
            raise PathConfigurationError(f"Root {key} must be a path string")
        if not value or str(value).startswith("/path/to/"):
            option = "--data-root, " if key == "data_root" else ""
            raise PathConfigurationError(
                f"Missing {key}: set {option}{variable}, or {key} in configs/paths.toml "
                "(copy configs/paths.example.toml first)."
            )
        return Path(value).expanduser().resolve()

    def resolve(self, key: str, **fields: str) -> Path:
        if key.endswith("_root") and "." not in key:
            return self.root(key)
        group, separator, name = key.partition(".")
        if not separator:
            candidates = [table for table, values in self.layout.items() if key in values]
            if len(candidates) != 1:
                raise PathConfigurationError(f"Unknown or ambiguous resource {key!r}")
            group, name = candidates[0], key
        try:
            template = self.layout[group][name]
        except KeyError as error:
            raise PathConfigurationError(
                f"Unknown resource {key!r} in configs/data_layout.toml"
            ) from error
        try:
            relative = self._relative(template.format(**fields), key)
        except KeyError as error:
            raise PathConfigurationError(
                f"Resource {key} needs layout field {error.args[0]!r}"
            ) from error
        root_key = {
            "weights": "weights_root",
            "assets": "assets_root",
            "outputs": "output_root",
        }.get(group, "data_root")
        root = self.root(root_key)
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise PathConfigurationError(f"Resource {key} resolves outside {root_key}")
        return result


_default: Paths | None = None


def configure(*, data_root: str | Path | None = None, config_dir: Path | None = None) -> Paths:
    """Install the path context shared by entry points and training functions."""
    global _default
    _default = Paths(data_root=data_root, config_dir=config_dir)
    return _default


def resolve(key: str, **fields: str) -> Path:
    """Resolve a logical layout key or a root; never accept an arbitrary path."""
    global _default
    if _default is None:
        _default = Paths()
    return _default.resolve(key, **fields)
