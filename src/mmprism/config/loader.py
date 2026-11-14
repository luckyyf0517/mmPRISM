import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml

from mmprism.config.schema import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

type EnvironmentPolicy = Literal["preserve", "expand", "forbid"]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(f"Environment variable {name} is required by the configuration")

    return _ENV_PATTERN.sub(replace, value)


def reject_environment_expressions(value: object, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_environment_expressions(key, f"{location} key")
            reject_environment_expressions(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_environment_expressions(item, f"{location}[{index}]")
        return
    if isinstance(value, str) and _ENV_PATTERN.search(value) is not None:
        raise ConfigError(f"Environment expressions are forbidden in {location}")


def load_yaml_mapping(
    path: str | Path,
    *,
    environment: EnvironmentPolicy = "preserve",
) -> dict[str, Any]:
    """Load one strict YAML mapping under an explicit environment policy."""

    if environment not in {"preserve", "expand", "forbid"}:
        raise ConfigError(f"Unsupported environment policy: {environment!r}")
    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"Unable to load YAML mapping {config_path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    try:
        processed: object = raw
        if environment == "expand":
            processed = expand_environment(raw)
        elif environment == "forbid":
            reject_environment_expressions(raw, str(config_path))
    except ConfigError as error:
        raise ConfigError(f"Invalid configuration {config_path}: {error}") from error
    if not isinstance(processed, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    result: dict[str, Any] = {}
    for key, item in processed.items():
        if not isinstance(key, str):
            raise ConfigError(f"Configuration root keys must be strings: {config_path}")
        result[key] = item
    return result
