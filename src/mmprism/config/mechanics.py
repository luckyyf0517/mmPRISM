from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mmprism.config.schema import ConfigError

_STABLE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def reject_unknown(
    payload: Mapping[str, Any], allowed: set[str] | frozenset[str], location: str
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        raise ConfigError(f"Unknown keys in {location}: {rendered}")


def require_text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be non-empty text")
    return value.strip()


def require_integer(
    value: object,
    location: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{location} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{location} must be <= {maximum}")
    return value


def require_schema_version(value: object, expected: str, location: str) -> str:
    version = require_text(value, location)
    if version != expected:
        raise ConfigError(f"{location} must be {expected}")
    return version


def require_stable_id(value: object, location: str) -> str:
    if not isinstance(value, str) or _STABLE_ID_PATTERN.fullmatch(value) is None:
        raise ConfigError(f"{location} must be a stable lowercase ID")
    return value


def require_sha256(value: object, location: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ConfigError(f"{location} must be a lowercase SHA-256 digest")
    return value


def canonical_json_bytes(payload: object) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"payload is not canonical JSON: {error}") from error
    return serialized.encode("utf-8")


def payload_fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def resolve_project_path(project_root: str | Path, project_relative_path: str | Path) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(project_relative_path)
    if candidate.is_absolute():
        raise ConfigError("project path must be relative to the project root")
    if not candidate.parts or candidate == Path("."):
        raise ConfigError("project path must not be empty")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ConfigError(
            f"project path escapes the project root: {project_relative_path}"
        ) from error
    return resolved
