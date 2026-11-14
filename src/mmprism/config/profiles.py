from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from mmprism.config.loader import load_yaml_mapping, reject_environment_expressions
from mmprism.config.mechanics import (
    payload_fingerprint,
    reject_unknown,
    require_mapping,
    require_stable_id,
    require_text,
    resolve_project_path,
)
from mmprism.config.schema import ConfigError

_PROFILE_REF_FIELDS = frozenset({"path", "expected_kind", "expected_id"})

_PROFILE_ENVELOPE_FIELDS = frozenset({"schema_version", "profile_kind", "profile_id", "parameters"})


def _freeze_json_value(value: object, location: str) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"{location} keys must be strings")
            frozen[key] = _freeze_json_value(item, f"{location}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{location} must contain only finite numbers")
        return value
    raise ConfigError(f"{location} contains a non-JSON value of type {type(value).__name__}")


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProfileRef:
    path: PurePosixPath
    expected_kind: str
    expected_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, PurePosixPath):
            raise ConfigError("profile reference path must be a PurePosixPath")
        raw_path = self.path.as_posix()
        if self.path.is_absolute() or self.path == PurePosixPath(".") or ".." in self.path.parts:
            raise ConfigError("profile reference path must be project-root-relative")
        if "\\" in raw_path:
            raise ConfigError("profile reference path must be a POSIX path")
        object.__setattr__(
            self,
            "expected_kind",
            require_stable_id(self.expected_kind, "profile reference expected_kind"),
        )
        object.__setattr__(
            self,
            "expected_id",
            require_stable_id(self.expected_id, "profile reference expected_id"),
        )

    @classmethod
    def from_mapping(cls, value: object, location: str = "profile reference") -> ProfileRef:
        payload = require_mapping(value, location)
        reject_environment_expressions(payload, location)
        reject_unknown(payload, _PROFILE_REF_FIELDS, location)
        raw_path = require_text(payload.get("path"), f"{location}.path")
        if "\\" in raw_path:
            raise ConfigError(f"{location}.path must be a POSIX path")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
            raise ConfigError(f"{location}.path must be project-root-relative")
        if path.as_posix() != raw_path:
            raise ConfigError(f"{location}.path must use canonical POSIX syntax")
        return cls(
            path=path,
            expected_kind=require_stable_id(
                payload.get("expected_kind"), f"{location}.expected_kind"
            ),
            expected_id=require_stable_id(payload.get("expected_id"), f"{location}.expected_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path.as_posix(),
            "expected_kind": self.expected_kind,
            "expected_id": self.expected_id,
        }


@dataclass(frozen=True, slots=True)
class ProfileEnvelope:
    schema_version: str
    profile_kind: str
    profile_id: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        schema_version = require_text(self.schema_version, "profile envelope.schema_version")
        profile_kind = require_stable_id(self.profile_kind, "profile envelope.profile_kind")
        profile_id = require_stable_id(self.profile_id, "profile envelope.profile_id")
        parameters = require_mapping(self.parameters, "profile envelope.parameters")
        reject_environment_expressions(parameters, "profile envelope.parameters")
        frozen_parameters = _freeze_json_value(parameters, "profile envelope.parameters")
        if not isinstance(frozen_parameters, Mapping):
            raise TypeError("mapping freeze must preserve the root mapping")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "profile_kind", profile_kind)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "parameters", frozen_parameters)

    @classmethod
    def from_mapping(cls, value: object, location: str = "profile envelope") -> ProfileEnvelope:
        payload = require_mapping(value, location)
        reject_environment_expressions(payload, location)
        reject_unknown(payload, _PROFILE_ENVELOPE_FIELDS, location)
        parameters = require_mapping(payload.get("parameters"), f"{location}.parameters")
        return cls(
            schema_version=require_text(
                payload.get("schema_version"), f"{location}.schema_version"
            ),
            profile_kind=require_stable_id(payload.get("profile_kind"), f"{location}.profile_kind"),
            profile_id=require_stable_id(payload.get("profile_id"), f"{location}.profile_id"),
            parameters=parameters,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_kind": self.profile_kind,
            "profile_id": self.profile_id,
            "parameters": _thaw_json_value(self.parameters),
        }

    @property
    def fingerprint(self) -> str:
        return payload_fingerprint(self.to_dict())


type DomainProfileParser[DomainProfileT] = Callable[[str, Mapping[str, object]], DomainProfileT]


@dataclass(frozen=True, slots=True)
class ResolvedProfileBinding[DomainProfileT]:
    reference: ProfileRef
    envelope: ProfileEnvelope
    domain_profile: DomainProfileT

    @property
    def fingerprint(self) -> str:
        return self.envelope.fingerprint

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference.to_dict(),
            "resolved_profile": self.envelope.to_dict(),
            "fingerprint": self.fingerprint,
        }


def _validate_profile_source_path(reference: ProfileRef) -> None:
    parts = reference.path.parts
    if (
        len(parts) < 6
        or parts[0] != "workspaces"
        or parts[2:4] != ("configs", "profiles")
        or parts[4] != reference.expected_kind
    ):
        raise ConfigError(
            "profile path must match workspaces/<owner>/configs/profiles/<expected_kind>/..."
        )
    require_stable_id(parts[1], "profile path workspace owner")
    if reference.path.suffix != ".yaml":
        raise ConfigError("profile path must identify a .yaml file")


def resolve_profile_reference[DomainProfileT](
    reference: ProfileRef,
    *,
    project_root: str | Path,
    domain_parser: DomainProfileParser[DomainProfileT],
) -> ResolvedProfileBinding[DomainProfileT]:
    _validate_profile_source_path(reference)
    source = resolve_project_path(project_root, reference.path.as_posix())
    if not source.is_file():
        raise ConfigError(f"profile file does not exist: {source}")
    payload = load_yaml_mapping(source, environment="forbid")
    envelope = ProfileEnvelope.from_mapping(payload, f"profile {source}")
    if envelope.profile_kind != reference.expected_kind:
        raise ConfigError(
            "profile kind mismatch: "
            f"expected {reference.expected_kind}, got {envelope.profile_kind}"
        )
    if envelope.profile_id != reference.expected_id:
        raise ConfigError(
            f"profile ID mismatch: expected {reference.expected_id}, got {envelope.profile_id}"
        )
    domain_profile = domain_parser(envelope.schema_version, envelope.parameters)
    return ResolvedProfileBinding(
        reference=reference,
        envelope=envelope,
        domain_profile=domain_profile,
    )
