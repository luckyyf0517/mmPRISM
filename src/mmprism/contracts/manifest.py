import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a data manifest violates the canonical contract."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ManifestError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _required_text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str, location: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location}.{key} must be a non-empty string when provided")
    return value.strip()


@dataclass(frozen=True)
class ModalityRef:
    uri: str | None = None
    text: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, location: str) -> "ModalityRef":
        payload = _mapping(value, location)
        _reject_unknown(payload, {"uri", "text", "shape", "dtype", "sha256"}, location)
        uri = _optional_text(payload, "uri", location)
        text = _optional_text(payload, "text", location)
        if (uri is None) == (text is None):
            raise ManifestError(f"{location} must provide exactly one of uri or text")
        if uri is not None and "://" not in uri and Path(uri).is_absolute():
            raise ManifestError(f"{location}.uri must be relative to a configured root: {uri}")
        if uri is not None and "://" not in uri and ".." in Path(uri).parts:
            raise ManifestError(f"{location}.uri must not escape its configured root: {uri}")

        shape_value = payload.get("shape")
        shape: tuple[int, ...] | None = None
        if shape_value is not None:
            if not isinstance(shape_value, list) or not all(
                isinstance(size, int) and size >= 0 for size in shape_value
            ):
                raise ManifestError(f"{location}.shape must be a list of non-negative integers")
            shape = tuple(shape_value)

        dtype = payload.get("dtype")
        if dtype is not None and (not isinstance(dtype, str) or not dtype):
            raise ManifestError(f"{location}.dtype must be a non-empty string when provided")

        sha256 = payload.get("sha256")
        if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ManifestError(f"{location}.sha256 must be a lowercase SHA-256 digest")

        return cls(uri=uri, text=text, shape=shape, dtype=dtype, sha256=sha256)


@dataclass(frozen=True)
class SampleRecord:
    schema_version: str
    sample_id: str
    dataset: str
    modalities: dict[str, ModalityRef]
    sequence_id: str | None = None
    subject_id: str | None = None
    group_keys: dict[str, str] | None = None
    acquisition: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Any, location: str = "record") -> "SampleRecord":
        payload = _mapping(value, location)
        allowed = {
            "schema_version",
            "sample_id",
            "sequence_id",
            "subject_id",
            "dataset",
            "modalities",
            "group_keys",
            "acquisition",
            "provenance",
        }
        _reject_unknown(payload, allowed, location)

        schema_version = _required_text(payload, "schema_version", location)
        if schema_version != "mmprism.sample.v1":
            raise ManifestError(f"Unsupported sample schema at {location}: {schema_version}")

        modality_payload = _mapping(payload.get("modalities"), f"{location}.modalities")
        if not modality_payload:
            raise ManifestError(f"{location}.modalities must contain at least one modality")
        modalities: dict[str, ModalityRef] = {}
        for name, reference in modality_payload.items():
            if not isinstance(name, str) or not name.strip():
                raise ManifestError(f"{location}.modalities keys must be non-empty strings")
            modalities[name] = ModalityRef.from_mapping(
                reference, f"{location}.modalities.{name}"
            )

        group_keys_value = payload.get("group_keys")
        group_keys: dict[str, str] | None = None
        if group_keys_value is not None:
            group_mapping = _mapping(group_keys_value, f"{location}.group_keys")
            valid_group_keys = all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in group_mapping.items()
            )
            if not valid_group_keys:
                raise ManifestError(f"{location}.group_keys must map strings to strings")
            group_keys = dict(group_mapping)

        acquisition_value = payload.get("acquisition")
        acquisition = (
            dict(_mapping(acquisition_value, f"{location}.acquisition"))
            if acquisition_value is not None
            else None
        )
        provenance_value = payload.get("provenance")
        provenance = (
            dict(_mapping(provenance_value, f"{location}.provenance"))
            if provenance_value is not None
            else None
        )

        return cls(
            schema_version=schema_version,
            sample_id=_required_text(payload, "sample_id", location),
            sequence_id=_optional_text(payload, "sequence_id", location),
            subject_id=_optional_text(payload, "subject_id", location),
            dataset=_required_text(payload, "dataset", location),
            modalities=modalities,
            group_keys=group_keys,
            acquisition=acquisition,
            provenance=provenance,
        )


@dataclass(frozen=True)
class ManifestSummary:
    path: Path
    record_count: int
    datasets: tuple[str, ...]
    modalities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "record_count": self.record_count,
            "datasets": list(self.datasets),
            "modalities": list(self.modalities),
        }


def validate_manifest(path: str | Path) -> ManifestSummary:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"Manifest file does not exist: {manifest_path}")

    sample_ids: set[str] = set()
    datasets: set[str] = set()
    modalities: set[str] = set()
    record_count = 0

    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                message = f"Invalid JSON at {manifest_path}:{line_number}: {error}"
                raise ManifestError(message) from error

            record = SampleRecord.from_mapping(payload, f"line {line_number}")
            if record.sample_id in sample_ids:
                raise ManifestError(
                    f"Duplicate sample_id {record.sample_id!r} at {manifest_path}:{line_number}"
                )
            sample_ids.add(record.sample_id)
            datasets.add(record.dataset)
            modalities.update(record.modalities)
            record_count += 1

    if record_count == 0:
        raise ManifestError(f"Manifest contains no records: {manifest_path}")

    return ManifestSummary(
        path=manifest_path.resolve(),
        record_count=record_count,
        datasets=tuple(sorted(datasets)),
        modalities=tuple(sorted(modalities)),
    )
