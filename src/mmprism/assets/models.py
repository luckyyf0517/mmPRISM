from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast
from uuid import uuid4

import yaml

from mmprism.config import expand_environment

MODEL_ASSET_CONFIG_SCHEMA = "mmprism.model_asset_config.v1"
MODEL_ASSET_MANIFEST_SCHEMA = "mmprism.model_asset_manifest.v1"
MODEL_ASSET_COLLECTION_SCHEMA = "mmprism.model_asset_collection.v1"
MODEL_ASSET_SMOKE_SCHEMA = "mmprism.model_asset_smoke.v1"
MODEL_ASSET_MANIFEST_NAME = "mmprism_model_asset.json"
MODEL_ASSET_CHECKSUM_NAME = "SHA256SUMS"
MODEL_ASSET_COLLECTION_NAME = "mmprism_model_assets.json"

_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ASSET_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_REPO_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_SUPPORTED_LOADERS = frozenset({"sentence_transformers", "transformers_auto"})


class ModelAssetError(RuntimeError):
    """Raised when a model asset cannot be configured, acquired, or verified."""


class ModelHubClient(Protocol):
    def resolve_revision(self, repo_id: str, revision: str) -> str:
        """Resolve an immutable revision through the remote model registry."""

    def snapshot_download(
        self,
        repo_id: str,
        revision: str,
        required_files: Sequence[str],
        cache_dir: Path,
    ) -> Path:
        """Download the selected files and return their snapshot directory."""


class _HuggingFaceHubClient:
    def resolve_revision(self, repo_id: str, revision: str) -> str:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:  # pragma: no cover - exercised by integration smoke
            raise ModelAssetError(
                "huggingface-hub is required; install the evaluation extra"
            ) from error

        try:
            info = HfApi().model_info(repo_id=repo_id, revision=revision)
        except Exception as error:  # huggingface_hub exposes backend-specific subclasses
            raise ModelAssetError(
                f"Unable to resolve Hugging Face model {repo_id}@{revision}: {error}"
            ) from error
        resolved = getattr(info, "sha", None)
        if not isinstance(resolved, str) or not _SHA1_PATTERN.fullmatch(resolved):
            raise ModelAssetError(
                f"Hugging Face returned no immutable commit for {repo_id}@{revision}"
            )
        return resolved

    def snapshot_download(
        self,
        repo_id: str,
        revision: str,
        required_files: Sequence[str],
        cache_dir: Path,
    ) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:  # pragma: no cover - exercised by integration smoke
            raise ModelAssetError(
                "huggingface-hub is required; install the evaluation extra"
            ) from error

        try:
            snapshot = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=list(required_files),
                cache_dir=cache_dir,
            )
        except Exception as error:  # huggingface_hub exposes backend-specific subclasses
            raise ModelAssetError(
                f"Unable to download Hugging Face model {repo_id}@{revision}: {error}"
            ) from error
        return Path(snapshot).resolve()


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelAssetError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ModelAssetError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelAssetError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _relative_path(value: str, location: str) -> PurePosixPath:
    if "\\" in value:
        raise ModelAssetError(f"{location} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise ModelAssetError(f"{location} must be a normalized relative path")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class ModelAssetSpec:
    asset_id: str
    repo_id: str
    revision: str
    destination: PurePosixPath
    loader: str
    required_files: tuple[PurePosixPath, ...]

    @classmethod
    def from_mapping(cls, value: object, location: str) -> ModelAssetSpec:
        payload = _mapping(value, location)
        _reject_unknown(
            payload,
            {"asset_id", "repo_id", "revision", "destination", "loader", "required_files"},
            location,
        )
        asset_id = _text(payload, "asset_id", location)
        if not _ASSET_ID_PATTERN.fullmatch(asset_id):
            raise ModelAssetError(
                f"{location}.asset_id must match {_ASSET_ID_PATTERN.pattern}"
            )
        repo_id = _text(payload, "repo_id", location)
        if not _REPO_ID_PATTERN.fullmatch(repo_id):
            raise ModelAssetError(f"{location}.repo_id must be a namespaced model ID")
        revision = _text(payload, "revision", location)
        if not _SHA1_PATTERN.fullmatch(revision):
            raise ModelAssetError(
                f"{location}.revision must be an exact lowercase 40-character commit"
            )
        loader = _text(payload, "loader", location)
        if loader not in _SUPPORTED_LOADERS:
            supported = ", ".join(sorted(_SUPPORTED_LOADERS))
            raise ModelAssetError(f"{location}.loader must be one of: {supported}")

        files_value = payload.get("required_files")
        if not isinstance(files_value, list) or not files_value:
            raise ModelAssetError(f"{location}.required_files must be a non-empty list")
        files: list[PurePosixPath] = []
        for index, item in enumerate(files_value):
            if not isinstance(item, str) or not item.strip():
                raise ModelAssetError(
                    f"{location}.required_files[{index}] must be non-empty text"
                )
            files.append(
                _relative_path(item.strip(), f"{location}.required_files[{index}]")
            )
        if len(files) != len(set(files)):
            raise ModelAssetError(f"{location}.required_files must not contain duplicates")
        forbidden_names = {MODEL_ASSET_MANIFEST_NAME, MODEL_ASSET_CHECKSUM_NAME}
        if any(path.as_posix() in forbidden_names for path in files):
            raise ModelAssetError(
                f"{location}.required_files reserves the asset metadata filenames"
            )
        return cls(
            asset_id=asset_id,
            repo_id=repo_id,
            revision=revision,
            destination=_relative_path(
                _text(payload, "destination", location), f"{location}.destination"
            ),
            loader=loader,
            required_files=tuple(sorted(files, key=PurePosixPath.as_posix)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "destination": self.destination.as_posix(),
            "loader": self.loader,
            "required_files": [path.as_posix() for path in self.required_files],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelAssetSetConfig:
    asset_set_id: str
    models: tuple[ModelAssetSpec, ...]

    @classmethod
    def from_mapping(cls, value: object) -> ModelAssetSetConfig:
        payload = _mapping(value, "model asset config")
        _reject_unknown(payload, {"schema_version", "asset_set_id", "models"}, "model asset config")
        if payload.get("schema_version") != MODEL_ASSET_CONFIG_SCHEMA:
            raise ModelAssetError(f"schema_version must be {MODEL_ASSET_CONFIG_SCHEMA}")
        asset_set_id = _text(payload, "asset_set_id", "model asset config")
        if not _ASSET_ID_PATTERN.fullmatch(asset_set_id):
            raise ModelAssetError("model asset config.asset_set_id must be a stable lowercase ID")
        models_value = payload.get("models")
        if not isinstance(models_value, list) or not models_value:
            raise ModelAssetError("model asset config.models must be a non-empty list")
        models = tuple(
            ModelAssetSpec.from_mapping(item, f"model asset config.models[{index}]")
            for index, item in enumerate(models_value)
        )
        for label, values in (
            ("asset_id", [model.asset_id for model in models]),
            ("destination", [model.destination for model in models]),
        ):
            if len(values) != len(set(values)):
                raise ModelAssetError(f"model asset config has duplicate {label} values")
        return cls(asset_set_id=asset_set_id, models=models)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_ASSET_CONFIG_SCHEMA,
            "asset_set_id": self.asset_set_id,
            "models": [model.to_dict() for model in self.models],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_model_asset_config(path: str | Path) -> ModelAssetSetConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ModelAssetError(f"Unable to load model asset config: {error}") from error
    try:
        expanded = expand_environment(raw)
    except ValueError as error:
        raise ModelAssetError(str(error)) from error
    return ModelAssetSetConfig.from_mapping(expanded)


def _output_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise ModelAssetError(f"Model output root must be a real directory: {root}")
    return root


def _asset_path(root: Path, spec: ModelAssetSpec) -> Path:
    destination = root.joinpath(*spec.destination.parts)
    resolved = destination.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ModelAssetError(
            f"Model destination escapes output root: {spec.destination.as_posix()}"
        ) from error
    return destination


def _read_asset_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelAssetError(f"Unable to read model asset manifest {path}: {error}") from error
    return _mapping(payload, f"model asset manifest {path}")


def _verify_asset_directory(spec: ModelAssetSpec, target: Path) -> dict[str, object]:
    if not target.is_dir() or target.is_symlink():
        raise ModelAssetError(f"Model asset directory is missing or unsafe: {target}")
    actual_files: set[str] = set()
    for path in target.rglob("*"):
        if path.is_symlink():
            raise ModelAssetError(f"Model asset contains a symbolic link: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(target).as_posix())
    expected_files = {
        *(path.as_posix() for path in spec.required_files),
        MODEL_ASSET_MANIFEST_NAME,
        MODEL_ASSET_CHECKSUM_NAME,
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ModelAssetError(
            f"Model asset file inventory mismatch for {spec.asset_id}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    manifest_path = target / MODEL_ASSET_MANIFEST_NAME
    manifest = _read_asset_manifest(manifest_path)
    expected_identity = {
        "schema_version": MODEL_ASSET_MANIFEST_SCHEMA,
        "asset_id": spec.asset_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "destination": spec.destination.as_posix(),
        "loader": spec.loader,
        "spec_fingerprint": spec.fingerprint,
    }
    for key, value in expected_identity.items():
        if manifest.get(key) != value:
            raise ModelAssetError(
                f"Model asset manifest identity mismatch for {spec.asset_id}: {key}"
            )
    files_value = manifest.get("files")
    if not isinstance(files_value, list):
        raise ModelAssetError(f"Model asset manifest files are invalid for {spec.asset_id}")
    recorded: dict[str, tuple[int, str]] = {}
    for index, item in enumerate(files_value):
        row = _mapping(item, f"model asset manifest.files[{index}]")
        path_value = row.get("path")
        size_value = row.get("size_bytes")
        digest_value = row.get("sha256")
        if (
            not isinstance(path_value, str)
            or not isinstance(size_value, int)
            or isinstance(size_value, bool)
            or size_value < 0
            or not isinstance(digest_value, str)
            or not _SHA256_PATTERN.fullmatch(digest_value)
        ):
            raise ModelAssetError(
                f"Model asset manifest file entry {index} is invalid for {spec.asset_id}"
            )
        if path_value in recorded:
            raise ModelAssetError(f"Duplicate file in model asset manifest: {path_value}")
        recorded[path_value] = (size_value, digest_value)
    required_names = {path.as_posix() for path in spec.required_files}
    if set(recorded) != required_names:
        raise ModelAssetError(f"Model asset manifest inventory mismatch for {spec.asset_id}")

    checksum_lines: list[str] = []
    total_size = 0
    for relative in sorted(required_names):
        file_path = target.joinpath(*PurePosixPath(relative).parts)
        size = file_path.stat().st_size
        digest = _sha256(file_path)
        if recorded[relative] != (size, digest):
            raise ModelAssetError(f"Model asset checksum mismatch: {spec.asset_id}/{relative}")
        total_size += size
        checksum_lines.append(f"{digest}  {relative}\n")
    checksum_bytes = "".join(checksum_lines).encode("ascii")
    if (target / MODEL_ASSET_CHECKSUM_NAME).read_bytes() != checksum_bytes:
        raise ModelAssetError(f"SHA256SUMS mismatch for model asset {spec.asset_id}")
    return {
        "asset_id": spec.asset_id,
        "destination": spec.destination.as_posix(),
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "loader": spec.loader,
        "file_count": len(required_names),
        "size_bytes": total_size,
        "asset_manifest_sha256": _sha256(manifest_path),
    }


def plan_model_assets(
    config: ModelAssetSetConfig, output_root: str | Path
) -> dict[str, object]:
    root = _output_root(output_root)
    models: list[dict[str, object]] = []
    for spec in config.models:
        target = _asset_path(root, spec)
        if not target.exists():
            models.append(
                {
                    "asset_id": spec.asset_id,
                    "destination": spec.destination.as_posix(),
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "state": "missing",
                }
            )
            continue
        try:
            verified = _verify_asset_directory(spec, target)
        except ModelAssetError as error:
            models.append(
                {
                    "asset_id": spec.asset_id,
                    "destination": spec.destination.as_posix(),
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "state": "invalid",
                    "error": str(error),
                }
            )
        else:
            models.append({**verified, "state": "ready"})
    return {
        "schema_version": MODEL_ASSET_COLLECTION_SCHEMA,
        "asset_set_id": config.asset_set_id,
        "config_fingerprint": config.fingerprint,
        "output_root": str(root),
        "status": "ready" if all(model["state"] == "ready" for model in models) else "incomplete",
        "models": models,
    }


def verify_model_assets(
    config: ModelAssetSetConfig, output_root: str | Path
) -> dict[str, object]:
    plan = plan_model_assets(config, output_root)
    models = cast(list[dict[str, object]], plan["models"])
    failures = [model for model in models if model["state"] != "ready"]
    if failures:
        summary = ", ".join(
            f"{model['asset_id']}={model['state']}" for model in failures
        )
        raise ModelAssetError(f"Model asset verification failed: {summary}")
    return plan


def _builder_payload(runtime_report: Mapping[str, Any] | None) -> dict[str, object]:
    git_commit: object = None
    git_dirty: object = None
    python_version: object = None
    if runtime_report is not None:
        python_version = runtime_report.get("python")
        git = runtime_report.get("git")
        if isinstance(git, Mapping):
            git_commit = git.get("commit")
            git_dirty = git.get("dirty")
    return {
        "name": "mmprism",
        "version": _package_version("mmprism"),
        "python": python_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "huggingface_hub": _package_version("huggingface-hub"),
    }


def _materialize_asset(
    spec: ModelAssetSpec,
    target: Path,
    snapshot: Path,
    *,
    config_fingerprint: str,
    builder: Mapping[str, object],
    downloaded_at_utc: str,
) -> None:
    if not snapshot.is_dir():
        raise ModelAssetError(f"Downloaded snapshot is not a directory: {snapshot}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging.{uuid4().hex}"
    staging.mkdir()
    try:
        files: list[dict[str, object]] = []
        checksum_lines: list[str] = []
        for relative in spec.required_files:
            source = snapshot.joinpath(*relative.parts)
            if not source.is_file():
                raise ModelAssetError(
                    f"Required model file is absent from snapshot: {spec.repo_id}/{relative}"
                )
            destination = staging.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            digest = _sha256(destination)
            size = destination.stat().st_size
            path_text = relative.as_posix()
            files.append({"path": path_text, "size_bytes": size, "sha256": digest})
            checksum_lines.append(f"{digest}  {path_text}\n")
        (staging / MODEL_ASSET_CHECKSUM_NAME).write_text(
            "".join(checksum_lines), encoding="ascii"
        )
        manifest: dict[str, object] = {
            "schema_version": MODEL_ASSET_MANIFEST_SCHEMA,
            "asset_id": spec.asset_id,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "source_uri": f"https://huggingface.co/{spec.repo_id}/tree/{spec.revision}",
            "destination": spec.destination.as_posix(),
            "loader": spec.loader,
            "spec_fingerprint": spec.fingerprint,
            "config_fingerprint": config_fingerprint,
            "downloaded_at_utc": downloaded_at_utc,
            "builder": dict(builder),
            "files": files,
        }
        (staging / MODEL_ASSET_MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if target.exists():
            raise ModelAssetError(f"Model destination appeared during download: {target}")
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def download_model_assets(
    config: ModelAssetSetConfig,
    output_root: str | Path,
    *,
    hub_client: ModelHubClient | None = None,
    runtime_report: Mapping[str, Any] | None = None,
    downloaded_at_utc: str | None = None,
) -> dict[str, object]:
    root = _output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    client = hub_client or _HuggingFaceHubClient()
    builder = _builder_payload(runtime_report)
    timestamp = downloaded_at_utc or _utc_now()
    actions: dict[str, str] = {}
    lock_path = root / ".mmprism-model-assets.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        for spec in config.models:
            target = _asset_path(root, spec)
            if target.exists():
                _verify_asset_directory(spec, target)
                actions[spec.asset_id] = "reused"
                continue
            resolved_revision = client.resolve_revision(spec.repo_id, spec.revision)
            if resolved_revision != spec.revision:
                raise ModelAssetError(
                    f"Resolved revision mismatch for {spec.asset_id}: "
                    f"expected {spec.revision}, got {resolved_revision}"
                )
            snapshot = client.snapshot_download(
                spec.repo_id,
                spec.revision,
                [path.as_posix() for path in spec.required_files],
                root / ".cache" / "huggingface",
            )
            _materialize_asset(
                spec,
                target,
                snapshot,
                config_fingerprint=config.fingerprint,
                builder=builder,
                downloaded_at_utc=timestamp,
            )
            _verify_asset_directory(spec, target)
            actions[spec.asset_id] = "downloaded"

        verified = verify_model_assets(config, root)
        portable_models: list[dict[str, object]] = []
        verified_models = cast(list[dict[str, object]], verified["models"])
        for model in verified_models:
            row = dict(model)
            row.pop("state", None)
            portable_models.append(row)
        collection: dict[str, object] = {
            "schema_version": MODEL_ASSET_COLLECTION_SCHEMA,
            "asset_set_id": config.asset_set_id,
            "config_fingerprint": config.fingerprint,
            "generated_at_utc": timestamp,
            "builder": builder,
            "models": portable_models,
        }
        _write_bytes_atomic(_json_bytes(collection), root / MODEL_ASSET_COLLECTION_NAME)
        return {
            **verified,
            "collection_manifest": str(root / MODEL_ASSET_COLLECTION_NAME),
            "collection_manifest_sha256": _sha256(root / MODEL_ASSET_COLLECTION_NAME),
            "actions": actions,
        }


def _embedding_summary(name: str, embeddings: Any, input_count: int) -> dict[str, object]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised by integration smoke
        raise ModelAssetError("numpy is required for the model asset smoke") from error
    array = np.asarray(embeddings)
    if array.ndim != 2 or array.shape[0] != input_count or array.shape[1] < 1:
        raise ModelAssetError(f"{name} produced invalid embedding shape {array.shape}")
    if not np.isfinite(array).all():
        raise ModelAssetError(f"{name} produced non-finite embeddings")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0):
        raise ModelAssetError(f"{name} produced zero-norm embeddings")
    normalized = array / norms[:, None]
    similarities = normalized @ normalized.T
    diagonal = np.diag(similarities)
    if not np.isfinite(similarities).all() or np.any(diagonal < 0.999):
        raise ModelAssetError(f"{name} failed its cosine self-similarity check")
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "finite": True,
        "minimum_norm": float(norms.min()),
        "self_similarity_minimum": float(diagonal.min()),
        "pairwise_cosine": similarities.tolist(),
    }


def run_model_asset_smoke(
    config: ModelAssetSetConfig,
    output_root: str | Path,
    *,
    device: str = "cpu",
    texts: Sequence[str] = ("今天天气很好。", "请打开会议室的门。"),
) -> dict[str, object]:
    root = _output_root(output_root)
    verified = verify_model_assets(config, root)
    verified_models = cast(list[dict[str, object]], verified["models"])
    if len(texts) < 2 or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ModelAssetError("Model smoke requires at least two non-empty texts")
    results: list[dict[str, object]] = []
    for spec in config.models:
        model_path = _asset_path(root, spec)
        try:
            if spec.loader == "sentence_transformers":
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(str(model_path), device=device)
                embeddings = model.encode(
                    list(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            else:
                import torch
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                    model_path
                )
                model = AutoModel.from_pretrained(model_path).to(device)
                model.eval()
                batch = tokenizer(
                    list(texts), padding=True, truncation=True, return_tensors="pt"
                )
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                with torch.inference_mode():
                    output = model(**batch, return_dict=True)
                pooled = getattr(output, "pooler_output", None)
                if pooled is None:
                    pooled = output.last_hidden_state[:, 0, :]
                embeddings = pooled.detach().cpu().numpy()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ModelAssetError(f"Unable to load {spec.asset_id} for smoke: {error}") from error
        summary = _embedding_summary(spec.asset_id, embeddings, len(texts))
        verified_model = next(
            model for model in verified_models if model["asset_id"] == spec.asset_id
        )
        results.append(
            {
                "asset_id": spec.asset_id,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "loader": spec.loader,
                "asset_manifest_sha256": verified_model["asset_manifest_sha256"],
                "embedding": summary,
            }
        )
    return {
        "schema_version": MODEL_ASSET_SMOKE_SCHEMA,
        "status": "passed",
        "asset_set_id": config.asset_set_id,
        "config_fingerprint": config.fingerprint,
        "device": device,
        "input_count": len(texts),
        "models": results,
    }


def write_model_asset_smoke(payload: Mapping[str, object], destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    _write_bytes_atomic(_json_bytes(payload), path)
    return path
