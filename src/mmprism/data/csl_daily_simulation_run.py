"""Config-driven CSL-Daily pose -> simulated radar-cube materialization.

Reproduces the legacy ``run_simulation.py::process_sequence`` chain per pose
sequence (forensic reference only, never imported): load a ``[T, 2, 24, 3]``
dual-hand pose ``.npy``, drop NaN-contaminated frames, densify to a
63-point reflector cloud (z scaled by 0.6), Gaussian smooth (sigma=1),
finite-difference velocities (x10), decimate 30 -> 10 fps (``[::3]``),
then run the point-reflector FMCW :class:`~mmprism.simulation.Simulation`
and :class:`~mmprism.simulation.Processor` to obtain ``[T', 64, 32, 32, 32]``
power cubes.

Output contract (``mmprism.pose_reconstruction.sample_v1``): one manifest row
per *sequence*, with ``radar_cube [T', 64, 32, 32, 32] float32``,
``pose_gt [2, 24, 3] float32``, ``frame_mask [T'] bool`` and
``pose_valid [2, 24] bool``. This shape is dictated by the existing loader:
``PoseReconstructionManifest._pose_record`` requires a 5-dimensional
``radar_cube`` (leading time axis), a fixed ``(2, 24, 3)`` ``pose_gt``, and
``frame_mask`` of length ``radar_cube.shape[0]``; the OmniHand training
fixtures (``tests/integration/test_omnihand_run.py``) exercise exactly this
sequence-level layout with ``[frames, D, R, W, H]`` cubes, and
``collate_pose_reconstruction_samples`` pads the time axis across samples.
Because the contract stores exactly one pose target per sample, the target is
the raw (unsmoothed, metre) pose of the temporally central decimated frame.

NaN handling: a frame is valid only when every one of its ``2 x 24 x 3``
coordinates is finite. Invalid frames are dropped *before* densification and
smoothing because ``scipy.ndimage.gaussian_filter1d`` would otherwise smear
NaNs across the whole sequence. Every emitted frame is therefore valid, so
``frame_mask``/``pose_valid`` are all-true by construction. A sequence with
fewer than ``min_valid_frames`` valid input frames or fewer than
``min_output_frames`` decimated frames is recorded as a failed sample in the
run record; it never aborts the run.

This module is a pure library: it reads no environment variables, does no
logging, and parses no CLI arguments. ``${NAME}`` placeholders in the YAML
config are expanded from an explicit ``variables`` mapping supplied by the
caller. CLI wiring lands separately (the shared ``mmprism.config`` schema and
``mmprism.cli`` are intentionally untouched).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from mmprism.artifacts.run import sha256_file
from mmprism.contracts import validate_dual_hand_pose, validate_radar_cube
from mmprism.data.pose_reconstruction import (
    FRAME_MASK_MODALITY,
    POSE_RECONSTRUCTION_SAMPLE_PROTOCOL,
    POSE_TARGET_MODALITY,
    POSE_VALID_MODALITY,
    RADAR_CUBE_MODALITY,
    RADAR_CUBE_POWER_PROTOCOL,
)
from mmprism.simulation import (
    Processor,
    Simulation,
    densify_body_hand_frames,
    get_radar_config,
    temporal_smooth_decimate,
)
from mmprism.simulation.radar_config import RADAR_CONFIG_REGISTRY

CONFIG_SCHEMA = "mmprism.csl_daily_simulation.v1"
RUN_RECORD_SCHEMA = "mmprism.csl_daily_simulation_run.v1"

# DEC-012: the rebuilt simulator is registered as its own experiment protocol,
# never as a reproduction of the manuscript MANO pipeline.
SIMULATION_PROTOCOL = "csl_daily_skeleton_sim_v1"

SAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

POSE_ENTRY_KEYS = {"sample_id", "pose_uri", "pose_sha256", "sequence_id", "subject_id"}


class CslDailySimulationError(RuntimeError):
    """Raised when the CSL-Daily simulation materialization cannot proceed."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CslDailySimulationError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise CslDailySimulationError(f"unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CslDailySimulationError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _integer(
    payload: Mapping[str, Any], key: str, location: str, default: int, minimum: int
) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CslDailySimulationError(f"{location}.{key} must be an integer >= {minimum}")
    return value


def _number(
    payload: Mapping[str, Any], key: str, location: str, default: float
) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise CslDailySimulationError(f"{location}.{key} must be a positive number")
    return float(value)


def _simple_filename(value: str, location: str) -> str:
    if PurePosixPath(value).name != value:
        raise CslDailySimulationError(f"{location} must be a plain file name, got {value!r}")
    return value


def _expand_variables(value: Any, variables: Mapping[str, str]) -> Any:
    """Expand ``${NAME}``/``${NAME:-default}`` placeholders from an explicit mapping."""
    if isinstance(value, Mapping):
        return {key: _expand_variables(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_variables(item, variables) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        if name in variables:
            return variables[name]
        if default is not None:
            return default
        raise CslDailySimulationError(
            f"configuration placeholder {name} has no supplied value"
        )

    return _VARIABLE_PATTERN.sub(replace, value)


@dataclass(frozen=True, slots=True)
class CslDailySimulationConfig:
    """Resolved materialization configuration (all paths absolute)."""

    radar_config_id: str
    pose_manifest_path: Path
    pose_root: Path
    output_root: Path
    manifest_name: str
    run_record_name: str
    dataset: str
    coordinate_frame: str
    smoothing_sigma: float
    velocity_scale: float
    decimation: int
    min_valid_frames: int
    min_output_frames: int
    frames_per_batch: int
    device: str
    precision: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CslDailySimulationConfig:
        root = _mapping(payload, "root")
        _reject_unknown(
            root,
            {"schema_version", "radar", "input", "output", "dataset", "preprocessing", "runtime"},
            "root",
        )
        if root.get("schema_version") != CONFIG_SCHEMA:
            raise CslDailySimulationError(f"schema_version must be {CONFIG_SCHEMA}")

        radar = _mapping(root.get("radar"), "radar")
        _reject_unknown(radar, {"radar_config_id"}, "radar")
        radar_config_id = _text(radar, "radar_config_id", "radar")
        if radar_config_id not in RADAR_CONFIG_REGISTRY:
            known = ", ".join(sorted(RADAR_CONFIG_REGISTRY))
            raise CslDailySimulationError(
                f"unknown radar_config_id: {radar_config_id!r}; known: {known}"
            )

        input_section = _mapping(root.get("input"), "input")
        _reject_unknown(input_section, {"pose_manifest_path", "pose_root"}, "input")
        pose_manifest_path = Path(_text(input_section, "pose_manifest_path", "input"))
        pose_root = Path(_text(input_section, "pose_root", "input"))

        output = _mapping(root.get("output"), "output")
        _reject_unknown(output, {"output_root", "manifest_name", "run_record_name"}, "output")
        output_root = Path(_text(output, "output_root", "output"))
        manifest_name = _simple_filename(
            _text(output, "manifest_name", "output")
            if "manifest_name" in output
            else "manifest.jsonl",
            "output.manifest_name",
        )
        run_record_name = _simple_filename(
            _text(output, "run_record_name", "output")
            if "run_record_name" in output
            else "run_record.json",
            "output.run_record_name",
        )

        dataset = _mapping(root.get("dataset"), "dataset")
        _reject_unknown(dataset, {"name", "coordinate_frame"}, "dataset")

        preprocessing = _mapping(root.get("preprocessing", {}), "preprocessing")
        _reject_unknown(
            preprocessing,
            {
                "smoothing_sigma",
                "velocity_scale",
                "decimation",
                "min_valid_frames",
                "min_output_frames",
            },
            "preprocessing",
        )

        runtime = _mapping(root.get("runtime", {}), "runtime")
        _reject_unknown(runtime, {"device", "precision", "frames_per_batch"}, "runtime")
        device = _text(runtime, "device", "runtime") if "device" in runtime else "cpu"
        if device != "cpu":
            raise CslDailySimulationError(
                f"runtime.device must be 'cpu' for this pipeline, got {device!r}"
            )
        precision = (
            _text(runtime, "precision", "runtime") if "precision" in runtime else "float32"
        )
        if precision != "float32":
            raise CslDailySimulationError(
                f"runtime.precision must be 'float32', got {precision!r}"
            )

        return cls(
            radar_config_id=radar_config_id,
            pose_manifest_path=pose_manifest_path,
            pose_root=pose_root,
            output_root=output_root,
            manifest_name=manifest_name,
            run_record_name=run_record_name,
            dataset=_text(dataset, "name", "dataset"),
            coordinate_frame=_text(dataset, "coordinate_frame", "dataset"),
            smoothing_sigma=_number(preprocessing, "smoothing_sigma", "preprocessing", 1.0),
            velocity_scale=_number(preprocessing, "velocity_scale", "preprocessing", 10.0),
            decimation=_integer(preprocessing, "decimation", "preprocessing", 3, 1),
            min_valid_frames=_integer(
                preprocessing, "min_valid_frames", "preprocessing", 2, 2
            ),
            min_output_frames=_integer(
                preprocessing, "min_output_frames", "preprocessing", 1, 1
            ),
            frames_per_batch=_integer(runtime, "frames_per_batch", "runtime", 4, 1),
            device=device,
            precision=precision,
        )

    def portable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "radar": {"radar_config_id": self.radar_config_id},
            "input": {
                "pose_manifest_path": str(self.pose_manifest_path),
                "pose_root": str(self.pose_root),
            },
            "output": {
                "output_root": str(self.output_root),
                "manifest_name": self.manifest_name,
                "run_record_name": self.run_record_name,
            },
            "dataset": {
                "name": self.dataset,
                "coordinate_frame": self.coordinate_frame,
            },
            "preprocessing": {
                "smoothing_sigma": self.smoothing_sigma,
                "velocity_scale": self.velocity_scale,
                "decimation": self.decimation,
                "min_valid_frames": self.min_valid_frames,
                "min_output_frames": self.min_output_frames,
            },
            "runtime": {
                "device": self.device,
                "precision": self.precision,
                "frames_per_batch": self.frames_per_batch,
            },
        }

    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.portable_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


def load_csl_daily_simulation_config(
    path: str | Path, *, variables: Mapping[str, str]
) -> CslDailySimulationConfig:
    """Load the YAML config, expanding ``${NAME}`` placeholders from ``variables``.

    The library never reads process environment variables; the caller (CLI)
    injects roots such as ``MMPRISM_DATA_ROOT`` explicitly.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise CslDailySimulationError(f"configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            payload: object = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise CslDailySimulationError(f"invalid YAML in {config_path}: {error}") from error
    expanded = _expand_variables(payload, variables)
    return CslDailySimulationConfig.from_mapping(_mapping(expanded, "root"))


@dataclass(frozen=True, slots=True)
class PoseInputEntry:
    """One frozen pose-manifest row: a check-summed ``[T, 2, 24, 3]`` pose file."""

    sample_id: str
    pose_uri: str
    pose_sha256: str
    sequence_id: str | None
    subject_id: str | None


def load_pose_manifest(path: str | Path) -> tuple[PoseInputEntry, ...]:
    """Parse the frozen pose manifest JSONL (sample id, pose URI, SHA-256)."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise CslDailySimulationError(f"pose manifest does not exist: {manifest_path}")
    entries: list[PoseInputEntry] = []
    sample_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            location = f"{manifest_path}:{line_number}"
            try:
                payload: object = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CslDailySimulationError(f"invalid JSON at {location}: {error}") from error
            record = _mapping(payload, location)
            _reject_unknown(record, POSE_ENTRY_KEYS, location)
            sample_id = _text(record, "sample_id", location)
            if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
                raise CslDailySimulationError(
                    f"{location}.sample_id must match {SAMPLE_ID_PATTERN.pattern}"
                )
            if sample_id in sample_ids:
                raise CslDailySimulationError(f"duplicate sample_id {sample_id!r} at {location}")
            sample_ids.add(sample_id)
            pose_uri = _text(record, "pose_uri", location)
            uri_path = PurePosixPath(pose_uri)
            if uri_path.is_absolute() or ".." in uri_path.parts:
                raise CslDailySimulationError(
                    f"{location}.pose_uri must be a relative path inside the pose root"
                )
            if uri_path.suffix != ".npy":
                raise CslDailySimulationError(f"{location}.pose_uri must reference a .npy file")
            pose_sha256 = _text(record, "pose_sha256", location)
            if not SHA256_PATTERN.fullmatch(pose_sha256):
                raise CslDailySimulationError(
                    f"{location}.pose_sha256 must be a lowercase SHA-256 digest"
                )
            sequence_id = record.get("sequence_id")
            subject_id = record.get("subject_id")
            if sequence_id is not None and not isinstance(sequence_id, str):
                raise CslDailySimulationError(f"{location}.sequence_id must be a string")
            if subject_id is not None and not isinstance(subject_id, str):
                raise CslDailySimulationError(f"{location}.subject_id must be a string")
            entries.append(
                PoseInputEntry(
                    sample_id=sample_id,
                    pose_uri=pose_uri,
                    pose_sha256=pose_sha256,
                    sequence_id=sequence_id,
                    subject_id=subject_id,
                )
            )
    if not entries:
        raise CslDailySimulationError(f"pose manifest is empty: {manifest_path}")
    return tuple(entries)


def frame_validity(pose: NDArray[np.generic]) -> NDArray[np.bool_]:
    """Boolean ``[T]`` mask: a frame is valid only if every coordinate is finite."""
    return np.asarray(np.isfinite(pose).all(axis=(1, 2, 3)), dtype=np.bool_)


def select_target_pose(
    valid_pose: NDArray[np.generic], *, output_frames: int, decimation: int
) -> NDArray[np.generic]:
    """Pick the raw pose of the temporally central decimated frame.

    Decimated output frame ``i`` corresponds to valid-frame index
    ``i * decimation`` (``temporal_smooth_decimate`` aligns ``smoothed[:-1]``
    and then strides by ``decimation``).
    """
    index = min((output_frames // 2) * decimation, valid_pose.shape[0] - 1)
    return np.asarray(valid_pose[index])


@dataclass(frozen=True, slots=True)
class ArrayArtifact:
    """A written ``.npy`` array with manifest metadata."""

    uri: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def modality_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


def build_manifest_row(
    *,
    entry: PoseInputEntry,
    config: CslDailySimulationConfig,
    artifacts: Mapping[str, ArrayArtifact],
    pose_manifest_sha256: str,
) -> dict[str, Any]:
    """Build one ``mmprism.pose_reconstruction.sample_v1`` manifest row."""
    row: dict[str, Any] = {
        "schema_version": "mmprism.sample.v1",
        "sample_id": entry.sample_id,
        "sequence_id": entry.sequence_id or entry.sample_id,
        "dataset": config.dataset,
        "modalities": {
            modality: artifact.modality_dict() for modality, artifact in artifacts.items()
        },
        "acquisition": {
            "sample_protocol": POSE_RECONSTRUCTION_SAMPLE_PROTOCOL,
            "radar_cube_protocol": RADAR_CUBE_POWER_PROTOCOL,
            "pose_units": "m",
            "pose_coordinate_frame": config.coordinate_frame,
            "radar_config_id": config.radar_config_id,
        },
        "provenance": {
            "simulation_protocol": SIMULATION_PROTOCOL,
            "generator": "mmprism.data.csl_daily_simulation_run",
            "radar_config_id": config.radar_config_id,
            "source_pose_manifest_sha256": pose_manifest_sha256,
            "source_pose_uri": entry.pose_uri,
            "source_pose_sha256": entry.pose_sha256,
            "preprocessing": {
                "smoothing_sigma": config.smoothing_sigma,
                "velocity_scale": config.velocity_scale,
                "decimation": config.decimation,
            },
        },
    }
    if entry.subject_id is not None:
        row["subject_id"] = entry.subject_id
    return row


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    """Per-sample materialization status recorded in the run record."""

    sample_id: str
    status: str  # "emitted" or "failed"
    reason: str | None
    input_frames: int
    valid_frames: int
    output_frames: int
    source_pose_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "reason": self.reason,
            "input_frames": self.input_frames,
            "valid_frames": self.valid_frames,
            "output_frames": self.output_frames,
            "source_pose_sha256": self.source_pose_sha256,
        }


@dataclass(frozen=True, slots=True)
class CslDailySimulationResult:
    """Structured result of a materialization run."""

    manifest_path: Path
    run_record_path: Path
    config_fingerprint: str
    pose_manifest_sha256: str
    outcomes: tuple[SampleOutcome, ...]
    run_record: dict[str, Any]

    @property
    def emitted_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "emitted")

    @property
    def failed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "failed")


def _write_array(
    output_root: Path, modality: str, sample_id: str, array: NDArray[np.generic]
) -> ArrayArtifact:
    uri = f"{modality}/{sample_id}.npy"
    path = output_root / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return ArrayArtifact(
        uri=uri,
        shape=tuple(int(size) for size in array.shape),
        dtype=str(array.dtype),
        sha256=sha256_file(path),
    )


def _load_verified_pose(entry: PoseInputEntry, pose_root: Path) -> NDArray[np.floating[Any]]:
    pose_root = pose_root.resolve()
    path = (pose_root / entry.pose_uri).resolve()
    if not path.is_relative_to(pose_root):
        raise CslDailySimulationError(
            f"sample {entry.sample_id} pose URI escapes the pose root"
        )
    if not path.is_file():
        raise CslDailySimulationError(f"sample {entry.sample_id} pose file missing: {path}")
    observed = sha256_file(path)
    if observed != entry.pose_sha256:
        raise CslDailySimulationError(
            f"sample {entry.sample_id} pose SHA-256 mismatch: "
            f"expected {entry.pose_sha256}, got {observed}"
        )
    try:
        pose = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CslDailySimulationError(
            f"sample {entry.sample_id} pose is not a readable .npy: {error}"
        ) from error
    if not isinstance(pose, np.ndarray) or not np.issubdtype(pose.dtype, np.floating):
        raise CslDailySimulationError(
            f"sample {entry.sample_id} pose must be a floating NumPy array"
        )
    if pose.ndim != 4 or pose.shape[1:] != (2, 24, 3):
        raise CslDailySimulationError(
            f"sample {entry.sample_id} pose must have shape [T, 2, 24, 3], got {pose.shape}"
        )
    return np.asarray(pose, dtype=np.float32)


def _simulate_cube(
    simulation: Simulation,
    processor: Processor,
    points: NDArray[np.generic],
    velocities: NDArray[np.generic],
    *,
    frames_per_batch: int,
) -> NDArray[np.float32]:
    """Run the FMCW simulation chain over decimated frames on CPU."""
    chunks: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, int(points.shape[0]), frames_per_batch):
            stop = start + frames_per_batch
            batch_points = torch.from_numpy(np.ascontiguousarray(points[start:stop])).float()
            batch_velocities = torch.from_numpy(
                np.ascontiguousarray(velocities[start:stop])
            ).float()
            raw_frames = simulation(batch_points, batch_velocities)
            cube = processor(raw_frames)
            chunks.append(np.asarray(cube.cpu().numpy(), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _materialize_entry(
    entry: PoseInputEntry,
    *,
    config: CslDailySimulationConfig,
    simulation: Simulation,
    processor: Processor,
    pose_manifest_sha256: str,
) -> tuple[dict[str, Any], SampleOutcome]:
    """Run the full chain for one pose sequence and write its artifacts."""
    pose = _load_verified_pose(entry, config.pose_root)
    input_frames = int(pose.shape[0])
    validity = frame_validity(pose)
    valid_frames = int(validity.sum())
    if valid_frames < config.min_valid_frames:
        raise CslDailySimulationError(
            f"sample {entry.sample_id} has {valid_frames} valid frames, "
            f"requires at least {config.min_valid_frames}"
        )
    valid_pose = pose[validity]

    # NaN frames are already dropped: smoothing never sees non-finite input.
    densified = densify_body_hand_frames(valid_pose.reshape(valid_frames, 48, 3))
    points, velocities = temporal_smooth_decimate(
        densified,
        sigma=config.smoothing_sigma,
        velocity_scale=config.velocity_scale,
        decimation=config.decimation,
    )
    output_frames = int(points.shape[0])
    if output_frames < config.min_output_frames:
        raise CslDailySimulationError(
            f"sample {entry.sample_id} yields {output_frames} decimated frames, "
            f"requires at least {config.min_output_frames}"
        )

    cube = _simulate_cube(
        simulation,
        processor,
        points,
        velocities,
        frames_per_batch=config.frames_per_batch,
    )
    validate_radar_cube(cube, leading_axes=("time",))

    target = np.asarray(
        select_target_pose(
            valid_pose, output_frames=output_frames, decimation=config.decimation
        ),
        dtype=np.float32,
    )
    validate_dual_hand_pose(target, coordinate_frame=config.coordinate_frame)

    frame_mask = np.ones(output_frames, dtype=np.bool_)
    pose_valid = np.ones((2, 24), dtype=np.bool_)

    artifacts = {
        RADAR_CUBE_MODALITY: _write_array(
            config.output_root, RADAR_CUBE_MODALITY, entry.sample_id, cube
        ),
        POSE_TARGET_MODALITY: _write_array(
            config.output_root, POSE_TARGET_MODALITY, entry.sample_id, target
        ),
        FRAME_MASK_MODALITY: _write_array(
            config.output_root, FRAME_MASK_MODALITY, entry.sample_id, frame_mask
        ),
        POSE_VALID_MODALITY: _write_array(
            config.output_root, POSE_VALID_MODALITY, entry.sample_id, pose_valid
        ),
    }
    row = build_manifest_row(
        entry=entry,
        config=config,
        artifacts=artifacts,
        pose_manifest_sha256=pose_manifest_sha256,
    )
    outcome = SampleOutcome(
        sample_id=entry.sample_id,
        status="emitted",
        reason=None,
        input_frames=input_frames,
        valid_frames=valid_frames,
        output_frames=output_frames,
        source_pose_sha256=entry.pose_sha256,
    )
    return row, outcome


def run_csl_daily_simulation(
    config: CslDailySimulationConfig, *, created_at: datetime | None = None
) -> CslDailySimulationResult:
    """Materialize every pose-manifest entry into simulated radar cubes.

    Per-sample failures (missing/corrupt pose, SHA-256 mismatch, too few valid
    frames, contract violations) are recorded in the run record and never
    abort the run. The output manifest refuses to clobber an existing file.
    """
    timestamp = (created_at or datetime.now(UTC)).isoformat()
    pose_manifest_path = config.pose_manifest_path
    if not pose_manifest_path.is_file():
        raise CslDailySimulationError(f"pose manifest does not exist: {pose_manifest_path}")
    pose_manifest_sha256 = sha256_file(pose_manifest_path)
    entries = load_pose_manifest(pose_manifest_path)

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / config.manifest_name
    run_record_path = output_root / config.run_record_name
    if manifest_path.exists():
        raise CslDailySimulationError(
            f"output manifest already exists (no-clobber): {manifest_path}"
        )
    if run_record_path.exists():
        raise CslDailySimulationError(
            f"run record already exists (no-clobber): {run_record_path}"
        )

    radar_config = get_radar_config(config.radar_config_id)
    simulation = Simulation(radar_config)
    processor = Processor(process_range=True)

    rows: list[dict[str, Any]] = []
    outcomes: list[SampleOutcome] = []
    for entry in entries:
        try:
            row, outcome = _materialize_entry(
                entry,
                config=config,
                simulation=simulation,
                processor=processor,
                pose_manifest_sha256=pose_manifest_sha256,
            )
        except Exception as error:  # per-sample failure: record, never crash
            row = None
            outcome = SampleOutcome(
                sample_id=entry.sample_id,
                status="failed",
                reason=f"{type(error).__name__}: {error}",
                input_frames=0,
                valid_frames=0,
                output_frames=0,
                source_pose_sha256=entry.pose_sha256,
            )
        if row is not None:
            rows.append(row)
        outcomes.append(outcome)

    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    result_outcomes = tuple(outcomes)
    emitted = sum(1 for outcome in result_outcomes if outcome.status == "emitted")
    run_record: dict[str, Any] = {
        "schema_version": RUN_RECORD_SCHEMA,
        "created_at": timestamp,
        "config": config.portable_dict(),
        "config_sha256": config.fingerprint(),
        "radar_config_id": config.radar_config_id,
        "simulation_protocol": SIMULATION_PROTOCOL,
        "pose_manifest": {
            "path": str(pose_manifest_path),
            "sha256": pose_manifest_sha256,
            "entry_count": len(entries),
        },
        "counts": {
            "entries": len(entries),
            "emitted": emitted,
            "failed": len(entries) - emitted,
            "input_frames": sum(outcome.input_frames for outcome in result_outcomes),
            "valid_frames": sum(outcome.valid_frames for outcome in result_outcomes),
            "output_frames": sum(outcome.output_frames for outcome in result_outcomes),
        },
        "samples": [outcome.to_dict() for outcome in result_outcomes],
        "outputs": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "record_count": len(rows),
        },
    }
    with run_record_path.open("w", encoding="utf-8") as stream:
        json.dump(run_record, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return CslDailySimulationResult(
        manifest_path=manifest_path,
        run_record_path=run_record_path,
        config_fingerprint=config.fingerprint(),
        pose_manifest_sha256=pose_manifest_sha256,
        outcomes=result_outcomes,
        run_record=run_record,
    )
