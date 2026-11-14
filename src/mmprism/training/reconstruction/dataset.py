"""Aligned radar windows, deterministic rank cycles, binding and target caches."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from mmprism.data.cascade_compact_raw import CompactSegmentResolver
from mmprism.data.compact_radar import PackedRadar
from mmprism.data.reconstruction_membership import (
    deterministic_cycle_batches,
    deterministic_window_start,
    parse_membership,
    role_counts,
)
from mmprism.release.paths import resolve
from mmprism.training.reconstruction.dense_token_deterministic_mhr import UPPER_BODY_INDICES

MHR_STATE_DIMENSION = 191
BINDING_SCHEMA = "mmprism.reconstruction.binding"
TARGET_CACHE_SCHEMA = "mmprism.reconstruction.target_cache"


class DatasetError(RuntimeError):
    """The aligned training data contract cannot be satisfied."""


def _section_path(config, section, key):
    return resolve(config[section][key])


def _resolve_nested_path(config, *, section, group, key):
    return resolve(config[section][group][key])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetError(f"cannot read {description} {path}: {error}") from error
    _require(isinstance(value, dict), f"{description} must be a JSON object")
    return cast(dict[str, Any], value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class LabelSummary:
    frame_count: int
    valid_frame_count: int


@dataclass(frozen=True, slots=True)
class BoundSegment:
    corpus: str
    role: str
    source_id: str
    sample_id: str
    source_segment_id: str
    reference_id: str
    parent_id: str
    frame_count: int
    valid_frame_count: int
    target_offset: int


def _segments_for_role(segments: Sequence[BoundSegment], role: str) -> tuple[BoundSegment, ...]:
    selected = tuple(segment for segment in segments if segment.role == role)
    _require(selected, f"reconstruction {role} segment support is empty")
    return selected


def load_prepared_segments(output_root: Path) -> tuple[BoundSegment, ...]:
    payload = _read_json(
        output_root / "training_binding.json", "prepared reconstruction trainer binding"
    )
    _require(
        payload.get("schema_version") == BINDING_SCHEMA,
        "prepared reconstruction binding schema differs",
    )
    rows = payload.get("segments")
    _require(isinstance(rows, list) and rows, "prepared reconstruction binding has no segments")
    segments: list[BoundSegment] = []
    for row in rows:
        _require(isinstance(row, Mapping), "prepared reconstruction segment row differs")
        try:
            segment = BoundSegment(
                corpus=str(row["corpus"]),
                role=str(row["role"]),
                source_id=str(row["source_id"]),
                sample_id=str(row["sample_id"]),
                source_segment_id=str(row["source_segment_id"]),
                reference_id=str(row["reference_id"]),
                parent_id=str(row["parent_id"]),
                frame_count=int(row["frame_count"]),
                valid_frame_count=int(row["valid_frame_count"]),
                target_offset=int(row["target_offset"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError(
                f"prepared reconstruction segment row is malformed: {error}"
            ) from error
        _require(
            segment.role in {"train", "validation", "test"}
            and segment.corpus in {"old", "new", "v4"}
            and (segment.frame_count > 0)
            and (0 <= segment.valid_frame_count <= segment.frame_count)
            and (segment.target_offset >= 0),
            "prepared reconstruction segment values differ",
        )
        segments.append(segment)
    return tuple(segments)


def _normalization_arrays(target_root: Path) -> dict[str, np.ndarray]:
    path = target_root / "train_only_normalization.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = {
                "mean": np.asarray(archive["mean"], dtype=np.float32),
                "scale": np.asarray(archive["scale"], dtype=np.float32),
                "active_dimensions": np.asarray(archive["active_dimensions"], dtype=bool),
            }
    except (OSError, KeyError, ValueError) as error:
        raise DatasetError(
            f"cannot read reconstruction train-only normalization: {error}"
        ) from error
    _require(
        all(value.shape == (MHR_STATE_DIMENSION,) for value in result.values()),
        "reconstruction train-only normalization shape differs",
    )
    _require(
        bool(np.isfinite(result["mean"]).all())
        and bool(np.isfinite(result["scale"]).all())
        and bool((result["scale"] > 0.0).all()),
        "reconstruction train-only normalization values differ",
    )
    return result


class WindowDataset:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        segments: Sequence[BoundSegment],
        role: str,
        target_root: Path,
    ) -> None:
        self.config = config
        self.segments = _segments_for_role(segments, role)
        self.role = role
        self.window_frames = int(_mapping(config.get("data"), "data")["train_window_frames"])
        self.window_seed = int(_mapping(config.get("optimization"), "optimization")["sampler_seed"])
        (targets_path, valid_path) = _target_paths(target_root, role)
        self.target = np.load(targets_path, mmap_mode="r")
        self.target_valid = np.load(valid_path, mmap_mode="r")
        self.global_pose = np.load(_global_pose_path(target_root, role), mmap_mode="r")
        _require(
            self.target.shape == (len(self.target_valid), MHR_STATE_DIMENSION)
            and self.global_pose.shape == (len(self.target_valid), 2, 24, 3),
            f"{role} target/global cache shapes differ",
        )
        _require(
            sum(segment.frame_count for segment in self.segments) == len(self.target_valid),
            f"{role} segment frame counts differ from target cache",
        )
        self.resolvers = _load_source_resolvers(config)

    def __len__(self) -> int:
        return len(self.segments)

    def _read_window(
        self, *, segment_index: int, start: int, sampler_cycle: int
    ) -> dict[str, np.ndarray | str | int]:
        _require(
            0 <= segment_index < len(self.segments),
            "reconstructionsegment index is outside role support",
        )
        segment = self.segments[segment_index]
        _require(0 <= start < segment.frame_count, "reconstructionwindow start is outside segment")
        stop = min(start + self.window_frames, segment.frame_count)
        count = stop - start
        resolver = self.resolvers.get(segment.source_id)
        _require(
            resolver is not None, f"reconstructionsource resolver is missing: {segment.source_id}"
        )
        raw = np.asarray(
            resolver.read_radar(segment.source_segment_id, slice(start, stop)), dtype=np.float32
        )
        _require(
            raw.shape == (count, 64, 116, 32, 2), f"{segment.sample_id} raw radar shape differs"
        )
        offset = segment.target_offset + start
        raw_padded = np.zeros((self.window_frames, 64, 116, 32, 2), dtype=np.float32)
        target_padded = np.zeros((self.window_frames, MHR_STATE_DIMENSION), dtype=np.float32)
        pose_padded = np.zeros((self.window_frames, 2, 24, 3), dtype=np.float32)
        temporal_mask = np.zeros(self.window_frames, dtype=bool)
        target_mask = np.zeros(self.window_frames, dtype=bool)
        raw_padded[:count] = raw
        target_padded[:count] = self.target[offset : offset + count]
        pose_padded[:count] = self.global_pose[offset : offset + count]
        temporal_mask[:count] = True
        target_mask[:count] = self.target_valid[offset : offset + count]
        _require(bool(temporal_mask.any()), f"{segment.sample_id} has an empty temporal window")
        _require(bool(np.isfinite(target_padded[:count]).all()), "MHR target is non-finite")
        return {
            "raw_radar": raw_padded,
            "target_mhr191": target_padded,
            "target_global_pose": pose_padded,
            "temporal_mask": temporal_mask,
            "target_mask": target_mask,
            "sample_id": segment.sample_id,
            "reference_id": segment.reference_id,
            "source_id": segment.source_id,
            "segment_index": segment_index,
            "sampler_cycle": sampler_cycle,
            "window_start": start,
        }

    def random_window(
        self, segment_index: int, sampler_cycle: int
    ) -> dict[str, np.ndarray | str | int]:
        segment = self.segments[segment_index]
        start = deterministic_window_start(
            frame_count=segment.frame_count,
            window_frames=self.window_frames,
            base_seed=self.window_seed,
            sampler_cycle=sampler_cycle,
            sample_id=segment.sample_id,
        )
        return self._read_window(
            segment_index=segment_index, start=start, sampler_cycle=sampler_cycle
        )

    def fixed_window(self, segment_index: int, start: int) -> dict[str, np.ndarray | str | int]:
        return self._read_window(segment_index=segment_index, start=start, sampler_cycle=-1)

    def __getitem__(self, request: tuple[int, int]) -> dict[str, np.ndarray | str | int]:
        _require(
            isinstance(request, tuple) and len(request) == 2,
            "reconstructiondataloader request must be (segment_index,sampler_cycle)",
        )
        return self.random_window(int(request[0]), int(request[1]))


class RankCycleBatchSampler:
    def __init__(
        self,
        *,
        segment_count: int,
        world_size: int,
        rank: int,
        per_device_batch: int,
        base_seed: int,
        sampler_cycle: int,
        start_batch: int,
    ) -> None:
        _require(
            0 <= rank < world_size and start_batch >= 0,
            "reconstructionrank or batch offset differs",
        )
        all_batches = deterministic_cycle_batches(
            segment_count=segment_count,
            world_size=world_size,
            per_device_batch=per_device_batch,
            base_seed=base_seed,
            sampler_cycle=sampler_cycle,
        )
        self._batches = all_batches[rank][start_batch:]
        self.sampler_cycle = sampler_cycle

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for batch in self._batches:
            yield [(segment_index, self.sampler_cycle) for segment_index in batch]

    def __len__(self) -> int:
        return len(self._batches)


def collate_windows(items: Sequence[Mapping[str, np.ndarray | str | int]]) -> dict[str, Any]:
    import torch

    _require(items, "reconstructioncannot collate an empty window batch")
    array_keys = (
        "raw_radar",
        "target_mhr191",
        "target_global_pose",
        "temporal_mask",
        "target_mask",
    )
    result: dict[str, Any] = {
        key: torch.from_numpy(np.stack([cast(np.ndarray, item[key]) for item in items]))
        for key in array_keys
    }
    result["sample_id"] = [str(item["sample_id"]) for item in items]
    result["reference_id"] = [str(item["reference_id"]) for item in items]
    result["source_id"] = [str(item["source_id"]) for item in items]
    result["segment_index"] = [int(item["segment_index"]) for item in items]
    result["window_start"] = [int(item["window_start"]) for item in items]
    return result


def _label_parts(label_root: Path) -> tuple[Path, ...]:
    parts = tuple(sorted((label_root / "parts").glob("*.parquet")))
    _require(parts, f"camera-label parquet parts are missing under {label_root}")
    return parts


def _validate_label_manifest(label_root: Path, *, expected_target_column: str) -> dict[str, Any]:
    manifest = _read_json(label_root / "build_manifest.json", "camera-label build manifest")
    _require(
        manifest.get("label_schema") == "mmprism.cascade_camera_global_label.v1",
        f"{label_root.name} label schema differs",
    )
    _require(
        manifest.get("training_target_column") in {None, expected_target_column}
        or manifest.get("training_target_column") == expected_target_column,
        f"{label_root.name} target column differs",
    )
    post = _mapping(manifest.get("post_processing"), f"{label_root.name}.post_processing")
    _require(
        post.get("local_smoothing") == "none (hands/fingers/arms untouched)",
        "camera-label smooths local geometry",
    )
    _require(
        post.get("operation") == "pose = full24 + (smooth_W5(root) - root)",
        "camera-label is not root-only W5",
    )
    return manifest


def _iter_label_groups(
    label_root: Path, *, columns: Sequence[str]
) -> Iterator[dict[str, list[Any]]]:
    try:
        from pyarrow import parquet
    except ImportError as error:
        raise DatasetError("reconstruction binding requires pyarrow parquet support") from error
    current_id: str | None = None
    current: dict[str, list[Any]] = {column: [] for column in columns}
    seen: set[str] = set()
    for part in _label_parts(label_root):
        source = parquet.ParquetFile(part)
        for batch in source.iter_batches(batch_size=65536, columns=list(columns)):
            rows = batch.to_pydict()
            count = batch.num_rows
            for row in range(count):
                sample_id = rows["sample_id"][row]
                _require(isinstance(sample_id, str) and sample_id, "camera-label sample ID differs")
                if current_id is not None and sample_id != current_id:
                    _require(
                        current_id not in seen,
                        f"camera-label sample {current_id} is non-contiguous",
                    )
                    seen.add(current_id)
                    yield current
                    current = {column: [] for column in columns}
                current_id = sample_id
                for column in columns:
                    current[column].append(rows[column][row])
    if current_id is not None:
        _require(current_id not in seen, f"camera-label sample {current_id} is non-contiguous")
        yield current


def _label_summaries(label_root: Path) -> dict[str, LabelSummary]:
    summaries: dict[str, LabelSummary] = {}
    for group in _iter_label_groups(
        label_root, columns=("sample_id", "frame_index", "episode_frame_count", "validity")
    ):
        sample_ids = group["sample_id"]
        sample_id = str(sample_ids[0])
        frame_index = np.asarray(group["frame_index"], dtype=np.int64)
        expected_count = int(group["episode_frame_count"][0])
        _require(len(frame_index) == expected_count, f"{sample_id} label frame count differs")
        _require(
            np.array_equal(frame_index, np.arange(expected_count, dtype=np.int64)),
            f"{sample_id} label frame indices are not local-contiguous",
        )
        _require(sample_id not in summaries, f"duplicate camera-label sample {sample_id}")
        summaries[sample_id] = LabelSummary(
            frame_count=expected_count,
            valid_frame_count=int(np.asarray(group["validity"], dtype=bool).sum()),
        )
    _require(summaries, f"camera-label {label_root} has no samples")
    return summaries


def build_binding(config: Mapping[str, Any]) -> tuple[tuple[BoundSegment, ...], dict[str, Any]]:
    data = _mapping(config.get("data"), "data")
    membership_path = _section_path(config, "data", "membership")
    records = parse_membership(_read_json(membership_path, "reconstruction membership"))
    expected_segments = _mapping(data.get("expected_segments"), "expected_segments")
    _require(
        role_counts(records)
        == {role: int(expected_segments[role]) for role in ("train", "validation", "test")},
        "reconstruction membership segment counts differ from reconstruction contract",
    )
    target_column = data.get("rootw5_target_column")
    _require(
        isinstance(target_column, str) and target_column == "pose3d_global", "target column differs"
    )
    label_roots = {
        corpus: _resolve_nested_path(config, section="data", group="camera_labels", key=corpus)
        for corpus in ("old", "new", "v4")
    }
    manifests = {
        corpus: _validate_label_manifest(root, expected_target_column=target_column)
        for (corpus, root) in label_roots.items()
    }
    summaries_by_corpus = {corpus: _label_summaries(root) for (corpus, root) in label_roots.items()}
    resolvers = _load_source_resolvers(config)
    record_by_sample = {record.sample_id: record for record in records}
    label_samples = set().union(*(set(values) for values in summaries_by_corpus.values()))
    _require(
        set(record_by_sample) == label_samples,
        "reconstruction membership and camera-label sample IDs do not match exactly",
    )
    bounded_without_offsets: list[BoundSegment] = []
    for record in records:
        summary = summaries_by_corpus[record.corpus][record.sample_id]
        resolver = resolvers.get(record.source_id)
        _require(
            resolver is not None, f"reconstruction source resolver is missing: {record.source_id}"
        )
        locator = resolver.locator(record.source_segment_id)
        _require(
            locator.frame_count == summary.frame_count,
            f"{record.sample_id} compact/camera-label frame count differs",
        )
        raw_root = _resolve_nested_path(
            config, section="data", group="raw_mhr_samples", key=record.corpus
        )
        s2_root = _resolve_nested_path(config, section="data", group="s2_roots", key=record.corpus)
        _require(
            (raw_root / f"{locator.item_id}.npz").is_file(),
            f"raw MHR parent missing: {locator.item_id}",
        )
        _require(
            (s2_root / f"{locator.item_id}.npz").is_file(), f"S2 parent missing: {locator.item_id}"
        )
        bounded_without_offsets.append(
            BoundSegment(
                corpus=record.corpus,
                role=record.role,
                source_id=record.source_id,
                sample_id=record.sample_id,
                source_segment_id=record.source_segment_id,
                reference_id=record.reference_id,
                parent_id=locator.item_id,
                frame_count=summary.frame_count,
                valid_frame_count=summary.valid_frame_count,
                target_offset=-1,
            )
        )
    bounded: list[BoundSegment] = []
    role_frame_counts: dict[str, int] = {}
    for role in ("train", "validation", "test"):
        offset = 0
        for segment in sorted(
            (value for value in bounded_without_offsets if value.role == role),
            key=lambda value: (value.source_id, value.sample_id),
        ):
            bounded.append(BoundSegment(**{**asdict(segment), "target_offset": offset}))
            offset += segment.frame_count
        role_frame_counts[role] = offset
    expected_frames = _mapping(data.get("expected_prevalid_frames"), "expected_prevalid_frames")
    _require(
        role_frame_counts
        == {role: int(expected_frames[role]) for role in ("train", "validation", "test")},
        "reconstruction pre-validity frame counts differ from contract",
    )
    report = {
        "schema_version": BINDING_SCHEMA,
        "membership": str(membership_path),
        "label_roots": {key: str(value) for (key, value) in label_roots.items()},
        "label_builds": {key: manifests[key].get("build_id") for key in manifests},
        "segment_counts": role_counts(records),
        "prevalid_frame_counts": role_frame_counts,
        "valid_frame_counts": {
            role: sum(segment.valid_frame_count for segment in bounded if segment.role == role)
            for role in ("train", "validation", "test")
        },
        "rootw5": "root-only edge-replicated W5; local joints untouched",
    }
    return (tuple(bounded), report)


class _ParentFields:
    def __init__(self, *, capacity: int = 8) -> None:
        self.capacity = capacity
        self.values: OrderedDict[tuple[str, str], dict[str, np.ndarray]] = OrderedDict()

    def load(self, *, raw_path: Path, s2_path: Path) -> dict[str, np.ndarray]:
        key = (str(raw_path), str(s2_path))
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        required_raw = (
            "raw_body_pose_params",
            "raw_hand_pose_params",
            "raw_global_rot",
            "frame_indices",
        )
        required_s2 = (
            "camera_translation_m",
            "frozen_scale_vector",
            "lambda_to_standard",
            "frame_indices",
        )
        try:
            with (
                np.load(raw_path, allow_pickle=False) as raw,
                np.load(s2_path, allow_pickle=False) as s2,
            ):
                _require(
                    all(name in raw.files for name in required_raw),
                    f"raw MHR fields missing: {raw_path}",
                )
                _require(
                    all(name in s2.files for name in required_s2), f"S2 fields missing: {s2_path}"
                )
                value = {
                    "body": np.asarray(raw["raw_body_pose_params"], dtype=np.float32).copy(),
                    "hand": np.asarray(raw["raw_hand_pose_params"], dtype=np.float32).copy(),
                    "global": np.asarray(raw["raw_global_rot"], dtype=np.float32).copy(),
                    "raw_frame": np.asarray(raw["frame_indices"], dtype=np.int64).copy(),
                    "translation": np.asarray(s2["camera_translation_m"], dtype=np.float32).copy(),
                    "morphology": np.asarray(s2["frozen_scale_vector"], dtype=np.float32).copy(),
                    "lambda": np.asarray(s2["lambda_to_standard"], dtype=np.float32).copy(),
                    "s2_frame": np.asarray(s2["frame_indices"], dtype=np.int64).copy(),
                }
        except OSError as error:
            raise DatasetError(f"cannot open raw/S2 MHR parent: {error}") from error
        frames = value["hand"].shape[0]
        _require(
            value["hand"].shape == (frames, 108)
            and value["body"].shape == (frames, 133)
            and (value["global"].shape == (frames, 3))
            and (value["translation"].shape == (frames, 3)),
            f"raw/S2 MHR field shapes differ for {raw_path.name}",
        )
        _require(value["morphology"].shape == (28,), f"MHR morphology differs for {raw_path.name}")
        _require(value["lambda"].shape == (), f"MHR standard scale differs for {raw_path.name}")
        _require(
            np.array_equal(value["raw_frame"], value["s2_frame"]),
            f"raw/S2 frame index differs for {raw_path.name}",
        )
        _require(
            all(np.isfinite(item).all() for item in value.values()),
            f"non-finite raw/S2 MHR fields: {raw_path.name}",
        )
        self.values[key] = value
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def _state_from_group(
    *, fields: Mapping[str, np.ndarray], absolute_indices: np.ndarray, delta_root: np.ndarray
) -> np.ndarray:
    _require(absolute_indices.ndim == 1, "MHR absolute indices differ")
    _require(
        len(absolute_indices) > 0
        and int(absolute_indices.min()) >= 0
        and (int(absolute_indices.max()) < fields["hand"].shape[0]),
        "MHR absolute indices escape parent",
    )
    _require(delta_root.shape == (len(absolute_indices), 3), "camera-label root delta differs")
    upper = fields["body"][absolute_indices][:, list(UPPER_BODY_INDICES)]
    state = np.concatenate(
        (
            upper,
            fields["hand"][absolute_indices, :54],
            fields["hand"][absolute_indices, 54:],
            fields["global"][absolute_indices],
            fields["translation"][absolute_indices] + delta_root,
            np.broadcast_to(fields["morphology"], (len(absolute_indices), 28)),
            np.full((len(absolute_indices), 1), float(fields["lambda"]), dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    _require(state.shape == (len(absolute_indices), MHR_STATE_DIMENSION), "MHR state width differs")
    _require(bool(np.isfinite(state).all()), "assembled MHR state is non-finite")
    return state


def _target_paths(target_root: Path, role: str) -> tuple[Path, Path]:
    return (target_root / role / "targets.npy", target_root / role / "valid.npy")


def _global_pose_path(target_root: Path, role: str) -> Path:
    return target_root / role / "pose3d_global.npy"


def _fit_train_normalization(target_root: Path) -> dict[str, np.ndarray]:
    (targets_path, valid_path) = _target_paths(target_root, "train")
    targets = np.load(targets_path, mmap_mode="r")
    valid = np.load(valid_path, mmap_mode="r")
    _require(targets.shape == (len(valid), MHR_STATE_DIMENSION), "train target cache shape differs")
    total = 0
    sum_values = np.zeros(MHR_STATE_DIMENSION, dtype=np.float64)
    square_values = np.zeros(MHR_STATE_DIMENSION, dtype=np.float64)
    for start in range(0, len(valid), 8192):
        stop = min(start + 8192, len(valid))
        values = np.asarray(
            targets[start:stop][np.asarray(valid[start:stop], dtype=bool)], dtype=np.float64
        )
        if len(values):
            total += len(values)
            sum_values += values.sum(axis=0)
            square_values += np.square(values).sum(axis=0)
    _require(total > 0, "reconstruction train target cache has no valid frames")
    mean = sum_values / total
    variance = np.maximum(square_values / total - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    active = scale >= 1e-06
    scale = np.where(active, scale, 1.0)
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "active_dimensions": active.astype(bool),
        "valid_train_frames": np.asarray(total, dtype=np.int64),
    }


def prepare_global_pose_cache(
    config: Mapping[str, Any], *, segments: Sequence[BoundSegment], target_root: Path
) -> dict[str, Any]:
    paths = {role: _global_pose_path(target_root, role) for role in ("train", "validation", "test")}
    if all(path.is_file() for path in paths.values()):
        for role, path in paths.items():
            value = np.load(path, mmap_mode="r")
            expected = sum(segment.frame_count for segment in segments if segment.role == role)
            _require(value.shape == (expected, 2, 24, 3), f"{role} global pose cache shape differs")
        return {
            "paths": {role: str(path) for (role, path) in paths.items()},
            "shape_per_frame": [2, 24, 3],
            "reused": True,
        }
    _require(not any(path.exists() for path in paths.values()), "global pose cache is partial")
    pose: dict[str, np.memmap] = {}
    written: dict[str, np.ndarray] = {}
    for role, path in paths.items():
        count = sum(segment.frame_count for segment in segments if segment.role == role)
        pose[role] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(count, 2, 24, 3)
        )
        written[role] = np.zeros(count, dtype=bool)
    by_sample = {segment.sample_id: segment for segment in segments}
    for corpus in ("old", "new", "v4"):
        label_root = _resolve_nested_path(config, section="data", group="camera_labels", key=corpus)
        for group in _iter_label_groups(label_root, columns=("sample_id", "pose3d_global")):
            sample_id = str(group["sample_id"][0])
            segment = by_sample.get(sample_id)
            _require(
                segment is not None and segment.corpus == corpus,
                f"unexpected global-pose camera-label sample {sample_id}",
            )
            values = np.asarray(group["pose3d_global"], dtype=np.float32).reshape(-1, 2, 24, 3)
            _require(
                values.shape[0] == segment.frame_count, f"{sample_id} global pose count differs"
            )
            start = segment.target_offset
            stop = start + segment.frame_count
            _require(
                not bool(written[segment.role][start:stop].any()),
                f"duplicate global pose rows: {sample_id}",
            )
            pose[segment.role][start:stop] = values
            written[segment.role][start:stop] = True
    for role, values in pose.items():
        _require(bool(written[role].all()), f"global pose cache has unwritten {role} rows")
        values.flush()
    return {
        "paths": {role: str(path) for (role, path) in paths.items()},
        "shape_per_frame": [2, 24, 3],
        "reused": False,
    }


def _load_source_resolvers(config):
    resolvers = {}
    for catalog_key in config["data"]["compact_sources"].values():
        catalog = _read_json(resolve(catalog_key), "compact source catalog")
        for source_id, row in _mapping(catalog.get("sources"), "sources").items():
            _require(source_id not in resolvers, f"duplicate compact source {source_id}")
            items = resolve(row["compact_items_root"])
            manifest = resolve(row["segments_manifest"])
            _require(
                items.is_dir() and manifest.is_file(), f"{source_id} compact source unavailable"
            )
            resolvers[source_id] = StatelessSegmentResolver(items, manifest)
    return resolvers


def prepare_target_cache(
    config: Mapping[str, Any], *, segments: Sequence[BoundSegment], output_root: Path
) -> dict[str, Any]:
    cache = _mapping(config.get("cache"), "cache")
    target_root_value = cache.get("target_root")
    _require(isinstance(target_root_value, str) and target_root_value, "target cache root missing")
    target_root = resolve(target_root_value)
    report_path = target_root / "target_cache_report.json"
    if report_path.is_file():
        report = _read_json(report_path, "existing reconstructiontarget cache report")
        _require(
            report.get("schema_version") == TARGET_CACHE_SCHEMA,
            "existing target cache schema differs",
        )
        for role in ("train", "validation", "test"):
            (target_path, valid_path) = _target_paths(target_root, role)
            _require(
                target_path.is_file() and valid_path.is_file(),
                "existing target cache is incomplete",
            )
        return report
    target_root.mkdir(parents=True, exist_ok=True)
    by_role = {
        role: tuple(value for value in segments if value.role == role)
        for role in ("train", "validation", "test")
    }
    targets: dict[str, np.memmap] = {}
    valid: dict[str, np.memmap] = {}
    written: dict[str, np.ndarray] = {}
    for role, role_segments in by_role.items():
        count = sum(value.frame_count for value in role_segments)
        (target_path, valid_path) = _target_paths(target_root, role)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        targets[role] = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.float32, shape=(count, MHR_STATE_DIMENSION)
        )
        valid[role] = np.lib.format.open_memmap(
            valid_path, mode="w+", dtype=np.bool_, shape=(count,)
        )
        written[role] = np.zeros(count, dtype=bool)
    by_sample = {segment.sample_id: segment for segment in segments}
    resolvers = _load_source_resolvers(config)
    panel_limit = int(
        _mapping(config.get("evaluation"), "evaluation").get("identity_panel_frames_per_source", 16)
    )
    panel_state: dict[str, list[np.ndarray]] = {f"p{index:02d}": [] for index in range(1, 19)}
    panel_pose: dict[str, list[np.ndarray]] = {f"p{index:02d}": [] for index in range(1, 19)}
    parents = _ParentFields()
    columns = (
        "sample_id",
        "frame_index",
        "episode_frame_count",
        "validity",
        "delta_root_m",
        "pose3d_global",
    )
    for corpus in ("old", "new", "v4"):
        label_root = _resolve_nested_path(config, section="data", group="camera_labels", key=corpus)
        raw_root = _resolve_nested_path(config, section="data", group="raw_mhr_samples", key=corpus)
        s2_root = _resolve_nested_path(config, section="data", group="s2_roots", key=corpus)
        for group in _iter_label_groups(label_root, columns=columns):
            sample_id = str(group["sample_id"][0])
            segment = by_sample.get(sample_id)
            _require(
                segment is not None and segment.corpus == corpus,
                f"unexpected camera-label sample {sample_id}",
            )
            frame_index = np.asarray(group["frame_index"], dtype=np.int64)
            _require(
                np.array_equal(frame_index, np.arange(segment.frame_count, dtype=np.int64)),
                f"{sample_id} cache group frames differ",
            )
            resolver = resolvers[segment.source_id]
            locator = resolver.locator(segment.source_segment_id)
            absolute = frame_index + locator.frame_start
            fields = parents.load(
                raw_path=raw_root / f"{segment.parent_id}.npz",
                s2_path=s2_root / f"{segment.parent_id}.npz",
            )
            state = _state_from_group(
                fields=fields,
                absolute_indices=absolute,
                delta_root=np.asarray(group["delta_root_m"], dtype=np.float32),
            )
            start = segment.target_offset
            stop = start + segment.frame_count
            _require(
                not bool(written[segment.role][start:stop].any()),
                f"duplicate target rows: {sample_id}",
            )
            targets[segment.role][start:stop] = state
            valid_values = np.asarray(group["validity"], dtype=bool)
            valid[segment.role][start:stop] = valid_values
            written[segment.role][start:stop] = True
            if segment.role == "train" and len(panel_state[segment.source_id]) < panel_limit:
                room = panel_limit - len(panel_state[segment.source_id])
                selected = np.flatnonzero(valid_values)[:room]
                if len(selected):
                    panel_state[segment.source_id].extend(state[selected])
                    panel_pose[segment.source_id].extend(
                        np.asarray(group["pose3d_global"], dtype=np.float32).reshape(-1, 2, 24, 3)[
                            selected
                        ]
                    )
    for role in ("train", "validation", "test"):
        _require(bool(written[role].all()), f"target cache has unwritten {role} rows")
        targets[role].flush()
        valid[role].flush()
    selected_state = [value for source in sorted(panel_state) for value in panel_state[source]]
    selected_pose = [value for source in sorted(panel_pose) for value in panel_pose[source]]
    _require(
        selected_state and len(selected_state) == len(selected_pose), "GT identity panel is empty"
    )
    panel_path = target_root / "gt_decode_identity_panel.npz"
    _write_npz(
        panel_path,
        state=np.asarray(selected_state, dtype=np.float32),
        pose3d_global=np.asarray(selected_pose, dtype=np.float32),
    )
    normalization = _fit_train_normalization(target_root)
    normalization_path = target_root / "train_only_normalization.npz"
    _write_npz(normalization_path, **normalization)
    report = {
        "schema_version": TARGET_CACHE_SCHEMA,
        "target_root": str(target_root),
        "state_dimension": MHR_STATE_DIMENSION,
        "state_order": [
            "upper_articulation_48",
            "left_hand_articulation_54",
            "right_hand_articulation_54",
            "global_rotation_3",
            "camera_translation_rootw5_3",
            "frozen_morphology_28",
            "lambda_to_standard_1",
        ],
        "role_frames": {role: int(targets[role].shape[0]) for role in targets},
        "role_valid_frames": {
            role: int(np.asarray(valid[role], dtype=bool).sum()) for role in valid
        },
        "normalization": {
            "path": str(normalization_path),
            "fit_role": "train",
            "valid_train_frames": int(normalization["valid_train_frames"]),
            "active_dimensions": int(normalization["active_dimensions"].sum()),
            "inactive_dimensions": int((~normalization["active_dimensions"]).sum()),
        },
        "gt_decode_identity_panel": str(panel_path),
        "target_is_task_local_read_performance_cache": True,
        "no_target_free_dense_cache_created": True,
    }
    _write_json(report_path, report)
    return report


class StatelessSegmentResolver(CompactSegmentResolver):
    """Read a compact radar record without persistent loader state."""

    def _packed_radar(self, item_id: str) -> PackedRadar:
        if item_id not in self._radar_maps:
            self._radar_maps[item_id] = PackedRadar(self.item(item_id).radar_payload)
        return self._radar_maps[item_id]
