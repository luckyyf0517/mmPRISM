"""Fixed-scale compact radar storage and record metadata.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

CASCADE_COMPACT_ITEM_SCHEMA = "cascade.collection-item.v2"

MMHAND_COMPACT_ITEM_SCHEMA = "mmprism.mmhand_collected_compact_raw_item.v1"

COMPACT_STORAGE_SCHEMA = "mmprism.compact_raw_storage.v1"

VIDEO_STORAGE_SCHEMA = "mmprism.lossless_rgb_video.v1"

RADAR_TAIL = (64, 116, 32, 2)

COLOR_TAIL = (360, 640, 3)

PACKED_RADAR_NAME = "mmwave.fp16.npy"

VIDEO_NAME = "color.mp4"

VIDEO_METADATA_NAME = "video.json"

RADAR_SCALE_DIVISOR = 16.0


class CompactRawStorageError(RuntimeError):
    """Raised when a compact raw payload violates its contract."""


@dataclass(frozen=True, slots=True)
class CompactCascadeItem:
    item_dir: Path
    logical_item_id: str
    radar_payload: Path
    radar_shape: tuple[int, int, int, int, int]
    video_payload: Path
    video_shape: tuple[int, int, int, int]
    frame_rate_hz: float
    channel_order: str


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompactRawStorageError(f"unable to read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CompactRawStorageError(f"{description} must be a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CompactRawStorageError(f"{description} must be an object")
    return cast(Mapping[str, object], value)


def _text(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompactRawStorageError(f"{description} must be non-empty text")
    return value


def _positive_float(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CompactRawStorageError(f"{description} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CompactRawStorageError(f"{description} must be a finite positive number")
    return result


def _shape(value: object, description: str, tail: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != len(tail) + 1:
        raise CompactRawStorageError(f"{description} has an invalid rank")
    if any(isinstance(component, bool) or not isinstance(component, int) for component in value):
        raise CompactRawStorageError(f"{description} must contain integer dimensions")
    result = tuple(cast(int, component) for component in value)
    if result[0] < 1 or result[1:] != tail:
        raise CompactRawStorageError(f"{description} must have tail {tail}, got {result}")
    return result


def _radar_shape(value: object, description: str) -> tuple[int, int, int, int, int]:
    return cast(tuple[int, int, int, int, int], _shape(value, description, RADAR_TAIL))


def _video_shape(value: object, description: str) -> tuple[int, int, int, int]:
    return cast(tuple[int, int, int, int], _shape(value, description, COLOR_TAIL))


def _load_array(path: Path, description: str) -> np.ndarray:
    try:
        return cast(np.ndarray, np.load(path, mmap_mode="r", allow_pickle=False))
    except (OSError, ValueError) as error:
        raise CompactRawStorageError(f"unable to load {description} {path}: {error}") from error


def unpack_fixed16_iq(values: np.ndarray) -> np.ndarray:
    """Restore fixed `/16` FP16 IQ to float32 before any radar processing."""

    if values.dtype != np.float16:
        raise CompactRawStorageError(f"packed radar must be float16, got {values.dtype}")
    restored = values.astype(np.float32) * RADAR_SCALE_DIVISOR
    if not bool(np.isfinite(restored).all()):
        raise CompactRawStorageError("restored radar contains non-finite values")
    return restored


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise CompactRawStorageError("ffmpeg is required for lossless RGB video storage")
    return executable


def _decode_lossless_video(payload: Path, shape: tuple[int, int, int, int]) -> np.ndarray:
    command = [
        _ffmpeg(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(payload),
        "-map",
        "0:v:0",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise CompactRawStorageError(f"lossless RGB video decoding failed: {error}")
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if len(result.stdout) != expected_bytes:
        raise CompactRawStorageError(
            f"decoded RGB byte count differs: {len(result.stdout)} != {expected_bytes}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(shape).copy()


def load_compact_cascade_item(item_dir: Path) -> CompactCascadeItem:
    """Validate one compact cascade v2 item header without materializing payloads."""

    root = item_dir.resolve()
    item = _read_json(root / "item.json", "compact cascade item metadata")
    schema = item.get("schema_version")
    if schema not in (CASCADE_COMPACT_ITEM_SCHEMA, MMHAND_COMPACT_ITEM_SCHEMA):
        raise CompactRawStorageError(f"compact cascade item schema mismatch: {root}")
    # Compact records without an inline vision block use their separate
    # metadata sidecar for geometry and channel order.
    cascade_envelope = schema == CASCADE_COMPACT_ITEM_SCHEMA
    radar = _mapping(item.get("radar"), "compact cascade item.radar")
    vision = (
        _mapping(item.get("vision"), "compact cascade item.vision") if cascade_envelope else None
    )
    storage = _mapping(item.get("storage"), "compact cascade item.storage")
    if storage.get("schema_version") != COMPACT_STORAGE_SCHEMA:
        raise CompactRawStorageError("compact storage schema mismatch")
    radar_storage = _mapping(storage.get("radar"), "compact cascade storage.radar")
    vision_storage = _mapping(storage.get("vision"), "compact cascade storage.vision")
    if (
        radar_storage.get("payload") != f"radar/{PACKED_RADAR_NAME}"
        or radar_storage.get("source_dtype") != "float32"
        or radar_storage.get("storage_dtype") != "float16"
        or radar_storage.get("scale_divisor") != 16
        or radar_storage.get("restored_dtype") != "float32"
    ):
        raise CompactRawStorageError(
            "compact radar storage metadata differs from fixed /16 contract"
        )
    if (
        vision_storage.get("payload") != f"vision/{VIDEO_NAME}"
        or vision_storage.get("metadata") != f"vision/{VIDEO_METADATA_NAME}"
    ):
        raise CompactRawStorageError("compact video storage metadata differs")
    radar_shape = _radar_shape(radar.get("tensor_shape"), "compact cascade radar shape")
    video = _read_json(root / "vision" / VIDEO_METADATA_NAME, "compact video metadata")
    if vision is not None:
        video_shape = _video_shape(vision.get("shape"), "compact cascade video shape")
        channel_order = _text(vision.get("color_order"), "compact cascade color_order")
    else:
        video_shape = _video_shape(video.get("shape"), "compact cascade video shape")
        channel_order = _text(video.get("channel_order"), "compact cascade color_order")
    if radar_shape[0] != video_shape[0]:
        raise CompactRawStorageError("compact cascade radar/video frame counts differ")
    if (
        video.get("schema_version") != VIDEO_STORAGE_SCHEMA
        or video.get("codec") != "libx264rgb"
        or video.get("crf") != 0
        or video.get("pixel_format") != "rgb24"
        or video.get("channel_order") != channel_order
        or _video_shape(video.get("shape"), "compact video shape") != video_shape
    ):
        raise CompactRawStorageError("compact video metadata differs from its contract")
    radar_payload = root / "radar" / PACKED_RADAR_NAME
    video_payload = root / "vision" / VIDEO_NAME
    if not radar_payload.is_file() or not video_payload.is_file():
        raise CompactRawStorageError("compact item payload is missing")
    return CompactCascadeItem(
        item_dir=root,
        logical_item_id=_text(item.get("logical_item_id"), "compact cascade logical_item_id"),
        radar_payload=radar_payload,
        radar_shape=radar_shape,
        video_payload=video_payload,
        video_shape=video_shape,
        frame_rate_hz=_positive_float(video.get("frame_rate_hz"), "compact video frame_rate_hz"),
        channel_order=channel_order,
    )


def load_compact_video(item: CompactCascadeItem, frame_slice: slice | None = None) -> np.ndarray:
    """Decode byte-identical native channel-order frames from a compact item."""

    decoded = _decode_lossless_video(item.video_payload, item.video_shape)
    return decoded if frame_slice is None else decoded[frame_slice]


CASCADE_SEGMENT_MANIFEST_REQUIRED_FIELDS = (
    "segment_id",
    "source_logical_item_id",
    "source_frame_start",
    "source_frame_end_exclusive",
    "color_order",
)

EXPECTED_VIDEO_TAIL = (360, 640, 3)

EXPECTED_RADAR_SEGMENT_TAIL = (64, 116, 32, 2)


@dataclass(frozen=True, slots=True)
class CompactSegmentLocator:
    """One segment reference: a parent compact item plus a frame window."""

    sample_id: str
    item_id: str
    frame_start: int
    frame_end_exclusive: int
    color_order: str

    @property
    def frame_count(self) -> int:
        return self.frame_end_exclusive - self.frame_start


class CompactSegmentResolver:
    """Resolve segment manifest references onto compact parent items.

    This is the shared read boundary for the reference-based cascade chain:
    annotation, label validation, and training all obtain radar or RGB/BGR
    frames through it and never touch copied per-segment arrays.  Radar is
    restored to float32 at this boundary (float32(packed) * 16); video is
    decoded natively and converted to BGR exactly once when the cut manifest
    declares the historical provider boundary.

    Instances cache parent headers and packed-radar memory maps, so one
    resolver per dataloader worker gives every worker an mmap cache without
    duplicating payloads.
    """

    def __init__(self, items_root: Path, segments_manifest: Path) -> None:
        self._items_root = Path(items_root).expanduser().resolve()
        manifest_path = Path(segments_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise CompactRawStorageError(f"segment manifest is missing: {manifest_path}")
        locators: dict[str, CompactSegmentLocator] = {}
        try:
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise CompactRawStorageError(f"unable to read segment manifest: {error}") from error
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise CompactRawStorageError(
                    f"{manifest_path}: malformed segment manifest row: {error}"
                ) from error
            if not isinstance(row, dict):
                raise CompactRawStorageError(f"{manifest_path}: segment row must be an object")
            missing = [
                field for field in CASCADE_SEGMENT_MANIFEST_REQUIRED_FIELDS if field not in row
            ]
            if missing:
                raise CompactRawStorageError(f"{manifest_path}: segment row is missing {missing}")
            locator = CompactSegmentLocator(
                sample_id=str(row["segment_id"]),
                item_id=str(row["source_logical_item_id"]),
                frame_start=int(row["source_frame_start"]),
                frame_end_exclusive=int(row["source_frame_end_exclusive"]),
                color_order=str(row["color_order"]),
            )
            if locator.sample_id in locators:
                raise CompactRawStorageError(
                    f"{manifest_path}: duplicate segment id {locator.sample_id}"
                )
            locators[locator.sample_id] = locator
        if not locators:
            raise CompactRawStorageError(f"{manifest_path}: segment manifest is empty")
        self._manifest_path = manifest_path
        self._locators = locators
        self._items: dict[str, CompactCascadeItem] = {}
        self._radar_maps: dict[str, np.ndarray] = {}

    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self._locators)

    @property
    def items_root(self) -> Path:
        return self._items_root

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def has(self, sample_id: str) -> bool:
        return sample_id in self._locators

    def locator(self, sample_id: str) -> CompactSegmentLocator:
        try:
            return self._locators[sample_id]
        except KeyError as error:
            raise CompactRawStorageError(
                f"unknown segment id {sample_id!r} in {self._manifest_path}"
            ) from error

    def item(self, item_id: str) -> CompactCascadeItem:
        if item_id not in self._items:
            self._items[item_id] = load_compact_cascade_item(self._items_root / item_id)
        return self._items[item_id]

    def checked_locator(self, sample_id: str) -> tuple[CompactSegmentLocator, CompactCascadeItem]:
        locator = self.locator(sample_id)
        item = self.item(locator.item_id)
        frames = item.video_shape[0]
        if locator.frame_end_exclusive > frames:
            raise CompactRawStorageError(
                f"{sample_id}: window [{locator.frame_start}, {locator.frame_end_exclusive}) "
                f"exceeds parent item {locator.item_id} with {frames} frames"
            )
        if tuple(item.video_shape[1:]) != EXPECTED_VIDEO_TAIL:
            raise CompactRawStorageError(
                f"{sample_id}: parent item geometry {item.video_shape[1:]} "
                f"is not {EXPECTED_VIDEO_TAIL}"
            )
        if tuple(item.radar_shape[1:]) != EXPECTED_RADAR_SEGMENT_TAIL:
            raise CompactRawStorageError(
                f"{sample_id}: parent radar tail {item.radar_shape[1:]} "
                f"is not {EXPECTED_RADAR_SEGMENT_TAIL}"
            )
        return locator, item

    def _packed_radar(self, item_id: str) -> np.ndarray:
        if item_id not in self._radar_maps:
            packed = _load_array(self.item(item_id).radar_payload, f"{item_id} packed radar")
            self._radar_maps[item_id] = packed
        return self._radar_maps[item_id]

    def read_radar(self, sample_id: str, frame_slice: slice | None = None) -> np.ndarray:
        """Restore one segment window (or a bounded sub-window) as float32 IQ."""

        locator, item = self.checked_locator(sample_id)
        if frame_slice is None:
            start, stop = 0, locator.frame_count
        else:
            if frame_slice.step not in (None, 1):
                raise CompactRawStorageError(f"{sample_id}: radar frame_slice step must be 1")
            start = 0 if frame_slice.start is None else frame_slice.start
            stop = locator.frame_count if frame_slice.stop is None else frame_slice.stop
        if not 0 <= start <= stop <= locator.frame_count:
            raise CompactRawStorageError(
                f"{sample_id}: radar window [{start}, {stop}) escapes the segment "
                f"[0, {locator.frame_count}) bound"
            )
        packed = self._packed_radar(locator.item_id)
        absolute_start = locator.frame_start + start
        absolute_stop = locator.frame_start + stop
        if tuple(packed.shape[1:]) != tuple(item.radar_shape[1:]):
            raise CompactRawStorageError(
                f"{locator.item_id}: packed radar shape differs from its header"
            )
        window = np.asarray(packed[absolute_start:absolute_stop])
        return unpack_fixed16_iq(window)

    def read_radar_frame(self, sample_id: str, frame_index: int) -> np.ndarray:
        """Restore exactly one segment frame as float32 [64,116,32,2]."""

        locator, _ = self.checked_locator(sample_id)
        if not 0 <= frame_index < locator.frame_count:
            raise CompactRawStorageError(
                f"{sample_id}: frame {frame_index} escapes the segment "
                f"[0, {locator.frame_count}) bound"
            )
        packed = self._packed_radar(locator.item_id)
        row = np.asarray(packed[locator.frame_start + frame_index])
        return unpack_fixed16_iq(row)

    def read_video(self, sample_id: str) -> np.ndarray:
        """Decode one segment window at the provider boundary (BGR contract)."""

        locator, item = self.checked_locator(sample_id)
        window = load_compact_video(
            item,
            slice(locator.frame_start, locator.frame_end_exclusive),
        )
        if locator.color_order != item.channel_order:
            window = window[..., ::-1]
        return np.ascontiguousarray(window)

    def read_video_group(
        self, item_id: str, sample_ids: list[str] | tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        """Decode one parent video once and return BGR windows per segment.

        Annotation runners group segments by parent and call this once per
        group so the compact video is FFmpeg-decoded a single time.
        """

        item = self.item(item_id)
        decoded = load_compact_video(item)
        windows: dict[str, np.ndarray] = {}
        for sample_id in sample_ids:
            locator, header = self.checked_locator(sample_id)
            if locator.item_id != item_id:
                raise CompactRawStorageError(
                    f"{sample_id}: segment belongs to {locator.item_id}, not {item_id}"
                )
            window = decoded[locator.frame_start : locator.frame_end_exclusive]
            if locator.color_order != header.channel_order:
                window = window[..., ::-1]
            windows[sample_id] = np.ascontiguousarray(window)
        return windows
