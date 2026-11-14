"""Nine-signer radar windows and camera-pose supervision used in personal adaptation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from mmprism.data.cascade_compact_raw import unpack_fixed16_iq
from mmprism.data.compact_radar import PackedRadar
from mmprism.release.paths import resolve

SIGNERS = ("p01", "p03", "p06", "p10", "p11", "p15", "p17", "p36", "p37")
WINDOW = 32


@dataclass(frozen=True)
class Segment:
    row: dict
    person: str
    cell: str

    @property
    def frame_count(self) -> int:
        return int(self.row["frames"])

    @property
    def sample_id(self) -> str:
        return self.row["segment_id"]


@lru_cache(maxsize=9)
def person_rows(signer: str) -> dict:
    """Segment rows for one signer, read from the manifest alone.

    The inference path needs the frame count and the segment offsets and nothing
    else, so this accessor never opens the camera pose or the validity mask.
    """
    if signer not in SIGNERS:
        raise ValueError(f"Unknown signer {signer!r}; choose one of {', '.join(SIGNERS)}")
    path = resolve("pack.pool", signer=signer)
    return {
        row["segment_id"]: row
        for line in (path / "manifest.jsonl").read_text().splitlines()
        if (row := json.loads(line))
    }


def _person(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    rows = {
        row["segment_id"]: row
        for line in (path / "manifest.jsonl").read_text().splitlines()
        if (row := json.loads(line))
    }
    return (
        rows,
        np.load(path / "pose_camera_m.npy", mmap_mode="r"),
        np.load(path / "valid_mask.npy", mmap_mode="r"),
    )


def person_data(signer: str) -> tuple[dict, np.ndarray, np.ndarray]:
    if signer not in SIGNERS:
        raise ValueError(f"Unknown signer {signer!r}; choose one of {', '.join(SIGNERS)}")
    return _person(resolve("pack.pool", signer=signer))


def calibration_ids(signer: str) -> list[str]:
    payload = json.loads(resolve("pack.calibration_selection").read_text())
    ids = payload["persons"][signer]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError(f"{signer}: calibration_selection must contain 50 distinct segments")
    return ids


def cell_segments(cell: str, signer: str) -> list[Segment]:
    cells = json.loads(resolve("pack.splits").read_text())["cells"]
    rows, _, _ = person_data(signer)
    ids = [
        sid
        for group in cells[cell]["groups"]
        if group["person"] == signer
        for sid in group["segment_ids"]
    ]
    return [Segment(rows[sid], signer, cell) for sid in ids]


class CellDataset:
    """The original non-overlapping window reader, with pack-relative radar references."""

    window_frames = WINDOW

    def __init__(self, segments):
        self.segments = tuple(segments)
        self._packed: dict[Path, PackedRadar] = {}

    def __len__(self):
        return len(self.segments)

    def fixed_window(self, index: int, start: int) -> dict[str, np.ndarray]:
        segment = self.segments[index]
        row = segment.row
        reference = row["radar_ref"]
        length = min(WINDOW, segment.frame_count - start)
        if not 0 <= start < segment.frame_count:
            raise IndexError(f"{segment.sample_id}: window starts outside the segment")
        path = resolve("pack.radar", signer=segment.person, segment=segment.sample_id)
        packed = self._packed.get(path)
        if packed is None:
            packed = self._packed[path] = PackedRadar(path)
        offset = int(reference["frame_offset"]) + start
        raw = np.zeros((WINDOW, 64, 116, 32, 2), np.float32)
        raw[:length] = unpack_fixed16_iq(packed[offset : offset + length])
        _, poses, masks = person_data(segment.person)
        offset = int(row["frame_offset"]) + start
        pose = np.zeros((WINDOW, 2, 24, 3), np.float32)
        pose[:length] = np.asarray(poses[offset : offset + length])
        temporal = np.zeros(WINDOW, bool)
        temporal[:length] = True
        target_mask = np.zeros(WINDOW, bool)
        target_mask[:length] = masks[offset : offset + length]
        return {
            "raw_radar": raw,
            "temporal_mask": temporal,
            "target_mask": target_mask,
            "target_global_pose": pose,
        }

    def whole_segment(self, index: int) -> np.ndarray:
        segment = self.segments[index]
        pieces = []
        for start in range(0, segment.frame_count, WINDOW):
            item = self.fixed_window(index, start)
            pieces.append(item["raw_radar"][: int(item["temporal_mask"].sum())])
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def targets(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        segment = self.segments[index]
        _, poses, masks = person_data(segment.person)
        offset = int(segment.row["frame_offset"])
        return (
            np.asarray(poses[offset : offset + segment.frame_count]),
            np.asarray(masks[offset : offset + segment.frame_count]),
        )
