"""L2 frame-level episode-atomic Parquet assembly for the mesh annotation corpus.

Consumes the per-episode fusion export corpus
(``mmprism.csl_daily_mesh_episode_export.v1`` npz records plus launcher
progress sidecars) and publishes the canonical training delivery registered as
``mmprism.csl_daily_mesh_annotation_stabilized.v1`` (layout L2, see
``DATA_DELIVERY_PARQUET.md``):

- one row = one frame; parts target ~4,096 rows packed greedily in
  ``sample_id`` byte order so an episode never straddles parts;
- one row group per episode;
- ``episode_index.parquet`` is the merged split+locator; chunk assignment
  is split-blind; split="unassigned" marks storage-complete episodes the
  split table does not cover (readers select by split name);
- payloads use FLAT fixed-size lists with logical shapes recorded in
  ``schema.json:static_dimensions``;
- a build-level ``normalization_stats.json`` sidecar carries per-axis
  mean/std over frames of TRAIN-split eligible episodes only, under the
  per-frame per-person joint-0 root-center convention.

Only episodes whose launcher progress is terminal ``completed`` are
emitted.  Quarantined/failed episodes are excluded explicitly and
accounted per split; this builder never fills an exclusion from another
pose source.  PyArrow is imported lazily (``data-parquet`` extra).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MESH_L2_SCHEMA_NAME = "mmprism.csl_daily_mesh_annotation_stabilized.v1"

MESH_NORMALIZATION_PROTOCOL_ID = "mmprism.mesh_pose_normalization.train_only_v1"

MESH_POSE_SOURCE = "mesh_stabilized_fk_v2"

POSE_FLAT = 2 * 24 * 3

ROOT_JOINT_INDEX = 0  # legacy graph root (shoulder) — bilateral, per person


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_digest(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
