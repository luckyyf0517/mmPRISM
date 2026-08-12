# Parquet Training Data Delivery Contract

Status: current
Owner: Data rebuild lane
Authority scope: The data intake, radar rebuild, split, or delivery boundary represented by this page.
Last reviewed: 2026-08-12

## Scope

This document defines the only final model-ready delivery format for rebuilt data: task-specific Parquet
products. One logical training sample occupies one Parquet row. It applies to the greenfield system and does
not require legacy file compatibility.

The current delivery metadata and Arrow-schema contract is v2. The v2 translation schema adds explicit input-mode
binding and permits `pose_only` without a radar-feature column. Existing v1 deliveries remain immutable historical
artifacts and are not accepted by the v2 reader or validator.

Parquet is a delivery format, not a source of truth. It must not replace raw inputs, frozen manifests,
annotation sidecars, source-integrity reports, failure records, or intermediate arrays required to reproduce an
annotation. A product can enter formal training only after its frozen inputs, split assignment, reader, and
validation evidence have passed the gates below.

The design explicitly rejects a sparsely populated universal schema, opaque pickle/Python-object/`np.save` byte
blobs, string-based path association, direct materialization from a live sidecar directory, and promotion of
current CSL-News visual RTMW3D pose as metric radar training data.

## Immutable Layers

```text
raw/          Original source bytes; immutable after intake.
interim/      Recoverable processing: NPZ, JSON sidecar, failures, QC reports and extraction cache.
manifests/    Frozen source and eligibility snapshots, exclusions and split assignments.
processed/    Immutable task-specific Parquet delivery products for formal data builds.
```

`cache/` is disposable and not a provenance input. `quarantine/` retains corrupt, superseded or ambiguous
material and is never selected by a delivery builder. Every transition writes a new versioned destination and
then publishes it atomically; no layer overwrites its input.

Sidecars intentionally retain more than a training row: native 133-joint RTMW3D output, transformed 2D output,
timestamps, model/runtime metadata, checksums and errors. Final rows retain only target-adapter tensors/text plus
identity and provenance needed to audit that row. This prevents every epoch from carrying audit-only arrays while
keeping every annotated sample reproducible.

## Delivery Layout And Placement

```text
processed/<product-id>/<schema-version>/<build-id>/
  delivery.json
  schema.json
  splits/
    train/chunk-00000/part-00000.parquet
    validation/chunk-00000/part-00000.parquet
    test/chunk-00000/part-00000.parquet
  inventories/parts.jsonl
  indices/sample_index.jsonl
  validation/report.json
  SHA256SUMS
```

`<build-id>` is derived from the frozen eligibility manifest hash, split hash, delivery-config fingerprint and
clean Git commit. The builder writes a hidden staging sibling, validates it fully, then atomically renames it;
an existing build may never be overwritten.

Split is a directory boundary. Each part and chunk contains exactly one split, so train globs cannot accidentally
include validation/test data. Archive ID is row provenance rather than a storage partition: archive directories
would weaken split isolation and the fixed part/chunk contract.

1. Select samples from exactly one frozen eligibility manifest and one frozen split assignment.
2. Within a split, order rows by UTF-8 byte order of `sample_id`; this is the only placement order.
3. A Parquet part has at most `1,024` rows and exactly one row group. Every non-terminal part has 1,024 rows;
   the terminal part may be shorter. Padding rows are prohibited.
4. A chunk has at most `64` parts, hence targets 65,536 rows. Every non-terminal chunk has 64 parts; the
   terminal chunk may be shorter.
5. Names are stable ordinals, `chunk-00000/part-00000.parquet`, never archive, worker or execution order IDs.
6. `parts.jsonl` records split/chunk/part, URI, row count, byte count, SHA-256, sample-ID digest, row-group
   count and schema fingerprint. `sample_index.jsonl` maps `sample_id` to split/chunk/part/row position.

The part/chunk boundary is a retry and streaming unit, not a scientific split.

## Shared Identity And Provenance

All final products include typed versions of the following columns. A non-established value is null, not an
inferred placeholder:

| Column | Type | Rule |
|---|---|---|
| `schema_version`, `build_id`, `dataset_id`, `sample_id` | `string` | Product/data/build identity; `sample_id` is globally unique. |
| `sequence_id`, `subject_id`, `source_archive_id`, `source_member` | nullable `string` | Exact source identity when known. |
| `group_id`, `split` | `string` | Frozen split identity; `split` equals enclosing directory. |
| `source_archive_sha256` | nullable `string` | Exact selected raw archive. |
| `source_member_crc32` | nullable `uint32` | Member integrity identity when available. |
| `caption`, `caption_language` | nullable `string` | Inline target text and language only where relevant. |
| `frame_count`, `fps` | nullable `int32`, `float32` | Sequence length and rate, when applicable. |
| `coordinate_frame`, `pose_units` | nullable `string` | Explicit; trainable metric products require `m`. |
| `annotation_config_fingerprint` | nullable `string` | Annotation/transform identity when applicable. |
| `source_manifest_sha256`, `split_assignment_sha256` | `string` | Immutable input bindings. |

Global details that do not distinguish rows, including full resolved config, Git state, writer version, source
registry copies, license and static schema, are recorded once in `delivery.json`. Neither rows nor metadata hold
machine absolute paths, staging paths or cache paths.

## Product-Specific Schemas

The current training adapters define final payloads. Numeric payloads are `float32`; masks are `bool`; a reader
may not silently coerce data. Variable temporal axes use typed Arrow lists and static axes use fixed-size lists.

### Pose Reconstruction

Protocol: `mmprism.pose_reconstruction.sample_v1`.

| Column | Arrow logical shape | Required contract |
|---|---|---|
| `radar_cube` | `list<fixed_size_list<fixed_size_list<fixed_size_list<fixed_size_list<float32,E>,A>,R>,D>>` | `[T,D,R,A,E]`, non-negative `mmprism.radar_cube.power_v1`. |
| `frame_mask` | `list<bool>` | `[T]`; source omission materializes as all true. |
| `pose_gt` | `fixed_size_list<fixed_size_list<fixed_size_list<float32,3>,24>,2>` | `[2,24,3]` metric target. |
| `pose_valid` | `fixed_size_list<fixed_size_list<bool,24>,2>` | `[2,24]`; source omission materializes as all true. |

Admission requires `pose_units == "m"`, explicit coordinate frame, agreed static dimensions, explicit radar
protocol and calibration/acquisition evidence that radar cube and pose share the coordinate contract.

### Sign Language Translation

Protocol: `mmprism.sign_language_translation.sample_v2`. `delivery.json` and `schema.json` bind one
`input_mode`; a delivery may not mix modes.

| Column | Arrow logical shape | Required contract |
|---|---|---|
| `pose` | `list<fixed_size_list<fixed_size_list<fixed_size_list<float32,3>,J>,2>>` | `[T,2,J,3]` metric pose. |
| `pose_confidence` | `list<fixed_size_list<fixed_size_list<float32,J>,2>>` | `[T,2,J]`, aligned to pose. |
| `radar_feature` | `list<fixed_size_list<float32,F>>`, fusion only | `[T,F]`, aligned to pose; required only for `pose_plus_radar_feature`. |
| `frame_mask` | `list<bool>` | `[T]`; source omission materializes as all true. |
| `caption` | `string` | Non-empty inline target text. |

All temporal modalities must have identical `T`; confidence is within `[0,1]`; the mask is non-empty; and pose
coordinate/radar-feature protocols are explicit. `pose_only` rows and Arrow schemas omit `radar_feature` and
`radar_feature_dim` entirely; their reader returns `radar_feature=None`. `pose_plus_radar_feature` retains the
non-null feature column and fixed feature dimension. Product schemas must remain separate even where both contain a
pose tensor.

### CSL-News Boundary

Current CSL-News output is named `intermediate_visual_pose_caption`. Its source-bound sidecar/NPZ stores native
133-joint pose, scores, transformed 2D keypoints, frame indices/timestamps, canonical `[T,2,24,3]` visual pose,
confidence/validity and caption. This is valid annotation evidence but not a final training product.

It has neither a calibrated radar cube and metric pose contract required by OmniHand nor a time-aligned radar
feature and metric coordinate contract required by WaveLLM. It must not be promoted to either product merely
because canonical pose shape matches a downstream tensor. Audit-heavy visual arrays remain in sidecars until a
new concrete and validated consumer requires them.

## Serialization And Build Gates

The canonical writer/reader uses PyArrow, a pinned compatible version, Zstandard compression and exactly one
row group per part (the final row group has the actual part row count, at most 1,024). It is added as a lazy
explicit `data-parquet` extra so basic contract tests do not import PyArrow or training dependencies.
`schema.json` records Arrow types, static dimensions, protocol versions, translation input mode where relevant,
and a schema fingerprint. Opaque objects, pickles and arbitrary `.npy` byte payloads are prohibited.

`delivery.json` records product/protocol/schema versions, translation input mode where relevant, row/part/chunk
policy, compression/writer versions,
clean Git state, a portable resolved delivery config and config fingerprint, portable runtime environment, build
time, deterministic-placement policy, frozen inputs, and validation outcome. It is the formal training input
binding; it excludes data roots, staging paths and other machine-local paths.

The materializer is a downstream consumer only; it must not annotate, simulate, assign splits or repair source
data while writing Parquet.

1. Validate delivery config, clean Git state and no-clobber destination.
2. Require exact frozen-manifest/split coverage, no duplicate IDs, and all target task modalities.
3. Run a capacity dry-run before writing, with payload/staging estimate and free-space floor recorded.
4. Stream deterministic payloads, validating shape, dtype, finite values, units, frames, masks and confidence.
5. Write staged parts/chunks; compute per-part checksums and inventories after finalization.
6. Validate exact membership, part/chunk/row-group bounds, schema fingerprint, checksum, index and no missing IDs.
7. Run first/middle/last plus seeded random row parity against source fixtures, then relevant adapter CPU batch
   smoke. Publish atomically only after all gates pass.

An interrupted or invalid build stays staging/quarantine evidence and cannot be used for training.

## Reader Migration

Existing JSONL-plus-NPY adapters remain forensic/build inputs during migration. Once a Parquet product is released,
its final training configs may not silently revert to those array paths.

1. `ParquetPoseReconstructionDataset` and `ParquetSignLanguageTranslationDataset` expose the current
   dependency-light sample/batch contracts with lazy PyArrow import.
2. `mmprism parquet-delivery-plan CONFIG` validates frozen inputs and returns placement/capacity estimates without
   writing. `mmprism parquet-delivery-build CONFIG` requires clean Git and atomically publishes a no-clobber
   delivery. `mmprism parquet-delivery-validate ROOT` rechecks copied input bindings, inventory, index, schema,
   rows and checksums.
3. Fixture parity covers both translation modes and pose reconstruction; placement, split isolation, no-clobber,
   input-mode/schema tampering, part tampering, inventory drift, unlisted parts, and the missing optional dependency
   are rejected. A real delivery still requires a dedicated
   capacity report and CPU/GPU one-batch adapter smoke before formal training.

## CSL-News Pipeline Boundary

The current service-level split is clear: source integrity, annotation, identity audit, QC, status and frozen
pose-manifest construction are separate modules with thin CLI/systemd orchestration. The annotation module itself
is too large because it combines strict config parsing, MMPose bootstrap, ZIP reader, transform, artifact
publication and worker loop.

It should remain in this repository. It shares contracts, manifests, configuration, tests, revision evidence and
the future Parquet materializer with the training system. A separate workspace would duplicate provenance and
create another cross-repository interface.

After live workers are deliberately stopped or migrated, split the implementation under `src/mmprism/data/csl_news/`:

```text
annotation_config.py     rtmw3d_estimator.py    archive_reader.py
annotation_artifacts.py  annotation_runner.py   pose_transform.py
annotation_audit.py      annotation_qc.py       annotation_status.py
pose_manifest.py         parquet_delivery.py
```

Until then `csl_news_annotation.py` remains the stable facade used by live workers. Refactoring its import path
while it is writing is forbidden. The first extraction preserves the facade and proves fixture/output parity
before a controlled worker migration.

## Acceptance Evidence

A production delivery is evidence-ready only with a frozen source/eligibility manifest, frozen split assignment,
complete delivery metadata/inventory/checksums, validator and capacity reports, reader parity, adapter smoke and
a formal run binding. Delivery completion alone is not paper evidence; manuscript claims additionally require an
entry in the paper evidence registry.
