# CSL-News Source and Rebuild Contract

Status: historical
Owner: CSL-News annotation lane
Authority scope: Historical CSL-News source and annotation contract retained for evidence recovery only.
Last reviewed: 2026-08-12

> **Archived on 2026-08-12.** The local CSL-News ZIP/metadata intake, source registries, source manifests, and
> extracted-video cache have been removed. This page records the historical pipeline and must not be used to
> download, resume, schedule, annotate, or build a training delivery. Completed pose outputs, frozen pose
> manifests, and splits are retained only as checkpoint-side visual-pose evidence; see
> [archival cleanup](../../logs/2026/08/20260812_ARCHIVAL_CLEANUP.md).

## Official Source

- Dataset: `ZechengLi19/CSL-News`
- Source: `https://huggingface.co/datasets/ZechengLi19/CSL-News`
- Pinned revision: `3a0601210333fe760efd09b5d9e2ae5f341ce339`
- License: `CC BY-NC 4.0`
- RGB package: 436 ZIP archives, `archive_001.zip` through `archive_436.zip`
- Compressed repository size: `935001573087` bytes, about 935 GB decimal
- Labels: `data/train/CSL_News_Labels.json` and `.csv`
- Official pose alternative: `ZechengLi19/CSL-News_pose`, revision
  `73f339ebe49c75ba497320c3610b87f42fd497ec`, about 256 GB compressed

Historically, the official RGB source was downloaded without extraction to:

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/
```

The download is resumable and managed by `scripts/download_csl_news.sh`. Archives remain immutable until
checksum, ZIP integrity, label coverage and license metadata are validated.

The downloader supports `curl` and `aria2` engines. The active intake uses four archive workers with eight
aria2 connections per archive after a controlled benchmark showed about 3.4 MiB/s for one aria2 transfer
while the previous sixteen single-connection curl transfers delivered about 4.7 MB/s in aggregate. Existing
curl `.part` files are contiguous prefixes and are resumed by aria2. Final promotion now requires an explicit
successful transfer status, no remaining aria2 control file, and a complete `unzip -t`/CRC pass before atomic
rename. After disabling aria2's per-fragment low-speed cutoff, a 60-second five-process measurement
reported 9.95 MB/s of effective writes with all processes healthy.

This promotion gate was added after `archive_001` received HTTP 403 at 93% and the former `xargs` child shell
failed to propagate aria2's non-zero exit, causing an incomplete `.part` to be renamed. The primary
`archive_001`, plus member-corrupt primary `archive_005` and `archive_008`, remain immutable quarantine
candidates. Only archives in a full-read integrity report's passed list may enter annotation or manifest
promotion.

## Replacement Overlay And Source Identity

Source-integrity registry v2 resolves one exact source path for every archive. A registry entry contains the
relative path, `source_kind` (`primary` or `replacement`), SHA-256, stat, audit path/hash and label identity.
Consumers must open `archive_path_relative`; reconstructing a path from `archive_id` is forbidden.

The verified replacement overlay is:

```text
incoming/20260811_csl_news_hf_3a060121/rgb_archives/
  replacements/20260811_recovery_hf_3a060121/rgb_archives/archive_{001,005,008}.zip
```

The three replacements passed full CRC, label coverage and deterministic decode validation. Their primary
counterparts are not moved, deleted or overwritten. The v2 registry snapshot at `2026-08-11T22:10Z` has
SHA-256 `ae6b2909e7b12c3f9519ffc493b67a556621d6e7203665b940ea4bee9878a02c` and contains 59 passed
archives/97,997 videos with zero failed entries; it remains partial relative to 436 archives.

Annotation reuse is source-bound. Archive SHA-256, labels SHA-256, member size and CRC must match the current
entry. Existing unbound or different-source output is retained, while recomputation is published beside it as
`<sample-id>--source_<full-archive-sha256>.{npz,json}`. A pose manifest selects exactly one current-source
sidecar and writes every superseded/unbound candidate to a checksum-covered
`source_identity_quarantine.jsonl`.

## Legacy Preprocessing Flow

The following describes the code that exists in the historical implementation. It is forensic evidence, not
the canonical rebuild API.

### 1. Archive extraction and video selection

`run_csl_news_annotation.py` expects:

```text
<base>/archives/archive_NNN.zip
```

For each archive it:

1. extracts the ZIP into `<base>/videos/archive_NNN/`;
2. recursively discovers MP4 files;
3. retains only paths containing `Common-Concerns` or `Dragon-TV`;
4. processes every selected video;
5. deletes the extracted archive directory after processing.

The legacy base path is hard-coded as `/root/autodl-tmp/datasets/csl-news`.

### 2. RTMPose3D annotation

For every selected MP4, `run_csl_news_annotation.py`:

1. decodes the full video with OpenCV and holds all frames in memory;
2. crops 20 pixels from the top and both horizontal edges;
3. treats the entire cropped frame as one person bounding box;
4. runs `mmpose.apis.inference_topdown` using RTMPose3D;
5. subtracts one sequence-level depth center computed from joints 6 and 7;
6. retains 17 body joints and 42 hand joints temporarily;
7. constructs two 24-joint branches: three arm joints plus 21 hand joints per side;
8. writes a `float32` NPY with intended shape `[T, 2, 24, 3]` under the parallel `poses/` tree.

The referenced MMPose checkout, model config and checkpoint are not present in the current repository. The
historical checkpoint name is:

```text
rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
```

### 3. Text labels

`run_csl_lables.py` reads `CSL_News_Labels.json` and creates a dictionary:

```text
video filename without extension -> Chinese text
```

This is format conversion, not text annotation. Duplicate basenames would overwrite silently, so the rebuild
must use a dataset-qualified stable sequence ID and validate one-to-one label coverage.

### 4. Synthetic radar generation

`run_simulation.py` historically:

1. reads pose NPY files;
2. selects body and hand points and interpolates skeleton segments;
3. smooths trajectories with a Gaussian filter;
4. estimates velocity and downsamples nominal 30 FPS pose to 10 FPS;
5. simulates complex FMCW signals using the legacy 64-chirp configuration;
6. sums the raw signal into range-azimuth and Doppler-azimuth projections;
7. saves the stacked signal under a path produced by replacing `poses` with `signals`.

### 5. Split and model inputs

`legacy/src/scripts/split.py` scans pose paths, hashes the basename with MD5 and writes train/validation JSON files.
The JSON values are machine-specific absolute paths and the split is not signer-, source-video- or
near-duplicate-aware.

Legacy OmniHand uses `SingleFrameDataset` to load `[T, 2, 24, 3]` pose targets and derive per-frame velocity.
Legacy WaveLLM reads GT/predicted pose paths plus the converted caption dictionary, then pads or samples the
sequence to a fixed maximum length.

## Confirmed Interface Drift

The checked-in scripts do not form a valid end-to-end pipeline:

1. `run_csl_news_annotation.py` currently writes `[T, 2, 24, 3]`.
2. `run_simulation.py` indexes its input as `[T, N, 3]` and slices body/hands from a flat joint axis.
3. `legacy/src/scripts/check.py` documents `[N, 57, 3]` but actually checks for `[N, 59, 3]`; its deletion calls
   are currently commented out even though the summary labels rejected paths as "removed".
4. `run_extract_feature.py` still contains comments and indexing for `[T, 59, 3]` in parts of its path.
5. The RTMPose dependency, config and weight files are absent from the locked environment.
6. Path relationships are inferred through string replacement rather than a manifest.

Therefore, none of the legacy cleanup/check scripts may be run against the new source intake. Their contracts
must first be recovered and expressed as non-destructive canonical validators.

## Canonical Rebuild Stages

The new pipeline will preserve the scientific intent while defining explicit contracts:

1. `source inventory`: archive SHA-256, ZIP integrity, MP4 member list and label coverage;
2. `source manifest`: stable sample ID, original video name, category/source program, caption and archive ID;
3. `video decode`: streaming frames with original FPS/timestamps and no implicit crop;
4. `pose annotation`: pinned estimator/config/checkpoint, keypoints plus confidence, failure status and QC sample;
5. `pose transform`: versioned joint mapping, depth convention, coordinate system and units;
6. `simulation`: explicit pose schema input and versioned radar configuration output;
7. `split`: group-disjoint source/signer/video policy with duplicate audit;
8. `materialization`: versioned pose/signal/features with provenance and validation reports.

The legacy output will only be compared against this pipeline on a small audit subset; no compatibility shim is
required.

## Elastic Annotation Scheduling

Annotation execution is controlled separately from the data contract. The static historical worker sharding
rule (`archive_id % worker_count`) remains supported only for targeted recovery and forensic reproduction; it
is not the operational production mode because changing worker capacity would reshuffle ownership.

The current operation uses a filesystem control plane below the annotation output root:

```text
scheduler/control.json             # source/config-bound running or paused intent
scheduler/claim.lock               # short critical section for archive claims
scheduler/leases/archive_NNN.json  # active worker lease and heartbeat
scheduler/leases/history/          # released lease records
scheduler/leases/expired/          # stale leases retained before recovery
```

A scheduled worker claims one currently integrity-passed, source-incomplete archive at a time. A lease is
renewed before every video, so `paused` takes effect after the in-flight video completes. No completed sample
is revisited: existing source-bound NPZ/sidecar validation remains the completion authority, and archive markers
continue to bind the exact source identity. An interrupted worker leaves its lease for a bounded expiry period;
only then may another worker recover the archive. More workers can be started or stopped at any time without
repartitioning existing work. The control plane is operational metadata, not a source manifest or paper-facing
artifact.

## Processed Delivery Boundary

CSL-News annotation output is an `interim` product, not final training data. Each published source-bound NPZ and
JSON sidecar preserves native 133-joint RTMW3D evidence, transform details, frame/timestamp arrays, canonical
visual 2x24 pose, confidence, validity, source identity and annotation checksum. A frozen pose manifest selects
eligible sidecars without copying or reinterpreting them.

Final model-ready data is delivered separately as task-specific Parquet according to
[Parquet delivery contract](../../../../data_rebuild/docs/authority/20_CONTRACTS/DATA_DELIVERY_PARQUET.md):
one training sample per row, at most 1,024 rows per Parquet part and at most 64 parts
per split-homogeneous chunk. The builder may consume only a frozen eligible manifest and frozen split assignment;
it is not allowed to read a live annotation directory or repair inputs during materialization.

Current CSL-News output is specifically `intermediate_visual_pose_caption`. It is not yet admissible as either
current training product: OmniHand additionally requires a calibrated non-negative radar cube and metric target
pose; WaveLLM additionally requires a time-aligned radar feature and metric pose-coordinate contract. The visual
RTMW3D arrays, including canonical `[T,2,24,3]`, must not be described as metre-space radar ground truth merely
because their shape matches a downstream hand-pose tensor. Audit fields remain in sidecars until a concrete,
validated consumer requires a separate product.

## Source Audit Smoke

`mmprism csl-news-audit` implements the first read-only gate for one complete archive. It rejects `.part`
inputs and records:

- archive and labels SHA-256;
- ZIP member counts, unsafe/encrypted/duplicate paths and optional full CRC validation;
- source-program counts and official label coverage;
- deterministic sampled MP4 full-decode results when `--decode-samples` is enabled.

The command writes an atomic `mmprism.csl_news_source_audit.v1` JSON report. Sample videos are copied only
to a temporary directory for decoding and removed before the command returns. This gate does not run pose
annotation, simulation or model training.

The first real-source trial is scheduled for `2026-08-12 08:00 Asia/Shanghai` and writes under:

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1/
```

## Canonical Source Manifest Snapshot

`mmprism csl-news-source-manifest` v2 freezes one exact source-integrity v2 registry byte snapshot. It reads
only typed `passed` entries, resolves each entry's `archive_path_relative` under the configured archive root,
and never infers an archive path from its ID. It never reads `.part` files and does not extract videos. The
versioned configuration keeps archive, registry, label, and output relationships relative to
`MMPRISM_DATA_ROOT`.

Each `mmprism.sample.v1` record contains:

- a stable sample ID derived from source ID, archive name, and member name;
- video as `zip://<registry-relative-archive-path>!/member.mp4`, resolved only with the configured archive root;
- the canonical JSON caption as an explicit inline-text modality with a UTF-8 SHA-256;
- archive path/source kind/SHA-256/stat/audit, exact registry SHA-256, ZIP member CRC/size, labels SHA-256,
  config fingerprint, and clean Git commit;
- archive and source-program group keys, while unavailable subject/scene fields remain explicit unknowns.

The builder requires registry schema v2 and rejects archive-root/label/count mismatches, path escapes,
symlinks, stat/SHA/video-count drift, unsafe/encrypted/duplicate members, cross-archive basename collisions,
missing labels, stable-ID collisions, source changes during the scan, insufficient disk space, and dirty Git
state. It copies the exact registry bytes, validates the JSONL through the general manifest contract, writes
`SHA256SUMS` for registry/manifest/summary, and only then atomically renames the snapshot directory.
Available-archive snapshots are intentionally `partial`; a final snapshot is complete only when all 436
archives and every canonical label are represented.

The old v1 source snapshot scanned primary ZIP names directly and remains historical linkage evidence only.
The v2 builder shares the same registry path semantics as annotation and pose-manifest builders, including the
versioned replacement paths for `001/005/008`.

The first real v2 snapshot was built from clean commit `7f86516` at
`source_manifest_v2/snapshot_20260811T224413.526848Z`. It freezes 63 archives and 104,658 records, copies
registry SHA-256 `dc2d7068...`, and writes manifest SHA-256 `a431d14c...`. Checksum replay, the general
manifest contract, portable paths, and first/middle/last exact ZIP/member reads passed. Its 4,945 replacement
records resolve `001/005/008` through their registered versioned paths rather than the preserved corrupt
primaries. `crc_checked=false` in this snapshot means the manifest builder did not repeat the expensive full
CRC pass; the copied typed registry already binds full CRC, label-coverage and decode evidence for every
selected archive. The snapshot remains partial until all 436 archives and 722,711 labels are represented.

The canonical label source is JSON. CSV is retained as immutable cross-check evidence because it contains
the same 722,711 unique keys but four additional conflicting duplicate rows; it may not override JSON.

## Canonical Pose And Caption Manifest

`mmprism csl-news-pose-manifest` freezes completed RTMW3D sidecars visible at scan start. It only accepts
artifacts whose archive has a typed `passed` entry in one exact cumulative integrity-registry byte snapshot.
It then validates stable identity, canonical JSON caption, annotation fingerprint, NPZ shape/dtype contract,
artifact size and optional SHA-256 before atomically publishing the snapshot.

`mmprism csl-news-annotation-audit` is the CPU-only full identity gate for a live annotation root. It freezes
the visible non-hidden sidecar list at start, hashes each paired NPZ as a stream, compares declared size and
SHA-256, and requires sidecar/artifact stat stability before and after reads. It accumulates every invalid pair
without loading NPZ arrays. Reports bind the frozen list hash, runtime Git state and aggregate bytes hashed.

An identity mismatch is never repaired in place and does not weaken the normal manifest checksum gate.
`validation.exclusions` can quarantine only an exact known pair: archive/sample ID, sidecar SHA-256,
declared and observed NPZ identities, clean-run audit path/SHA-256 and reason must all match. Every configured
entry must be present and applied exactly once; unused or drifted entries fail the snapshot. Accepted audit
evidence is copied into the immutable snapshot and included in `SHA256SUMS`.

Each output is a portable `mmprism.sample.v1` JSONL manifest. Native 133-joint arrays, transformed 2D
keypoints, frame/timestamp arrays and canonical `[T,2,24,3]` pose/confidence/valid arrays reference one NPZ
through relative URIs. The caption is inline. Source archive/audit, labels, annotation sidecar/model/transform,
builder config and clean Git commit remain explicit provenance. A dependency-light `CslNewsPoseManifest`
adapter provides random access and revalidates the shared container, shape and dtype contract without importing
PyTorch, Lightning or Transformers.

The first real partial snapshot was created from clean commit `390093b`:

```text
/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/pose_manifest_v1/
  snapshot_20260811T164204Z_partial/
```

It contains 2,157 records from 5 integrity-passed archives and references 1,169,173,125 bytes of pose
artifacts. Fifteen historical artifact/sidecar pairs from failed archives were retained but excluded. Its
manifest SHA-256 is `4161593fdbfc85a5c2fb392e3ef92d40da560db5c75a19d559f1f92878e31600`.
This snapshot is contract and pipeline evidence only; the final dataset requires all 436 archives to pass the
source gate and a new complete snapshot.

The first live publication conflict was discovered while building a later snapshot: one GalaxyFS-visible
sidecar declared the empty-file identity although its atomically promoted NPZ was complete. The original pair,
failure records and failed snapshot temp directory remain unchanged. Clean commit `3bdd31f` audited 9,519
published pairs and found only this mismatch. Clean commit `98549a9` then produced
`snapshot_20260811T212450.135852Z` with 9,551 included records, one checksum-bound exclusion, zero unpaired
eligible NPZ and manifest SHA-256
`8e3db8712bc61848e9d6dea9f5b3a3821365ffd102d6643977ad43107b2db0c4`. This remains a partial snapshot.

The first v2 source-bound snapshot was built from clean commit `11014a8` at
`snapshot_20260811T222941.214512Z`. It binds registry SHA-256 `ae6b2909...`, contains 10,011 records from
12 represented archives, and writes 1,875 superseded/unbound sidecars to a checksum-covered source-identity
quarantine ledger. The manifest SHA-256 is
`3412aeb2f7fea685796e17d85b3af6342b7ffe1b3a61895446295f5f71e073f7`. All snapshot checksums,
the general manifest contract and first/middle/last checksum-validating reads passed. It remains partial.
