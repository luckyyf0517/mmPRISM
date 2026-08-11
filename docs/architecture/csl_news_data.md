# CSL-News Source and Rebuild Contract

Status: `official_download_active_legacy_pipeline_audited`
Last Updated: `2026-08-11`
Role: `dataset_specific_architecture_contract`

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

The official RGB source is being downloaded without extraction to:

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/
```

The download is resumable and managed by `scripts/download_csl_news.sh`. Archives remain immutable until
checksum, ZIP integrity, label coverage and license metadata are validated.

The downloader supports `curl` and `aria2` engines. The active intake uses four archive workers with eight
aria2 connections per archive after a controlled benchmark showed about 3.4 MiB/s for one aria2 transfer
while the previous sixteen single-connection curl transfers delivered about 4.7 MB/s in aggregate. Existing
curl `.part` files are contiguous prefixes and are resumed by aria2; the final atomic rename contract is
unchanged. After disabling aria2's per-fragment low-speed cutoff, a 60-second five-process measurement
reported 9.95 MB/s of effective writes with all processes healthy.

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

`src/scripts/split.py` scans pose paths, hashes the basename with MD5 and writes train/validation JSON files.
The JSON values are machine-specific absolute paths and the split is not signer-, source-video- or
near-duplicate-aware.

Legacy OmniHand uses `SingleFrameDataset` to load `[T, 2, 24, 3]` pose targets and derive per-frame velocity.
Legacy WaveLLM reads GT/predicted pose paths plus the converted caption dictionary, then pads or samples the
sequence to a fixed maximum length.

## Confirmed Interface Drift

The checked-in scripts do not form a valid end-to-end pipeline:

1. `run_csl_news_annotation.py` currently writes `[T, 2, 24, 3]`.
2. `run_simulation.py` indexes its input as `[T, N, 3]` and slices body/hands from a flat joint axis.
3. `src/scripts/check.py` documents `[N, 57, 3]` but actually checks for `[N, 59, 3]`; its deletion calls
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
