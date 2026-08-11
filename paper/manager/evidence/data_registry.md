# Data Registry

Status: `upload_contract_ready_awaiting_sources`
Last Updated: `2026-08-11`
Role: `dataset_and_split_provenance`

## Data Families

| ID | Family | Source Location | License/Access | Raw Size | Manifest | Validation | Paper Role | Status |
|---|---|---|---|---|---|---|---|---|
| `DATASET-CSL-DAILY` | CSL-Daily | official source or incoming upload: unknown | restricted/unknown | unknown | missing | missing | visual pose/synthetic training and SLU | blocked |
| `DATASET-CSL-NEWS` | CSL-News | HF `ZechengLi19/CSL-News@3a060121`; incoming batch active | CC BY-NC 4.0 | 935001573087 B compressed | cumulative registry active；48 archives/79,813 videos passed at 21:24Z；9,551-record pose+caption partial manifest verified；final 436-archive manifest pending | JSON 722,711/722,711 valid unique records；`001/005/008` source failures isolated；9,519-pair identity audit found one checksum-bound exclusion | visual pose/synthetic training and SLU | in_progress_integrity_failure |
| `DATASET-COLLECTED-BASE` | collected_base | unknown | private | unknown | missing | missing | real radar pose | blocked |
| `DATASET-COLLECTED-DEMO` | collected_demo | unknown | private | unknown | missing | missing | development/demo | blocked |
| `DATASET-COLLECTED-CSL` | collected_csl | unknown | private | unknown | missing | missing | real sign language | blocked |
| `DATASET-REAL-STRESS-REV1` | revision orientation/occlusion/new-user set | not collected | private/ethics pending | unknown | missing | missing | reviewer real-world boundary tests | blocked |

## Supporting Source Assets

| ID | Asset | Required Contents | Access | Rebuild Rule | Status |
|---|---|---|---|---|---|
| `ASSET-COLLECTED-METADATA` | participant/session/sequence metadata | anonymous subject mapping, caption, scene, distance, orientation, occlusion, split, ethics scope | private | cannot infer from directory names | blocked |
| `ASSET-RADAR-CONFIG` | hardware and acquisition configs | profile/chirp/frame, firmware/software, 12Tx/16Rx channel order, config version mapping | private/project | must bind every sequence to config ID | blocked |
| `ASSET-RADAR-CALIBRATION` | calibration and coordinates | antenna/virtual-array coordinates, phase/range calibration, bad channels, radar-camera transform/sync | private/project | cannot regenerate from radar arrays alone | blocked |
| `ASSET-SIM-PROVENANCE` | original simulation inputs and method | MANO parameters/mesh/model or skeleton inputs, source/config/checkpoint, example output | mixed/license-bound | resolve manuscript-vs-code discrepancy before reproduction claim | blocked |
| `ASSET-HISTORICAL-RUNS` | paper experiment evidence | splits, configs, checkpoints, predictions, metrics, logs | private | optional for new training, required for original-paper audit | blocked |
| `ASSET-FIGURE-SOURCE` | display-item source data | sample-level values, plots, scripts, tables | private | needed for Source Data unless public raw+code fully reproduce figures | blocked |

## Downloadable Model Assets

| ID | Model | Source ID | Revision/Checksum | License | Rule | Status |
|---|---|---|---|---|---|---|
| `MODEL-MT5-BASE` | mT5 base | `google/mt5-base` | HF `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f`; 6 files/2,334,046,221 B；weight SHA-256 `180573b5...`；asset manifest `4edec505...` | upstream license/release redistribution review pending | pinned downloader + A100 two-step train/generate smoke；historical fine-tuned weights remain separate | evidence_ready |
| `MODEL-RTMW3D` | RTMPose3D | official OpenMMLab RTMW3D-L; MMPose `759b39c` | `794dbc78b04a43d81781f8ab0eba5b24f3dd5d71aaf6ae253940424159fb81ed` | upstream research code/model terms; release audit pending | checkpoint/config/commit hash gate before every run | evidence_ready |
| `MODEL-SIMCSE` | SimCSE evaluator | `cyclone/simcse-chinese-roberta-wwm-ext` | HF `871d7039a3fccd4869d545a25b63c545341ca7f4`; 6 files/409,532,074 B；asset manifest `e57f2eeb...` | HF card does not declare license；do not redistribute until clarified | pinned downloader + `AutoModel` CPU smoke `[2,768]` finite；`EVID-CODE-MODELS-V1` | evidence_ready |
| `MODEL-SBERT` | SBERT evaluator | `shibing624/text2vec-base-chinese` | HF `183bb99aa7af74355fb58d16edf8c13ae7c5433e`; 8 files/409,209,289 B；asset manifest `81395c0b...` | Apache-2.0 in HF card | pinned downloader + `SentenceTransformer` CPU smoke `[2,768]` finite；`EVID-CODE-MODELS-V1` | evidence_ready |

## Derived Dataset Builds

| ID | Source | Protocol | Location | Validation | Status |
|---|---|---|---|---|---|
| `BUILD-CSL-NEWS-RTMW3D-V1` | `DATASET-CSL-NEWS` | native 133-joint RTMW3D + historical crop/depth/2x24 mapping; config fingerprint `d7525ebb...` | `/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1` | GPU smoke/QC passed；4 registry workers active；clean identity audit 9,518/9,519 passed；second frozen manifest 9,551 records/9 archives, SHA-256 `8e3db871...`；15 failed-source pairs and one exact identity conflict retained/excluded | in_progress |

## Model-ready Contracts

| ID | Modalities | Protocol | Validation | Real Build Status |
|---|---|---|---|---|
| `CONTRACT-POSE-RECONSTRUCTION-V1` | required `radar_cube [T,D,R,A,E] float32`, `pose_gt [2,24,3] float32`; optional `frame_mask [T] bool`, `pose_valid [2,24] bool` | sample `mmprism.pose_reconstruction.sample_v1`; cube `mmprism.radar_cube.power_v1`; pose unit `m` and explicit coordinate frame | strict relative `.npy` URI、shape/dtype/SHA-256、finite/non-negative power、manifest-wide spatial/coordinate consistency、variable-time zero-padding/masking tests passed | adapter_ready；real radar cube/metric pose manifest blocked on collected source and calibration provenance |

## Split Registry

| ID | Dataset | Group Rule | Seed/Hash Rule | Train/Val/Test Count | Manifest Hash | Leakage Audit | Status |
|---|---|---|---|---|---|---|---|
| `SPLIT-CSL-NEWS-POSE-PARTIAL-V1` | `BUILD-CSL-NEWS-RTMW3D-V1` partial 2,157 records | `sequence_id` -> SHA-256 `group_id`; signer unavailable | seed `20260811`; `sha256_mod_weight_v1`; weights `8/1/1` | 1,701 / 219 / 237 | source `4161593f...`; assignments `133f32d5...` | coverage 2,157/2,157；duplicate/leakage 0；independent bucket recompute passed | partial_evidence_ready |
| `SPLIT-LEGACY-UNKNOWN` | unknown | unknown | unknown | unknown | unknown | unchecked | blocked |

## Validation Requirements

- file readability and checksum sampling
- modality coverage
- shape/dtype distribution
- pose NaN and coordinate statistics
- radar chirp/antenna/sample consistency
- annotation coverage and empty captions
- subject/signer/sequence/scene leakage
- duplicate and near-duplicate sequence check
- archive/source checksum and upload completeness
- subject/session/sequence/radar-config referential integrity
- orientation/distance/occlusion metadata coverage
- radar-camera timestamp and coordinate-system consistency
- license, ethics, reviewer access and public release classification

## Intake Location

Pending batches must use:

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/<YYYYMMDD_source_batch>/
```

Each batch requires `README.md`, `UPLOAD_MANIFEST.csv` and `SHA256SUMS`. No batch is promoted to
`raw/` or `external/` until validation is recorded here. Full upload contents and ordering are defined in
`../data_upload_checklist.md`.

Active public-source batch:

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/
```

Metadata at the pinned revision completed on `2026-08-11`: JSON SHA-256
`3381d80157fa75012ec2a220eb8a63c88968af2d60d5dbcb5a82bf680db8a3a5`, CSV SHA-256
`683e2c71bc48d9cb6210118799836c7afa4a11269a41bab1dfa4fbbb1d0cee79`, and README SHA-256
`cc0c6367538d1eedb07f199e1a4d56edf74a2026b0718feae112400911b5ba23`.

Machine-readable profile `profile_20260811T151215Z.json` has SHA-256
`90e24aa4236febca9aa5bc8faaa025751618210f05d6bfd32d76ab9d94f10c43` and status
`passed_with_warnings`. The canonical JSON has 722,711 valid unique records; CSV has the same key
set plus four conflicting duplicate rows. JSON is authoritative and CSV cannot override it. Reviewer-facing
interpretation and missing fields are recorded in `csl_news_metadata_profile.md`.

The first clean-commit source snapshot is recorded in `csl_news_source_manifest.md`. Its 18,095-record
manifest has SHA-256 `6984d0cc30a0f5a9e6baa58fa8a764e0c0b70ed1b0bb9224e9fca8faa1b1a1f5`;
it covers 11/436 archives and remains explicitly partial. A subsequent full CRC audit found `005/008`
corrupt, so this snapshot is retained only as contract/linkage evidence. The integrity summary is recorded in
`csl_news_source_integrity.md` with SHA-256 `ea8062f546cdf10abdde5b5b27e0e78e5e39e3df538e0d68b983e6ac4b7c9a00`.
The same evidence document records the later `archive_001` incomplete-final incident and report SHA-256
`379b72f34a1f749a246891901e746defae3331dcb65fab64662686a7f260a723`, plus the first fixed-gate
promotion `archive_002` with passed report SHA-256
`3f2eaffd97c1f48481d92f7f88f5bd8ce68d78cce3bc74f0acbb9d8e0c43c4e9`.
The cumulative registry supersedes manual scheduling lists. The exact `2026-08-11T16:42Z` byte snapshot
bound to the first pose manifest has SHA-256
`183743fbb60bb85b75dd63f6c112e0c1a3081b2b6a391e32fa6ce2a21cb5b02d`, with 14 passed archives/
23,020 videos and failed entries `001/005/008` retained in place. The derived 2,157-record manifest is
documented in `csl_news_pose_manifest.md`; it is partial pipeline evidence, not the final dataset manifest.
At `21:24Z`, the cumulative registry reached 51 final archives: 48 passed/79,813 videos and the same three
failed IDs. A clean-commit CPU-only audit froze 9,519 published sidecar/NPZ pairs, streamed 5,115,703,846
artifact bytes, and reported exactly one mismatch. Report SHA-256 is
`55478cbb6078d7e4c7b0c9a95577e6260e249239514ec584d082d5b0b4c538b4`; the affected
`archive_006/3af7db9841fb2ac483721620` pair remains unchanged. Snapshot
`snapshot_20260811T212450.135852Z` binds that clean report and excludes only the exact declared/observed
identity conflict. It contains 9,551 records from 9 archives, manifest SHA-256
`8e3db8712bc61848e9d6dea9f5b3a3821365ffd102d6643977ad43107b2db0c4`, and passed all four
`SHA256SUMS` entries plus first/middle/last checksum-validating adapter reads. It remains partial evidence.
The first canonical sequence split is recorded in `csl_news_pose_split.md`. It binds that exact partial pose
manifest and has assignment SHA-256 `133f32d58b213947edf09c7c1e1b7c3ee30b8588a9f2b7a863d6a668bce2d7d9`.
It has zero sequence-group leakage but is not subject-independent because signer metadata is unavailable.

Pinned source and download implementation are recorded in `../../../docs/architecture/csl_news_data.md`.

First source-audit artifact target:

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1/
```

This trial is not promoted as dataset validation until its CRC, label coverage and video decode report passes.
