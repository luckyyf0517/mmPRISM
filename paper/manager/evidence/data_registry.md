# Data Registry

Status: `upload_contract_ready_awaiting_sources`
Last Updated: `2026-08-11`
Role: `dataset_and_split_provenance`

## Data Families

| ID | Family | Source Location | License/Access | Raw Size | Manifest | Validation | Paper Role | Status |
|---|---|---|---|---|---|---|---|---|
| `DATASET-CSL-DAILY` | CSL-Daily | official source or incoming upload: unknown | restricted/unknown | unknown | missing | missing | visual pose/synthetic training and SLU | blocked |
| `DATASET-CSL-NEWS` | CSL-News | HF `ZechengLi19/CSL-News@3a060121`; incoming batch active | CC BY-NC 4.0 | 935001573087 B compressed | cumulative integrity registry active；14 archives/23,020 videos passed at 16:42Z；2,157-record pose+caption partial manifest verified；final 436-archive manifest pending | JSON 722,711/722,711 valid unique records；`001/005/008` failed and isolated；promotion + consumption + derived-manifest gates active | visual pose/synthetic training and SLU | in_progress_integrity_failure |
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
| `MODEL-MT5-BASE` | mT5 base | `google/mt5-base` | unknown | unknown | pin and download; upload historical fine-tuned weights separately | not_started |
| `MODEL-RTMW3D` | RTMPose3D | official OpenMMLab RTMW3D-L; MMPose `759b39c` | `794dbc78b04a43d81781f8ab0eba5b24f3dd5d71aaf6ae253940424159fb81ed` | upstream research code/model terms; release audit pending | checkpoint/config/commit hash gate before every run | evidence_ready |
| `MODEL-SIMCSE` | SimCSE evaluator | `cyclone/simcse-chinese-roberta-wwm-ext` | unknown | unknown | pin HF revision and download | not_started |
| `MODEL-SBERT` | SBERT evaluator | `shibing624/text2vec-base-chinese` | unknown | unknown | pin HF revision, download and run evaluator smoke | not_started |

## Derived Dataset Builds

| ID | Source | Protocol | Location | Validation | Status |
|---|---|---|---|---|---|
| `BUILD-CSL-NEWS-RTMW3D-V1` | `DATASET-CSL-NEWS` | native 133-joint RTMW3D + historical crop/depth/2x24 mapping; config fingerprint `d7525ebb...` | `/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1` | GPU smoke/QC passed；4 registry workers active；first frozen manifest has 2,157 records/5 archives and SHA-256 `4161593f...`；15 ineligible historical pairs retained and excluded | in_progress |

## Split Registry

| ID | Dataset | Group Rule | Seed/Hash Rule | Train/Val/Test Count | Manifest Hash | Leakage Audit | Status |
|---|---|---|---|---|---|---|---|
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

Pinned source and download implementation are recorded in `../../../docs/architecture/csl_news_data.md`.

First source-audit artifact target:

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1/
```

This trial is not promoted as dataset validation until its CRC, label coverage and video decode report passes.
