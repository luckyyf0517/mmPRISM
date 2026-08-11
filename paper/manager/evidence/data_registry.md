# Data Registry

Status: `upload_contract_ready_awaiting_sources`
Last Updated: `2026-08-11`
Role: `dataset_and_split_provenance`

## Data Families

| ID | Family | Source Location | License/Access | Raw Size | Manifest | Validation | Paper Role | Status |
|---|---|---|---|---|---|---|---|---|
| `DATASET-CSL-DAILY` | CSL-Daily | official source or incoming upload: unknown | restricted/unknown | unknown | missing | missing | visual pose/synthetic training and SLU | blocked |
| `DATASET-CSL-NEWS` | CSL-News | HF `ZechengLi19/CSL-News@3a060121`; incoming batch active | CC BY-NC 4.0 | 935001573087 B compressed | metadata profile ready; source adapter audit in progress; full archive manifest pending | JSON 722,711/722,711 valid unique records; CSV contains 4 conflicting duplicate rows but covers every canonical JSON record; `archive_003`: SHA/CRC/1,657 labels/decode passed; full video validation pending | visual pose/synthetic training and SLU | in_progress |
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
| `BUILD-CSL-NEWS-RTMW3D-V1` | `DATASET-CSL-NEWS` | native 133-joint RTMW3D + historical crop/depth/2x24 mapping; config fingerprint `d7525ebb...` | `/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1` | GPU smoke passed; live status `healthy`; QC `20260811T145656Z` sampled 100/246 artifacts and 24,628 frames, 100 passed/0 warnings, checksum/shape/finite/frame/FPS contracts passed; reports under `reports/` and `qc/` | in_progress |

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

Pinned source and download implementation are recorded in `../../../docs/architecture/csl_news_data.md`.

First source-audit artifact target:

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1/
```

This trial is not promoted as dataset validation until its CRC, label coverage and video decode report passes.
