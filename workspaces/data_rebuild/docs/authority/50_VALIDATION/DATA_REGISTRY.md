# Data Registry

Status: current
Owner: Data rebuild lane
Authority scope: The data intake, radar rebuild, split, or delivery boundary represented by this page.
Last reviewed: 2026-08-12

## Data Families

| ID | Family | Source Location | License/Access | Raw Size | Manifest | Validation | Paper Role | Status |
|---|---|---|---|---|---|---|---|---|
| `DATASET-CSL-DAILY` | CSL-Daily | official source or incoming upload: unknown | restricted/unknown | unknown | missing | missing | visual pose/synthetic training and SLU | blocked |
| `DATASET-CSL-NEWS` | CSL-News | HF `ZechengLi19/CSL-News@3a060121`; immutable primary + replacement overlay | CC BY-NC 4.0 | 935001573087 B compressed | v2 partial source snapshot 63 archives/104,658 records, SHA-256 `a431d14c...`；live registry 78 archives/129,539 videos at 00:45Z；final 436-archive snapshot pending | JSON 722,711/722,711 valid unique records；replacement `001/005/008` passed full CRC/coverage/decode；primary failures retained | visual pose/synthetic training and SLU | in_progress_download |
| `DATASET-COLLECTED-BASE` | collected_base | unknown | private | unknown | missing | missing | legacy non-semantic radar gesture/pose evidence only | blocked_legacy_inventory |
| `DATASET-COLLECTED-DEMO` | collected_demo | unknown | private | unknown | missing | missing | legacy non-semantic development/demo only | blocked_legacy_inventory |
| `DATASET-COLLECTED-CSL` | historical directory label only; contents are non-semantic gestures | unknown | private | unknown | missing | missing | legacy pose/forensic evidence; prohibited from semantic SLU totals or translation claims | blocked_legacy_inventory |
| `DATASET-SEMANTIC-SIGN-V2` | new video-guided Chinese Sign Language (CSL) collection | not collected; owned by `sign_language_collection` | private; minimal consent pending | unknown | missing | reference set/setup/pilot pending | prompted CSL reproduction, cross-participant and real-world testing; approximately 30 total participants planned, with 3--4 professional/proficient contributors if available and remaining video-guided volunteers | planning |
| `DATASET-REAL-STRESS-REV1` | compact orientation/occlusion subset of `DATASET-SEMANTIC-SIGN-V2` | not collected | private/ethics pending | unknown | missing | condition matrix and pilot pending | reviewer real-world boundary tests; not an independent participant cohort | planning |

## Processed Delivery Profiles

All previously self-collected project families are currently classified as non-semantic gestures. Their historical
directory names do not establish semantic labels, and none may enter `DELIVERY-SLU-V1` or paper translation totals.
New semantic real-data delivery begins only from a frozen, validated `DATASET-SEMANTIC-SIGN-V2` handoff.

| ID | Product | Frozen Inputs Required | Payload Boundary | Status |
|---|---|---|---|---|
| `DELIVERY-POSE-RECON-V1` | `mmprism.pose_reconstruction.sample_v1` | eligible manifest, split, radar/calibration evidence | radar cube `[T,D,R,A,E]`, frame mask, metric `[2,24,3]` pose and validity | reader/materializer/validator fixture-verified; blocked on real calibrated source |
| `DELIVERY-SLU-V1` | `mmprism.sign_language_translation.sample_v1` | eligible manifest, split, metric pose and aligned radar feature | pose/confidence `[T,2,J,*]`, radar feature `[T,F]`, mask and caption | reader/materializer/validator fixture-verified; blocked on real aligned source |
| `INTERIM-CSLNEWS-VISPOSE-V1` | `intermediate_visual_pose_caption` | source-bound CSL-News pose manifest | native/2D/canonical visual pose arrays and caption remain sidecar/NPZ | in_progress; explicitly not final training delivery |

All final delivery profiles use immutable, task-specific Parquet rather than a mixed universal table. The row/part/
chunk policy, typed payloads, provenance and validator gates are defined in
[Parquet delivery contract](../20_CONTRACTS/DATA_DELIVERY_PARQUET.md) and `DEC-038`; a profile is not paper evidence until its
frozen delivery inventory, reader parity and formal run are registered.

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
| `BUILD-CSL-NEWS-RTMW3D-V1` | `DATASET-CSL-NEWS` | native 133-joint RTMW3D + historical crop/depth/2x24 mapping; config fingerprint `d7525ebb...` | `/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1` | GPU smoke/QC passed；immutable conflict recovery preserved；clean `6e9cc5e` snapshot 12,057 records/12 archives, SHA-256 `cdd450e4...`；01:32Z expanded to 8 registry lanes: 0--3 on GPU 7 and 4--7 on GPU 5, `archive_id % 8`; baseline 16,162 current-source pairs/81 passed archives, first 3 min +157 pairs/0 new failures, all lanes active | in_progress |
| `EVID-CSL-NEWS-LEGACY-POSE-002-V1` | historical NAS-derived `archive_002` pose export | immutable ZIP `3b3af27c...`, 1,624 finite float64 `[T,59,3]` arrays; read-only comparison against current source SHA `a1086401...` | upload `/home/yanyifan/upload/20260812/archive_002.zip`; report `interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/legacy_comparisons/20260812_archive_002_v2` | ZIP test, exact source member/frame/depth-center coverage 1,624/1,624; 1,567 current-source-bound, numerical non-equivalence documented; 57 identity-unbound excluded from strict result | forensic_only_environment_equivalence_open |

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
[data intake operation](../40_OPERATIONS/DATA_INTAKE.md).

Active public-source batch:

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/
```

Pending historical comparison batch:

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260812_csl_news_legacy_pose_pair_v1/legacy_evidence/
```

该目录保留为规范化两样本 intake 目标。作者随后在 `/home/yanyifan/upload/20260812/archive_002.zip`
提供了完整 archive-level historical pose export；原始上传位置保持不变，未复制至 intake target。该 ZIP 已按
`annotation.legacy_pose_name` 关联并完成只读 full-archive 对照，细节见
[historical pose comparison](../../../../csl_news_annotation/docs/logs/2026/08/20260812_LEGACY_POSE_COMPARISON.md)。
它是 forensic evidence，不是 registered source，也不得进入任何
canonical manifest；`DATA-005-D` 仍需补齐 historical environment 与第二个 clean archive 交叉验证。

Metadata at the pinned revision completed on `2026-08-11`: JSON SHA-256
`3381d80157fa75012ec2a220eb8a63c88968af2d60d5dbcb5a82bf680db8a3a5`, CSV SHA-256
`683e2c71bc48d9cb6210118799836c7afa4a11269a41bab1dfa4fbbb1d0cee79`, and README SHA-256
`cc0c6367538d1eedb07f199e1a4d56edf74a2026b0718feae112400911b5ba23`.

Machine-readable profile `profile_20260811T151215Z.json` has SHA-256
`90e24aa4236febca9aa5bc8faaa025751618210f05d6bfd32d76ab9d94f10c43` and status
`passed_with_warnings`. The canonical JSON has 722,711 valid unique records; CSV has the same key
set plus four conflicting duplicate rows. JSON is authoritative and CSV cannot override it. Reviewer-facing
interpretation and missing fields are recorded in the
[metadata profile](../../../../csl_news_annotation/docs/logs/2026/08/20260811_METADATA_PROFILE.md).

The first clean-commit source snapshot is recorded in the
[source manifest Log](../../../../csl_news_annotation/docs/logs/2026/08/20260811_SOURCE_MANIFEST.md).
Its 18,095-record
manifest has SHA-256 `6984d0cc30a0f5a9e6baa58fa8a764e0c0b70ed1b0bb9224e9fca8faa1b1a1f5`;
it covers 11/436 archives and remains explicitly partial. A subsequent full CRC audit found `005/008`
corrupt, so this snapshot is retained only as contract/linkage evidence. The integrity summary is recorded in
[source integrity Log](../../../../csl_news_annotation/docs/logs/2026/08/20260812_SOURCE_INTEGRITY.md)
with SHA-256 `ea8062f546cdf10abdde5b5b27e0e78e5e39e3df538e0d68b983e6ac4b7c9a00`.
The same evidence document records the later `archive_001` incomplete-final incident and report SHA-256
`379b72f34a1f749a246891901e746defae3331dcb65fab64662686a7f260a723`, plus the first fixed-gate
promotion `archive_002` with passed report SHA-256
`3f2eaffd97c1f48481d92f7f88f5bd8ce68d78cce3bc74f0acbb9d8e0c43c4e9`.
The cumulative registry supersedes manual scheduling lists. The exact `2026-08-11T16:42Z` byte snapshot
bound to the first pose manifest has SHA-256
`183743fbb60bb85b75dd63f6c112e0c1a3081b2b6a391e32fa6ce2a21cb5b02d`, with 14 passed archives/
23,020 videos and failed entries `001/005/008` retained in place. The derived 2,157-record manifest is
documented in the
[pose manifest Log](../../../../csl_news_annotation/docs/logs/2026/08/20260811_POSE_MANIFEST.md);
it is partial pipeline evidence, not the final dataset manifest.
At `21:24Z`, the cumulative registry reached 51 final archives: 48 passed/79,813 videos and the same three
failed IDs. A clean-commit CPU-only audit froze 9,519 published sidecar/NPZ pairs, streamed 5,115,703,846
artifact bytes, and reported exactly one mismatch. Report SHA-256 is
`55478cbb6078d7e4c7b0c9a95577e6260e249239514ec584d082d5b0b4c538b4`; the affected
`archive_006/3af7db9841fb2ac483721620` pair remains unchanged. Snapshot
`snapshot_20260811T212450.135852Z` binds that clean report and excludes only the exact declared/observed
identity conflict. It contains 9,551 records from 9 archives, manifest SHA-256
`8e3db8712bc61848e9d6dea9f5b3a3821365ffd102d6643977ad43107b2db0c4`, and passed all four
`SHA256SUMS` entries plus first/middle/last checksum-validating adapter reads. It remains partial evidence.
The first canonical sequence split is recorded in the
[pose split Log](../../../../csl_news_annotation/docs/logs/2026/08/20260811_POSE_SPLIT.md).
It binds that exact partial pose
manifest and has assignment SHA-256 `133f32d58b213947edf09c7c1e1b7c3ee30b8588a9f2b7a863d6a668bce2d7d9`.
It has zero sequence-group leakage but is not subject-independent because signer metadata is unavailable.

The current source control plane is
`manifests/csl_news/source_integrity_v2/registry.json`. Snapshot SHA-256
`ae6b2909e7b12c3f9519ffc493b67a556621d6e7203665b940ea4bee9878a02c` binds 59 passed archives/
97,997 videos and exact replacement paths for `001/005/008`; failed count is zero. The prior v1 registry and
pose snapshots remain immutable incident/pipeline evidence but are not the current training source because
their source identities predate replacement selection. The live v2 status at `22:13Z` counted 9,394
current-source pairs, 1,875 quarantined old/unbound pairs, zero duplicate-current-source samples and 3/3
validated samples. The subsequent clean audit and v2-bound pose snapshot are tracked below.

The first v2-bound snapshot is
`pose_manifest_v1/snapshot_20260811T222941.214512Z`. It binds the registry hash above and contains 10,011
current-source records from 12 represented archives. Manifest SHA-256 is
`3412aeb2f7fea685796e17d85b3af6342b7ffe1b3a61895446295f5f71e073f7`; its 1,875-entry
source-identity quarantine ledger has SHA-256
`1b03721b4fc64601d8dff0fc247e6d7a1a319ac93d2dc25c6cc463f0cd659586`. All five checksum entries,
the general manifest contract, and first/middle/last checksum-validating adapter reads passed. It remains
partial and is not a paper-facing dataset-size claim.

The source-manifest v2 snapshot `snapshot_20260811T224413.526848Z` was built on clean commit `7f86516`.
It freezes 63 archives/104,658 records and registry SHA-256 `dc2d7068...`; manifest SHA-256 is
`a431d14cd5f693a82d8f21c3c5c7ee05c9d27d2ee003c801db21dcfdc7434263`. All three checksum entries,
the general manifest contract, portable-path scan and first/middle/last exact ZIP/member reads passed. The
summary's `crc_checked=false` means CRC was not redundantly rerun while freezing the manifest; the copied v2
registry had already gated every selected archive with full CRC/coverage/decode. The older 18,095-record v1
snapshot remains historical linkage evidence only, and the v2 snapshot remains partial until all 436 archives
and 722,711 labels are represented.

Clean commit `6e9cc5e` recovered the registered `archive_006/3af7...` canonical identity conflict without
overwriting it. The valid current-source result uses the deterministic full-source-SHA suffix and is selected by
`pose_manifest_v1/snapshot_20260811T232708.554551Z`; the original bad pair remains bound to copied audit
evidence as the one explicit exclusion. The snapshot contains 12,057 records from 12 archives, has manifest
SHA-256 `cdd450e4d7e17d4f34266f199ed4ff61f1ead9584715f1d4b9d3286a97d086e5`, and passed five checksum
entries, the general contract, portable-path scan, and first/middle/last/recovery checksum-validating reads.
The `23:30Z` live registry has since advanced to 71 archives/118,075 videos; it does not mutate this frozen
70-archive snapshot.

Pinned source and download implementation are recorded in the
[CSL-News pipeline](../../../../csl_news_annotation/docs/authority/30_ARCHITECTURE/CSL_NEWS_PIPELINE.md).

First source-audit artifact target:

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1/
```

This trial is not promoted as dataset validation until its CRC, label coverage and video decode report passes.

The scheduled `2026-08-12T00:00Z` trial preserved a useful negative artifact: the old runner selected the
known-corrupt primary `archive_001.zip` and failed with `BadZipFile`. Clean commit `96701de` replaced filename
discovery with typed registry selection and live path/stat/archive-SHA/labels-SHA verification. The clean rerun
at `00:25Z` selected replacement `archive_001` (SHA-256 `911ed805...`), passed all 1,694 members, complete label
coverage and 3/3 decode probes, and is stored under
`source_trial_v1/20260812T002504Z_archive_001_da5711261201/`. Its selection binds registry SHA-256
`da5711261201917ac42f6036f4533642662290cf1019ad5b05c7d379d8e35c9c`.
