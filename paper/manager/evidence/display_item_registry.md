# Display Item Provenance Registry

Status: `all_current_items_registered_provenance_pending`
Last Updated: `2026-08-11`
Role: `display_item_to_source_data_control`
Audit ID: `PAPER-AUDIT-001`

## Scope And Evidence Boundary

This registry covers the current Overleaf snapshot `paper/manuscript@3242a40631ec5198e66fa8592763235c108513b2`.
The machine-readable source is `artifacts/manuscript_inventory_v2.json` with SHA-256
`db01f16e1fcfe7b22743eab8671820cbc428d816f55c0ed69af35c25ba0647d7`.

- Main manuscript: 6 figure environments containing 7 figure display items, plus 2 table items.
- Supplementary ZIP: 5 figure items and 6 table items.
- Total: 19 LaTeX environments and 20 independently captioned display items.
- Registration proves location and asset linkage only. It does not validate any scientific value.
- `expected artifact` paths below are deliverable contracts, not claims that those files exist now.
- Supplementary Tables S2-S6 carry the literal comment `示例表格内容（替换为真实数据）`; their current
  numbers are excluded from evidence and remain `placeholder_unverified` until regenerated.

## Status Vocabulary

| Status | Meaning |
|---|---|
| `asset_provenance_pending` | Diagram/photo exists, but editable source, authorship, license or generation lineage is incomplete. |
| `scientific_provenance_pending` | Quantitative or qualitative result lacks a complete dataset -> split -> run -> metric -> artifact chain. |
| `placeholder_unverified` | Source explicitly marks the content as example data; no number may be cited or promoted. |
| `evidence_ready` | All registered fields and hashes are verified. No current display item has this status. |

## Main Manuscript

| Display ID | Location / Label / Asset | Caption Scope | Source Data Requirement And Expected Artifact | Provenance Chain Required | Reviewer / Evidence | Status |
|---|---|---|---|---|---|---|
| `DISPLAY-MAIN-FIG-01` | `chapter/1_introduction.tex:16`; `fig:teaser`; `pics/ZZQ-25112636041-3780.jpg` | “Schematic of the mmPRISM framework…”; panels d/e also state pose and translation results | Mixed schematic/result item. Preserve editable panel sources plus sample-level values/predictions behind d/e in `source_data/main_fig01/` with `figure_manifest.json`. | Dataset/split, mmPRISM and mmHand run/checkpoint, MPJPE/PCK and translation protocols, panel-generation script: all pending. | `R1-2`,`R1-4`,`R2-1`,`ED-COMP-8`; `EVID-REV-ARCH`,`EVID-REV-EFF`,`EVID-REV-DATASET` | `scientific_provenance_pending` |
| `DISPLAY-MAIN-FIG-02` | `chapter/2_results.tex:13`; `fig:network`; `pics/network.png` | “Architecture of the volumetric feature extraction module.” | No numeric Source Data apparent. Preserve editable diagram source, export command, font/icon licenses and final hash in `source_data/main_fig02/figure_manifest.json`. | Dataset/split/run/metric: n/a for schematic. Exact code/config commit and diagram generator/source: pending. | `R2-5`,`ED-COMP-8`; `EVID-REV-ATTN` | `asset_provenance_pending` |
| `DISPLAY-MAIN-FIG-03` | `chapter/2_results.tex:21`; `fig:temporal`; `pics/temporal.png` | “Temporal aggregation for dynamic consistency.” | No numeric Source Data apparent. Preserve editable diagram source and export provenance in `source_data/main_fig03/figure_manifest.json`. | Exact temporal implementation/config commit and diagram generator/source: pending; any claim of occlusion resolution must bind experiments. | `R1-5`,`ED-COMP-8`; `EVID-REV-REAL` | `asset_provenance_pending` |
| `DISPLAY-MAIN-FIG-04` | `chapter/2_results.tex:56`; `fig:overall_comparison`; `pics/overall_comparison.png` | “Robustness of kinematic reconstruction across operational regimes.” | Required: per-sample/per-sequence MPJPE and PCK for each method and distance, aggregation inputs and uncertainty in `source_data/main_fig04/`. | Collected-radar dataset and held-out split; mmPRISM/mmHand runs and checkpoints; versioned MPJPE/PCK protocol; generation script: all pending. | `R1-3`,`R1-4`,`R2-3`,`ED-COMP-8`; `EVID-REV-REAL`,`EVID-REV-DATASET` | `scientific_provenance_pending` |
| `DISPLAY-MAIN-FIG-05` | `chapter/2_results.tex:65`; `fig:qualitative_demo`; `pics/rec_result.png` | “Visual validation of structural integrity…” | Required: selected sample IDs, uncropped reference frames, ground truth, both methods' predictions, selection rule and consent/release metadata in `source_data/main_fig05/`. | Collected-radar dataset/split, exact run/checkpoint per prediction, qualitative selection protocol and render script: all pending. | `R1-5`,`R2-3`,`ED-COMP-8`; `EVID-REV-REAL` | `scientific_provenance_pending` |
| `DISPLAY-MAIN-FIG-06` | `chapter/2_results.tex:139`; `fig:training`; `pics/training.pdf` | “Domain adaptation via shallow-layer alignment.” | Required: plotted embeddings/labels, sampled IDs, before/after checkpoints, t-SNE parameters/seeds and any quantitative alignment values in `source_data/main_fig06/`. | Matched synthetic/real manifests and splits; shallow-adaptation run/checkpoints; versioned embedding/alignment protocol; plot script: all pending. | `R1-4a`,`R2-2`,`ED-COMP-8`; `EVID-REV-DA`,`EVID-REV-SYNREAL` | `scientific_provenance_pending` |
| `DISPLAY-MAIN-FIG-07` | `chapter/2_results.tex:158`; `fig:llm`; `pics/llm.pdf` | “Context-aware translation framework.” | No numeric Source Data apparent. Preserve editable diagram source, export provenance and exact architecture/config binding in `source_data/main_fig07/figure_manifest.json`. | Pose/mmWave feature contracts, mT5 checkpoint/config and diagram generator/source: pending; downstream evidence belongs to Table 2. | `R2-1`,`R1-5`,`ED-COMP-8`; `EVID-REV-ARCH`,`EVID-REV-REAL` | `asset_provenance_pending` |
| `DISPLAY-MAIN-TABLE-01` | `chapter/1_introduction.tex:49`; `tab:related_work`; inline table | “Comparison with previous wireless hand sensing works.” | Required: row-level citation/extraction ledger, protocol comparability decisions and mmPRISM sample-level result links in `source_data/main_table01/`. | Literature sources and extraction date; mmPRISM dataset/split/run/checkpoint; metric definitions and table generator: all pending. | `R1-1`,`R2-6`,`ED-COMP-8`; `EVID-REV-XMODAL`,`EVID-REV-EFF` | `scientific_provenance_pending` |
| `DISPLAY-MAIN-TABLE-02` | `chapter/2_results.tex:166`; `tab:llm_results`; inline table | “Benchmarking translation performance.” | Required: sample IDs, references, predictions for every row/model, per-sample metric inputs and aggregation outputs in `source_data/main_table02/`. | Exact translation datasets/splits, each run/checkpoint, BLEU/ROUGE/SBERT/SimCSE protocol versions and table generator: all pending. | `R1-4`,`R2-1`,`R2-6`,`ED-COMP-8`; `EVID-REV-ARCH`,`EVID-REV-XMODAL`,`EVID-REV-DATASET` | `scientific_provenance_pending` |

## Supplementary Information

All supplementary locations refer to `mian.tex` inside
`paper/manuscript/supplementary/Supplementary_Information.zip`.

| Display ID | Location / Label / Asset | Caption Scope | Source Data Requirement And Expected Artifact | Provenance Chain Required | Reviewer / Evidence | Status |
|---|---|---|---|---|---|---|
| `DISPLAY-SUPP-FIG-01` | `mian.tex:91`; `fig:s1`; `pics/mmwave_cube.pdf` | “4D mmWave Cube…” | Required: representative raw ADC/cube sample, transform configuration, displayed slice/voxel values, colour mapping and render manifest in `source_data/supp_fig01/`. | Radar source sample/manifest, FMCW/beamforming config and commit, cube contract, render script: pending. | `R1-6`,`ED-COMP-4`,`ED-COMP-8`; `EVID-PAPER-INVENTORY` | `scientific_provenance_pending` |
| `DISPLAY-SUPP-FIG-02` | `mian.tex:101`; `fig:s2`; `pics/tsne.pdf` | “t-SNE visualization of feature alignment…” | Required: embedding matrix, domain labels, stable sample IDs, sampling rule, seed/perplexity/iterations and plotting environment in `source_data/supp_fig02/`. | Synthetic/real manifests and splits, before/after checkpoints, embedding protocol and plot script: pending. | `R1-4a`,`R2-2`,`ED-COMP-8`; `EVID-REV-DA`,`EVID-REV-SYNREAL` | `scientific_provenance_pending` |
| `DISPLAY-SUPP-FIG-03` | `mian.tex:111`; `fig:s3`; `pics/scene.png` | “Experimental setups for real-world mmPRISM data collection…” | Preserve original uncropped photos, capture date/scene/device metadata, panel-selection/export record, permission/consent and privacy review in `source_data/supp_fig03/`. | Collected-radar dataset/scene registry and collection protocol; run/checkpoint/metric: n/a; composition source: pending. | `R1-3`,`R1-4b/c/d`,`R2-3`,`R2-4`,`ED-COMP-8`; `EVID-REV-REAL`,`EVID-REV-DATASET` | `asset_provenance_pending` |
| `DISPLAY-SUPP-FIG-04` | `mian.tex:121`; `fig:s4`; `pics/antenna_array_scale_impact.pdf` | “Impact of antenna array size…” | Required: sample-level MPJPE/PCK for every antenna configuration, configuration files and aggregate/uncertainty outputs in `source_data/supp_fig04/`. | Synthetic or collected dataset/split, four matched runs/checkpoints, versioned MPJPE/PCK protocol and plot script: pending. | `ED-COMP-8`; `EVID-PAPER-INVENTORY` | `scientific_provenance_pending` |
| `DISPLAY-SUPP-FIG-05` | `mian.tex:131`; `fig:s5`; `pics/point_quality_vs_llm_performance.pdf` | “Impact of mmw-pose reconstruction quality…” | Required: checkpoint-by-sample pose/translation predictions, MPJPE/BLEU inputs, aggregation/correlation output and selection rule in `source_data/supp_fig05/`. | Dataset/split, ordered CubeNet and SLU checkpoints, MPJPE/BLEU protocol and plot script: pending. | `R2-1`,`ED-COMP-8`; `EVID-REV-ARCH` | `scientific_provenance_pending` |
| `DISPLAY-SUPP-TABLE-01` | `mian.tex:143`; `tab:s1`; inline table | “Specifications of mmWave radar antenna array configurations…” | Required configuration provenance: hardware design/source, virtual-array derivation and versioned configuration export in `source_data/supp_table01/`. | Hardware/config registry; run/checkpoint/metric: n/a; table generator/source: pending. | `ED-COMP-8`; `EVID-PAPER-INVENTORY` | `scientific_provenance_pending` |
| `DISPLAY-SUPP-TABLE-02` | `mian.tex:165`; `tab:s2`; inline table | “Comparison of temporal aggregation methods…” | Current values are prohibited as evidence. Expected: per-sample MPJPE/PCK, matched seeds/configs and aggregates in `source_data/supp_table02/`. | Dataset/split, single-frame/LSTM/Transformer runs and checkpoints, MPJPE/PCK protocol and table generator: missing. | `R1-5`,`ED-COMP-8`; `EVID-REV-REAL` | `placeholder_unverified` |
| `DISPLAY-SUPP-TABLE-03` | `mian.tex:186`; `tab:s3`; inline table | “Ablation study on multi-modal fusion strategies…” | Current values are prohibited as evidence. Expected: sample predictions, matched no/additive/adaptive runs and all metric inputs in `source_data/supp_table03/`. | Translation dataset/split, matched runs/checkpoints, BLEU/ROUGE/SBERT/SimCSE protocols and table generator: missing. | `R1-5`,`ED-COMP-8`; `EVID-REV-REAL` | `placeholder_unverified` |
| `DISPLAY-SUPP-TABLE-04` | `mian.tex:205`; `tab:s4`; inline table | “Ablation study on the impact of multi-stage training…” | Current values are prohibited as evidence. Expected: sample predictions, matched training budgets/configs and metric inputs in `source_data/supp_table04/`. | Cam-pose/synthetic/real datasets and splits, stage checkpoints/runs, translation protocols and table generator: missing. | `R2-2`,`ED-COMP-8`; `EVID-REV-DA` | `placeholder_unverified` |
| `DISPLAY-SUPP-TABLE-05` | `mian.tex:224`; `tab:s5`; inline table | “Performance of mmPRISM on generalization tests…” | Current values are prohibited as evidence. Expected: subject/sequence/environment assignments, leakage audit, sample-level MPJPE/PCK and uncertainty in `source_data/supp_table05/`. | Collected-radar manifest and three frozen splits, matched runs/checkpoints, MPJPE/PCK protocol and table generator: missing. | `R1-3`,`R1-4c/d`,`R2-3`,`R2-4`,`ED-COMP-8`; `EVID-REV-REAL`,`EVID-REV-DATASET` | `placeholder_unverified` |
| `DISPLAY-SUPP-TABLE-06` | `mian.tex:243`; `tab:s6`; inline table | “Performance under real-world noisy scenarios.” | Current values are prohibited as evidence. Expected: condition registry, stable sample IDs, per-sample MPJPE/PCK and uncertainty in `source_data/supp_table06/`. | Collected-radar condition-stratified split, matched runs/checkpoints, MPJPE/PCK protocol and table generator: missing. | `R1-3`,`R2-3`,`ED-COMP-8`; `EVID-REV-REAL` | `placeholder_unverified` |

## Promotion Gate

A display item may move to `evidence_ready` only when:

1. dataset manifest and frozen split hashes are present;
2. every paper-facing run and checkpoint is registered with code/config/environment hashes;
3. metric protocol and sample-level inputs/predictions are retained;
4. generation script reproduces the final asset or table from those artifacts;
5. expected Source Data artifact exists, has a checksum and passes an independent read/reproduction check;
6. reviewer/evidence mappings and manuscript wording are synchronized;
7. placeholder markers are removed only after real regenerated values replace them.

The original-submission comparison remains pending. If its display inventory differs, create a separate snapshot;
do not reuse current display IDs for materially different items.
