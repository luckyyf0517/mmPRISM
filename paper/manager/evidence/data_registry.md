# Data Registry

Status: `awaiting_asset_discovery`
Last Updated: `2026-08-11`
Role: `dataset_and_split_provenance`

## Data Families

| ID | Family | Source Location | License/Access | Raw Size | Manifest | Validation | Paper Role | Status |
|---|---|---|---|---|---|---|---|---|
| `DATASET-CSL-DAILY` | CSL-Daily | unknown | unknown | unknown | missing | missing | training/evaluation candidate | blocked |
| `DATASET-CSL-NEWS` | CSL-News | unknown | unknown | unknown | missing | missing | training/evaluation candidate | blocked |
| `DATASET-COLLECTED-BASE` | collected_base | unknown | private | unknown | missing | missing | real radar pose | blocked |
| `DATASET-COLLECTED-DEMO` | collected_demo | unknown | private | unknown | missing | missing | development/demo | blocked |
| `DATASET-COLLECTED-CSL` | collected_csl | unknown | private | unknown | missing | missing | real sign language | blocked |

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
