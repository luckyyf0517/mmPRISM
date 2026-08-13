# Data Rebuild Authority

Status: current
Owner: Data rebuild lane
Authority scope: Current source intake, radar rebuild, split, quarantine, and data delivery workflow.
Last reviewed: 2026-08-13

## Boundary

This workspace turns frozen source identities into validated, task-specific model-ready products. It owns
radar processing recovery until calibrated cube production has an independent stable handoff. New participant
recruitment, sign-language content, acquisition and session QC are owned by `sign_language_collection`; this
workspace begins only after receiving its immutable raw-session manifest and validation status.

## Current State

- Private project data is confirmed on NAS but source inventory and accepted transfer batches are incomplete.
- Raw/range-Doppler contracts and analytic tests exist; beamforming and physical axes await calibration.
- Task-specific Parquet delivery v2 reader, planner, materializer, and validator are implemented and fixture-verified;
  real products remain blocked by source, calibration, and final-split evidence.
- Archived CSL-News visual pose output is optional checkpoint-side forensic evidence only; CSL-Daily intake and
  the new semantic CSL collection are the active public/real-data paths. No CSL-News source bytes remain local.
- `/mnt/gfs/yanyifan/mmPRISM/{dataset,pretrained_models,log}` is an old-project mirror preserved in its historical
  layout. `dataset` is legacy metadata/split evidence, `pretrained_models` is a legacy model mirror, and `log` is
  incoming historical run evidence with checkpoints still transferring; canonical code must not write to or train
  directly from any of these paths.
- The complete CSL-Daily raw source is being uploaded directly to the dedicated preservation root
  `external/csl_daily/csl_daily_original_20260812/`. It remains an unaccepted source until stable inventory and
  validation, despite its direct `external/` location. The next gated path is receipt -> baseline camera-pose
  QC -> separately labelled skeleton-simulation delivery -> OmniHand/WaveLLM controls; it is documented in the
  [CSL-Daily reproduction operation](40_OPERATIONS/CSL_DAILY_REPRODUCTION.md).
- CE-CNSL is registered as a P1 follow-on source for vocabulary and heterogeneous-domain expansion, with an
  independent manifest, split, label namespace, artifact root, and result identity. Execution is paused: do not
  download the source, implement an adapter, run the 120--240-sequence pilot, or allocate GPU capacity until the
  CSL-Daily end-to-end stable loop is accepted and the coordinator explicitly reactivates this work. See the
  [CE-CNSL intake and pose pilot](40_OPERATIONS/CE_CNSL_INTAKE_AND_POSE_PILOT.md).
- A historical WaveLLM bundle is uploading under the project mirror `log/archived/`. It is preservation-only until
  stable inventory, checksum, format, and controlled-load receipt completes. The recovered CSL-News-derived mT5-only
  export remains a fallback, not historical end-to-end evidence.

Active blockers: NAS intake, acquisition/channel/calibration evidence, simulation provenance, and final
subject-aware splits.

Next actions: preserve the incoming historical bundle and await upload completion, then receipt/audit it; accept
CSL-Daily metadata/source, freeze `annotation_v1` and QC before any candidate improvement, create a distinct
control split (the legacy validation/test files are identical), and build immutable synthetic task products. Real
radar calibration/metadata and the subject-aware real-data split remain separate blockers.

CE-CNSL has no active next action. Revisit its saved OpenSpec only after the CSL-Daily stable-loop activation gate.

Full CSL-News source reconstruction and training may be useful for future provenance or ceiling work, but it does not
block the CSL-Daily revision path. It requires an explicit future decision and cannot restore the missing historical
end-to-end checkpoint.

## Accepted Upstream Delivery

The workspace accepts the CSL-News source-manifest v2 snapshot only as interim visual-source evidence:

```text
producer: csl_news_annotation
producer commit: 7f86516
immutable snapshot: snapshot_20260811T224413.526848Z
manifest SHA-256: a431d14cd5f693a82d8f21c3c5c7ee05c9d27d2ee003c801db21dcfdc7434263
validation: exact registry/checksums/contract/portable paths and sampled reads passed
boundary: partial 63-archive snapshot; not a final model-ready delivery
```

Canonical evidence: [source manifest log](../../../csl_news_annotation/docs/logs/2026/08/20260811_SOURCE_MANIFEST.md).

## Canonical Locations

- Code: `src/mmprism/data/`, `src/mmprism/radar/`
- Config: `configs/data/`
- Scripts: data download, integrity, and rebuild scripts under `scripts/`
- Tests: data/radar unit and contract tests under `tests/`

## Authority And Operations

- [Parquet delivery contract](20_CONTRACTS/DATA_DELIVERY_PARQUET.md)
- [Data intake](40_OPERATIONS/DATA_INTAKE.md)
- [Data rebuild runbook](40_OPERATIONS/DATA_REBUILD.md)
- [CSL-Daily reproduction operation](40_OPERATIONS/CSL_DAILY_REPRODUCTION.md)
- [Data registry](50_VALIDATION/DATA_REGISTRY.md)
- [CE-CNSL intake and pose pilot](40_OPERATIONS/CE_CNSL_INTAKE_AND_POSE_PILOT.md)
- [Changelog](90_CHANGELOG.md)
- [Logs](../logs/README.md)
