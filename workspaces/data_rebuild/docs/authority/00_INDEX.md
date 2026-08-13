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
  preserved historical run evidence pending its separate receipt/audit; canonical code must not write to or train
  directly from any of these paths.
- The CSL-Daily raw upload is complete at the dedicated preservation root
  `external/csl_daily/csl_daily_original_20260812/`. The retained source of record is the expanded JPEG-frame tree,
  official labels/split metadata, and review MP4s; the complete tarball and its transfer splits were removed with a
  recorded cleanup inventory. No historical full CSL-Daily pose, feature, signal, or predicted-pose
  product is present: legacy JSON files are metadata/path evidence only. The existing 54-sample `annotation_v1` pilot
  has material contract failures (NaN hand values and missing native/scores/confidence/validity) and is diagnostic-only.
  A complete, new versioned `annotation_v2` camera-pose rebuild is thus P0 and mandatory before simulation, not an
  optional improvement. The gated path is GPU smoke -> frozen full-corpus `annotation_v2` and QC -> separately labelled pre-beamforming synthetic-FMCW
  delivery -> runtime cube conversion/OmniHand -> pose-only WaveLLM controls. Feature/fusion is a non-blocking
  later comparison. It is documented in the
  [CSL-Daily reproduction operation](40_OPERATIONS/CSL_DAILY_REPRODUCTION.md).
- CE-CNSL is registered as a P1 follow-on source for vocabulary and heterogeneous-domain expansion, with an
  independent manifest, split, label namespace, artifact root, and result identity. Execution is paused. During the
  late stable phase of CSL-Daily, explicit project-owner authorization may unlock source download and immutable
  receipt only. Adapter, label repair, the 120--240-sequence pilot, processing, and GPU work remain inactive until the
  CSL-Daily end-to-end stable loop is accepted and this work is explicitly reactivated. See the
  [CE-CNSL intake and pose pilot](40_OPERATIONS/CE_CNSL_INTAKE_AND_POSE_PILOT.md).
- A historical WaveLLM bundle is preserved under the project mirror `log/archived/`. It is preservation-only until
  stable inventory, checksum, format, and controlled-load receipt completes. The recovered CSL-News-derived mT5-only
  export remains a fallback, not historical end-to-end evidence.

Active blockers: NAS intake, acquisition/channel/calibration evidence, simulation provenance, and final
subject-aware splits.

Next actions: complete the `annotation_v2` GPU gate, then run/freeze
and QC the full corpus, create a distinct
control split (the legacy validation/test files are identical), and
build immutable synthetic-FMCW
and pose-only products for OmniHand and WaveLLM. Cube generation occurs in the runtime adapter rather than a
persisted delivery. Feature/fusion does not block the first training loop. The historical bundle audit, real-radar
calibration/metadata, and subject-aware real-data split remain separate blockers.

CE-CNSL has no active next action. In CSL-Daily's late stable phase, await explicit project-owner authorization before
any optional source download; otherwise revisit its saved OpenSpec only after the stable-loop activation gate.

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
- [CSL-Daily legacy-path replay runbook](40_OPERATIONS/CSL_DAILY_LEGACY_REPLAY.md)
- [CE-CNSL intake and pose pilot](40_OPERATIONS/CE_CNSL_INTAKE_AND_POSE_PILOT.md)
- [Data registry](50_VALIDATION/DATA_REGISTRY.md)
- [Changelog](90_CHANGELOG.md)
- [Logs](../logs/README.md)
