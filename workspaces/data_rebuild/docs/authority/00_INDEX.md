# Data Rebuild Authority

Status: current
Owner: Data rebuild lane
Authority scope: Current source intake, radar rebuild, split, quarantine, and data delivery workflow.
Last reviewed: 2026-08-12

## Boundary

This workspace turns frozen source identities into validated, task-specific model-ready products. It owns
radar processing recovery until calibrated cube production has an independent stable handoff.

## Current State

- Private project data is confirmed on NAS but source inventory and accepted transfer batches are incomplete.
- Raw/range-Doppler contracts and analytic tests exist; beamforming and physical axes await calibration.
- Task-specific Parquet delivery is accepted; readers/materializer and real products remain pending.
- Partial CSL-News visual pose data is upstream evidence, not final radar/model-ready delivery.

Active blockers: NAS intake, acquisition/channel/calibration evidence, simulation provenance, and final
subject-aware splits.

Next actions: accept metadata/calibration first, bind a real radar fixture, validate delivery capacity and
reader parity, then build immutable task products.

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
- [Data registry](50_VALIDATION/DATA_REGISTRY.md)
- [Changelog](90_CHANGELOG.md)
- [Logs](../logs/README.md)
