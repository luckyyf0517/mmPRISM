# CSL-News Annotation Changelog

Status: current
Owner: CSL-News annotation lane
Authority scope: Material changes to CSL-News annotation boundaries, contracts, and supported operations.
Last reviewed: 2026-08-12

## 2026-08-12

- Retired CSL-News source acquisition and annotation from the revision data-rebuild path. Removed the local
  ZIP/partial-download/label intake, extracted-video cache, live source registry, and ZIP-dependent source
  manifest after disabling their timers; retained published pose outputs, sidecars, QC/review/failure evidence,
  frozen pose manifests, and splits solely as optional checkpoint-side visual-pose evidence.
- Established this workspace from the CSL-News-specific architecture, runbook, and evidence stream.
- Preserved source replacement, quarantine, and immutable incident identities during migration.
- Replaced the operational fixed-shard worker design with a source/config-bound filesystem lease scheduler.
  It supports cooperative pause/resume, stale-worker recovery, and adding/removing workers without redistributing
  live archives; the previous workers were stopped before the scheduler deployment and all data remains retained.
- Paused the old eight-worker transient service set for operator scheduler validation. The new scheduler has not
  been initialized against the live output root and cannot resume annotation without an explicit operator action.
- Benchmarked RTMW3D-L frame batching against the historical single-frame MMPose path. Although batch 16/64
  increased isolated throughput, the outputs were not numerically equivalent; batch 1 remains the only allowed
  setting for the current v1 artifact lineage, and non-default batch configurations receive distinct fingerprints.
