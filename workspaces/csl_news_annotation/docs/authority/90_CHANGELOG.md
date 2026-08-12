# CSL-News Annotation Changelog

Status: current
Owner: CSL-News annotation lane
Authority scope: Material changes to CSL-News annotation boundaries, contracts, and supported operations.
Last reviewed: 2026-08-12

## 2026-08-12

- Established this workspace from the CSL-News-specific architecture, runbook, and evidence stream.
- Preserved source replacement, quarantine, and immutable incident identities during migration.
- Replaced the operational fixed-shard worker design with a source/config-bound filesystem lease scheduler.
  It supports cooperative pause/resume, stale-worker recovery, and adding/removing workers without redistributing
  live archives; the previous workers were stopped before the scheduler deployment and all data remains retained.
- Paused the old eight-worker transient service set for operator scheduler validation. The new scheduler has not
  been initialized against the live output root and cannot resume annotation without an explicit operator action.
