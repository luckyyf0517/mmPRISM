# CSL-News Annotation Archive Authority

Status: current
Owner: CSL-News annotation lane
Authority scope: Archived CSL-News visual-pose evidence and the boundary preventing its accidental resumption.
Last reviewed: 2026-08-12

## Boundary

This workspace retains completed source-bound CSL-News visual pose+caption artifacts as intermediate
evidence. It does not consume a local CSL-News source, produce new annotations, or deliver calibrated radar
training data.

## Current State

- The CSL-News ZIPs, partial downloads, labels, extracted-video cache, live source registry, and ZIP-dependent
  source-manifest artifacts were removed on 2026-08-12 after the revision priority shifted to CSL-Daily.
- The RTMW3D annotation root, its sidecars, failure evidence, QC/reviews/run metadata, frozen pose manifests,
  and partial split evidence are retained as immutable checkpoint-side visual-pose evidence.
- Scheduler intent remains `paused`, and its source-integrity/status timers are disabled. It is not resumable:
  the local source registry and source bytes no longer exist.
- The retained partial outputs are neither final training data nor a paper dataset-size/generalization claim.

There are no active CSL-News operational blockers because this lane is archived. Any future reactivation
requires a new source intake, source/license review, immutable manifest lineage, and an explicit decision.

Next action: do not run CSL-News download, integrity, scheduler, or annotation commands. CSL-Daily intake
and the real semantic CSL collection are the active data-rebuild paths.

## Canonical Locations

- Code: `src/mmprism/data/csl_news*.py`
- Config and scripts: retained as historical/recovery references only; not supported operations
- Tests: `tests/unit/test_csl_news*.py`

## Authority And Operations

- [Pipeline contract](30_ARCHITECTURE/CSL_NEWS_PIPELINE.md)
- [Annotation runbook](40_OPERATIONS/ANNOTATION_RUNBOOK.md)
- [Changelog](90_CHANGELOG.md)
- [Logs](../logs/README.md)
