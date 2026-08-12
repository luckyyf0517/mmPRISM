# CSL-News Annotation Authority

Status: current
Owner: CSL-News annotation lane
Authority scope: Current CSL-News source validation, annotation, QC, and manifest workflow.
Last reviewed: 2026-08-12

## Boundary

This workspace consumes the pinned official CSL-News RGB/label source and produces source-bound visual
pose+caption manifests. Its outputs are intermediate visual evidence, not calibrated radar training data.

## Current State

- Source-integrity v2 selects passed primary or replacement archives without overwriting incident evidence.
- RTMW3D workers publish source-bound immutable artifacts and quarantine old/unbound results.
- The former static eight-lane run is intentionally paused for scheduler work. The replacement control plane
  uses durable archive leases, cooperative pause/resume, and elastic worker capacity; it has not yet been
  enabled for production consumption.
- Partial source, pose, metadata, and split snapshots are validated; the full 436-archive build is incomplete.

Active blockers: full-source completion, signer/subject metadata, and historical inference environment.

Next actions: CPU-test and operator-test the lease scheduler while paused, resume only through the new
control plane, then finish source/annotation coverage, rerun full identity audit, freeze the final manifests,
and perform the planned clean historical cross-check.

## Canonical Locations

- Code: `src/mmprism/data/csl_news*.py`
- Config: `configs/data/csl_news_*.yaml`
- Scripts: `scripts/run_csl_news_*.sh`
- Tests: `tests/unit/test_csl_news*.py`

## Authority And Operations

- [Pipeline contract](30_ARCHITECTURE/CSL_NEWS_PIPELINE.md)
- [Annotation runbook](40_OPERATIONS/ANNOTATION_RUNBOOK.md)
- [Changelog](90_CHANGELOG.md)
- [Logs](../logs/README.md)
