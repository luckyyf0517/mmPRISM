# Paper Revision Authority

Status: current
Owner: Paper revision lane
Authority scope: Current reviewer, evidence-promotion, manuscript, response, and submission workflow.
Last reviewed: 2026-08-13

## Boundary

This workspace audits and promotes frozen evidence into the manuscript and response. It does not own the
production of upstream data, model checkpoints, predictions, or metrics.

## Current State

- Decision letter and reviewer comments are indexed with stable IDs.
- The current manuscript/supplement display inventory and evidence map exist.
- Historical WaveLLM artifacts are staged and are preservation-only until the producing workspace completes a
  stable receipt and controlled audit; original-submission linkage and historical metrics remain unaccepted.
- Results-based response text is intentionally blocked until corresponding evidence is accepted.

Active blockers: original submission import, historical run mapping, the new approximately 30-participant
video-guided CSL collection, reviewer-driven real experiments, availability/license decisions, and Source Data
completion. Historical non-semantic gestures cannot close translation claims, and volunteer recordings must not be
presented as natural fluent-signer generalization.

Next actions: import the original submission, close display-item provenance, await a frozen validated delivery from
[semantic sign-language collection](../../../sign_language_collection/docs/authority/00_INDEX.md), promote completed
workspace evidence, then write manuscript and response changes without strengthening unsupported claims.

## Canonical Locations

- Manuscript: `paper/manuscript/`
- Audit tool: `paper/manager/tools/audit_manuscript.py`
- Artifact records: `paper/manager/evidence/artifacts/`
- Tests: `tests/unit/test_manuscript_audit.py`

## Authority And Operations

- [Availability contract](20_CONTRACTS/AVAILABILITY.md)
- [Display-item registry](20_CONTRACTS/DISPLAY_ITEM_REGISTRY.md)
- [Operator guide](40_OPERATIONS/OPERATOR_GUIDE.md)
- [Reviewer comments brief](50_VALIDATION/REVIEWER_COMMENTS_BRIEF.md)
- [Experiment reproduction](40_OPERATIONS/EXPERIMENT_REPRODUCTION.md)
- [Final submission audit](40_OPERATIONS/FINAL_SUBMISSION_AUDIT.md)
- [Original submission intake](40_OPERATIONS/ORIGINAL_SUBMISSION.md)
- [Manuscript status](50_VALIDATION/MANUSCRIPT_STATUS.md)
- [Editorial language](50_VALIDATION/EDITORIAL_LANGUAGE.md)
- [Review analysis](50_VALIDATION/REVIEW_ANALYSIS.md)
- [Response tracker](50_VALIDATION/RESPONSE_TRACKER.md)
- [Reviewer closure](50_VALIDATION/REVIEWER_CLOSURE.md)
- [Paper evidence map](50_VALIDATION/PAPER_EVIDENCE_MAP.md)
- [Changelog](90_CHANGELOG.md)
- [Logs](../logs/README.md)
