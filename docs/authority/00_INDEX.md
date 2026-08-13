# mmPRISM Project Authority

Status: current
Owner: mmPRISM coordinator
Authority scope: Cross-workspace project state, boundaries, shared contracts, and reading order.
Last reviewed: 2026-08-13

## Project State

mmPRISM is rebuilding the paper pipeline from immutable data and explicit provenance. Canonical code
remains under `src/mmprism/`; business execution and status are organized by workspace.

Current cross-workspace blockers:

- Private source inventory, calibration, and historical run provenance are not yet complete.
- A historical WaveLLM bundle is uploading to the project mirror. It is preservation-only until a stable,
  checksum-bound receipt, checkpoint-format audit, and controlled load complete; directory names and partial bytes
  establish neither original-submission linkage nor usable weights. The recovered mT5-only export remains a
  load-smoke-verified fallback initialization.
- Full physical radar-cube reconstruction remains blocked on acquisition and calibration evidence.
- The new CSL collection targets approximately 30 recorded participants: ideally 3--4 professional/proficient
  signers if available, plus volunteers who learn from fixed reference videos. Reference content, minimal consent,
  synchronization and pilot QC remain to be frozen.
- Original-submission evidence and paper-facing experiment provenance remain incomplete.

The revision-critical execution path is:

```text
preserve and receipt the incoming historical WaveLLM bundle
-> audit its model/data/provenance boundary and select a controlled language initialization
-> recover the CSL-Daily simulation/OmniHand second stage and train new geometry adapters
-> run matched sim2real adaptation and new-real-data experiments
```

The CSL-Daily control is planned as three separately reported stages: camera-pose semantic ceiling, cross-fitted
predicted-mmWave-pose translation, and the same predicted pose plus checkpoint-bound CubeNet frame features. This
is a synthetic-control/historical-replay lane and cannot replace the new real-radar reviewer evidence. Its interface
change is specified by OpenSpec `add-csl-daily-reproduction-controls` and its executing workspace operations are
linked from the Data Rebuild, OmniHand Training, WaveLLM Training, and Paper Revision indexes.

CE-CNSL remains registered as a P1 follow-on public source, but execution is paused under `DEC-054`. No source
download, adapter implementation, pose pilot, or GPU work starts until the CSL-Daily line has produced an accepted
`annotation_v2 -> synthetic FMCW -> OmniHand -> pose-only WaveLLM` stable loop and the coordinator explicitly
reactivates this change. The completed literature/label audit and independent dataset/split identity remain available
for that later review; CE-CNSL never blocks CSL-Daily or the new real-radar collection.

Full CSL-News reconstruction and retraining do not block the CSL-Daily revision path. The incoming historical bundle
does not become a hidden precondition and cannot support historical reproduction claims until its audit passes. See
`DEC-046` in the [decision log](60_DECISIONS/DECISION_LOG.md).

## Workspaces

- [CSL-News annotation archive](../../workspaces/csl_news_annotation/docs/authority/00_INDEX.md)
- [Data rebuild](../../workspaces/data_rebuild/docs/authority/00_INDEX.md)
- [OmniHand training](../../workspaces/omnihand_training/docs/authority/00_INDEX.md)
- [WaveLLM training](../../workspaces/wavellm_training/docs/authority/00_INDEX.md)
- [Semantic sign-language collection](../../workspaces/sign_language_collection/docs/authority/00_INDEX.md)
- [Paper revision](../../workspaces/paper_revision/docs/authority/00_INDEX.md)

Workspace ownership and handoffs are defined in the
[workspace map](10_SCOPE/WORKSPACE_MAP.md). A routine task updates only its owning workspace index.

## Shared Authority

- [Shared contracts](20_CONTRACTS/SHARED_CONTRACTS.md)
- [Engineering rules](20_CONTRACTS/ENGINEERING_RULES.md)
- [Tensor contracts](20_CONTRACTS/TENSOR_CONTRACTS.md)
- [Data splits](20_CONTRACTS/DATA_SPLITS.md)
- [Run artifacts](20_CONTRACTS/RUN_ARTIFACTS.md)
- [Model assets](20_CONTRACTS/MODEL_ASSETS.md)
- [Documentation governance](20_CONTRACTS/DOCUMENT_GOVERNANCE.md)
- [System architecture](30_ARCHITECTURE/SYSTEM_ARCHITECTURE.md)
- [Research terms and end-to-end execution](30_ARCHITECTURE/RESEARCH_EXECUTION_MODEL.md)
- [Release audit](40_OPERATIONS/RELEASE_AUDIT.md)
- [Decision log](60_DECISIONS/DECISION_LOG.md)
- [Project changelog](90_CHANGELOG.md)

## Research Notes

- [Literature reading notes](../literature/README.md) are non-authoritative research context. Any project rule
  derived from a paper must be explicitly promoted into the decision log or another indexed Authority page.

## Verification

```bash
uv run python scripts/audit_docs.py
uv run pytest tests/unit/test_document_governance.py
git diff --check
```

Project history is indexed under [docs/logs](../logs/README.md). Logs do not become current
Authority without an explicit decision and Authority update.
