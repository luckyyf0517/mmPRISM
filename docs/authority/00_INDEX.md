# mmPRISM Project Authority

Status: current
Owner: mmPRISM coordinator
Authority scope: Cross-workspace project state, boundaries, shared contracts, and reading order.
Last reviewed: 2026-08-12

## Project State

mmPRISM is rebuilding the paper pipeline from immutable data and explicit provenance. Canonical code
remains under `src/mmprism/`; business execution and status are organized by workspace.

Current cross-workspace blockers:

- Private source inventory, calibration, and historical run provenance are not yet complete.
- The original-submission cam-pose WaveLLM checkpoint still needs an immutable intake record, SHA-256,
  configuration/model identity, historical data/split linkage, and an independent holdout evaluation.
- Full physical radar-cube reconstruction remains blocked on acquisition and calibration evidence.
- The new CSL collection targets approximately 30 recorded participants: ideally 3--4 professional/proficient
  signers if available, plus volunteers who learn from fixed reference videos. Reference content, minimal consent,
  synchronization and pilot QC remain to be frozen.
- Original-submission evidence and paper-facing experiment provenance remain incomplete.

The revision-critical execution path is:

```text
audit and freeze the original-submission WaveLLM semantic initialization
-> recover the CSL-Daily simulation/OmniHand second stage
-> run matched sim2real adaptation and new-real-data experiments
```

Full 436-archive CSL-News reconstruction and retraining do not block revision experiments that use the audited
original-submission checkpoint. Retraining is triggered only if that checkpoint cannot be loaded or audited, is
incompatible with the required historical pose contract, contains split leakage, or is shown by controlled
evaluation to be the downstream bottleneck. See `DEC-044` in the
[decision log](60_DECISIONS/DECISION_LOG.md).

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
