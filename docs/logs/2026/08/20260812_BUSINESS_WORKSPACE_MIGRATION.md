# Business Workspace Migration

Status: historical
Owner: mmPRISM coordinator
Evidence scope: Final acceptance record for the documentation and business-workspace reorganization.
Recorded: 2026-08-12

## Identity

- Baseline repository commit: `decae360c7e51497030183fcf1545a4fa5aaf3c7`
- Manuscript submodule: `3242a40631ec5198e66fa8592763235c108513b2`
- Decision: `DEC-039`
- OpenSpec archive: `2026-08-12-reorganize-business-workspaces`
- Canonical capability: `business-workspace-governance`

## Result

- Established project Authority and five business workspaces.
- Kept shared source, configuration, and tests at repository root.
- Replaced 69 old Markdown authority/evidence paths with compatibility entrypoints.
- Preserved all 325 baseline stable identifiers and added only `DEC-039`.
- Preserved all eight tracked evidence JSON SHA-256 values and the manuscript submodule identity.
- Validated one frozen CSL-News annotation-to-data-rebuild delivery without a duplicate handoff report.

## Verification

Passed:

```text
documentation governance audit: 0 issues
document governance + manuscript audit tests: 9 passed
Ruff for governance tool/tests: passed
OpenSpec strict validation: passed
git diff --check: passed
```

The full repository test run reached 210 passed and 3 failed. The failures are isolated to pre-existing
uncommitted WaveLLM/distributed training work outside this migration: an undefined
`write_single_rank_predictions`, a missing `_runtime_payload` argument, and scalar tensor byte hashing in
`src/mmprism/training/distributed.py`. This migration did not modify those files.
