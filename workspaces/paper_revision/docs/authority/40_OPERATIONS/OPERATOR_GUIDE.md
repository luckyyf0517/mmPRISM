# Paper Revision Operator Guide

Status: current
Owner: Paper revision lane
Authority scope: Daily intake, evidence promotion, manuscript update, and handoff procedure.
Last reviewed: 2026-08-12

## Start

Read only what is needed:

1. [paper revision index](../00_INDEX.md)
2. the scoped Authority page for the selected operation
3. the producing workspace index when consuming data or model evidence
4. the relevant dated Log or immutable artifact

Do not reconstruct current status from migration snapshots or compatibility entrypoints.

## Select Work

- Use a stable existing task/reviewer/evidence ID when one applies.
- Pick the highest-priority actionable item from the workspace index.
- Update only the owning index for routine status changes.
- Use OpenSpec before changing code, interfaces, data contracts, or ownership.

## Accept Evidence

Before promoting producer evidence, verify:

```text
producer workspace
producer commit
immutable artifact location
manifest or inventory hash
validation status
```

The identity may come from an existing manifest, receipt, or Log. Do not create a second handoff report.
Paper-facing numbers also require sample-level predictions and a versioned metric protocol.

## Update The Manuscript

1. Confirm the target claim/display item in the
   [paper evidence map](../50_VALIDATION/PAPER_EVIDENCE_MAP.md).
2. Confirm evidence promotion status and claim strength.
3. Modify and commit within the `paper/manuscript` submodule.
4. Update [manuscript status](../50_VALIDATION/MANUSCRIPT_STATUS.md) and the relevant tracker.
5. Run the manuscript audit and the applicable narrow verification.

Private Overleaf credentials remain in the root `.env` and must not enter Git.

## Finish

Routine handoff is concise:

```text
Task: <stable ID>
State: <commit or working-tree note>
Result: <one sentence>
Evidence: <existing path/artifact, or n/a>
Next: <one action>
Blocker: <one item, or none>
```

Add a dated Log only for a durable audit, incident, import, milestone, or completed formal run.
