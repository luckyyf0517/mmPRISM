# Documentation Governance

Status: current
Owner: mmPRISM coordinator
Authority scope: Placement, update, handoff, compatibility, and validation rules for project documents.
Last reviewed: 2026-08-12

## Placement

- Authority states current truth and ownership.
- A Runbook explains a supported named operation.
- A dated Log preserves a run, incident, audit, import, or milestone.
- OpenSpec plans code, interface, data-contract, or ownership changes.

Each workspace has one current-status summary: `docs/authority/00_INDEX.md`. Scoped Authority pages
contain durable contracts or validation registers and must be linked from that index. A Log never becomes
current truth by recency alone.

## Minimal Updates

- Routine progress: update only the owning workspace index when its actionable state changed.
- Contract or ownership change: use OpenSpec, then update the owning Authority and changelog.
- Completed run or incident: add a dated Log or reference an immutable artifact.
- Paper promotion: update the paper evidence map only after producer evidence is frozen.
- Project index: update only for cross-workspace phase or blocker changes.

README files describe stable scope and should not track daily progress. Do not create a dashboard, task
tracker, handoff report, or optional directory without an operational need.

## Handoff

Routine handoff uses task, state, result, evidence, next action, and blocker. Cross-workspace delivery also
provides producer commit, immutable location, manifest/inventory hash, and validation status, preferably in
an existing manifest summary or receipt.

## Compatibility

When a canonical document moves, its old path contains only one link to the new location and the standard
compatibility statement. It contains no metadata or independent authority claims.

## Verification

```bash
uv run python scripts/audit_docs.py
uv run pytest tests/unit/test_document_governance.py
git diff --check
```

The versioned behavior contract is
[business workspace governance](../../../openspec/specs/business-workspace-governance/spec.md).
