## Why

mmPRISM currently mixes current architecture, operational status, dated run evidence, task tracking, and paper revision control across `docs/architecture/` and `paper/manager/`. This creates competing sources of truth and makes routine handoff require updates to several large files.

The repository needs lightweight business workspaces that make ownership and execution paths obvious while keeping reusable Python code, configuration infrastructure, and tests in their canonical root locations.

## What Changes

- Introduce project authority under `docs/authority/` for cross-workspace scope, shared contracts, and governance.
- Introduce five business workspaces: `csl_news_annotation`, `data_rebuild`, `omnihand_training`, `wavellm_training`, and `paper_revision`.
- Give each workspace one concise `docs/authority/00_INDEX.md` as its current-status entrypoint and one `90_CHANGELOG.md`; create other authority pages and log directories only when they carry real content.
- Classify existing documents as project authority, workspace authority, reusable runbook, immutable log, or compatibility pointer before moving them.
- Replace old authority locations with compatibility pointers during one coordinated cutover.
- Reduce routine handoff to task ID, commit, result, evidence reference, next action, and blocker. Require a frozen artifact identity only for cross-workspace data or evidence transfer.
- Use OpenSpec only for changes to code, interfaces, contracts, or ownership. One-off runs use a runbook when needed and produce a dated log.
- Preserve stable task, decision, evidence, dataset, split, and run IDs throughout migration.
- Keep shared implementation in `src/mmprism/`, shared configuration infrastructure in `configs/`, and automated verification in `tests/`.

### Non-goals

- Moving or duplicating shared code to match workspace directories.
- Changing runtime behavior, schemas, training protocols, data formats, or public APIs.
- Rewriting, moving, or regenerating runtime artifacts as part of documentation cleanup.
- Deleting legacy forensic code or changing the `paper/manuscript` submodule.
- Requiring a document or log for every development session.

## Capabilities

### New Capabilities

- `business-workspace-governance`: Defines business-workspace boundaries, minimal authority structure, lightweight handoff, evidence placement, and migration compatibility requirements.

### Modified Capabilities

None. This repository has no archived OpenSpec capability baseline yet.

## Impact

- Documentation paths under `docs/`, `paper/manager/`, and new `workspaces/` paths.
- Root navigation and agent instructions in `README.md` and `AGENTS.md`.
- New Markdown/link/governance validation tooling or tests.
- No intended change to `src/mmprism/`, `configs/`, `tests/`, runtime artifacts, or manuscript contents during this change.
