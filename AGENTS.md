# mmPRISM Agent Guide

## Canonical Sources

- New implementation: `src/mmprism/`
- Versioned configuration: `configs/`
- Automated verification: `tests/`
- Project authority: `docs/authority/00_INDEX.md`
- Business workspaces: `workspaces/*/docs/authority/00_INDEX.md`
- OpenSpec changes: `openspec/changes/`
- Manuscript submodule: `paper/manuscript/`

## Legacy Boundary

The original-submission codebase is preserved read-only under `legacy/` (entry points, `legacy/config/`,
`legacy/src/{data,fmcw,model,eval,scripts,utils}`, `legacy/scripts/{omnihand,wavellm}`). It was relocated from
the repository root, `config/`, `src/`, and `scripts/` on 2026-08-12; see `legacy/README.md`.

- Do not add new features to legacy modules.
- Do not import legacy modules from `src/mmprism/`.
- Do not create compatibility shims unless a documented evidence-recovery task requires one.
- Preserve historical code until the original manuscript evidence audit is complete.

## Engineering Rules

1. Use the UV-managed Python 3.12 environment and absolute `mmprism.*` imports.
2. Keep models free of path resolution, logging, checkpoint writing, and CLI parsing.
3. Express data relationships through validated manifests, never path string replacement.
4. Inject data roots, artifact roots, devices, precision, and model locations through configuration.
5. Make CLI commands thin orchestration layers over testable package functions.
6. Save resolved configuration, Git state, data manifest hash, environment, seed, and metrics for every formal run.
7. Treat raw data as immutable; write rebuilt data to versioned destinations.
8. Never promote a metric or claim to the manuscript without an evidence-registry entry.

## Verification Order

Run the narrowest applicable checks first:

```text
unit -> contract -> integration -> GPU smoke -> paper evidence audit
```

CPU-only contract tests must not import optional training dependencies such as PyTorch, Lightning, or Transformers.

Canonical commands run through UV:

```bash
scripts/bootstrap_env.sh research
uv run ruff check src/mmprism tests
uv run mypy
uv run pytest
```

## Workspace And Handoff

Workspaces represent executable business workflows, not Python package boundaries. Shared code remains in
`src/mmprism/`, shared configuration in `configs/`, and verification in `tests/`.

Before changing architecture, data definitions, or paper-facing evidence, read:

1. `docs/authority/00_INDEX.md`
2. the owning workspace `docs/authority/00_INDEX.md`
3. only the scoped contract, runbook, or dated evidence needed for the task

Routine handoff uses task, state, result, evidence, next action, and blocker; no standalone handoff document
is required. Cross-workspace data or paper-evidence delivery additionally requires producer commit,
immutable location, manifest/inventory hash, and validation status, preferably in an existing artifact.

Use OpenSpec before changing code, interfaces, data contracts, or ownership. Routine progress updates only
the owning workspace index when its actionable state changes.

Run `uv run python scripts/audit_docs.py` before completing documentation or Authority changes.
