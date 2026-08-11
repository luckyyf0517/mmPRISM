# mmPRISM Agent Guide

## Canonical Sources

- New implementation: `src/mmprism/`
- Versioned configuration: `configs/`
- Automated verification: `tests/`
- Architecture documentation: `docs/architecture/`
- Revision control plane: `paper/manager/`
- Manuscript submodule: `paper/manuscript/`

## Legacy Boundary

The root `run_*.py` files, `config/`, and the pre-existing modules under `src/data`, `src/fmcw`, `src/model`, `src/eval`, `src/scripts`, and `src/utils` are read-only historical references during the rebuild.

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

## Agent Handoff

Before changing architecture, data definitions, or paper-facing evidence, read:

1. `paper/manager/dashboard.md`
2. `paper/manager/current/core_rules.md`
3. `paper/manager/current/issues.md`
4. `paper/manager/tasks/todo.md`
5. the relevant runbook and registry

Update the documents required by `paper/manager/sync_map.md` before ending the task.
