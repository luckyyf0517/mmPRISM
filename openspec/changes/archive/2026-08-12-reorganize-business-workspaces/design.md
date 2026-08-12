## Context

See `proposal.md` for motivation. The current control plane is mature but structurally overloaded:

- `paper/manager/` contains about 6,000 lines spanning current state, tasks, runbooks, decisions, reviews, and evidence.
- `docs/architecture/` contains both shared contracts and business-specific architecture.
- `dashboard.md`, `current/*`, `tasks/*`, and registries repeat portions of the same current status.
- Existing authority-like pages use `Status`/`Last Updated`/`Role`; none currently carries the required `Owner`/`Authority scope`/`Last reviewed` metadata.
- The repository has active uncommitted training work. Migration must not absorb, revert, or reinterpret it.
- Stable IDs and artifact hashes already form a useful evidence graph and must survive path changes.

## Goals / Non-Goals

**Goals:**

- Make the current state of any business workflow reachable from one short index.
- Separate current truth from dated execution evidence without increasing routine update count.
- Define business ownership without forcing package duplication or code movement.
- Migrate incrementally with a reversible, coordinator-controlled cutover.

**Non-Goals:**

- Designing a monorepo package-per-workspace architecture.
- Replacing Git commits, manifests, run receipts, or formal artifacts with prose handoff documents.
- Reformatting every historical evidence document for visual consistency.
- Creating new active task trackers after the current task tables are retired.

## Decisions

### 1. Use five business workspaces

| Workspace | Owns | Does not own |
|---|---|---|
| `csl_news_annotation` | official-source intake gates, source integrity, RTMW3D annotation, QC, pose/source manifests | generic data delivery, model training, shared manifest implementation |
| `data_rebuild` | private/public data intake, radar processing and simulation recovery, split, quarantine, model-ready delivery | OmniHand/WaveLLM training behavior, CSL-News worker operations |
| `omnihand_training` | pose-reconstruction model train/resume/evaluate lifecycle | shared artifact framework, upstream data production |
| `wavellm_training` | sign-language generation train/resume/evaluate lifecycle | shared model acquisition and generic evaluation infrastructure |
| `paper_revision` | reviewer requirements, evidence promotion, manuscript audit/writeback, response closure | production of training/data evidence owned by other workspaces |

`data_rebuild` intentionally includes radar and materialization initially. These responsibilities share source provenance, calibration, split, and delivery gates; splitting them now would create an extra handoff before either flow is independently operational. A future split requires its own OpenSpec change.

Alternatives considered:

- **Workspace per package directory:** rejected because `contracts`, `runtime`, `artifacts`, and `evaluation` are shared implementation layers rather than business outcomes.
- **Separate radar and dataset-delivery workspaces now:** deferred until they have independent owners and stable frozen handoffs.
- **Documentation-governance workspace:** rejected because project governance is coordinator-owned project Authority.

### 2. Keep source roots canonical and assign logical business ownership

No source, config, script, or test is moved during this change. Workspace indexes link to these current locations:

| Workspace | Primary canonical paths |
|---|---|
| `csl_news_annotation` | `src/mmprism/data/csl_news*.py`, CSL-News configs and scripts, matching unit tests |
| `data_rebuild` | `src/mmprism/data/{split,pose_reconstruction,sign_language_translation}.py`, `src/mmprism/radar/`, data configs/scripts/tests |
| `omnihand_training` | `src/mmprism/models/cubenet.py`, `src/mmprism/training/omnihand_*`, pose evaluation and matching tests |
| `wavellm_training` | `src/mmprism/models/{stgcn,translation}.py`, `src/mmprism/training/{mt5,wavellm}_*`, language evaluation and matching tests |
| `paper_revision` | `paper/manuscript/`, manuscript audit tooling/tests, paper evidence maps |

Shared paths such as `contracts/`, `config/`, `runtime/`, `artifacts/`, `assets/`, `evaluation/`, `release/`, and `cli.py` remain project-owned even when a workspace consumes them. Where an evaluation module is domain-specific, the workspace owns requirements and validation while the reusable evaluation API remains shared code.

### 3. Use a deliberately small documentation surface

Project-level initial surface:

```text
docs/authority/
  00_INDEX.md
  10_SCOPE/WORKSPACE_MAP.md
  20_CONTRACTS/SHARED_CONTRACTS.md
  60_DECISIONS/DECISION_LOG.md
  90_CHANGELOG.md
```

Workspace minimum:

```text
workspaces/<name>/
  README.md
  docs/authority/00_INDEX.md
  docs/authority/90_CHANGELOG.md
```

`20_CONTRACTS`, `40_OPERATIONS`, `50_VALIDATION`, and `docs/logs` are added only when migrated content requires them. There is no permanent workspace dashboard, roadmap, todo collection, or per-session handoff ledger. The index contains current phase, at most the active blockers needed to act, next actions, canonical locations, and reading order. Commit chronology is excluded.

### 4. Classify before moving

Every existing document receives one target classification in a temporary migration inventory:

| Classification | Test |
|---|---|
| Project Authority | Defines truth across two or more workspaces |
| Workspace Authority | Defines current truth owned by exactly one business workflow |
| Runbook | Reusable instructions for a named operation that is still supported |
| Log | Dated run, incident, audit, imported source, or completed milestone evidence |
| Compatibility pointer | Old location retained solely for inbound links |

This inventory is an implementation aid and need not become a permanently maintained registry after cutover.

Initial document routing:

| Current material | Target owner |
|---|---|
| `current/core_rules.md`, shared parts of overview/architecture, `sync_map.md` | project Authority or `AGENTS.md` |
| `current/issues.md`, `dashboard.md`, master todo | concise project index plus owning workspace indexes; chronology to Logs |
| `decisions/decision_log.md` | project decision Authority, preserving every `DEC-*` ID |
| shared architecture contracts: tensors, run artifacts, splits, assets, release | project Authority contracts/architecture |
| CSL-News architecture, runbook, integrity, manifests, pose comparison | `csl_news_annotation` Authority/Operations/Logs |
| general data status, upload intake, Parquet delivery, radar audit | `data_rebuild` Authority/Operations/Logs |
| OmniHand architecture and smoke/formal evidence | `omnihand_training` Authority/Logs |
| WaveLLM architecture and smoke/formal evidence | `wavellm_training` Authority/Logs |
| reviews, manuscript status/audit, display registry, evidence map, compliance | `paper_revision` Authority/Logs |
| generic architecture-refactor instructions | merge irreducible rules into `AGENTS.md`; historical plan becomes Log |

Existing evidence documents may be moved with `git mv`; their scientific content is not rewritten except for metadata, relative links, and an explicit classification header where necessary.

### 5. Separate routine handoff from frozen delivery

Ordinary handoff remains conversational and concise:

```text
Task: <stable ID>
State: <commit or dirty-worktree note>
Result: <one sentence>
Evidence: <existing path/artifact, or n/a>
Next: <one action>
Blocker: <one item, or none>
```

Cross-workspace transfer uses existing machine-readable identities wherever possible. A manifest summary, run receipt, or immutable Log is sufficient when it contains producer, commit, immutable location, hash, and validation status. No parallel handoff form is introduced.

### 6. Cut over incrementally but switch authority atomically

Scaffolding and content preparation are additive. Until cutover, `paper/manager/` remains the declared authority. During cutover, one coordinator commit:

1. promotes project and workspace indexes to current;
2. updates `AGENTS.md`, root `README.md`, and navigation links;
3. changes every migrated old authority path to a compatibility pointer;
4. marks migrated historical material as Log/historical;
5. runs governance and relative-link validation.

This avoids a period where old and new current pages both claim authority. Workspace migration can be reviewed in batches before the final switch, but those pages remain `draft` until cutover.

### 7. Add narrow governance validation

The checker should use standard-library Markdown link parsing or the repository's existing test conventions and must not import optional ML dependencies. It validates structure and links only; it does not rewrite Markdown or inspect external HTTP targets.

Required checks:

- mandatory metadata on Authority pages;
- required project/workspace entrypoints;
- every scoped Authority page linked from its owning index;
- relative Markdown links resolve, excluding generated/runtime/vendor areas;
- compatibility pointers match the prescribed form and target an existing file;
- deprecated current-authority markers do not survive under old paths;
- no optional empty directories or placeholder-only pages are introduced.

## Risks / Trade-offs

- **Large one-time link churn** -> Build a migration inventory and run the link checker before and after each batch; leave compatibility pointers.
- **Loss of historical nuance while shortening indexes** -> Move chronology to dated Logs and preserve stable IDs/hashes rather than deleting it.
- **Old and new pages compete during migration** -> Keep new indexes `draft` and use one coordinator cutover commit.
- **Business ownership overlaps shared code** -> Record logical ownership in workspace indexes while retaining project ownership for shared APIs.
- **Paper evidence needs stronger indexing than other workspaces** -> Permit scoped registries in `paper_revision` because they support claim-to-artifact audit; do not generalize them to every workspace.
- **Migration collides with active code work** -> Restrict edits to documentation, OpenSpec, and dependency-light validation; never stage unrelated dirty files.

## Migration Plan

### Phase 0: Baseline and inventory

- Freeze a tracked-document inventory with source path, type, owner, target, stable IDs, and inbound-link count.
- Record the current Git state without requiring unrelated work to be clean.
- Identify immutable evidence artifacts and exclude them from rewrite operations.

### Phase 1: Project authority and pilot

- Create project Authority as `draft`.
- Create all five minimal workspace skeletons.
- Pilot the pattern with `omnihand_training`, which has a bounded architecture/evidence set.
- Review information density and remove any field that does not support an actual decision or operation.

### Phase 2: Business migration

- Migrate `wavellm_training`.
- Migrate `csl_news_annotation` while preserving all source-identity incident evidence.
- Migrate `data_rebuild`, separating current data/radar/delivery truth from historical progress.
- Migrate `paper_revision` last because it consumes evidence from all other workspaces.

### Phase 3: Evidence and operation normalization

- Retain only reusable active Runbooks.
- Group completed evidence by meaningful milestone or incident; do not create a Log per session or commit.
- Split the central experiment registry into workspace evidence references; keep the paper evidence map as the promotion index for manuscript-facing claims.

### Phase 4: Authority cutover

- Update root navigation and agent handoff rules.
- Replace migrated old locations with compatibility pointers.
- Set new Authority status to `current` in the same commit.
- Run structural, link, terminology, whitespace, and existing documentation-related tests.

### Phase 5: Stabilization and archive

- Use the new structure for at least one real update in each active workspace.
- Fix only demonstrated friction; do not add speculative templates.
- Archive this OpenSpec change after all acceptance gates pass.

### Rollback

Before cutover, delete only the new draft paths if the design is rejected. After cutover, revert the coordinator cutover commit; because immutable artifacts and legacy code are untouched and old paths are retained as files, rollback does not require data restoration.

## Open Questions

- Whether `data_rebuild` should later split into radar and delivery workspaces depends on independent ownership and stable handoff artifacts; this does not block the current migration.
- Whether Markdown governance validation lives under `scripts/` or as a dependency-light package tool can be chosen during implementation without changing the behavior contract.
