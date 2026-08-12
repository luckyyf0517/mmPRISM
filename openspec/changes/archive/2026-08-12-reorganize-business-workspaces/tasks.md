## 1. Baseline And Guardrails

- [x] 1.1 Resolve or isolate unrelated dirty-worktree changes, then record the migration baseline commit and the `paper/manuscript` submodule identity.
- [x] 1.2 Build a temporary migration inventory for tracked Markdown under `docs/` and `paper/manager/`, recording current role, target owner, target classification, stable IDs, inbound links, and immutable artifact references.
- [x] 1.3 Mark runtime artifacts, manuscript contents, legacy forensic code, and unrelated source changes as explicit no-touch paths for the migration.

## 2. Governance Validation

- [x] 2.1 Implement a dependency-light Markdown governance validator for Authority metadata, required entrypoints, index reachability, relative links, and compatibility-pointer targets.
- [x] 2.2 Add fixtures and tests covering broken links, missing metadata, unindexed Authority, invalid pointers, duplicate current-status entrypoints, and valid minimal workspaces.
- [x] 2.3 Document one canonical validation command that runs without optional ML or training dependencies.

## 3. Project Authority

- [x] 3.1 Create draft `docs/authority/00_INDEX.md` with project scope, current phase, cross-workspace blockers, and the shortest useful reading order.
- [x] 3.2 Create `10_SCOPE/WORKSPACE_MAP.md` defining the five business workspaces, shared-code boundary, ownership, inputs, outputs, and cross-workspace relationships.
- [x] 3.3 Create `20_CONTRACTS/SHARED_CONTRACTS.md` by extracting only cross-workspace manifest, split, run-evidence, and frozen-delivery rules from current sources.
- [x] 3.4 Migrate the decision register with every `DEC-*` identity and status preserved, and initialize the project changelog without copying routine progress history.

## 4. Minimal Workspace Setup

- [x] 4.1 Create meaningful minimal skeletons for `csl_news_annotation`, `data_rebuild`, `omnihand_training`, `wavellm_training`, and `paper_revision`, without unused optional directories.
- [x] 4.2 Populate each workspace index with boundary, current state, actionable blockers, next actions, root-owned code/config/script/test links, and reading order.
- [x] 4.3 Pilot the structure with `omnihand_training`; confirm that one ordinary status update needs no additional dashboard, task tracker, or handoff document before migrating the other workspaces.

## 5. Business Documentation Migration

- [x] 5.1 Migrate WaveLLM architecture, supported operations, and accepted validation evidence into `wavellm_training`, preserving artifact hashes and stable IDs.
- [x] 5.2 Migrate CSL-News source integrity, annotation operations, manifests, QC, and incident evidence into `csl_news_annotation`; retain only repeatable active runbooks and keep incident history immutable.
- [x] 5.3 Migrate data intake, radar provenance, split, quarantine, and task-specific delivery truth into `data_rebuild`, separating current rules from dated progress.
- [x] 5.4 Migrate reviewer sources, manuscript status, display-item registry, evidence promotion map, compliance, and response closure into `paper_revision` after producer workspace references are stable.
- [x] 5.5 Reclassify completed engineering smokes, formal-run reports, audits, and incidents as milestone Logs or immutable artifact references; do not create per-session Logs.

## 6. Navigation And Authority Cutover

- [x] 6.1 Update `AGENTS.md` and root navigation so new work starts from project Authority and the owning workspace index while retaining the existing canonical code and verification rules.
- [x] 6.2 Replace every migrated old authority path with a compatibility pointer and ensure no old page retains an independent current-status claim.
- [x] 6.3 In one coordinator-owned trunk change, set the new indexes to `current`, update review dates, and switch all canonical navigation links.

## 7. Acceptance And Stabilization

- [x] 7.1 Run the governance validator, Markdown relative-link audit, superseded-path and stale-status scans, `git diff --check`, and the narrow documentation/tool tests.
- [x] 7.2 Verify that stable IDs and tracked evidence artifact hashes match the baseline inventory and that no runtime artifact, manuscript content, shared source, or legacy forensic code changed unintentionally.
- [x] 7.3 Exercise one real routine update in each active workspace and one frozen cross-workspace delivery; remove demonstrated redundancy rather than adding new templates.
- [x] 7.4 Run `openspec validate --all --strict`, archive `reorganize-business-workspaces`, and confirm the resulting `business-workspace-governance` spec is absent from the active-change list.

## 8. Deferred Structural Changes

- [x] 8.1 Record, without implementing, any demonstrated need to split `data_rebuild` into separate radar and delivery workspaces as a future OpenSpec candidate.
- [x] 8.2 Require separate future OpenSpec changes for physical source/config/test movement, public API changes, or cross-workspace ownership changes discovered during migration.
