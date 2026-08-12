# business-workspace-governance Specification

## Purpose
Define a lightweight, testable documentation and ownership model for mmPRISM business workflows while preserving shared canonical code and immutable research evidence.
## Requirements
### Requirement: Workspaces represent executable business workflows
The repository SHALL use workspaces for business workflows that have distinct inputs, operations, outputs, and acceptance criteria. The initial workspace set SHALL be:

- `csl_news_annotation`
- `data_rebuild`
- `omnihand_training`
- `wavellm_training`
- `paper_revision`

The repository SHALL NOT create workspaces solely for shared package layers such as contracts, configuration, runtime, artifacts, assets, evaluation, release, or CLI composition.

#### Scenario: Place a model training activity
- **WHEN** an activity trains or evaluates the OmniHand reconstruction model
- **THEN** its business status and operations belong to `omnihand_training`
- **AND** reusable implementation remains in the canonical root package

#### Scenario: Place a shared runtime change
- **WHEN** a change affects provenance, artifact writing, configuration, or evaluation shared by multiple workflows
- **THEN** it remains project-owned rather than creating a new workspace
- **AND** an OpenSpec change is used when its interface or ownership contract changes

### Requirement: Shared canonical implementation remains at repository root
This documentation reorganization SHALL NOT move or duplicate canonical implementation from `src/mmprism/`, versioned project configuration from `configs/`, or automated verification from `tests/` merely to mirror workspace names. Workspace indexes SHALL link to the root-owned code, configuration, scripts, and tests they use.

#### Scenario: Establish workspace ownership without moving code
- **WHEN** the `wavellm_training` workspace is created
- **THEN** its index links to the relevant `src/mmprism/models/`, `src/mmprism/training/`, `configs/`, `scripts/`, and `tests/` paths
- **AND** no duplicate workspace-local implementation is required

### Requirement: Project authority is minimal and cross-workspace
Project authority SHALL live under `docs/authority/` and cover only project scope, workspace boundaries, shared contracts, cross-workspace decisions, and project-level current blockers. `docs/authority/00_INDEX.md` SHALL be the project authority entrypoint.

#### Scenario: Record a shared manifest rule
- **WHEN** a manifest requirement applies to more than one business workflow
- **THEN** it is owned by project authority or its linked canonical code contract
- **AND** workspace pages reference it without copying an independently maintained version

### Requirement: Each workspace has one current-status entrypoint
Every workspace SHALL contain `README.md`, `docs/authority/00_INDEX.md`, and `docs/authority/90_CHANGELOG.md`. The index SHALL contain `Status`, `Owner`, `Authority scope`, and `Last reviewed` metadata and SHALL be the only page that summarizes that workspace's current phase, blockers, and next actions.

Optional authority sections, operations, validation pages, logs, source, configuration, scripts, and tests SHALL be created only when they contain real workspace-owned content.

#### Scenario: Create a workspace skeleton
- **WHEN** a business workspace is introduced
- **THEN** its README, authority index, and changelog are created with meaningful content
- **AND** no empty optional directories or placeholder pages are created

#### Scenario: Update routine progress
- **WHEN** routine progress changes the current next action without changing a contract or producing durable evidence
- **THEN** only the owning workspace index needs a concise update
- **AND** no separate dashboard, status page, roadmap, or session log is required

### Requirement: Authority, runbooks, logs, and OpenSpec have non-overlapping roles
Current truth SHALL be maintained in Authority. Reusable named operations SHALL be maintained as Runbooks while active. Dated execution, incident, audit, and milestone evidence SHALL be immutable Logs. Code, interface, data-contract, or ownership changes SHALL be planned through OpenSpec.

A completed one-off run SHALL NOT remain a current Authority page. A new Log SHALL NOT supersede Authority unless an explicit promotion or decision updates the owning Authority.

#### Scenario: Complete a one-off experiment
- **WHEN** a one-off experiment finishes without changing an interface or contract
- **THEN** its durable result is recorded as a dated Log or referenced immutable artifact
- **AND** no OpenSpec change is created solely for that run

#### Scenario: Change a data contract
- **WHEN** a proposed implementation changes a manifest schema or cross-workspace handoff contract
- **THEN** an OpenSpec change is required before implementation
- **AND** the accepted contract is promoted to the appropriate Authority after implementation

### Requirement: Routine handoff is concise
A routine same-workspace handoff SHALL require no standalone handoff document. When a handoff is needed, the minimum message SHALL be limited to task ID, commit or working-tree state, result, evidence reference when applicable, next action, and blocker.

#### Scenario: Hand off an ordinary implementation task
- **WHEN** an agent finishes a task that does not transfer a dataset or paper-facing evidence across workspaces
- **THEN** the agent provides the concise handoff fields
- **AND** no manifest, report, or new Authority page is required only for handoff formality

### Requirement: Cross-workspace delivery uses frozen identity
A data or paper-evidence delivery consumed by another workspace SHALL identify the producer workspace, producer commit, immutable location, relevant manifest or inventory hash, and validation status. These fields MAY be carried by an existing manifest summary, artifact receipt, or dated Log; a separate handoff report SHALL NOT be required when that information already exists.

#### Scenario: Deliver training data to OmniHand
- **WHEN** `data_rebuild` hands model-ready data to `omnihand_training`
- **THEN** the consumer receives an immutable location, producer commit, manifest and split identities, and validation status
- **AND** the same identity is referenced rather than recopied into a second report

### Requirement: Migration preserves identity and compatibility
The reorganization SHALL preserve stable task, decision, evidence, dataset, split, run, and display-item IDs. Existing immutable artifacts SHALL not be rewritten. Every moved or renamed document that may have inbound links SHALL leave a compatibility pointer at its old path with no independent authority claims.

#### Scenario: Move an existing authority document
- **WHEN** current content is promoted from `paper/manager/current/` to a workspace authority page
- **THEN** the old path becomes a compatibility pointer to the canonical path
- **AND** it does not retain a competing status summary

#### Scenario: Preserve run evidence
- **WHEN** an existing engineering smoke document is reclassified as a Log
- **THEN** its stable IDs and artifact hashes remain unchanged
- **AND** runtime artifact bytes are not modified

### Requirement: Governance is mechanically verifiable
The repository SHALL provide a non-networked validation command that checks Authority metadata, required workspace entrypoints, index reachability for scoped Authority pages, Markdown relative-link resolution, compatibility-pointer targets, and prohibited duplicate current-status entrypoints. Migration completion SHALL also require `git diff --check` and a scan for superseded paths or terminology in current Authority.

#### Scenario: Validate a completed migration
- **WHEN** the coordinator runs the documented governance validation
- **THEN** missing metadata, broken links, unindexed Authority pages, invalid compatibility targets, and duplicate current-status pages cause a non-zero result
- **AND** immutable runtime artifacts are outside the validator's rewrite scope

### Requirement: Coordinator owns cross-workspace authority cutover
Only the coordinator SHALL edit project-level Authority and perform the final migration cutover on trunk. A lane MAY edit its own workspace and reference project Authority, but SHALL NOT write another lane's workspace.

#### Scenario: Integrate a lane result
- **WHEN** a lane's workspace documentation and evidence pass acceptance
- **THEN** the coordinator merges it to trunk and updates project Authority only when the project-level truth changed
- **AND** no unrelated workspace documentation is rewritten
