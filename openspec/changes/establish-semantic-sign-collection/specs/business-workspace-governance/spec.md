## MODIFIED Requirements

### Requirement: Workspaces represent executable business workflows

The repository SHALL use workspaces for business workflows that have distinct inputs, operations, outputs, and
acceptance criteria. The workspace set SHALL be:

- `csl_news_annotation`
- `data_rebuild`
- `sign_language_collection`
- `omnihand_training`
- `wavellm_training`
- `paper_revision`

The repository SHALL NOT create workspaces solely for shared package layers such as contracts, configuration,
runtime, artifacts, assets, evaluation, release, or CLI composition. `sign_language_collection` SHALL own new
participant recruitment, semantic content, synchronized acquisition and session acceptance; `data_rebuild` SHALL
own processing, split and model-ready delivery after a frozen collection handoff.

#### Scenario: Place a model training activity

- **WHEN** an activity trains or evaluates the OmniHand reconstruction model
- **THEN** its business status and operations belong to `omnihand_training`
- **AND** reusable implementation remains in the canonical root package.

#### Scenario: Place a new human-data acquisition activity

- **WHEN** an activity recruits signers or records and accepts new semantic radar/reference sessions
- **THEN** its business status, protocol and operations belong to `sign_language_collection`
- **AND** reusable schemas and validators remain in the canonical root package.

#### Scenario: Process an accepted collection

- **WHEN** a frozen semantic collection is transformed into radar cubes, participant-disjoint splits or model-ready
  products
- **THEN** the activity belongs to `data_rebuild`
- **AND** the collection's immutable source identity remains bound to every derived product.

#### Scenario: Place a shared runtime change

- **WHEN** a change affects provenance, artifact writing, configuration or evaluation shared by multiple workflows
- **THEN** it remains project-owned rather than creating a new workspace
- **AND** an OpenSpec change is used when its interface or ownership contract changes.
