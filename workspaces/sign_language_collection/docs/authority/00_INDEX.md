# Semantic Sign-Language Collection Authority

Status: current
Owner: Semantic sign-language collection lane
Authority scope: Current decisions, readiness, execution boundary, and handoff state for the new real-data collection.
Last reviewed: 2026-08-12

## Boundary

This workspace owns the design and collection of a new, high-quality, semantically meaningful sign-language
dataset. It covers recruitment, consent, content freeze, synchronized radar/reference recording, sign-language
review, session QC and publication of a frozen collection manifest.

It does not process raw radar into model tensors, create canonical train/validation/test assignments, train
models, or promote paper claims. Those responsibilities remain with `data_rebuild`, training workspaces and
`paper_revision`. Shared schemas, validators and CLI code remain under `src/mmprism/`, `configs/` and `tests/`.

## Confirmed Decisions

- All previously self-collected project data consists of non-semantic gestures. It is legacy pose/reconstruction
  or forensic material, not semantic sign-language data.
- The new planning target is approximately 30 usable participants, substantially beyond the previously reported
  cohort of 12. Attempts that fail eligibility or session QC do not count as usable participants.
- The primary collection unit is a meaningful continuous sign-language utterance with a verified semantic
  target. Isolated or non-semantic gestures may be recorded for calibration but cannot constitute the primary
  translation dataset.
- Every usable participant completes the core semantic protocol. Expensive orientation and occlusion conditions
  use a frozen compact stress subset rather than repeating the full corpus in every condition.
- The new dataset is identity-disjoint by participant for downstream evaluation. Exact split assignments are
  produced later by `data_rebuild` from the frozen participant-aware manifest.

## Current State

Planning is active; recruitment and collection have not started. The target language/variant, semantic content
pack, signer eligibility rubric, ethics/consent scope, compensation, exact recording modalities, synchronization
tolerance, session duration, stress-subset size and downstream release scope remain to be frozen.

No production collection may start until readiness gates `G0` through `G4` pass. A small pilot precedes production,
and pilot data is excluded from the production dataset unless the protocol and hardware remain identical and an
explicit acceptance decision is recorded.

## Phase Status

| Phase | State | Exit condition |
|---|---|---|
| Protocol and ownership | in progress | scope, language, content, identity and handoff contracts frozen |
| Ethics and recruitment | blocked | approval/consent, compensation and signer eligibility accepted |
| Hardware and synchronization | blocked | radar/reference/calibration protocol and tolerances pass bench test |
| Pilot | blocked | complete small-cohort dry run passes semantic, signal, sync and operator review |
| Production collection | blocked | approximately 30 usable participants accepted under one frozen protocol |
| Freeze and handoff | blocked | immutable raw sessions, QC ledger and manifest validate and are accepted by data rebuild |

## Authority And Operations

- [Collection plan](10_SCOPE/COLLECTION_PLAN.md)
- [Acquisition and identity contract](20_CONTRACTS/ACQUISITION_CONTRACT.md)
- [Collection runbook](40_OPERATIONS/COLLECTION_RUNBOOK.md)
- [Readiness and QC gates](50_VALIDATION/READINESS_AND_QC.md)
- [Changelog](90_CHANGELOG.md)
- [Logs](../logs/README.md)

Ownership and data-contract changes are tracked by the active OpenSpec change
[`establish-semantic-sign-collection`](../../../../openspec/changes/establish-semantic-sign-collection/proposal.md).
