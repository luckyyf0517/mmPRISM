# Semantic Sign-Language Collection Authority

Status: current
Owner: Semantic sign-language collection lane
Authority scope: Current decisions, readiness, execution boundary, and handoff state for the new real-data collection.
Last reviewed: 2026-08-12

## Boundary

This workspace owns the new paper-revision CSL collection. It covers reference-video/text freeze, lightweight
participant coordination, consent, synchronized radar/reference recording, basic take QC and a frozen handoff.

It does not process raw radar into model tensors, create canonical train/validation/test assignments, train
models, or promote paper claims. Those responsibilities remain with `data_rebuild`, training workspaces and
`paper_revision`. Shared schemas, validators and CLI code remain under `src/mmprism/`, `configs/` and `tests/`.

## Confirmed Decisions

- All previously self-collected project data consists of non-semantic gestures. It is legacy pose/reconstruction
  or forensic material, not semantic sign-language data.
- The target is approximately 30 participants with technically usable recordings, substantially beyond the
  previously reported cohort of 12. There is no separate enrollment-funnel accounting.
- The primary signing language is Chinese Sign Language (CSL). The precise CSL regional/register boundary,
  written translation target and reference-video set still require a simple content check before the pilot.
- Seek 3--4 professional/proficient CSL contributors for demonstration, content checking and a small reference
  subset if they can be found. Their availability does not block volunteer collection.
- Most scale comes from volunteers who watch the same frozen CSL video and reproduce it. Their records are typed
  `video_guided_volunteer`, not represented as natural signing by fluent CSL users.
- Every recorded participant completes the core video-guided set. Expensive orientation and occlusion conditions
  use a frozen compact stress subset rather than repeating the full corpus in every condition.
- The new dataset is identity-disjoint by participant for downstream evaluation. Exact split assignments are
  produced later by `data_rebuild` from the frozen participant-aware manifest.

## Current State

Planning is active; collection has not started. CSL is selected. Remaining practical blockers are the fixed
reference-video/text list, a minimal consent form, radar/reference setup and synchronization, session length, a
small pilot and the compact stress subset. Professional/proficient CSL support remains best-effort.

Production follows one small pilot. Pilot data may be retained when the same reference set and hardware setup are
used and its takes pass the same QC.

## Phase Status

| Phase | State | Exit condition |
|---|---|---|
| Reference set | in progress | CSL videos, text meanings and core/stress lists frozen |
| Minimal collection setup | blocked | consent, radar/reference setup, IDs and storage confirmed |
| Pilot | blocked | a few video-guided takes pass playback, signal, sync and file checks |
| Production collection | blocked | approximately 30 participants have usable core recordings |
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
