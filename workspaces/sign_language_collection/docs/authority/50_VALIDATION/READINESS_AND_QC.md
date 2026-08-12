# Collection Readiness And QC Gates

Status: current
Owner: Semantic sign-language collection lane
Authority scope: Go/no-go gates, acceptance states, coverage checks, and evidence required before pilot, production, and handoff.
Last reviewed: 2026-08-12

## Current Readiness

| Gate | Requirement | State | Blocking evidence |
|---|---|---|---|
| `G0` scope | legacy exclusion, approximately 30-participant target, core/stress boundary and ownership accepted | passed | `DEC-040` and workspace Authority |
| `G1` language/content | target language/variant, task definition, reviewed content pack and non-manual policy frozen | blocked | approved content-pack version |
| `G2` ethics/recruitment | ethics/consent, compensation, access/release scope, signer eligibility and withdrawal policy accepted | blocked | approved documents and rubric |
| `G3` acquisition | radar/reference modalities, config, placement, calibration, sync method/tolerance and storage plan frozen | blocked | bench report and protocol bundle |
| `G4` operator dry run | complete no-participant/authorized dry run with immutable package and automated QC | blocked | dry-run validation report |
| `G5` pilot | small eligible-signer pilot passes burden, semantics, signal, sync, backup and annotation review | blocked | dated pilot report and go decision |
| `G6` production | approximately 30 usable core participants plus frozen stress coverage accepted | blocked | live manifest-derived coverage report |
| `G7` handoff | immutable snapshot independently validates and data rebuild accepts it | blocked | snapshot hash and acceptance receipt |

Only `G0` is currently passed. This table is the current readiness truth; task completion elsewhere cannot promote a
gate without the named evidence.

## Take States

Every attempt has exactly one current processing state in the QC ledger:

```text
recorded -> technical_review -> semantic_review -> accepted
                                             \-> rejected
                                             \-> rerecord_requested
```

Acceptance requires all applicable checks below. Rejection retains the attempt and reason. A re-record is a new
take, not a replacement file.

## Technical QC

- Raw radar and reference artifacts exist, are readable, non-empty and checksum-valid.
- Device/config/channel-map and calibration identities match the frozen protocol.
- Observed radar/reference counts and durations are within frozen tolerances.
- Synchronization offset/drift passes the frozen hardware-derived threshold.
- Participant and hands are visible enough in the approved reference modality for the intended pose/semantic QC.
- Orientation, occlusion, distance, environment and retry metadata are present when required.
- No direct identifier appears in research paths, manifests or preview artifacts.

## Semantic QC

- `utterance_id` exists in the exact frozen content pack.
- A qualified reviewer confirms intended meaning or an explicitly allowed paraphrase.
- Completeness, ordering and material signing errors follow a frozen rubric.
- Non-manual dependency is recorded and handled by the task policy.
- Prompt text alone is not treated as proof of what the participant performed.
- Ambiguous or disputed takes remain deferred/rejected until resolved; they are not silently accepted to improve
  coverage.

## Session And Participant Acceptance

A session is accepted only when required calibration/preflight evidence exists, every planned take has a visible
outcome, missing/rejected material is quantified, backup/checksum publication passed and deviations are resolved.

A participant counts toward the approximately 30-person target only after at least one core semantic session is
accepted. Recruitment contacts, consented-but-unrecorded people, pilot-only participants and failed sessions are not
counted. Final reporting distinguishes attempted, consented, recorded and accepted counts.

## Coverage Report

Coverage is generated from the immutable accepted/rejected ledgers and includes:

- attempted, accepted and excluded participant/session/take counts;
- content coverage and accepted repetitions by utterance;
- sentence length, vocabulary and semantic-category statistics;
- non-manual dependency coverage;
- participant-level diversity fields only where consented and scientifically justified;
- nominal/orientation/occlusion/environment condition coverage;
- missingness, re-record and rejection reasons;
- modality, synchronization and calibration failure rates;
- protocol version strata.

No frame count substitutes for participant or utterance coverage. Legacy non-semantic gestures appear only in a
separate historical inventory and never in these semantic totals.

## Pilot Go/No-Go Review

The pilot review answers:

1. Are prompts linguistically correct and performed naturally by eligible signers?
2. Is the primary task genuinely continuous semantic signing rather than isolated gesture concatenation?
3. Can operators complete the session within an acceptable burden while preserving breaks?
4. Are radar/reference signals, pose visibility, sync and backup consistently valid?
5. Are semantic and technical rejection reasons reproducible between reviewers/operators?
6. Is the core/stress workload feasible for approximately 30 usable participants?

Any material content, sensor, synchronization or consent change after the pilot creates a new protocol version.
Pilot data may join production only through an explicit decision proving it is identical under all paper-facing
conditions.

## Handoff Validation

Before `G7`, validate exact participant/session/take identity uniqueness, pseudonymization, content-pack coverage,
required modality bindings, checksums, protocol/config/calibration copies, QC/exclusion states and consent-scope
codes. The handoff must state the actual accepted cohort and limitations; the approximately 30-person target cannot
be reported as achieved before validation.
