## Why

All historical self-collected project recordings are non-semantic gestures. They may support pose reconstruction
or historical forensics, but they cannot validate continuous sign-language translation, semantic dataset coverage
or unseen-signer semantic generalization. The revision therefore needs a separately owned, high-quality real
sign-language collection targeting approximately 30 usable participants.

The author has confirmed Chinese Sign Language (CSL) as the primary signing language. The exact CSL
variety/register, written translation target and content inventory remain pre-pilot decisions.

## What Changes

- Add a `sign_language_collection` business workspace owning a primarily CSL protocol, recruitment, consent, semantic content,
  synchronized acquisition, semantic/session QC and frozen collection handoff.
- Explicitly classify legacy self-collected recordings as non-semantic and exclude them from the new cohort count
  and translation evidence.
- Define a two-tier acquisition: a core meaningful continuous-sign protocol for every accepted participant and a
  compact orientation/occlusion stress subset.
- Define pseudonymous participant/session/take identities, raw modality/config/calibration bindings, semantic
  review, immutable session publication and handoff to `data_rebuild`.
- Add shared schema/config/validation implementation only after language, ethics, hardware and pilot decisions are
  frozen; shared code remains in root canonical paths.

## Non-Goals

- Treating documentation as ethics approval or starting recruitment before approval.
- Inventing the CSL variety/register, written translation target, sentence inventory, proficiency rubric or release permissions.
- Rewriting, relabeling or deleting legacy raw data.
- Moving shared implementation into the workspace directory.
- Claiming that the approximately 30-participant target has been achieved before accepted sessions validate.

## Impact

- One additional project business workspace and corresponding governance-audit coverage.
- New downstream frozen collection-manifest input for `data_rebuild`.
- Future versioned acquisition configs, schemas, validators and tests under `configs/`, `src/mmprism/` and `tests/`.
- Paper-facing real-data claims remain blocked until the final collection and downstream evidence chain validate.
