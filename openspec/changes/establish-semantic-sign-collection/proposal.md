## Why

All historical self-collected project recordings are non-semantic gestures. They may support pose reconstruction
or historical forensics, but they cannot validate continuous sign-language translation, semantic dataset coverage
or unseen-signer semantic generalization. The revision therefore needs a separately owned real CSL collection
targeting approximately 30 participants with usable recordings.

The author has confirmed Chinese Sign Language (CSL) as the primary signing language. The exact CSL
variety/register, written translation target and content inventory remain pre-pilot decisions.

## What Changes

- Add a `sign_language_collection` business workspace owning fixed CSL reference videos/text, lightweight
  participant coordination, synchronized acquisition, basic QC and frozen collection handoff.
- Explicitly classify legacy self-collected recordings as non-semantic and exclude them from the new cohort count
  and translation evidence.
- Seek 3--4 professional/proficient CSL contributors if available and use video-guided volunteers for scale, with
  participant type preserved in data and paper reporting.
- Define a common video-guided core and compact orientation/occlusion subset.
- Define minimal participant/session/take identities, raw modality/config bindings, basic QC and handoff to
  `data_rebuild`.

## Non-Goals

- Building a participant registration/screening/replacement funnel.
- Requiring professional CSL contributors before volunteer pilot recording can start.
- Inventing the CSL variety/register, reference-video inventory or release permissions.
- Rewriting, relabeling or deleting legacy raw data.
- Moving shared implementation into the workspace directory.
- Claiming video-guided volunteers are fluent/natural CSL signers.

## Impact

- One additional project business workspace and corresponding governance-audit coverage.
- New downstream frozen collection-manifest input for `data_rebuild`.
- Future compact acquisition config/manifest validation may remain under canonical root paths.
- Paper-facing real-data claims remain blocked until the final collection and downstream evidence chain validate.
