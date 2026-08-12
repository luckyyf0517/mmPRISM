# Minimal CSL Acquisition Contract

Status: draft
Owner: Semantic sign-language collection lane
Authority scope: Minimum identities, files, metadata, privacy boundary, and handoff required for the revision dataset.
Last reviewed: 2026-08-12

## Identities

Use opaque `participant_id`, `session_id`, `utterance_id` and `take_id`. Names and contact details stay outside the
research data. Each participant declares exactly one collection type:

- `professional_or_proficient_signer`;
- `video_guided_volunteer`.

This field must survive into the final manifest and paper statistics.

## Accepted Take

An accepted take needs only:

1. raw radar file and SHA-256;
2. synchronized reference video and SHA-256;
3. participant/session/utterance/take IDs;
4. participant type;
5. frozen reference-video/content version and Chinese text meaning;
6. condition: frontal, orientation angle, hand occlusion or object occlusion;
7. radar configuration/calibration identity;
8. basic synchronization result and `accepted`/`rejected` status.

The raw files are not overwritten. A retry receives a new `take_id`. Relationships are stored in the manifest,
never inferred by replacing path strings.

## Basic QC

Accept a take when radar and reference files are readable, the reference recording shows a complete reproduction,
the streams are synchronized well enough for the downstream window, and the IDs/condition are correct. Otherwise
record a short reason and repeat if practical.

For volunteer data, the target meaning comes from the frozen reference clip and text. QC checks faithful visible
reproduction; it does not certify that the volunteer is a CSL signer. When a professional/proficient reviewer is
available, record any content/take check separately, but this is not required for every volunteer take.

## Minimal Consent And Privacy

Before recording, participants must agree to radar and identifiable reference-video collection and the intended
research use. Use the project's approved minimal form/process. Direct identifiers are not placed in research
filenames or manifests. Public release of identifiable reference video is a separate decision and is not assumed.

## Frozen Handoff

The collection hands `data_rebuild`:

```text
producer commit
reference content list/version
participant/session/take manifest
raw radar and reference-video files
radar configuration/calibration reference
accepted/rejected take status
SHA256SUMS
```

`data_rebuild` verifies checksums, IDs, participant-type coverage, required files and participant-disjoint grouping
before processing or splitting. No additional handoff report is needed when these fields are in the snapshot.
