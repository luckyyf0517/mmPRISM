# Semantic Acquisition And Identity Contract

Status: draft
Owner: Semantic sign-language collection lane
Authority scope: Required identities, modalities, metadata, privacy boundaries, immutability, and frozen handoff for each new collection.
Last reviewed: 2026-08-12

## Identity Hierarchy

The acquisition contract uses stable opaque identifiers:

```text
dataset_id
protocol_id
participant_id
session_id
utterance_id
take_id
modality_artifact_id
```

`participant_id` is pseudonymous and stable within the dataset. Names, contact details, consent signatures and the
identity-to-participant map live in a separate restricted registry and never appear in the research manifest,
filenames, paper tables or model artifacts. Paths never encode names or infer modality relationships.

## Dataset Classes

Every record declares one class:

- `semantic_continuous`: eligible for the primary translation corpus;
- `semantic_isolated`: secondary isolated-sign material only;
- `calibration_nonsemantic`: hardware/pose calibration only;
- `legacy_nonsemantic`: historical self-collected gestures, stored outside the new dataset identity;
- `pilot`: pre-production evidence unless explicitly promoted;
- `invalid_attempt`: retained in the QC ledger but excluded from training/evaluation manifests.

Only `semantic_continuous` accepted takes count toward the primary corpus. No relabeling of legacy non-semantic
gestures can turn them into semantic sign-language evidence without a new verified semantic source.

## Required Per-Take Bindings

An accepted semantic take binds:

1. raw radar/ADC artifact and its SHA-256;
2. exact radar acquisition configuration, firmware/software identity and channel map;
3. synchronized reference artifact sufficient to verify signing and derive/validate pose;
4. synchronization evidence and measured offset/drift under the frozen tolerance;
5. `utterance_id`, canonical meaning, prompt version and semantic review status;
6. participant, session, environment, orientation and occlusion group keys;
7. calibration identity and sensor-placement metadata;
8. consent/release scope code without embedding the consent document;
9. operator identity by role, timestamps, retry lineage and acceptance/rejection state.

Derived pose, radar cubes, features and captions are not raw acquisition truth. They are created later in versioned
destinations and bind back to the immutable take and source checksums.

## Minimum Metadata

### Participant-level, pseudonymous

```text
participant_id
eligibility_rubric_version and result
target sign-language proficiency category
consented scientific attributes only
consent_scope_code
withdrawal/status state
```

### Session-level

```text
session_id, participant_id, protocol_id
date/time and site/environment code
radar/reference device and config IDs
calibration ID
operator roles
planned and completed content/condition counts
session-level QC state
```

### Take-level

```text
take_id, session_id, utterance_id, attempt_number
dataset_class
orientation and measured/nominal angle
occlusion type and severity rubric
distance/placement and environment code
start/end/sync markers
semantic correctness and reviewer state
signal/reference/synchronization QC states
artifact relative URIs, byte sizes, media/tensor metadata and SHA-256
accepted/rejected state and reason codes
```

Exact enumerations and numeric tolerances are frozen in versioned configuration before the pilot. Free-text notes
may supplement, but never replace, typed fields.

## Semantic Ground Truth

The primary signing language is Chinese Sign Language (CSL). Every content pack and semantic take declares CSL plus
the frozen variety/register code; unspecified generic `sign_language` labels are not accepted for production data.

The canonical semantic unit is an immutable `utterance_id` from the frozen content pack. An accepted take must not
derive its target text only from the prompt shown to the participant: a qualified reviewer confirms that the
performed utterance matches the intended meaning or records an allowed paraphrase/ambiguity.

For each utterance, preserve:

- sign language `CSL` and its frozen variety/register;
- canonical target-language translation;
- allowed reference translations when justified;
- lexical/content tags and length statistics;
- non-manual dependency state;
- review version and reviewer role.

Gloss annotations are included only if a consistent language-expert protocol exists. Missing gloss must not be
silently replaced by automatic labels.

## Acquisition And Synchronization

- Radar raw data, reference stream and event/sync markers are recorded without destructive preprocessing.
- All clocks, frame rates, trigger paths and expected durations are declared in the protocol.
- The session package stores observed frame/sample counts and timing diagnostics, not only configured values.
- A take with missing raw bytes, unexplained frame loss, failed checksum, unacceptable sync drift or absent semantic
  review cannot be accepted.
- Radar and reference streams use the same `take_id`; relationships are explicit manifest fields rather than path
  replacement.

The precise synchronization tolerance is intentionally unresolved until hardware characterization and downstream
frame-window requirements are measured. It must be frozen before production.

## Privacy, Consent, And Access

Reference video may be directly identifiable even when radar is not. Consent must separately describe collection,
internal processing, derived skeletons/features, reviewer access, public release, paper figures and retention.
Access policy distinguishes at least restricted direct identifiers, restricted identifiable raw reference media,
controlled raw radar/session data and release-eligible derived artifacts.

Withdrawal and deletion behavior follows the approved consent and ethics protocol. The technical manifest records
tombstone/status changes without exposing the participant identity and never rewrites historical evidence silently.

## Immutable Session Publication

An operator writes to a temporary session area. After local QC and backup verification, the data steward publishes
an immutable versioned session package containing raw artifacts, protocol/config references, metadata, QC report and
checksums. Existing accepted packages are never overwritten; a re-record is a new `take_id` linked to the prior
attempt.

## Handoff To Data Rebuild

The final collection snapshot contains:

```text
producer workspace and commit
dataset/protocol/content-pack IDs
immutable session roots
pseudonymous take manifest
participant/session/take coverage summary
QC and exclusion ledger
radar/reference/calibration/config bindings
consent-scope codes and release boundary
SHA256SUMS
validation status
```

`data_rebuild` independently verifies checksums, identity uniqueness, participant metadata coverage, required
modalities and QC states before accepting the snapshot. Acceptance does not permit changing raw data; subsequent
processing and subject-disjoint split assignments are new immutable products.
