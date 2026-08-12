## Context

The previous self-collected cohort contains non-semantic gestures, while the paper claims continuous sign-language
understanding and reviewers request clearer data characterization and real-world generalization. Collection is a
distinct business workflow: it deals with human recruitment, linguistic correctness, synchronized sensors and raw
session acceptance. `data_rebuild` should not own participant operations, and `paper_revision` should only consume
frozen evidence.

## Decisions

### Chinese Sign Language is the primary language

The new semantic collection primarily records Chinese Sign Language (CSL). CSL selection is fixed, while its exact
regional/register boundary, written translation target, content inventory and non-manual policy remain subject to
expert review. The protocol does not treat arbitrary gestures or manually encoded spoken-Chinese word sequences as
CSL evidence.

### Separate semantic collection ownership

`sign_language_collection` owns new acquisition through immutable session and collection manifests. It hands off
to `data_rebuild`, which owns processing, split assignment and model-ready delivery. The workspace is an ownership
view, not a Python package.

### Legacy evidence boundary

Historical self-collected data is typed `legacy_nonsemantic` and never contributes to semantic participant,
utterance or translation totals. It can remain available for pose/hardware forensics under its original provenance.

### Approximately 30 usable participants

Progress counts accepted participants with a valid core semantic session, not contacts, attempted sessions or raw
frame volume. Exact reported count is derived from the final validated manifest. Participant identity becomes the
minimum downstream split group.

### Core plus compact stress subset

All accepted participants perform a frozen continuous-semantic core. A smaller frozen matrix repeats selected
utterances for off-axis and occlusion conditions, including reviewer-cited `30 degree`/`60 degree` and partial hand
or object occlusion. This avoids repeating the entire corpus in every condition while preserving real-boundary
evidence.

### Semantic verification is independent of prompts

Every primary take binds a reviewed `utterance_id` and canonical meaning. A sign-language reviewer confirms the
performed meaning; the shown prompt alone is not ground truth. Non-manual dependencies are explicit so a hand-only
radar task cannot silently claim coverage it does not observe.

### Immutable, privacy-aware identities

Direct identifiers and consent records remain in a restricted registry. Research manifests use stable opaque
participant/session/take IDs and explicit artifact relations. Raw radar/reference streams, acquisition config,
calibration, synchronization evidence, QC state and checksums are published without overwrite.

## Risks

- Recruiting approximately 30 eligible signers may exceed the revision schedule; the protocol prioritizes quality
  and supports an early editor-extension decision rather than substituting non-signers or non-semantic gestures.
- Reference video is identifiable; access/release scopes must be explicit even if radar is described as private.
- Mid-collection protocol drift can create incomparable strata; every material change receives a new protocol ID.
- An oversized condition matrix can exhaust participants and operators; stress coverage stays compact and frozen.
- A sentence corpus can overstate linguistic scope; CSL variety/register, written target, task and non-manual
  boundaries are mandatory.

## Handoff

The collection publishes immutable raw session packages, a pseudonymous manifest, content/protocol/config and
calibration identities, QC/exclusion ledger, coverage summary and checksums. `data_rebuild` independently validates
that snapshot before constructing subject-disjoint splits or derived tensors.
