## Context

The previous self-collected cohort contains non-semantic gestures, while the paper claims continuous sign-language
understanding and reviewers request clearer data characterization and real-world generalization. Collection is a
distinct business workflow: it deals with reference content, participants, synchronized sensors and raw session
acceptance. `data_rebuild` should not own recording operations, and `paper_revision` should only consume frozen
evidence.

## Decisions

### Chinese Sign Language is the primary language

The new collection uses fixed Chinese Sign Language (CSL) reference videos and their Chinese meanings. The precise
video set remains to be selected. The protocol does not relabel arbitrary legacy gestures as CSL.

### Separate semantic collection ownership

`sign_language_collection` owns new acquisition through immutable session and collection manifests. It hands off
to `data_rebuild`, which owns processing, split assignment and model-ready delivery. The workspace is an ownership
view, not a Python package.

### Legacy evidence boundary

Historical self-collected data is typed `legacy_nonsemantic` and never contributes to semantic participant,
utterance or translation totals. It can remain available for pose/hardware forensics under its original provenance.

### Mixed participant types

The target is approximately 30 participants with usable recordings. Seek 3--4 professional/proficient CSL
contributors for demonstration/checking/reference recordings if possible; use volunteers who learn from frozen
videos for the remaining scale. Do not maintain enrollment funnel statistics. Preserve participant type and make
participant identity the minimum downstream split group.

### Core plus compact stress subset

All recorded participants perform a common video-guided core. A smaller matrix repeats selected clips for
off-axis and occlusion conditions, including reviewer-cited `30 degree`/`60 degree` and partial hand or object
occlusion. This avoids repeating the entire corpus in every condition.

### Claim boundary follows participant type

Every take binds the frozen reference video and its known text meaning. Basic QC checks visible reproduction. A
professional/proficient contributor may check content or takes when available, but is not required for each
volunteer take. The paper must describe volunteers as video-guided and cannot claim natural fluent-signer
generalization from their recordings.

### Immutable, privacy-aware identities

Research manifests use opaque participant/session/take IDs and explicit artifact relations. Raw radar/reference
streams, acquisition config, synchronization result, participant type, QC state and checksums are not overwritten.

## Risks

- Professional/proficient CSL contributors may be unavailable; proceed with volunteers but state that limitation.
- Reference video is identifiable; access/release scopes must be explicit even if radar is described as private.
- Changing the reference videos or radar setup mid-collection can create incomparable groups; version the change.
- An oversized condition matrix can exhaust participants and operators; stress coverage stays compact and frozen.
- Volunteer imitation can be mistaken for natural CSL; participant types and prompted task wording are mandatory.

## Handoff

The collection publishes raw session packages, an opaque-ID manifest, reference-content/config identities, compact
QC, participant-type coverage and checksums. `data_rebuild` validates that snapshot before participant-disjoint
splits or derived tensors.
