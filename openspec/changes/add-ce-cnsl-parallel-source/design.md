## Context

CSL-Daily is the revision-critical controlled baseline, but its historical derived assets are unavailable and the
new `annotation_v2` path is already being built. CE-CNSL adds vocabulary and capture diversity, but its adapter
should be designed from the completed CSL-Daily path rather than assumptions made during that rebuild. Its official
layout and labels also contain source-specific risks: heterogeneous video
formats, regional CSL metadata, an acknowledged stale signer mapping, and a split containing every signer in every
partition.

## Decisions

### Priority and ownership

CSL-Daily remains P0. CE-CNSL is a paused P1 follow-on source inside `data_rebuild`; no new workspace is justified.
During CSL-Daily's late stable phase, explicit project-owner authorization may unlock only source download, checksum,
and immutable receipt. Adapter, label-repair, pilot, full-processing, and GPU tasks still require an accepted
CSL-Daily end-to-end stable loop and explicit reactivation. The OpenSpec change ID is retained for stable links; it
no longer means concurrent execution.

### Shared core and dataset adapters

Reusable functions accept validated source records rather than CSL-Daily paths. The shared boundary covers scheduler
leases, native/canonical pose payloads, confidence/validity masks, QC, synthetic FMCW generation, and typed delivery.
Adapters own only source discovery, video decode metadata, label parsing, sample identity, and official split fields.

No generic abstraction is introduced until the same behavior is exercised by both sources. Existing CSL-Daily
commands may remain thin wrappers while shared package functions are extracted behind them.

### Identity and labels

`DATASET-CE-CNSL` has its own manifest and artifact root. Stable sample number is the primary upstream identity.
Signer is a repaired, evidence-bearing metadata field because the published old mapping is unreliable from H onward.

Four label layers remain distinct: `spoken_chinese`, `gloss_raw`, `gloss_normalized` with transform version, and
`regional_note`. Training-only vocabulary construction is mandatory. A run records whether it predicts Chinese text,
raw Gloss, or a named normalized Gloss representation.

### Evaluation

The official CE-CNSL split is retained for benchmark comparability but is not participant-disjoint. Any repaired
participant-disjoint analysis is an additional internal split and is never reported as official WER. Adaptation
experiments evaluate both datasets separately so vocabulary/domain gain and CSL-Daily forgetting remain visible.

## Gates

1. Optional early download: CSL-Daily is in its late stable phase and the project owner explicitly authorizes source
   download and immutable receipt.
2. Activation: accepted CSL-Daily `annotation_v2 -> synthetic FMCW -> OmniHand -> pose-only WaveLLM` stable loop
   plus explicit reactivation before any adapter or pilot task.
3. Source: checksum-bound full archive/extraction, exact video/CSV coverage, decode metadata, and license status.
4. Identity: frozen signer repair and reversible label representation.
5. Pose pilot: 120--240 stratified sequences pass contract coverage and manual overlay QC.
6. Promotion: explicit acceptance binds source, labels, pilot, config/model, and QC before full processing.

## Risks And Mitigations

- Missing license: allow only private pilot processing; block redistribution until clarified.
- Pose failure from heterogeneous framing: stratify the pilot, preserve aspect ratio, and retain explicit masks.
- Dialect collapse: preserve raw variants and regional notes; make normalization versioned and reversible.
- Participant leakage: never use the official split for new-user claims.
- Premature execution: manual download authorization is recorded separately and cannot unlock adapter, pilot, or GPU
  work.
- Premature abstraction: extract shared functions only when both adapters exercise the contract.
