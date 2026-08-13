## Context

CSL-Daily is the revision-critical controlled baseline, but its historical derived assets are unavailable and the
new `annotation_v2` path is already being built. CE-CNSL is small enough to intake in parallel and adds vocabulary
and capture diversity. Its official layout and labels also contain source-specific risks: heterogeneous video
formats, regional CSL metadata, an acknowledged stale signer mapping, and a split containing every signer in every
partition.

## Decisions

### Priority and ownership

CSL-Daily remains P0. CE-CNSL is P1-parallel and remains inside `data_rebuild`; no new workspace is justified.
Intake, metadata audit, and a bounded pilot may run now. Full-corpus CE-CNSL GPU work yields to CSL-Daily capacity.

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

1. Source: checksum-bound full archive/extraction, exact video/CSV coverage, decode metadata, and license status.
2. Identity: frozen signer repair and reversible label representation.
3. Pose pilot: 120--240 stratified sequences pass contract coverage and manual overlay QC.
4. Promotion: explicit acceptance binds source, labels, pilot, config/model, and QC before full processing.

## Risks And Mitigations

- Missing license: allow only private pilot processing; block redistribution until clarified.
- Pose failure from heterogeneous framing: stratify the pilot, preserve aspect ratio, and retain explicit masks.
- Dialect collapse: preserve raw variants and regional notes; make normalization versioned and reversible.
- Participant leakage: never use the official split for new-user claims.
- P0 resource contention: CE-CNSL full workers yield to the CSL-Daily critical path.
- Premature abstraction: extract shared functions only when both adapters exercise the contract.
