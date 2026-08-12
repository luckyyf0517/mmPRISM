## Context

The accepted delivery contract keeps source, sidecar, manifest, split, and Parquet layers separate.  The
in-progress CSL-Daily simulation change produces the upstream camera-pose and simulated-cube route.  The
existing `GeometryGuidedMT5` always expects pose, confidence, and radar features, while `OmniHandCubeNet`
already exposes per-frame features as an internal output.  Neither fact establishes a valid pose-only baseline
or a reusable feature artifact.

The historical mirror contains a CSL-Daily train/validation mapping, but its `val.json` and `test.json` are
byte-identical.  It also contains incoming historical WaveLLM checkpoints.  Both are evidence sources only
until independently receipted.

The raw upload may additionally contain existing `sentence/poses`, synthetic signals, or feature directories.
Those are historical derived candidates rather than raw source. They are valuable for direct replay because they
may be the exact old camera-pose input the former CubeNet/WaveLLM pipeline consumed. They must be receipted and
audited before a regeneration changes any interpretation.

## Decisions

### Dataset roles

Every CSL-Daily result declares exactly one role:

| Role | Inputs | Permitted conclusion |
|---|---|---|
| `synthetic_csl_daily_control` | visual pose, skeleton-simulated cube, or their predictions | Controlled data/pipeline behavior only; no real-radar or real-world generalization claim. |
| `historical_replay` | receipt-bound historical split/config/checkpoint/prediction | Whether a historical convention is reproduced, explained, or unavailable. |
| `revision_real_radar` | accepted new real-radar manifest and participant-disjoint split | Potential reviewer-facing reconstruction/translation evidence. |

`val.json == test.json` is admissible only in `historical_replay` and must be labelled `legacy_validation_as_test`.
New CSL-Daily controls use an explicit validation partition and either an independently derived eligible holdout
or no `test` summary; no result may imply that the duplicated file is an independent test set.

### Annotation versions and selection

First classify every transferred historical derived candidate by path, count, shape/dtype, checksum, source-frame
identity, split mapping, producer/config evidence, and deterministic source-overlay samples. It receives a stable
`legacy_received` asset ID and remains immutable. It can be selected for a `historical_replay` direct training run
only after these checks bind it to the source/caption records. It is never silently renamed to a new annotation
version or promoted as a new independent split.

`annotation_v1` is the frozen canonical historical-transform replay: full-image single-person RTMW3D inference,
confidence thresholding, sequence depth centering, 17-body plus 42-hand selection, dual-hand 24-joint
projection, and failure quarantine. It is a newly generated, separately versioned comparison baseline; it must not
overwrite or be presumed byte-identical to a received historical pose asset. It is the valid starting point for
controlled `annotation_v2` comparison.

`annotation_v2` is a separately versioned candidate.  It may improve person selection/tracking, temporal
consistency, left/right identity handling, crop policy, or confidence handling, but must preserve the raw frames,
raw RTMW3D output, configuration, failure records, and a direct v1 comparison.  It is selected only after a
predeclared stratified review and diagnostic report; lower jitter alone is insufficient proof of better pose.

### WaveLLM modality modes

The task configuration chooses an explicit mode:

| Mode | Required data | Trainable components | Prohibited behavior |
|---|---|---|---|
| `pose_only` | pose, pose confidence, mask, caption | pose encoder and language adapter/backbone scope | Loading, projecting, or zero-filling a radar feature. |
| `pose_plus_radar_feature` | pose, confidence, radar feature, mask, caption | pose encoder, radar projector, fusion, language adapter/backbone scope | Inferring a feature path or mixing a feature protocol/checkpoint. |

The model/checkpoint metadata stores this mode, the exact adapter tensor inventory, and the corresponding data
contract.  Evaluation rejects a mismatch.  A pose-only run has a distinct checkpoint scope because fusion and
radar-projector tensors are absent, not merely unused.

### Predicted-pose and feature provenance

The full synthetic cascade has three separately reported inputs:

1. `cam_pose`: camera-pose annotation and confidence, using `pose_only`.
2. `predicted_mmw_pose`: CubeNet pose prediction from a simulated cube, using `pose_only`.
3. `predicted_mmw_pose_plus_cube_feature`: the same predicted pose plus a frame-aligned CubeNet frame feature,
   using `pose_plus_radar_feature`.

An OmniHand feature export is an immutable intermediate artifact, not a model-side path lookup.  Each record
binds the OmniHand checkpoint and checkpoint metadata hashes, source cube manifest hash, split hash, model
fingerprint, feature dimension, frame mask, tensor dtype/shape/checksum, and inference device/precision.  The
exporter fails if the sample identity, frame count, coordinate frame, or model contract differs.

For a paper-relevant cascade, training-row predictions/features are cross-fitted: a fold's CubeNet did not train
on any of that fold's rows.  Validation/test rows use a model fitted only on the training partition.  Fold
assignments, checkpoints, exports, and coverage reports are immutable inputs to WaveLLM.  In-sample features may
be used for fast debugging only if the run/config/result explicitly says `diagnostic_in_sample`; they cannot enter
the primary comparison table.

### Direct replay sequence

Once source and old derived assets are receipted, the quickest historical check is:

```text
received cam-pose (and received synthetic signal/cube when provenance passes)
-> legacy-replay CubeNet train/evaluate or controlled simulation regeneration
-> received cam-pose and replay predicted-mmWave-pose WaveLLM controls
-> metric recomputation against receipt-bound historical predictions/configuration when available
```

The replay uses the old duplicated validation-as-test mapping only under the `historical_replay` role. A received
signal/cube may be used as an input only after its simulator/config/shape/source binding is accepted; otherwise
regenerate it from the same selected received cam-pose with the separately labelled canonical simulation protocol.

### Historical initialization branch

New canonical CSL-Daily baseline work may begin with `MODEL-MT5-BASE`, already pinned as an official asset, once
the data product is accepted.  This does not await the historical log bundle.  The local mT5 export and incoming
historical bundle are parallel forensic/controlled-initialization candidates; each requires its own immutable
receipt before a formal run.  Any initialization comparison uses the same frozen data, mode, optimization budget,
and evaluation protocol and is reported separately from architectural comparisons.

### Annotation quality assessment

The baseline and any candidate version produce a deterministic, stratified sample with source frames, overlays,
and blinded review labels.  Strata include sequence length, estimated confidence, motion/jitter quantile, hand
overlap/occlusion proxy, image cropping, and signer/recording fields when official metadata establishes them.
The report includes coverage, failed/quarantined count, NaN/finite rate, arm/hand visibility, confidence, frame
speed/acceleration, bone-length stability, left/right swap alerts, and reviewer agreement.  Manual reviewers label
tracking, hand identity, gross 3D plausibility, and failure category.  Candidate selection requires no material
regression in these review labels and an explicit decision record.

## Interfaces

The detailed manifest fields and command surface are specified in the accompanying OpenSpec requirement.  All
paths remain portable relative references; roots/devices/assets come from configuration.  CPU-only contract tests
must validate configuration and manifest consistency without importing Torch, MMPose, or Transformers.

## Risks And Mitigations

- CSL-Daily may not expose sufficient signer metadata for a new participant-disjoint split.  Bind the official
  split without inventing a signer key and state the limitation.
- Cross-fitting multiplies CubeNet compute.  Start with a small smoke and fixed fold count, then schedule only
  after a capacity and queue plan.  Do not replace it with in-sample data in paper-facing runs.
- Annotation v2 may improve smoothness while degrading semantics.  Require frame overlays and blinded review
  before promotion.
- Received legacy poses may have incomplete or unrecorded producer metadata. Preserve and report them as
  `unavailable_asset` or `explained_data_drift` when linkage cannot be established; do not overwrite them to force
  a comparison.
- Historical evaluator/config artifacts can be incomplete.  Classify the outcome as unavailable or an explained
  gap, not a failed new baseline.
