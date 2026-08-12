## ADDED Requirements

### Requirement: CSL-Daily result roles are explicit

Every formal CSL-Daily run SHALL record one of `synthetic_csl_daily_control`, `historical_replay`, or
`revision_real_radar` as its evidence role.  A synthetic CSL-Daily control SHALL NOT be promoted as evidence of
real-radar fidelity, real-world generalization, or the manuscript-described MANO/ray-tracing simulator.

#### Scenario: Report a synthetic CubeNet result

- **WHEN** a CubeNet is trained on a CSL-Daily camera-pose-derived simulated cube
- **THEN** its run receipt records `synthetic_csl_daily_control`
- **AND** its report labels the simulation protocol
- **AND** paper evidence promotion rejects it for a real-radar claim.

### Requirement: Duplicated legacy validation/test files are replay-only

When historical CSL-Daily validation and test assignment files have identical content, the system SHALL record
that identity and SHALL permit their use only for a labelled historical replay.  A new independent test claim
SHALL require a distinct frozen assignment.

#### Scenario: Prepare a new CSL-Daily formal test

- **WHEN** a new formal run requests an independent CSL-Daily test metric
- **AND** the configured test assignment hashes to the same bytes as validation
- **THEN** preparation fails unless the run role is `historical_replay`
- **AND** a replay result is labelled `legacy_validation_as_test`.

### Requirement: Received historical derived CSL-Daily assets are preserved before regeneration

Uploaded historical pose, synthetic signal/cube, feature, or split assets SHALL be independently receipted before
any canonical annotation or simulation regeneration. The receipt SHALL preserve original relative identity and
record count, shape/dtype, checksum, source/caption linkage, and producer/configuration evidence or its absence.

#### Scenario: Use old CSL-Daily pose for a direct replay

- **WHEN** an existing uploaded CSL-Daily pose directory is proposed as a reconstruction or translation input
- **THEN** the system records it as a receipt-bound historical derived asset without changing its bytes
- **AND** verifies or explicitly rejects its source-frame and caption/split linkage
- **AND** any run using it records `historical_replay` rather than treating it as a regenerated canonical annotation.

### Requirement: Pose-only WaveLLM has no synthetic radar input

The canonical WaveLLM surface SHALL support an explicit `pose_only` mode.  That mode SHALL require pose,
confidence, mask, and caption; it SHALL NOT require, load, project, or fabricate a radar-feature tensor.
Checkpoint metadata SHALL bind the selected mode and evaluation SHALL reject a different mode.

#### Scenario: Train a camera-pose semantic control

- **WHEN** a `pose_only` WaveLLM task is prepared
- **THEN** the manifest validator accepts no radar-feature reference
- **AND** the constructed model has no radar projector or fusion parameters
- **AND** the run receipt records `input_mode=pose_only`.

### Requirement: Fusion WaveLLM uses checkpoint-bound frame features

The `pose_plus_radar_feature` WaveLLM mode SHALL consume only a radar feature exported from a declared producer
checkpoint and frozen cube manifest.  The export SHALL record sample identity, frame mask, feature shape/dtype,
feature checksum, source/split hashes, producer checkpoint/metadata hashes, producer model fingerprint, and
inference precision.

#### Scenario: Attempt to train with an incompatible feature export

- **WHEN** the feature export's checkpoint, frame mask, coordinate frame, sample set, or split hash differs from
  the configured translation delivery
- **THEN** formal preparation fails before model construction
- **AND** the error identifies the failed binding.

### Requirement: Predicted-mmWave-pose training avoids in-sample reconstruction leakage

For any paper-relevant `predicted_mmw_pose` or `predicted_mmw_pose_plus_cube_feature` WaveLLM comparison, each
training-row prediction and feature SHALL be produced by a CubeNet fold that did not train on that row.  Evaluation
rows SHALL be produced by a checkpoint fitted only to the corresponding training partition.

#### Scenario: Prepare a primary predicted-pose comparison

- **WHEN** a formal primary predicted-mmWave-pose WaveLLM run is prepared
- **THEN** its input manifest proves one excluded fold for every training row
- **AND** validation/test rows bind a training-partition checkpoint
- **AND** missing, duplicated, or in-sample fold coverage fails preparation.

### Requirement: Improved annotations preserve and compare against the baseline

The historical-transform CSL-Daily annotation SHALL be frozen as `annotation_v1`.  An annotation candidate SHALL
use a new version and immutable output root, retain raw pose output and failure records, and publish deterministic
QC plus stratified visual-review comparison before it can become a training source.

#### Scenario: Promote an annotation candidate

- **WHEN** an operator proposes `annotation_v2` for model delivery
- **THEN** the promotion report includes v1/v2 coverage, failure, temporal, identity, and blinded review results
- **AND** the report identifies raw source/config/model artifacts for both versions
- **AND** the source manifest remains bound to exactly one chosen annotation version.
