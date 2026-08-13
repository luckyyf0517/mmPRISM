## ADDED Requirements

### Requirement: CE-CNSL is an independent parallel source

The system SHALL identify CE-CNSL as `DATASET-CE-CNSL` and SHALL keep its source manifest, artifact root, split,
label transform, and metrics separate from CSL-Daily. CE-CNSL work SHALL NOT block the CSL-Daily P0 path.

#### Scenario: Run both source pipelines concurrently

- **WHEN** CSL-Daily annotation and CE-CNSL intake or pilot are active
- **THEN** each artifact binds exactly one dataset ID and source manifest
- **AND** CE-CNSL full-corpus work yields resources required by the CSL-Daily critical path
- **AND** neither result is published as one mixed CSL-Daily metric.

### Requirement: CE-CNSL source and signer identity are audited before promotion

The system SHALL receipt the complete source and verify sample-number video/CSV coverage. Because the published old
signer mapping is acknowledged to be stale from H onward, participant metadata SHALL use a frozen repair table with
published, observed, repaired, evidence, and review fields before participant-based analysis.

#### Scenario: Prepare a participant-based CE-CNSL split

- **WHEN** a CE-CNSL run groups samples by signer
- **THEN** preparation rejects directory or CSV signer labels without an accepted repair table
- **AND** the split receipt binds the repair-table checksum
- **AND** the official all-signers-in-all-splits assignment is not described as participant-disjoint.

### Requirement: CE-CNSL label normalization is reversible

Every CE-CNSL record SHALL preserve spoken Chinese, raw Gloss, regional notes, and any normalized Gloss together with
the normalization version. Formal vocabulary construction SHALL use only the configured training partition.

#### Scenario: Normalize a gesture variant

- **WHEN** an experiment maps a raw variant, direction, or subject/object annotation to a normalized Gloss
- **THEN** the raw label remains unchanged and addressable
- **AND** the record stores the transform version
- **AND** dev/test labels do not add entries to the training vocabulary.

### Requirement: Full processing requires a bounded pose pilot

The system SHALL NOT start full-corpus CE-CNSL annotation or simulation until a frozen 120--240-sequence pilot covers
the declared signer/device/frame-geometry/length/difficulty strata and passes the shared pose contract and review gate.

#### Scenario: Promote CE-CNSL for full annotation

- **WHEN** an operator requests full-corpus processing
- **THEN** the promotion receipt binds source, repaired labels, pilot manifest, model/config, coverage, QC, and review
- **AND** missing strata, incomplete coverage, or unresolved material pose failures reject promotion.

### Requirement: CE-CNSL reuses contracts without inheriting CSL-Daily assumptions

Shared annotation, scheduling, QC, simulation, and delivery functions SHALL consume validated source records. A
CE-CNSL adapter SHALL own its source layout and labels and SHALL NOT emulate CSL-Daily through path replacement.

#### Scenario: Process a CE-CNSL sequence

- **WHEN** the CE-CNSL adapter emits a validated source record
- **THEN** shared annotation emits native pose/scores, finite canonical pose, confidence, validity, imputation, and mask
- **AND** source-specific paths or label fields do not enter model code
- **AND** aspect ratio is preserved rather than forcing a 256x256 stretch.

### Requirement: Adaptation results remain dataset-specific

The first promoted transfer experiment SHALL use the stable CSL-Daily base followed by CE-CNSL adaptation. Evaluation
SHALL report CSL-Daily and CE-CNSL separately; CE-CNSL synthetic products SHALL NOT support real-radar claims.

#### Scenario: Report sequential adaptation

- **WHEN** a `CSL-Daily -> CE-CNSL` model is evaluated
- **THEN** the report includes each dataset's split and label policy
- **AND** shows CE-CNSL gain and CSL-Daily retention separately
- **AND** does not characterize visual background diversity as radar-domain diversity.
