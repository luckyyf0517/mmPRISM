## ADDED Requirements

### Requirement: Final training data is task-specific Parquet

The system SHALL materialize final model-ready data as one of the supported task-specific Parquet products. Each
logical sample SHALL occupy one row and SHALL contain only the tensors/text required by its target adapter plus
typed identity and provenance. The system SHALL NOT use a universal mixed schema or opaque Python/NPY blobs.

#### Scenario: Reject a non-model-ready CSL-News visual pose record

- **WHEN** a source record contains only interim visual pose/caption modalities and lacks the target radar contract
- **THEN** either final product materializer rejects it before writing
- **AND** no processed delivery is published.

### Requirement: Delivery placement is deterministic and split-homogeneous

The builder SHALL order selected samples by UTF-8 sample ID within each frozen split. A part SHALL contain no more
than 1,024 rows, and a chunk SHALL contain no more than 64 parts. A part and chunk SHALL contain one split only.

#### Scenario: Build a delivery larger than one part

- **WHEN** a split contains 1,025 valid selected samples
- **THEN** the builder produces a full 1,024-row first part and a one-row terminal second part
- **AND** the output paths are stable ordinal chunk/part names independent of worker or source archive order.

### Requirement: Delivery preserves frozen input identity

A completed delivery SHALL include copied frozen source-manifest and split-assignment input files, their SHA-256
identities, a part inventory, sample index, schema metadata, validation report and top-level checksum list. The
builder SHALL reject an existing output destination and publish only after atomic staging validation.

#### Scenario: Detect copied input or part tampering

- **WHEN** a copied input or completed Parquet part changes after publication
- **THEN** delivery validation fails before the delivery is accepted by a reader with checksum verification enabled.

### Requirement: Optional Parquet support is lazy

Parquet support SHALL be an explicit optional dependency. Importing the base data package or validating non-Parquet
manifest contracts SHALL NOT import PyArrow, Torch, Lightning or Transformers.

#### Scenario: Use a base environment without PyArrow

- **WHEN** PyArrow is unavailable and a caller requests a Parquet read or write operation
- **THEN** the system raises a clear delivery error that names the required optional extra
- **AND** ordinary non-Parquet imports remain usable.

### Requirement: Parquet readers preserve current adapter tensor contracts

The pose reconstruction and sign-language translation readers SHALL return the same sample and collated batch
contracts as the existing JSONL-plus-NPY adapters, including shape, dtype, frame-mask and coordinate validation.

#### Scenario: Read a Parquet translation row

- **WHEN** a valid translation row is read from the selected split
- **THEN** its pose, confidence, radar feature, mask and caption satisfy the existing translation sample contract
- **AND** its collated batch is equivalent to a batch sourced from the corresponding frozen JSONL manifest.
