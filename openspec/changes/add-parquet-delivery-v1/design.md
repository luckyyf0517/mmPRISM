## Context

The repository uses business workspaces for ownership only. Canonical shared code remains in `src/mmprism/`.
The current `PoseReconstructionManifest` and `SignLanguageTranslationManifest` establish the target tensor
contracts: radar cube plus metric pose for OmniHand; aligned metric pose/confidence/radar feature/caption for
WaveLLM. The frozen JSONL manifest and split assignment remain the upstream evidence source.

## Decisions

### Task-specific products

The implementation exposes exactly two products:

- `pose_reconstruction` / `mmprism.pose_reconstruction.sample_v1`
- `sign_language_translation` / `mmprism.sign_language_translation.sample_v1`

They share identity/provenance columns but have separate typed tensor schemas. Current CSL-News pose+caption is
not accepted by either materializer because it lacks the required radar/metric contracts.

### Self-contained immutable delivery

A completed delivery root contains `delivery.json`, `schema.json`, copied input
`source_manifest.jsonl` and `split_assignments.jsonl`, part inventory, sample index, validation report and
checksums. Output paths stored inside metadata are portable relative paths. The builder writes a unique hidden
staging sibling and atomically publishes only after validation. Existing destinations are rejected.

### Streaming file policy

A part contains at most 1,024 rows and a split-homogeneous chunk contains at most 64 parts. Rows are ordered by
UTF-8 `sample_id`. Each part is one Parquet row group. The materializer streams samples into a part-sized staged
buffer, and capacity planning records the transient estimate before writing; a future change may introduce smaller
row groups only with a compatible schema/validator update.

### Optional dependency boundary

PyArrow is a `data-parquet` optional extra. Imports are lazy: metadata-only contract code can import without it,
and only read/write/Parquet-validation paths request the dependency. Standard package import retains no PyArrow,
Torch, Lightning or Transformers dependency.

### Validation

Planning validates frozen manifest/split coverage, product-specific model-ready input contracts, deterministic
placement, and conservative free-space requirement before writes. Completed validation checks copied input hashes,
inventory/index membership, part checksum/count/schema/row-group bounds, row identity/split/group consistency and
random-access reader parity. The materializer uses existing source adapters, so source NPY checksum and tensor
checks remain enforced.

## Risks

- PyArrow nested-list type behavior may vary across releases: pin a compatible optional version and test exact
  fixture round trip.
- Reading a full part for random access is initially acceptable for correctness, but future reader performance
  work can add row-group-aware caching without altering the format.
- A real 4D cube can be large: fixed 1,024-row files remain the user-level unit, while smaller row groups constrain
  transient memory.
