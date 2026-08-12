## Why

The data rebuild workspace has accepted a Parquet delivery contract, but canonical training adapters still read
JSONL manifests and NPY sidecars only. Without an executable delivery contract, a future large materialization
could silently mix splits, lose provenance, or promote interim CSL-News visual poses as model-ready radar data.

## What Changes

- Add a lazy PyArrow-backed Parquet delivery implementation under `src/mmprism/data/`.
- Define immutable delivery metadata, copied input bindings, split-homogeneous part/chunk placement, inventory,
  index, checksum and validation behavior.
- Implement distinct pose-reconstruction and sign-language-translation readers that return the existing
  dependency-light sample/batch tensor contracts.
- Implement a deterministic materializer from a frozen JSONL manifest plus split assignment, with capacity
  planning and atomic no-clobber publication.
- Add fixture parity and tamper-rejection coverage. No real CSL-News sidecar directory is materialized.

## Non-Goals

- Producing a real or paper-facing Parquet dataset before calibrated radar and complete task modalities exist.
- Changing current model architecture, formal training orchestration, raw data, live annotation workers, or legacy
  forensic code.
- Creating a universal sparse schema or serializing Python/NPY blobs into Parquet.
- Moving shared code into workspace directories.

## Impact

- New optional `data-parquet` dependency and lockfile entries.
- New public data reader/materializer APIs and CPU-only tests that opt into the optional dependency.
- Data rebuild Authority records implementation status and exact safety boundary.
