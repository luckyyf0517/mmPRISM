# CSL-Daily OmniHand Reconstruction Operation

Status: current
Owner: OmniHand training lane
Authority scope: CubeNet training and export requirements for the CSL-Daily synthetic reconstruction control.
Last reviewed: 2026-08-13

## Role

This operation consumes only an accepted, immutable CSL-Daily pre-beamforming synthetic-FMCW pose-reconstruction
delivery. Its runtime processor derives the simulated cube on the selected device, then CubeNet reconstructs the
selected camera-pose target and exports predicted poses to the WaveLLM pose-only lane. Frame features are a separate
later fusion export. It is a synthetic control, not evidence of reconstruction from real measured radar or of the
manuscript's MANO/ray-tracing pipeline.

The cross-workspace meanings of cam-pose, synthetic-domain mmw-pose and radar feature are fixed by the
[research execution model](../../../../../docs/authority/30_ARCHITECTURE/RESEARCH_EXECUTION_MODEL.md). This page
owns only OmniHand training, prediction and feature export.

## Formal Inputs

Before a formal run, require a raw-radar pose-reconstruction receipt with source/annotation/simulation provenance,
manifest and split hashes, Parquet inventory/checksums, signal representation/dtype/shape, simulator and processor
fingerprints, coordinate frame/units, reader parity, capacity/throughput report, and one-batch GPU runtime-processor
smoke. Legacy path JSON, live sidecars, and persisted power cubes are not trainable inputs.

Historical CSL-Daily `val.json` and `test.json` are byte-identical. A run using that mapping is a labelled
historical replay only; a new synthetic control must use a distinct frozen control holdout or report validation
without presenting it as an independent test.

## Training And Export Sequence

```text
accepted pre-beamforming synthetic-FMCW delivery
-> runtime range/Doppler/beamforming adapter smoke
-> small CubeNet train/reload/evaluate smoke
-> frozen formal CubeNet protocol and checkpoint(s)
-> fold-bound cross-fitted predicted-pose export for WaveLLM training rows
-> pose-only WaveLLM consumer receipt
-> optional checkpoint-bound frame-feature export
-> optional fusion WaveLLM consumer receipt
```

The existing model exposes `frame_features [B,T,F]`; feature availability inside a forward output is not by itself
a feature-delivery contract. The exporter must bind each output to sample ID, frame identity/mask, source cube
manifest, split assignment, producing checkpoint and metadata checksums, model fingerprint, feature dimension,
dtype/shape/checksum, and inference precision/device.

For a primary predicted-mmWave-pose WaveLLM experiment, a training-row prediction comes from a checkpoint whose
training membership excludes that row. Validation/test outputs come from a checkpoint trained only on the
corresponding training partition. Export coverage must be exactly one valid producer per consumer row. Apply the
same requirement to a later feature export. In-sample exports may support debugging only under
`diagnostic_in_sample` and cannot populate comparison tables.

## Required Reports

- reconstruction metrics with sample-level predictions/targets and frozen pose metric protocol;
- source/split/checkpoint/export linkage plus exact row/fold coverage;
- GPU training/inference cost profile under the final configuration;
- synthetic-control limitation and simulation protocol in every downstream handoff.

The later WaveLLM fusion comparison must use the same predicted poses for its pose-only and feature-fusion rows, so
the feature effect is not confounded with a different CubeNet checkpoint or split.
