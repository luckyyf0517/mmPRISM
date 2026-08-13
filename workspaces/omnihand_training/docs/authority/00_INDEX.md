# OmniHand Training Authority

Status: current
Owner: OmniHand training lane
Authority scope: Current OmniHand model, training, resume, prediction, and evaluation workflow.
Last reviewed: 2026-08-13

## Boundary

This workspace starts from a validated radar-cube/metric-pose manifest and split. It does not infer paths,
perform beamforming, own shared artifact infrastructure, or manufacture paper-facing evidence claims.

## Current State

- Canonical CubeNet/temporal model and versioned pose metrics are implemented.
- Single-device formal train/checkpoint/reload/evaluate and completed-epoch exact resume are validated.
- Two-process CPU/Gloo formal training is integration-validated with one rank-zero run, exact prediction coverage,
  merged metrics, consistent model-state hashes, and a checkpoint numerically matched to a single-process reference.
- Synthetic A100 evidence is accepted as engineering validation, not a paper result.
- CSL-Daily is the next synthetic control input after its accepted pre-beamforming synthetic-FMCW delivery. The
  runtime processor derives CubeNet inputs on device; its cross-fitted predicted poses are the first WaveLLM handoff.
  Checkpoint-bound frame features are a later fusion-only export and do not block pose-only translation.

Active blockers: accepted CSL-Daily raw-radar delivery and runtime processor adapter, checkpoint-bound cross-fitted
predicted-pose export, real calibrated model-ready manifests, production training/evaluation, and multi-GPU NCCL
smoke. Frame-feature
export blocks only the later fusion comparison. DDP completed-
epoch resume remains unsupported because per-rank RNG and sampler state are not yet captured.

Next action: after data-rebuild receipt, validate the raw-radar runtime processor on a CSL-Daily pilot, then run the
small synthetic smoke, freeze its reconstruction protocol, and produce the cross-fitted predicted-pose export for
the pose-only WaveLLM control. Feature export and fusion follow independently; separately run the multi-GPU NCCL
smoke.

## Canonical Locations

- Code: `src/mmprism/models/cubenet.py`, `src/mmprism/training/omnihand_*.py`
- Config: `configs/examples/omnihand_*.yaml`
- Scripts: `scripts/run_omnihand_*.sh`
- Tests: OmniHand unit, integration, resume, and distributed tests under `tests/`

## Authority And Evidence

- [Model and lifecycle](30_ARCHITECTURE/OMNIHAND.md)
- [CSL-Daily reconstruction operation](40_OPERATIONS/CSL_DAILY_RECONSTRUCTION.md)
- [Changelog](90_CHANGELOG.md)
- [Accepted engineering logs](../logs/README.md)
