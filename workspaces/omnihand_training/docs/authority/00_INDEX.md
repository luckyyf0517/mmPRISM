# OmniHand Training Authority

Status: current
Owner: OmniHand training lane
Authority scope: Current OmniHand model, training, resume, prediction, and evaluation workflow.
Last reviewed: 2026-08-12

## Boundary

This workspace starts from a validated radar-cube/metric-pose manifest and split. It does not infer paths,
perform beamforming, own shared artifact infrastructure, or manufacture paper-facing evidence claims.

## Current State

- Canonical CubeNet/temporal model and versioned pose metrics are implemented.
- Single-device formal train/checkpoint/reload/evaluate and completed-epoch exact resume are validated.
- Synthetic A100 evidence is accepted as engineering validation, not a paper result.

Active blockers: real calibrated model-ready manifests and production training/evaluation. Distributed
execution is active implementation work and must be accepted separately before becoming current truth.

Next action: complete distributed lifecycle validation, then run the frozen real-data protocol when its
cross-workspace delivery is available.

## Canonical Locations

- Code: `src/mmprism/models/cubenet.py`, `src/mmprism/training/omnihand_*.py`
- Config: `configs/examples/omnihand_*.yaml`
- Scripts: `scripts/run_omnihand_*.sh`
- Tests: OmniHand unit, integration, resume, and distributed tests under `tests/`

## Authority And Evidence

- [Model and lifecycle](30_ARCHITECTURE/OMNIHAND.md)
- [Changelog](90_CHANGELOG.md)
- [Accepted engineering logs](../logs/README.md)
