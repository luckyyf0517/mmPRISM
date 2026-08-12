# WaveLLM Training Authority

Status: current
Owner: WaveLLM training lane
Authority scope: Current WaveLLM model, training, resume, prediction, and evaluation workflow.
Last reviewed: 2026-08-12

## Boundary

This workspace starts from aligned pose, confidence, radar-feature, caption, and split identities. It does
not reconstruct radar/pose inputs, resolve model paths inside models, or own paper claim promotion.

## Current State

- mT5 is the sole canonical generation backend; legacy Phi-3 support is excluded.
- Single-device formal train/checkpoint/reload/evaluate and completed-epoch exact resume are validated.
- Current character metrics and synthetic A100 runs are engineering protocols, not paper results.

Active blockers: real aligned model-ready data, production BLEU/ROUGE/semantic protocols, distributed
execution, and final paper training.

Next action: close distributed lifecycle and production metric contracts, then execute the frozen real-data
protocol when delivered.

## Canonical Locations

- Code: `src/mmprism/models/{stgcn,translation}.py`, `src/mmprism/training/{mt5,wavellm}_*.py`
- Config: `configs/models/mt5_base_v1.yaml`, `configs/examples/{mt5,wavellm}_*.yaml`
- Scripts: `scripts/download_mt5.sh`, `scripts/run_mt5_smoke.sh`, `scripts/run_wavellm_train.sh`
- Tests: WaveLLM, mT5, language metric, and resume tests under `tests/`

## Authority And Evidence

- [Model support](20_CONTRACTS/MODEL_SUPPORT.md)
- [Model and lifecycle](30_ARCHITECTURE/WAVELLM.md)
- [Changelog](90_CHANGELOG.md)
- [Accepted engineering logs](../logs/README.md)
