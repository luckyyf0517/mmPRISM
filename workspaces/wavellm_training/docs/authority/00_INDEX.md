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
- Formal train/evaluate now use the shared distributed lifecycle, exact prediction sharding, cross-rank metric
  merging, and rank-zero artifact publication; WaveLLM-specific multi-process validation remains open.
- Current character metrics and synthetic A100 runs are engineering protocols, not paper results.
- The recovered CSL-News-derived mT5-only export is load-smoke-verified as a controlled language initialization.
  The original complete cam-pose WaveLLM checkpoint and historical hand-pose encoder are unavailable, so historical
  end-to-end reproduction, pose compatibility, and historical metrics are explicitly out of scope.

Active blockers: local-derived mT5 asset receipt/import, real aligned model-ready data, production
BLEU/ROUGE/semantic protocols, WaveLLM multi-process/NCCL validation, and final paper evaluation.

Next action: implement a checksum-bound local-derived mT5 receipt/import, restore the CSL-Daily simulation
second-stage protocol, then train the new canonical geometry adapters and declared language-model scope.

Full CSL-News reconstruction and retraining are not P0. They remain a separately reported future ceiling or
provenance task, not a replacement for the unavailable historical end-to-end checkpoint.

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
