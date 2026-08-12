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
- CSL-Daily will be evaluated as a three-row controlled matrix: camera-pose semantic ceiling, cross-fitted
  predicted-mmWave-pose, and predicted-mmWave-pose plus checkpoint-bound CubeNet frame feature. JSONL+NPY and final
  Parquet WaveLLM delivery now support the required first-class pose-only mode; checkpoint-bound cross-fitted
  feature export remains the separate pending contract.
- A historical WaveLLM bundle is uploading under the project mirror `log/archived/`. It is preservation-only until
  transfer completion, a stable checksum-bound receipt, format/world-size audit, and controlled load establish its
  identity and completeness. The CSL-News-derived mT5-only export remains a load-smoke-verified fallback;
  no historical end-to-end reproduction, pose compatibility, or metric claim is currently accepted.

Active blockers: accepted CSL-Daily delivery, checkpoint-bound cross-fitted feature
export, production BLEU/ROUGE/semantic protocols, historical bundle receipt/audit, WaveLLM multi-process/NCCL
validation, and final paper evaluation.

Next action: preserve the inbound bundle and await upload completion for its independent receipt/audit. In parallel,
accept CSL-Daily through the data-rebuild gate, implement the producer-bound feature-export contract, then run the
small three-row control matrix with an accepted language asset.

Full CSL-News reconstruction and retraining are not P0. They remain a separately reported future ceiling or
provenance task, regardless of the result of this historical bundle audit.

## Canonical Locations

- Code: `src/mmprism/models/{stgcn,translation}.py`, `src/mmprism/training/{mt5,wavellm}_*.py`
- Config: `configs/models/mt5_base_v1.yaml`, `configs/examples/{mt5,wavellm}_*.yaml`
- Scripts: `scripts/download_mt5.sh`, `scripts/run_mt5_smoke.sh`, `scripts/run_wavellm_train.sh`
- Tests: WaveLLM, mT5, language metric, and resume tests under `tests/`

## Authority And Evidence

- [Model support](20_CONTRACTS/MODEL_SUPPORT.md)
- [Model and lifecycle](30_ARCHITECTURE/WAVELLM.md)
- [CSL-Daily training operation](40_OPERATIONS/CSL_DAILY_TRAINING.md)
- [Changelog](90_CHANGELOG.md)
- [Accepted engineering logs](../logs/README.md)
