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
- The revision baseline is the original-submission cam-pose WaveLLM checkpoint trained on the reported first
  approximately 100 CSL-News archives. Its file location, hash, exact configuration, historical input/split linkage,
  and independent evaluation are still pending intake and audit.

Active blockers: original-checkpoint intake/audit, real aligned model-ready data, production
BLEU/ROUGE/semantic protocols, distributed execution, and final paper evaluation.

Next action: audit and freeze the original checkpoint, restore the CSL-Daily simulation second-stage protocol,
then use the same semantic initialization for matched architecture, DA, stress, and sim2real comparisons.

Full 436-archive CSL-News retraining is not P0. A complete source intake and retraining are opened only by one of the
audit triggers in `DEC-044` or as a separately reported future ceiling experiment.

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
