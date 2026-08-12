# WaveLLM Training Changelog

Status: current
Owner: WaveLLM training lane
Authority scope: Material changes to the WaveLLM training and evaluation workflow.
Last reviewed: 2026-08-12

## 2026-08-12

- Established a focused training workspace without moving canonical model or training code.
- Preserved the mT5-only support boundary and reclassified prior engineering runs as Logs.
- Superseded the unavailable complete original WaveLLM baseline with a CSL-News-derived mT5-only initialization for
  new canonical CSL-Daily training; historical end-to-end model, hand-pose encoder, and metrics are unavailable and
  excluded from reproduction claims.
- Recorded the verified local mT5 load/canonical-wrapper smoke and made a checksum-bound local-derived asset receipt
  an explicit formal-training gate.
- Integrated WaveLLM train/evaluate with rank-zero formal-run ownership, distributed checkpoint consistency, exact
  prediction sharding, and cross-rank metrics; multi-process WaveLLM validation and DDP resume remain open.
