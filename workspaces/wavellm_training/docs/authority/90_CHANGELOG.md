# WaveLLM Training Changelog

Status: current
Owner: WaveLLM training lane
Authority scope: Material changes to the WaveLLM training and evaluation workflow.
Last reviewed: 2026-08-12

## 2026-08-12

- Established a focused training workspace without moving canonical model or training code.
- Preserved the mT5-only support boundary and reclassified prior engineering runs as Logs.
- Designated the audited original-submission CSL-News-100 cam-pose WaveLLM checkpoint as the revision-wide shared
  semantic initialization, with explicit provenance, compatibility, holdout-evaluation, and retraining triggers.
- Integrated WaveLLM train/evaluate with rank-zero formal-run ownership, distributed checkpoint consistency, exact
  prediction sharding, and cross-rank metrics; multi-process WaveLLM validation and DDP resume remain open.
