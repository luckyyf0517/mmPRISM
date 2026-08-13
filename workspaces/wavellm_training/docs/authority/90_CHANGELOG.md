# WaveLLM Training Changelog

Status: current
Owner: WaveLLM training lane
Authority scope: Material changes to the WaveLLM training and evaluation workflow.
Last reviewed: 2026-08-13

## 2026-08-12

- Established a focused training workspace without moving canonical model or training code.
- Preserved the mT5-only support boundary and reclassified prior engineering runs as Logs.
- Recorded the incoming historical WaveLLM bundle as preservation-only while transfer is in progress. The historical
  model role is pending stable receipt and audit; no recovered checkpoint, encoder, or metric claim is yet accepted.
- Retained the verified local CSL-News-derived mT5 smoke as fallback and made an immutable asset receipt an explicit
  formal-training gate for either candidate initialization.
- Integrated WaveLLM train/evaluate with rank-zero formal-run ownership, distributed checkpoint consistency, exact
  prediction sharding, and cross-rank metrics; multi-process WaveLLM validation and DDP resume remain open.
- Defined the CSL-Daily reproduction matrix and its missing controls: explicit pose-only mode, checkpoint-bound
  CubeNet feature export, cross-fitted predicted inputs, production semantic metrics, and replay-only treatment of
  the historical duplicated validation/test mapping.
- Implemented the v2 JSONL+NPY WaveLLM input-mode contract: `pose_only` omits and forbids radar features, has no
  radar projector/fusion parameter surface, and binds its checkpoint metadata and evaluation path to the mode.
  The focused CPU train/evaluate integration test covers this path.
- Implemented final Parquet `pose_only` translation delivery: the schema, rows, static dimensions, delivery metadata
  and reader all bind the mode and omit radar features rather than serializing empty placeholders. CubeNet feature
  provenance remains pending.

## 2026-08-13

- Recorded migration staging of the historical WaveLLM bundle as a read-only handoff observation. Upload completion
  does not replace the `DEC-046` receipt, format/world-size, metadata/tensor, or controlled-load audit.

- Applied `DEC-048`: CSL-Daily first-loop execution is `CSLD-WL-01` camera-pose and `CSLD-WL-02` cross-fitted
  predicted-pose, both `pose_only`. The checkpoint-bound feature/fusion comparison (`CSLD-WL-03`) remains a required
  separately reported third stage, but cannot block the first two runs.
