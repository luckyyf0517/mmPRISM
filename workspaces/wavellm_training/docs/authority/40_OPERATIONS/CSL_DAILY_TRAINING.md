# CSL-Daily WaveLLM Training Operation

Status: current
Owner: WaveLLM training lane
Authority scope: WaveLLM input modes, upstream handoffs, experimental matrix, and historical-replay boundary for CSL-Daily.
Last reviewed: 2026-08-13

## Purpose

This operation turns accepted CSL-Daily products into controlled WaveLLM experiments. Its first delivery measures
semantic degradation from camera pose to reconstructed pose using two pose-only controls. A third feature/fusion
comparison follows independently. It does not establish real-radar robustness or replace the participant-disjoint
revision dataset.

The cross-workspace meanings of cam-pose, synthetic-domain mmw-pose and radar feature are fixed by the
[research execution model](../../../../../docs/authority/30_ARCHITECTURE/RESEARCH_EXECUTION_MODEL.md). This page
owns only the translation modes, comparison matrix and WaveLLM acceptance gates.

The corresponding interface and provenance changes are planned in OpenSpec
`add-csl-daily-reproduction-controls`. Its JSONL+NPY and final Parquet pose-only vertical slices are implemented and
integration-tested: the model has no projector/fusion tensors, checkpoint evaluation enforces the selected input
mode, and pose-only delivery omits rather than fabricates radar features. Checkpoint-bound feature-export
provenance remains the separate unfinished contract.

## Inputs And Evidence Roles

All runs require an immutable source/eligibility manifest, frozen split assignment, non-empty captions, explicit
coordinate frame/units, data-delivery validation, accepted language-model asset, resolved config, clean commit,
and sample-level predictions. A CSL-Daily result has one declared role:

- `synthetic_csl_daily_control`: newly generated camera-pose/skeleton-simulation control; not real-radar evidence.
- `historical_replay`: a receipt-bound reproduction of a former data/config/metric convention.
- `revision_real_radar`: reserved for the later accepted real-radar delivery.

Legacy CSL-Daily `val.json` and `test.json` are byte-identical. They can support only
`historical_replay/legacy_validation_as_test`; they cannot be named a new independent test result.

## Required Matrix

The primary control matrix fixes language initialization, tokenizer, data eligibility, split, caption processing,
optimization budget, generation, and metric protocol unless a row explicitly studies one of these variables.

| ID | Geometry input | Feature input | WaveLLM mode | Question answered |
|---|---|---|---|---|
| `CSLD-WL-01` | accepted camera pose | none | `pose_only` | Semantic ceiling of the selected camera-pose annotation. |
| `CSLD-WL-02` | cross-fitted CubeNet predicted mmWave-pose | none | `pose_only` | Translation degradation attributable to reconstructed pose. |
| `CSLD-WL-03` | same cross-fitted predicted pose | checkpoint-bound CubeNet frame feature | `pose_plus_radar_feature` | Incremental value of the geometry encoder's frame feature beyond predicted pose. |

`CSLD-WL-01` and `CSLD-WL-02` may compare pose sources. `CSLD-WL-02` and `CSLD-WL-03` test the feature/fusion
increment under the same predicted pose. None is a substitute for Reviewer 2's direct cube-to-LLM baseline; that
remains a separately parameter/budget-matched architecture experiment.

## Execution Priority

`CSLD-WL-01` and `CSLD-WL-02` are the first-loop deliverables. The received historical CSL-Daily checkpoints are
also pose-only by a read-only checkpoint namespace audit, so this path is the most direct canonical and historical
comparison boundary. `CSLD-WL-03` waits for the independent checkpoint-bound feature-export contract; it must not
delay either pose-only run. This ordering does not remove the fusion comparison or relax its provenance contract.

## Mode Contract

`pose_only` is a first-class model/data mode. It requires pose, confidence, frame mask, and caption and has no
radar projector or fusion parameters. Supplying all-zero, random, duplicated, or inferred radar features is not
pose-only and must never be reported as such.

`pose_plus_radar_feature` requires a feature with exactly matching sample identity, temporal frame identity,
mask, coordinate frame, source cube manifest, split assignment, producer model fingerprint, checkpoint checksum,
and declared dimension. The feature is exported outside the model and stored in an immutable delivery; model code
may not discover it through sibling paths.

## Predicted-Pose Leakage Gate

For paper-relevant predicted-pose/fusion rows, each training-row prediction and feature must come from a CubeNet
fold that did not train on that row. Validation/test predictions use a checkpoint trained only on their training
partition. The WaveLLM manifest binds fold assignment, producer checkpoint/metadata, feature export hashes, and
coverage report. Missing, duplicate, mismatched, or in-sample primary data is a formal-run error.

An in-sample result may be generated only for debugging with role `diagnostic_in_sample`; it has a distinct output
root and cannot populate comparison tables or response evidence.

## Initialization And Metrics

`MODEL-MT5-BASE` can initialize new control runs after data delivery acceptance. The local mT5 export remains a
receipt-pending controlled alternative. The incoming historical `log/archived/` bundle remains entirely out of
scope until stable receipt/audit; directory names and partial evaluation files are not model provenance.

Use one receipt-bound language asset for the main matrix. Any initialization study is a separately labelled,
same-budget comparison. Production translation metrics must freeze BLEU, ROUGE, SBERT, and SimCSE versions and
tokenization/normalization before reporting; all predictions/references and per-sample metric inputs are retained.
The existing character metric is orchestration validation only.

## Execution Sequence

```text
accepted CSL-Daily source + receipt of any existing historical pose/signal/feature candidates
-> canonical pose-only delivery and small CSLD-WL-01 smoke
-> accepted skeleton-simulation reconstruction delivery
-> CubeNet small smoke and frozen formal checkpoint(s)
-> cross-fitted predicted-pose export and CSLD-WL-02 smoke
-> frozen production pose-only metric protocol and formal first-loop controls
-> optional checkpoint-bound feature export and CSLD-WL-03 fusion smoke
-> shortest labelled historical replay, retaining legacy inputs unchanged
-> historical checkpoint/config/evaluation audit, reported separately
```

The historical replay and new canonical control use separate output roots and receipts. The latter does not become
a successful reproduction merely because it is trained from the same captions; historical metric reproduction also
needs accepted former configuration, split, prediction, and evaluator evidence.

## Paper Boundary

CSL-Daily numbers may support a clearly labelled controlled pipeline analysis or explain a historical convention.
They cannot support statements of real-world radar fidelity, off-axis robustness, object occlusion robustness, or
new-user generalization. Paper promotion still requires the paper workspace's evidence registry and reviewer
closure gate.
