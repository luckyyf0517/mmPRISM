# CSL-Daily Legacy-Path Replay Runbook

Status: current
Owner: Data rebuild lane
Authority scope: Isolated replay of the legacy CSL-Daily camera-pose-to-WaveLLM route.
Last reviewed: 2026-08-13

## Purpose

The immediate goal is operational, not a paper result: identify and execute the
smallest data-backed vertical slice of the former CSL-Daily workflow. It
answers whether the recovered raw source, labels, models, simulator semantics,
and current package interfaces can produce an end-to-end prediction. Gates 2--5
remain operator-review actions; this runbook does not authorize their execution
by itself.

This is deliberately separate from the formal CSL-Daily reconstruction in
[CSL-Daily reproduction](CSL_DAILY_REPRODUCTION.md).  The formal route must
persist pre-beamforming synthetic FMCW and use a frozen raw-radar/Parquet
delivery (`DEC-049`).  This runbook may use a temporary direct-cube fixture only
to find compatibility gaps.  Its artifacts are `diagnostic_legacy_replay` and
must not enter paper tables, evidence, or a formal training input.

## Recovered Legacy Semantics

The available historical code establishes these intended transformations:

```text
CSL-Daily sentence frames + csl2020ct_v2.pkl
-> RTMW3D-L full-image inference, confidence < 0.5 -> NaN
-> sequence-global shoulder depth centering
-> 17 body + 42 hand joints -> [T,2,24,3] dual-hand pose
-> skeleton densification, temporal Gaussian smoothing, 30 Hz -> 10 Hz
-> IWR1843 point-reflector FMCW simulation
-> range/Doppler/beamforming power cube -> CubeNet/OmniHand
-> predicted dual-hand pose -> HandPoseEncoder + mT5 -> Chinese text
```

Two historical details require empirical confirmation rather than assumption:

1. `run_simulation.py` emits aggregated two-view signals and is CSL-News
   archive-oriented, so it is not evidence that those files were the input of
   the CSL-Daily CubeNet job.
2. The legacy CSL-Daily OmniHand configuration has `use_simulator: true`; its
   forward method therefore generates raw FMCW online from pose and derives its
   cube inside the model.  The legacy training split's `mmwave` field was not
   necessarily consumed in that run.

The initial replay should therefore prove current offline simulation and
CubeNet ingestion first.  Do not claim exact historical numerical equivalence
until a received historical pose/signal/checkpoint/config tuple establishes it.

## Inputs Already Observed

| Need | Observed location | State | Replay use |
|---|---|---|---|
| CSL-Daily frames | `external/csl_daily/csl_daily_original_20260812/CSL-Daily/sentence/frames_512x512/` | Upload is in final read-only rsync verification | Required after receipt |
| CSL-Daily videos | `.../sentence/video/` | Same upload candidate | Read-only spot check only |
| Official labels | `.../sentence_label/csl2020ct_v2.pkl` and split metadata | Present | Required for captions/identity binding |
| Historical split descriptors | `dataset/csl-daily/{train,val,test,all}.json` | Present; val and test are byte-identical | Historical replay only |
| RTMW3D-L | `external/models/rtmw3d/.../794dbc78...pth` plus pinned MMPose cache | Expected by config | Validate with one GPU sequence |
| mT5 base | registered `MODEL-MT5-BASE` asset | Evidence-ready | Required for current WaveLLM smoke |
| Historical WaveLLM runs | `log/archived/` | Receipt/audit pending | Do not load for this replay |
| Historical evaluator mirrors | `pretrained_models/{sbert,simcse}` | Preservation candidate | Not required for the first generated-text check |

## Execution Plan

### Gate 0: Source Acceptance

Wait for the final `rsync -n` verification to exit.  Confirm no regular
transfer process remains and two short inventories are stable.  Then run the
read-only receipt and validation:

```bash
export MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM
uv run mmprism csl-daily-source-receipt \
  --source-root "$MMPRISM_DATA_ROOT/external/csl_daily/csl_daily_original_20260812" \
  --receipt-root "$MMPRISM_DATA_ROOT/interim/csl_daily/source_receipts" \
  --source-id DATASET-CSL-DAILY \
  --legacy-split-root "$MMPRISM_DATA_ROOT/dataset/csl-daily" \
  --stability-wait-seconds 60
uv run mmprism csl-daily-source-receipt-validate RECEIPT_ROOT
```

Success: immutable receipt, source inventory, label identity, and legacy split
hashes are recorded.  Failure: stop; do not annotate or train from the source.

### Gate 1: Label-to-Frame Binding

Read `csl2020ct_v2.pkl` through the existing CSL-Daily annotation parser and
compare a small deterministic set against frame directories and the legacy
JSON descriptors.  Record the mapping rules, exact caption text, signer ID,
frame count, and missing/duplicate count.  Choose a bounded replay cohort:

- smoke: 8--16 complete sequences from at least two signers;
- diagnostic mini-train: 64--128 train sequences plus a separate 16--32
  historical-validation sequences;
- never call the legacy `val`/`test` duplicate pair an independent test.

Success: every selected replay row maps to one source sequence and non-empty
caption.  Failure: repair only the adapter/manifest mapping; never rewrite
the legacy JSON or source labels.

### Gate 2: Camera-Pose GPU Smoke

Run the canonical RTMW3D adapter on the chosen smoke cohort under Slurm.  The
current prepared pilot is `scripts/slurm/csl_daily_annotation_pilot.sbatch`;
override its `MMPRISM_CSL_DAILY_PILOT_MAX_SEQUENCES` to the bounded cohort if
needed.  It writes versioned sidecars under `interim/`, never under `external/`.

Inspect representative native and canonical overlays and record:

- source-frame and output-frame coverage;
- pose shape `[T,2,24,3]`, dtype, finite/NaN and confidence rates;
- depth-centering and left/right identity behavior;
- per-sequence failures/quarantine reason.

Success: selected sequences produce readable pose sidecars with reviewable
overlays.  Failure categories are source decode, model bootstrap, mapping,
pose quality, or resource scheduling; fix only the category observed.

### Gate 3: Synthetic-Radar/Cube Smoke

For the same accepted pose rows, run the existing direct-cube materializer on
a *small temporary diagnostic output root*.  It is only a numerical probe of
the recovered skeleton simulator and processor.  Check simulator input/output
shape, finite non-negative cube values, coordinate scale, and runtime/storage
cost.  Do not run it corpus-wide or turn its cubes into a final Parquet
delivery.

In parallel, compare one raw-FMCW frame and processor result against the
frozen legacy-equivalence fixture.  Any material discrepancy blocks calling
the path a legacy replay and is recorded with the simulator/config fingerprint.

Success: a current CubeNet can consume one replay batch without shape or
coordinate errors.  Failure: distinguish pose normalization, skeleton
densification, simulator, processor, or model-adapter mismatch.

### Gate 4: OmniHand Mini-Train And Predicted-Pose Export

Use the diagnostic cube fixture to run the smallest current `omnihand-train`
protocol that produces a checkpoint, reloads it, evaluates it, and exports
predicted poses for the held-out replay rows.  This checks the training
lifecycle, not reconstruction quality.  Retain resolved config, Git state,
input manifest/split hashes, step losses, prediction IDs and metrics.

This mini-run can use in-sample predicted pose solely for plumbing.  It must
be labelled `diagnostic_in_sample`; formal predicted-mmWave-pose experiments
require the cross-fitted export gate in the canonical operation.

Success: exact held-out row coverage and checkpoint reload/evaluation pass.
Failure: repair the current manifest/model interface; do not use historical
checkpoints as an undocumented substitute.

### Gate 5: WaveLLM Pose-Only Mini-Train And Generation

First run camera-pose text generation (`CSLD-WL-01` shape) using the same
bounded cohort and accepted mT5 base.  Then, only if Gate 4 produced complete
diagnostic predictions, replace camera poses with predicted poses for a second
plumbing run.  Keep feature input disabled for both runs.

For each run save sample IDs, references, generated text, loss and the
generation configuration.  The first milestone is one non-empty generated
Chinese output linked to a source sequence; BLEU/ROUGE/SBERT/SimCSE are added
only after the evaluator asset receipt and metric protocol are frozen.

Success: train -> checkpoint -> reload -> generate works from both pose
sources, with exact prediction/reference membership.  Failure: classify as
caption binding, tokenizer/asset, pose shape/mask, checkpoint lifecycle, or
generation issue.

### Gate 6: Decision

Publish one short replay report listing all artifact hashes and the first
failing gate, or record a successful diagnostic chain.  Then decide:

- if Gates 0--5 pass, move to the formal raw-FMCW runtime-adapter/Parquet path
  and a clean first-loop training protocol;
- if only Gates 0--2 pass, prioritize raw pose QC/annotation work;
- if Gate 3 fails, prioritize simulation/processor interface recovery;
- if Gate 4 or 5 fails, prioritize the corresponding current training adapter;
- do not begin feature fusion, full CSL-News reconstruction, or broad code
  refactoring before this decision.

## Artifact And Isolation Rules

- Put replay artifacts under a new versioned root such as
  `interim/csl_daily/legacy_replay/<run-id>/` and mark every record
  `diagnostic_legacy_replay` or `diagnostic_in_sample`.
- Source under `external/`, legacy mirror content, and incoming historical
  checkpoints remain read-only.
- The current `csl-daily-simulate` command persists cubes; it may be used only
  for Gate 3 temporary diagnostics until the raw-FMCW materializer and runtime
  adapter exist.
- Do not use historical `val.json` and `test.json` as separate benchmark sets.
- Do not resume or load `log/archived/` until the historical-bundle receipt,
  world-size/metadata audit, and controlled load are completed.

## Today's Definition Of Done

Today is complete when the source receipt has passed and the following
diagnostic chain is either demonstrated or stopped at a documented first
failure:

```text
one source sequence + caption
-> RTMW3D dual-hand pose sidecar and visual review
-> simulated radar/cube batch consumed by CubeNet
-> checkpointed OmniHand mini-run with predicted pose
-> checkpointed pose-only mT5 mini-run with saved generation
```

This is an execution-confidence milestone.  It neither reproduces the
manuscript numbers nor closes reviewer evidence requirements.
