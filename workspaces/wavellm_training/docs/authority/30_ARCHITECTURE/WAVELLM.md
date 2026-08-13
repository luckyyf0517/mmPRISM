# Canonical WaveLLM Translation

Status: current
Owner: WaveLLM training lane
Authority scope: The WaveLLM training and evaluation boundary represented by this page.
Last reviewed: 2026-08-13

## Boundary

WaveLLM starts from temporally aligned, model-ready geometry and optional checkpoint-bound radar features. It does
not infer file relationships, reconstruct poses, resolve model paths, parse CLI input, or write artifacts.

```text
pose             [batch,time,2,24,3] float32, metres
pose confidence  [batch,time,2,24]   float32, [0,1]
radar features   [batch,time,F]       float32, fusion mode only
frame mask       [batch,time]         bool/long
caption          non-empty text
```

The v2 JSONL+NPY manifest owns input mode, coordinate frame, units, shapes, dtypes, checksums, and inline caption.
It requires a frame mask in both modes. Variable-length samples are zero-padded during collation and every model
path receives the explicit frame mask. `pose_only` accepts pose, confidence, mask, and caption only; it never
loads, fabricates, projects, or persists a radar feature. Its final Parquet delivery has no `radar_feature` column
or feature dimension and its reader returns `None`. `pose_plus_radar_feature` additionally requires one declared
feature sequence and its protocol. Supplying zero radar features to the fusion model is not a valid pose-only
experiment.

## Model

- Pose path: dual-hand ST-GCN maps metric geometry to one embedding per frame in both modes.
- Pose-only path: feeds pose embeddings directly to mT5 and has no radar projector or fusion parameters.
- Fusion path: a feature projector maps the declared feature dimension to mT5 hidden size; mean pose confidence
  and a learned gate then combine pose and radar embeddings.
- Prompt: token embeddings are concatenated with fused frame embeddings and their attention masks.
- Language model: the sole supported backend is a pinned local mT5 asset.

The model class contains no path resolution, logging, checkpoint writing, or CLI parsing. The mT5
backbone can be frozen for adapter training or included in a full-model checkpoint through explicit
configuration. The engineering recipe freezes mT5 and is not the final scientific protocol.

## Revision Language Initialization

A historical WaveLLM bundle is staged under `/mnt/gfs/yanyifan/mmPRISM/log/archived/`. It may include the former
cam-pose checkpoint and historical hand-pose encoder, but its run labels and staged files do not establish that fact.
No inbound file may be loaded, converted, moved, or used for initialization until the
[receipt and audit](../../logs/2026/08/20260813_HISTORICAL_WAVELLM_STAGING_HANDOFF.md) establish a stable inventory,
checksums, format/world-size completeness, metadata, and a controlled load report.

The retained local fallback is a CSL-News-derived mT5-only export, recorded as `MODEL-MT5-CSLNEWS-HISTORICAL-V1` and
smoke-verified in the [historical initialization log](../../logs/2026/08/20260812_HISTORICAL_MT5_INITIALIZATION_SMOKE.md).

The export initializes only `GeometryGuidedMT5.language_model`. The current dual-hand ST-GCN pose encoder, radar
projector, and confidence-aware fusion are deliberately new components and must be freshly initialized and trained on
frozen CSL-Daily or later real-data manifests. No legacy module is imported and no compatibility shim is permitted.

Before a formal run, the chosen asset must have a checksum-bound, immutable receipt. The mT5 fallback needs a
local-derived asset receipt/import; an accepted historical bundle needs an additional DeepSpeed/Lightning format and
world-size audit. A remote Hugging Face asset configuration is not a substitute: the fallback's weight SHA-256 differs
from the pinned official `google/mt5-base` asset.

Controlled architecture, DA, stress, and sim2real comparisons must register the same accepted language initialization
or explicitly state an alternative initialized under the same data and optimization budget. They must not claim that
historical pose/radar modules, checkpoint metrics, split, or predictions were reused unless the asset-specific audit
accepts and documents them. Full CSL-News source reconstruction is P1/future provenance work and is not a training
gate for CSL-Daily.

## Data And Split Gates

`mmprism.sign_language_translation.sample_v2` requires local relative `.npy` references for pose, confidence,
and frame mask. Fusion records additionally require radar features and the radar-feature protocol; pose-only
records forbid both. Construction validates metadata; formal runs additionally verify every array checksum and
load-time shape, dtype, finite value, confidence range, and non-empty-mask contracts. A training run rejects
input-mode, feature/joint dimension, coordinate-frame, sample-ID, sequence-ID, or maximum-frame drift.

## CSL-Daily Control Matrix

The controlled CSL-Daily route consists of camera pose -> text (`pose_only`), cross-fitted predicted mmWave-pose
-> text (`pose_only`), and that same predicted pose plus a checkpoint-bound CubeNet frame feature -> text
(`pose_plus_radar_feature`). The latter feature is an immutable producer artifact binding source/split/checkpoint,
frame identity/mask, shape/dtype/checksum and model fingerprint; it is never inferred by path substitution.
Training-row predicted inputs are cross-fitted, so their CubeNet producer did not train on those rows. A diagnostic
in-sample result is separately labelled and cannot serve as the primary comparison. The legacy CSL-Daily validation
and test JSON are byte-identical, so they only support an explicitly labelled historical replay, not a new test.
See the [CSL-Daily training operation](../40_OPERATIONS/CSL_DAILY_TRAINING.md).

## Formal Train And Evaluate

`mmprism wavellm-train` requires a clean Git commit, a sign-language-translation experiment config,
strict task and model-asset configs, separate train/validation manifests, and one hashed canonical split
assignment file. It verifies every sample's declared train/validation membership in addition to the
sample/sequence leakage gates. It writes:

- generic resolved experiment, environment, input hashes, and run lifecycle metadata;
- `wavellm.resolved.json` and `wavellm.runtime.json`;
- atomic `checkpoint.safetensors` plus checksum-bound `checkpoint.json`;
- immutable JSON/Safetensors training-state pairs for every fully completed epoch;
- `history.json`, `performance.json`, `predictions.jsonl`, and `metrics.json`.

For a frozen mT5 backbone, the checkpoint contains only non-language-model tensors and records
`scope=adapter_only`. Metadata binds the input mode, base-model repository/revision and asset/collection
manifests, model and task fingerprints, coordinate frame, units, Git commit, runtime, and all input hashes.
Evaluation rejects a checkpoint with a different mode before tensor loading; pose-only adapter inventories cannot
contain radar projector or fusion tensors.

`mmprism wavellm-evaluate` registers the manifest, split assignments, weights, metadata, configs, and model
manifests as independent hashed inputs. Before state loading it validates evaluation split membership,
weight checksum/format/scope, complete model and task fingerprints, model-asset identity, units, coordinate
frame, and exact adapter tensor inventory.

`wavellm-train` uses the shared completed-epoch resume contract. It restores AdamW, GradScaler, all RNG
states, loader generator, history, global step, and the configured model checkpoint scope. Adapter-only
states contain the non-language-model tensors and bind the exact frozen mT5 asset identity. Resume
requires exact Git/data/split/model/runtime compatibility and permits only nondecreasing epoch/step
targets. A deterministic CPU integration test proves final adapter tensor and history equality for
uninterrupted versus segmented two-epoch training.

Formal train/evaluate also detect the `torchrun` environment and share the OmniHand distributed lifecycle: rank zero
owns run initialization/finalization and checkpoint publication; model-state hashes must agree across ranks; exact
no-padding prediction shards are aggregated against the full sample set; and character-metric states and performance
are merged across ranks. DDP resume is rejected until per-rank RNG and sampler state have a complete contract.

## Metric Protocol

`mmprism.language_metric.character_v1` stores sample-level exact match, Unicode code-point Levenshtein
distance, and reference/prediction character counts. Summary exact match is count-weighted by samples;
character error rate is total edit distance divided by total reference code points. This minimal protocol
validates formal orchestration. Production paper evaluation still requires frozen BLEU/ROUGE/semantic
protocols reconciled with the manuscript and historical evaluator definitions.

## Evidence Boundary

Clean commit `e31000b` completed A100/BF16 train, adapter checkpoint, reload, prediction, and standalone
evaluation on a deterministic 4/2-record synthetic fixture. An external 250-gate audit verified all
hashes, split separation, tensor inventory, prediction replay, metric recomputation, runtime, performance,
and temporary-file gates. Synthetic outputs and metric values are engineering evidence only.

Real pose/radar feature preparation remains blocked on the upstream OmniHand/radar provenance chain. Historical
checkpoint audit is also open and must close before this asset is used for paper-facing comparisons.
WaveLLM-specific multi-process/NCCL validation, production metrics, and real-data validation remain open; the shared
distributed implementation, prediction aggregation, and single-process epoch-boundary resume are implemented.
