# Canonical WaveLLM Translation

Status: current
Owner: WaveLLM training lane
Authority scope: The WaveLLM training and evaluation boundary represented by this page.
Last reviewed: 2026-08-12

## Boundary

WaveLLM starts from temporally aligned, model-ready geometry and radar features. It does not infer file
relationships, reconstruct poses, resolve model paths, parse CLI input, or write artifacts.

```text
pose             [batch,time,2,24,3] float32, metres
pose confidence  [batch,time,2,24]   float32, [0,1]
radar features   [batch,time,F]       float32
frame mask       [batch,time]         bool/long
caption          non-empty text
```

The manifest owns the sample and radar-feature protocol, coordinate frame, units, shapes, dtypes,
checksums, and inline caption. Variable-length samples are zero-padded during collation and every model
path receives the explicit frame mask.

## Model

- Pose path: dual-hand ST-GCN maps metric geometry to one embedding per frame.
- Radar path: a feature projector maps the declared feature dimension to mT5 hidden size.
- Fusion: mean pose confidence and a learned gate combine pose and radar embeddings.
- Prompt: token embeddings are concatenated with fused frame embeddings and their attention masks.
- Language model: the sole supported backend is a pinned local mT5 asset.

The model class contains no path resolution, logging, checkpoint writing, or CLI parsing. The mT5
backbone can be frozen for adapter training or included in a full-model checkpoint through explicit
configuration. The engineering recipe freezes mT5 and is not the final scientific protocol.

## Revision Semantic Initialization

The paper revision uses the original-submission cam-pose WaveLLM checkpoint, reported as trained from the first
approximately 100 CSL-News archives, as a shared immutable semantic initialization. It is a historical paper asset,
not a canonical formal run, until its provenance is sufficiently recovered. Intake must establish at minimum:

- checkpoint bytes and SHA-256;
- mT5 and tokenizer identity/revision plus model structure and resolved configuration;
- strict or explicitly controlled state loading with a complete missing/unexpected-tensor report;
- compatibility with the historical pose representation used by the checkpoint;
- the recoverable caption mapping, sample/sequence list, and train/validation/test split for the first-100 scope;
- an independent fixed-holdout evaluation with sample-level predictions and versioned metrics.

All applicable revision architecture, DA, stress, and sim2real experiments must register the same checkpoint ID and
hash. If an alternative architecture cannot consume some architecture-specific tensors, it must retain the identical
semantic backbone initialization, declare every unmatched tensor, and keep data, optimization budget, and evaluation
protocol controlled. It may not silently substitute a newly full-data-pretrained language base.

A new 436-archive CSL-News source intake or completed manifest would not trigger training by itself. Full-data training
remains P1 or future work unless the historical checkpoint cannot be loaded or audited, has split leakage, is
incompatible with the required historical pose contract, or controlled evaluation demonstrates that the semantic base
masks the sim2real differences under study.

## Data And Split Gates

`mmprism.sign_language_translation.sample_v1` requires local relative `.npy` references for pose,
confidence, radar features, and optional masks. Construction validates metadata; formal runs additionally
verify every array checksum and load-time shape, dtype, finite value, confidence range, and non-empty-mask
contracts. A training run rejects overlapping sample IDs, overlapping non-null sequence IDs, feature or
joint dimension drift, coordinate-frame drift, and sequences longer than the configured maximum.

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
`scope=adapter_only`. Metadata binds the base-model repository/revision and asset/collection manifests,
model and task fingerprints, coordinate frame, units, Git commit, runtime, and all input hashes.

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
DDP model execution, distributed checkpoint aggregation, production metrics, and real-data validation
remain open; prediction aggregation and single-process epoch-boundary resume are implemented.
