# mmPRISM

Geometry-guided millimeter-wave perception for continuous sign language understanding.

Status: `major_revision_greenfield_rebuild`

This repository is being rebuilt for the Nature Communications major revision. Data preparation and model training will be rerun from scratch using the canonical package under `src/mmprism/`. Historical scripts remain temporarily available for forensic review, but they are not supported as the new execution path.

## Current Scope

Implemented foundation:

- standard Python packaging through `pyproject.toml`;
- strict experiment configuration with early validation;
- environment-based path injection without machine-specific paths;
- versioned JSONL sample-manifest contract;
- side-effect-free run planning, runtime provenance reporting, and atomic formal-run initialization;
- SHA-256-bound run inputs plus versioned finite-metric artifacts;
- explicit radar, pose, feature, and caption tensor contracts;
- a NumPy range-Doppler v1 transform with analytic signal tests;
- a canonical CubeNet/OmniHand pose regressor with independently ablatable attention,
  mask-aware temporal aggregation, and versioned metric-pose evaluation;
- a checksum-bound radar-cube/metric-pose adapter with variable-time collation;
- single-device formal OmniHand train/evaluate services with Safetensors checkpoints, streaming
  sample predictions, history, runtime/performance identity, and count-weighted metrics;
- pinned, checksummed SBERT and SimCSE evaluator-model acquisition;
- pinned mT5-base acquisition plus a two-step pose/radar/fusion train-generate GPU smoke;
- strict model-ready translation manifests plus single-device formal WaveLLM train/evaluate services
  with adapter-only Safetensors, sample predictions, runtime/performance identity, and character metrics;
- a single `mmprism` CLI surface;
- dependency-light unit and contract tests.

Not yet implemented in the canonical package:

- antenna calibration, beamforming, physical radar axes, and radar simulation;
- real-data OmniHand training validation, resumable/distributed training, and distributed prediction;
- real-data WaveLLM validation, resumable/distributed training, rank-safe prediction aggregation, and
  production paper metrics;
- remaining production data adapters, distributed prediction/checkpoint writers, and GPU integration tests.

Do not interpret the current package as a reproducible release of the paper results yet. Range-Doppler
processing is independently tested, but the complete 4D cube remains blocked on acquisition, channel-map,
virtual-array, and calibration evidence.

## Quick Start

The canonical environment uses UV, Python 3.12, and the committed `uv.lock`. On this project host, the research profile targets PyTorch CUDA 12.8 for the installed A100 GPUs.

```bash
scripts/bootstrap_env.sh research
uv run mmprism doctor
uv run mmprism config configs/examples/pose_smoke.yaml
uv run mmprism plan configs/examples/pose_smoke.yaml
uv run mmprism manifest tests/fixtures/manifests/pose_smoke.jsonl
uv run mmprism models-plan configs/models/evaluation_models_v1.yaml
# Requires a clean Git worktree and the train profile.
MMPRISM_DEVICE=cuda:0 scripts/run_omnihand_smoke.sh
# Expected to return 1 while the listed reviewer-release blockers remain.
uv run mmprism release-audit configs/release/reviewer_release_v1.yaml
uv run pytest
```

Initialize a formal provenance envelope without running a model:

```bash
uv run mmprism run-init configs/examples/pose_smoke.yaml \
  --input manifest:data_manifest=tests/fixtures/manifests/pose_smoke.jsonl
```

This command writes resolved config, environment/Git state and input hashes atomically. Canonical
OmniHand and WaveLLM train/evaluate services use that envelope directly; data preparation, resume, and
distributed orchestration remain under construction.

Profiles:

- `foundation`: package and development checks without ML extras.
- `research`: training, radar, evaluation, tracking, and visualization dependencies.
- `distributed`: research profile plus optional DeepSpeed support.

Do not use the legacy `requirements.txt` for the canonical package. Update `pyproject.toml`, run `uv lock`, and commit the resulting `uv.lock` whenever dependencies change.

Machine-specific roots are injected through environment variables:

```bash
export MMPRISM_DATA_ROOT=/path/to/mmprism-data
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
export MMPRISM_CACHE_ROOT=/path/to/mmprism-cache
export MMPRISM_MODEL_ROOT=/path/to/mmprism-models
```

## Evaluation Models

The semantic evaluators are pinned to immutable Hugging Face commits in
`configs/models/evaluation_models_v1.yaml`. The canonical downloader acquires both SimCSE and SBERT,
materializes only the declared files, records per-file SHA-256 checksums, and atomically promotes each
verified asset. It never resolves the moving `main` branch.

```bash
export MMPRISM_MODEL_ROOT=/path/to/mmprism-models
scripts/download_models.sh
uv run --frozen --extra evaluation mmprism models-smoke \
  configs/models/evaluation_models_v1.yaml \
  --output-root "${MMPRISM_MODEL_ROOT}" \
  --device cpu
```

`models-plan` is dependency-light and network-free. `models-download` reuses a complete verified
asset, but refuses a corrupt or unexpected existing directory instead of overwriting it. Download
resume state remains under `${MMPRISM_MODEL_ROOT}/.cache/huggingface`; consumers use only the
materialized `simcse/` and `sbert/` directories plus their manifests.
The wrapper defaults `HF_HUB_DISABLE_XET=1` after an observed stalled Xet transfer; explicitly set it
to `0` before invocation if a deployment has a validated Xet path.

## Pose Reconstruction

The canonical reconstruction path consumes non-negative radar-cube power with axes
`[batch,time,doppler,range,azimuth,elevation]` and emits metric dual-hand joints with shape
`[batch,2,24,3]`. CubeNet uses depthwise-separable 3D residual blocks, optional PAFPN aggregation,
and separately configurable channel, spatial, and squeeze-excitation attention. The temporal path is
mask-aware and combines CLS, valid-frame mean, and learned-attention summaries.

Run the deterministic two-step engineering smoke on an available CUDA device:

```bash
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
MMPRISM_DEVICE=cuda:0 scripts/run_omnihand_smoke.sh
```

The smoke checks spatial/temporal/head gradients and parameter updates, single-frame inference,
padding invariance, sample-level MPJPE, wrist-relative MPJPE/PCK, runtime provenance, and peak CUDA
memory. Synthetic cubes and targets are used only to validate the executable model boundary; their
metrics are not paper results. Physical 4D-cube reproduction remains gated on acquisition and
calibration provenance.

Run formal training after producing disjoint model-ready manifests:

```bash
export MMPRISM_DATA_ROOT=/path/to/model-ready-data
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
export MMPRISM_TRAIN_MANIFEST=/path/to/train.jsonl
export MMPRISM_VALIDATION_MANIFEST=/path/to/validation.jsonl
export MMPRISM_SPLIT_ASSIGNMENTS=/path/to/split_assignments.jsonl
scripts/run_omnihand_train.sh
```

Run `mmprism prepare` with the same experiment config, manifests, split assignment file, and explicit
manifest-to-split bindings before starting a formal job. The preflight requires a clean Git worktree,
validates every input hash and contract, checks the planned destination without creating it, and emits
one JSON report to stdout. `omnihand-train` independently enforces the same split membership and records
the assignment file in `inputs.json`; it also verifies every manifest-bound array checksum. It
writes a Safetensors checkpoint plus metadata, resolved task/runtime configuration, history,
validation predictions, and `mmprism.pose_metric.dual_hand_metric_v1` metrics. `omnihand-evaluate`
requires the checkpoint weights and metadata as separate hashed inputs and rejects any checksum,
model-config, unit, or coordinate-frame mismatch. The example is a two-step engineering recipe, not a
paper training protocol.

Every fully completed epoch also publishes an immutable `training-state.epoch-NNNNN.json` and
`.safetensors` pair. Resume by setting both `MMPRISM_RESUME_STATE_METADATA` and
`MMPRISM_RESUME_STATE_TENSORS` before invoking the same script. The state restores model, AdamW,
GradScaler, RNG, loader generator, history, and global step only when all Git/data/split/model/runtime
bindings match. Epoch and step targets may only stay fixed or increase; an interrupted partial epoch is
rerun from the preceding complete boundary.

## Language Model Support

mT5 is the sole canonical language-generation backend. The pinned engineering smoke exercises the
dual-hand ST-GCN pose path, radar-feature projection, confidence-aware fusion, real mT5 forward/backward,
optimizer updates and beam generation:

```bash
export MMPRISM_MT5_MODEL_ROOT=/path/to/mt5-assets
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
scripts/download_mt5.sh
MMPRISM_DEVICE=cuda:0 scripts/run_mt5_smoke.sh
```

The smoke freezes the mT5 backbone and updates only the canonical adapters. It proves runnable module
integration, not paper-result reproduction or the final training protocol.

Run the formal single-device path after producing sequence-disjoint model-ready manifests:

```bash
export MMPRISM_MT5_MODEL_ROOT=/path/to/mt5-assets
export MMPRISM_DATA_ROOT=/path/to/model-ready-data
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
export MMPRISM_TRAIN_MANIFEST=/path/to/train.jsonl
export MMPRISM_VALIDATION_MANIFEST=/path/to/validation.jsonl
export MMPRISM_SPLIT_ASSIGNMENTS=/path/to/split_assignments.jsonl
scripts/run_wavellm_train.sh
```

`wavellm-train` verifies every manifest-bound pose, confidence, and frame-mask array. The v2 input contract
selects either `pose_only`, which forbids radar features, or `pose_plus_radar_feature`, which additionally
requires one radar-feature sequence. It rejects sample/sequence leakage, requires each manifest sample to match
the registered split assignment, and writes an adapter-only checkpoint when mT5 is frozen. The separate
`wavellm-evaluate` command re-registers weights and metadata as hashed inputs and rejects checksum,
model/task-config, input-mode, model-asset, unit, coordinate-frame, or tensor-inventory drift before generation.
`wavellm-train` uses the same paired epoch-state resume contract and, for adapter-only training, stores
the trainable adapter model state without duplicating the frozen mT5 backbone.
The included character metric is an orchestration protocol; full fine-tuning, real-data validation,
distributed model/checkpoint execution, and production BLEU/ROUGE/semantic metrics remain under construction.
Historical alternative backend definitions are excluded from the release rather than exposed as
unsupported claims.

## Project Navigation

- Current project truth: `docs/authority/00_INDEX.md`
- Business workflow status and operations: `workspaces/*/docs/authority/00_INDEX.md`
- Proposed code, interface, contract, or ownership changes: `openspec/changes/`
- Dated project evidence: `docs/logs/`

Workspaces describe executable business workflows. They reference shared code, configuration, scripts, and
tests at the repository root rather than duplicating those assets. Historical links under
`docs/architecture/` and `paper/manager/` remain as compatibility entrypoints.

## Canonical Layout

```text
configs/                 validated experiment configuration
docs/authority/          current project-level boundaries and shared contracts
docs/logs/               immutable project-level history and evidence
workspaces/              business workflow Authority, operations, and Logs
openspec/                proposed and archived implementation/contract changes
src/mmprism/
  contracts/             data and artifact schemas
  assets/                pinned external model acquisition and verification
  config/                strict configuration loading
  data/                  manifest-backed datasets and splits
  radar/                 simulation and signal processing
  models/                pure reconstruction and language models
  training/              training and distributed orchestration
  evaluation/            versioned pose and language metrics
  artifacts/             run metadata, predictions, and paper exports
  runtime/               paths, environment, seeds, devices, run plans
  cli.py                  user-facing command composition
tests/                    unit, contract, integration, and fixtures
```

The public package boundary is enforced by the versioned release audit. Internal agent guidance,
revision management, manuscript sources, credentials, architecture work logs, and legacy forensic code
are not part of the reviewer release inventory.

## Legacy Code

The development repository retains the root `run_*.py` files, `config/`, and original modules under
`src/data`, `src/fmcw`, `src/model`, `src/eval`, `src/scripts`, and `src/utils` only to audit the original
submission. They are absent from the reviewer release selection.

- New code must not import them.
- They will not receive feature work or compatibility shims.
- The reviewer release will contain only validated canonical entry points.
- They will be archived or removed after historical evidence is extracted.

## Release Audit

`mmprism release-audit` builds the public inventory from Git-tracked files, hashes every selected file,
checks required and forbidden paths, scans for local absolute paths and credentials, validates the console
entrypoint, and statically audits canonical imports for missing modules, legacy dependencies, and cycles.
The reviewer profile remains intentionally failing until all required runnable examples, model download
instructions, and the author-approved license are present.

## License

The publication license has not yet been approved by the authors. No license should be inferred from earlier repository documentation. License selection is tracked as `OPS-REV-002` in the revision workspace.
