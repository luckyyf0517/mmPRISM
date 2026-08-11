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
- pinned, checksummed SBERT and SimCSE evaluator-model acquisition;
- a single `mmprism` CLI surface;
- dependency-light unit and contract tests.

Not yet implemented in the canonical package:

- antenna calibration, beamforming, physical radar axes, and radar simulation;
- CubeNet/OmniHand training and evaluation;
- WaveLLM/mT5 training, generation, and evaluation;
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
`prepare`, `train` and `evaluate` services are still under construction.

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

## Canonical Layout

```text
configs/                 validated experiment configuration
docs/architecture/       package boundaries and rebuild design
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
