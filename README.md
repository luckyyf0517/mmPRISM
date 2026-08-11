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
- a single `mmprism` CLI surface;
- dependency-light unit tests;
- revision and reviewer-evidence management under `paper/manager/`.

Not yet implemented in the canonical package:

- radar simulation and FMCW processing;
- CubeNet/OmniHand training and evaluation;
- WaveLLM/mT5 training, generation, and evaluation;
- remaining production data adapters, distributed prediction/checkpoint writers, and GPU integration tests.

Do not interpret the foundation scaffold as a reproducible release of the paper results yet. Read `paper/manager/dashboard.md` for the current blockers and work order.

## Quick Start

The canonical environment uses UV, Python 3.12, and the committed `uv.lock`. On this project host, the research profile targets PyTorch CUDA 12.8 for the installed A100 GPUs.

```bash
scripts/bootstrap_env.sh research
uv run mmprism doctor
uv run mmprism config configs/examples/pose_smoke.yaml
uv run mmprism plan configs/examples/pose_smoke.yaml
uv run mmprism manifest tests/fixtures/manifests/pose_smoke.jsonl
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
```

## Canonical Layout

```text
configs/                 validated experiment configuration
docs/architecture/       package boundaries and rebuild design
src/mmprism/
  contracts/             data and artifact schemas
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
paper/manager/            revision control plane and evidence tracking
paper/manuscript/         private Overleaf Git submodule
```

Architecture rules are defined in `AGENTS.md` and `docs/architecture/README.md`.

## Legacy Code

The root `run_*.py` files, `config/`, and the original modules under `src/data`, `src/fmcw`, `src/model`, `src/eval`, `src/scripts`, and `src/utils` are retained only to audit the original submission.

- New code must not import them.
- They will not receive feature work or compatibility shims.
- The reviewer release will contain only validated canonical entry points.
- They will be archived or removed after historical evidence is extracted.

## Paper Revision

- Management entry: `paper/manager/README.md`
- Current dashboard: `paper/manager/dashboard.md`
- Reviewer comments: `paper/manager/reviews/`
- Architecture status: `paper/manager/current/architecture_status.md`
- Data rebuild status: `paper/manager/current/data_status.md`
- Overleaf workflow: `paper/manager/current/operator_guide.md`

## License

The publication license has not yet been approved by the authors. No license should be inferred from earlier repository documentation. License selection is tracked as `OPS-REV-002` in the revision workspace.
