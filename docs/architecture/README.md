# mmPRISM Rebuild Architecture

Status: `foundation_v1`
Last Updated: `2026-08-11`

## Direction

The rebuilt package is the only implementation target. Legacy scripts and modules remain available for forensic comparison, but there is no runtime compatibility requirement and no new code may depend on them.

## Dependency Direction

```text
contracts <- config
contracts <- data <- radar
contracts <- models
models + data + evaluation <- training
config + runtime + training + artifacts <- cli
```

Lower layers must not import CLI, trainer, experiment logger, or manuscript code.

## Package Boundaries

- `contracts`: stable, dependency-light data and artifact schemas.
- `config`: strict configuration parsing and early validation.
- `data`: manifest-backed indexing, transforms, collate, split, and validation.
- `radar`: simulation and FMCW processing with explicit tensor contracts.
- `models`: pure PyTorch modules for reconstruction and translation.
- `training`: Lightning or native PyTorch orchestration and distributed adapters.
- `evaluation`: versioned pose and language metric protocols.
- `artifacts`: atomic run metadata, predictions, metrics, and paper exports.
- `runtime`: project discovery, environment reporting, seeds, and devices.
- `cli`: user-facing composition only.

## Rebuild Slices

1. Foundation: packaging, configuration, contracts, CLI, CPU tests.
2. Data: inventory, manifest, validation, deterministic group-disjoint splits.
3. Radar: simulation and processing from explicit arrays to versioned cubes.
4. Pose: CubeNet reconstruction training and evaluation.
5. Language: pose/feature encoders, fusion, mT5 training and generation.
6. Reviewer experiments: direct baseline, DA matrix, stress tests, ablations, cost profile.
7. Release: clean environment reproduction and publication archive.

Each slice owns its schema, unit tests, contract tests, smoke command, resolved config, and artifact definition before formal training starts.

Dataset-specific audits:

- `csl_news_data.md`: official source, legacy preprocessing flow, interface drift and canonical rebuild stages.
- `run_artifacts.md`: atomic formal-run initialization, input hashing, metrics and lifecycle contract.
- `data_splits.md`: deterministic group assignment, portable split artifacts and leakage gates.
- `tensor_contracts.md`: canonical radar/pose/feature axes and the NumPy range-Doppler protocol.
- `release_audit.md`: Git-backed public inventory, dependency graph, and reviewer-release gates.
- `model_support.md`: mT5-only generation boundary, verified engineering smoke, and excluded legacy backend.
- `omnihand.md`: canonical CubeNet input/output, attention, temporal, metric, formal train/evaluate,
  checkpoint, and prediction contracts.
- `wavellm.md`: model-ready pose/feature contract, mT5 fusion, formal train/evaluate, adapter checkpoint,
  language metric, and real-data boundary.

## Environment Contract

- Python: `3.12.x`, pinned by `.python-version`.
- Resolver and installer: UV `0.11.x`.
- Lockfile: `uv.lock` is mandatory and committed.
- GPU runtime: PyTorch cu128 on the project A100 host.
- Default environment: development tools plus explicitly selected project extras.
- DeepSpeed: optional `distributed` profile, not a base dependency.

Use `scripts/bootstrap_env.sh research` for the normal project environment. Dependency changes must be made in `pyproject.toml` and resolved with `uv lock`; direct `pip install` is not part of the supported workflow.
