# Formal Run Artifact Contract

Status: `foundation_implemented`
Last Updated: `2026-08-11`
Schema: `mmprism.run.v1`

## Purpose

Every canonical training, evaluation or data-preparation service must initialize its output through
`mmprism.artifacts.RunArtifactWriter`. A logger URL or checkpoint directory alone is not a formal run.
The writer is dependency-light and does not import PyTorch, Lightning or Transformers.

`run-init` only creates the provenance envelope. It does not execute a model or imply that canonical
train/evaluate/prepare services already exist.

## Initial Layout

```text
<artifact-root>/<experiment-name>/<run-id>/
  run.json
  config.resolved.json
  environment.json
  inputs.json
```

Initialization writes the complete directory to a same-filesystem temporary location and promotes it with
one atomic rename. Existing run IDs are never overwritten. No partial target directory is visible after a
failed initialization.

- `run.json`: schema, run ID, status, UTC time, task, seed, command, source-config hash, Git state and hashes
  of registered artifacts.
- `config.resolved.json`: environment-expanded, absolute-path canonical experiment configuration.
- `environment.json`: Python/platform/package versions, project root, Git commit and dirty state.
- `inputs.json`: named manifests, splits, checkpoints, model assets and other inputs with absolute local path,
  byte size and SHA-256.

The local paths make a run auditable on the execution host. Portable publication artifacts must separately
use manifest URIs and the release/export contract; they must not copy these paths into a public package.

## Lifecycle

1. Build a side-effect-free `RunPlan` with a timezone-aware UTC timestamp.
2. Capture every data manifest, split, checkpoint and model asset as a named `RunInput`; initialization
   requires at least one `manifest` input and a non-empty launch command.
3. Initialize the run atomically. The source YAML must resolve to the exact config hash in the plan.
4. The task service writes checkpoints/predictions through their future typed writers.
5. Write `metrics.json` once with `mmprism.metrics.v1`, a non-empty protocol ID, split, sample count and only
   finite numeric values.
6. Finalize as `completed`, `failed` or `aborted`. `completed` is rejected until `metrics.json` exists;
   non-completed states require a reason.

The writer is rank-zero orchestration. Rank-local prediction append/aggregation remains `ARCH-006-A` and
must not be implemented by concurrent writes to these JSON files.

## CLI Smoke

```bash
MMPRISM_DATA_ROOT=/path/to/data \
MMPRISM_ARTIFACT_ROOT=/path/to/artifacts \
MMPRISM_CACHE_ROOT=/path/to/cache \
uv run mmprism run-init configs/examples/pose_smoke.yaml \
  --input manifest:data_manifest=tests/fixtures/manifests/pose_smoke.jsonl
```

Input syntax is `KIND:NAME=PATH`. Supported kinds are `manifest`, `split`, `checkpoint`, `model`, `config`
and `other`; names must match `[a-z0-9][a-z0-9._-]*`. Relative input paths resolve from the project root.

## Evidence Gate

A future formal experiment is eligible for the experiment registry only if:

1. `run.json` is finalized and its Git/config/environment fields are present;
2. all manifests, splits, checkpoints and model assets used by the service appear in `inputs.json`;
3. artifact hashes recompute exactly;
4. `metrics.json` names a versioned protocol and retains its sample count;
5. sample-level prediction artifacts are present when the metric or paper claim requires them.
