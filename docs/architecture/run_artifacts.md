# Formal Run Artifact Contract

Status: `formal_preflight_and_single_rank_runs_implemented`
Last Updated: `2026-08-11`
Schema: `mmprism.run.v1`

## Purpose

Every canonical training, evaluation or data-preparation service must initialize its output through
`mmprism.artifacts.RunArtifactWriter`. A logger URL or checkpoint directory alone is not a formal run.
The writer is dependency-light and does not import PyTorch, Lightning or Transformers.

`mmprism prepare` is the side-effect-free formal-run gate. `run-init` only creates the provenance envelope;
it does not execute a model or apply the full data/split preflight.

## Side-Effect-Free Preflight

`mmprism prepare EXPERIMENT_CONFIG` validates a planned formal run before any artifact directory is
created. A passing `mmprism.prepare_report.v1` report proves, for the exact invocation:

- the project root has a valid clean Git commit and the source YAML resolves to the loaded config;
- the data root exists, artifact/cache destinations have a writable existing ancestor, and the planned
  run directory does not already exist;
- all named inputs still match their captured SHA-256 and satisfy their manifest/split contracts;
- there is at least one manifest, exactly one split assignment file, no cross-manifest sample overlap,
  and every manifest sample belongs to its explicitly bound split.

The command emits JSON only to stdout and does not create the artifact root, cache root, or run directory.
Its input syntax is `KIND:NAME=PATH`; every manifest additionally requires
`--split-binding MANIFEST_NAME=SPLIT`.

```bash
MMPRISM_DATA_ROOT=/path/to/data \
MMPRISM_ARTIFACT_ROOT=/path/to/artifacts \
MMPRISM_CACHE_ROOT=/path/to/cache \
uv run --frozen mmprism prepare configs/examples/pose_smoke.yaml \
  --input manifest:data_manifest=tests/fixtures/manifests/pose_smoke.jsonl \
  --input split:split_assignments=tests/fixtures/splits/pose_smoke.jsonl \
  --split-binding data_manifest=train
```

The report is a preflight result, not a formal run artifact. The subsequent train/evaluate service must
repeat the relevant split validation and register the assignment file in its own `inputs.json`, because
inputs or repository state may change after preflight.

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
2. Capture every data manifest, split, checkpoint and model asset as a named `RunInput`; canonical formal
   train/evaluate services require at least one `manifest`, exactly one split assignment file, explicit
   manifest-to-split membership, and a non-empty launch command.
3. Initialize the run atomically. The source YAML must resolve to the exact config hash in the plan.
4. The task service writes typed artifacts. The OmniHand service atomically promotes Safetensors,
   streams strict JSONL predictions, and registers every completed top-level artifact.
5. Write `metrics.json` once with `mmprism.metrics.v1`, a non-empty protocol ID, split, sample count and only
   finite numeric values.
6. Finalize as `completed`, `failed` or `aborted`. `completed` is rejected until `metrics.json` exists;
   non-completed states require a reason.

The writer is rank-zero orchestration. Rank-local prediction append/aggregation remains `ARCH-006-A` and
must not be implemented by concurrent writes to these JSON files.

The generic JSONL writer consumes an iterable and writes atomically, so single-rank evaluation does not
retain a full prediction set in memory. Empty streams, non-mapping records, non-finite JSON values,
collisions, unsafe names, and duplicate registration are rejected. This is not yet a distributed merge
protocol.

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
