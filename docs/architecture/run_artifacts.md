# Formal Run Artifact Contract

Status: `formal_preflight_prediction_and_epoch_resume_implemented`
Last Updated: `2026-08-12`
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
4. The task service writes typed artifacts. Model processes publish immutable rank-local prediction
   shards and receipts; rank zero validates, aggregates, and registers the complete result set. A
   single-process trainer publishes an immutable state pair after each fully completed epoch.
5. Write `metrics.json` once with `mmprism.metrics.v1`, a non-empty protocol ID, split, sample count and only
   finite numeric values.
6. Finalize as `completed`, `failed` or `aborted`. `completed` is rejected until `metrics.json` exists;
   non-completed states require a reason.

## Distributed Prediction Artifacts

Every rank writes only its own pair and never modifies shared `run.json`:

```text
predictions.rank-00000-of-00004.jsonl
predictions.rank-00000-of-00004.json
```

The JSON receipt uses `mmprism.prediction_shard.v1` and binds the run ID, domain prediction schema,
rank/world size, record count, ordered sample-ID digest, JSONL SHA-256 and byte size. Publication uses
atomic no-clobber promotion; an existing rank artifact is evidence of a collision and is never replaced.
An empty rank is valid, which covers distributed samplers where world size exceeds local sample count.

Rank zero calls `aggregate_prediction_shards` only after all receipts exist. It requires every expected
rank exactly once, rejects unexpected rank files, verifies receipt and shard identity, enforces canonical
strict JSON and the domain schema, and rejects missing, extra or duplicate sample IDs. The aggregator uses
a temporary SQLite table to sort by the UTF-8 sample ID key without retaining large pose payloads in
memory. It writes:

```text
predictions.jsonl
predictions.index.json
```

The index uses `mmprism.prediction_aggregation.v1` and records exact coverage plus every shard, receipt and
merged-file checksum. Only rank zero updates `run.json`, registering the whole completed set in one atomic
metadata update. Shards remain as provenance and are not deleted. OmniHand and WaveLLM world-size-one
formal runs use this same path so the distributed contract is continuously exercised.

## Epoch-Boundary Training Resume

OmniHand and WaveLLM use `mmprism.training_state.v1` for exact single-process recovery. Every fully
completed epoch publishes and registers one no-clobber pair:

```text
training-state.epoch-00001.json
training-state.epoch-00001.safetensors
```

The strict canonical JSON records source run ID, completed epoch, cumulative global step, full history,
configured epoch/step targets, tensor checksum/size, and exact compatibility bindings for Git commit,
manifests, split assignments, model/training configuration, runtime seed/precision/determinism, device
type, coordinate frame, and applicable model assets/checkpoint scope. The Safetensors payload contains
the resumable model scope, AdamW tensor state, Torch CPU/CUDA RNG, and DataLoader generator state. JSON
contains the remaining AdamW scalars/options, GradScaler state, and Python/NumPy RNG.

Both `--resume-state-metadata` and `--resume-state-tensors` are mandatory when resuming. They are captured
as independent `checkpoint` inputs before loading. The loader verifies strict JSON bytes, filename,
checksum, size, schema, bindings, optimizer topology, model keys, tensor shapes, device expectations, and
that every Safetensors tensor is referenced exactly once. Training may keep or increase `epochs` and
`max_steps`; any other compatibility drift or target decrease is rejected.

No state is published for a partial epoch. After mid-epoch interruption, the next run restores the last
complete boundary and reruns only the incomplete epoch. This contract does not claim mid-epoch recovery,
DDP model execution, or distributed checkpoint aggregation.

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
