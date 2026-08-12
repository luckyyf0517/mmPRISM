# Canonical Configuration

`configs/` contains validated configuration for the rebuilt `mmprism` package. The legacy `config/` directory is retained only for historical investigation.

Rules:

1. Every config declares a `schema_version`, experiment `name`, and `task`.
2. Machine-specific roots use environment expansion or a private ignored config.
3. Device, precision, seed, and determinism belong to `runtime`.
4. Unknown keys fail during configuration loading.
5. Formal runs must save the fully resolved configuration with their artifacts.

Validate the example without installing the package:

```bash
uv run mmprism config configs/examples/pose_smoke.yaml
uv run mmprism plan configs/examples/pose_smoke.yaml
uv run mmprism manifest tests/fixtures/manifests/pose_smoke.jsonl
uv run mmprism prepare configs/examples/pose_smoke.yaml \
  --input manifest:data_manifest=tests/fixtures/manifests/pose_smoke.jsonl \
  --input split:split_assignments=tests/fixtures/splits/pose_smoke.jsonl \
  --split-binding data_manifest=train
uv run mmprism run-init configs/examples/pose_smoke.yaml \
  --input manifest:data_manifest=tests/fixtures/manifests/pose_smoke.jsonl
```

`prepare` is the side-effect-free gate for a formal run. It requires a clean Git worktree, exactly one
split assignment input, and one explicit split binding for every manifest; its JSON report is written only
to stdout. `run-init` is a lower-level artifact-writer smoke and does not enforce those data bindings.
`run-init` creates only the atomic run provenance envelope. It does not execute training or evaluation.
The full artifact contract is documented in `docs/authority/20_CONTRACTS/RUN_ARTIFACTS.md`.

Audit the allowlisted reviewer release with:

```bash
uv run mmprism release-audit configs/release/reviewer_release_v1.yaml
```

This command is expected to return `1` until every required release deliverable is present. A structured
failed report is a valid blocker inventory; configuration or execution errors return `2`.

Prepare and validate the pinned semantic-evaluation models with:

```bash
export MMPRISM_MODEL_ROOT=/path/to/mmprism-models
uv run mmprism models-plan configs/models/evaluation_models_v1.yaml \
  --output-root "${MMPRISM_MODEL_ROOT}"
scripts/download_models.sh
uv run --frozen --extra evaluation mmprism models-smoke \
  configs/models/evaluation_models_v1.yaml \
  --output-root "${MMPRISM_MODEL_ROOT}" \
  --device cpu
```

The config contains immutable upstream commits and portable relative destinations only. The output
root is always supplied by CLI or `MMPRISM_MODEL_ROOT`; it does not belong in versioned config.

Prepare the pinned mT5 asset and run the geometry-fusion engineering smoke with:

```bash
export MMPRISM_MT5_MODEL_ROOT=/path/to/mt5-assets
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
scripts/download_mt5.sh
MMPRISM_DEVICE=cuda:0 scripts/run_mt5_smoke.sh
```

`configs/models/mt5_base_v1.yaml` fixes the upstream revision; machine-specific roots and the CUDA
device remain runtime inputs. `configs/examples/mt5_smoke.yaml` is a deterministic two-step integration
configuration with a frozen language backbone. It is not the production paper-training configuration.

Run the canonical CubeNet/OmniHand engineering smoke with:

```bash
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
MMPRISM_DEVICE=cuda:0 scripts/run_omnihand_smoke.sh
```

`configs/examples/omnihand_smoke.yaml` exercises the manuscript-facing 10-frame, 8-layer,
16-head temporal contract with a compact synthetic radar cube. It verifies execution and provenance;
it is not a production data or paper-result configuration.

The formal single-device path separates the experiment envelope from the task recipe:

```bash
export MMPRISM_DATA_ROOT=/path/to/model-ready-data
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
export MMPRISM_TRAIN_MANIFEST=/path/to/train.jsonl
export MMPRISM_VALIDATION_MANIFEST=/path/to/validation.jsonl
export MMPRISM_SPLIT_ASSIGNMENTS=/path/to/split_assignments.jsonl
scripts/run_omnihand_train.sh
```

`configs/examples/omnihand_train_smoke_experiment.yaml` owns roots, seed, device, bf16 precision, and determinism;
`configs/examples/omnihand_train_smoke.yaml` owns CubeNet, loader, optimizer, and metric settings.
Both source files, both manifests, and the canonical split assignment file are hashed as formal inputs.
Every manifest sample must match its declared train or validation assignment. The task config is deliberately
limited to two steps and must not be used as a paper-result protocol.

Build the pinned CSL-News partial sequence split with:

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  scripts/run_csl_news_pose_split.sh
```

This config is deliberately bound to the first 2,157-record partial pose manifest. It is engineering
evidence and must not be reused as the final dataset split.

Freeze a CSL-News source manifest from one exact source-integrity v2 registry snapshot with:

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  scripts/run_csl_news_source_manifest.sh
```

The v2 source-manifest config never scans primary archive names as authority. It resolves only registry-passed
paths, including versioned replacements, and writes the copied registry, manifest, summary, and `SHA256SUMS`
to a new atomic snapshot directory.

Plan or build a final model-ready Parquet delivery with a frozen source manifest and split assignment:

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  uv run --extra data-parquet mmprism parquet-delivery-plan \
  configs/data/parquet_delivery_example.yaml
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  uv run --extra data-parquet mmprism parquet-delivery-build \
  configs/data/parquet_delivery_example.yaml
```

Replace the example's zero hashes with the exact frozen source-manifest and split-assignment SHA-256 values.
The plan command does not write. Build requires clean Git, performs a capacity and source-adapter gate, creates a
new no-clobber delivery directory, copies its two frozen inputs, writes split-isolated Parquet parts, and validates
inventory/index/checksums before atomic publication. The current CSL-News RTMW3D pose/caption snapshot is not a
valid input to either model-ready product; it lacks the required metric radar modalities.
