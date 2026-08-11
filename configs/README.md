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
uv run mmprism run-init configs/examples/pose_smoke.yaml \
  --input manifest:data_manifest=tests/fixtures/manifests/pose_smoke.jsonl
```

`run-init` creates only the atomic run provenance envelope. It does not execute training or evaluation.
The full artifact contract is documented in `docs/architecture/run_artifacts.md`.

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

Build the pinned CSL-News partial sequence split with:

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  scripts/run_csl_news_pose_split.sh
```

This config is deliberately bound to the first 2,157-record partial pose manifest. It is engineering
evidence and must not be reused as the final dataset split.
