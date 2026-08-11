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
```

Build the pinned CSL-News partial sequence split with:

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  scripts/run_csl_news_pose_split.sh
```

This config is deliberately bound to the first 2,157-record partial pose manifest. It is engineering
evidence and must not be reused as the final dataset split.
