# Model Asset Acquisition

Status: current
Owner: mmPRISM coordinator
Authority scope: Cross-workspace mmPRISM rules, contracts, architecture, operations, or decisions defined by this page.
Last reviewed: 2026-08-12

## Boundary

`mmprism.assets` owns external model acquisition, immutable source identities, local materialization,
and checksum verification. Model classes and evaluators consume verified directories; they do not
download weights or resolve paths.

The v1 evaluator set is defined by `configs/models/evaluation_models_v1.yaml`:

| Asset | Upstream | Loader |
|---|---|---|
| `simcse` | `cyclone/simcse-chinese-roberta-wwm-ext@871d7039a3fccd4869d545a25b63c545341ca7f4` | `transformers.AutoModel` |
| `sbert` | `shibing624/text2vec-base-chinese@183bb99aa7af74355fb58d16edf8c13ae7c5433e` | `SentenceTransformer` |

## Lifecycle

1. `models-plan` parses the strict config and inspects local state without network or optional ML imports.
2. `models-download` resolves the configured commit through Hugging Face and rejects any mismatch.
3. Hugging Face cache state provides resumable transfer; selected files are copied into a same-filesystem staging directory.
4. Each file is hashed, `SHA256SUMS` and `mmprism_model_asset.json` are written, and the directory is atomically promoted.
5. An existing asset is reused only after exact inventory, size and checksum validation. Invalid state is never overwritten.
6. `models-smoke` loads both models from their materialized directories and verifies finite, nonzero embeddings and cosine self-similarity.

The portable manifests contain no machine-specific root. The collection manifest maps the asset-set
fingerprint to relative destinations and immutable per-model manifest hashes. Formal evaluation runs
must register that collection manifest as an input artifact.

## Failure Semantics

- Moving revisions such as `main` are rejected by schema validation.
- Missing remote files fail before promotion.
- Interrupted transfers remain recoverable through the Hugging Face cache.
- The shell wrapper defaults to direct HTTP (`HF_HUB_DISABLE_XET=1`) after a reproducible stalled Xet
  transfer on the reference host; callers may explicitly opt back into Xet.
- A corrupt final directory is reported as `invalid`; operator review is required before quarantine or removal.
- Optional Hugging Face, Transformers, Sentence Transformers and PyTorch imports occur only for download or smoke commands.
