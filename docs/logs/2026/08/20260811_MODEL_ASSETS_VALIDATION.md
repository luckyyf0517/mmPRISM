# Evaluator Model Asset Evidence

Status: historical
Owner: mmPRISM coordinator
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12
Legacy evidence role: `ARCH-REV-002_R2-CODE-2_evidence`
Evidence ID: `EVID-CODE-MODELS-V1`

## 1. Source Identity

```text
config: configs/models/evaluation_models_v1.yaml
config SHA-256: 047e073ee0b511d0fe442177d9532eb56fd94b8acc4a5f2edbcc3426c223b464
config fingerprint: 73bbfbf3f24123657e0f19c29745423778f190ebf64890f60823c93521ee3e10
asset builder commit: d73e9431dac017d1ab4ac92489426c5286811d72 (clean)
collection builder commit: 3ae69c3ce5a5a8a872a9bcf765f570b3467a8133 (clean)
asset root: /mnt/gfs/yanyifan/mmPRISM/external/models/evaluation_models_v1
```

| Asset | Immutable upstream revision | Files / bytes | Asset manifest SHA-256 |
|---|---|---:|---|
| SimCSE | `cyclone/simcse-chinese-roberta-wwm-ext@871d7039a3fccd4869d545a25b63c545341ca7f4` | 6 / 409,532,074 | `e57f2eeb745a184615f9396ed3a8e31e1b8ec57101fec412b0e77bcc697629af` |
| SBERT | `shibing624/text2vec-base-chinese@183bb99aa7af74355fb58d16edf8c13ae7c5433e` | 8 / 409,209,289 | `81395c0bd6046dc31e82e448df44f0a21134cb467d7383df2a2248ee6218234b` |

The collection manifest SHA-256 is
`5cb656d038459ec60c1ce8f2fe958358c809e0d1628ba86b605427fd61b81b22`. Each asset directory contains
only the allowlisted loader files, `SHA256SUMS`, and `mmprism_model_asset.json`. The main SimCSE weight
hash is `bf5f42e93a4b20e774b4c1867c999fa6f89ee5a1c5a84d907644111c886c94b1`; the SBERT safetensors hash
is `0c855515479137398ce4ea985628548d4e8ed8c5764656dac966d6a24f39e721`.

## 2. Reproduction

```bash
export MMPRISM_MODEL_ROOT=/path/to/mmprism-models
scripts/download_models.sh
uv run --frozen --extra evaluation mmprism models-smoke \
  configs/models/evaluation_models_v1.yaml \
  --output-root "${MMPRISM_MODEL_ROOT}" \
  --device cpu
```

The first successful model was retained after a later remote disconnect; the next invocation revalidated
and reused it. A subsequent Xet transfer stalled without creating a final SBERT directory. Direct HTTP
resumed the cache and completed the fixed 409,098,104-byte SBERT weight. The wrapper now defaults
`HF_HUB_DISABLE_XET=1`; this changes transport only, not repository or revision identity.

## 3. Loader Smoke

```text
artifact: paper/manager/evidence/artifacts/evaluation_models_smoke_v1.json
artifact SHA-256: e957ac79f620f0a982019befa4938c393357764f5d912b4b6a7c27996f789b39
runtime commit: 3ae69c3ce5a5a8a872a9bcf765f570b3467a8133 (clean)
Python: 3.12.13
huggingface-hub: 0.36.2
device: CPU
inputs: 2 Chinese sentences
status: passed
```

| Loader | Shape | Dtype | Finite | Minimum norm | Minimum self-similarity |
|---|---:|---|---|---:|---:|
| `transformers.AutoTokenizer/AutoModel` (SimCSE) | `[2, 768]` | float32 | yes | 19.275297 | 0.99999976 |
| `SentenceTransformer` (SBERT) | `[2, 768]` | float32 | yes | 17.894876 | 0.99999982 |

This closes model acquisition and loader readiness for `ARCH-REV-002`. It does not yet validate the
full canonical translation metric protocol, paper-level metric parity, or mT5 checkpoints; those remain
under `ARCH-006-B` and the WaveLLM vertical slice. The unsupported legacy generation backend boundary is
closed separately by `EVID-CODE-MODEL-SUPPORT-V1`.
