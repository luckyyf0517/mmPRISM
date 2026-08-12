# Historical CSL-News mT5 Initialization Smoke

Status: historical
Owner: WaveLLM training lane
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12
Evidence ID: `EVID-HISTORICAL-MT5-CSLNEWS-V1`

## Scope

This log records a recovered local mT5 export that can initialize the canonical CSL-Daily WaveLLM
language backbone. It is not an evaluation of translation quality, a reproduction of the original
submission's end-to-end cam-pose WaveLLM system, or paper-facing performance evidence.

The original full WaveLLM checkpoint and its standalone historical `HandPoseEncoder` export are not
available in the project mirror, working tree, reachable Git history, or fetched branches. Their absence
does not block the greenfield CSL-Daily training path. The canonical geometry adapters are intentionally
different from the historical architecture and must be initialized and trained anew.

## Recovered Asset Identity

The read-only source directory is:

```text
/mnt/gfs/yanyifan/mmPRISM/huggingface/mt5-pretrained-news/
```

It contains a Hugging Face-compatible mT5 export. The six required model/tokenizer files have the following
SHA-256 values:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 730 | `caa30525e0679c4b78e9fb32128f09bd207d7088dca8c713aabf66681bb25dd1` |
| `generation_config.json` | 147 | `f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a` |
| `pytorch_model.bin` | 1,186,855,228 | `6bab01d48ca5ec835ee39f70a7e030f9955feff9638a9c9c8f5cdf4d81606172` |
| `special_tokens_map.json` | 65 | `09059cedc26bc46bc09a52f05b92d4922e11917e87f3b92059bb1a63a59ab2c4` |
| `spiece.model` | 4,309,802 | `ef78f86560d809067d12bac6c09f19a462cb3af3f54d2b8acbba26e1433125d6` |
| `tokenizer_config.json` | 376 | `3be4bbd24c65845b1dc86186f3c7d1d61ac2e29154cb0d71133753562f8d1959` |

Legacy `src/utils/save_pretrained.py` records the claimed historical parent as
`log/archived/wavellm_mt5_gt_pose_0523/last.ckpt` and exports only `model.model.state_dict()` to this
directory. That parent checkpoint is not present and the claim is therefore recorded as provenance context,
not independently verified ancestry. The mirror's Git remote must likewise not be used to label these bytes
as an untouched official `google/mt5-base` revision: its official pinned weight SHA-256 is
`180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614`, which differs from the recovered export.

## Inventory And Load Checks

- Deserializing `pytorch_model.bin` found 284 bfloat16 tensor keys limited to `shared`, `encoder`,
  `decoder`, and `lm_head`; no pose encoder, radar projector, fusion, optimizer, trainer state, captions,
  predictions, or other end-to-end state is present.
- `MT5ForConditionalGeneration.from_pretrained(..., local_files_only=True, dtype=bfloat16)` loaded the
  recovered asset with 582,401,280 parameters. Deterministic local generation completed successfully.
- On GPU 6, the current `GeometryGuidedMT5` accepted synthetic contract-shaped pose
  `[1,8,2,24,3]`, confidence `[1,8,2,24]`, radar feature `[1,8,1024]`, and frame-mask `[1,8]` inputs.
  Its embeddings and confidence gates were finite, and `generate()` completed.
- The current dual-hand ST-GCN, radar projector, and confidence-aware fusion contain 2,170,432 newly
  initialized parameters. Generated text from this synthetic invocation is deliberately not retained as a
  quality result.

## Permitted Use And Boundary

`MODEL-MT5-CSLNEWS-HISTORICAL-V1` may be used only as a controlled language-backbone initialization for
new canonical CSL-Daily and later real-data runs. The run receipt must bind the imported local-derived model
asset and all newly trained adapter/model state. The historical complete cam-pose checkpoint is separately
registered as unavailable and must not be represented as loaded, reproduced, evaluated, or shared across
revision comparisons.

Before a formal run, the asset service needs a narrow local-derived mT5 importer that copies no source bytes,
checks the six files above, writes an immutable asset manifest and `SHA256SUMS`, and records the historical
parent claim as unverified. The generic remote Hugging Face asset service cannot be used directly because it
requires a verified upstream repo and commit identity.

## Remaining Work

- Implement and test the local-derived asset receipt/import path.
- Freeze CSL-Daily source, model-ready manifests, and subject-aware split before training.
- Train the canonical geometry adapters and selected language-model scope under a formal run receipt.
- Establish all revision metrics from newly produced predictions; do not recover or cite historical end-to-end
  metrics without their own independently auditable evidence.
