# mT5 Geometry-Fusion Vertical Smoke Evidence

Status: historical
Owner: WaveLLM training lane
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12
Legacy evidence role: `ARCH-005_ARCH-REV-003_engineering_evidence`
Evidence ID: `EVID-CODE-MT5-SMOKE-V1`

## 1. Scope

This evidence verifies that the canonical mT5-only generation surface is portable and executable. It does
not reproduce a paper result. The smoke uses deterministic synthetic tensors, freezes the mT5 backbone,
and updates only the pose, radar and fusion adapters for two steps.

## 2. Asset Identity

```text
source: google/mt5-base
revision: 2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f
files / bytes: 6 / 2,334,046,221
weight bytes: 2,329,735,129
weight SHA-256: 180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614
asset config fingerprint: 776fa93e54f6eb12fb15b726da1197497daae31b2f62b639c164c48e3e3c4516
asset manifest SHA-256: 4edec505643e4b05ea606346599942901ab55a6b505ac6c0946f94f2a15585b9
collection manifest SHA-256: 2350101b38c5ee9c860ae5d8c28918e360eb57b47d39fc1b24a3d36773418bc6
```

The model was acquired through the same fixed-revision, resumable, per-file checksum and atomic-promotion
asset service used for the evaluator models. The versioned configuration contains no machine path.

## 3. Reproduction

```bash
export MMPRISM_MT5_MODEL_ROOT=/path/to/mt5-assets
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
scripts/download_mt5.sh
MMPRISM_DEVICE=cuda:0 scripts/run_mt5_smoke.sh
```

The smoke requires a clean Git state. Model, artifact and device locations are injected at runtime.

## 4. Verified Result

```text
artifact: paper/manager/evidence/artifacts/mt5_smoke_v1.json
artifact SHA-256: 57fd48b2028c8ee68b465b2aa2eaee2278596cb2905e46527d431b83c0b58df4
runtime commit: 79b45b58d803b3b07a8b7476f87c208e6f17399d (clean)
Python / Torch / CUDA: 3.12.13 / 2.11.0+cu128 / 12.8
device / precision: NVIDIA A100-SXM4-80GB / bfloat16
status: passed
```

| Gate | Result |
|---|---:|
| finite token cross-entropy steps | `11.368079`, `12.955701` |
| loss protocol | `ignore_index=-100`, `label_smoothing=0.0` |
| pose/radar/fusion gradient norms | nonzero in both steps |
| pose/radar/fusion parameter deltas | `0.002197` / `0.002075` / `0.002441` |
| zero-confidence pose gate mean | `0.0` |
| unit-confidence pose gate mean | `0.438375` |
| beam output | 2 sample-level predictions, token shape `[2,9]` |
| peak allocated / reserved CUDA memory | 1,334,382,592 / 1,354,760,192 B |

The randomly initialized adapters and two updates are not expected to produce meaningful translations;
the stored predictions prove generation and sample-level artifact paths only. No prediction or loss from
this smoke may be cited as scientific performance.

## 5. Remaining Boundary

- The language backbone is frozen; production full/parameter-efficient training remains unvalidated.
- Input tensors are deterministic synthetic fixtures, not a frozen paper dataset manifest.
- No checkpoint, distributed writer or paper metric is produced.
- Full `ARCH-005` closure still requires production pose/feature/fused training, generation and evaluation
  against versioned real manifests and splits.
