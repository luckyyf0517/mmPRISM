# Experiment Registry

Status: historical
Owner: Paper revision lane
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12

## Historical Runs

| Run ID | Purpose | Git Commit | Config | Dataset/Split | Checkpoint | Predictions | Metrics | Verification | Status |
|---|---|---|---|---|---|---|---|---|---|
| `RUN-LEGACY-UNKNOWN` | original submission experiments | unknown | unknown | unknown | unknown | unknown | unknown | not audited | blocked |

## Engineering Smokes

### RUN-20260811-mt5-geometry-fusion-smoke

- Task / Reviewer IDs: `ARCH-005-A`, `ARCH-005-B`, `ARCH-REV-003`.
- Purpose: verify the portable canonical mT5 module boundary, not a scientific hypothesis.
- Git commit / dirty state: `79b45b58d803b3b07a8b7476f87c208e6f17399d` / clean.
- Environment lock: `uv.lock`; Python 3.12.13; Torch 2.11.0+cu128; Transformers 4.57.6.
- Resolved config: fingerprint `dd6c5d9574aede0c65baf00e78c4ab50c7b5f430817e8dd5456b38adb5022c11`.
- Model asset: `google/mt5-base@2eb15465...`; collection manifest `2350101b...`.
- Dataset manifest / split: none; deterministic synthetic tensor fixture only.
- Seed / device / precision: `20260811` / A100 GPU 5 / bfloat16.
- Training: two finite token-cross-entropy adapter updates; mT5 backbone frozen.
- Predictions: two stored sample-level beam outputs; not scientifically meaningful.
- Evidence: `mt5_vertical_smoke.md`; `artifacts/mt5_smoke_v1.json`.
- Paper destination: none; engineering/release evidence only.
- Status: `passed_engineering_smoke`.

### RUN-20260811-omnihand-cubenet-smoke

- Task / Reviewer IDs: `ARCH-004-A`, `ARCH-004-B`, `EXP-001-B`, `EXP-001-C`.
- Purpose: verify the canonical CubeNet reconstruction model and metric boundary, not a scientific
  hypothesis or paper result.
- Git commit / dirty state: `688d44d18d7441c0c79706546e97683db2713ce9` / clean.
- Environment lock: `uv.lock`; Python 3.12.13; Torch 2.11.0+cu128; CUDA 12.8.
- Resolved config: fingerprint `9111d597d37b2eab5662b2c492a5ce09caef21733afbac2c0540d419df39c13b`.
- Dataset manifest / split: none; deterministic synthetic non-negative cube and metric-pose tensors only.
- Seed / device / precision: `20260811` / A100 GPU 5 / bfloat16 autocast.
- Training: random initialization; two finite masked-L1 optimizer updates; all spatial, temporal and
  pose-head gradient norms and tracked parameter deltas nonzero.
- Predictions: two sample-level synthetic pose metric records; not scientifically meaningful.
- Validation: single-frame finite output, temporal padding invariance `0.0`, independent attention
  toggles, PAFPN odd-shape checks, 135-test suite, deterministic normalized replicate hash `282bd16e...`.
- Evidence: `omnihand_vertical_smoke.md`; `artifacts/omnihand_smoke_v1.json`.
- Paper destination: none; engineering/release evidence only.
- Status: `passed_engineering_smoke`.

### RUN-20260811-omnihand-formal-gpu-smoke

- Task / Reviewer IDs: `ARCH-004-A`, `ARCH-006-A`, `EXP-001`, `EXP-REV-006` engineering prerequisite.
- Purpose: verify the full single-device formal train/checkpoint/reload/predict/evaluate lifecycle, not a
  scientific hypothesis or manuscript result.
- Git commit / dirty state: `81e9b89896a25bc26eece5f789b9a842004a4d4a` / clean.
- Environment lock: `uv.lock`; Python 3.12.13; Torch 2.11.0+cu128; CUDA 12.8.
- Resolved config: run fingerprint `eb99098e6e9ad0b7a5f1f1339bdd01d05ebe200942962db5d4a9d98115698508`;
  model config SHA-256 `f374c8e3df02ee3db74e1d39b2d74575418d84f6f946f088249460324baa3c42`.
- Dataset manifest / split: deterministic synthetic model-ready fixture; train 4 records
  (`0020607a...`), validation/test 2 records (`48be61e0...`). No scientific split.
- Seed / device / precision: `20260811` / A100 physical GPU 5 exposed as `cuda:0` / `bf16-mixed`.
- Train run: `omnihand-train-smoke__20260811T201542Z__eb99098e`; 2 optimizer steps; completed.
- Evaluate run: `omnihand-train-smoke__20260811T201630Z__eb99098e`; checkpoint reload; completed.
- Checkpoint: Safetensors SHA-256 `18b941a3161a10978ca91033ed670a9881a09339b797c80fc4aed13e9c9b8010`.
- Predictions: 2/2 records; train and standalone evaluate payloads are byte-identical, SHA-256
  `c80928b022877d4857b87940d109ebf171e7edbbac449daaf87844c491ee0f6c`.
- Metrics: `mmprism.pose_metric.dual_hand_metric_v1`; finite count-weighted summary replayed identically.
  Synthetic values are intentionally excluded from this registry and are not paper evidence.
- Performance: train 1.2892 s core / 4.0648 s end-to-end / 48,897,536 B peak allocated; evaluate
  0.8647 s prediction / 3.2978 s end-to-end / 18,224,640 B peak allocated.
- Validation: independent 13-gate audit passed; all artifact/input hashes, clean Git, A100 BF16,
  checkpoint linkage, coverage, replay identity, finite metrics, performance fields, and temp-file gate passed.
- Evidence: `omnihand_formal_run.md`; `artifacts/omnihand_formal_run_v1.json`; mounted audit SHA-256
  `4450c5be6684dc51a1cee43a70361c721707165250f9a4a1709642648b3ea4d4`.
- Paper destination: none; engineering/release evidence only.
- Status: `passed_engineering_formal_run`.

### RUN-20260811-wavellm-formal-gpu-smoke

- Task / Reviewer IDs: `ARCH-005-B`, `ARCH-006-A`, `ARCH-006-B`, `EXP-001`, `EXP-REV-006`
  engineering prerequisite.
- Purpose: verify the full single-device WaveLLM train/adapter-checkpoint/reload/predict/evaluate
  lifecycle, not a scientific hypothesis or manuscript result.
- Git commit / dirty state: `e31000b3f55718d36df15e2013d80e18f7b690e1` / clean.
- Environment lock: `uv.lock`; Python 3.12.13; Torch 2.11.0+cu128; Transformers 4.57.6; CUDA 12.8.
- Model asset: `google/mt5-base@2eb15465...`; collection manifest `2350101b...`; mT5 frozen.
- Dataset manifest / split: deterministic synthetic model-ready fixture; train 4 records
  (`3b1e68bf...`), validation/test 2 records (`193ae2dd...`); sample/sequence overlap 0. No scientific split.
- Seed / device / precision: `20260811` / A100 physical GPU 5 exposed as `cuda:0` / `bf16-mixed`.
- Train run: `wavellm-train-smoke__20260811T205116Z__90a769db`; 2 optimizer steps; completed.
- Evaluate run: `wavellm-train-smoke__20260811T205146Z__90a769db`; adapter checkpoint reload; completed.
- Checkpoint: adapter-only Safetensors SHA-256
  `e4aab4edcc00f0ed51e290a3bb841e8549732b4c542daeb4d6b77d32229f5f44`; 62 tensors and zero
  `language_model.*` keys.
- Predictions: 2/2 records; train and standalone evaluate payloads are byte-identical, SHA-256
  `5f95172f06efdce35b318ad91ccc2e7aa29098978b46b736140c7679ac90ca03`.
- Metrics: `mmprism.language_metric.character_v1`; Unicode code-point edit counts and count-weighted
  summary independently replayed. Synthetic values are intentionally excluded and are not paper evidence.
- Performance: train 1.5687 s core / 17.3319 s end-to-end / 1,291,147,776 B peak allocated; evaluate
  1.0769 s prediction / 16.2700 s end-to-end / 1,210,208,256 B peak allocated.
- Validation: independent 250-gate audit passed; all artifact/input/array hashes, clean Git, A100 BF16,
  split separation, adapter inventory, coverage, prediction and metric replay, finite values, performance,
  and temp-file gates passed.
- Evidence: `wavellm_formal_run.md`; `artifacts/wavellm_formal_run_v1.json`; mounted audit SHA-256
  `ea2d074092c865615077242b156d96d45c01b933577a25d4d62757ef4ead6458`.
- Paper destination: none; engineering/release evidence only.
- Status: `passed_engineering_formal_run`.

## New Run Record Template

```markdown
### RUN-YYYYMMDD-short-name

- Task / Reviewer IDs:
- Hypothesis:
- Git commit / dirty state:
- Environment lock:
- Resolved config:
- Command:
- Dataset manifest / split hash:
- Seed / device / precision:
- Start / end time:
- Checkpoint:
- Predictions:
- Sample-level metrics:
- Summary metrics:
- Validation checks:
- Result interpretation:
- Paper destination:
- Status:
```

## Artifact Completeness Gate

正式 run 至少必须有：

1. `run.json` 或等价 metadata。
2. resolved config。
3. code/environment/data identity。
4. checkpoint 或明确 `evaluation_only` 标记。
5. sample-level prediction。
6. versioned metric summary。
