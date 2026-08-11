# Experiment Registry

Status: `awaiting_artifact_discovery`
Last Updated: `2026-08-11`
Role: `run_and_artifact_provenance`

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
