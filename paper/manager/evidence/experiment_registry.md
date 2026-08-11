# Experiment Registry

Status: `awaiting_artifact_discovery`
Last Updated: `2026-08-11`
Role: `run_and_artifact_provenance`

## Historical Runs

| Run ID | Purpose | Git Commit | Config | Dataset/Split | Checkpoint | Predictions | Metrics | Verification | Status |
|---|---|---|---|---|---|---|---|---|---|
| `RUN-LEGACY-UNKNOWN` | original submission experiments | unknown | unknown | unknown | unknown | unknown | unknown | not audited | blocked |

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
