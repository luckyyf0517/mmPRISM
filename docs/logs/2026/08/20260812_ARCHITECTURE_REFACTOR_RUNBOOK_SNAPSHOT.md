# Architecture Refactor Runbook

Status: historical
Owner: mmPRISM coordinator
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12

## 1. Refactor Unit

每次只实现一个 canonical vertical slice，并完成：

1. paper/scientific requirement capture
2. explicit typed contract
3. new implementation
4. unit and contract verification
5. two-batch/GPU smoke
6. artifact and provenance verification
7. documentation and decision update

## 2. Recommended Slice Order

### Slice A — Runtime and Config

- `pyproject.toml` and environment lock
- environment/path/seed/device/precision
- strict typed config validation
- resolved config and run metadata
- single CLI and dry-run plan

### Slice B — Data

- canonical record and manifests
- adapters and transforms
- datamodule/collate
- split and validation

### Slice C — Radar

- antenna selection
- range/Doppler FFT
- beamforming
- simulation
- CPU/GPU numerical checks

### Slice D — OmniHand

- pure model
- Lightning training wrapper
- loss/metrics
- checkpoint adapter

### Slice E — WaveLLM

- modality encoders
- fusion
- MT5 wrapper
- train/generate/evaluate

### Slice F — Artifact and Evaluation

- prediction writer
- distributed aggregation
- metric versioning
- paper summary exporter

## 3. Legacy Isolation Rules

- `src/mmprism` 不得导入旧 `src.data`、`src.fmcw`、`src.model`、`src.eval` 或根脚本。
- 不维护旧 CLI、旧 YAML 动态 import 或旧 checkpoint compatibility。
- legacy 代码只用于提取原稿 protocol、形状、超参数和指标定义，并在 registry 标注来源。
- reviewer release 只包含 canonical package 和经过验证的入口。
- 移除 legacy 文件前仍需确认没有 paper evidence、历史 config 或 active forensic task 依赖。

## 4. Verification

每个 slice 至少运行：

```text
unit -> contract -> golden fixture -> 2-batch integration -> GPU smoke
```

从头实现仍需分层 gate，不允许用“无需兼容”跳过 contract。性能优化必须在 correctness gate 后进行，并单独报告吞吐、显存和数值差异。

Foundation 的最低验证命令：

```bash
scripts/bootstrap_env.sh research
uv run ruff check src/mmprism tests
uv run mypy
uv run pytest
uv run mmprism doctor
uv run mmprism config configs/examples/pose_smoke.yaml
uv run mmprism plan configs/examples/pose_smoke.yaml
uv run mmprism manifest tests/fixtures/manifests/pose_smoke.jsonl
```
