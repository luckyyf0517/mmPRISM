# Original Experiment Reproduction Runbook

Status: `planned`
Last Updated: `2026-08-11`
Role: `paper_result_reproduction`

## 1. 固定输入

每个原投稿结果需要收集：

- manuscript table/figure/value
- original config and command
- code commit
- environment/CUDA/library versions
- dataset source, manifest and split
- checkpoint and prediction
- metric implementation/protocol

## 2. 执行顺序

1. 先重算已有 prediction 的指标，区分 metric drift 与 model drift。
2. 再用已有 checkpoint 重跑 inference，区分 preprocessing/inference drift。
3. 最后才重新训练，避免把所有差异归因于随机性。
4. 正式重训使用至少原 seed；若 reviewer 关注稳定性，再扩展多 seed。

## 3. 结果分类

- `reproduced_exact`：在严格容差内一致。
- `reproduced_with_variance`：差异可由 seed/硬件随机性解释。
- `explained_metric_drift`：指标 protocol 变化导致。
- `explained_data_drift`：数据/预处理/split 变化导致。
- `explained_code_drift`：模型或训练实现变化导致。
- `unavailable_asset`：关键数据/checkpoint 缺失。
- `unexplained_gap`：需要继续调查，不得直接写入返修结论。

## 4. 报告要求

每个结果保存 original、rerun、delta、容差、诊断、artifact 和 paper-facing decision。不得静默覆盖 manuscript 数值。
