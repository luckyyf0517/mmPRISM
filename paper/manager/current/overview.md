# Revision Overview

Status: `bootstrap_single_source_of_truth`
Last Updated: `2026-08-11`
Role: `single_source_of_truth`

## 1. 项目定位

mmPRISM 当前代码承载两个相互依赖的研究系统：

1. `OmniHand`：从仿真或真实毫米波雷达数据估计双手及手臂 3D 关节。
2. `WaveLLM`：把预测姿态或预计算雷达特征映射到语言模型 embedding，并生成中文手语描述。

当前主数据流是：

```text
video / pose / raw radar
  -> pose annotation or FMCW processing
  -> mmWave cube / pose sequence / feature sequence
  -> OmniHand pose estimation
  -> pose or feature encoder
  -> WaveLLM multimodal fusion
  -> MT5 text generation
  -> pose and language metrics
```

## 2. 本轮返修的工程目标

本轮不是单纯“把旧脚本跑起来”，而是建立一套能够支撑返修和后续复核的研究工程：

- 原始数据、派生数据、split 和 paper evidence 可追踪。
- OmniHand 与 WaveLLM 可以独立训练、评测，也可以组合成端到端流程。
- 配置、环境、随机种子和 checkpoint 可以重现。
- 原投稿结果作为历史证据审计；所有返修结果由新实现从头生成并保留完整 provenance。
- 每条审稿意见都有明确的 evidence、experiment、manuscript 和 response 状态。

## 2.1 Major Revision 的科学闭环

编辑和审稿人要求本轮证据集中回答：

1. 显式 pose reconstruction 相比 direct 4D radar-to-text 是否有独立价值。
2. 当前 shallow domain adaptation 相比 full/adversarial/MMD 是否具有公平条件下的效果或效率优势。
3. 系统在 30°/60°方向、双手重叠、物体遮挡、不同环境和未见用户上如何退化。
4. 合成数据与真实数据的接近程度、数据集规模/语义覆盖/split 是否透明。
5. spatial/channel/SE attention 是否各自必要，以及 4D encoder + LLM 的计算成本。
6. reviewer 是否能在干净环境中从下载模型、配置路径、示例数据到评测完整执行。

完整诊断和优先级见 `paper/manager/reviews/analysis.md`。

## 3. 总体策略

采用 greenfield 全量重构，旧代码只作 forensic reference：

1. 冻结当前 commit `518a402` 作为 legacy baseline。
2. 旧 `run_*.py`、`config/` 和原 `src/*` 模块不再维护，只用于提取论文使用的定义、参数和历史证据。
3. `src/mmprism/` 是唯一新实现；不建立 legacy compatibility shim，也不要求旧 checkpoint 可加载。
4. 从 manifest、配置、运行 provenance 和指标协议开始，再实现 radar、pose、language 垂直切片。
5. 每个切片通过 unit、contract、integration 和 GPU smoke gate 后，才允许正式从头训练。
6. reviewer-driven 实验全部基于同一 canonical data contract、split 和 artifact schema 执行。

## 4. 目标证据链

每个论文表格、图片或 claim 必须形成：

```text
Reviewer Ask / Paper Claim
  -> Dataset Manifest + Split Hash
  -> Resolved Config + Git Commit + Environment
  -> Checkpoint / Prediction Artifact
  -> Metric Summary + Sample-Level Output
  -> Manuscript Location + Response Letter Item
```

## 5. 当前边界

- 已获得正式审稿意见并接入当前 manuscript，但历史实验与原投稿定稿仍待导入，因此不填写未经验证的结果或最终 response。
- 尚未找到历史数据，因此所有数据规模、可用率和复现实验状态均为待核对。
- legacy 代码暂时保留用于历史审计，但不会成为新训练链的依赖或发布接口。
