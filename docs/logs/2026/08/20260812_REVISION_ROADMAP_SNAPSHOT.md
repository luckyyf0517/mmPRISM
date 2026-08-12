# Comprehensive Revision Roadmap

Status: historical
Owner: mmPRISM coordinator
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12

本路线图按验收门槛而不是固定日期推进。返修截止日期导入后，再将各 phase 映射到具体日历。

当前内部规划目标为 `2026-11-11`，对应编辑邮件“within three months”。邮件同时允许更长返修周期，但若 P0 真实数据采集无法按时完成，应提前联系编辑，而不是临近目标日期才申请延期。

## Phase 0 — Revision Intake and Legacy Freeze

目标：明确审稿范围，冻结可调查的旧项目状态。

工作：

- 导入原投稿、decision letter、reviewer comments、当前 manuscript 和 response draft。
- 保存当前 commit、远程分支、历史删除模块和旧配置索引。
- 建立锁定环境、最小 fixture 和 legacy smoke 命令。
- 完成代码、配置、路径和依赖静态审计。

Exit Gate：每条 reviewer comment 有稳定 ID；至少一个 OmniHand 和一个 WaveLLM 路径能够 smoke run，或有明确 blocker 记录。

## Phase 1 — Data Discovery and Provenance Reconstruction

目标：找到全部历史数据/权重/结果，建立只读 inventory。

工作：

- 对所有来源生成目录、大小、文件数、mtime、格式和 checksum 抽样。
- 区分 raw、annotation、derived pose、mmWave、feature、split、checkpoint 和 paper result。
- 建立数据 registry 和历史实验 artifact registry。
- 识别重复、损坏、缺帧、shape 漂移、subject 泄漏和缺失标注。

Exit Gate：所有 paper-relevant 数据族有 location、owner、状态、容量和风险；不存在未经确认的大规模复制计划。

## Phase 2 — Data Contract and Rebuild Pipeline

目标：把数据从路径约定升级为 manifest/schema 驱动。

工作：

- 定义 `sample_id`、`sequence_id`、`subject_id`、模态 URI、shape、dtype、采集信息和 provenance schema。
- 建立 raw/interim/processed/manifests/splits/quarantine 分层。
- 实现 inventory、validate、build-manifest、build-split 和 materialize CLI。
- 为 CSL-Daily、CSL-News、collected 数据分别建立 adapter。

Exit Gate：小样本可以从 raw 通过确定性流水线生成 processed 数据；manifest/split 校验和泄漏检查通过。

## Phase 3 — Greenfield Architecture and Vertical Build

目标：建立唯一 canonical package，并以新数据和新训练链从头生成返修结果。

工作：

- 维护 `pyproject.toml`、锁定依赖、统一 CLI 和 strict typed config。
- canonical package 分为 `contracts`、`config`、`data`、`radar`、`models`、`training`、`evaluation`、`artifacts`、`runtime` 和 `cli`。
- 按 data contract、radar、pose、language、evaluation 的顺序实现新垂直切片。
- 旧 `src.*` 与根入口只读取证；canonical package 禁止导入，不编写 compatibility shim。

Exit Gate：新 package clean install；CPU contract tests、两批次 integration 和 GPU smoke 通过；正式配置无绝对路径、动态 import 或缺失模块。

## Phase 4 — Original Result Reproduction

目标：用 canonical 新实现从头重建原投稿核心结果，并解释与历史数值的差异。

工作：

- 锁定原投稿使用的数据 split、预处理、checkpoint 和 metric protocol。
- 分别复现 OmniHand pose 指标与 WaveLLM 文本指标。
- 对无法复现的数值分类：数据缺失、代码漂移、指标漂移、随机性或记录缺失。
- 形成 reproduction report，不静默替换原结果。

Exit Gate：每个原投稿表格/图片有 `reproduced / explained_gap / unavailable` 状态和 evidence。

## Phase 5 — Reviewer-Driven Experiments

目标：只做审稿意见真正要求、且能改善证据闭环的实验。

工作：

- 从 response tracker 生成实验优先级。
- 每项实验先写 hypothesis、protocol、acceptance criterion 和 paper destination。
- 先运行小规模 protocol validation，再启动正式多 seed 实验。
- 失败结果也保留 provenance 和结论边界。

Exit Gate：所有 P0/P1 reviewer evidence gap 为 `evidence_ready`、有明确诚实边界，或经作者决定不做并有理由。

## Phase 6 — Manuscript and Response Closure

目标：让正文、response letter、图表和 evidence 完全一致。

工作：

- 逐条 reviewer closure audit。
- 全文结构、术语、claim strength、数值和图表引用同步。
- 对新增数据和实验补充方法细节、限制和复现信息。
- 建立 final numeric provenance pass。

Exit Gate：reviewer tracker 无未解释 P0/P1；每个 response claim 有 manuscript evidence；正文和 response 可编译。

## Phase 7 — Final Audit and Submission Package

目标：形成可提交、可复查、可交接的最终包。

工作：

- 运行 placeholder/stale-term/number/reference scan。
- 执行最终测试、配置校验、数据/实验 provenance 审计。
- PDF 视觉检查、补充材料和代码说明检查。
- 冻结 submission tag、environment lock 和 evidence snapshot。

Exit Gate：submission package ready，已知限制和未完成项均显式记录。
