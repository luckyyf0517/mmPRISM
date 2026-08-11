# Revision Dashboard

Status: `round0_greenfield_foundation_active`
Last Updated: `2026-08-11`
Role: `control_panel`

本页只保留当前阶段、最高优先级、blocker 和下一步动作。

## 当前阶段

`Round0: Legacy Baseline, Asset Discovery, and Revision Bootstrap`

- 旧代码结构已完成首轮静态盘点。
- 返修管理控制面已建立。
- Major revision decision、编辑硬性要求和两位 reviewer comments 已保留原文并完成中英文结构化整理。
- Reviewer Stage 1 诊断、P0–P2 实验计划、response tracker 和 evidence ID 已建立。
- 当前返修稿已通过私有 Overleaf Git 子模块接入 `paper/manuscript`；主入口为 `sn-article.tex`。
- 稿件首轮静态扫描确认 Availability 正式章节缺失，且仍含编辑明确要求删除的夸张/首创式表达。
- 作者已确认数据与训练从头重建；`src/mmprism` greenfield package、strict config、manifest contract、CLI 和 CPU 基础测试已开始落地。
- `/mnt/gfs/yanyifan` 当前未发现 mmPRISM、CSL、OmniHand 或 WaveLLM 数据目录。
- canonical UV 环境已锁定 Python 3.12/PyTorch 2.11 cu128；8 项测试、Ruff、Mypy、wheel、核心依赖 import 和 A100 CUDA smoke 均通过。
- 原投稿定稿与独立 response letter 尚未导入；当前子模块中已有主稿、章节、参考文献、图和 supplementary 压缩包。

## 当前最高优先级

1. 登记当前 Overleaf 稿件中的章节、表图与 supplementary，并补充导入原投稿定稿。
2. 先建立 sober-language 与 Availability 修改清单，不在新增实验完成前强化或改写结果主张。
3. 确认历史 mmPRISM 数据、权重、日志和实验结果的真实存储位置。
4. 确认可新增真实数据的人数、伦理边界、方向/遮挡采集条件和时间预算。
5. 完成 canonical 环境锁定、data/radar vertical slice 和最小 GPU smoke，再冻结 P0 实验 protocol。

## 当前 Blocker

- `BLOCK-DATA-ROOT`：挂载盘未发现预期数据，来源位置未知。
- `BLOCK-MANUSCRIPT`：当前工作稿已接入，但原投稿定稿与 response letter 缺失，且稿件表图尚未完成 provenance 登记。
- `BLOCK-PROVENANCE`：历史 checkpoint、split 和 paper-facing 数值尚未建立对应关系。
- `BLOCK-REAL-COLLECTION`：新增参与者、方向/遮挡 protocol 和伦理/同意范围尚未确认。

## 下一步动作

1. 继续执行 `PAPER-001B`，登记当前 manuscript/supplement，并导入原投稿定稿。
2. 执行 `DATA-001`，生成全量资产 inventory，而不是直接移动数据。
3. 对 `DATA-REV-002` 做作者/伦理/采集资源确认。
4. 继续 `ARCH-001/002`，完成正式 artifact writer 和 canonical train/eval/prepare CLI。
5. 数据定位后执行 `ARCH-003`，从真实 manifest fixture 建立 radar vertical slice。

## Source Of Truth

- 总路线：`current/roadmap.md`
- 架构：`current/architecture_status.md`
- 数据：`current/data_status.md`
- 风险：`current/issues.md`
- 任务：`tasks/todo.md`
- 决策：`decisions/decision_log.md`
- 审稿原文：`reviews/review_en.md`
- 中文整理：`reviews/review_cn.md`
- 审稿诊断：`reviews/analysis.md`
