# Master Revision Todo

Status: `active`
Last Updated: `2026-08-11`
Role: `master_task_index`

## P0 — 立即处理

| ID | Task | Dependency | Status | Exit Criterion |
|---|---|---|---|---|
| `PAPER-001A` | 导入 decision letter 和 reviewer comments | supplied email | done | 原文、英文索引、中文工作版和稳定 ID 已建立 |
| `PAPER-001B` | 导入原投稿、supplement 和当前返修稿 | user materials | in_progress | 当前稿与 supplement 静态 inventory 已完成；补齐原投稿并完成逐图表 provenance/差异审计 |
| `DATA-001` | 定位并分批接收所有 mmPRISM source、权重、日志和论文结果资产 | source location/capacity | in_progress | CSL-News 下载 active；其余 upload gate 通过且 registry 覆盖所有 data family 和历史 run |
| `OPS-001` | 建立可复现 Python/CUDA 环境 | CUDA/package selection | done | UV/Python 3.12/cu128 lock、wheel、import 与 A100 smoke 通过 |
| `ARCH-001` | 建立 greenfield package、配置、CLI 与基础 contract | author rebuild decision | in_progress | foundation tests 通过，legacy 隔离规则生效 |
| `EXP-001` | 建立 canonical OmniHand/WaveLLM 最小 vertical smoke | `DATA-001`, `OPS-001`, `ARCH-001` | in_progress | OmniHand/mT5 GPU module smoke 与 OmniHand synthetic-manifest CPU formal train/checkpoint/prediction/eval 已通过；待 clean GPU formal run、WaveLLM formal path 和真实 manifest |
| `REV-001` | 建立 reviewer diagnosis、tracker 和 closure matrix | `PAPER-001A` | done | 所有评论分类并映射到 task/evidence |
| `DATA-REV-001` | 完整数据集统计与 split audit | `DATA-001`, manuscript | in_progress | CSL-News metadata profile 已完成；待各数据族 frozen manifest、subject/scene/non-manual 和 leakage report |
| `DATA-REV-002` | 设计并采集方向/遮挡/新用户真实测试集 | ethics/capacity/protocol | blocked | condition-stratified held-out real manifest |
| `DATA-REV-003` | 建立 matched synthetic-real fidelity set | `DATA-001` | blocked | paired/category-matched manifest |
| `EXP-REV-001` | matched direct end-to-end architecture baseline | baseline/data ready | blocked | 多 seed pose/translation + matched compute |
| `EXP-REV-002` | domain adaptation comparison matrix | baseline/data ready | blocked | shallow/full/adversarial/MMD equal-budget results |
| `EXP-REV-003` | orientation/occlusion/new-user stress evaluation | `DATA-REV-002` | blocked | pose+translation condition curves |
| `EXP-REV-004` | synthetic-real fidelity analysis | `DATA-REV-003` | blocked | direct distribution + downstream evidence |
| `EXP-REV-005` | attention leave-one-out ablation | baseline/data ready | blocked | spatial/channel/SE contribution table |
| `EXP-REV-006` | training/inference cost profile | runnable models | blocked | params/FLOPs/time/memory/latency/throughput |
| `EXP-REV-007` | cross-modal benchmark feasibility and execution | manuscript/data audit | not_started | fair comparison or justified scope response |
| `ARCH-REV-001` | eliminate hard-coded release paths | architecture foundation | in_progress | clean-machine config/CLI path smoke |
| `ARCH-REV-002` | complete SBERT/SimCSE evaluator setup | environment | evidence_ready | pinned download + checksum + real SimCSE/SBERT loader smoke passed；`EVID-CODE-MODELS-V1` |
| `ARCH-REV-003` | align release docs/files and exclude internal docs | architecture audit | in_progress | release manifest contains only existing paths |
| `ARCH-REV-004` | decide and validate Phi-3 support boundary | greenfield scope policy | evidence_ready | unsupported public claim removed；release exclusion + zero-hit regression gate verified |
| `OPS-REV-001` | editorial compliance package | manuscript/data/code | in_progress | checklists/availability/source data/colour/ORCID tracked |
| `OPS-REV-002` | confirm and add code license | author approval | blocked | LICENSE and README agree |
| `PAPER-REV-001` | related work, 4D notation and sober-language revision | manuscript import | in_progress | 可重复 scan 和 30 项语言清单已完成；待核实文献、证据 gate 后修订正文并重扫 |
| `PAPER-REV-002` | point-by-point response and tracked-change manuscript | evidence ready | blocked | all comments verbatim and evidence-grounded |

## P1 — 基础设施与重建

| ID | Task | Dependency | Status | Exit Criterion |
|---|---|---|---|---|
| `DATA-002` | 定义 sample/sequence manifest schema | source data audit | in_progress | 通用 schema/validator 已落地；待真实数据 adapter 校准 |
| `DATA-003` | 为各数据族生成只读 source manifest | `DATA-002` | in_progress | CSL-News registry 为 28 archive/46,521 videos passed，2,157-record pose+caption partial manifest 已验收；`001/005/008` 隔离；待 replacement 和 436-archive complete manifest |
| `DATA-004` | 重建无泄漏 split | `DATA-003` | in_progress | CSL-News 2,157-record partial sequence split 已通过 coverage/leakage audit；待 complete source、signer/subject 和 final split |
| `ARCH-002` | 完成 package/config/CLI/runtime foundation | environment lock | in_progress | pyproject、strict config、doctor/plan CLI 与 run metadata 完整 |
| `ARCH-003` | 实现 canonical data/radar vertical slice | `DATA-002`, `ARCH-002` | in_progress | tensor/range-Doppler contract 已通过；待真实 radar manifest fixture、beamforming provenance 和 cube gate |
| `ARCH-004` | 实现 canonical OmniHand vertical slice | `ARCH-003` | evidence_ready | clean commit `688d44d` 两步 A100 smoke、single/temporal path、pose metric 与 deterministic replicate 通过；真实数据训练仍由 `ARCH-003` 阻塞 |
| `ARCH-005` | 实现 canonical WaveLLM vertical slice | `ARCH-003`, `ARCH-004` | in_progress | synthetic tensor 上 pose/radar/fused adapter 两步 train-generate smoke 已通过；待真实 manifest、checkpoint、prediction 和 metric 路径 |
| `EXP-002` | 重建原投稿 experiment registry | `DATA-001`, `PAPER-001B` | not_started | 每个原始表图有 experiment/provenance 状态 |

## P2 — 返修证据与提交

| ID | Task | Dependency | Status | Exit Criterion |
|---|---|---|---|---|
| `EXP-003` | 用 canonical 新实现从头复现原投稿关键结果 | `EXP-001`, `EXP-002`, `DATA-004` | not_started | reproduced/explained_gap report |
| `REV-002` | 冻结 reviewer-driven 新实验 protocol | `REV-001`, `EXP-001`, `PAPER-001B` | not_started | hypothesis/protocol/acceptance/paper target 完整 |
| `EXP-004` | 执行返修新增实验 | `REV-002` | not_started | 多 seed artifact 和 analysis ready |
| `PAPER-002` | 正文与 response letter 回写 | `REV-001`, evidence ready | not_started | comment-to-evidence-to-text 闭环 |
| `PAPER-003` | 数值、图表和 provenance 最终审计 | `PAPER-002` | not_started | paper evidence map 无空缺 |
| `OPS-002` | 最终测试、编译和 submission package | `PAPER-003` | not_started | final audit 全通过 |

详细分解：

- 代码：`tasks/todo_code.md`
- 数据：`tasks/todo_data.md`
- 实验：`tasks/todo_experiments.md`
- 合规：`tasks/todo_compliance.md`
