# Response Letter Tracker

Status: `initial_mapping_complete_evidence_pending`
Last Updated: `2026-08-11`
Role: `reviewer_response_index`

当前只完成 Stage 1 诊断和行动映射；没有新结果，因此不撰写结果型 response。原文见 `review_en.md`，中文工作版见 `review_cn.md`，优先级解释见 `analysis.md`。

## Editor

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `ED-SCI-1` | EMPIRICAL_SUPPORT | P0 | 用架构、DA、真实 stress、fidelity、ablation 组成 stronger validation package | `EXP-REV-001`–`006` | `EVID-REV-*` | blocked | High |
| `ED-SCI-2` | FAIR_COMPARISON | P0 | matched direct 4D-cube-to-LLM baseline | `EXP-REV-001` | `EVID-REV-ARCH` | blocked | High |
| `ED-SCI-3` | FAIR_COMPARISON | P0 | shallow/full/adversarial/MMD 相同预算比较 | `EXP-REV-002` | `EVID-REV-DA` | blocked | High |
| `ED-SCI-4` | REPRODUCIBILITY | P0 | 完整数据统计、split 与 provenance | `DATA-REV-001` | `EVID-REV-DATASET` | in_progress | High |
| `ED-SCI-5` | GENERALIZATION | P0 | 方向、遮挡、场景、新用户真实测试 | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `ED-WRITE-1/2` | SCOPE_OR_OVERCLAIM | P0 | 全文 sober-language scan 与 claim narrowing | `PAPER-REV-001` | manuscript diff | in_progress | Medium |
| `ED-WRITE-3/4/5` | REVIEW_PROCESS | P0 | 原文逐条回复、修改标记、未完成项解释 | `PAPER-REV-002` | response/manuscript | blocked | High |
| `ED-COMP-1`–`12` | REVIEW_PROCESS | P0 | checklists、availability、source data、颜色、ORCID、提交包 | `OPS-REV-001` | compliance package | in_progress | High |

## Reviewer 1

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `R1-1` | RELATED_WORK | P2 | 核实并讨论 RadarLLM/mmExpert | `PAPER-REV-001` | bibliography/manuscript | not_started | Low |
| `R1-2` | EFFICIENCY | P0 | 参数、FLOPs、GPU-hours、显存、latency、throughput | `EXP-REV-006` | `EVID-REV-EFF` | blocked | Medium |
| `R1-3` | GENERALIZATION | P0 | 0°/30°/60° orientation-stratified real test | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `R1-4a` | EMPIRICAL_SUPPORT | P0 | paired/category-matched synthetic-real fidelity | `DATA-REV-003`, `EXP-REV-004` | `EVID-REV-SYNREAL` | blocked | High |
| `R1-4b` | REPRODUCIBILITY | P0 | synthetic/train/test 数据详情和 split audit | `DATA-REV-001` | `EVID-REV-DATASET` | in_progress | High |
| `R1-4c` | GENERALIZATION | P0/P1 | 扩充真实参与者并记录多样性 | `DATA-REV-002` | `EVID-REV-REAL` | blocked | High |
| `R1-4d` | GENERALIZATION | P0 | held-out user 与 low-data adaptation 分析 | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `R1-5` | ROBUSTNESS | P0 | hand-overlap/occlusion stratified pose+translation test 与机制解释 | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `R1-6` | CLARITY | P2 | 当前 Methods 已有 4D 维度定义；补齐首次出现、单位和 tensor layout 并标记修订 | `PAPER-REV-001` | manuscript diff | in_progress | Low |

## Reviewer 2 — Manuscript

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `R2-1` | FAIR_COMPARISON | P0 | matched direct end-to-end baseline | `EXP-REV-001` | `EVID-REV-ARCH` | blocked | High |
| `R2-2` | FAIR_COMPARISON | P0 | DA method matrix under equal real-data budget | `EXP-REV-002` | `EVID-REV-DA` | blocked | High |
| `R2-3` | ROBUSTNESS | P0 | orientation + hand/object occlusion，pose+translation 双任务 | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `R2-4` | REPRODUCIBILITY | P0 | sign type/vocab/sentences/length/non-manual/split statistics | `DATA-REV-001` | `EVID-REV-DATASET` | in_progress | High |
| `R2-5` | EMPIRICAL_SUPPORT | P0 | spatial/channel/SE leave-one-out ablation | `EXP-REV-005` | `EVID-REV-ATTN` | blocked | High |
| `R2-6` | MISSING_BASELINE | P1 | 先做 cross-modal protocol feasibility audit，再决定重实现/定位 | `EXP-REV-007` | `EVID-REV-XMODAL` | not_started | Medium |

## Reviewer 2 — Code Availability

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `R2-CODE-1` | REPRODUCIBILITY | P0 | 移除本地绝对路径，统一 config/CLI | `ARCH-REV-001` | clean-room run | in_progress | High |
| `R2-CODE-2` | REPRODUCIBILITY | P0 | 修复 SBERT 下载和评测模型准备 | `ARCH-REV-002` | evaluation smoke | not_started | High |
| `R2-CODE-3` | REPRODUCIBILITY | P0 | 重写 release README；清除不存在文件；release 排除 CLAUDE/internal docs | `ARCH-REV-003` | release manifest | in_progress | High |
| `R2-CODE-4` | REVIEW_PROCESS | P0 | 作者确认许可证，添加 LICENSE 或修正文档 | `OPS-REV-002` | license file/decision | blocked | High |
| `R2-CODE-5` | REPRODUCIBILITY | P0 | Phi-3 二选一：完整可运行支持，或删除 unsupported claim | `ARCH-REV-004` | supported-model smoke | blocked | Medium-High |

## Closure Gate

每条 item 标记 `done` 前必须同时满足：Direct Answer、verified evidence、manuscript revision、response revision 和 claim-strength audit。
