# Response Letter Tracker

Status: current
Owner: Paper revision lane
Authority scope: The manuscript revision and evidence-promotion boundary represented by this page.
Last reviewed: 2026-08-12

当前只完成 Stage 1 诊断和行动映射；没有新结果，因此不撰写结果型 response。原文见
[review EN](../../logs/2026/08/20260811_REVIEW_EN.md)，中文工作版见
[review CN](../../logs/2026/08/20260811_REVIEW_CN.md)，优先级解释见
[review analysis](REVIEW_ANALYSIS.md)。

下表的 `Planned Action` 是作者方案，不是原文复述。`ED-SCI-1` 至 `ED-SCI-5` 是对编辑同一个科学总括
段落的内部跟踪切片，不代表编辑另列了五条意见；原意和措辞强度见
[reviewer comments brief](REVIEWER_COMMENTS_BRIEF.md)。

一组历史 WaveLLM bundle 正在接收，尚未完成 inventory、checksum、format/world-size 或 controlled-load
audit。因此，当前不能把其目录名、partial checkpoint 或 historical evaluation 文件写成可用的投稿模型或
结果。待 receipt 后才选择本轮各方法共享的 language initialization；此前 mT5-only export 仅为 fallback。
无论选择何者，CSL-Daily 数据、canonical geometry adapters、预测和指标都必须重新生成。完整 CSL-News
重训不属于 P0，边界见 [review analysis](REVIEW_ANALYSIS.md) 和项目 `DEC-046`。

## Editor

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `ED-SCI-1` | EMPIRICAL_SUPPORT | P0 | 用架构、DA、真实 stress、fidelity、ablation 组成 stronger validation package | `EXP-REV-001`–`006` | `EVID-REV-*` | blocked | High |
| `ED-SCI-2` | FAIR_COMPARISON | P0 | matched direct 4D-cube-to-LLM baseline | `EXP-REV-001` | `EVID-REV-ARCH` | blocked | High |
| `ED-SCI-3` | FAIR_COMPARISON | P0 | shallow/full/adversarial/MMD 相同预算比较 | `EXP-REV-002` | `EVID-REV-DA` | blocked | High |
| `ED-SCI-4` | REPRODUCIBILITY | P0 | 完整数据统计、split 与 provenance | `DATA-REV-001` | `EVID-REV-DATASET` | in_progress | High |
| `ED-SCI-5` | GENERALIZATION | P0 | 约 30 名实际完成者的 video-guided CSL 跨参与者测试和小型方向/遮挡 stress 测试；专业/熟练人员单列 | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `ED-WRITE-1/2` | SCOPE_OR_OVERCLAIM | P0 | 全文 sober-language scan 与 claim narrowing | `PAPER-REV-001` | `EVID-PAPER-INVENTORY`; manuscript diff pending | in_progress | Medium |
| `ED-WRITE-3/4/5` | REVIEW_PROCESS | P0 | 原文逐条回复、修改标记、未完成项解释 | `PAPER-REV-002` | response/manuscript | blocked | High |
| `ED-COMP-1`–`12` | REVIEW_PROCESS | P0 | checklists、availability、source data、颜色、ORCID、提交包 | `OPS-REV-001` | `EVID-PAPER-INVENTORY`; compliance package pending | in_progress | High |

## Reviewer 1

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `R1-1` | RELATED_WORK | P2 | 核实并讨论 RadarLLM/mmExpert | `PAPER-REV-001` | bibliography/manuscript | not_started | Low |
| `R1-2` | EFFICIENCY | P0 | 确定可复现的训练/推理成本指标和测量条件；原文未指定指标集 | `EXP-REV-006` | `EVID-REV-EFF` | blocked | Medium |
| `R1-3` | GENERALIZATION | P0 | cross-orientation analysis；可与 `R2-3` 的 30°/60° 条件合并，角度不是 R1 原话 | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `R1-4a` | EMPIRICAL_SUPPORT | P0 | 选择并论证 synthetic-ground-truth closeness 度量；原文未指定 paired protocol 或指标 | `DATA-REV-003`, `EXP-REV-004` | `EVID-REV-SYNREAL` | blocked | High |
| `R1-4b` | REPRODUCIBILITY | P0 | synthetic/train/test 数据详情和 split audit | `DATA-REV-001` | `EVID-REV-DATASET` | in_progress | High |
| `R1-4c` | GENERALIZATION | P0/P1 | 从原 12 人扩充到约 30 名实际完成者；分别报告专业/熟练 CSL 人员和 video-guided volunteers，不记录报名漏斗 | `DATA-REV-002` | `EVID-REV-REAL` | blocked | High |
| `R1-4d` | GENERALIZATION | P0 | participant-disjoint 测试不同执行者；只有专业/熟练 CSL 子集可支持自然 signer 泛化，志愿者仅支持 prompted reproduction | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `R1-5` | CLARITY/ROBUSTNESS | P1 | 先解释 hand-to-hand occlusion 区分机制；原文未明确要求独立实验，必要时复用 `R2-3` 证据 | `PAPER-REV-001`; optional `EXP-REV-003` | manuscript/evidence if needed | in_progress | Medium |
| `R1-6` | CLARITY | P2 | 当前 Methods 已有 4D 维度定义；补齐首次出现、单位和 tensor layout 并标记修订 | `PAPER-REV-001` | manuscript diff | in_progress | Low |

## Reviewer 2 — Manuscript

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `R2-1` | FAIR_COMPARISON | P0 | matched direct end-to-end baseline | `EXP-REV-001` | `EVID-REV-ARCH` | blocked | High |
| `R2-2` | FAIR_COMPARISON | P0 | DA method matrix under equal real-data budget | `EXP-REV-002` | `EVID-REV-DA` | blocked | High |
| `R2-3` | ROBUSTNESS | P0 | 在冻结 CSL clips 的小型 stress 子集上做 30°/60° 和 hand/object occlusion，报告 pose+translation；无需所有参与者重复全句库 | `DATA-REV-002`, `EXP-REV-003` | `EVID-REV-REAL` | blocked | High |
| `R2-4` | REPRODUCIBILITY | P0 | 补原文列出的 sign type/vocab/sentence count/average length/non-manual features | `DATA-REV-001` | `EVID-REV-DATASET` | in_progress | High |
| `R2-5` | EMPIRICAL_SUPPORT | P0 | spatial/channel/SE leave-one-out ablation | `EXP-REV-005` | `EVID-REV-ATTN` | blocked | High |
| `R2-6` | MISSING_BASELINE | P1 | 先做 cross-modal protocol feasibility audit，再决定重实现/定位 | `EXP-REV-007` | `EVID-REV-XMODAL` | not_started | Medium |

## Reviewer 2 — Code Availability

| ID | Type | Priority | Planned Action | Task IDs | Evidence IDs | Status | Risk |
|---|---|---|---|---|---|---|---|
| `R2-CODE-1` | REPRODUCIBILITY | P0 | 移除本地绝对路径，统一 config/CLI | `ARCH-REV-001` | clean-room run | in_progress | High |
| `R2-CODE-2` | REPRODUCIBILITY | P0 | 修复 SBERT 下载和评测模型准备 | `ARCH-REV-002` | `EVID-CODE-MODELS-V1`：pinned download/checksum + two-loader CPU smoke passed；response/final archive pending | evidence_ready | Low |
| `R2-CODE-3` | REPRODUCIBILITY | P0 | 重写 release README；清除不存在文件；release 排除 CLAUDE/internal docs | `ARCH-REV-003` | `EVID-CODE-RELEASE-V1` + `EVID-CODE-MT5-SMOKE-V1`；mT5 example 已验收，final archive pending | in_progress | High |
| `R2-CODE-4` | REVIEW_PROCESS | P0 | 作者确认许可证，添加 LICENSE 或修正文档 | `OPS-REV-002` | `EVID-CODE-RELEASE-V1` missing LICENSE；author decision pending | blocked | High |
| `R2-CODE-5` | REPRODUCIBILITY | P0 | Phi-3 二选一：完整可运行支持，或删除 unsupported claim | `ARCH-REV-004` | `EVID-CODE-MODEL-SUPPORT-V1`：public claim removed、legacy excluded、content gate zero-hit；response/final archive pending | evidence_ready | Low |

## Closure Gate

每条 item 标记 `done` 前必须同时满足：Direct Answer、verified evidence、manuscript revision、response revision 和 claim-strength audit。

所有使用 WaveLLM translation 的架构、DA、真实 stress 和 sim2real 行必须额外满足：

- 结果登记同一个 receipt-bound accepted language initialization；上传中的
  `MODEL-WAVELLM-HISTORICAL-BUNDLE-20260812` 不能作为 run input；
- 表中各方法使用相同语义初始化，不能把完整 CSL-News 新预训练混入某一方法；
- 无法复用 architecture-specific tensors 的 matched direct baseline 记录完整 tensor 差异，但保留相同 mT5
  初始化和受控数据/预算；
- canonical pose/radar/fusion modules、CSL-Daily/real-data manifests、split、预测和指标均以新 formal run
  生成；
- Methods/response 如实披露最终接受 asset 的受控初始化角色；在 historical bundle receipt/audit 通过前，
  不得对其 pose contract、split、evaluation JSON 或端到端指标作任何 reproduction claim。

`EVID-REV-REAL` 必须给出实际有效录制人数和 participant-disjoint split，并把
`professional_or_proficient_signer` 与 `video_guided_volunteer` 分开统计。约 30 人和 3--4 名专业/熟练
人员均为作者侧采集计划，不是审稿原文要求。若专业/熟练人员不足或为零，response 应直接说明，并把结论
限定为视频提示式 CSL 复现的跨参与者与方向/遮挡鲁棒性，不声称自然手语用户泛化。
