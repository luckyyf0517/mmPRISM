# Major Revision Advisor Discussion Brief

Status: current
Owner: Paper revision lane
Authority scope: 2026-08-12 major-revision advisor discussion agenda, decision options, and required outputs.
Last reviewed: 2026-08-12

## 会议目标

本次不逐条讨论文案，而是请老师对返修范围、资源和作者立场做决定。会后应能立即冻结实验优先级、
真实数据计划、资产责任人和发布边界。

**一句话结论：** 这是值得投入但证据密集的 major revision。编辑明确要求替代架构、域适配、
数据描述和真实世界泛化；只修改文字或补少量消融不足以再次送审。

规划目标为 `2026-11-11`。这是按编辑邮件“within three months”推算的内部日期，不是邮件给出的
硬性拒收日期；若真实数据无法及时落实，需要尽早联系编辑，而不是临近提交时补弱证据。

## 建议的 30 分钟议程

1. `5 min`：确认是否按完整大修投入，以及可以接受的 claim 降级边界。
2. `10 min`：拍板真实数据扩采、伦理边界、场景矩阵和负责人。
3. `8 min`：确认必须完成的模型对照实验与 GPU 预算。
4. `5 min`：确认原投稿资产、历史实验和仿真证据由谁补齐。
5. `2 min`：确认数据/代码发布、许可证和延期触发条件。

## 审稿意见合并视图

| 决策包 | 编辑/审稿人实际关切 | 当前证据 | 建议处置 | 风险 |
|---|---|---|---|---|
| 真实泛化 | `ED-SCI-5`, `R1-3`, `R1-4c/d`, `R1-5`, `R2-3`：方向、遮挡、不同用户和真实多样性是否成立 | 缺失；12 名 20–30 岁参与者不足以关闭问题 | 统一做 held-out-user + 方向 + 遮挡 stress matrix，同时报告 pose 与 translation | 最高 |
| 架构归因 | `ED-SCI-2`, `R2-1`：收益是否来自显式 pose bottleneck，而非容量差异 | 缺 matched direct baseline | 相同输入、split、优化预算和近似参数量，比较 direct radar-to-text | 最高 |
| 域适配 | `ED-SCI-3`, `R2-2`：shallow fine-tuning 是否在相同真实数据预算下有效 | 缺横向对照 | shallow/full/adversarial/MMD，统一数据预算、steps、seed，并报告效率 | 最高 |
| 数据可信度 | `ED-SCI-4`, `R1-4a/b`, `R2-4`：synthetic-real 是否接近，数据规模和 split 是否透明 | CSL-News metadata 仅部分就绪；原实验 provenance 未闭合 | 先冻结 manifest/split；做 paired 或 category-matched fidelity 分析 | 最高 |
| 模块与成本 | `R1-2`, `R2-5`：三类 attention 是否必要，4D encoder + LLM 成本如何 | 缺消融和统一 profile | attention leave-one-out；报告参数、GPU-hours、显存、延迟和吞吐 | 高 |
| 跨模态定位 | `R2-6`：为何没有 WiFi/声学连续手语 baseline | reviewer 意图与协议可比性不完全明确 | 先审计可比协议；只有数据、任务和指标可对齐时做数值比较 | 中 |
| 代码与合规 | `R2-CODE-1`–`5`, `ED-COMP-*`：第三方能否运行，提交材料是否完整 | mT5/SBERT 部分证据已就绪；LICENSE、最终 archive、availability、Source Data 未闭合 | 继续 clean-room release；老师/作者确认许可证和数据公开边界 | 高 |

详细逐条映射见 [review analysis](../50_VALIDATION/REVIEW_ANALYSIS.md) 和
[response tracker](../50_VALIDATION/RESPONSE_TRACKER.md)。审稿原文的中文工作版见
[review CN](../../logs/2026/08/20260811_REVIEW_CN.md)。

## 请老师拍板的六件事

### D1. 返修投入和论文主张

**建议：** 接受完整大修，优先补决定性证据；若关键对照为负结果，诚实缩小“必要性、最优性、
普适性”主张，不为保留原叙事选择性报告。

需要决定：

- 是否确认以完整返修为目标，并允许根据实验结果调整论文核心表述？
- 哪些贡献是作者组必须保留的，哪些可以改为 modularity、interpretability 或 lightweight trade-off？

### D2. 真实数据扩采和伦理边界

**建议的最小实验结构：** unseen-user holdout；`0°/30°/60°`；无遮挡、双手重叠、物体部分遮挡；
pose reconstruction 和 translation 同时评估。参与者数量、人口统计字段和是否可公开必须根据伦理、
同意书和统计设计确认，不在会上凭经验虚构数字。

需要决定：

- 能否新增采集？可投入的参与者、设备时间、采集人员和完成日期是什么？
- 现有伦理审批/豁免与同意书是否覆盖新增参与者、上述场景和数据发布？
- 真实 radar 数据可公开到什么粒度；不能公开时，合理的受限访问流程是什么？

### D3. 核心实验与算力预算

**建议 P0 顺序：** 先恢复原结果与冻结协议，再并行执行 direct baseline、DA matrix、真实 stress、
synthetic-real fidelity、attention ablation，最后统一做 compute profile。跨模态 baseline 暂列 P1。

需要决定：

- 可用 GPU 类型、数量和连续时间窗口是什么？
- 公平对照采用多少 seeds；在预算不足时优先保留哪些实验？
- 是否认可 cross-modal 先做 feasibility audit，而不预先承诺不可比的数值表？

### D4. 原投稿与历史资产责任人

当前最关键的阻塞不是写作，而是原投稿结果与数据身份未闭合。请现场指定每类资产的提供者和日期：

| 必需资产 | 当前缺口 | 负责人 | 日期 |
|---|---|---|---|
| 原投稿最终 PDF/TeX、supplement、原表图 | 无法做新旧稿和数值差异审计 | 待定 | 待定 |
| paper-facing checkpoint、prediction、split、metric output | 原结果无法复算或绑定 evidence | 待定 | 待定 |
| 12 人真实数据协议、metadata、伦理/同意范围 | 无法设计合规扩采与公开方案 | 待定 | 待定 |
| radar acquisition、channel map、阵列与校准 | 论文与 legacy 物理配置存在冲突 | 待定 | 待定 |
| MANO/mesh/ray-tracing simulator、输入与配置 | 当前可见 legacy 与稿件方法描述不一致 | 待定 | 待定 |

### D5. 数据、代码和许可证

**建议：** 正式 release 只声称已经 clean-room 验证的 mT5 路径；不恢复 Phi-3 伪支持。数据和权重
分别按来源、同意书与第三方许可决定公开或受限访问，不能笼统写 “available on request”。

需要决定：

- 作者组批准使用哪一种代码许可证？
- 哪些自采数据、派生数据、模型权重可以公开，哪些只能受限访问？
- 谁负责最终 repository/archive/DOI 和访问测试？

### D6. 时间表和联系编辑的触发条件

**建议：** 会上设明确触发条件。若 `2026-08-26` 前仍无法确认真实数据采集与伦理边界，或
`2026-09-09` 前没有可验收的首批 stress/new-user 数据，则由通讯作者评估立即联系编辑说明进度并申请
延期。日期是内部风险控制建议，可由老师调整。

需要决定：通讯作者、内部初审日期、延期联系人和触发阈值。

## 不建议在会上承诺的事项

- 不承诺两阶段架构、shallow adaptation 或所有 attention 一定更优；先跑公平对照。
- 不承诺 WiFi/声学数值 baseline，除非协议审计证明可比。
- 不把新增参与者人数、统计显著性或 synthetic-real closeness 作为尚未测量的既定结论。
- 不承诺 MIT 或其他许可证，除非全体作者确认并核对第三方边界。
- 不使用当前 supplementary S2-S6 占位数字，也不把工程 smoke 指标晋升为论文结果。

## 会后必须形成的记录

| Decision | 结论 | Owner | Due date | Evidence/next action |
|---|---|---|---|---|
| `D1` 返修与 claim 边界 | 待会议记录 | 待定 | 待定 | 待定 |
| `D2` 真实采集与伦理 | 待会议记录 | 待定 | 待定 | 待定 |
| `D3` 实验与算力 | 待会议记录 | 待定 | 待定 | 待定 |
| `D4` 历史资产 | 待会议记录 | 待定 | 待定 | 待定 |
| `D5` 发布与许可证 | 待会议记录 | 待定 | 待定 | 待定 |
| `D6` 时间与延期触发 | 待会议记录 | 待定 | 待定 | 待定 |

会后将已确认结论写入 dated Log；只有改变当前边界、合同或论文策略的结论才同步对应 Authority，
本会前简报不作为实验结果或作者决定的证据。
