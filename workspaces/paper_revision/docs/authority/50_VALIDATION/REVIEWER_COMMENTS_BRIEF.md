# 审稿意见汇总与分析

Status: current
Owner: Paper revision lane
Authority scope: 面向作者讨论的编辑与审稿意见完整汇总、问题分析和建议处理方向。
Last reviewed: 2026-08-12

## 1. 总体判断

稿件收到的是有明确返修机会的 major revision，不是对研究问题或基本正确性的否定。两位审稿人都能
准确理解论文从毫米波手部重建到连续手语翻译的两阶段主线；Reviewer 1 还明确肯定了语言语义解释、
隐私友好感知和双手交互的价值。

但编辑认为现有证据不足以支撑当前结论，并明确指出：若没有实质性修改，稿件很可能不会再次送审。
意见的核心不是补充文字，而是验证四件事：两阶段架构是否必要、域适配是否公平有效、合成数据与真实
数据是否可信，以及系统能否在真实用户、方向和遮挡条件下泛化。

综合优先级：

- `P0`：影响编辑是否再次送审，必须以新增证据或明确的 claim 降级处理。
- `P1`：重要，但存在协议可比性或资源边界，应先论证可行性。
- `P2`：主要通过论文修改、引用或说明即可关闭。

## 2. 编辑意见

### `ED-SCI-1` 整体框架验证不足

**意见：** 编辑要求 substantially stronger validation of the proposed framework。

**分析：** 这是对所有科学意见的总括。当前实验主要证明系统能够工作，但不足以区分性能来自核心设计、
模型容量、数据规模还是训练策略。只增加一个消融或多写局限性不能关闭该意见。

**建议处理：** 用一组相互配合的实验回答，而不是堆零散结果：direct end-to-end baseline、域适配对照、
真实 stress/new-user 测试、synthetic-real fidelity、attention 消融和统一效率分析。

**优先级：** `P0`。

### `ED-SCI-2` 缺少替代架构比较

**意见：** 编辑明确要求 comparisons against alternative architectures，对应 `R2-1`。

**分析：** 当前无法证明显式 pose reconstruction 是性能提升的原因。直接模型若容量、数据或训练预算不同，
比较仍然无效。

**建议处理：** 构建由相同 4D voxel feature 直接接入 LLM 的 baseline，对齐数据、split、优化策略、训练
步数和尽可能接近的参数预算，同时报告 pose bottleneck 带来的精度、可解释性与模块化权衡。

**优先级：** `P0`。

### `ED-SCI-3` 缺少域适配策略比较

**意见：** 编辑明确要求 comparisons against domain adaptation strategies，对应 `R2-2`。

**分析：** 现有 shallow fine-tuning 可能只是一个轻量选择，尚不能声称最优或更高效。比较必须控制真实
数据量，否则不同方法的结论不可解释。

**建议处理：** 在相同 synthetic checkpoint、真实数据预算、训练步数和 seed 下比较 shallow fine-tuning、
full fine-tuning、adversarial DA 和 MMD，并同时报告准确率、可训练参数、训练时间和显存。

**优先级：** `P0`。

### `ED-SCI-4` 数据集描述不足

**意见：** 编辑要求 clearer dataset characterization，对应 `R1-4b` 和 `R2-4`。

**分析：** 目前“12 subjects、200k frames”不足以判断数据覆盖、split 隔离和结论适用范围，也无法支持
第三方复现。

**建议处理：** 补全数据来源、手语类型、词汇、句子、句长、帧数、参与者、session、scene、方向、
非手部语法、标注、缺失值与 split group；所有统计从冻结 manifest 自动生成。

**优先级：** `P0`。

### `ED-SCI-5` 真实世界泛化证据不足

**意见：** 编辑要求 evidence of real-world generalization，对应 `R1-3`、`R1-4c/d`、`R1-5` 和 `R2-3`。

**分析：** 这是风险最高的一组意见。现有 12 名 20–30 岁参与者和可控实验室测试不能支撑跨用户、方向、
遮挡与环境的强泛化表述。

**建议处理：** 统一设计 held-out-user、`0/30/60` 度方向、双手重叠、部分手部遮挡和物体遮挡测试，
同时报告重建与翻译结果、置信区间和失败案例。若无法扩采，应降低人群泛化结论并说明限制。

**优先级：** `P0`。

### `ED-WRITE-1/2` 写作不够克制

**意见：** 删除 `new`、`novel`、`first`、`unique`、`unprecedented` 等首创表达，以及 `superior`、
`remarkable`、`pave the way` 等主观或夸张表达。

**分析：** 这是编辑的明确写作要求，不取决于新实验是否成功。不能只机械替换词，还要检查对应句子的
证据强度和适用边界。

**建议处理：** 全稿逐项审计；可验证的句子改为具体、量化表述，证据不足的 necessity、optimality 和
generalization claim 直接缩小。

**优先级：** `P0`，但执行成本低于科学实验。

### `ED-WRITE-3/4/5` 返修写作格式

**意见：** response letter 必须逐字复现审稿意见并逐条回复；稿件修改必须使用 track changes 或颜色标记；
无法完成的请求必须明确解释。

**分析：** 属于提交形式的硬要求。不能用任务已完成代替 response 中的 direct answer、证据和稿件位置。

**建议处理：** 每条按“直接回答 -> 新证据 -> 稿件修改位置 -> 局限性”组织；未满足项说明客观原因和
替代分析，不回避问题。

**优先级：** `P0`，提交前关闭。

### `ED-COMP-1` 至 `ED-COMP-12` 合规要求

**意见：** 完成代码/软件 checklist、机器学习 checklist、色觉友好图表、Data Availability、Code
Availability、持久化数据存储、Source Data、ORCID、必要的作者变更表和文献更新，并提交 revised
manuscript、supplement、response、cover letter 等完整材料。

**分析：** 这些不是可选润色。当前 Data/Code Availability 正式章节缺失，20 个 display item 的科学
provenance 尚未闭合，Supplementary Tables S2-S6 仍含未验证占位数据。

**建议处理：** 为每个表图绑定 raw/processed input、sample-level values、run、checkpoint、metric、脚本
和 hash；据此生成 Source Data。仓库、DOI、许可证和访问方式只能在实际验证后写入。

**优先级：** `P0`，与实验和发布工作同步推进。

## 3. Reviewer 1

### `R1-POS` 正面评价

**意见：** 审稿人肯定从传统毫米波分类走向语言语义解释的方向，认为隐私友好的双手交互感知具有应用
价值，并对文中报告的 optical-level fidelity 表现出积极态度。

**分析：** 说明研究问题和总体叙事有吸引力，返修应保留“geometry-guided wireless sign-language
understanding”主线。但正面评价引用了当前性能数字，该数字仍必须完成 provenance 审计，不能把赞扬
当作指标已经被验证。

### `R1-1` 补充 RadarLLM 和 mmExpert

**意见：** 建议引用并讨论 RadarLLM（AAAI 2026）与 mmExpert（MobiHoc 2025）。

**分析：** 不能只在参考文献中增加条目，需要说明它们与本文在输入表示、任务、LLM 用法、数据生成和
输出目标上的关系。文献的正式出版信息也应核实。

**建议处理：** 在 related work 中做实质比较，避免使用“首次将 LLM 用于毫米波”等可能被这些工作
直接反驳的表述。

**优先级：** `P2`。

### `R1-2` 训练和推理成本

**意见：** 4D volumetric encoder 与 LLM 组合计算开销较大，需要展示训练和推理成本。

**分析：** 单报参数量不足以回答部署和复现成本。成本还应与关键 baseline 在相同硬件、输入长度和 batch
条件下比较。

**建议处理：** 报告总参数/可训练参数、FLOPs 或 MACs、GPU-hours、峰值显存、batch-1 latency 和
throughput，并给出效果与成本的权衡。

**优先级：** `P0`，可在模型冻结后统一 profile。

### `R1-3` 跨方向泛化

**意见：** 毫米波对传播路径敏感，用户不一定始终正对并严格对齐雷达，需要 cross-orientation 分析。

**分析：** 这是现实使用条件，而非普通噪声增强。训练和测试是否跨方向、角度标签如何获取、是否包含
同一用户泄漏都必须清楚。

**建议处理：** 至少报告正对与 off-axis 条件的分层结果；可采用 `0/30/60` 度，但应按真实采集能力和
校准精度确定。pose 与 translation 都需评估。

**优先级：** `P0`。

### `R1-4a` 合成数据与真实数据的接近程度

**意见：** 论文主要依赖合成数据，需要直接衡量 synthetic data 与 ground truth 的接近程度。

**分析：** 仅凭 synthetic-trained model 在真实数据上的下游表现，不能说明信号分布本身接近。审稿人也
可能期待相同动作/句子的 paired 比较。

**建议处理：** 优先构建 paired synthetic-real set；若无法逐样本配对，则使用 category-matched set 并
明确限制。报告信号统计、冻结特征分布、可分性/检索和 downstream transfer，避免用单一距离概括真实性。

**优先级：** `P0`。

### `R1-4b` 合成、训练与测试数据细节

**意见：** 需要说明合成数据的 diversity、signs、size、environment domain，以及训练集和测试集构成。

**分析：** 这既是复现问题，也是验证 split 是否真正跨用户、跨场景的前提。描述性文字不能替代可审计
manifest。

**建议处理：** 与 `ED-SCI-4`、`R2-4` 合并处理，公开或提供冻结 manifest、split hash 和统计生成方法。

**优先级：** `P0`。

### `R1-4c` 真实数据规模和多样性

**意见：** 12 名、20–30 岁参与者在规模和多样性上有限，建议利用可用手语/书写数据并采集更多 radar
数据。

**分析：** “更多”没有给出固定人数，不能随意承诺样本量。应依据伦理许可、统计设计和可行资源确定，
并诚实报告年龄、手型、手势风格等可合法收集的变量。

**建议处理：** 优先增加能够回答 unseen-user 和 stress condition 的参与者，而不是只增加同分布帧数；
无法扩充时缩小 population claim。

**优先级：** `P0/P1`，取决于采集与伦理可行性。

### `R1-4d` 新用户泛化与重训练需求

**意见：** 需要证明系统可迁移到不同手型、手部大小和手势风格的新用户，且不需要大量重新训练。

**分析：** 普通随机 frame split 不能回答该问题；必须按 participant 隔离。还应区分 zero-shot、少样本
适配和 full retraining。

**建议处理：** 使用 held-out-user split，报告 zero-shot 和固定少量 calibration budget 下的曲线、方差与
置信区间，明确“不需要大量重训练”的操作定义。

**优先级：** `P0`。

### `R1-5` 双手重叠和遮挡

**意见：** 两手从雷达视角重叠时，稀疏点云可能坍缩成一个簇，需要说明模型如何区分 hand-to-hand
occlusion。

**分析：** 只解释 temporal/geometry prior 不足以证明模型确实解决了 ambiguity。该问题与方向测试和
真实 stress test 高度重合。

**建议处理：** 按遮挡程度分层报告双手 pose 与 translation；展示输入、重建、预测和失败案例，再结合
时序先验解释机制。若模型无法稳定区分，应把它写成明确失败边界。

**优先级：** `P0`。

### `R1-6` 主文中的 4D cube 符号

**意见：** 不能只在 supplementary 解释 4D cube，主文也需引入相关 notation。

**分析：** 当前主文已有部分维度描述，但仍需检查首次出现位置、每个维度的物理含义和单位、索引顺序及
tensor layout 是否完整。

**建议处理：** 在 Methods 首次使用前增加统一定义，并确保正文、图、代码和数据契约的轴顺序一致。

**优先级：** `P2`。

## 4. Reviewer 2：论文内容

### `R2-POS` 对论文主线的理解

**意见：** 审稿人准确概括了 CubeNet 重建双手骨架、mT5 翻译自然语言的两阶段框架。

**分析：** 说明方法主线总体可理解。其后续问题集中在“为何必须这样设计”以及实验是否足以归因，而不是
未理解论文。因此回应应提供控制变量证据，不宜只重复架构动机。

### `R2-1` 两阶段架构必要性

**意见：** 当前优势可能来自更大容量或更强特征，而非显式姿态重建；需要参数和训练策略对齐的 direct
4D voxel-to-LLM baseline。

**分析：** 该意见直接挑战论文核心设计归因，也是编辑明确点名的 alternative architecture comparison。
如果 direct baseline 相近，两阶段仍可从可解释性、模块化或数据效率角度定位，但不能再声称性能必要性。

**建议处理：** 完成 matched direct baseline，统一 split、优化器、训练 budget、seed 和 metric；同时报告
精度、成本、可解释性及误差传播。

**优先级：** `P0`。

### `R2-2` 域适配方法是否最优

**意见：** shallow fine-tuning 缺少与 full fine-tuning、adversarial DA、MMD 的公平横向比较。

**分析：** “optimal”是高风险主张。即使 shallow 方法不是最高精度，也可能位于 accuracy-efficiency
Pareto frontier，回应不应预设结果。

**建议处理：** 使用相同真实数据量和训练预算，报告多 seed 的 pose/translation 指标、可训练参数、时间
和显存；根据结果决定表述为 best、competitive 或 lightweight option。

**优先级：** `P0`。

### `R2-3` 真实边界场景测试

**意见：** 当前噪声实验过于可控，需要在 `30/60` 度偏轴、部分手部遮挡和物体遮挡下评估 reconstruction
与 translation。

**分析：** 审稿人给出了较明确的最小场景。它与 `R1-3` 和 `R1-5` 可合并为一个信息密集的真实 stress
matrix，避免重复采集和重复实验。

**建议处理：** 分条件报告两个任务，不只给总体均值；保留场景 metadata 和失败案例，并严格按用户隔离
split。

**优先级：** `P0`。

### `R2-4` 数据集透明度

**意见：** 补充手语类型、词汇量、句子数、平均句长和非手部语法特征等完整统计。

**分析：** 非手部语法特征尤其重要，因为 radar-hand pipeline 可能天然无法观察面部和身体语法线索。
该项不仅要求统计，也可能暴露任务定义的适用边界。

**建议处理：** 补齐 reviewer 明确列出的字段，并说明数据是否包含、忽略或无法感知非手部信息；同步报告
subject、session、scene、split 和 annotation 统计。

**优先级：** `P0`。

### `R2-5` 三种 attention 是否冗余

**意见：** spatial、channel 和 SE attention 同时加入但无单组件消融，可能属于任意模块堆叠。

**分析：** leave-one-out 可以检验在完整模型中的边际贡献，但若多个模块功能重叠，最好同时包含无
attention base，必要时补单组件组合。无贡献模块应删除或弱化其设计主张。

**建议处理：** 至少比较 full、w/o spatial、w/o channel、w/o SE 和 base-none，统一 data、seed 和 budget，
报告 reconstruction 与 downstream translation。

**优先级：** `P0`。

### `R2-6` WiFi/声学跨模态 benchmark

**意见：** 现有 baseline 只含毫米波方法，建议加入 1–2 个 WiFi 或声学 continuous SLU baseline。

**分析：** 诉求是把工作放进 broader non-contact sensing 语境，但不同模态通常没有相同输入、采集数据、
词汇和 split。直接搬公开数字可能制造不公平比较。该意见的具体预期存在不确定性。

**建议处理：** 先审计是否存在任务、数据和指标可对齐的公开方法；可比时重实现 1–2 个 baseline。不可比
时扩充 related work，明确 modality/protocol 差异，并解释为何不进行误导性的数值排名。

**优先级：** `P1`。

## 5. Reviewer 2：代码可用性

### `R2-CODE-0` 总体评价

**意见：** 核心模型、数据、评估和仿真代码已经提交，但 README/CLAUDE 与代码不一致，路径硬编码且
缺少 LICENSE。

**分析：** 审稿人实际执行或深入检查了代码，后续回复必须对应可运行 artifact，不能只称“已修复”。
最终目标是第三方从安装、下载、示例输入到评测能够完成 clean-room reproduction。

### `R2-CODE-1` 本地路径硬编码

**意见：** 多个统计、数据拆分、FMCW、压缩解压、仿真和标注脚本包含作者本地绝对路径，第三方无法运行。

**分析：** 逐个替换字符串仍容易遗漏。正式实现应统一通过配置和 CLI 注入 data root、artifact root、模型
位置、device 和 precision；release 还应扫描绝对路径。

**建议处理：** 公开 release 以 canonical `src/mmprism/` 路径为准，提供可执行配置和最小示例；legacy
代码作为历史参考排除出 reviewer archive，而不是继续扩展。

**优先级：** `P0`。

### `R2-CODE-2` SBERT 下载与评估链断裂

**意见：** 下载脚本只准备 SimCSE，评估却强制加载 SBERT，按 README 执行必然失败。

**分析：** 这是明确、可复现的发布缺陷。当前 canonical 模型下载、checksum 和 CPU loader smoke 已形成
初步证据，但仍需纳入最终 archive 的 clean-room 测试。

**建议处理：** 固定模型标识、revision、下载位置和 checksum；评估入口应验证依赖并给出明确错误，README
命令必须在无作者缓存环境执行通过。

**优先级：** `P0`；当前为 `evidence_ready`，最终 release 尚待关闭。

### `R2-CODE-3` README 漂移和内部 CLAUDE.md

**意见：** README/CLAUDE 引用了不存在的 config 和 temporal/TVAN 模块；CLAUDE.md 属于内部开发说明，
不应进入学术代码档案。

**分析：** 问题本质是发布内容与实际支持范围不一致。删除单个文件不够，还需用 manifest 固定最终 archive
中允许的代码、配置、文档和示例。

**建议处理：** 公开 README 只描述经过验证的路径；release manifest 排除内部 agent 文档、paper manager、
legacy、凭据和未授权资产，并对 archive 做内容审计。

**优先级：** `P0`。

### `R2-CODE-4` 缺少 LICENSE

**意见：** README 声称 MIT，但仓库没有 LICENSE 文件。

**分析：** 不能根据旧 README 自动补 MIT。许可证是作者决策，还受第三方代码、模型、数据和权重许可
约束。

**建议处理：** 全体作者确认代码许可证；分别记录代码、数据、模型权重和第三方资产的发布边界，再让
README、LICENSE、Code Availability 和 release archive 保持一致。

**优先级：** `P0`，目前受作者决定阻塞。

### `R2-CODE-5` Phi-3 无可运行路径

**意见：** README 强调 Phi-3，但没有配置、运行脚本和完整训练/评估示例。

**分析：** 若论文贡献不依赖 Phi-3，删除不实支持声明比为应付意见临时扩展模型更稳妥。当前正式支持边界
已收敛到经过验证的 mT5，legacy Phi-3 不进入 release。

**建议处理：** 在 response 中明确正式 release 只支持 mT5，并给出完整 mT5 train/eval 示例与验证证据；
全文和 archive 扫描不再出现 Phi-3 支持 claim。

**优先级：** `P0`；支持边界已确定，最终 archive 尚待关闭。

## 6. 合并后的核心问题

1. **架构归因：** 显式 pose bottleneck 是否真正贡献性能，还是仅增加容量？
2. **域适配公平性：** shallow fine-tuning 在相同真实数据预算下处于什么位置？
3. **真实泛化：** 方向、遮挡、不同环境和未见用户下能否维持重建与翻译质量？
4. **数据可信度：** synthetic-real 差距、数据组成、split 和历史结果是否可审计？
5. **模块与效率：** attention 组件是否各有贡献，系统资源成本是否合理？
6. **跨模态定位：** WiFi/声学方法是否存在公平、可复现的比较协议？
7. **代码与合规：** reviewer 能否从干净环境运行，数据/代码/Source Data 声明是否真实一致？

科学工作应先恢复原投稿结果并冻结 dataset/split/metric protocol，再执行新增比较。任何负结果都应驱动
论文主张调整，而不是被排除。逐条执行状态以 [response tracker](RESPONSE_TRACKER.md) 为准，实验优先级
和最小协议见 [review analysis](REVIEW_ANALYSIS.md)。
