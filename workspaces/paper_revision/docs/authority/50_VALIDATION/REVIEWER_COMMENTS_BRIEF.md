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

结合当前投稿正文核对后，这些意见并非只要求新增实验。当前稿还需要系统补齐 Methods、数据与 split、
雷达输入、tensor/坐标、训练、解码、指标和 baseline 协议。否则即使增加结果表，也仍无法建立公平比较和
可复现的证据链。当前正文核对以 `sn-article.tex` 实际加载内容及正式 supplementary 为准。

综合优先级：

- `P0`：影响编辑是否再次送审，必须以新增证据或明确的 claim 降级处理。
- `P1`：重要，但存在协议可比性或资源边界，应先论证可行性。
- `P2`：主要通过论文修改、引用或说明即可关闭。

## 2. 编辑意见

### `ED-SCI-1` 整体框架验证不足

**意见：** 编辑要求 substantially stronger validation of the proposed framework。

**分析：** 这是对所有科学意见的总括。当前实验主要证明系统能够工作，但不足以区分性能来自核心设计、
模型容量、数据规模还是训练策略。进一步核对发现，当前 Methods 中 pose reconstruction、temporal
aggregation、confidence fusion 和 mT5 generation 的核心架构内容没有形成有效正文，而 Results 已对这些
模块作出较强机制归因。只增加一个消融或多写局限性不能关闭该意见。

**建议处理：** 一方面用相互配合的实验回答：matched direct baseline、域适配对照、真实 stress/new-user
测试、synthetic-real fidelity、attention 消融和统一效率分析；另一方面同步重写 Methods，使每项结果都能
绑定明确的输入、模型、训练和指标协议。

**优先级：** `P0`。

### `ED-SCI-2` 缺少替代架构比较

**意见：** 编辑明确要求 comparisons against alternative architectures，对应 `R2-1`。

**分析：** 当前 Table 2 已列出 `End-to-End (Cube)`，因此问题不是完全没有 direct baseline，而是正文没有
定义该 baseline 的 encoder/projector、参数量、LLM、预训练方式和训练预算。现有一行结果不足以证明显式
pose reconstruction 是性能提升原因；直接模型若容量、数据或训练预算不同，比较仍然无效。

**建议处理：** 构建由相同 4D voxel feature 直接接入 LLM 的 baseline，对齐数据、split、优化策略、训练
步数和尽可能接近的参数预算，同时报告 pose bottleneck 带来的精度、可解释性与模块化权衡。

**优先级：** `P0`。

### `ED-SCI-3` 缺少域适配策略比较

**意见：** 编辑明确要求 comparisons against domain adaptation strategies，对应 `R2-2`。

**分析：** 当前稿只描述冻结深层、微调 stem 的 shallow alignment，并主要用 t-SNE cluster mixing 支撑
domain alignment。t-SNE 不能证明方法最优、真实信号已对齐或下游泛化。现有 shallow fine-tuning 可能只是
一个轻量选择；比较必须控制真实数据量，否则不同方法的结论不可解释。

**建议处理：** 在相同 synthetic checkpoint、真实数据预算、训练步数和 seed 下比较 shallow fine-tuning、
full fine-tuning、adversarial DA 和 MMD，并同时报告准确率、可训练参数、训练时间和显存。

**优先级：** `P0`。

### `ED-SCI-4` 数据集描述不足

**意见：** 编辑要求 clearer dataset characterization，对应 `R1-4b` 和 `R2-4`。

**分析：** 当前 Methods 的 “Datasets and synthetic data generation” 实际主要描述仿真；“12 subjects、
200k frames”只出现在 supplementary 场景说明中。稿件没有给出数据版本、每阶段数据量、participant/
session/scene split、词汇与句子覆盖，因此不足以判断泄漏、数据覆盖和结论适用范围，也无法支持第三方复现。

**建议处理：** 补全数据来源、手语类型、词汇、句子、句长、帧数、参与者、session、scene、方向、
非手部语法、标注、缺失值与 split group；所有统计从冻结 manifest 自动生成。

**优先级：** `P0`。

### `ED-SCI-5` 真实世界泛化证据不足

**意见：** 编辑要求 evidence of real-world generalization，对应 `R1-3`、`R1-4c/d`、`R1-5` 和 `R2-3`。

**分析：** 这是风险最高的一组意见。当前稿展示了距离、low-light、multi-person 和 reflector 等场景，但没有
方向分层、受控手部/物体遮挡、严格 held-out-user 和适配预算。Supplementary 中声称 generalization/noisy
performance 的对应表仍为 `placeholder_unverified`，不能支撑跨用户、方向、遮挡与环境的强泛化表述。

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

**分析：** 当前稿没有报告总/可训练参数、FLOPs、训练硬件与时长、显存、latency 或 throughput，Methods
也没有完整训练配置。单报参数量仍不足以回答部署和复现成本；成本应与关键 baseline 在相同硬件、输入
长度和 batch 条件下比较。

**建议处理：** 报告总参数/可训练参数、FLOPs 或 MACs、GPU-hours、峰值显存、batch-1 latency 和
throughput，并给出效果与成本的权衡。

**优先级：** `P0`，可在模型冻结后统一 profile。

### `R1-3` 跨方向泛化

**意见：** 毫米波对传播路径敏感，用户不一定始终正对并严格对齐雷达，需要 cross-orientation 分析。

**分析：** 当前稿按 near/mid/far 距离报告结果，也描述若干环境干扰，但距离或环境变化不能替代
cross-orientation。这是现实使用条件，而非普通噪声增强；训练和测试是否跨方向、角度标签如何获取、是否
包含同一用户泄漏都必须清楚。

**建议处理：** 至少报告正对与 off-axis 条件的分层结果；可采用 `0/30/60` 度，但应按真实采集能力和
校准精度确定。pose 与 translation 都需评估。

**优先级：** `P0`。

### `R1-4a` 合成数据与真实数据的接近程度

**意见：** 论文主要依赖合成数据，需要直接衡量 synthetic data 与 ground truth 的接近程度。

**分析：** 当前稿用 t-SNE 的 synthetic/real feature mixing 支撑对齐，但 t-SNE 不能度量原始信号接近程度，
也不能单独证明下游 transfer。仅凭 synthetic-trained model 的结果不能说明信号分布本身接近；审稿人也
可能期待相同动作/句子的 paired 比较。

**建议处理：** 优先构建 paired synthetic-real set；若无法逐样本配对，则使用 category-matched set 并
明确限制。报告信号统计、冻结特征分布、可分性/检索和 downstream transfer，避免用单一距离概括真实性。

**优先级：** `P0`。

### `R1-4b` 合成、训练与测试数据细节

**意见：** 需要说明合成数据的 diversity、signs、size、environment domain，以及训练集和测试集构成。

**分析：** 当前稿没有列出 synthetic/train/test 的数据身份、样本或序列数量、真实/合成占比和 split unit。
这既是复现问题，也是验证 split 是否真正跨用户、跨场景的前提。描述性文字不能替代可审计 manifest。

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

**分析：** 当前 Results 已声称 temporal context 可以推断被遮挡 joint，fusion 也可在低 confidence 时依赖
radar feature，但稿件没有定义 occlusion 条件、pose confidence 的来源或分层结果。只解释 temporal/geometry
prior 不足以证明模型确实解决了 ambiguity；该问题与方向测试和真实 stress test 高度重合。

**建议处理：** 按遮挡程度分层报告双手 pose 与 translation；展示输入、重建、预测和失败案例，再结合
时序先验解释机制。若模型无法稳定区分，应把它写成明确失败边界。

**优先级：** `P0`。

### `R1-6` 主文中的 4D cube 符号

**意见：** 不能只在 supplementary 解释 4D cube，主文也需引入相关 notation。

**分析：** 当前主文已经给出 Doppler/Range/Azimuth/Elevation 四轴和部分 FFT/beamforming 公式，所以不是
完全没有 notation；但仍缺每轴 bin/单位、索引顺序、最终 shape、normalization、时间轴、dtype 及输入模型
前的 layout。该意见只能判为部分覆盖，不能直接关闭。

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

**分析：** 当前 Table 2 已有 `End-to-End (Cube)` 一行，但没有 matched architecture definition、参数量和
训练公平性说明。因此 Reviewer 的实质要求仍未满足。该意见直接挑战论文核心设计归因，也是编辑明确点名的
alternative architecture comparison；如果 direct baseline 相近，两阶段仍可从可解释性、模块化或数据效率
角度定位，但不能再声称性能必要性。

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

**分析：** 当前 Introduction 明确承认 sign language 包含 facial expression 和 body posture，但 Methods 没有
说明数据是否覆盖这些线索、系统实际输入包含哪些上肢/手部 joint、任务是否筛除了依赖 non-manual grammar
的句子。该项不仅要求统计，也决定任务定义和结论适用边界。

**建议处理：** 补齐 reviewer 明确列出的字段，并说明数据是否包含、忽略或无法感知非手部信息；同步报告
subject、session、scene、split 和 annotation 统计。

**优先级：** `P0`。

### `R2-5` 三种 attention 是否冗余

**意见：** spatial、channel 和 SE attention 同时加入但无单组件消融，可能属于任意模块堆叠。

**分析：** 当前 Results 和图注将三类 attention 描述为协同定位手部结构并抑制 multipath，但没有对应的
spatial/channel/SE leave-one-out。机制性表述强于现有证据。leave-one-out 可以检验完整模型中的边际贡献；
若多个模块功能重叠，还应包含无 attention base，必要时补单组件组合。

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
4. **数据可信度：** synthetic-real 差距、数据组成、split 和 paper-facing 结果是否可审计？
5. **模块与效率：** attention 组件是否各有贡献，系统资源成本是否合理？
6. **跨模态定位：** WiFi/声学方法是否存在公平、可复现的比较协议？
7. **代码与合规：** reviewer 能否从干净环境运行，数据/代码/Source Data 声明是否真实一致？

## 7. 结合当前稿必须完成的横向修订

以下内容并非新增审稿意见，而是关闭上述意见时必须同步补齐的论文技术说明。

### 7.1 重建 Methods 技术闭环

当前稿需要把 Results 中实际依赖的模块正式写入 Methods：CubeNet backbone、spatial/channel/SE、PAFPN、
temporal encoder 与 aggregation、dual-hand regression、STGCN、pose confidence、radar feature fusion、mT5
输入与 generation。每个模块至少给出输入输出 shape、关键层配置、mask、可训练范围和完整 tensor flow。

### 7.2 固定雷达输入与几何契约

补充硬件型号、ADC/chirp/frame 配置、TDM/channel map、array/calibration、FFT/window/crop、clutter removal、
angle grid、beamforming 和 cube normalization。明确 cube 的轴顺序、bin/单位、时间轴和 dtype；明确双手 joint
数量、拓扑、左右手顺序、坐标原点/轴向/单位、camera-radar 标定、valid mask 和 confidence。

### 7.3 明确数据、任务与 split

分别列出 cam-pose、synthetic radar/pose、real radar 和 translation 数据的来源、版本、规模、语言、词汇、
句子、帧率、参与者、session、scene 和标注方式。明确输入是否预分段，区分 isolated gesture、SLR、
continuous SLT/SLU，并给出 subject/session/scene-level train/validation/test split 与泄漏审计。

### 7.4 披露训练、适配与解码协议

每个 stage 都应报告 optimizer、learning rate/schedule、batch/effective batch、epochs/steps、seed、precision、
hardware、gradient clipping、augmentation、freeze/unfreeze、real-data budget、checkpoint selection。mT5 还需
给出 exact model/revision、tokenizer、prompt/embedding injection、maximum length 和 generation parameters。

### 7.5 冻结指标与统计口径

Pose 指标需明确 MPJPE/PCK 的单位、PCK threshold、absolute/root-relative、joint mask、对齐与 frame/sequence/
subject 聚合。Translation 指标需固定 BLEU-4、ROUGE-L、SBERT、SimCSE 的实现、tokenization、model revision、
pooling 和 aggregation。

当前 `84.30% Rel. Perf.` 可由表中四个指标各自相对 Vision Oracle 的比率再取无权平均复算得到，但正文
没有定义公式，仍需作者确认这是否是实际计算口径。优先逐项报告相对差距；若保留 composite，必须给出
公式、权重依据、敏感性和 uncertainty，避免把它直接称为标准的 optical-level fidelity。

### 7.6 重新建立 baseline 公平性

为 End-to-End、Heatmap 和 Point-based baseline 给出明确 architecture、parameter count、input representation、
LLM、pretraining、data budget、optimizer、seed 和 compute。作者构造的输入表示 baseline 与具名外部工作应
分开呈现；新的 direct baseline 必须尽可能对齐参数和训练策略。

### 7.7 增加 uncertainty 与失败边界

主表和图不能只给 point estimate。按实验单位报告样本量、多 seed 或重复采集、标准差/置信区间、配对方式和
effect size；只有在定义并执行统计检验后使用 `significant`。方向、遮挡、新用户和 domain gap 均应展示
condition-level curve 与失败案例，不用单一平均值掩盖边界。

### 7.8 统一收缩全文 claim

Abstract、Introduction、figure captions、Results 和 Discussion 必须使用同一证据强度。`high-fidelity`、
`optical-level`、`generalizable`、`robust`、`practical deployment`、`inherent privacy` 等表述只能覆盖实际验证
条件；否则改为具体观察或局限性。同步补齐 ethics/consent、Data Availability、Code Availability 和 Source
Data，且 supplementary 的占位结果不得进入返修提交包。

科学工作应先冻结 dataset/split/radar/metric protocol 并恢复 paper-facing 结果，再执行新增比较。任何负结果
都应驱动论文主张调整，而不是被排除。逐条执行状态以 [response tracker](RESPONSE_TRACKER.md) 为准，实验
优先级和最小协议见 [review analysis](REVIEW_ANALYSIS.md)。
