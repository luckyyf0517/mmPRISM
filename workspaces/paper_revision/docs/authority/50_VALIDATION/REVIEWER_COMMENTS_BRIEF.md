# 审稿意见原文核对与客观整理

Status: current
Owner: Paper revision lane
Authority scope: 面向作者讨论的编辑与审稿意见原意、措辞强度及有限分析。
Last reviewed: 2026-08-12

## 1. 核对结论

上一版整理所引用的事项基本都能在邮件或审稿报告中找到来源，但呈现方式不够克制，主要有两个问题：

1. 编辑的科学意见原文只有一个段落，其中点名四个方向。此前为便于任务管理拆成了
   `ED-SCI-1` 至 `ED-SCI-5`，容易误读为编辑另外提出了五条独立意见。
2. 此前把正文技术审计、建议实验协议和内部执行方案紧跟在“意见”后面，部分措辞比审稿人原话更具体、
   更强，容易把作者推导误认为审稿要求。

本版以 [decision email](../../logs/2026/08/20260811_DECISION_EMAIL_REDACTED.md) 为唯一原始依据。
`R1-4a` 至 `R1-4d` 只是把 Reviewer 1 的一个长段落拆成可跟踪子项，不代表审稿人列了四条编号意见。
正文手稿审计和实现建议不再混入本文件。

## 2. 编辑原文到底提了什么

### 2.1 科学要求：一个总括段落

编辑的原话是：

> the reviewers request substantially stronger validation of the proposed framework, including comparisons
> against alternative architectures and domain adaptation strategies, clearer dataset characterization and
> evidence of real-world generalization.

客观上，这是一条“需要显著加强验证”的总要求，其中明确强调四个方面：

1. 替代架构比较；
2. 域适配策略比较；
3. 更清楚的数据集描述；
4. 真实世界泛化证据。

这些内容来自审稿报告，编辑是在决定信中强调其重要性，并没有另行指定实验数量、参与者人数、角度、
指标、置信区间或训练协议。现有跟踪 ID `ED-SCI-1` 至 `ED-SCI-5` 均是这一个段落的内部拆分，不能称为
五条编辑原始意见。

编辑还明确警告：如果不能完成 substantial revisions，稿件很可能不会再次送审。这说明上述四个方向是
返修主线，但不意味着此前内部规划的所有实验都由编辑逐项要求。

### 2.2 写作要求

编辑明确要求使用克制的写作风格，删除或避免 `new`、`novel`、`first`、`unique`、`unprecedented`、
`superior`、`remarkable`、`pave the way` 等首创、夸张或主观表达。这是独立且明确的修改要求。

### 2.3 回复和提交要求

编辑明确要求：

- point-by-point response 中逐字复现审稿意见；
- 使用 track changes 或颜色标出正文修改；
- 无法满足或认为不成立的请求，要在回复中说明原因；
- 提交 revised manuscript、supplement、response、cover letter 和适用的 checklist。

邮件还包含期刊通用的代码/软件 checklist、机器学习 checklist、色觉友好图表、Data Availability、Code
Availability、Source Data、ORCID、数据存储和作者变更等政策要求。这些是返修合规事项，不应混成新增
科学意见或新增实验。

## 3. Reviewer 1：六个意见段落

Reviewer 1 先给出正面评价，认可从毫米波分类走向语言语义解释的方向，并引用了论文报告的 `84.3%`
结果。这段没有提出修改要求，也不能替代对该数字的证据核查。

### `R1-1` 相关工作

**原意：** 建议引用并讨论 RadarLLM（AAAI 2026）和 mmExpert（MobiHoc 2025）。

**措辞强度：** `would suggest`，属于明确建议，不是新增实验。

### `R1-2` 训练和推理成本

**原意：** 4D volumetric encoder 与 LLM 的组合计算开销大，作者应展示 mmPRISM 的训练和推理成本。

**措辞强度：** `should demonstrate`，属于明确要求。审稿人没有指定必须报告哪些成本指标，也没有明确
要求与所有 baseline 做成本对比；具体指标属于作者的实施选择。

### `R1-3` 跨方向分析

**原意：** 毫米波对传播路径敏感，真实用户未必静止、正对或严格对齐雷达，论文忽略了
cross-orientation analysis。

**措辞强度：** 明确指出缺失，但没有给出具体角度、样本量或只评估重建还是同时评估翻译。`30°/60°`
来自 Reviewer 2，不是 Reviewer 1 的原话。

### `R1-4` 合成数据、数据说明、真实验证和新用户

这是 Reviewer 1 的一个复合段落，而不是四条编号意见。原文依次提出：

- `R1-4a`：询问如何衡量合成数据与 ground truth 的接近程度；
- `R1-4b`：要求进一步解释合成数据的 diversity、signs、size、environment domains，以及训练集和测试集；
- `R1-4c`：强调要确保 synthetic-trained model 在真实数据上也有效，并高度建议利用现有手语/书写数据、
  采集更多雷达数据；指出当前 12 名、20–30 岁参与者在规模和多样性上有限；
- `R1-4d`：期望展示对不同手型、手部大小和动作风格的新用户的泛化，且不需要太多重新训练。

**措辞强度：** 这里混合了问题、说明要求、`highly recommended` 和 `expected to show`。新增更多真实雷达
数据确实是审稿人的明确建议，但原文没有规定新增人数，也没有要求必须建立此前规划中的完整人口统计矩阵。
新用户泛化是明确期待，但 zero-shot、few-shot 曲线等具体协议是作者方案，不是原文要求。

### `R1-5` 双手重叠

**原意：** 当两手从雷达视角重叠时，稀疏点云可能合成单一簇；作者应说明方法如何区分 hand-to-hand
occlusions。

**措辞强度：** 原文是 `should clarify`。它首先要求解释，不是明确要求单独新增一组实验。是否补实验，
应由现有证据能否支持解释决定。

### `R1-6` 4D cube notation

**原意：** 虽然补充材料已经解释，仍应在主文中引入 4D cube 的 notation。

**措辞强度：** 明确的正文修改，不是实验。

## 4. Reviewer 2：六条论文意见

### `R2-1` 两阶段架构必要性

**原意：** 当前优势可能来自模型容量或特征增强，而非显式姿态重建；应增加将 4D voxel feature 直接输入
LLM 的 end-to-end baseline，并对齐参数和训练策略。

**措辞强度：** `should add`，是明确实验要求。当前表中即使已有 `End-to-End (Cube)`，若没有证明参数与
训练策略对齐，也不能视为已经回答原意见。

### `R2-2` 域适配比较

**原意：** shallow fine-tuning 缺少同等条件下的横向比较；建议在使用相同真实数据量时，与 full
fine-tuning、adversarial DA 和 MMD 等主流方法比较，以展示效率优势。

**措辞强度：** `recommended`，但该方向又被编辑明确点名，因此应视为返修重点。原文规定了相同真实数据
量，并举出三类方法；没有规定 seed、训练时长、显存等完整协议。

### `R2-3` 真实边界场景

**原意：** 现有噪声评估依赖可控实验室环境；应在偏轴 `30°/60°`、部分手部遮挡或物体遮挡等极端边界
场景中评估 reconstruction 和 translation。

**措辞强度：** `should evaluate`，是最具体的真实场景实验要求。原文使用 `such as`，列出的场景是明确
示例，但没有给出参与者人数、每条件样本量或更多角度网格。

### `R2-4` 数据集透明度

**原意：** 只报告 12 subjects 和 200k frames 不足以复现；Section 3.3 应补充手语类型、词汇量、句子数、
平均句长和非手部语法特征。

**措辞强度：** `should be expanded`，是明确的文档和统计补充。subject/session/scene、split hash 等是合理
的复现信息，但不在该条原文列举的字段中，不能写成 Reviewer 2 的逐字要求。

### `R2-5` Attention 消融

**原意：** spatial、channel、SE attention 同时加入而没有单组件消融，容易被认为是任意堆叠；需要对每个
attention module 做 leave-one-out ablation，验证各自贡献和必要性。

**措辞强度：** `is needed`，是明确实验要求。原文只明确 leave-one-out；`base-none`、所有单组件组合和
downstream translation 指标均属于可能的加强方案，不是原文硬性指定。

### `R2-6` 跨模态 baseline

**原意：** 现有 baseline 限于毫米波端到端模型；应加入 1–2 个来自其他非接触模态的 continuous SLU
baseline，例如 WiFi 或声学方法。

**措辞强度：** `should include`，是明确要求，但原文没有说明如何解决跨模态数据、任务和 split 不一致。
可比协议是否存在需要先核实；不能自行把“加入 baseline”改写成必须招募参与者并重新采集对应模态。

## 5. Reviewer 2：五条代码意见

### `R2-CODE-1` 本地路径硬编码

审稿人列出包含本地路径的脚本，明确要求将路径改成 CLI 或配置参数，否则第三方无法执行。

### `R2-CODE-2` SBERT 下载链断裂

审稿人指出下载脚本只准备 SimCSE，而评估代码强制加载 SBERT，因此按说明运行必然失败，并建议修复。

### `R2-CODE-3` README/CLAUDE 与仓库不一致

审稿人指出文档引用不存在的配置和 temporal/TVAN 模块，并建议从学术代码档案中移除内部 `CLAUDE.md`。

### `R2-CODE-4` 缺少 LICENSE

审稿人指出 README 声称 MIT，但仓库不存在 LICENSE，属于需要解决的发布合规问题。

### `R2-CODE-5` Phi-3 不可运行

审稿人指出 Phi-3 没有可运行配置、脚本或完整训练/评估示例，并鼓励作者补充。删除未经验证的 Phi-3
支持声明可能是作者选择，但不是审稿人原文给出的方案。

## 6. 按原文强度归类

### 明确要求或被编辑重点强调

- 替代架构比较：`R2-1`；
- 域适配比较：`R2-2`，且被编辑点名；
- `30°/60°` 偏轴、部分手部/物体遮挡下的 reconstruction 与 translation：`R2-3`；
- attention leave-one-out：`R2-5`；
- 1–2 个 WiFi/声学 continuous-SLU baseline：`R2-6`；
- 训练和推理成本：`R1-2`；
- 数据集字段与 4D cube notation：`R1-4b`、`R2-4`、`R1-6`；
- 代码可执行性与发布问题：`R2-CODE-1` 至 `R2-CODE-5`。

### 明确关切或期待，但协议未指定

- cross-orientation analysis：`R1-3`；
- synthetic-ground-truth closeness 如何衡量，以及真实数据验证：`R1-4a/c`；
- 采集更多、更有说服力的真实雷达数据：`R1-4c`；
- 不同手型、大小和动作风格的新用户泛化：`R1-4d`。

### 首先属于解释、引用或正文修改

- RadarLLM 和 mmExpert：`R1-1`；
- 双手重叠如何区分：`R1-5`，原文只明确要求 `clarify`；
- 4D cube notation：`R1-6`；
- 克制写作、逐条回复和期刊合规事项。

上述归类只反映原文。具体需要多少新数据、如何招募、采用什么统计或训练协议，应另行评估，不能反向
写成编辑或审稿人的原始要求。逐项执行状态仍由 [response tracker](RESPONSE_TRACKER.md) 维护；正文技术
缺失见 dated manuscript audit，不作为本文件中的新增审稿意见。
