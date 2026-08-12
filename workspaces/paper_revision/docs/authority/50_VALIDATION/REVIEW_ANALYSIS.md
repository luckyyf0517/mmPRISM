# Major Revision Internal Triage

Status: current
Owner: Paper revision lane
Authority scope: Author-derived prioritization and possible response actions; not a restatement of reviewer text.
Last reviewed: 2026-08-12

本文件是作者侧的内部诊断和候选执行方案，不是审稿意见原文。原意、措辞强度及哪些事项并未被要求，
以 [reviewer comments brief](REVIEWER_COMMENTS_BRIEF.md) 为准。下文的优先级、实验合并方式、指标和协议均
属于作者规划；当前不撰写结果型 response，也不填入未经验证的新结果。

## A. 返修可行性判断

Classification: `PROMISING, EVIDENCE-INTENSIVE MAJOR REVISION`

### 正面信号

- 编辑明确邀请 major revision，而不是拒稿。
- Reviewer 1 高度认可研究问题、语言语义解释方向和隐私友好应用价值。
- Reviewer 2 能准确复述两阶段系统，说明方法主线总体可理解。
- 没有 reviewer 指出数据泄漏、数学错误或核心方法不成立等致命正确性问题。

### 决定性风险

- 编辑明确表示，没有 substantial revisions 很可能不会再次送审。
- 两阶段架构、域适配策略和 attention 堆叠缺少控制变量实验，当前增益归因不充分。
- 合成数据主导的结果缺少直接 synthetic-real fidelity 证据。
- 真实数据只有 12 名 20–30 岁参与者，方向、遮挡、环境和新用户泛化证据不足。
- 代码审查已发现第三方必然遇到的执行失败与发布合规问题。

### 总体判断

值得投入完整返修。最有价值的路径不是新增大量零散实验，而是设计少数能够同时回答多条意见的实验矩阵，并先修复数据/代码 provenance。若无法扩充真实用户或完成方向/遮挡测试，应尽早联系编辑说明时间需求，而不是在提交前用弱实验替代。

## B. Underlying Concern Matrix

| ID | Surface Comment | Underlying Decision Question | Class | Severity | Sharedness | Resolution Confidence |
|---|---|---|---|---|---|---|
| `R2-1` | 需要 direct end-to-end baseline | 增益究竟来自 explicit pose bottleneck，还是更大模型/更强 feature？ | FAIR_COMPARISON | MAJOR | EDITOR+SINGLE | HIGH |
| `R2-2` | 比较 full FT/adversarial/MMD | shallow adaptation 是否在相同真实数据预算下有效且高效？ | FAIR_COMPARISON | MAJOR | EDITOR+SINGLE | HIGH |
| `R1-3`,`R2-3` | 方向和极端场景 | 方法是否只在正对、无遮挡实验室条件下成立？ | GENERALIZATION/ROBUSTNESS | MAJOR | MULTIPLE+EDITOR | MEDIUM |
| `R1-4a` | synthetic closeness | 合成信号是否真的接近真实信号，而不仅是生成可用标签？ | EMPIRICAL_SUPPORT | MAJOR | SINGLE+EDITOR-THEME | MEDIUM |
| `R1-4b`,`R2-4` | dataset 细节不足 | 数据规模、语义覆盖、split 和非手部因素是否足以支撑结论并可复现？ | REPRODUCIBILITY | MAJOR | MULTIPLE+EDITOR | HIGH |
| `R1-4c`,`R1-4d` | 12 人过少、新用户泛化 | real calibration 是否对人群过拟合，能否迁移到不同手型/风格？ | GENERALIZATION | MAJOR | SINGLE+EDITOR-THEME | MEDIUM |
| `R1-5`,`R2-3` | 双手重叠/遮挡 | geometry reconstruction 在 radar ambiguity 下如何工作，是否需要额外证据？ | ROBUSTNESS | MODERATE/MAJOR | MULTIPLE | MEDIUM |
| `R2-5` | 三种 attention 同时堆叠 | CubeNet 设计是否有机制依据，还是任意模块堆叠？ | EMPIRICAL_SUPPORT | MAJOR | SINGLE | HIGH |
| `R1-2` | 4D encoder + LLM 昂贵 | 论文性能提升的资源代价是否可接受、可部署、可复现？ | EFFICIENCY | MODERATE | SINGLE | HIGH |
| `R2-6` | WiFi/声学 baseline | mmPRISM 在 broader non-contact SLU 中的位置是否清楚？ | MISSING_BASELINE | MODERATE | SINGLE | LOW-MEDIUM |
| `R1-1` | 补两篇文献 | 相关工作是否覆盖最接近的 radar-language 研究？ | RELATED_WORK | MINOR | SINGLE | HIGH |
| `R1-6` | 主文解释 4D cube notation | 核心输入和物理维度是否可独立理解？ | CLARITY | MINOR | SINGLE | HIGH |
| `R2-CODE-1`–`5` | release 无法直接执行 | 第三方是否能从 README 到数据、模型和评测完整复现？ | REPRODUCIBILITY | MAJOR | SINGLE+EDITOR-POLICY | HIGH |
| `ED-WRITE-1`,`2` | 删除夸张/首创语言 | 论文是否符合 Nature Communications 克制客观风格？ | SCOPE_OR_OVERCLAIM | MODERATE-MANDATORY | EDITOR | HIGH |
| `ED-COMP-*` | checklist/availability/source data | 返修是否符合不可豁免的期刊政策？ | REVIEW_PROCESS | MANDATORY | EDITOR | HIGH |

## C. Intent Diagnosis Cards

### `R2-6` Cross-modal benchmarking

- Most likely concern：现有 baseline 范围过窄，缺少 broader sensing context。
- Alternative interpretation：要求在同一 continuous SLU protocol 上真正重实现 WiFi/声学模型。
- Confidence：`medium-low`。
- Why uncertain：不同模态通常没有共享输入和同一采集数据，直接数值对比可能不公平。
- Evidence answering both：先做 baseline protocol audit；若存在可用的公开连续 SLU 模型和可对齐数据，进行同 split/同语言目标比较；否则提供明确不可比性说明、扩大 related-work benchmark，并避免把异构公开数值放入同一排名表。
- Author confirmation needed：是否拥有同步 WiFi/声学数据，或是否能在返修期采集/获得？

### `R1-4a` Synthetic-to-real closeness

- Most likely concern：合成信号的分布真实性不足以支撑 synthetic-trained model 的真实泛化。
- Alternative interpretation：审稿人可能要求 paired sign-level synthetic/real ground truth，而不是全局统计距离。
- Confidence：`medium`。
- Evidence answering both：对同一 sign/sentence 建立 paired 或 category-matched synthetic/real set，同时报告信号/特征分布诊断和 downstream real transfer。
- Author confirmation needed：是否存在相同动作或句子的 synthetic-real 对应关系？

### `R1-5` Hand-to-hand occlusion

- Most likely concern：模型在两个手部散射簇合并时是否仍能恢复双手 geometry。
- Alternative interpretation：仅需要解释 temporal/geometry prior，而不是要求独立 benchmark。
- Confidence：`medium-high`。
- Safe strategy：先核对正文是否能准确解释机制。Reviewer 1 原文只要求 `clarify`，没有明确要求独立实验；
  若现有证据不足，可复用 `R2-3` 的遮挡实验或补失败案例，而不是自动新建一套实验。

## D. Author-Proposed Experiment and Analysis Plan

下表是为了节省采集和训练成本而提出的内部合并方案。除单元格直接复述的条件外，不代表编辑或审稿人
规定了这些具体协议。

| Priority | Concern IDs | Underlying Question | Proposed Experiment / Analysis | Minimum Viable Protocol | Result Interpretation | Fallback |
|---|---|---|---|---|---|---|
| `P0-1` | all scientific items | 当前数据、split 和指标是否可信？ | Data/provenance reconstruction + original-result baseline | 锁定 paper split、manifest、metric protocol；已有 prediction 先重算指标 | 所有后续实验的前提 | 若资产缺失，明确 unavailable 并重建可审计 split |
| `P0-2` | `R2-1`,`ED-SCI-2` | pose bottleneck 是否必要？ | Matched direct 4D-cube-to-LLM baseline | 相同 CubeNet 输入、数据、优化预算和 seed；移除 pose supervision，通过 matched projector 接 mT5；对齐参数/训练策略 | 若两阶段更优，支持 geometry bottleneck；若相近，缩小 necessity claim | 报告负结果并将方法定位为 interpretability/modularity trade-off |
| `P0-3` | `R2-2`,`ED-SCI-3` | shallow adaptation 是否有效率优势？ | DA matrix: shallow FT / full FT / adversarial DA / MMD | 相同 synthetic checkpoint、真实样本预算、训练 steps、seed；同时报告 pose/translation、trainable params、time/memory | 比较 accuracy-efficiency Pareto，而非只报最高分 | 若当前方法不最优，改为 lightweight option 并诚实呈现 |
| `P0-4` | `R1-3`,`R1-4c/d`,`R2-3`; `R1-5` 可复用 | 是否真实泛化到方向、遮挡和新用户？ | 在可行时合并 cross-orientation、new-user 和边界场景采集 | Reviewer 2 明确举出 30°/60°、部分手部或物体遮挡，并要求同时评估 reconstruction 与 translation；Reviewer 1 另要求 cross-orientation 和新用户证据；人数和采样协议待定 | 按原文条件分别报告，避免增加没有来源的场景 | 若无法扩充人群，说明限制并缩小 population claim；必要时联系编辑 |
| `P0-5` | `R1-4a` | 如何衡量 synthetic-ground-truth closeness？ | 选择并论证可直接回答该问题的度量/分析 | Reviewer 未规定 paired set、距离函数或统计协议；应先依据现有真实/合成数据关系确定 | 清楚说明所选度量回答什么、不回答什么 | 若数据无法支持直接度量，明确限制，不用 t-SNE 单独替代真实性证据 |
| `P0-6` | `R2-5` | attention 组件是否必要？ | Leave-one-out CubeNet ablation | full、w/o spatial、w/o channel、w/o SE、必要时 base-none；同 data/seed/budget；报告 MPJPE/PCK 和 downstream translation | 验证单组件贡献与交互 | 无贡献组件应删除或弱化设计主张 |
| `P0-7` | `R1-2` | 训练和推理成本是多少？ | Compute profile | Reviewer 未指定成本指标；作者需选取能复现训练和推理代价的指标，并写明硬件和测量条件 | 直接报告成本，比较范围按可用 baseline 确定 | 对最终模型做可复现测量，不虚构历史成本 |
| `P0-8` | `R1-4b`,`R2-4` | 数据描述是否可复现？ | Dataset characterization | 明确补充 synthetic diversity/signs/size/environment、train/test composition，以及 sign type、vocab、sentence count、average length、non-manual features；其他字段仅在定义 split 或复现确有必要时增加 | 形成主文或 supplement 统计 | 无法披露的信息说明限制和访问方式 |
| `P0-9` | `R2-CODE-*`,`ED-COMP-*` | Reviewer 能否执行代码？ | Clean-environment release reproduction | new environment；configurable paths；complete model download；example data/output；train/eval smoke；release manifest；license decision | reviewer-ready zip/repo | 不支持的 Phi-3 功能应移除 claim，而非保留坏入口 |
| `P1-1` | `R2-6` | broader modality positioning 是否公平？ | Cross-modal feasibility audit, then 1–2 baselines if alignable | 明确 modality/data/task differences；优先同 vocabulary/split 或公开 continuous SLU protocol | 只有协议可比时才进行数值排名 | 用定性定位和 limitation 回答，不制造伪公平表格 |
| `P1-2` | `R1-4c` | 12 subjects 是否足够？ | Additional participant collection integrated with stress matrix | 数量、年龄/手型/风格范围由伦理许可和统计设计决定；预注册 held-out usage | 重点看 unseen-user confidence interval | 若返修期不足，联系编辑并降低 claim |
| `P2-1` | `R1-1`,`R1-6` | 文献和符号是否完整？ | Related-work and notation revision | 核实文献、增加 4D cube definition/table | clarity closure | 无需实验 |

## E. Clarification and Compliance Actions

### 可直接通过文字/资产解决

- `R1-1`：核实并讨论 RadarLLM 与 mmExpert，不只添加引用。
- `R1-6`：主文定义 4D cube 的四个维度、单位、符号和 tensor layout。
- `R1-5`：原文首先要求解释双手重叠时如何区分；现有证据不足时再决定是否补实验。
- `ED-WRITE-*`：全稿扫描 novelty/primacy/exaggeration 表达。
- `ED-COMP-*`：checklist、Data/Code Availability、Source Data、色觉友好图表、ORCID、cover letter。

### 不能只靠解释解决

- `R2-1` 两阶段必要性。
- `R2-2` 域适配最优性/效率。
- `R1-3`,`R2-3` 方向与遮挡。
- `R1-4a` synthetic-real fidelity。
- `R1-4d` unseen-user generalization。
- `R2-5` attention 组件必要性。

## F. Three-Month Time Budget

以 `2026-11-11` 为内部目标，建议：

1. Week 1–2：导入 manuscript、定位数据/历史结果、环境和 legacy baseline、冻结 protocol。
2. Week 2–4：完成最小必要重构、dataset characterization、实验脚本和新采集 protocol。
3. Week 3–7：同步执行真实 stress/new-user 数据采集与 P0 架构/DA/attention 实验。
4. Week 6–9：synthetic-real fidelity、compute profiling、多 seed 补跑和结果诊断。
5. Week 9–11：正文、supplement、response letter、Source Data 和 checklist 回写。
6. Week 12：numeric provenance、code release clean-room test、PDF/colour/reference audit。
7. Week 13：缓冲、导师/合作者审阅和最终提交；若 P0 真实数据进度不足，在此之前联系编辑延期。

## G. Author Inputs Needed

- 当前 manuscript/abstract/supplement 和原投稿表图。
- 原投稿 paper-facing checkpoint、prediction、split 和 metric artifact。
- 现有 12 人真实数据的采集协议、伦理/同意范围和 metadata。
- 可新增采集的人数、时间、设备和可覆盖方向/遮挡条件。
- 可用 GPU 类型、数量和并行预算。
- 是否有同步 WiFi/声学数据或可运行 cross-modal baseline。
- Phi-3 是否出现在论文贡献/实验中；若没有，建议从正式支持范围移除。
- 最终开源许可证由全体作者确认，不能仅根据旧 README 自动添加 MIT。

## H. Result Reporting Template

```text
Experiment ID:
Concern IDs addressed:
Status: completed / failed / partial
Protocol:
Dataset manifest and split hash:
Baselines and controls:
Metric protocol:
Number of runs or seeds:
Result:
Uncertainty:
Unexpected findings:
Artifacts:
Claim supported:
Claim not supported:
Preferred manuscript change:
```
