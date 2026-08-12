# 当前投稿手稿技术完整性审计

Status: historical
Owner: Paper revision lane
Evidence scope: 对当前 Overleaf 投稿正文、当前 supplementary 和历史会议稿的只读技术审计快照。
Recorded: 2026-08-12

## 1. 审计边界

本审计绑定 manuscript submodule commit
`3242a40631ec5198e66fa8592763235c108513b2`，不修改论文正文，也不把当前工程重建结果自动视为原论文
实现或实验事实。

| 层级 | 角色 | 本轮使用方式 |
|---|---|---|
| `sn-article.tex` + `chapter/1_introduction.tex` 至 `chapter/4_discussion.tex` | 当前审稿提交正文 | 唯一用于判断“当前主文是否披露”的来源 |
| `supplementary/Supplementary_Information.zip` | 当前补充材料 | 判断是否补足主文，以及补充结果是否可验证 |
| `chapter_org/` | 之前的会议版本论文 | 仅作为历史技术细节和实验线索；不算当前投稿已披露，不自动构成当前证据 |

`sn-article.tex:226-235` 只加载 `chapter/` 下四个文件，没有加载 `chapter_org/`。当前文件身份：

| Artifact | SHA-256 |
|---|---|
| `sn-article.tex` | `75ca93da51c24b47839635b3ff50f87c10966d8d145bab0d016384f7e8d7ff7a` |
| `chapter/1_introduction.tex` | `ec47f5a4a67951c364e88c587dc0df0d2ca513464bec22d65e092c236b014493` |
| `chapter/2_results.tex` | `aeeef304cb8a65edb17ba9a18ce0124a656e4215df4e22e9d2b64a469e5917cd` |
| `chapter/3_methods.tex` | `e84f332c5ab62d58dbb70012f0f62c2369921f0e3f1c6cef0083b2037e769bf0` |
| `chapter/4_discussion.tex` | `4f8374faecec2b25279c1fe9b1df24a4d908b25d705388daf7a7b6f7b9d0f9fb` |
| Supplementary ZIP | `f74f4eb0ac8c9e870a964e0cb5c72075cface6aa375d06ad81ba44d74d44abcd` |

## 2. 总体结论

审稿人关于技术细节、数据透明度、比较公平性和真实泛化的主要判断成立。当前投稿正文能够说明概念主线，
但尚不能让独立研究者重建同一输入、模型、split、训练和指标协议。最严重的问题不是个别参数遗漏，而是
Results 对 CubeNet、temporal transformer、confidence fusion、三阶段训练和仿真真实性作出强机制归因，
有效 Methods 却没有提供这些模块的可执行定义。

本轮识别出：

- `9` 组阻断科学复现或结果解释的 major findings；
- `6` 组需要在返修中补齐的 moderate findings；
- `chapter_org/` 中有部分可恢复说明，但全部需要按当前数据、代码和实验重新验真；
- 当前 supplementary 的 Tables S2-S6 源码明确标记为“示例表格内容（替换为真实数据）”，不能用于
  证明主文结论，也不能从历史会议稿直接复制数值替换。

## 3. Major Findings

### `MANU-TECH-001` 有效 Methods 缺少两套核心模型的架构定义

**位置：** 当前 `chapter/3_methods.tex:24-45`；Results `chapter/2_results.tex:9-36,144-160`。

**发现：** Methods 声称会详述 pose reconstruction 和 translation 两个核心组件，但对应的 CubeNet、
temporal aggregation、pose encoder、confidence fusion 和 mT5 generation 段落全部处于 LaTeX 注释中。
有效 Methods 从 4D cube 直接跳到 synthetic generation。相反，Results 和图注声称使用 spatial/channel/SE、
PAFPN、RoPE temporal transformer、CLS/mean/attention aggregation、STGCN、joint-confidence gate 和三阶段
curriculum。

**影响：** 模型的输入输出、层次、维度、参数量、mask、confidence 来源和训练边界均无法从当前稿复现；
读者无法判断表中 baseline 是否容量匹配。这直接支持 `R2-1`、`R2-5` 和 `R1-2`。

**最小修复：** 在当前 Methods 中给出经过实现核对的完整模块定义和 tensor flow；层宽、stage 数、temporal
窗口、attention heads、position encoding、aggregation、regression head、STGCN graph、fusion equation、
mT5 接入方式与可训练范围至少进入主文或正式 supplementary。

### `MANU-TECH-002` 数据集、任务定义和 split 基本缺失

**位置：** 当前 `chapter/3_methods.tex:47-66`；supplement `mian.tex:105-112`。

**发现：** 标题为 “Datasets and synthetic data generation” 的小节只写了仿真，没有列出数据集名称、版本、
手语类型、语言、词汇、句子、序列、帧率、参与者/session/scene、标注流程、train/validation/test 数量或
split unit。Supplement 只给出 12 名 20-30 岁参与者、200,000+ frames 和若干场景/距离。

**影响：** 无法判断 participant/sequence leakage、continuous SLU 的边界、真实与合成数据占比或未见用户
泛化。主文承认 sign language 含 facial/body non-manual grammar (`chapter/1_introduction.tex:3-4`)，但系统以
手部/有限上肢几何为主，没有说明任务是否排除或覆盖这些语言信息。直接支持 `R1-4b/c/d`、`R2-4`。

**最小修复：** 从冻结 manifest 自动生成 dataset table；明确 sign language、caption language、sentence
segmentation、signer qualification、non-manual coverage、每阶段数据量和 subject/session/scene-level split。

### `MANU-TECH-003` 雷达采集与 4D cube 协议不足以复现

**位置：** 当前 `chapter/3_methods.tex:3-22`。

**发现：** 已给出 12 Tx/16 Rx、77 GHz、3.85 GHz bandwidth、FFT 和 steering-vector 概念，但缺少硬件型号、
ADC samples/rate、chirp slope/duration/count、frame rate、TDM 顺序、channel map、window/zero-padding、range
crop、angle grid、array coordinates/calibration、normalization、cube bin sizes 和最终 dtype。式 (3) 在完成
Doppler FFT 后跨 Doppler bins 减均值，被称为 static clutter removal，但处理顺序和 DC 定义没有解释。
`W_learn` 如何从扁平 `N_angles` 形成 azimuth/elevation、是否保持复数物理约束、如何校准也未说明。

**影响：** 不同实现会生成不同物理坐标和 cube。当前 forensic 状态也记录了稿件与 legacy 在 chirp、bandwidth、
clutter order、array 和 steering convention 上存在待关闭冲突，因此不能靠代码猜测论文协议。

**最小修复：** 发布逐序列 acquisition/config identity、array/channel/calibration、完整信号处理顺序和一个带
checksum 的 ADC-to-cube fixture；Methods 与冻结 radar contract 同步。

### `MANU-TECH-004` pose 与 translation tensor contract 未定义

**位置：** 当前 `chapter/3_methods.tex:22,69-76,82-114`。

**发现：** 4D cube 仅定义为 `[D,R,A,E]`；注释中的 pose 仅为 `[J,3]`。当前稿没有说明 batch/time 轴、双手
顺序、`J` 的值和 joint topology、坐标原点/轴向/单位、camera-radar 标定、有效 joint mask、sequence padding、
pose confidence 的生成方式，或 radar feature 与 pose 的逐帧对齐方式。

**影响：** 即使模型结构已知，也无法构造相同 target 或比较 MPJPE/PCK。当前 canonical engineering contract
明确使用 `[B,T,D,R,A,E] -> [B,2,24,3]`、米制和显式 coordinate frame；这说明这些字段是必要信息，但并不
证明原投稿已经采用该 contract。

**最小修复：** 在论文中给出完整 shape/axis/unit/coordinate/joint table，并绑定 camera-radar calibration、
valid mask、confidence 和 time-alignment protocol。

### `MANU-TECH-005` synthetic radar 方法和真实性结论之间存在显著信息差

**位置：** 当前 Methods `chapter/3_methods.tex:47-66`；Results `chapter/2_results.tex:124-140`。

**发现：** Methods 只给出 MANO mesh、理想化 IF sum 和 amplitude 项，未说明视频到 MANO 的工具/版本与
误差、mesh sampling、material/scattering parameters、ray count/order、self/inter-hand occlusion、multipath、
noise/hardware impairments、radar discretization、随机种子和 simulator validation。Results 却声称 high-fidelity
ray tracing “rigorously” captures micro-Doppler、complex multipath 和 real-world stochastic properties，并由
t-SNE 推断 domain gap 已被成功弥合。

**影响：** 方法描述不足以生成相同 synthetic data；t-SNE intermingling 不能单独证明信号真实性、transfer
有效性或 unseen-environment generalization。当前 forensic audit 仅能看到以 skeleton interpolation 为主的
legacy 路径，原 MANO/mesh/ray-tracing provenance 仍未定位；这是一项未决冲突，不是已经证明稿件错误。

**最小修复：** 恢复 simulator 资产和配置，报告 paired/category-matched synthetic-real fidelity、下游 transfer
与剩余 domain gap；在此之前降级 “high-fidelity/physically consistent/bridged” 表述。

### `MANU-TECH-006` 训练协议只剩两个 loss，无法复现

**位置：** 当前 `chapter/3_methods.tex:68-77`；Results `chapter/2_results.tex:132,146-150`。

**发现：** 当前稿未报告 optimizer、learning rate/schedule、batch/effective batch、epochs/steps、seeds、GPU、
precision、gradient clipping、checkpoint selection、augmentation、loss weighting、stage-specific freeze/unfreeze、
real-data budget 或 decoding config。Pose loss 被称为 L2/MSE，但式子只按 `J` 平均 squared norm，没有定义
双手、batch、time 和 invalid joints 的归一化。Translation loss 引用的 `E_fused` 在有效 Methods 中没有定义。

**影响：** 无法重跑主结果、比较 compute cost 或验证 shallow adaptation 的数据效率；直接支持 `R1-2` 和
`R2-2`。

**最小修复：** 每个 training stage 提供完整 resolved recipe、数据输入、可训练参数范围、停止/选择规则、
seed 和资源统计；公式与实现的 mask/normalization 保持一致。

### `MANU-TECH-007` 指标和自定义 `Rel. Perf.` 协议不完整

**位置：** 当前 `chapter/3_methods.tex:79-114`；`chapter/2_results.tex:163-179`。

**发现：** PCK 的阈值 `T` 没有给值，MPJPE/PCK 未说明 absolute 或 root-relative、对齐、单位、有效 joints、
双手聚合、frame/sequence/subject aggregation 和 uncertainty。BLEU 未说明 BLEU-4 implementation、tokenization、
smoothing 与 corpus/sentence aggregation；ROUGE 表中有时写 `ROUGE`、Methods 写 `ROUGE-L`；SBERT/SimCSE
缺少 exact model/revision、pooling 和分数尺度。

当前正文也没有定义 `Rel. Perf.`。从表中可逆推出它是四个异质指标各自除以 Vision Oracle 后的无权平均：
Ours 的四项相对值为 `73.7185%/81.2524%/86.3704%/95.8700%`，平均为 `84.3028%`。历史
`chapter_org/6_evaluation.tex:75` 只说明单个分数与 benchmark 的比率，没有明确跨四个指标再平均。

**影响：** 84.30% 看似单一 fidelity 指标，实际混合 lexical 与 embedding metrics，权重和统计意义未经论证。
它不应被写成 “optical-level translation fidelity” 而不披露公式和 sensitivity。

**最小修复：** 冻结每项 metric implementation 和 sample-level input；优先逐项报告相对差距。若保留 composite，
须给公式、选择依据、权重敏感性和 uncertainty，避免把它称为标准 fidelity。

### `MANU-TECH-008` baseline 虽出现在表中，但公平性没有建立

**位置：** 当前 `chapter/2_results.tex:152,163-179`。

**发现：** Table 2 已包含 `End-to-End (Cube)`，因此不能简单说当前稿完全没有 direct baseline。但正文没有
定义其 encoder/projector、参数量、LLM、预训练、三阶段训练、真实数据预算、优化器或 compute。Heatmap 和
Point-based 也没有引用或实现定义。表注称其为 state-of-the-art RF-based methods，但它们看起来更像作者构造的
输入表示 baseline，而非具名外部 SOTA。

历史 `chapter_org/6_evaluation.tex:63-77` 补充了三种 baseline 的概念，并声称使用 identical LLM 和 multi-stage
pipeline；但仍没有 Reviewer 2 要求的 aligned parameters，而且该历史文字不在当前投稿中。

**影响：** `R2-1` 仍然成立：表中有一行不等于已经完成 matched architecture comparison。当前数据也不能隔离
pose bottleneck、multi-stage training、fusion 和容量的各自贡献。

**最小修复：** 为每个 baseline 给出可运行定义、参数量、训练数据/预算、seed 和资源；增加真正 matched direct
baseline，并把内部 baseline 与外部 literature comparison 分开。

### `MANU-TECH-009` 当前 supplementary 含明示占位结果

**位置：** Supplement `mian.tex:160-256`。

**发现：** Tables S2-S6 每张表源码前都有“示例表格内容（替换为真实数据）”，但 captions 仍将其数字解释为
真实实验，并据此声称 temporal transformer、adaptive fusion、多阶段训练、cross-individual/generalization 和
noisy-scenario robustness 已得到验证。

历史会议稿确实包含类似实验，但数值并不一致。例如历史 temporal single-frame 为 `27.35` MPJPE，当前
supplement 占位表为 `42.56`；历史 No Fusion BLEU-4 为 `14.58`，当前占位为 `12.35`；历史 pre-train-only
BLEU-4 为 `0.52`，当前占位为 `8.76`。

**影响：** supplement 不能关闭 `R1-3/4/5`、`R2-2/3` 或模型消融问题；若提交包已包含这些表，属于最高优先级
的科学 provenance 风险。

**最小修复：** 在重新生成真实实验前删除表和结果性 caption，或明确标记为非提交草稿。历史数值只能在找到
dataset/split/run/checkpoint/metric 后作为待恢复 evidence，不能直接复制。

## 4. Moderate Findings

### `MANU-TECH-010` 机制性语言强于已展示证据

- `chapter/2_results.tex:28-35` 将 attention 描述为定位 hand-like geometry、抑制 multipath，并称 temporal
  unit 可推断 occluded joints；没有 attention leave-one-out 或 occlusion-stratified evidence。
- `chapter/2_results.tex:45-48` 从少量 qualitative examples 推导显式 geometry 是可靠翻译的 prerequisite。
- `chapter/2_results.tex:132-139` 用 t-SNE cluster mixing 推导 domain-invariant feature 和 unseen-environment
  generalization。

这些陈述应改为设计假设，直到 `R1-5`、`R2-2/3/5` 对应实验完成。

### `MANU-TECH-011` 真实部署、隐私和泛化表述超出测试边界

**位置：** Abstract `sn-article.tex:200-202`；Introduction `chapter/1_introduction.tex:39-43`；Discussion
`chapter/4_discussion.tex:1-3`。

当前稿从受控距离、low-light、reflector 和 multi-person 场景直接推到 inherent privacy、dynamic environments、
real-world deployment、generalizable paradigm、practical、accessible 和 trustworthy。Low light 是 radar 的
模态属性，不等于系统整体鲁棒性；privacy 也需要明确 threat model 和 radar representation 的信息边界。
在方向、遮挡、未见用户和更广人口统计未验证前应缩小结论。

### `MANU-TECH-012` 缺少不确定性、重复运行和统计检验

主表/图只给 point estimates，没有样本量、seed、标准差/置信区间、effect size 或 statistical test，却使用
“significant/superior/remarkable”。距离区间也只用 `<0.5 m`、`~1 m`、`>2 m`，没有每档样本数和实际分布。
定量比较应保留 sample/sequence/subject 层级结果，并说明配对与聚合方法。

### `MANU-TECH-013` sign-language task 的 linguistic scope 不清楚

当前稿使用 open-vocabulary、continuous、translation、understanding、gesture classification 和 semantic
equivalence 等不同任务词，但没有定义输入是否预分段、输出语言、词汇开放性的测试方式、未见句子/未见 signer、
annotation/reference 数量或 human evaluation。正文也没有交代缺少 facial/body non-manual signals 时结论适用于哪类
sign-language content。需要将 SLR、isolated gesture、continuous SLT/SLU 清晰分开。

### `MANU-TECH-014` mT5 与 confidence fusion 仍不可复现

当前稿没有 mT5 exact variant/revision、tokenizer、prompt、embedding injection、freeze/full fine-tune、maximum
length、beam/greedy decoding、generation parameters。Joint confidence 的监督/计算与 gate 的形状、归一化、
radar feature 来源层也未定义。历史 `chapter_org/4_method_slu.tex` 只有概念说明，不能填补这些实现参数。

### `MANU-TECH-015` 伦理、可用性和 Source Data 尚未进入有效稿

当前有效主稿没有 ethics/consent、Data Availability 或 Code Availability 正式章节；相关词只出现在模板注释。
对参与者数据、RGB ground truth、原始 radar、派生 pose 和 model weights 的公开/受限边界均未说明。20 个 display
item 的科学 provenance 仍未闭合，详见 [display-item registry](../../../authority/20_CONTRACTS/DISPLAY_ITEM_REGISTRY.md)。

## 5. `chapter_org/` 可恢复内容与使用限制

| 历史位置 | 可恢复线索 | 为什么不能直接迁入当前稿 |
|---|---|---|
| `chapter_org/3_method_rec.tex:45-84` | CubeNet、三种 attention、PAFPN、hybrid temporal、RoPE、aggregation | 需核对当前/原投稿实现；当前 canonical temporal encoder 使用 learned positions，不能用工程重建反向证明历史 RoPE |
| `chapter_org/4_method_slu.tex:10-43` | STGCN、confidence fusion、mT5、三阶段训练 | 仍缺 shape、exact model、training/decoding；需绑定真实实现和运行 |
| `chapter_org/5_implementation.tex:19-29` | 8 subjects、2000 segments、CubeNet/mT5 optimizer 等 | 这是会议版本；当前 supplement 写 12 participants、200,000+ frames，数据 cohort/单位明显不同 |
| `chapter_org/6_evaluation.tex:63-97` | 三种 translation baseline 和 Relative Performance 概念 | 没有参数匹配；需要当前数据/split/run 重新验证 |
| `chapter_org/6_evaluation.tex:99-259` | beamforming、temporal、fusion、pretraining ablations | 与当前 supplement 占位数字存在差异，不能选择性搬运 |
| `chapter_org/7_case_study.tex` | noisy scenes、8 gestures、20 sentences | 当前投稿未加载；协议和原始结果未完成 provenance，不算当前稿已有实验 |

因此，`chapter_org/` 的正确用途是 migration inventory：先判断内容是否仍属于当前系统，再绑定 evidence，最后
重写到 `chapter/` 或正式 supplementary。它不是可以直接复制的“遗漏章节”。

## 6. 对审稿意见准确性的复核

| Reviewer item | 结合当前稿的判断 |
|---|---|
| `R1-6` 4D cube notation | **部分成立。** 当前主文已命名 D/R/A/E 四轴并给部分公式，但缺 axis layout、bin/units、shape、normalization 和时序 contract；不能标记已关闭。 |
| `R2-1` direct baseline | **成立但需精确表述。** Table 2 已有 End-to-End (Cube)，缺的是 matched definition、参数和训练公平性，而不是完全没有一行 baseline。 |
| `R2-2` DA comparisons | **成立。** 只有 shallow alignment 描述和 t-SNE，没有 full/adversarial/MMD 或相同真实预算比较。 |
| `R1-3/4d/5`, `R2-3` 真实泛化 | **成立。** 当前可见结果未提供方向、真实遮挡、严格 held-out-user 的可验证证据；supplement 对应表是占位数据。 |
| `R1-4a` synthetic-real closeness | **成立。** t-SNE 不是信号 closeness 或 paired fidelity 测量。 |
| `R1-4b/c`, `R2-4` 数据透明度 | **完全成立。** dataset/split/linguistic coverage 是当前 Methods 最大缺口之一。 |
| `R2-5` attention ablation | **成立。** 当前稿机制描述很强，没有 spatial/channel/SE leave-one-out。 |
| `R1-2` compute cost | **成立。** 当前稿没有参数、FLOPs、GPU-hours、显存、latency 或 throughput。 |

## 7. 建议的修订顺序

1. 立即隔离或删除 supplementary S2-S6 占位数字，避免继续被引用。
2. 冻结当前论文真正使用的 dataset/split/radar/metric identities，恢复原投稿 run/checkpoint/prediction。
3. 以实现和 resolved config 为准重写 Methods，不从 `chapter_org/` 凭记忆搬运参数。
4. 明确定义 dataset、task、tensor、coordinate、training、decoding、metric 和 baseline protocols。
5. 完成 matched architecture/DA/attention、synthetic-real 和 real-world experiments 后再回写 Results。
6. 最后统一收缩 Abstract、Introduction、captions 和 Discussion 的 claim strength，并补 availability、ethics、
   Source Data 和 uncertainty。

本审计只确认当前稿的披露与证据缺口。任何历史数值是否真实、是否属于当前投稿数据版本，仍需通过
dataset -> split -> run -> checkpoint -> prediction -> metric 链条单独确认。
