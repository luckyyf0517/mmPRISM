# Full-Reproduction Data Upload Checklist

Status: `ready_for_source_intake`
Last Updated: `2026-08-12`
Role: `author_facing_upload_contract`

本文档回答“从零完整复现 mmPRISM 需要重新上传什么”。它是上传执行清单，不替代
`current/data_status.md` 的状态真值或 `evidence/data_registry.md` 的 provenance 登记。

## 1. 当前约束

- Canonical data root：`/mnt/gfs/yanyifan/mmPRISM/`。
- `/mnt/gfs` 为约 10 TB 共享盘；`2026-08-12T00:38Z` 复核时约余 3.1 TB，容量会动态变化。
- 作者已确认全部项目数据仍在其 NAS；当前机器尚待接收 source-side inventory 和分批上传内容。
- 上传前必须先提供文件清单和总大小；未通过容量 gate 前不得解压或生成派生数据。
- 第一优先级是不可替代的私人原始采集与标定资料，不是 pose、feature 或 synthetic radar 缓存。

## 2. 立即上传清单

### P0-A 私人真实雷达采集：必须上传

这是完整复现中最不可替代的数据。保留采集设备产生的原始包，不要先拆帧、转格式、重命名或覆盖。

- 历史 `collected_base`、`collected_demo`、`collected_csl` 的原始 sequence/session 目录。
- 每条 sequence 的原始雷达数据。旧代码曾处理 `mmwave.npy`，也可能存在设备 native binary、
  ADC dump、metadata 或日志；这些都应保留。
- 与雷达同步的原始 RGB/视频。旧代码曾处理 `color.npy`；若还有 MP4、帧时间戳或相机原始文件，
  一并保留。
- 每条 sequence 的中文文本、gloss/sign label、动作 ID、句子 ID 和失败/重采标记。
- sequence 到 subject、session、scene、采集批次和 radar configuration 的原始映射。
- 原始帧率、帧数、时间戳、同步偏移、丢帧记录和采集程序版本。
- 原始目录说明或采集日志，即使其中部分内容看似重复。

旧代码推断出的雷达数组候选形状是 sequence
`[T, chirp, antenna, sample, real_or_imag]` 或每帧
`[chirp, antenna, sample, real_or_imag]`，并出现过约
`[T, 128, 86, 256, 2]` 的历史约定。该形状只能用于 intake 检查，不能当作已确认事实。

### P0-B 参与者、场景和条件元数据：必须上传

Reviewer 要求新用户、方向和遮挡泛化，只有雷达数组不足以完成统计或无泄漏 split。

- `subject_id`/`signer_id` 的匿名映射；禁止上传姓名、手机号等直接身份信息。
- 在伦理和同意范围允许时，提供年龄段、性别等用于多样性统计的字段及字段说明。
- `session_id`、采集日期或批次、房间/场景、背景运动、距离和设备位置。
- 相对 radar 的方向角；至少区分 `0°`、`30°`、`60°`，未知值显式标记为 unknown。
- 遮挡类型：无、双手重叠、部分手部遮挡、物体遮挡；记录物体类型和遮挡程度（若有）。
- 手尺寸或 gesturing style 相关字段仅在已合法采集且允许科研使用时提供。
- train/validation/test 的原始分配理由；如果没有可靠记录，不要凭目录编号补写。
- 是否包含面部表情、头部/身体动作等 non-manual grammatical features，以及这些字段是否有标注。
- 伦理批件/知情同意允许的数据使用、公开和 reviewer sharing 边界。管理文档只记录范围和编号，
  不提交包含个人信息的原始表格。

建议至少提供以下三个表及数据字典：

```text
subjects.csv
sessions.csv
sequences.csv
metadata_dictionary.md
```

### P0-C 雷达配置、阵列与标定：必须上传

- 精确硬件型号、固件/采集软件版本和采集脚本。
- 原始 profile/chirp/frame 配置；包括起始频率、斜率、采样率、ADC samples、chirps、frame period、
  Tx/Rx TDM 顺序及启用通道。
- 12 Tx、16 Rx、192 virtual antenna 到 86 horizontal/4 vertical element 的映射和去重规则。
- 物理天线坐标、virtual array 坐标、channel order、极性/相位约定和单位。
- range/phase/耦合/通道标定文件、坏通道记录和标定日期。
- radar、相机和人体坐标系定义、外参、单位、轴方向及变换矩阵。
- radar-camera 同步方法、触发方式、时间偏移和校准采集。

当前存在必须解释的版本冲突：legacy 仿真配置是 64 chirps、256 ADC samples，而历史真实数组约定可能是
128 chirps、86 antennas、256 samples。每个 sequence 必须通过 `radar_config_id` 绑定真实配置，不能用一份
默认配置覆盖全部数据。

### P0-D 不可公开下载的外部源数据：必须上传或提供重新获取方式

#### CSL-Daily

- 原始 sentence image sequences：历史路径形态为 `sentence/images/<sequence>/*.jpg`。
- 原始 annotation：旧代码使用 `sentence_label/csl2020ct_v2.pkl`。
- signer/identity、sentence、gloss、sequence 和官方 split 元数据。
- 数据集版本、下载日期、来源 URL、license/terms 及当前访问凭据的管理方式。

#### CSL-News

- 原始视频压缩包或已验证的原始 MP4 文件。
- 原始 `CSL_News_Labels.json`，以及若仍存在的 historical converted mapping。
- archive、sequence、栏目/类别、caption、signer 和官方 split 元数据。
- 数据集版本、下载日期、来源 URL、license/terms。旧代码至少筛选过
  `Common-Concerns` 和 `Dragon-TV`，但新 manifest 必须以源数据实际字段为准。

作者已确认 CSL-News 原视频可以重新下载，因此默认不进入上传队列。执行下载前仍需固定官方来源、
数据版本、访问条款和 archive checksum；原始 `CSL_News_Labels.json` 必须属于同一版本。

当前已固定 `ZechengLi19/CSL-News@3a0601210333fe760efd09b5d9e2ae5f341ce339`，许可证为
`CC BY-NC 4.0`，436 archives 总计约 935 GB compressed，下载正在 versioned incoming batch 中进行。

如果这两个数据集可以从官方来源重新获取，不需要重复占用 GFS；先提供准确版本、URL、访问方式和
checksum。若链接已失效、需要审批或版本无法确定，则按 P0 上传原始包。

### P0-E MANO/仿真来源证据：必须澄清，条件性上传

当前稿件描述“由 MANO mesh 和 ray tracing 生成 synthetic mmWave”，但仓库中可见的 legacy
`run_simulation.py` 主要对 24-joint skeleton 插值后仿真。正式重跑前必须确认原投稿实际使用哪条路径：

- 若确实使用 MANO：上传原始 MANO 参数、拟合输出/mesh、mesh-to-radar 输入、仿真配置、运行脚本、
  关键 checkpoint 和可合法保存的 MANO model 资产；同时记录 MANO license/访问限制。
- 若使用另一份未提交的 mesh/ray-tracing simulator：上传其源码、依赖、配置、输入和小样例输出。
- 若原结果实际使用 skeleton simulator：上传对应输入、配置和历史运行证据，并将稿件方法描述列为待修正。

在该问题关闭前，不允许把新 skeleton 仿真结果登记为原投稿 MANO pipeline 的直接复现。

### P0-F 返修新增真实 stress set：后续必须采集

这部分不是原项目从零复现所需的历史上传，但它是关闭 reviewer real-world generalization 意见的必需数据。
如果已有符合条件的未上传采集，可按 P0-A 到 P0-C 直接 intake；否则在历史配置与伦理范围确认后再采集：

- 与训练 subject 完全隔离的新用户，记录匿名 subject/session 和允许披露的多样性字段。
- 至少 `0°`、`30°`、`60°` off-axis orientation，并记录角度测量方法。
- 无遮挡、双手重叠、部分手部遮挡和 object occlusion 条件。
- 至少覆盖可复现的距离、环境/场景和背景干扰条件。
- 条件间尽量使用 matched signs/sentences，并同时保留 reconstruction 与 translation target。
- split 在采集前冻结；test 数据不得用于 adaptation、checkpoint selection 或阈值选择。

该数据登记为独立 revision dataset，不能并入原投稿 test set 后重新报告为同一 protocol。

## 3. 历史审计资产：P1 优先上传

这些资产不是“从头训练”的前置条件，但用于解释原投稿数字、判断实现差异和准备 Source Data。

- 原投稿实际使用的 split JSON/TXT/PKL 和生成脚本输出。
- OmniHand/CubeNet、WaveLLM/mT5、domain adaptation 和 baseline checkpoints。
- 每个正式 run 的 resolved config、命令、Git commit、环境、seed 和训练日志。
- sample-level predictions、pose outputs、translation outputs 和 per-sample metrics。
- 汇总 metrics、消融表、复杂度/速度记录、失败样本和可视化输入。
- 主文与补充材料每张图/表的 source values、绘图脚本和原始统计文件。
- 原投稿对应的 historical poses、predicted poses、features 和 synthetic signals，可作为 forensic evidence
  保留，但不能和新重建产物混放。

已发现的历史 split/data 名称线索包括：

```text
csl-daily
csl-news-200
csl-news-demo01
collected-700
collected-demo
collected-cross-individual-*
```

名称不等同于 provenance；上传时仍需说明其来源 run 或原投稿表图。

### 首批 CSL-News pose 对照样本

NAS 上约 `0-99` archive 的历史 RTMPose3D 派生结果不需要整体优先上传。先提供两条原始 `.npy`，用于
与当前重建结果做逐帧等价性审计，目标目录为：

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260812_csl_news_legacy_pose_pair_v1/legacy_evidence/
```

不要只上传重命名后的两个数组。请保留它们的原始相对路径/文件名，并附带：

- archive ID 和匹配的 ZIP member/视频文件名；
- historical caption/mapping/split entry（如有）；
- 生成时使用的模型 config/checkpoint 或环境记录（如仍可找到）。

样本只作历史审计，不替代可重新下载的视频 source，也不会直接进入训练 manifest。

## 4. 不要优先上传的可再生资产

以下内容默认从已验证 source 重新生成。只有为了历史审计且有明确来源时才放入 P1：

- RTMPose3D 生成的 pose annotation。
- synthetic radar、range/Doppler/azimuth/elevation cube。
- predicted pose、learned feature 和临时可视化。
- 新 split、训练 checkpoint、metrics 和 cache。
- Hugging Face 或 OpenMMLab 可稳定重新下载的公共模型副本。

## 5. 公共模型：固定版本后下载

| Asset | Historical Reference | Intake Rule |
|---|---|---|
| Language backbone | `google/mt5-base` 或历史本地 `mt5-pretrained` | 确认原始 revision；公共 base 下载，历史 fine-tuned 权重按 P1 上传 |
| Pose estimator | `rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth` | 从官方源固定 URL/revision/checksum；若官方源不可用则上传 |
| SimCSE | `cyclone/simcse-chinese-roberta-wwm-ext` | 固定 Hugging Face revision 后下载 |
| SBERT | `shibing624/text2vec-base-chinese` | 固定 Hugging Face revision 后下载；必须补齐评估 smoke |

不得只记录 mutable model name；正式 registry 需要 revision/commit、下载日期、license 和 checksum。

## 6. 上传 staging 结构

上传内容先进入版本化 incoming batch，不直接写入 `raw/`：

```text
/mnt/gfs/yanyifan/mmPRISM/
  incoming/
    <YYYYMMDD_source_batch>/
      README.md
      UPLOAD_MANIFEST.csv
      SHA256SUMS
      private_real/
      external_sources/
      hardware_calibration/
      legacy_evidence/
```

通过 checksum、可读性、权限和 metadata gate 后，再由 intake 工具登记到 canonical `raw/`、`external/`
或 historical evidence 位置。原始上传包保持只读，不原地修复。

`UPLOAD_MANIFEST.csv` 至少包含：

```text
source_id,relative_path,category,dataset,size_bytes,sha256,
source_owner,access_class,original_format,notes
```

上传 token、密码和需要保密的下载凭据只能放在 `.env` 或独立 secret manager，不能放入 manifest、README
或 Git。

## 7. 每批上传前后的 Gate

### 上传前

- [ ] 先给出每个 archive/目录的估计大小和文件数。
- [ ] 标明 P0/P1/P2、是否可重新下载、是否含个人/敏感数据。
- [ ] 确认可用空间和解压峰值；共享容量即使暂时充足，也不允许边下载边批量解压。
- [ ] 生成相对路径 manifest 和 SHA-256；大目录可先生成 archive checksum。
- [ ] 确认数据 owner、license、伦理和 reviewer/public sharing 边界。

### 上传后

- [ ] checksum 与来源端一致。
- [ ] 文件/压缩包可读，未发生截断。
- [ ] 抽样验证 shape、dtype、NaN/Inf、帧数和 complex representation。
- [ ] sequence、subject、session、caption 和 radar config 可关联。
- [ ] 原始目录保持只读；所有修复进入 versioned `interim/`。
- [ ] `evidence/data_registry.md` 已登记 owner/location/size/access/validation 状态。
- [ ] inventory 报告记录 bytes/files、缺失模态、重复和 quarantine。

## 8. 建议上传顺序

1. 先发送“目录/压缩包名称 + 大小 + 是否可重新下载”的小清单，不传内容。
2. 上传匿名 metadata、雷达配置、阵列映射和标定文件；这些体积小且决定后续 reader contract。
3. 分批上传私人真实采集 raw package，优先 `collected_csl` 和原投稿 test split 对应批次。
4. 上传 MANO/mesh/simulator provenance，关闭仿真方法不一致风险。
5. 从固定官方版本重新下载 CSL-News；仅在不能官方重下时上传 CSL-Daily 或其他外部原始包。
6. 容量允许后上传原投稿 checkpoints、predictions、metrics 和 figure source data。
7. 不上传可从以上资产重新生成的 pose/signal/feature/cache，除非它们是唯一历史证据。

完成上述 P0 intake 后，才能冻结 canonical reader、split 和返修新增采集 protocol。
