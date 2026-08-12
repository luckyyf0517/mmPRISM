# Data Status and Rebuild Plan

Status: historical
Owner: Data rebuild lane
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12

## 1. 当前盘点结果

`/mnt/gfs/yanyifan` 当前只发现：

```text
Falcon/
Falcon_checkpoint_archive/
huggingface/
```

盘点开始时未发现目录名包含 `mmPRISM`、`CSL-Daily`、`CSL-News`、`collected_base`、
`collected_demo`、`OmniHand` 或 `WaveLLM` 的资产。现已建立 canonical incoming root，并开始下载
CSL-News 官方源。`2026-08-12` 作者确认全部项目数据仍保存在其 NAS；这确认了来源仍存在，但当前机器
尚无 NAS 路径、只读 inventory、容量清单或 checksum，因此状态从“来源未知”改为“等待分批传入和验收”。

`2026-08-12T00:38Z` 复核时，共享 `/mnt/gfs` 约 10 TB，使用率 70%，剩余约 3.1 TB。
空间属于共享动态状态，每次批量下载或解压前仍必须重新检查。

面向上传人员的完整 P0/P1/P2 清单见 `../data_upload_checklist.md`。

## 2. 代码推断出的历史数据族

| Data Family | 历史用途 | 预期主要模态 | 当前状态 |
|---|---|---|---|
| CSL-Daily | OmniHand simulation、WaveLLM caption | images, pose, pred_pose, feature, annotation | author_nas_confirmed_transfer_pending |
| CSL-News | pose annotation、simulation、WaveLLM caption | video, pose, signal/feature, caption | official_download_active_v2_registry_78_archives_129539_videos_passed；legacy_pose_range_0_99_reported_on_author_nas |
| Collected Base | 真实毫米波 OmniHand | color, raw mmWave, pose | author_nas_confirmed_transfer_pending |
| Collected Demo | 真实毫米波开发/演示 | color, raw mmWave, pose, pred_pose | author_nas_confirmed_transfer_pending |
| Collected CSL | 真实手语采集 | color, raw mmWave, pose, caption | author_nas_confirmed_transfer_pending |
| Model Weights | RTMPose3D、CubeNet、MT5、semantic evaluators | checkpoints/tokenizers | RTMW3D + mT5 base + SimCSE + SBERT verified；historical training checkpoints missing |
| Historical Runs | paper metrics and checkpoints | config, ckpt, predictions, metrics | missing_location |

## 3. 完整复现所需来源分类

| Category | Required Source | Intake Strategy | Priority |
|---|---|---|---|
| 私人真实采集 | 原始 radar/ADC、同步 RGB/视频、caption/label | 必须重新上传；保留 native package | P0 |
| 采集元数据 | subject/session/scene/orientation/occlusion/split/ethics scope | 必须重新上传；匿名化、带字段字典 | P0 |
| 雷达与标定 | 硬件/固件、chirp/frame、阵列映射、channel order、外参和标定 | 必须重新上传；每条 sequence 绑定 config ID | P0 |
| CSL-Daily | 原始图像、`csl2020ct_v2.pkl`、signer/sequence/version/license | 官方版本可重下则固定版本下载，否则上传 | P0 |
| CSL-News | 原始视频、`CSL_News_Labels.json`、archive/category/version/license | 官方 HF revision 已固定；935 GB compressed download active | P0 |
| MANO/仿真来源 | MANO 参数/mesh/model 或原始 simulator 输入、配置和运行证据 | 依据原投稿真实 pipeline 条件性上传 | P0 |
| 返修真实 stress set | 新用户、0°/30°/60°、双手重叠和物体遮挡 | 原始数据 intake 后冻结 protocol 并新采；不得混入原 test protocol | P0 |
| 历史论文证据 | split、checkpoint、prediction、metric、log、figure source | 为原投稿 provenance 优先上传 | P1 |
| 公共模型 | mT5、RTMPose3D、SimCSE、SBERT | 固定 revision/checksum 后下载 | P1 |
| 派生数据 | pose、synthetic signal、radar cube、feature、新 checkpoint | 从 source 重新生成；默认不上传 | P2 |

### 当前关键 provenance 冲突

1. legacy radar config 使用 64 chirps/256 ADC samples，但历史真实采集约定可能是
   128 chirps/86 antennas/256 samples；不能假设所有数据使用同一配置。
2. 稿件描述 MANO mesh + ray tracing，而当前可见 legacy `run_simulation.py` 主要使用 24-joint
   skeleton 插值输入。原投稿实际仿真源码、输入和 checkpoint 尚未定位。
3. `collected_base`、`collected_demo` 和 `collected_csl` 的目录编号不是可靠 subject/split 元数据，
   必须恢复原始采集映射。

## 4. Canonical 数据根目录与 staging

作者已确认数据统一放在 `/mnt/gfs/yanyifan`。canonical root 接受为：

```text
/mnt/gfs/yanyifan/mmPRISM/
  incoming/            versioned upload batches; validated before promotion
  raw/                 immutable source assets
  external/            third-party annotations/models metadata
  interim/             recoverable processing outputs
  processed/           model-ready versioned datasets
  manifests/           sample manifests and validation reports
  splits/              versioned split manifests
  checkpoints/         verified checkpoints
  experiments/         run artifacts
  cache/               disposable cache with quotas
  quarantine/          corrupted or ambiguous assets
```

`processed/` is reserved for immutable, task-specific Parquet delivery products only. A final product has one
logical training sample per row, at most 1,024 rows per Parquet part and at most 64 parts per split-homogeneous
chunk. `interim/` remains the recoverable per-sample sidecar/NPZ layer; neither a live sidecar directory nor a
partial manifest can be used as formal training input. The payload is defined by the current target adapter, so
CSL-News visual pose+caption is currently an intermediate product rather than OmniHand/WaveLLM training data.
The full contract and gates are in `../../../docs/architecture/data_delivery_parquet.md`.

上传批次使用 `incoming/<YYYYMMDD_source_batch>/`，附带 `UPLOAD_MANIFEST.csv` 和 `SHA256SUMS`。
容量审计完成前不批量创建目录、解压或 materialize 数据。

### 4.1 CSL-News 历史 pose 对照样本

作者确认 NAS 上保留了约 `0-99` archive 的历史处理后 pose，并先提供两条样本用于验证新旧预处理是否
一致。已建立只读 intake 目标：

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260812_csl_news_legacy_pose_pair_v1/legacy_evidence/
```

上传时保留原目录层级和文件名，不 flatten、不改名、不覆盖。每条 `.npy` 需要同时给出原始 archive ID、
ZIP member/视频相对路径；如有 historical label、mapping 或 split entry 也一并保留。历史脚本预期输出
`float32 [T,2,24,3]`，当前 canonical NPZ 同时保留 native 133-joint、score、2D transform、frame/time 和
`canonical_pose [T,2,24,3]`。对照必须先以精确 archive/member identity 定位当前结果，再报告：

- shape、dtype、帧数和 caption identity；
- bitwise equality、`allclose`、最大/平均绝对误差；
- 每轴、每关节误差、depth center 和左右手顺序；
- 不一致时区分视频解码、crop、模型/依赖版本、深度中心或 joint mapping 差异。

该批次只作为 historical forensic evidence；验证前不得进入 canonical training manifest。

### 4.2 已验证模型资产

`evaluation_models_v1` 已通过 canonical pinned downloader 写入
`external/models/evaluation_models_v1/`。SimCSE/SBERT 共 14 个 loader 文件、818,741,363 bytes，
collection manifest SHA-256 为
`5cb656d038459ec60c1ce8f2fe958358c809e0d1628ba86b605427fd61b81b22`。clean commit `3ae69c3`
上的 CPU smoke 使用两条中文文本，两个 loader 均输出 finite `[2,768]` float32 embedding；报告见
`../evidence/artifacts/evaluation_models_smoke_v1.json`，SHA-256 为
`e957ac79f620f0a982019befa4938c393357764f5d912b4b6a7c27996f789b39`。模型版权边界仍按各上游
条款处理；SimCSE HF card 未声明 license，因此当前只提供固定 revision 下载器，不把权重纳入 release。

`mt5_base_v1` 已通过同一 canonical asset service 写入 `external/models/mt5_base_v1/`。固定来源为
`google/mt5-base@2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f`，6 个 loader 文件共
2,334,046,221 bytes；主权重 SHA-256 为
`180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614`，collection manifest
SHA-256 为 `2350101b38c5ee9c860ae5d8c28918e360eb57b47d39fc1b24a3d36773418bc6`。clean commit
`79b45b5` 的 A100 smoke 已验证两步 adapter update 和 beam generation。该 base asset 不替代仍缺失的
原投稿/历史 fine-tuned checkpoint，也不构成论文指标证据；详情见 `../evidence/mt5_vertical_smoke.md`。

## 5. Canonical Sample Record

每个 sequence/sample 至少需要以下字段：

```yaml
schema_version: mmprism.sample.v1
sample_id: stable-id
sequence_id: stable-sequence-id
subject_id: optional-subject-id
dataset: csl_daily | csl_news | collected
modalities:
  video: optional-uri
  pose_gt: optional-uri
  pose_pred: optional-uri
  radar_raw: optional-uri
  radar_cube: optional-uri
  feature: optional-uri
  caption: optional-text-or-uri
shapes: {}
dtypes: {}
acquisition: {}
provenance:
  source_id: source-record
  processor_version: optional
  git_commit: optional
  config_hash: optional
checksums: {}
group_keys:
  signer: optional
  subject: optional
  scene: optional
```

URI 应为相对于配置 root 的相对路径或显式 storage URI，不使用历史机器绝对路径。

`acquisition` 至少需要 `session_id`、`radar_config_id`、scene、distance、orientation、occlusion、
timestamp/synchronization 和 coordinate-system references；无法恢复的字段必须显式为 unknown。

## 6. 数据重建顺序

1. `preflight`：来源端先提供 archive/目录大小、文件数、类别、访问级别和可重新下载性。
2. `upload incoming`：先 metadata/calibration，再分批上传不可替代 raw；不立即解压。
3. `inventory`：只读扫描目录、大小、文件数、后缀、mtime 和抽样 shape。
4. `identify`：把历史路径映射为稳定 data family 和 source ID。
5. `validate raw`：检查 checksum、可读性、复数表示、帧数、pose shape、NaN 和 annotation coverage。
6. `build manifest`：不复制数据，先建立 source manifest。
7. `deduplicate`：基于路径、大小和 checksum 抽样识别重复资产。
8. `build interim`：只生成必要且可恢复的姿态、雷达帧或特征。
9. `build splits`：按 subject/signer/sequence/scene group 生成确定性 split。
10. `materialize processed`：仅为正式实验生成 model-ready 数据，并设置 quota。
11. `validate processed`：schema、shape、统计、泄漏和小样本可视化。
12. `deliver`：冻结 `delivery.json`、Parquet inventory/checksum、reader parity 和 adapter smoke；只有通过后
    才可把该 delivery 绑定到 formal run。

## 7. 必须回答的盘点问题

- 原始 CSL-Daily/CSL-News 是否仍受许可证或访问限制？
- collected 数据中的 sequence ID 如何映射到采集者、动作、场景和采集日期？
- 雷达原始数组 `[chirp, antenna, sample, real/imag]` 是否始终一致？
- 历史 `128 x 86 x 256`、天线子阵列和 radar config 是否存在版本差异？
- pose 的 24 joints 定义、坐标系、单位和 normalization 是否在所有数据族一致？
- paper 使用的是哪个 split，是否存在同 subject/sequence 泄漏？
- 旧 features/pred_pose 是由哪个 checkpoint、commit 和预处理生成？
- 哪些 checkpoint/metrics 是投稿版本真正使用的？

## 8. 当前下一步

1. 作者先提供 NAS 上所有待上传 archive/目录的名称、估计大小和“可重下/不可重下”标记。
2. 先把两条 CSL-News historical pose 样本按原相对路径放入已建立的对照批次，完成新旧逐帧审计。
3. 优先上传体积小的匿名 metadata、雷达配置、阵列映射、标定和仿真 provenance。
4. 分批上传私人 raw captures，优先原投稿 test split 与 `collected_csl` 对应来源。
5. 每批完成 checksum、只读 inventory 和 data registry 登记后，再批准下一批。
6. 监控 CSL-News 下载服务；每个 final ZIP 必须先通过完整 CRC gate，才可进入标注或 manifest promotion。
7. 保留 primary 异常 `archive_001/005/008`、partial output 和失败 sidecar；当前只通过 v2 registry
   选择已验证的 versioned replacement，不移动、覆盖或删除原件。

## 9. CSL-News 官方下载状态

```text
source: ZechengLi19/CSL-News
revision: 3a0601210333fe760efd09b5d9e2ae5f341ce339
license: CC BY-NC 4.0
compressed_size: 935001573087 bytes
archives: 436
incoming: /mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121
```

运行服务：

```bash
systemctl --user status mmprism-csl-news-metadata.service
systemctl --user status mmprism-csl-news-archives.service
```

metadata unit 已于 `2026-08-11T14:57Z` 以 `Result=success`、exit code 0 完成；archive unit
继续下载。固定 revision 下的 metadata 已原子落盘：

| File | Bytes | SHA-256 |
|---|---:|---|
| `CSL_News_Labels.json` | 199,441,318 | `3381d80157fa75012ec2a220eb8a63c88968af2d60d5dbcb5a82bf680db8a3a5` |
| `CSL_News_Labels.csv` | 148,851,954 | `683e2c71bc48d9cb6210118799836c7afa4a11269a41bab1dfa4fbbb1d0cee79` |
| `README.md` | 2,670 | `cc0c6367538d1eedb07f199e1a4d56edf74a2026b0718feae112400911b5ba23` |

`csl-news-metadata-profile` 对固定 revision 的三份 metadata 做完整扫描。报告
`profile_20260811T151215Z.json` 为 `passed_with_warnings`：JSON 的 722,711 条记录全部有效、
非空且 video/pose key 唯一；CSV 覆盖所有 JSON key，但包含 4 个冲突重复行。canonical pipeline
固定使用 JSON，CSV 只作交叉审计，不修改任何上游文件。完整译文长度统计、字符集定义和 reviewer
缺失字段见 `../evidence/csl_news_metadata_profile.md`。

`2026-08-11T15:25Z`，clean commit `96ccc6e` 生成首个 available-archive source snapshot：11 个
archive、18,095 条 portable `caption/video` record，manifest SHA-256 为
`6984d0cc30a0f5a9e6baa58fa8a764e0c0b70ed1b0bb9224e9fca8faa1b1a1f5`。通用 contract、绝对路径
扫描和当前 676 个 pose sidecar 的 ID/text 交叉检查均通过，但该 snapshot 未执行完整 CRC。
后续逐 member 审计确认其中 `archive_005/008` 损坏，因此它只保留为 schema/linkage evidence，
不能作为 integrity-verified 输入或论文统计。详见 `../evidence/csl_news_source_manifest.md`。

完整 CRC audit artifact 位于
`/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1/audit_20260811T154138Z`；
11 个 archive 中 9 个通过（14,844 videos）、2 个失败（3,251 videos），missing label/empty text 均为 0。
总表 SHA-256 为 `ea8062f546cdf10abdde5b5b27e0e78e5e39e3df538e0d68b983e6ac4b7c9a00`。
primary `005/008` 保持原位并作为历史 source failure 排除；当前 replacement 已由 v2 registry 验收；详见
`../evidence/csl_news_source_integrity.md`。

`2026-08-11T15:51Z`，新出现的 `archive_001.zip` 无法打开 central directory。下载日志确认 aria2
在 93% 时因 HF 临时签名 URL 返回 HTTP 403，但旧 `xargs` 子 shell 未传播非零状态，错误执行了
`.part -> .zip`。clean-commit incident report SHA-256 为
`379b72f34a1f749a246891901e746defae3331dcb65fab64662686a7f260a723`。下载器现要求 transfer exit 0、
无残留 `.aria2`、完整 `unzip -t`/CRC 通过后才 promotion；修复版 service 已恢复，原 `001` 未移动或覆盖。

`archive_002` 是修复版 downloader 首个完成 promotion 的 ZIP，随后通过 canonical full-read audit：
1,624 个视频、label coverage 1,624/1,624、missing/empty 0，archive SHA-256 为
`a10864019a02d5abefe1045b1ce7fc3f3350562889e4b6c95cfe766981334fde`，report SHA-256 为
`3f2eaffd97c1f48481d92f7f88f5bd8ce68d78cce3bc74f0acbb9d8e0c43c4e9`。

`2026-08-11T16:16Z` 起，`configs/data/csl_news_source_integrity.yaml` 和
`csl-news-integrity-scan` 维护 cumulative atomic registry。首次 clean-commit 扫描覆盖 14 个 final ZIP，
后续增量扫描复用全部 14 个结果并自动审计新晋升的 `archive_017`。`16:17Z` registry 覆盖 15 个 final：
12 个通过、3 个失败（`001/005/008`），白名单共 19,760 videos；该快照 SHA-256 为
`070bcc4446894577cab6e05f632049a2a53143b508e50523dd27c20daea52b66`。每个 archive 有独立
SHA-256、source stat、audit report/hash 和 clean builder commit；标签 hash 变化会强制全部重审。

`2026-08-11T22:10Z`，source-integrity v2 registry 完成 replacement overlay 切换。primary
`001/005/008` 保持原位；通过验证的新文件位于
`replacements/20260811_recovery_hf_3a060121/rgb_archives/`，registry 逐项保存精确相对路径和
`source_kind=replacement`。三份 replacement 均通过完整 SHA-256、逐 member CRC、label coverage
和单视频 decode：

| Archive | Replacement SHA-256 | Videos | Audit SHA-256 |
|---|---|---:|---|
| `001` | `911ed805d80842867c0ecebc86c2f8ad0fbd6790269861dbdc964ebaa9bab7ec` | 1,694 | `eee22ef84c43c62f623b660985c246970b2bcabf31a30e9a02faac3398f0978a` |
| `005` | `3450d136994df60739ff8bf62382b36005de81a91c911921348e88f378542dd3` | 1,632 | `0a3542633b0aac14c5b6b0bff3d559565a8dd03b10121634110e8ddfba7303de` |
| `008` | `b258e4bebaf36623e65066438c3956a6f0ba8579e8df36f4a399297d5b291153` | 1,619 | `39615ae9f529f6b11c025062f4f2a5ddcee89e5daa16fd237e2c601da2c747c6` |

该 registry byte snapshot SHA-256 为
`ae6b2909e7b12c3f9519ffc493b67a556621d6e7203665b940ea4bee9878a02c`，覆盖 59 个 present archive，
59/59 passed、97,997 videos、failed 0。它仍是 436-archive 下载中的 partial registry。

下载使用 `scripts/download_csl_news.sh`，当前引擎为 aria2：4 个 archive worker、每文件 8 个连接、
断点续传、ZIP 完整性通过后原子 promotion、只下载不展开，并保留至少 1 TiB 可用空间。切换前短时基准中，
单个 aria2 传输稳定约 3.4 MiB/s，原 16 路 curl 同窗口合计约 4.7 MB/s。Legacy 预处理链与接口冲突见
`../../../docs/architecture/csl_news_data.md`。

切换并稳定后，5 个 aria2 进程的 60 秒有效写入为 9.95 MB/s。多段 Range 写入会提前扩展
`.part` 的表观文件大小，因此后续进度不再通过 `.part` stat/du 求和，而以完成 `.zip` 数和 aria2
日志中的完成字节为准。

首批只读审计由 `scripts/run_csl_news_source_trial.sh` 执行，产物写入：

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1/
```

`2026-08-11T14:29Z`，`archive_003.zip` 审计通过：1,657 个视频、CRC 无失败、无不安全或
重复 member、全部命中官方非空文本；archive SHA-256 为
`ae348f6cc3088cc755d5af4f4320c3f6851a5564fb27345b3fb0e150f1a655d4`。

`2026-08-12T00:00Z` 的计划 trial 按时触发，但旧脚本按 primary 文件名选择了已知事故原件
`archive_001.zip`，因此以 `BadZipFile` 失败；该失败不代表 v2 当前来源回退。clean commit
`96701de` 将 trial 改为只接受 typed source-integrity registry entry，并在执行前复验 registry/config、
路径、stat、archive SHA-256 和 labels SHA-256。`00:25Z` 补跑正确选择 replacement `001`，完整
1,694-member CRC/label coverage 与 3 个视频 decode 全部通过；selection registry SHA-256 为
`da5711261201917ac42f6036f4533642662290cf1019ad5b05c7d379d8e35c9c`，通过产物位于
`source_trial_v1/20260812T002504Z_archive_001_da5711261201/`。失败目录和 primary 原件均保留。

canonical pose annotation 使用 `configs/data/csl_news_rtmw3d_overnight.yaml` 和
`paper/manager/runbooks/csl_news_annotation_overnight.md`。首个真实样本生成 125 帧原生
`[T,133,3]` 与 canonical `[T,2,24,3]`，全部数值有限，峰值显存约 262 MiB；正式输出和
scratch 在次晨人工检查前全部保留。

夜间 4-worker pool 固定 GPU 7，现由 source-integrity v2 registry 动态分片；只有 `passed` archive
的精确登记路径可见，
新 final 由 5 分钟 integrity timer 审计后自动进入对应 worker。运行后由
`csl-news-annotation-status` 做只读健康快照。
`2026-08-11T14:47Z` 报告为 `healthy`：10 个完整 archive、16,476 个当前可用视频、
101 个成功样本、latest run 新增失败 0、缺失 artifact/sidecar 0、抽样校验 3/3 通过。
报告目录为：

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/reports/
```

GPU worker 已获明确授权与其他任务共享 GPU；运行门槛只检查可用显存，不以 GPU 利用率判断
是否启动、暂停或迁移。`2026-08-11T14:56Z` 的正式只读 QC 在 246 个候选产物中确定性抽检
100 个、共 24,628 帧，状态为 `passed` 且无 warning：canonical valid ratio 为 0.99245，
transformed 2D in-bounds ratio 为 0.98769，frame count/checksum/shape/finite/FPS 契约全部通过。
QC 报告目录为：

```text
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/qc/
```

`15:00 UTC` 首次 timer 自动触发已以 `0/SUCCESS` 验收；对应状态为 `healthy`，11 个完整
archive、18,095 个可用视频、291 个成功样本、当前 run 新增失败 0、缺失配对 0、抽样 3/3
通过。worker 保持 `NRestarts=0`。

`16:00 UTC` 自动报告为 `attention_required`，原因仅为已隔离的 `archive_001` central-directory
错误；annotation artifact 配对完整、最新 3/3 校验通过、latest run 新失败 0。4-lane aggregate
近期吞吐约 1,394 samples/hour；该 ETA 包含未调度/异常 archive，仅作运维参考。

`16:20 UTC` registry-aware 报告只统计 12 个 eligible archive/19,760 videos：1,687 个 eligible NPZ，
15 个来自损坏 archive 的历史 NPZ/sidecar 单列为 ineligible 且不抵扣进度；最新 3/3 校验通过、
dynamic run 新失败 0，近期约 1,353 samples/hour。报告保持 `attention_required`，原因是 registry
显式保留 3 个失败 source，而不是 annotation 失败。

`16:30 UTC`，两个后续 timer 周期自动审计通过 `archive_007/013`。当前 registry 覆盖 17 个 final，
14 个通过/23,020 videos，失败项仍仅为 `001/005/008`；registry SHA-256 为
`d48a35dc8196400b962ba704258136edc25553a491917371795236b9d9299512`。同步状态报告统计
1,889 个 eligible NPZ、15 个 ineligible 历史 pair、抽检 3/3、dynamic run 新失败 0，近期约
1,428 samples/hour。

`16:42 UTC`，clean commit `390093b` 生成首个 integrity-gated pose+caption partial snapshot：
2,157 条 record、5 个 represented archive、1,169,173,125 bytes referenced artifact。snapshot 绑定
当时 14 passed archive/23,020 videos 的 exact registry bytes（SHA-256
`183743fbb60bb85b75dd63f6c112e0c1a3081b2b6a391e32fa6ce2a21cb5b02d`），manifest SHA-256 为
`4161593fdbfc85a5c2fb392e3ef92d40da560db5c75a19d559f1f92878e31600`。15 个 failed-archive 历史
pair 被保留并排除，eligible NPZ 无未配对项；checksum、portable path、通用 contract 和首/中/末
adapter 读取均通过。详见 `../evidence/csl_news_pose_manifest.md`。该 snapshot 仍为 partial，后台
下载和标注继续运行，不能作为全量论文统计。

`16:52 UTC`，clean commit `eb5de64` 对上述 frozen pose manifest 生成首个 canonical partial split。
协议 `csl_news_pose_sequence_hash_80_10_10_v1` 使用 `sequence_id`、seed `20260811` 和整数权重
`8/1/1`；train/validation/test 分别为 1,701/219/237 samples。2,157 条 source 全覆盖，2,157 个
sequence/group 全部唯一，跨 split group leakage、重复 sample、missing/extra coverage 均为 0；
2,157 个 group ID 和 hash bucket 独立重算一致。assignments SHA-256 为
`133f32d58b213947edf09c7c1e1b7c3ee30b8588a9f2b7a863d6a668bce2d7d9`，详见
`../evidence/csl_news_pose_split.md`。由于 source 仍是 partial 且无 signer/subject metadata，本结果
只证明 sequence-disjoint split 工程链，不支撑 subject-independent、新用户泛化或最终论文 split。

## 10. CSL-News Derived Identity Incident And Recovery

`2026-08-11T21:00Z` 后一次 pose-manifest 构建在
`archive_006/3af7db9841fb2ac483721620` 处按 checksum gate 停止。sidecar 声明 0 bytes 和空文件
SHA-256，但实际 NPZ 为 813,674 bytes，SHA-256 为
`6914b6bb0f26304d87b14d7cd7e8b00ac13e6d65202a97c0d4a89e3b0d38bca3`。原 pair、failure records
和 `.snapshot_20260811T210301.065913Z.tmp.925176` 均保留，未覆盖或清理。annotation publisher 已加入
same-directory temp、flush/fsync、promotion 后 size/SHA 重验、size+SHA resume gate 和 conflict-continue。

clean commit `3bdd31f6b0b9f43c8c3458df79a653346eda8c4e` 的 CPU-only full identity audit 冻结
9,519 个 published pair，流式哈希 5,115,703,846 bytes；9,518 通过，唯一异常即上述 pair，且其
sidecar/NPZ 在 hash 前后 stat 完全稳定。报告位于
`identity_audits/audit_20260811T212324Z.json`，SHA-256 为
`55478cbb6078d7e4c7b0c9a95577e6260e249239514ec584d082d5b0b4c538b4`，`audit_failures` 为空。

按 `DEC-029`，该异常不做原地修补，而由 versioned checksum-bound exclusion 隔离。clean commit
`98549a92b7ca22adbcbed6a241d139f07ed64ec0` 生成
`snapshot_20260811T212450.135852Z`：冻结 9,552 个 eligible sidecar，显式排除 1 个，最终 manifest
9,551 records/9 represented archives、0 unpaired NPZ，manifest SHA-256
`8e3db8712bc61848e9d6dea9f5b3a3821365ffd102d6643977ad43107b2db0c4`。四项 `SHA256SUMS`、通用
contract 和首/中/末 checksum-validating adapter 读取均通过。snapshot 绑定的 registry 覆盖 51 个
final archive，其中 48 passed/79,813 videos，失败仍仅 `001/005/008`。这些都是 partial pipeline
evidence，不作为最终数据规模或论文指标。

`17:14 UTC`，integrity timer 在 clean commit `8b64d0f` 下自动审计通过新晋升的 `archive_026`：
1,598 videos、完整 CRC/label coverage 通过。当前 registry 覆盖 18 个 final，15 个通过、24,618 videos，
失败项仍仅为 `001/005/008`；registry SHA-256 为
`b150b679877568a092f1dcb61b0c9a35648434e339ce7603ff841dde29ae0ce1`。上一周期因主仓库有未提交
稿件审计修改而按 clean-Git gate 拒绝更新，提交后下一周期自动恢复，无需重启 worker。

`17:15 UTC` 状态快照统计 2,916 个 eligible pose/sidecar pair、21,702 个当前白名单剩余样本，
missing artifact/sidecar 均为 0、latest run 新失败 0、抽检 3/3 通过；近期约 1,409 samples/hour，
当前白名单 ETA 约 15.4 小时。报告 SHA-256 为
`a43efcb4c7529809323b4b300cd890a6ef060afb341bf510b307913c89ba5139`。同期 100-sample QC 为
`passed`、warning/failure 均为 0，共检查 24,601 帧；canonical valid ratio 0.99196、transformed
in-bounds ratio 0.95260，报告 SHA-256 为
`c6521e0d1c40fcf92e40a7d71ebf2b31331e190b5225e6139f2226ddc9dbe2ca`。

`17:30 UTC`，提交稿件 display registry 的 clean commit `1fc0d55` 后，手工触发同一 integrity
oneshot 并以 `0/SUCCESS` 完成；新 final `archive_027/030` 分别以 1,577/1,780 videos 通过完整
CRC、路径安全和 label coverage。当前 registry 为 20 final、17 passed/27,975 videos，失败项仍仅
`001/005/008`，SHA-256 为 `ed848abce94683d74aca8bbc985a365315fec5983ed74e44261a09165927d804`。
`17:31 UTC` 状态快照统计 3,287 个 eligible pair、missing artifact/sidecar 0、latest run 新失败 0、
抽检 3/3 通过，近期约 1,436 samples/hour；四个 worker 均 `active/running`、`NRestarts=0`。

`17:52 UTC`，clean commit `8c27fb9` 后手工触发 integrity oneshot 并以 `0/SUCCESS` 完成；
`archive_032/034` 分别新增 1,678/1,770 个通过完整 CRC、路径安全和 label coverage 的视频。当前
registry 为 22 final、19 passed/31,423 videos，失败项仍仅 `001/005/008`，SHA-256 为
`6ad8310cdbe934ff291a3e68d6ea151231e2b84c13c650e3cc939f8bf23b1338`。`17:53 UTC` 状态快照统计
3,795 个 eligible pair、missing artifact/sidecar 0、latest run 新失败 0、抽检 3/3 通过，近期约
1,382 samples/hour；四个 worker、下载服务和 integrity timer 均为 active。

`19:18 UTC`，clean commit `84f2c52` 后手工恢复一次先前被 dirty-Git gate 拒绝的 integrity scan，
并以 `0/SUCCESS` 完成；新增 `archive_046/051` 后 registry 为 31 final、28 passed/46,521 videos，
失败项仍仅 `001/005/008`，SHA-256 为
`0c96b5ed3f2acda9f5484731b44eb1f3f658cbee2aea3575d357ee311b4c30a4`。`19:17 UTC` 状态快照统计
5,710 个 eligible pair、missing artifact/sidecar 0、latest run 新失败 0、抽检 3/3 通过，近期约
1,484 samples/hour；报告 SHA-256 为
`01f489f238891cf6fd3ff392bfcbb57efc2ce708a40cb793f2bfa3e67196df0a`。四个 registry worker 均
`active/running`、`NRestarts=0`；status 的 `attention_required` 只来自三个已隔离 source failure。

`19:23 UTC`，integrity timer 在 clean commit `10a30e5` 下完整审计并通过 `archive_052`：1,689 个
视频的逐 member CRC、路径安全和 label coverage 均通过，archive SHA-256 为
`dfa60be4fb10bd3eb46465e62f62f0938677795e88b9f08134614c90af86ecc0`，audit SHA-256 为
`3fb760832e7b26a3b5ed7c34f2bd7936fe60bd908c89cf156dd21cb9a72a3ba1`。registry 更新为 32 final、
29 passed/48,210 videos，失败仍仅 `001/005/008`，SHA-256 为
`f1a5cd753c32df399dbac59d9102470bdec7262396ca0f0b50c6245386c3ce94`。

`19:30 UTC` 自动状态报告统计 6,017 个 eligible pair、missing artifact/sidecar 0、latest run 新失败
0、抽检 3/3 通过，近期约 1,488 samples/hour；报告 SHA-256 为
`b60c277162be81e981a9c261e10c0dbfc2d71ba0db2f037e2d9ed21f8db6e27e`。四个 registry worker 均
`active/running`、`NRestarts=0`；`attention_required` 仍只来自三个已隔离 source failure。

`22:13 UTC` 的 v2 source-aware 状态报告为 `healthy`：59 个 archive/97,997 videos 可调度，
9,394 个当前来源 NPZ/sidecar pair，缺失 pair 0，当前来源重复 0，latest run 新失败 0，抽检 3/3
通过。另外 1,875 个缺少或不匹配当前 source identity 的历史 pair 单独隔离且不计入完成度。四个
`registry{0..3}-v3` worker 的 orchestration metadata 分别为 `worker_index=0..3`、
`worker_count=4`，共同使用 GPU 7；GPU 利用率不作为停止条件。

clean commit `11014a82627726758e3f6f24b82455e976c61c2b` 的新 identity audit 冻结 11,815 个
pair、哈希 6,373,342,155 bytes，11,814 通过；唯一失败仍为已登记的 `archive_006/3af7...`，
没有新增 conflict。报告 SHA-256 为
`23278c988156ce27e52405794642f7e77ab0ec44d93c43be93da1626d5864105`。

同一 clean commit 构建的 v2-bound snapshot
`snapshot_20260811T222941.214512Z` 包含 10,011 records/12 represented archives，manifest SHA-256
为 `3412aeb2f7fea685796e17d85b3af6342b7ffe1b3a61895446295f5f71e073f7`。1,875 个旧来源或
unbound sidecar 进入 `source_identity_quarantine.jsonl`，ledger SHA-256 为
`1b03721b4fc64601d8dff0fc247e6d7a1a319ac93d2dc25c6cc463f0cd659586`；当前来源 unpaired NPZ
为 0。五项 `SHA256SUMS`、通用 manifest contract 和首/中/末 adapter checksum 读取全部通过。
该 snapshot 是 partial pipeline evidence，不是最终数据集规模或论文结果。

`22:30 UTC` 自动 status 报告返回 `attention_required`，唯一新增 failure 是已登记的
`archive_006/3af7...` 在当前 worker run 中再次触发 preserve-on-conflict；没有新增 sample identity，
current-source duplicate 0、missing pair 0、抽检 3/3 通过。status service 以 exit 1 保留该告警，
四个 annotation worker 和下载/integrity 服务不受影响。

`22:32 UTC` integrity timer 在 clean commit `11014a8` 下继续通过 `091/094/096`，live v2 registry
更新为 62/62 passed、102,949 videos、failed 0，SHA-256
`b461c9efd619ca2a049f4f64c9758bf7d6c64fb603a06ea64123148d13542e1a`。已发布的 10,011-record
snapshot 仍绑定其冻结时的 59-archive registry bytes，不受 live registry 后续更新影响。

clean commit `7f86516403612b9bb48a7668c4f78b833929e745` 随后生成首个真实 source-manifest v2 snapshot
`snapshot_20260811T224413.526848Z`。它冻结 63 个 archive/104,658 records，绑定并复制 exact registry
bytes（SHA-256 `dc2d7068f562dacb054b709845d38d57b4d6668205007a6f0f7a4900d2b81011`）；manifest/
summary SHA-256 分别为 `a431d14cd5f693a82d8f21c3c5c7ee05c9d27d2ee003c801db21dcfdc7434263` 和
`8758923881aa17edd2b89b7e7a24efe3f7850466c2ca3f028b6b4dc1d53ae02b`。三项 checksum、通用 contract、
portable path 和首/中/末 exact ZIP/member 读取全部通过，replacement `001/005/008` 共 4,945 条记录。
summary 的 `crc_checked=false` 只表示 manifest 冻结时未重复执行全量 CRC；复制的 registry 已对所有
入选 archive 完成逐 member CRC、label coverage 和 decode gate。该 snapshot 覆盖 14.4813% labels，
仍是 partial evidence，不能作为最终数据集规模。

`22:47 UTC` live v2 registry 已继续更新为 66/66 passed、109,797 videos、failed 0；该更新不追写
上述 frozen snapshot。下载、integrity timer 和四个 GPU 7 v3 worker 继续运行，worker 均为
`NRestarts=0`；项目负责人已批准共卡运行，GPU 利用率不作为启动、暂停或迁移条件。

`23:24 UTC` 仅滚动重启 registry lane 2，使其加载 clean commit `6e9cc5e` 的 immutable-conflict
recovery。`archive_006/3af7db9841fb2ac483721620` 被确定性路由到完整 source SHA 后缀并成功生成；
原 canonical NPZ/JSON 的 size、SHA-256 和 mtime 未变化，历史 failure records 仍为 4 条。恢复 sidecar
记录 `artifact.variant`、current source identity 和 clean run commit。lane 0/1/3、下载、timer 和其他
GPU 进程未触碰。

clean commit `6e9cc5e` 随后冻结 `snapshot_20260811T232708.554551Z`：70 个 passed source archive、
12,057 records/12 represented archives，manifest SHA-256 为
`cdd450e4d7e17d4f34266f199ed4ff61f1ead9584715f1d4b9d3286a97d086e5`。原坏 canonical pair 仍由
checksum-bound audit exclusion 精确隔离，恢复 variant 则作为同一 sample ID 的唯一 current-source
record 入选。五项 `SHA256SUMS`、通用 contract、portable path 和首/中/末/恢复样本 checksum adapter
读取全部通过。扫描中 1 个 NPZ 在冻结边界后约 20 ms 完成 sidecar 发布，因此只作为
`unpaired_npz_at_scan` 记录且未入选；源文件没有删除或覆盖。

`23:30 UTC` status 为 `healthy`：70 个 archive/116,202 videos、12,165 completed current-source
sample、recovered/shadowed-invalid 为 1/1、duplicate/missing pair 为 0、latest-run failure 为 0、抽检
3/3 通过。随后 integrity scan 于 `23:30:58Z` 将 live registry 推进到 71/71 passed、118,075 videos、
failed 0；该 live 更新不追写已冻结 snapshot。下载、四个 worker 和两个 timer 继续 active。

`2026-08-12T00:11 UTC`，epoch-resume 代码提交 `f0c6205` 使 worktree 恢复 clean 后，手工触发
integrity oneshot 并以 `0/SUCCESS` 完成。新 `archive_120` 含 1,636 个视频，完整 CRC、路径安全、
label coverage 和 decode probe 均通过；archive/audit SHA-256 分别为
`5a0c7b151714469067d008b84463a9fbb4de28bdc7b808b189eabb12f6705e10` 和
`2ee2bb7b9fc8095eafbd15c29cf96562c7e7ac2d8ec2b38c050df972242e526f`。live v2 registry 更新为
73/73 passed、121,465 videos、failed 0，SHA-256 为
`1f49b3e621c60b8bf9fd5ac96d49f0afdf9ba4abbae3d0773f26fa1ed989bcbf`。

`00:12 UTC` 手工 status `status_20260812T001155Z.json` 为 `healthy`：13,580 个 completed
current-source pair、remaining available 107,885、duplicate/missing pair 0、latest-run failure 0、
抽检 3/3 通过，近期约 1,799 samples/hour、32.07 frames/s；报告 SHA-256 为
`18ed292719a7a44fdecd26e52dd59cd4685a581e5f4035b18b7dba0a175294e0`。下载服务和四个 GPU 7
registry worker 均 `active/running`、`NRestarts=0`。当前 source intake 约 249 GB、pose output 7.9 GB、
annotation scratch 20 GB；原始 ZIP、partial、scratch、失败和 pose artifact 均未清理。

`00:30 UTC` 定时 status `status_20260812T003003Z.json` 继续为 `healthy`：74 个 registry-passed
archive、123,129 videos、14,125 个 completed current-source pair、remaining available 109,004、
duplicate/missing pair 0、抽检 3/3；近期约 2,180 samples/hour、36.85 frames/s。`00:38 UTC` 下载目录
有 76 个 final ZIP 和 51 个 `.part`，aria2 正处理约 `archive_123-127`；下载和四个 GPU 7 worker 均
`active/running`，没有清理任何 source、partial、scratch、failure 或 historical pose artifact。

`00:43 UTC` 起，integrity timer 从会随开发编辑变脏的主 worktree 隔离到 clean detached worktree
`/home/yanyifan/.cache/mmprism-runtime/csl_news_integrity_3f36094`，固定 commit `3f36094`，并使用其独立
UV `.venv`。clean-state gate 保持不变；连续两次 scan 均 exit 0，首轮新验收 `122/124/125/126`。
`00:45 UTC` 手工状态 `status_20260812T004515Z.json` 为 `healthy`：78 个 passed archive/
129,539 videos、14,655 个 current-source pair、missing pair 0、抽检 3/3，近期约 2,173 samples/hour、
37.91 frames/s；报告 SHA-256 为
`788a4e0988fed5112c21995ff64ae8fa0f71af9b8e237c37f746c909c52bfebc`。下载、四个 worker 和 clean
integrity timer 均为 `active`；主开发 worktree 的未提交 DDP 文件未被移动、覆盖或提交。

`01:31:43 UTC` 切换前状态为 `healthy`：81 个 integrity-passed archive、134,660 个可用视频、
16,162 个 current-source pose/text pair，duplicate/missing pair 均为 0，最近窗口约 1,857 samples/hour。
于 `01:32:52 UTC` 受控停止旧 GPU 7 的 `worker_index=0..3/worker_count=4` pool 后，重编排为 8 个互斥
worker：GPU 7 lane `0..3`、空闲 GPU 5 lane `4..7`，统一按 `archive_id % 8 == worker_index` 消费同一
source-integrity v2 registry。启动后约 3 分钟产生 157 个 current-source pair、0 新 failure；最近 120 秒
产生 111 个，即约 3.1k--3.3k samples/hour。8/8 worker 均 `active/running`、`NRestarts=0`，GPU 5/7
均约 99% utilization。短窗值仅为扩容验收，正式稳定吞吐仍由下一次 CPU-only status snapshot 记录；下载、
integrity timer、raw ZIP、`.part`、scratch、failure、quarantine 和既有 pose artifact 均未移动、删除或覆盖。

`2026-08-12` 收到 `/home/yanyifan/upload/20260812/archive_002.zip`（SHA-256
`3b3af27c...`）；ZIP 完整性通过，含 1,624 条 historical `float64 [T,59,3]` pose。这不是预期的
2x24 final tensor，而是 old `17 body + 42 hand` intermediate representation。只读对照依据 archive_002
current source SHA `a1086401...`、sidecar legacy pose identity、native 133-joint 重建相同 depth-center/
59-joint view：coverage、frame count、shape、depth center 全部为 1,624/1,624；1,567 条 current-source-bound
序列的 median sequence mean absolute error 为 `5.257e-05`，但没有任何 bitwise 或
`allclose(rtol=1e-5, atol=1e-6)` 等价项，且少数 body-joint frame 有最高 `1.379` 的离散跳变。
这证实 transform/coverage 对齐但 historical inference environment 未实现数值等价；该 ZIP 与 57 条
source-identity-unbound current pair 均保持 forensic-only，不能进入 canonical manifest。详细证据与报告 hash
见 `../evidence/csl_news_legacy_pose_comparison.md`。
