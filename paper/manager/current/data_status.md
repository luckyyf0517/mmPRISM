# Data Status and Rebuild Plan

Status: `csl_news_download_active_other_asset_intake_blocked`
Last Updated: `2026-08-11`
Role: `data_source_of_truth`

## 1. 当前盘点结果

`/mnt/gfs/yanyifan` 当前只发现：

```text
Falcon/
Falcon_checkpoint_archive/
huggingface/
```

盘点开始时未发现目录名包含 `mmPRISM`、`CSL-Daily`、`CSL-News`、`collected_base`、
`collected_demo`、`OmniHand` 或 `WaveLLM` 的资产。现已建立 canonical incoming root，并开始下载
CSL-News 官方源；其他数据族仍未到位。

`2026-08-11T12:10Z` 复核时，共享 `/mnt/gfs` 约 10 TB，使用率 65%，剩余约 3.6 TB；
`/mnt/gfs/yanyifan` 约占 698 GB。空间属于共享动态状态，每次批量下载或解压前仍必须重新检查。

面向上传人员的完整 P0/P1/P2 清单见 `../data_upload_checklist.md`。

## 2. 代码推断出的历史数据族

| Data Family | 历史用途 | 预期主要模态 | 当前状态 |
|---|---|---|---|
| CSL-Daily | OmniHand simulation、WaveLLM caption | images, pose, pred_pose, feature, annotation | missing_location |
| CSL-News | pose annotation、simulation、WaveLLM caption | video, pose, signal/feature, caption | official_archives_download_in_progress_metadata_complete |
| Collected Base | 真实毫米波 OmniHand | color, raw mmWave, pose | missing_location |
| Collected Demo | 真实毫米波开发/演示 | color, raw mmWave, pose, pred_pose | missing_location |
| Collected CSL | 真实手语采集 | color, raw mmWave, pose, caption | missing_location |
| Model Weights | RTMPose3D、CubeNet、MT5、semantic evaluators | checkpoints/tokenizers | missing_or_partial |
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

上传批次使用 `incoming/<YYYYMMDD_source_batch>/`，附带 `UPLOAD_MANIFEST.csv` 和 `SHA256SUMS`。
容量审计完成前不批量创建目录、解压或 materialize 数据。

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

1. 作者先提供所有待上传 archive/目录的名称、估计大小和“可重下/不可重下”标记。
2. 优先上传体积小的匿名 metadata、雷达配置、阵列映射、标定和仿真 provenance。
3. 分批上传私人 raw captures，优先原投稿 test split 与 `collected_csl` 对应来源。
4. 每批完成 checksum、只读 inventory 和 data registry 登记后，再批准下一批。
5. 监控 CSL-News 下载服务；完成后生成 file/member manifest、SHA-256、ZIP integrity 和 label coverage report。
6. `archive_003` source-audit smoke 已提前通过；保留次晨 timer 作为独立复核。

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

下载使用 `scripts/download_csl_news.sh`，当前引擎为 aria2：4 个 archive worker、每文件 8 个连接、
断点续传、`.part` 原子完成、只下载不解压，并保留至少 1 TiB 可用空间。切换前短时基准中，
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

canonical pose annotation 使用 `configs/data/csl_news_rtmw3d_overnight.yaml` 和
`paper/manager/runbooks/csl_news_annotation_overnight.md`。首个真实样本生成 125 帧原生
`[T,133,3]` 与 canonical `[T,2,24,3]`，全部数值有限，峰值显存约 262 MiB；正式输出和
scratch 在次晨人工检查前全部保留。

夜间 worker 固定 GPU 7，运行后由 `csl-news-annotation-status` 做只读健康快照。
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
