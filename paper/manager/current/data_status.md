# Data Status and Rebuild Plan

Status: `asset_location_blocked_target_schema_proposed`
Last Updated: `2026-08-11`
Role: `data_source_of_truth`

## 1. 当前盘点结果

`/mnt/gfs/yanyifan` 当前只发现：

```text
Falcon/
Falcon_checkpoint_archive/
huggingface/
```

未发现目录名包含 `mmPRISM`、`CSL-Daily`、`CSL-News`、`collected_base`、`collected_demo`、`OmniHand` 或 `WaveLLM` 的资产。

共享 `/mnt/gfs` 当前约 10TB，总使用率 98%，剩余约 207GB。正式数据重建前必须先确认源数据位置并做容量预算，不能直接复制历史目录。

## 2. 代码推断出的历史数据族

| Data Family | 历史用途 | 预期主要模态 | 当前状态 |
|---|---|---|---|
| CSL-Daily | OmniHand simulation、WaveLLM caption | images, pose, pred_pose, feature, annotation | missing_location |
| CSL-News | pose annotation、simulation、WaveLLM caption | video, pose, signal/feature, caption | missing_location |
| Collected Base | 真实毫米波 OmniHand | color, raw mmWave, pose | missing_location |
| Collected Demo | 真实毫米波开发/演示 | color, raw mmWave, pose, pred_pose | missing_location |
| Collected CSL | 真实手语采集 | color, raw mmWave, pose, caption | missing_location |
| Model Weights | RTMPose3D、CubeNet、MT5、semantic evaluators | checkpoints/tokenizers | missing_or_partial |
| Historical Runs | paper metrics and checkpoints | config, ckpt, predictions, metrics | missing_location |

## 3. 建议数据根目录

在来源与容量确认后，建议使用：

```text
/mnt/gfs/yanyifan/mmPRISM/
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

该路径当前仅为 proposed decision；在 `DEC-003` 接受且容量审计完成前，不批量 materialize 数据。

## 4. Canonical Sample Record

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

## 5. 数据重建顺序

1. `inventory`：只读扫描目录、大小、文件数、后缀、mtime 和抽样 shape。
2. `identify`：把历史路径映射为稳定 data family 和 source ID。
3. `validate raw`：检查可读性、复数表示、帧数、pose shape、NaN 和 annotation coverage。
4. `build manifest`：不复制数据，先建立 source manifest。
5. `deduplicate`：基于路径、大小和 checksum 抽样识别重复资产。
6. `build interim`：只生成必要且可恢复的姿态、雷达帧或特征。
7. `build splits`：按 subject/signer/sequence/scene group 生成确定性 split。
8. `materialize processed`：仅为正式实验生成 model-ready 数据，并设置 quota。
9. `validate processed`：schema、shape、统计、泄漏和小样本可视化。

## 6. 必须回答的盘点问题

- 原始 CSL-Daily/CSL-News 是否仍受许可证或访问限制？
- collected 数据中的 sequence ID 如何映射到采集者、动作、场景和采集日期？
- 雷达原始数组 `[chirp, antenna, sample, real/imag]` 是否始终一致？
- 历史 `128 x 86 x 256`、天线子阵列和 radar config 是否存在版本差异？
- pose 的 24 joints 定义、坐标系、单位和 normalization 是否在所有数据族一致？
- paper 使用的是哪个 split，是否存在同 subject/sequence 泄漏？
- 旧 features/pred_pose 是由哪个 checkpoint、commit 和预处理生成？
- 哪些 checkpoint/metrics 是投稿版本真正使用的？
