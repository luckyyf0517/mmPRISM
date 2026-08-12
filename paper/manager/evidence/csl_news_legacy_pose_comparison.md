# CSL-News Historical Pose Comparison

Status: `archive_002_forensic_comparison_complete_environment_equivalence_open`
Last Updated: `2026-08-12`
Role: `DATA-005-D_historical_pose_forensic_evidence`

## 1. Immutable Intake

作者上传的历史 pose archive 保持在原位置，未复制、解压、改名或作为训练输入：

```text
path: /home/yanyifan/upload/20260812/archive_002.zip
SHA-256: 3b3af27cb8dfdcb6d284d354d8e860af13d0b57e3252dc6762361795abfdb869
ZIP validation: unzip -t passed
members: 1,624
layout: archive_002/<legacy_video_stem>.npy
array contract observed: finite float64 [T,59,3]
```

这不是此前约定的两条 `float32 [T,2,24,3]` 最终 pose 样本，而是一整份历史
`17 body + 42 hand` 中间表示。它仍然可以用于验证当前 native RTMW3D 输出的对应视图。

当前官方 source 已通过 integrity v2 registry：

```text
archive: archive_002.zip
source SHA-256: a10864019a02d5abefe1045b1ce7fc3f3350562889e4b6c95cfe766981334fde
status: passed
videos: 1,624
```

## 2. Comparison Protocol

`src/mmprism/data/csl_news_legacy_pose_comparison.py` 是本次只读、CPU-only 对照实现。
对每一个 historical `.npy`，它通过 `annotation.legacy_pose_name` 的 stem 唯一关联当前 sidecar，
从当前 `.npz` 的 immutable `native_keypoints_3d [T,133,3]` 按 historical rule 派生：

```text
depth_center = mean(native[:, [6, 7], z]) over the complete sequence
legacy_view = native[:, [0:17, 91:133], :]
legacy_view[..., z] -= depth_center
```

该规则等价于 root `run_csl_news_annotation.py` 中保存最终 2x24 前的 59-joint 中间张量。每项检查：

- legacy/canonical member identity、shape、dtype、frame count、caption hash；
- embedded source SHA identity；
- depth center sidecar value 与 native-output 重算值；
- `array_equal`、`allclose(rtol=1e-5, atol=1e-6)`、每轴/每关节 mean/max absolute error。

报告仅写入版本化 derived destination：

```text
report directory:
  /mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/
  rtmw3d_l_794dbc78_v1/legacy_comparisons/20260812_archive_002_v2/
summary.json SHA-256:
  ec6d2ece3b38af5b158a4f222771e5dcfb11586624371465bb43f11c563faa2e
entries.jsonl SHA-256:
  6c7a6eb69a7e56cd324887c18c8411c8f05a4ef8d975e99003462750a8eba60b
```

## 3. Coverage And Results

| Check | Result |
|---|---:|
| historical `.npy` members | 1,624 / 1,624 readable |
| matched canonical sidecars and exact source video member stem | 1,624 / 1,624 |
| matched frame count and `[T,59,3]` shape | 1,624 / 1,624 |
| reported vs recomputed depth center exactly equal | 1,624 / 1,624 |
| current-source-bound sidecars | 1,567 |
| old/unbound sidecars | 57 |
| strict current-source bitwise equal | 0 / 1,567 |
| strict current-source `allclose(1e-5, 1e-6)` | 0 / 1,567 |

对于 1,567 条 current-source-bound 项：

```text
sequence mean absolute error: median 5.257e-05, mean 5.980e-05, max 5.540e-04
sequence peak absolute error: median 3.520e-01, 99th percentile 1.009e+00,
                              maximum 1.379e+00
aggregate mean absolute error by axis (x, y, z):
  [2.390e-05, 2.699e-05, 1.285e-04]
```

单个大峰值并不表示整段偏移。例如最大项
`Common-Concerns_20240322_68762-69237_304101` 有 475 帧，sequence mean absolute error 为
`6.030e-05`；只有 10 帧 peak error 大于 `0.01`、4 帧大于 `0.1`、1 帧大于 `1.0`。
最大误差发生在 frame 414、59-joint index 16、y axis，historical/current 值分别为
`0.6532466802` 与 `-0.7257693`。此类离散帧跳变集中于 body 末端 joints 13--16，不能由固定的
left/right 交换、59-joint mapping 或 depth-center rule 解释。

## 4. Interpretation Boundary

本对照确认当前 pipeline 与历史数据在以下方面一致：完整 sequence coverage、video/label identity、
frame count、59-joint selection，以及 sequence-wide depth centering。它**不**确认 historical 与 current
RTMW3D inference 为逐帧数值等价：所有可比序列均不满足 bitwise 或严格 `allclose`。

最可能的未闭合变量是 historical MMPose/PyTorch/CUDA/decoder runtime 或历史生成脚本版本；上传产物
为 `float64 [T,59,3]`，而当前 root historical script 可见的最终保存路径为 `float32 [T,2,24,3]`，也说明
上传 archive 并非该可见脚本的同构最终产物。不得把该差异归因于重构错误，也不得据此宣称 numerical
reproduction；需要原历史环境 lockfile、MMPose commit/config/checkpoint 文件与实际 invocation 才能关闭。

所有 historical entries 仍只作 forensic evidence，且 57 条 `source_identity_unbound` 项只保留数值参考，
明确排除在 strict current-source 等价结论和 canonical training manifest 外。

## 5. Next Action

1. 保留 `archive_002.zip` 原位置；不写回、不 promotion、不删除。
2. 若 NAS 保留历史 environment/config，优先上传相应的 MMPose checkout revision、RTMW3D config、
   checkpoint checksum 和启动命令，而不是重复上传可再生 pose。
3. 选择 `archive_007` 的一条 Common-Concerns 和一条 Dragon-TV final historical pose（原始路径/格式）
作为第二轮 clean source-identity cross-check；或上传整个 `archive_007`，其当前 1,708 条 sidecar 均已
绑定 current primary source SHA。
