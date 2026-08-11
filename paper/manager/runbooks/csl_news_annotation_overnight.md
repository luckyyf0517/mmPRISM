# CSL-News 夜间 RTMW3D 标注 Runbook

Status: `approved_for_execution`
Last Updated: `2026-08-11`
Role: `unattended_pose_annotation_operations`

## 1. 今晚目标与授权

- 对已经完整下载并原子命名为 `archive_*.zip` 的 CSL-News 视频持续生成 RTMW3D-L 姿态。
- 今晚只运行一个 GPU worker；后续可按 archive ID 取模扩展多个 worker。
- 操作者已明确批准与其他任务共享 GPU。选卡和运行期间只以可用显存为资源门槛，GPU 利用率不是启动、暂停或退出条件。
- 授权边界：可以在已有计算任务的卡上同时运行本 worker；只要满足最低可用显存即可，已有任务的 GPU 利用率不构成冲突或迁移理由。
- worker 将 OpenMP/BLAS/PyTorch 限制为 4 个 CPU 线程，并将 OpenCV 限制为 1 个线程。
- 不结束、暂停或修改其他用户进程。若任何卡达不到最低空闲显存，worker 退出并交给 systemd 稍后重试。

## 2. 严格禁止清理

在次日上午人工检查前，不得删除、移动或覆盖以下任何内容：

- 已完成或未完成的 ZIP、`.part` 和 aria2 控制文件；
- 已提取的视频和 `.source.json`；
- 成功 `.npz`、JSON sidecar、archive marker 和 run metadata；
- 失败记录、异常样本和进程中断留下的临时文件。

本 runbook 和 annotation service 故意不提供 cleanup 步骤。

## 3. 固定输入

```text
source: ZechengLi19/CSL-News@3a0601210333fe760efd09b5d9e2ae5f341ce339
archives: /mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/rgb_archives
labels: /mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/metadata/CSL_News_Labels.json
mmpose: 759b39c13fea6ba094afc1fa932f51dc1b11cbf9
checkpoint SHA-256: 794dbc78b04a43d81781f8ab0eba5b24f3dd5d71aaf6ae253940424159fb81ed
config: configs/data/csl_news_rtmw3d_overnight.yaml
```

环境使用 `mmcv-lite 2.1.0`。MMPose 会 eager import 与本任务无关的 EDPose head，而该 head 需要
当前 PyTorch/CUDA 组合没有官方 wheel 的 MMCV 扩展；annotation adapter 只将 EDPose 明确标记为
不可用，RTMW3D 实际依赖的 RTMW head、CSPNeXt、codec 和 checkpoint 仍按固定源码执行。
官方 checkpoint 使用旧版 PyTorch pickle 格式；其 SHA-256 验证通过后，worker 设置 PyTorch 官方
兼容开关 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。该开关不用于任何未登记或 checksum 不匹配的权重。

只读取最终 `.zip`，绝不读取 `.part`。标签缺失/重复、ZIP 结构不安全、模型提交或 checksum 不一致时停止任务。

## 4. 输出契约

每个样本用稳定 ID 写入：

```text
interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/
  samples/archive_NNN/<sample-id>.npz
  samples/archive_NNN/<sample-id>.json
  failures/archive_NNN/<sample-id>/attempt_<UTC>.json
  archives/archive_NNN.json
  runs/run_<UTC>_<pid>.json
```

NPZ 同时保留 `[T,133,3]` 原生 3D keypoints、`[T,133]` confidence、
`[T,133,2]` transformed 2D keypoints、帧号/时间戳，以及历史映射得到的
`[T,2,24,3]` canonical pose/confidence/valid mask。JSON 保存原始中文文本、archive/member、
源 revision、视频 checksum、crop/bbox/depth 规则、模型/config/checkpoint hash 和运行环境。

已提取视频保留在：

```text
/mnt/gfs/yanyifan/mmPRISM/cache/csl_news_annotation/rtmw3d_l_794dbc78_v1/videos/
```

## 5. 执行顺序

1. `uv sync --extra annotation` 安装锁定环境。
2. 对一个已完成 archive 做 SHA-256/CRC/label/decode source audit。
3. 在空闲显存最多且至少 30 GiB 可用的卡上执行单视频 smoke；允许该卡同时有高利用率任务。
4. 检查输出 shape、有限值、中文文本、sidecar checksum、峰值显存和每帧速度。
5. smoke 通过后，以同一物理 GPU 启动一个 `systemd --user` worker。
6. worker 持续轮询新下载完成的 archive，已验证输出自动跳过；逐视频普通失败写 sidecar 后继续。

## 6. 停止条件

- `/mnt/gfs` 可用空间低于 1 TiB；
- labels 无效、缺失或重复；
- ZIP 损坏、路径不安全或同 archive 视频 basename 冲突；
- MMPose commit、配置或 checkpoint hash 不匹配；
- 连续 2 个 CUDA OOM；
- 可用显存低于启动门槛时不加载模型，由 systemd 重试。

GPU 利用率高不属于停止条件，这是本次夜间运行的显式授权。

## 7. 启动与观察

单视频 smoke：

```bash
scripts/run_csl_news_annotation_worker.sh --gpu auto -- \
  --archive-id 3 --max-videos 1 --once
```

正式任务由管理员用 `systemd-run --user` 托管，设置 `Restart=on-failure` 和 300 秒重试；
启动后记录实际物理 GPU 到本页运行记录。观察命令：

```bash
systemctl --user status mmprism-csl-news-annotation.service
journalctl --user -u mmprism-csl-news-annotation.service -f
```

机器可读状态快照由独立 CPU-only 命令生成，不导入 MMPose/PyTorch，也不读取隐藏临时文件：

```bash
scripts/run_csl_news_annotation_status.sh
systemctl --user status mmprism-csl-news-annotation-status.timer
```

快照保存在 `.../rtmw3d_l_794dbc78_v1/reports/status_<UTC>.json`，包括 archive/video
可用量、成功/失败/缺失配对、latest run 吞吐和 ETA，以及最近 3 个样本的 contract/checksum 校验。

只读数值 QC 使用确定性的均匀抽样，不暂停 worker：

```bash
scripts/run_csl_news_annotation_qc.sh
```

报告保存在 `.../rtmw3d_l_794dbc78_v1/qc/qc_<UTC>.json`；失败返回非零，warning 保留报告但不自动停止 worker。

## 8. 次晨验收

- 服务状态、实际 GPU、下载与标注并行状态；
- completed/failed/skipped 数和最近一次失败原因；
- 随机检查至少 3 个 NPZ/sidecar，核对 shape、文本、checksum、帧数和 finite values；
- 统计速度、峰值显存、磁盘占用和预计完成时间；
- 人工确认前保持全部源、scratch、失败与输出不变。

## 9. 运行记录

- 单视频 smoke：GPU 5，125 帧，10.49 秒（含首次模型加载），峰值显存 274,832,896 B；
  native/canonical shape、finite values、文本和 artifact SHA-256 均通过。
- 正式 unit：`mmprism-csl-news-annotation.service`，选择 GPU 7，首次启动于
  `2026-08-11T14:38:12Z`；成功生成第二个样本后发现上游 CPU 线程池过量，主动停止并加入
  4-thread 限制。该停止不清理任何 artifact 或 scratch。
- 最终恢复：`2026-08-11T14:39:47Z`，GPU 7，main PID `1525289`；PyTorch/OpenCV
  报告的线程限制分别为 4/1，unit 另设 `CPUQuota=400%`。
- `2026-08-11T14:41Z` 健康检查：`active/running`、`NRestarts=0`、GPU memory 约 838 MiB，
  累计 19 个成功 NPZ。抽查最近 3 个样本的 shape、finite values、非空文本和 SHA-256 均通过。
- failure 目录中保留 2 个正式启动前的依赖诊断记录；最终恢复后未观察到新增失败。
- `2026-08-11T14:47Z` 首份修正后 status report 为 `healthy`：10 个完整 archive、
  16,476 个可用视频、101 个成功样本、当前 run 新增失败 0、缺失配对 0、抽样 3/3 通过；
  近期约 786 samples/hour、59.1 frames/s，当前已下载 archive ETA 约 20.8 小时。
- `2026-08-11T14:50Z` 启用 `mmprism-csl-news-annotation-status.timer`，每 30 分钟触发，
  独立 service 设 `CPUQuota=100%`。手工触发验收以 `0/SUCCESS` 完成；快照仍为 `healthy`，
  147 个成功样本、当前 run 新增失败 0、抽样 3/3 通过。首次自动触发为 `15:00 UTC`。
- `2026-08-11` 操作者再次明确批准 GPU 共享策略：可以与其他任务挤在同一张卡上，调度只看
  可用显存，GPU 利用率不作为 gate。当前 worker 继续使用 GPU 7，不因此重启或迁移。
- `2026-08-11T14:56Z` 首份正式数值 QC 为 `passed`：在 246 个候选产物中确定性抽检
  100 个、共 24,628 帧，100/100 通过且无 warning；校验和、shape、finite、连续帧号、
  reported frame count 和 FPS 契约均通过。canonical valid ratio 为 0.99245，transformed 2D
  in-bounds ratio 为 0.98769；报告为 `qc/qc_20260811T145656Z.json`。
- `2026-08-11T15:00:03Z` 首次 timer 自动触发并以 `0/SUCCESS` 完成；报告
  `reports/status_20260811T150003Z.json` 为 `healthy`：11 个完整 archive、18,095 个可用视频、
  291 个成功样本、当前 run 新增失败 0、缺失配对 0、抽样 3/3 通过。worker 保持
  `active/running`、`NRestarts=0`，timer 下一次触发为 `15:30 UTC`。
