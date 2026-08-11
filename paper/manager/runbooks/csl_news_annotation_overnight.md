# CSL-News 夜间 RTMW3D 标注 Runbook

Status: `approved_for_execution`
Last Updated: `2026-08-11`
Role: `unattended_pose_annotation_operations`

## 1. 今晚目标与授权

- 对已经完整下载并原子命名为 `archive_*.zip` 的 CSL-News 视频持续生成 RTMW3D-L 姿态。
- 使用 4 个 registry-driven 轮询 worker；每个 worker 只读取 cumulative integrity registry 中
  `status=passed` 的 archive，并按 `archive_id % 4` 分片。GPU 7 总 worker 数上限为 4。
- 项目负责人已明确批准与其他任务共享 GPU。选卡和运行期间只以可用显存为资源门槛，GPU 利用率不是启动、暂停、迁移或退出条件。
- 授权边界：可以在已有计算任务的卡上同时运行本 worker；只要满足最低可用显存即可，已有任务的 GPU 利用率不构成冲突或迁移理由。不得为了本任务结束、暂停或修改其他用户进程。
- 默认启动门槛为 2,048 MiB free memory；该值高于当前单 worker 约 838 MiB 的稳定占用并保留加载余量，可通过 `--min-free-mib` 按模型实测结果调整。30 GiB 不再作为固定门槛。
- worker 将 OpenMP/BLAS/PyTorch 限制为 4 个 CPU 线程，并将 OpenCV 限制为 1 个线程。
- 若指定卡达不到最低空闲显存，worker 退出并交给 systemd 稍后重试；自动选卡时在满足门槛的卡中选择空闲显存最多者。

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
integrity config: configs/data/csl_news_source_integrity.yaml
integrity registry: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v2/registry.json
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
registry worker 额外在每个成功/失败 sidecar 和 archive marker 中保存实际消费时的 registry SHA-256、
archive SHA-256、audit path/hash、audit builder commit、audited time 和 labels SHA-256。

已提取视频保留在：

```text
/mnt/gfs/yanyifan/mmPRISM/cache/csl_news_annotation/rtmw3d_l_794dbc78_v1/videos/
```

## 5. 执行顺序

1. `uv sync --extra annotation` 安装锁定环境。
2. 对每个已完成 archive 做完整 SHA-256/逐 member CRC/label source audit；未通过者不进入 worker。
3. 在满足配置启动门槛的卡上执行单视频 smoke；允许该卡同时有高利用率任务，默认门槛为 2,048 MiB free memory。
4. 检查输出 shape、有限值、中文文本、sidecar checksum、峰值显存和每帧速度。
5. smoke 通过后，启动 4 个 `systemd --user` registry worker；CLI override 只改变 orchestration
   shard，不改变 artifact config fingerprint。
6. integrity timer 每 5 分钟扫描新 final ZIP；审计通过后由 worker 最迟在下一次 60 秒轮询消费。
7. 已验证输出自动跳过；逐视频普通失败写 sidecar 后继续。registry/source stat 变化时停止而非继续消费。

## 6. 停止条件

- `/mnt/gfs` 可用空间低于 1 TiB；
- labels 无效、缺失或重复；
- ZIP 损坏、路径不安全或同 archive 视频 basename 冲突；
- MMPose commit、配置或 checkpoint hash 不匹配；
- 连续 2 个 CUDA OOM；
- 可用显存低于启动门槛时不加载模型，由 systemd 重试。

GPU 利用率高不属于停止条件，这是本次夜间运行的显式授权。

所有 worker 只允许使用 cumulative source integrity registry 中的 `passed_archive_ids`。发现损坏时
停止对应 archive worker，但不停止已经通过 source gate 的其他 worker；不得用 partial manifest
或 central-directory CRC 字段替代完整读取验证。

## 7. 启动与观察

单视频 smoke：

```bash
scripts/run_csl_news_annotation_worker.sh --gpu auto -- \
  --archive-id 3 --max-videos 1 --once
```

正式任务由管理员用 `systemd-run --user` 托管，设置 `Restart=on-failure` 和 300 秒重试；
4 个 worker 分别传入 `--worker-index 0..3 --worker-count 4 --integrity-registry <path>`。
启动后记录实际物理 GPU、registry SHA-256 和 shard 到 run metadata。观察命令：

```bash
systemctl --user status 'mmprism-csl-news-annotation-registry*.service'
journalctl --user -f \
  -u mmprism-csl-news-annotation-registry0.service \
  -u mmprism-csl-news-annotation-registry1.service \
  -u mmprism-csl-news-annotation-registry2.service \
  -u mmprism-csl-news-annotation-registry3.service
systemctl --user status mmprism-csl-news-integrity-scan.timer
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

冻结当前全部 published sidecar/NPZ pair 的 CPU-only identity audit：

```bash
scripts/run_csl_news_annotation_audit.sh
```

报告保存在 `.../rtmw3d_l_794dbc78_v1/identity_audits/audit_<UTC>.json`。命令只流式读取字节并校验
声明/实际 size、SHA-256 和 hash 前后 stat，不加载 pose arrays。存在任何 mismatch 时返回非零，但仍写出
包含全部异常的报告；不得据此自动删除、覆盖或重算原 pair。正式 quarantine 只能使用 clean Git 报告，
并由 `configs/data/csl_news_pose_manifest_available.yaml` 中 checksum-bound exclusion 精确绑定。

## 8. 次晨验收

- 服务状态、实际 GPU、下载与标注并行状态；
- completed/failed/skipped 数和最近一次失败原因；
- 随机检查至少 3 个 NPZ/sidecar，核对 shape、文本、checksum、帧数和 finite values；
- 运行一次 clean-Git 全量 identity audit，确认异常集合未扩张；
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
- `2026-08-11` 项目负责人再次明确批准 GPU 共享策略：可以与其他任务挤在同一张卡上，调度只看
  可用显存，GPU 利用率不作为 gate。worker 默认启动门槛据实测占用由 30,000 MiB 调整为
  2,048 MiB；当前 worker 继续使用 GPU 7，不因此重启或迁移。
- `2026-08-11T14:56Z` 首份正式数值 QC 为 `passed`：在 246 个候选产物中确定性抽检
  100 个、共 24,628 帧，100/100 通过且无 warning；校验和、shape、finite、连续帧号、
  reported frame count 和 FPS 契约均通过。canonical valid ratio 为 0.99245，transformed 2D
  in-bounds ratio 为 0.98769；报告为 `qc/qc_20260811T145656Z.json`。
- `2026-08-11T15:00:03Z` 首次 timer 自动触发并以 `0/SUCCESS` 完成；报告
  `reports/status_20260811T150003Z.json` 为 `healthy`：11 个完整 archive、18,095 个可用视频、
  291 个成功样本、当前 run 新增失败 0、缺失配对 0、抽样 3/3 通过。worker 保持
  `active/running`、`NRestarts=0`，timer 下一次触发为 `15:30 UTC`。
- `2026-08-11T15:25Z` clean-commit source snapshot 冻结 11 个 archive/18,095 条 record；
  对当时 676 个 `archive_003` pose sidecar 逐一检查，stable sample ID 缺失 0、caption mismatch 0。
  snapshot 为 partial，且未执行完整 CRC，后续降级为 contract/linkage evidence。
- `2026-08-11T15:35Z` 同 GPU 7 archive-specific canary 暂扩至总计 4 workers，单进程约 828 MiB；
  `archive_005` 出现真实解压失败后，立即停止 `004/005/006` canary，不清理任何输出。
- `2026-08-11T15:44Z` 对 frozen 11 archives 的 clean-commit 完整 CRC audit 完成：9 个通过、
  `005/008` 损坏。总表 SHA-256 为 `ea8062f546cdf10abdde5b5b27e0e78e5e39e3df538e0d68b983e6ac4b7c9a00`。
  仅保留 `archive_003` 主 worker；恢复并行池前必须读取 `passed_archive_ids`，原件和 partial/failure
  artifacts 保留到次晨人工检查。
- `2026-08-11T15:49Z` 将 unrestricted main worker 替换为 4 条 CRC-valid fixed lane：
  `003,010,011`；`004,014`；`006,015`；`009,020`。最终 unit 名为
  `mmprism-csl-news-annotation-lane{0..3}-v2.service`，均在 GPU 7、`CPUQuota=400%`、
  `Restart=on-failure`；模型加载后 4/4 `active/running`、`NRestarts=0` 并持续产出。
- lane 启动时 GPU 7 已有约 8.9 GiB 其他任务；4 个标注 worker 共卡后总显存约 12.3 GiB，
  高利用率不触发迁移，符合本次显式授权。
- 首次无 `-v2` lane unit 因 systemd shell quoting 将 archive ID 展开为空，在模型加载和样本处理前
  以参数错误退出；这些 unit 已停止，未生成或修改 artifact。
- `2026-08-11T15:51Z` 状态扫描发现新 final `archive_001` central directory 不可读。日志确认旧下载
  脚本在 aria2 93%/HTTP 403 后误晋升 incomplete `.part`。下载已短暂停止并以修复版恢复：transfer
  非零、残留 `.aria2` 或完整 `unzip -t` 失败均禁止 promotion；`001` 原件保持不变且未进入 lane。
- `2026-08-11T16:00Z` timer report `status_20260811T160003Z.json` 为 `attention_required`，唯一
  source error 是已隔离 `001`；artifact/sidecar 缺失均为 0、latest run 新失败 0、最近 3/3 样本
  校验通过。4-lane aggregate 近期约 1,394 samples/hour。
- 修复版 downloader 首个晋升的 `archive_002` 随后通过 1,624-video canonical audit，report
  SHA-256 为 `3f2eaffd97c1f48481d92f7f88f5bd8ce68d78cce3bc74f0acbb9d8e0c43c4e9`；为保持 frozen lane
  边界，`002` 不动态插入当前进程，进入下一轮调度。
- `2026-08-11T16:16Z`，clean commit `b182512` 首次生成 cumulative integrity registry：首轮
  审计 14 个 final ZIP，`001/005/008` 失败，其余 11 个通过。紧接着的增量扫描复用 14 个结果并
  自动审计新晋升的 `017`，当前白名单 12 个 archive/19,760 videos，registry SHA-256 为
  `070bcc4446894577cab6e05f632049a2a53143b508e50523dd27c20daea52b66`。
- `2026-08-11T16:19Z`，4 个 fixed lane 完全停止后才启动
  `mmprism-csl-news-annotation-registry{0..3}.service`；4/4 `active/running`、`NRestarts=0`，
  GPU 7 尚余约 52 GiB。run metadata 记录 registry hash 和 shard；无新 annotation failure。
- `mmprism-csl-news-integrity-scan.timer` 每 5 分钟运行一次，使用 `flock` 防止重叠并原子更新 registry。
  `16:20Z` 白名单状态报告统计 1,687 个 eligible NPZ，另有 15 个来自损坏 archive 的历史 NPZ/
  sidecar 被标为 ineligible、未计入进度；抽检 3/3 通过，当前 run 新失败 0。
- `16:24Z` 与 `16:29Z` timer 自动审计通过新晋升的 `007/013`；`16:30Z` registry 为 14 个 passed
  archive/23,020 videos。提交 `7847f4f` 后平滑替换为 `registry{0..3}-v2`，重启前 `007` 尚无产物；
  v2 的首个新 sidecar 已核对包含 registry snapshot、archive/audit/labels hash 和 builder commit。
- `16:42Z`，clean commit `390093b` 首次生成 integrity-gated pose+caption partial snapshot：冻结
  2,157 条 record/5 个 represented archive，15 个 failed-archive 历史 pair 保留但排除，eligible NPZ
  未配对为 0。manifest SHA-256 为 `4161593fdbfc85a5c2fb392e3ef92d40da560db5c75a19d559f1f92878e31600`；
  `SHA256SUMS`、portable path scan 和首/中/末 adapter 读取均通过。后台 4 worker 未暂停，后续
  新 artifact 只进入新 snapshot。
- `17:09Z` integrity timer 因主仓库正在编辑、Git state 非 clean 而按设计以 exit 2 拒绝写 registry；
  `8b64d0f` 提交后，`17:14Z` 下一周期自动以 `0/SUCCESS` 恢复并完整审计通过 `archive_026`。
  当前 registry 为 18 final、15 passed/24,618 videos，失败项仍仅 `001/005/008`，SHA-256 为
  `b150b679877568a092f1dcb61b0c9a35648434e339ce7603ff841dde29ae0ce1`。
- `17:15Z` 状态报告：2,916 eligible pair、missing artifact/sidecar 0、当前 run 新失败 0、抽检 3/3，
  近期约 1,409 samples/hour、白名单 ETA 约 15.4 小时。同期 100-sample/24,601-frame QC 为
  `passed`，warning/failure 0；4 个 registry worker 均 `active/running`、`NRestarts=0`。状态仍为
  `attention_required`，原因是 registry 持续保留 3 个 failed source，不是 annotation failure。
- `17:30Z`，clean commit `1fc0d55` 后手工触发 integrity oneshot 并以 `0/SUCCESS` 完成；新 final
  `archive_027/030` 均通过完整 CRC/路径安全/label coverage。registry 更新为 20 final、17 passed/
  27,975 videos。`17:31Z` 状态报告为 3,287 eligible pair、missing artifact/sidecar 0、latest run
  新失败 0、抽检 3/3，近期约 1,436 samples/hour；四个 worker 均 `active/running`、`NRestarts=0`。
- `17:52Z`，clean commit `8c27fb9` 后手工触发 integrity oneshot 并以 `0/SUCCESS` 完成；
  `archive_032/034` 通过完整 CRC/路径安全/label coverage，registry 更新为 22 final、19 passed/
  31,423 videos。`17:53Z` 状态报告为 3,795 eligible pair、missing artifact/sidecar 0、latest run
  新失败 0、抽检 3/3，近期约 1,382 samples/hour；四个 worker、下载服务和 timer 均为 active。
- `19:18Z`，clean commit `84f2c52` 后 integrity scan 从一次预期的 dirty-Git 拒绝中恢复，并以
  `0/SUCCESS` 完成；registry 更新为 31 final、28 passed/46,521 videos，失败仍仅 `001/005/008`。
  `19:17Z` 状态报告已有 5,710 eligible pair、missing artifact/sidecar 0、latest run 新失败 0、
  抽检 3/3，近期约 1,484 samples/hour；四个 registry worker 均 active 且 `NRestarts=0`。
- `19:23Z` integrity timer 在 clean commit `10a30e5` 下审计通过 `archive_052` 的 1,689 个视频；
  registry 更新为 32 final、29 passed/48,210 videos。`19:30Z` 自动报告已有 6,017 eligible pair、
  missing artifact/sidecar 0、latest run 新失败 0、抽检 3/3，近期约 1,488 samples/hour；四个 worker
  继续 `active/running`、`NRestarts=0`，未清理任何 source、scratch、failure 或 pose artifact。
- `21:00Z` 后一次 pose-manifest snapshot 在 `archive_006/3af7db9841fb2ac483721620` 按 checksum
  gate 失败：sidecar 错误声明 0 bytes/空文件 SHA，实际 NPZ 为 813,674 bytes、SHA-256
  `6914b6bb0f26304d87b14d7cd7e8b00ac13e6d65202a97c0d4a89e3b0d38bca3`。失败 snapshot temp、
  原 pair 和 failure records 全部保留；worker publisher 已加 fsync、promotion 后 identity 重验和
  preserve-on-conflict 后继续，四个 worker 均以 `NRestarts=0` 恢复持续产出。
- `21:23Z` clean commit `3bdd31f` 执行 CPU-only 全量 identity audit：冻结 9,519 个 published pair、
  流式哈希 5,115,703,846 bytes，9,518 通过且唯一异常仍为上述 pair；hash 前后 stat 稳定，未发现
  第二个异常。报告 `identity_audits/audit_20260811T212324Z.json` SHA-256 为
  `55478cbb6078d7e4c7b0c9a95577e6260e249239514ec584d082d5b0b4c538b4`，Git clean，
  `audit_failures` 为空。
- `21:24Z` registry 为 51 final、48 passed/79,813 videos，失败仍仅 `001/005/008`。clean commit
  `98549a9` 的新 pose snapshot 冻结 9,552 eligible sidecar，通过 checksum-bound exclusion 精确隔离
  一个 preserved conflict，写入 9,551 records/9 archives、0 unpaired NPZ；manifest SHA-256 为
  `8e3db8712bc61848e9d6dea9f5b3a3821365ffd102d6643977ad43107b2db0c4`，四项 `SHA256SUMS`、
  contract 和首/中/末 adapter 读取全部通过。后台下载和四个 GPU 7 worker 未暂停。
- `21:32Z–21:34Z` 在 clean HEAD `e821044` 上对 registry worker 0/1/2/3 逐个滚动重启，统一替换
  `85d1143`/`c21b818` 的混合运行 provenance；每次只重启一个本项目 unit，不触碰下载、integrity timer
  或其他 GPU 进程。最终四个 worker 均 `active/running`、`NRestarts=0`，run metadata 的 Git commit/
  dirty state 全部为 `e821044`/`false`，四个 shard 均重新加载模型并产出新样本。`21:34Z` 当前 NPZ/
  sidecar 为 9,907/9,907。
- `22:05Z–22:10Z` versioned recovery 完成并切换到 source-integrity v2。replacement `001/005/008`
  全部通过 SHA-256、逐 member CRC、label coverage 和单视频 decode；primary 坏 ZIP 未移动或覆盖。
  v2 registry SHA-256 为 `ae6b2909e7b12c3f9519ffc493b67a556621d6e7203665b940ea4bee9878a02c`，
  59/59 present archive passed、97,997 videos、failed 0。
- 四个 `registry{0..3}-v3` worker 在 GPU 7 滚动启动，实际 process argv 和 run metadata 均确认
  `worker_index=0..3/worker_count=4`、registry v2、Git `0f2e635` clean；未停止下载、timer 或其他
  GPU 进程。source identity 不匹配时写 `--source_<archive-sha256>` 变体，不覆盖旧产物。
- `22:13Z` 手工 source-aware 状态报告为 `healthy`：9,394 个 current-source pair、1,875 个
  old/unbound quarantine pair、duplicate-current-source 0、missing pair 0、latest run 新失败 0、
  抽检 3/3 通过；四个 worker `active/running`、`NRestarts=0`。
- `22:27Z–22:28Z` 在 clean commit `11014a8` 上执行全量 identity audit：冻结 11,815 pair、
  哈希 6,373,342,155 bytes，11,814 通过，唯一失败仍为既有 `archive_006/3af7...`；报告 SHA-256
  `23278c988156ce27e52405794642f7e77ab0ec44d93c43be93da1626d5864105`，无新增 conflict。
- `22:29Z` 同一 clean commit 冻结 v2-bound partial snapshot `snapshot_20260811T222941.214512Z`：
  10,011 records/12 archives，1,875 个旧来源/unbound sidecar 写入 checksum-covered quarantine，
  当前来源 unpaired NPZ 0。五项 `SHA256SUMS`、通用 contract 和首/中/末 checksum adapter 读取
  全部通过；manifest SHA-256 为 `3412aeb2f7fea685796e17d85b3af6342b7ffe1b3a61895446295f5f71e073f7`。
- `22:30Z` 自动 status 为 `attention_required`，原因仅为既有 `archive_006/3af7...` 在当前 run
  再次触发 preserve-on-conflict。current-source duplicate/missing pair 均为 0、抽检 3/3 通过；timer
  保留 exit 1 告警，但四个 worker 不停止、不迁移。
- `22:32Z` clean integrity timer 继续审计通过 `091/094/096`，live v2 registry 为 62/62 passed、
  102,949 videos、failed 0，SHA-256
  `b461c9efd619ca2a049f4f64c9758bf7d6c64fb603a06ea64123148d13542e1a`。既有 snapshot 仍绑定
  冻结时的 59-archive registry，不追写 live 更新。
- `22:44Z` clean commit `7f86516` 冻结首个 source-manifest v2 snapshot
  `snapshot_20260811T224413.526848Z`：复制并绑定 63-archive registry exact bytes，生成 104,658 条
  source+caption record，manifest SHA-256 为
  `a431d14cd5f693a82d8f21c3c5c7ee05c9d27d2ee003c801db21dcfdc7434263`。三项 `SHA256SUMS`、
  通用 manifest contract、portable path scan 和首/中/末 exact ZIP/member 读取全部通过；snapshot
  仍为 partial。manifest summary 的 `crc_checked=false` 仅表示本次冻结没有重复执行全量 CRC；所选
  archive 已由复制的 v2 registry 逐一通过完整 CRC、label coverage 和 decode gate。
- `22:47Z` live v2 registry 已继续推进到 66/66 passed、109,797 videos、failed 0；四个 v3 worker
  与下载服务均为 `active/running`、`NRestarts=0`。GPU 7 利用率约 99%、可用显存约 77.8 GiB；按
  项目负责人批准的规则继续共卡运行，不因利用率重启、暂停或迁移，也不干预其他 GPU 任务。
