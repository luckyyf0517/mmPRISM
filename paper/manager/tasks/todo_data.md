# Data Rebuild Todo

Status: `upload_preflight_in_progress`
Last Updated: `2026-08-11`
Role: `data_execution_tracker`

| ID | Priority | Task | Status | Acceptance |
|---|---|---|---|---|
| `DATA-001-A` | P0 | 确认历史数据实际位置和访问方式 | blocked | 每个来源有 owner/location/access |
| `DATA-001-B` | P0 | 只读统计容量、文件数、后缀和目录层次 | blocked | inventory report，不生成副本 |
| `DATA-001-C` | P0 | 定位 checkpoint、log、prediction、metric artifacts | blocked | historical artifact inventory |
| `DATA-001-D` | P0 | 评估 `/mnt/gfs` 容量和迁移策略 | in_progress | 峰值空间与清理/归档方案 |
| `DATA-001-E` | P0 | 定义重新上传范围、优先级和 intake gate | done | `data_upload_checklist.md` 覆盖 P0/P1/P2、下载和重建边界 |
| `DATA-001-F` | P0 | 收集待上传 archive/目录大小及可重下标记 | blocked | source-side preflight inventory 可计算上传/解压峰值 |
| `DATA-001-G` | P0 | 上传并验证 metadata/radar config/calibration | blocked | checksum、字段字典、config mapping 和 access scope 通过 |
| `DATA-001-H` | P0 | 分批上传私人真实采集 raw package | blocked | incoming batch checksum 完整且原始包只读 |
| `DATA-001-I` | P0 | 固定 CSL-Daily/CSL-News 重新下载或上传路径 | in_progress | CSL-News 已确认可重下；待补齐各数据集 version/URL/license/checksum 或 incoming batch |
| `DATA-001-J` | P0 | 恢复原投稿 MANO/mesh/skeleton simulation provenance | blocked | 实际 simulator、输入、配置和历史证据一致 |
| `DATA-001-K` | P0 | 下载并验证 CSL-News 官方 RGB/labels | in_progress | versioned replacement `001/005/008` 已通过完整 gate；v2 registry 当前 59 archives/97,997 videos passed，待其余 377 archives + labels 全量验证 |
| `DATA-002-A` | P1 | 定义 sample/sequence/acquisition/provenance schema | in_progress | schema v1 reviewed against real source |
| `DATA-002-B` | P1 | 定义 pose joint/坐标系/单位规范 | in_progress | metric `[left/right,24,x/y/z]` contract 已冻结；待各数据族单位/坐标 mapping，RTMW3D 当前仅 shape/order 已证 |
| `DATA-002-C` | P1 | 定义 raw radar complex representation 与 radar config version | in_progress | complex `[chirp,antenna,sample]` contract 与 range-Doppler v1 已通过；待真实 reader/config fixture |
| `DATA-003-A` | P1 | CSL-Daily source adapter 和 manifest | not_started | coverage/shape/annotation report |
| `DATA-003-B` | P1 | CSL-News source adapter 和 manifest | in_progress | v2 registry/atomic audit/exact-path worker whitelist 已验证；59 archive/97,997 videos passed；待 v2-bound pose snapshot 和最终 436-archive CRC/coverage/decode report |
| `DATA-003-C` | P1 | collected source adapter 和 manifest | not_started | subject/scene/action metadata report |
| `DATA-003-D` | P1 | damaged/missing/ambiguous asset quarantine | in_progress | primary `001/005/008` 和 1,875 个 unbound/旧来源 pair 保持隔离；replacement 已通过；唯一 sidecar/NPZ identity conflict 由 clean audit + checksum-bound exclusion 隔离且原件保留；待最终全量 audit |
| `DATA-004-A` | P1 | 建立 subject/signer/sequence group split | in_progress | 2,157-record CSL-News partial sequence split 已通过 deterministic coverage/leakage audit；待 full manifest、signer/subject metadata 和最终 split |
| `DATA-004-B` | P1 | 识别原投稿 split | not_started | paper split hash/provenance |
| `DATA-005-A` | P2 | 重建 pose annotation pipeline | in_progress | durable publication、source-bound resume、source-versioned coexistence、4-worker v2 registry shard 和 source-aware status 已通过；9,394 个当前来源 pair 健康，夜间全量 build 与新 frozen manifest 待完成 |
| `DATA-005-B` | P2 | 重建 radar processing/simulation pipeline | in_progress | NumPy range-Doppler v1/analytic tests 已通过；beamforming/simulation 等 acquisition、array、calibration evidence |
| `DATA-005-C` | P2 | 重建 pred_pose/feature pipeline | not_started | checkpoint-bound provenance |
| `DATA-006-A` | P2 | 生成 model-ready processed dataset | in_progress | strict dependency-light radar-cube/metric-pose manifest adapter、checksum 和 variable-time collator 已通过；待真实 collected cube/pose build、validation report 与 frozen manifest hash |
| `DATA-REV-001` | P0 | 统计 sign type/vocab/sentences/length/non-manual/subjects/scenes/splits | in_progress | CSL-News 722,711 条 metadata profile 已生成；sign vocab/non-manual/subjects/scenes/splits 仍缺，待 manuscript-ready table + frozen manifest summary |
| `DATA-REV-002` | P0 | 方向/双手重叠/物体遮挡/新用户真实数据 protocol 与采集 | blocked | ethics-cleared held-out manifest and QC |
| `DATA-REV-003` | P0 | paired/category-matched synthetic-real evaluation set | blocked | same-sign fidelity manifest |

## 当前上传 Gate

1. GFS 在 `2026-08-11T12:10Z` 约余 3.6 TB，但为共享动态容量；bulk job 必须持续保留至少 1 TiB。
2. 先传匿名 metadata、radar config/calibration，再传私人 raw captures。
3. 每批进入 `incoming/<batch-id>`，完成 checksum 和只读 inventory 后才登记为 source。
4. 公共模型和可重新生成的 pose/signal/feature/cache 不占用首批上传预算。
5. 完整操作清单：`../data_upload_checklist.md`。

当前 public download 由 `mmprism-csl-news-metadata.service` 和
`mmprism-csl-news-archives.service` 托管；完成前不得解压。

`archive_003` 的只读 SHA-256/CRC/label/decode smoke 已通过；原定
`mmprism-csl-news-source-trial.timer` 保留为 `2026-08-12 08:00 Asia/Shanghai` 独立复核。
RTMW3D 单视频 smoke 也已通过，但不代表 436 个 archive 已完成。

11-archive 完整 CRC audit 中 9 个通过，`005/008` 损坏。总表位于
`manifests/csl_news/source_integrity_v1/audit_20260811T154138Z/summary.json`，SHA-256 为
`ea8062f546cdf10abdde5b5b27e0e78e5e39e3df538e0d68b983e6ac4b7c9a00`。损坏原件和所有 partial
artifact 禁止清理或 promotion。

后续完成的 `archive_001` 是 aria2 HTTP 403 后由旧脚本误晋升的 incomplete final。结构化报告
SHA-256 为 `379b72f34a1f749a246891901e746defae3331dcb65fab64662686a7f260a723`；下载器已改为 transfer
exit、残留 `.aria2` 和完整 `unzip -t` 三重 gate，并恢复断点下载。
新 gate 首个晋升的 `archive_002` 已通过 1,624-video canonical audit，report SHA-256 为
`3f2eaffd97c1f48481d92f7f88f5bd8ce68d78cce3bc74f0acbb9d8e0c43c4e9`。cumulative registry
现由 5 分钟 timer 增量维护。source-integrity v2 registry 在 `22:10Z` 为 59 个 archive/
97,997 videos passed，并通过 exact relative path 选择 replacement `001/005/008`；4 个 dynamic worker
仅消费 typed passed entry，并保存 registry hash/shard provenance。

replacement batch 位于 `replacements/20260811_recovery_hf_3a060121/rgb_archives/`。三份 replacement
分别通过完整 SHA-256、逐 member CRC、label coverage 和单视频 decode，primary 坏文件未移动、未覆盖。
annotation resume 现在同时匹配 archive/labels SHA-256、member size/CRC；旧来源或缺少 identity 的产物
保持原名，新结果以 `--source_<archive-sha256>` 后缀共存。`22:13Z` source-aware status 为 `healthy`：
9,394 个当前来源 pair、1,875 个旧来源隔离 pair、当前来源重复 0、抽检 3/3 通过。

首个 clean-commit pose+caption snapshot 冻结 2,157 条 eligible record，manifest SHA-256 为
`4161593fdbfc85a5c2fb392e3ef92d40da560db5c75a19d559f1f92878e31600`；15 个 failed-archive
历史 pair 被保留并排除，首/中/末 adapter checksum/shape/dtype 读取通过。该证据不关闭全量任务。

clean commit `3bdd31f` 的 CPU-only identity audit 冻结 9,519 个 published sidecar/NPZ pair，流式哈希
5,115,703,846 bytes；9,518 通过，唯一异常为 `archive_006/3af7db9841fb2ac483721620`。该 pair
不覆盖、不删除；`98549a9` 的第二个 snapshot 通过证据 SHA 绑定排除该 pair，纳入其余 9,551 条记录，
manifest SHA-256 为 `8e3db8712bc61848e9d6dea9f5b3a3821365ffd102d6643977ad43107b2db0c4`。

该 manifest 的 sequence-disjoint partial split 为 train/validation/test 1,701/219/237；assignment
SHA-256 `133f32d58b213947edf09c7c1e1b7c3ee30b8588a9f2b7a863d6a668bce2d7d9`，coverage 和
cross-group leakage audit 通过。缺少 signer/subject 且 source 未完成，因此 `DATA-004-A` 保持进行中。

## 禁止事项

- 在找到源数据前创建大规模占位副本。
- 直接覆盖旧 pose/mmWave/features。
- 在 split 中保存新的机器绝对路径。
- 未做容量预算就展开 NPY 或生成重复缓存。
