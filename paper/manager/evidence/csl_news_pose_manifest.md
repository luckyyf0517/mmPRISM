# CSL-News Pose Manifest Evidence

Status: `v2_partial_snapshot_evidence_ready`
Last Updated: `2026-08-11`
Role: `DATA-003-B_ARCH-003-A_pose_caption_manifest_evidence`

## 1. Snapshot Identity

```text
source: huggingface:ZechengLi19/CSL-News
revision: 3a0601210333fe760efd09b5d9e2ae5f341ce339
builder commit: 390093b95c5da9d74924029ae5a8496bc1a01cb4
builder Git state: clean
annotation config fingerprint: d7525ebbf4e524589eeb9fa71198d10ee525300adb9244a7470424b211399889
snapshot config fingerprint: 3d8cbf7066aad8f2991dc03c73d058cc33d691a2bf06a65fe2fb19a769e71829
snapshot: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/pose_manifest_v1/snapshot_20260811T164204Z_partial
integrity registry SHA-256: 183743fbb60bb85b75dd63f6c112e0c1a3081b2b6a391e32fa6ce2a21cb5b02d
manifest SHA-256: 4161593fdbfc85a5c2fb392e3ef92d40da560db5c75a19d559f1f92878e31600
summary SHA-256: 52dc11ebc8bf599acc1bed6f078c830d875cb55a9b4da0bb306433ba6c909d7e
status: partial
```

复现入口：

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  scripts/run_csl_news_pose_manifest.sh
```

正式 snapshot 要求 clean Git、至少 1 TiB 可用空间，并创建新目录；不会覆盖已有 snapshot。

## 2. Frozen Boundary

扫描开始时，cumulative integrity registry 有 17 个 final ZIP，其中 14 个 `passed`，共 23,020
个视频；`001/005/008` 为 typed `failed`。snapshot 只冻结扫描开始时已完成且位于 passed archive
下的 sidecar/NPZ pair：

| Item | Result |
|---|---:|
| manifest records | 2,157 |
| represented passed archives | 5 (`002/003/004/006/009`) |
| referenced artifact bytes | 1,169,173,125 |
| excluded failed-archive sidecar/NPZ pairs | 15 / 15 |
| unpaired eligible NPZ at scan | 0 |
| sidecars with embedded registry provenance | 297 |
| older sidecars supplemented from frozen registry | 1,860 |

较早 sidecar 尚未内嵌 per-sample integrity block，但它们的 archive、source stat、stable ID、caption、
artifact checksum 均重新对当前 frozen registry 校验，并在 manifest record 中补入 registry/audit
provenance；这不等同于篡改旧 sidecar。

## 3. Record And Adapter Contract

每条 `mmprism.sample.v1` 记录包含：

- inline canonical JSON caption；
- native `[T,133,3]` keypoints、scores 和 transformed 2D keypoints；
- frame indices、timestamps；
- canonical pose `[T,2,24,3]`、confidence 和 valid mask；
- source archive/member/audit、labels、annotation model/transform/sidecar 和 clean builder commit。

所有数组使用相对 URI 指向同一个 NPZ，并共享 artifact SHA-256。`CslNewsPoseManifest` 在初始化时
校验 record contract，在读取时限制路径不得逃逸 artifact root，并可重新校验 checksum、shape 和 dtype。
该 adapter 不依赖 PyTorch、Lightning 或 Transformers。

## 4. Independent Verification

| Gate | Result |
|---|---|
| `SHA256SUMS` | registry/manifest/summary 3/3 `OK` |
| JSONL line count | 2,157 |
| general manifest contract | passed；9 modalities |
| local absolute path scan in manifest | 0 matches |
| first/middle/last adapter load | 3/3 passed with checksum verification |
| sampled caption/shape/dtype | matched sidecar and NPZ |
| source mutation | none; build is read-only over ZIP/pose source |

完整代码门同时通过 51 tests、Ruff、strict Mypy、shell syntax、sdist 和 wheel build。

## 5. Evidence Boundary

- 本 snapshot 只证明 canonical pose+caption manifest、provenance gate 和读取 adapter 可用。
- 它不代表 436 个 archive 下载/标注完成，也不能作为论文最终数据规模或最终 split 的依据。
- `001/005/008` 和其历史派生产物仍保留；本 snapshot 未包含这些记录。
- 后台 worker 在 snapshot 之后继续产生 artifact；它们只会进入后续新 snapshot。
- `DATA-003-B`、`DATA-005-A` 和 `ARCH-003-A` 保持 `in_progress`，直到全量 source、pose build、
  final manifest、split 和 leakage audit 完成。

## 6. Clean Identity Audit And Second Snapshot

`2026-08-11T21:23:25Z`，clean commit `3bdd31f6b0b9f43c8c3458df79a653346eda8c4e`
运行 CPU-only `csl-news-annotation-audit`。命令在开始时冻结全部非隐藏 published sidecar，逐对流式
校验 JSON/schema/sample/config identity、NPZ 存在性、声明 size/SHA-256、实际 SHA-256，以及 sidecar
和 artifact 的 hash 前后 stat 稳定性；不会加载姿态数组。

```text
report: interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/identity_audits/audit_20260811T212324Z.json
report SHA-256: 55478cbb6078d7e4c7b0c9a95577e6260e249239514ec584d082d5b0b4c538b4
frozen sidecar list SHA-256: 1d5f9704a7d597c015b0e45b19a8c4e1eaf5669ba38a5a281efd5ccc00ddeb8e
audited pairs: 9,519
passed pairs: 9,518
failed pairs: 1
artifact bytes hashed: 5,115,703,846
audit failures: none
```

唯一异常为 `archive_006/3af7db9841fb2ac483721620`：sidecar 声明 0 bytes/空文件 SHA-256，
实际 NPZ 为 813,674 bytes，SHA-256
`6914b6bb0f26304d87b14d7cd7e8b00ac13e6d65202a97c0d4a89e3b0d38bca3`；sidecar SHA-256 为
`d7791e9633a48a40f587e5c6b6281cfff6623d68ed09f785334de2166dc18142`。两文件在审计前后
device/inode/size/mtime/ctime 完全一致，未发现第二个异常。该 pair、failure records 和失败 snapshot
临时目录均未删除、移动或覆盖。

`DEC-029` 固定处理方式为 checksum-bound explicit exclusion。manifest builder 不放宽普通 checksum
gate；只有同时匹配 sample/archive、sidecar SHA、声明/实际 NPZ identity、clean-run audit report SHA
与 clean Git provenance 的条目才能排除，任一漂移硬失败。audit report 会复制进 snapshot 并进入
`SHA256SUMS`。

随后 clean commit `98549a92b7ca22adbcbed6a241d139f07ed64ec0` 生成第二个 partial snapshot：

```text
snapshot: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/pose_manifest_v1/snapshot_20260811T212450.135852Z
snapshot config fingerprint: 8344e14ef984580c0f1b5bff2eacce3adb386b4d8de0ae5e0f53775320e83c4c
integrity registry SHA-256: c94763b4cd2e77a670be7de9f66f44a716a4adba2af86e3fdc99354984e205b4
manifest SHA-256: 8e3db8712bc61848e9d6dea9f5b3a3821365ffd102d6643977ad43107b2db0c4
summary SHA-256: 99f7c3ec92cb4255c2600766b7014f7ad642e19444d41e0e6ad943ef528843ae
SHA256SUMS SHA-256: ae1914651f0b10d39342e0c58959cf27356e41aba373be3c901e4962cb3c124a
records: 9,551
represented archives: 9
frozen eligible sidecars: 9,552
explicit exclusions: 1
unpaired eligible NPZ: 0
referenced artifact bytes: 5,135,316,654
status: partial
```

独立验收结果：registry/manifest/summary/copied audit evidence 的四项 `SHA256SUMS` 全部 `OK`；
通用 manifest contract 为 9,551 records/9 modalities；排除 sample ID 在 JSONL 中零命中；首/中/末
adapter 以 `verify_checksum=True` 读取 3/3 通过。该 snapshot 仍不代表 436 archive 全量完成，也不替代
基于 signer/subject 的最终 split 或任何论文结果。

## 8. Current v2 Source-Bound State

source-integrity v2 将 `001/005/008` 切换到 versioned replacement 后，旧 sidecar 不能因 archive ID
相同而被静默复用。annotation resume 现在要求 archive SHA-256、labels SHA-256、member size 和 CRC
全部匹配；不匹配或缺少 identity 时，重算产物使用：

```text
<sample-id>--source_<full-archive-sha256>.npz
<sample-id>--source_<full-archive-sha256>.json
```

原产物不删除、不覆盖。pose-manifest builder 对每个 sample 只接受一个与当前 v2 entry 完全匹配的
sidecar；旧来源、unbound 或 superseded candidate 写入 `source_identity_quarantine.jsonl`，并由
snapshot `SHA256SUMS` 覆盖。

`2026-08-11T22:13Z` 的只读 source-aware 状态报告绑定 registry SHA-256
`ae6b2909e7b12c3f9519ffc493b67a556621d6e7203665b940ea4bee9878a02c`：当前来源 pair 9,394，
旧来源/unbound 隔离 pair 1,875，当前来源重复 0，missing pair 0，抽检 3/3 通过，状态 `healthy`。
此前 v1-bound snapshot 保留为 pipeline/incident evidence，不作为 current-source training manifest。
后续 clean full identity audit 与 v2-bound partial snapshot 的验收结果记录如下。

## 9. v2-Bound Partial Snapshot

clean commit `11014a82627726758e3f6f24b82455e976c61c2b` 的全量 identity audit 冻结 11,815 个
published pair，流式哈希 6,373,342,155 bytes；11,814 通过，唯一失败仍是已登记的
`archive_006/3af7db9841fb2ac483721620`，没有新冲突。报告为
`identity_audits/audit_20260811T222742Z.json`，SHA-256
`23278c988156ce27e52405794642f7e77ab0ec44d93c43be93da1626d5864105`，Git clean，
`audit_failures` 为空。

同一 clean commit 随后冻结 current-source snapshot：

```text
snapshot: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/pose_manifest_v1/snapshot_20260811T222941.214512Z
snapshot config fingerprint: dc33e47ef79468fa0b9158ce92ca73771056740a760cef3a6f03a1934ed069fc
integrity registry SHA-256: ae6b2909e7b12c3f9519ffc493b67a556621d6e7203665b940ea4bee9878a02c
manifest SHA-256: 3412aeb2f7fea685796e17d85b3af6342b7ffe1b3a61895446295f5f71e073f7
summary SHA-256: 35502576b1bf4cc7d530c6d7f4c0a44d38a3d881ee13087f226cd91d5297305e
source identity quarantine SHA-256: 1b03721b4fc64601d8dff0fc247e6d7a1a319ac93d2dc25c6cc463f0cd659586
SHA256SUMS SHA-256: 0c450ea95c596a1e2abf3076d8834876da032181b1291876870ac0ad75e7f611
records: 10,011
represented archives: 12
frozen current-source sidecars: 10,012
explicit exclusions: 1
source-identity quarantine entries: 1,875
unpaired current-source NPZ: 0
referenced artifact bytes: 5,389,225,123
status: partial
```

五项 `SHA256SUMS` 全部 `OK`；通用 manifest contract 读取 10,011 records/9 modalities；首/中/末
adapter 在 `verify_checksum=True` 下 3/3 通过。quarantine ledger 的 1,875 条记录覆盖
`002/003/004/005/006/009`，原因均为旧产物缺少或不匹配当前 source content identity；它们未计入
manifest。manifest 中 `005/008` 的抽样记录分别绑定 replacement SHA-256 `3450d136...`/
`b258e4be...` 和 v2 精确相对路径。`001` 在冻结时尚无 current-source 完成产物，因此未出现在该
partial snapshot；后续 worker 产物进入新的 snapshot，不修改本 snapshot。
