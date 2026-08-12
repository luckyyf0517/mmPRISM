# CSL-News Source Integrity Evidence

Status: `v2_replacement_overlay_verified_download_partial`
Last Updated: `2026-08-12`
Role: `DATA-001-K_source_integrity_evidence`

## 1. Audit Identity

Frozen recovery control snapshot used by the first v2 pose snapshot:

```text
registry writer commit: 0f2e635114e4bda3b359c9b795e50d9dd4b2532c
registry: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v2/registry.json
registry SHA-256 at 2026-08-11T22:10Z: ae6b2909e7b12c3f9519ffc493b67a556621d6e7203665b940ea4bee9878a02c
labels SHA-256: 3381d80157fa75012ec2a220eb8a63c88968af2d60d5dbcb5a82bf680db8a3a5
present archives: 59
passed: 59 archives / 97,997 videos
failed: 0
selected replacements: 001,005,008
```

v2 registry 不再假设 `archive_NNN.zip` 必须来自 primary root；每项显式保存
`archive_path_relative` 和 `source_kind`，worker 必须读取该精确路径。primary 坏文件保持不变，
replacement 位于 versioned overlay；registry 仍为 436-archive 下载中的 partial snapshot。

The live registry subsequently advanced at `2026-08-11T22:47Z`:

```text
registry SHA-256: 1a50f062d332379f125a215937225a5735d91bf07e8f5fc0389cacf2a545f0a2
present/passed: 66/66 archives
videos: 109,797
failed: 0
selected replacements: 001,005,008
```

该 live 更新不改变已发布 snapshot 内复制的 registry bytes/hash；后续 archive 只进入新的 snapshot。

The clean post-commit scan at `2026-08-12T00:11Z` advanced the live registry again:

```text
registry writer commit: f0c6205845f6111087fab0071d102779a76271d2
registry SHA-256: 1f49b3e621c60b8bf9fd5ac96d49f0afdf9ba4abbae3d0773f26fa1ed989bcbf
present/passed: 73/73 archives
videos: 121,465
failed: 0
new archive: 120 / 1,636 videos
archive SHA-256: 5a0c7b151714469067d008b84463a9fbb4de28bdc7b808b189eabb12f6705e10
audit SHA-256: 2ee2bb7b9fc8095eafbd15c29cf96562c7e7ac2d8ec2b38c050df972242e526f
```

该值是持续变化的 live gate，不追写任何已冻结 source/pose manifest；最终证据仍需 436 archives
全部完成后重新冻结。

Historical v1 cumulative snapshot retained for incident provenance:

```text
registry writer commit at snapshot: 8c27fb95b221c19f00a8a4d89c7073d1f4b34f6d
registry: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1/registry.json
registry SHA-256 at 2026-08-11T17:52Z: 6ad8310cdbe934ff291a3e68d6ea151231e2b84c13c650e3cc939f8bf23b1338
labels SHA-256: 3381d80157fa75012ec2a220eb8a63c88968af2d60d5dbcb5a82bf680db8a3a5
present final archives: 22
passed: 19 archives / 31,423 videos
failed: 001,005,008
```

Registry 更新使用非阻塞 `flock`、per-archive audit 和原子替换。archive source stat 或标签 hash
变化时旧结果不可复用；标注 worker 同时核对 source ID/revision/stat，只消费 typed `passed` entry。
registry 以单次 byte snapshot 读取并同时计算 hash；成功/失败 sidecar 和 archive marker 都记录该
snapshot hash 及对应 archive audit provenance，避免长运行 worker 只绑定启动时的旧 registry。

## 2. Replacement Recovery Result

`mmprism-csl-news-recovery-20260811.service` 已以 `Result=success` 完成。三份 replacement 均通过
完整 SHA-256、逐 member CRC、路径/重复/encryption 检查、canonical label coverage 和一个确定性
视频全解码 probe：

| Archive | Primary SHA-256 | Replacement SHA-256 | Videos | Audit SHA-256 |
|---|---|---|---:|---|
| `001` | `07c9b956e9c42f9623b5bce57cc6e49de4fa2e0554c4963d99770e4b92beabdc` | `911ed805d80842867c0ecebc86c2f8ad0fbd6790269861dbdc964ebaa9bab7ec` | 1,694 | `eee22ef84c43c62f623b660985c246970b2bcabf31a30e9a02faac3398f0978a` |
| `005` | `fbc00d7148c2cc23717c21026775b3ce09a702b32201cd6a2fedce3f3ee18b6a` | `3450d136994df60739ff8bf62382b36005de81a91c911921348e88f378542dd3` | 1,632 | `0a3542633b0aac14c5b6b0bff3d559565a8dd03b10121634110e8ddfba7303de` |
| `008` | `ec596092c412e5a8530911c3be4855ecc715af208a7e4419c0b94f76756ecbe7` | `b258e4bebaf36623e65066438c3956a6f0ba8579e8df36f4a399297d5b291153` | 1,619 | `39615ae9f529f6b11c025062f4f2a5ddcee89e5daa16fd237e2c601da2c747c6` |

恢复结果关闭的是这三个 archive 的当前-source 缺口，不关闭 436-archive 下载任务。primary 文件、
v1 registry、历史 pose 和 failure sidecar 均保留；任何下游输入必须绑定 v2 entry 的精确来源 identity。

## 3. Historical v1 Evidence

Initial frozen manual audit retained for historical provenance:

```text
source: huggingface:ZechengLi19/CSL-News@3a0601210333fe760efd09b5d9e2ae5f341ce339
labels SHA-256: 3381d80157fa75012ec2a220eb8a63c88968af2d60d5dbcb5a82bf680db8a3a5
builder commit: 8e26322bacdddffbeb5201c851c237f42b19a407
builder Git state: clean
scope: frozen 11 final ZIP files available at scan start
artifact: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1/audit_20260811T154138Z
summary SHA-256: ea8062f546cdf10abdde5b5b27e0e78e5e39e3df538e0d68b983e6ac4b7c9a00
status: failed_with_corrupt_archives
```

`SHA256SUMS` 覆盖 11 个 archive audit JSON 和 `summary.json`，复核结果为 12/12 `OK`。
审计逐 member 完整读取数据并检查 ZIP 解压/CRC、member safety、重复、加密和官方标签覆盖；
本批未执行额外视频 decode probe。

### Initial Results

| Result | Archive IDs | Videos |
|---|---|---:|
| CRC/integrity passed | `003,004,006,009,010,011,014,015,020` | 14,844 |
| corrupt | `005,008` | 3,251 |
| total audited | 11 archives | 18,095 |

所有 18,095 个 member 均命中 canonical JSON 非空文本；missing label 和 empty text 均为 0。
未发现不安全路径、重复 member/video basename 或加密 member。

损坏证据：

| Archive | Archive SHA-256 | First failing member | Error |
|---|---|---|---|
| `005` | `fbc00d7148c2cc23717c21026775b3ce09a702b32201cd6a2fedce3f3ee18b6a` | `20230927_Dragon-TV__11187-11462_560428.mp4` | `zlib.error: invalid stored block lengths` |
| `008` | `ec596092c412e5a8530911c3be4855ecc715af208a7e4419c0b94f76756ecbe7` | `Dragon-TV_20230215_937-1162_701211.mp4` | `zlib.error: invalid stored block lengths` |

### Post-scope download incident: `archive_001`

`archive_001` was not part of the frozen 11-archive audit. It appeared as a final `.zip` afterward, but a
status scan could not open its central directory. A clean-commit structured audit recorded:

```text
builder commit: 1bfc33be407b6192a803c2eec45848ccbe4280b8
artifact: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1/incident_20260811T155614Z/archive_001.json
report SHA-256: 379b72f34a1f749a246891901e746defae3331dcb65fab64662686a7f260a723
archive bytes: 2100991351
archive SHA-256: 07c9b956e9c42f9623b5bce57cc6e49de4fa2e0554c4963d99770e4b92beabdc
error: BadZipFile: File is not a zip file
```

下载日志证明 aria2 在 93% 时因 HF 临时签名 URL 返回 HTTP 403，并保留了 `.aria2` 控制文件；
旧脚本的 `xargs` 子 shell 未继承 `set -e`，随后错误地将 incomplete `.part` 改名为 final。
修复后的下载器显式传播 transfer exit code、拒绝残留 `.aria2`，并在完整 `unzip -t`/CRC 通过后
才执行原子 promotion。修复版 service 已恢复运行并主动拒绝现有 `archive_001.zip`。

### Post-scope passed archive: `archive_002`

修复版 downloader 首个 promotion 的 `archive_002` 随后通过 canonical full-read CRC、SHA-256、
member safety 和 1,624/1,624 label coverage gate：

```text
builder commit: 1bfc33be407b6192a803c2eec45848ccbe4280b8
artifact: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1/incremental_20260811T160003Z/archive_002.json
report SHA-256: 3f2eaffd97c1f48481d92f7f88f5bd8ce68d78cce3bc74f0acbb9d8e0c43c4e9
archive SHA-256: a10864019a02d5abefe1045b1ce7fc3f3350562889e4b6c95cfe766981334fde
status: passed
```

该独立增量报告随后被 cumulative registry 纳入；不再维护手工 fixed lane 清单。

### Passed archives at `17:30Z`: `archive_027/030`

`2026-08-11T17:14Z` timer 在 clean commit `8b64d0f` 下自动发现并完整审计新 final
`archive_026`。逐 member CRC/full read、路径安全和 1,598/1,598 label coverage 均通过：

```text
artifact: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1/audits/archive_026/audit_20260811T171449.865453Z_812389d64074.json
report SHA-256: 5b8a4a64d85a15857e67936921c80446bfacf513c15a4d40c57c5b0429c9cd18
archive SHA-256: 812389d6407443529655182b510bf0563c159a65f238aebffc41cc7df300231a
status: passed
```

该周期前一次 timer 在主仓库有未提交稿件审计修改时按设计以 exit 2 拒绝更新；提交后无需人工重启，
下一周期以 `0/SUCCESS` 恢复。这验证了 registry 不会绑定 dirty builder state。

`2026-08-11T17:30Z` 在 clean commit `1fc0d55` 下手工触发同一 systemd oneshot，完整审计并通过
新 final `archive_027/030`。两者分别覆盖 1,577 和 1,780 个视频，完整 CRC、路径安全与 label
coverage 均通过：

```text
archive_027 audit: manifests/csl_news/source_integrity_v1/audits/archive_027/audit_20260811T173039.942634Z_782f364263aa.json
archive_027 report SHA-256: 37e598df1ee5705c969c17de89540fa3e3f18c8b97d76849e1b2672cde0dc883
archive_027 SHA-256: 782f364263aa9e81b7edadd6d70bcdf0242e0024a33edfc05999a676dae56091
archive_030 audit: manifests/csl_news/source_integrity_v1/audits/archive_030/audit_20260811T173049.343510Z_35aaffe6c97a.json
archive_030 report SHA-256: 737c7b5b0a8238390dd817030553f62debce61d20d2097f0f553105498d7c597
archive_030 SHA-256: 35aaffe6c97a602a7ebe979e27468d93d0748b58ae4da86e025a67c0ce2345b2
```

该 scan 以 `0/SUCCESS` 完成，registry 现为 20 final、17 passed/27,975 videos，失败项仍仅为
`001/005/008`。四个 GPU worker 未重启且 `NRestarts=0`。`17:31Z` 状态快照记录 3,287 个
eligible pose/sidecar pair、missing artifact/sidecar 0、latest run 新失败 0、抽检 3/3 通过，近期
约 1,436 samples/hour；报告 SHA-256 为
`b0f7b94ed04a0b4e0cf1b5a33786809fb6adf9ad944811de7f8e0a7028ef536b`。状态保持
`attention_required` 只因为 registry 保留三个 failed source，不是 annotation failure。

### Latest passed archives: `archive_032/034`

`archive_032` 在 clean commit `efb008a` 下完成 full-read audit，1,678 个视频全部通过 CRC、路径安全
和 label coverage；`archive_034` 在提交 canonical radar contract 的 clean commit `8c27fb9` 后由手工
触发的同一 oneshot 审计，1,770 个视频全部通过：

```text
archive_032 SHA-256: fd6a5b451e09e24e3c2bde316d5cdc0fdb224e7d4bcd9634e646ee2397729f28
archive_032 audit SHA-256: a37d9d224729f5a2cffb1fb4bea20f1630c8068bb04bd021d9d1a3f9e0d44c2e
archive_034 SHA-256: 316d5b6e8e0c4d902c6baaef730ca99438ee6c9adf3aa27d37bfae1a11aee33e
archive_034 audit SHA-256: e2d79f81bb93d314333ae73fd83d5239203267fb64d1cd23820a2c36bb9f1449
```

`17:52Z` scan 以 `0/SUCCESS` 完成，registry 为 22 final、19 passed/31,423 videos，SHA-256 为
`6ad8310cdbe934ff291a3e68d6ea151231e2b84c13c650e3cc939f8bf23b1338`。`17:53Z` 状态快照统计
3,795 个 eligible pose/sidecar pair、missing artifact/sidecar 0、latest run 新失败 0、抽检 3/3，
近期约 1,382 samples/hour；报告 SHA-256 为
`cd7718e92c3a0aa50b76151416bb3fbd493743d052a676e676939ddf6bc8da66`。四个 worker、下载服务和
integrity timer 均保持 active；`attention_required` 仍只表示三个历史 failed source 被保留。

### Latest passed archive: `archive_052`

`2026-08-11T19:23Z` integrity timer 在 clean commit `10a30e5` 下完整审计 `archive_052`。该 archive
包含 1,689 个视频，逐 member CRC/full read、路径安全和 label coverage 全部通过：

```text
archive SHA-256: dfa60be4fb10bd3eb46465e62f62f0938677795e88b9f08134614c90af86ecc0
audit: manifests/csl_news/source_integrity_v1/audits/archive_052/audit_20260811T192349.038520Z_dfa60be4fb10.json
audit SHA-256: 3fb760832e7b26a3b5ed7c34f2bd7936fe60bd908c89cf156dd21cb9a72a3ba1
```

registry 更新为 32 final、29 passed/48,210 videos，失败仍仅 `001/005/008`，SHA-256 为
`f1a5cd753c32df399dbac59d9102470bdec7262396ca0f0b50c6245386c3ce94`。`19:30Z` 状态报告统计
6,017 个 eligible pair、missing artifact/sidecar 0、latest run 新失败 0、抽检 3/3，近期约
1,488 samples/hour；报告 SHA-256 为
`b60c277162be81e981a9c261e10c0dbfc2d71ba0db2f037e2d9ed21f8db6e27e`。四个 registry worker 均
`active/running`、`NRestarts=0`；`attention_required` 仅表示三个已知 failed source 被保留。

## 4. Evidence Boundary

- 先前 18,095-record source snapshot 使用 `verify_crc=false`，因此只证明 manifest contract、
  portable URI、stable identity 和 source-to-pose text linkage；它不再被称为 source-integrity verified。
- primary `001`、`005`、`008` 及其旧来源 partial pose/failure artifact 全部保留，但不得进入 processed
  dataset、split、训练或论文统计；v2 replacement 是当前可用来源。
- 只有当前 v2 cumulative registry 的 typed `passed` entry 及其精确 `archive_path_relative` 可以进入
  标注池；历史 summary 只作证据。
- `archive_003` 另有 CRC + deterministic video decode smoke；本批其余通过 archive 尚未做 decode probe。
- frozen manual report 仅覆盖当时的 11/436 archives；cumulative registry 仍是 partial，不是完整数据集验证。

## 5. Ongoing Gate

1. 不删除、移动或覆盖当前 ZIP、`.part`、pose、scratch 或 failure sidecar。
2. `001`、`005`、`008` replacement 已固定在 versioned recovery overlay；不得将其覆盖回 primary 路径。
3. annotation resume 必须匹配 archive/labels SHA-256、member size/CRC；不匹配时写 source-versioned 新产物。
4. pose manifest 必须只选与当前 v2 source identity 唯一匹配的 sidecar，其余候选写入 quarantine 清单。
5. 后续每个新完成 archive 必须先通过同等完整 CRC/coverage/decode gate，才能调度标注。
