# CSL-News Source Integrity Evidence

Status: `cumulative_registry_active_failures_isolated`
Last Updated: `2026-08-11`
Role: `DATA-001-K_source_integrity_evidence`

## 1. Audit Identity

Current cumulative control artifact:

```text
builder commit: b18251204ece87b67702d1f6735c9eb9cb416a9b
registry: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1/registry.json
registry SHA-256 at 2026-08-11T16:17Z: 070bcc4446894577cab6e05f632049a2a53143b508e50523dd27c20daea52b66
labels SHA-256: 3381d80157fa75012ec2a220eb8a63c88968af2d60d5dbcb5a82bf680db8a3a5
present final archives: 15
passed: 12 archives / 19,760 videos
failed: 001,005,008
```

Registry 更新使用非阻塞 `flock`、per-archive audit 和原子替换。archive source stat 或标签 hash
变化时旧结果不可复用；标注 worker 同时核对 source ID/revision/stat，只消费 typed `passed` entry。
registry 以单次 byte snapshot 读取并同时计算 hash；成功/失败 sidecar 和 archive marker 都记录该
snapshot hash 及对应 archive audit provenance，避免长运行 worker 只绑定启动时的旧 registry。

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

## 2. Results

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

## 3. Evidence Boundary

- 先前 18,095-record source snapshot 使用 `verify_crc=false`，因此只证明 manifest contract、
  portable URI、stable identity 和 source-to-pose text linkage；它不再被称为 source-integrity verified。
- `001`、`005`、`008` 及其已生成的 partial pose/failure artifact 全部保留，但不得进入 processed dataset、
  split、训练或论文统计。
- 只有当前 cumulative registry 的 typed `passed` entry 可以进入标注池；历史 summary 只作证据。
- `archive_003` 另有 CRC + deterministic video decode smoke；本批其余通过 archive 尚未做 decode probe。
- frozen manual report 仅覆盖当时的 11/436 archives；cumulative registry 仍是 partial，不是完整数据集验证。

## 4. Recovery Gate

1. 不删除、移动或覆盖当前 ZIP、`.part`、pose、scratch 或 failure sidecar。
2. 人工复核后，把 `001`、`005`、`008` 重新下载到新的 versioned incoming/recovery 位置。
3. 对 replacement 执行完整 SHA-256、逐 member CRC、label coverage 和 decode probe。
4. replacement 通过前保持原文件为 quarantine candidate；通过后登记新 source identity，再决定 promotion。
5. 后续每个新完成 archive 必须先通过同等完整 CRC gate，才能调度标注。
