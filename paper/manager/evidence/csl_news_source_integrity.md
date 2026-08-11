# CSL-News Source Integrity Evidence

Status: `partial_audit_failed_corrupt_archives_isolated`
Last Updated: `2026-08-11`
Role: `DATA-001-K_source_integrity_evidence`

## 1. Audit Identity

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

## 3. Evidence Boundary

- 先前 18,095-record source snapshot 使用 `verify_crc=false`，因此只证明 manifest contract、
  portable URI、stable identity 和 source-to-pose text linkage；它不再被称为 source-integrity verified。
- `005`、`008` 及其已生成的 partial pose/failure artifact 全部保留，但不得进入 processed dataset、
  split、训练或论文统计。
- 只有本报告 `passed_archive_ids` 中的 archive 可以进入临时并行标注池。
- `archive_003` 另有 CRC + deterministic video decode smoke；本批其余通过 archive 尚未做 decode probe。
- 该报告仅覆盖扫描时的 11/436 archives，不是完整数据集验证。

## 4. Recovery Gate

1. 不删除、移动或覆盖当前 ZIP、`.part`、pose、scratch 或 failure sidecar。
2. 人工复核后，把 `005`、`008` 重新下载到新的 versioned incoming/recovery 位置。
3. 对 replacement 执行完整 SHA-256、逐 member CRC、label coverage 和 decode probe。
4. replacement 通过前保持原文件为 quarantine candidate；通过后登记新 source identity，再决定 promotion。
5. 后续每个新完成 archive 必须先通过同等完整 CRC gate，才能调度标注。

