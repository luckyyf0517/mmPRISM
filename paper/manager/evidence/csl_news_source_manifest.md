# CSL-News Source Manifest Evidence

Status: `v1_historical_v2_builder_verified_snapshot_pending`
Last Updated: `2026-08-11`
Role: `DATA-003-B_source_manifest_evidence`

## Current v2 Builder Contract

`mmprism.csl_news_source_manifest.v2` no longer scans primary archive names as the source of truth. It requires
`source.integrity_registry`, accepts only registry schema v2 and typed `passed` entries, and resolves each exact
`archive_path_relative` under the configured archive root. Every registered archive is checked against its
size, mtime, SHA-256, video count and audit provenance; root/count/labels identity mismatches, path escapes and
symlinks fail before publication.

Each snapshot copies the exact registry bytes to `integrity_registry.json`, writes registry SHA-256 and
source-kind/path/audit identity into records and summary, validates the general manifest contract, and creates
`SHA256SUMS` for registry/manifest/summary before atomic publication. Unit tests cover primary and replacement
selection, preserved corrupt primary input, stat drift, same-stat content drift, symlink rejection, labels,
clean Git, portable paths and checksum replay. A real clean v2 snapshot is the next evidence gate.

## 1. Historical v1 Snapshot Identity

```text
source: huggingface:ZechengLi19/CSL-News
revision: 3a0601210333fe760efd09b5d9e2ae5f341ce339
builder commit: 96ccc6e966c216c4dfc4196b105646bbfa2f9c25
builder Git state: clean
config fingerprint: 00b84ee1199535022a2303bce0ed554a5c47d5709a2660a5d0d8080913c89cf6
snapshot: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_manifest_v1/snapshot_20260811T152502.395066Z
manifest SHA-256: 6984d0cc30a0f5a9e6baa58fa8a764e0c0b70ed1b0bb9224e9fca8faa1b1a1f5
summary SHA-256: 9fbe74f4655b1f37a951af3ff707c86c2c4199a2177aa2de2551c3a95874f8ac
status: partial
```

Historical entrypoint (the versioned config now targets v2 and will create a new snapshot root):

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM scripts/run_csl_news_source_manifest.sh
```

正式 snapshot 要求 clean Git；因此新运行会生成 v2 新目录，不覆盖本 v1 历史目录。

## 2. Frozen Input Boundary

扫描开始时冻结 11 个 final ZIP，archive ID 为：

```text
003, 004, 005, 006, 008, 009, 010, 011, 014, 015, 020
```

- 当前 snapshot video/record：18,095；
- canonical labels：722,711；当前覆盖 2.5038%；
- source program：Common-Concerns 11,873，Dragon-TV 6,127，unknown 95；
- compressed/uncompressed ZIP members：23,577,466,794 / 23,615,216,510 bytes；
- scan start 可见 `.part`：13；这些文件未读取；
- `verify_crc=false`，当前只记录 central-directory member CRC；最终 436-archive snapshot 必须执行
  完整 ZIP/CRC gate。
- 后续完整逐 member 审计确认该 frozen scope 中 `archive_005`、`archive_008` 损坏。因此本 snapshot
  不能再表述为 source-integrity verified，也不能作为训练或论文统计的直接输入。

后台 aria2 在构建期间可能完成新 archive，但它们不属于本次 frozen input，会进入下一次 snapshot。

## 3. Record Contract

每条 `mmprism.sample.v1` 包含：

- stable `sample_id` 和原始 `sequence_id`；
- portable `zip://archive_NNN.zip!/member.mp4` video URI；
- inline canonical JSON caption、UTF-8 dtype 与 caption SHA-256；
- archive SHA-256、member CRC/size、labels SHA-256；
- source/revision、config fingerprint 和 clean builder commit；
- archive/source-program group keys；subject 和 scene 明确为 source metadata unavailable。

manifest 不包含 `/mnt/` 或 `/home/` 绝对路径。存储根只在 versioned config/运行环境注入。

## 4. Independent Verification

| Gate | Result |
|---|---|
| line count | 18,095 |
| `mmprism manifest` contract | passed；dataset `csl_news`；modalities `caption`,`video` |
| manifest SHA-256 recomputation | matched |
| summary SHA-256 | `9fbe74f4...` |
| absolute local path scan | 0 matches |
| prior `archive_003` SHA-256 | matched `ae348f6c...` |
| pose sidecar IDs checked | 676；missing 0 |
| pose sidecar caption comparison | mismatch 0 |

上述 gate 只验证 manifest contract/linkage。后续 source integrity audit 的结果为 9 个 archive 通过、
2 个 archive 损坏；机器可读总表 SHA-256 为 `ea8062f546cdf10abdde5b5b27e0e78e5e39e3df538e0d68b983e6ac4b7c9a00`，
详见 `csl_news_source_integrity.md`。

## 5. Remaining Gate

本 snapshot 证明 manifest schema、stable identity 和 source-to-pose linkage，但不能支撑全量数据声明。
`DATA-003-B` 仍为 `in_progress`，最终验收还需要：

1. 436 个 archive 全部 final 后生成 `complete` snapshot；
2. `001/005/008` replacement 已验证；继续对其余 archive 执行 SHA-256、CRC、member safety、label coverage；
3. 对 deterministic video sample 做 decode/shape/FPS 验收；
4. 以 complete manifest hash 生成 split，并做 group/duplicate leakage audit；
5. 论文写回只引用 complete/frozen manifest 的统计，不引用本 partial snapshot 作为全量数字。
