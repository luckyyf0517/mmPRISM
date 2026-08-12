# CSL-News Pose Split Evidence

Status: historical
Owner: CSL-News annotation lane
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12
Legacy evidence role: `DATA-004-A_sequence_split_contract_evidence`

## 1. Snapshot Identity

```text
source manifest: manifests/csl_news/pose_manifest_v1/snapshot_20260811T164204Z_partial/manifest.jsonl
source manifest SHA-256: 4161593fdbfc85a5c2fb392e3ef92d40da560db5c75a19d559f1f92878e31600
source records / scope: 2,157 / partial
builder commit: eb5de64d9a0c4cb11610709b502a5218661b0157
builder Git state: clean
config fingerprint: 0a62b351aae8df1d1166ec54762942736be81f3eef388195e37f5440be8e1fcf
protocol: csl_news_pose_sequence_hash_80_10_10_v1
seed / algorithm: 20260811 / sha256_mod_weight_v1
snapshot: /mnt/gfs/yanyifan/mmPRISM/splits/csl_news/pose_sequence_hash_80_10_10_v1/snapshot_20260811T165233Z_partial
assignments SHA-256: 133f32d58b213947edf09c7c1e1b7c3ee30b8588a9f2b7a863d6a668bce2d7d9
config SHA-256: 1c7d1cf2c85202ea110ea2a89208b7cc3d8f034a922b729dd03c449c3a54931e
summary SHA-256: 378a6cdf4a144cad473d98f140c66c6fbf6de3ccdfce28f4439d95a816c393d9
status: partial
```

复现入口：

```bash
MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM \
  scripts/run_csl_news_pose_split.sh
```

## 2. Protocol

当前官方 CSL-News metadata 没有 signer/subject/session ID，因此本次只使用 `sequence_id` 建立
sequence-disjoint engineering split。raw sequence value 不进入 assignments；`group_id` 是
`namespace + dataset + selector + value` 的完整 SHA-256。分配对
`protocol_id + seed + group_id` 再做 SHA-256，并按有序整数权重 `8/1/1` 取模。

| Split | Samples | Sequence groups |
|---|---:|---:|
| train | 1,701 | 1,701 |
| validation | 219 | 219 |
| test | 237 | 237 |
| total | 2,157 | 2,157 |

当前 source 中 `sequence_id` 全部唯一，因此一个 group 对应一个 sample。权重定义长期确定性分配，
不保证有限 partial snapshot 的计数恰好等于 80%/10%/10%。

## 3. Independent Verification

| Gate | Result |
|---|---|
| source manifest hash/count/dataset | matched |
| assignment coverage | 2,157/2,157；missing 0；extra 0 |
| duplicate sample ID | 0 |
| duplicate source sequence ID | 0 |
| cross-split `group_id` leakage | 0 |
| configured split presence/minimum groups | 3/3 passed |
| independent group ID and hash-bucket recomputation | 2,157/2,157 matched |
| `SHA256SUMS` | assignments/config/summary 3/3 `OK` |
| absolute local path scan in assignments/config | 0 matches |
| dependency-light `SplitIndex` load | passed |

代码质量门为 56 tests、Ruff、strict Mypy、shell syntax、sdist 和 wheel build 全部通过。

## 4. Evidence Boundary

- 该 split 只绑定第一份 2,157-record partial pose manifest；后台新增样本不属于本 snapshot。
- 它证明 deterministic sequence grouping、portable artifact 和 leakage validator 可用，不是最终训练 split。
- CSL-News source 尚未完成 436-archive 下载/标注，final manifest 必须生成新的 complete split。
- signer/subject metadata 缺失，因此不能声称 subject-independent 或 new-user generalization。
- caption/class balance、近重复视频、scene/session 和 non-manual 分层尚未审计。
- 原投稿使用的历史 split 仍未定位；本 split 不用于冒充或重建原投稿数值。

因此 `DATA-004-A` 和 master `DATA-004` 推进为 `in_progress`，不得标记完成。
