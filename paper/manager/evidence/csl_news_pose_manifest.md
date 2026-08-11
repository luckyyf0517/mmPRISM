# CSL-News Pose Manifest Evidence

Status: `partial_snapshot_evidence_ready`
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
