# Data Discovery and Rebuild Runbook

Status: `active_plan`
Last Updated: `2026-08-11`
Role: `repeatable_data_rebuild_process`

## 0. 原则

- 先只读 inventory，再决定迁移或转换。
- raw immutable，派生版本可重建。
- manifest 是模态关系真值，目录名不是。
- split 由 group metadata 生成，不从当前文件排列推断。
- 每一步都有 dry-run、容量估计和 validation report。

## 1. Source Discovery

对每个可能来源记录：

- filesystem/storage URI
- owner 和权限
- 总大小、文件数、mtime 范围
- 目录结构和扩展名分布
- 是否为原始、派生、缓存、checkpoint 或 result
- 是否允许移动/删除

不得在 source discovery 阶段计算全量 checksum；先对大小/mtime/抽样 checksum 建立成本模型。

当前来源尚未在 GFS 上，先按 `../data_upload_checklist.md` 执行 upload preflight：来源端提供目录/归档
大小、文件数、类别、可重新下载性和 access class。任何内容先进入
`/mnt/gfs/yanyifan/mmPRISM/incoming/<batch-id>/`，不得直接写入 canonical `raw/`。

## 2. Inventory

inventory 输出建议为 JSONL/Parquet，每个文件至少包含：

```text
source_id, relative_path, size_bytes, mtime, suffix,
sampled_checksum, detected_family, detected_modality, status
```

每个 incoming batch 还必须保留来源端 `UPLOAD_MANIFEST.csv` 和 `SHA256SUMS`。全量 archive checksum
在来源端生成，GFS 端验证；解压后文件级 checksum 可按数据规模分阶段补齐。

另生成 summary：

- bytes/files by family/modality/suffix
- largest directories/files
- suspected duplicates
- unreadable/corrupt files
- estimated materialization space

## 3. Schema and Manifest

1. 先为每个数据族实现 source adapter。
2. adapter 只读 source 并产生 canonical sample records。
3. manifest validator 检查 ID 唯一性、路径存在、shape/dtype、模态对齐和 provenance。
4. 不完整 sample 进入 quarantine report，不静默丢弃。

## 4. Split

优先 group key：

1. subject/signer
2. sequence/session
3. scene/acquisition batch

生成 split 后必须检查：

- group disjointness
- source manifest SHA-256 和 100% sample coverage
- class/action/caption distribution
- sequence length distribution
- modality missing rate
- 与原投稿 split 的差异

## 5. Materialization

只有正式实验需要的版本才 materialize。每个版本目录必须有：

```text
dataset manifest snapshot + SHA-256
split snapshot/assignments.jsonl + SHA256SUMS
resolved build config
build and validation reports
provenance/runtime metadata
```

## 6. Acceptance

- 小样本可端到端读取。
- manifest hash 稳定。
- 重复运行不产生不同 split 或输出。
- 处理失败有明确记录。
- 存储峰值不超过批准预算。
- upload checksum 与来源端一致，incoming 原始包保持只读。
- subject/session/sequence/radar-config 引用完整；unknown 字段显式记录。
- license、伦理和 reviewer/public access 范围已登记。
