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

## 2. Inventory

inventory 输出建议为 JSONL/Parquet，每个文件至少包含：

```text
source_id, relative_path, size_bytes, mtime, suffix,
sampled_checksum, detected_family, detected_modality, status
```

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
- class/action/caption distribution
- sequence length distribution
- modality missing rate
- 与原投稿 split 的差异

## 5. Materialization

只有正式实验需要的版本才 materialize。每个版本目录必须有：

```text
dataset_manifest.jsonl
split_manifest.json
build_config.yaml
build_report.json
validation_report.json
provenance.json
```

## 6. Acceptance

- 小样本可端到端读取。
- manifest hash 稳定。
- 重复运行不产生不同 split 或输出。
- 处理失败有明确记录。
- 存储峰值不超过批准预算。
