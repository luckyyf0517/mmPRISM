# Evidence and Provenance Registry

Status: `active_bootstrap`
Last Updated: `2026-08-11`

本目录把论文结论与数据、实验和 artifact 绑定，防止返修过程中出现“表格里有数字，但找不到来源”的情况。

## 文件职责

- `data_registry.md`：数据族、manifest、split、license 和质量状态。
- `csl_news_source_integrity.md`：CSL-News frozen archive 的完整 CRC、隔离和恢复证据。
- `csl_news_pose_manifest.md`：CSL-News integrity-gated pose+caption partial snapshot 与 adapter 验收证据。
- `csl_news_pose_split.md`：CSL-News pose partial manifest 的 sequence-disjoint split 与 leakage audit。
- `experiment_registry.md`：每次正式 run 的配置、commit、数据、checkpoint、prediction 和 metrics。
- `paper_evidence_map.md`：paper claim/table/figure 到 data/experiment/reviewer/manuscript 的映射。

## Evidence ID

- 数据：`DATASET-*`
- split：`SPLIT-*`
- 实验：`RUN-*`
- 指标协议：`METRIC-*`
- 论文证据块：`EVID-*`

paper-facing 数值不允许只引用 logger URL；必须有本地或挂载盘 artifact 路径和 hash。
