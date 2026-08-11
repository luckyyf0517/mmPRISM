# Evidence and Provenance Registry

Status: `active_bootstrap`
Last Updated: `2026-08-11`

本目录把论文结论与数据、实验和 artifact 绑定，防止返修过程中出现“表格里有数字，但找不到来源”的情况。

## 文件职责

- `data_registry.md`：数据族、manifest、split、license 和质量状态。
- `csl_news_source_integrity.md`：CSL-News frozen archive 的完整 CRC、隔离和恢复证据。
- `csl_news_pose_manifest.md`：CSL-News integrity-gated pose+caption partial snapshot 与 adapter 验收证据。
- `csl_news_pose_split.md`：CSL-News pose partial manifest 的 sequence-disjoint split 与 leakage audit。
- `manuscript_inventory.md`：当前 Overleaf 主稿和 supplementary 的结构、引用、资产与合规静态证据。
- `display_item_registry.md`：20 个当前 display item 的 Source Data 与 provenance 控制表。
- `radar_contract_audit.md`：雷达张量/range-Doppler 契约、稿件与 legacy 冲突及 beamforming gate。
- `release_audit.md`：公开 release allowlist、依赖图、逐文件 hash、排除边界和剩余交付 blocker。
- `model_assets.md`：SimCSE/SBERT fixed revision、逐文件 checksum 与真实 loader smoke 证据。
- `model_support_boundary.md`：caption-generation supported scope、legacy Phi-3 排除和自动回归 gate。
- `mt5_vertical_smoke.md`：pinned mT5、geometry-fusion 两步 GPU smoke 和非论文证据边界。
- `omnihand_vertical_smoke.md`：canonical CubeNet/OmniHand 两步 GPU smoke、pose metric 和非论文证据边界。
- `omnihand_formal_run.md`：clean-commit A100 上 train/checkpoint/reload/prediction/evaluate 正式闭环证据。
- `artifacts/manuscript_inventory_v2.json`：由审计工具生成的逐文件、逐行、逐 display item 机器可读 inventory。
- `artifacts/manuscript_inventory_v1.json`：首次环境级审计的历史快照，不再作为当前 canonical inventory。
- `artifacts/release_audit_v1.json`：clean commit 上生成的 reviewer release 机器可读预审报告。
- `artifacts/evaluation_models_smoke_v1.json`：clean commit 上的 SimCSE/SBERT 本地加载与 embedding smoke。
- `artifacts/mt5_smoke_v1.json`：脱敏的 clean-commit mT5 forward/backward/update/generate smoke。
- `artifacts/omnihand_smoke_v1.json`：脱敏的 clean-commit CubeNet forward/backward/update/metric smoke。
- `artifacts/omnihand_formal_run_v1.json`：OmniHand 正式 GPU 闭环的哈希、复现一致性与性能摘要。
- `experiment_registry.md`：每次正式 run 的配置、commit、数据、checkpoint、prediction 和 metrics。
- `paper_evidence_map.md`：paper claim/table/figure 到 data/experiment/reviewer/manuscript 的映射。

## Evidence ID

- 数据：`DATASET-*`
- split：`SPLIT-*`
- 实验：`RUN-*`
- 指标协议：`METRIC-*`
- 论文证据块：`EVID-*`

paper-facing 数值不允许只引用 logger URL；必须有本地或挂载盘 artifact 路径和 hash。
