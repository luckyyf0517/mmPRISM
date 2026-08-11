# Data Rebuild Todo

Status: `blocked_on_asset_location`
Last Updated: `2026-08-11`
Role: `data_execution_tracker`

| ID | Priority | Task | Status | Acceptance |
|---|---|---|---|---|
| `DATA-001-A` | P0 | 确认历史数据实际位置和访问方式 | blocked | 每个来源有 owner/location/access |
| `DATA-001-B` | P0 | 只读统计容量、文件数、后缀和目录层次 | blocked | inventory report，不生成副本 |
| `DATA-001-C` | P0 | 定位 checkpoint、log、prediction、metric artifacts | blocked | historical artifact inventory |
| `DATA-001-D` | P0 | 评估 `/mnt/gfs` 容量和迁移策略 | in_progress | 峰值空间与清理/归档方案 |
| `DATA-002-A` | P1 | 定义 sample/sequence/acquisition/provenance schema | not_started | schema v1 reviewed |
| `DATA-002-B` | P1 | 定义 pose joint/坐标系/单位规范 | not_started | 所有数据族 mapping 明确 |
| `DATA-002-C` | P1 | 定义 raw radar complex representation 与 radar config version | not_started | reader/validator fixture |
| `DATA-003-A` | P1 | CSL-Daily source adapter 和 manifest | not_started | coverage/shape/annotation report |
| `DATA-003-B` | P1 | CSL-News source adapter 和 manifest | not_started | coverage/shape/annotation report |
| `DATA-003-C` | P1 | collected source adapter 和 manifest | not_started | subject/scene/action metadata report |
| `DATA-003-D` | P1 | damaged/missing/ambiguous asset quarantine | not_started | quarantine reason registry |
| `DATA-004-A` | P1 | 建立 subject/signer/sequence group split | not_started | deterministic split + leakage audit |
| `DATA-004-B` | P1 | 识别原投稿 split | not_started | paper split hash/provenance |
| `DATA-005-A` | P2 | 重建 pose annotation pipeline | not_started | versioned pose output + QC |
| `DATA-005-B` | P2 | 重建 radar processing/simulation pipeline | not_started | versioned radar output + QC |
| `DATA-005-C` | P2 | 重建 pred_pose/feature pipeline | not_started | checkpoint-bound provenance |
| `DATA-006-A` | P2 | 生成 model-ready processed dataset | not_started | validation report + manifest hash |
| `DATA-REV-001` | P0 | 统计 sign type/vocab/sentences/length/non-manual/subjects/scenes/splits | blocked | manuscript-ready table + machine-readable summary |
| `DATA-REV-002` | P0 | 方向/双手重叠/物体遮挡/新用户真实数据 protocol 与采集 | blocked | ethics-cleared held-out manifest and QC |
| `DATA-REV-003` | P0 | paired/category-matched synthetic-real evaluation set | blocked | same-sign fidelity manifest |

## 禁止事项

- 在找到源数据前创建大规模占位副本。
- 直接覆盖旧 pose/mmWave/features。
- 在 split 中保存新的机器绝对路径。
- 未做容量预算就展开 NPY 或生成重复缓存。
