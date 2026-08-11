# Revision Sync Map

Status: `active`
Last Updated: `2026-08-11`
Role: `minimal_cross_document_sync`

本文件规定一次改动完成后最少需要同步哪些管理文档，避免每次全量改写。

| 触发事件 | 必须更新 | 条件性更新 |
|---|---|---|
| 发现新数据目录或权重 | `current/data_status.md`, `evidence/data_registry.md` | blocker 改变时更新 `dashboard.md` |
| 上传范围、batch 或容量 gate 改变 | `data_upload_checklist.md`, `current/data_status.md`, `tasks/todo_data.md` | source 到达时更新 `evidence/data_registry.md`；风险变化时更新 `current/issues.md` 和 `dashboard.md` |
| 数据 schema、split 或 manifest 改变 | `evidence/data_registry.md`, `tasks/todo_data.md` | 影响实验时更新 `evidence/experiment_registry.md` |
| 架构决定拍板 | `decisions/decision_log.md`, `current/architecture_status.md` | 产生任务时更新 `tasks/todo_code.md` |
| 完成一次实验 | `evidence/experiment_registry.md` | 支撑论文时更新 `evidence/paper_evidence_map.md` |
| reviewer comment 状态变化 | `reviews/response_letter_tracker.md` | P0/blocker 变化时更新 `dashboard.md` |
| 正文或 response letter 修改 | `current/manuscript_status.md` | 证据映射变化时更新 `evidence/paper_evidence_map.md` |
| Overleaf 子模块 URL、分支、主入口或同步方式变化 | `current/manuscript_status.md`, `current/operator_guide.md` | 阶段/blocker 改变时更新 `dashboard.md` 和 `decisions/decision_log.md` |
| round 开始或结束 | 当前 `rounds/*/README.md`, `current/roadmap.md` | 阶段变化时更新 `dashboard.md` |
| 新风险出现或关闭 | `current/issues.md` | P0 风险变化时更新 `dashboard.md` |

## 最小提交前检查

1. 任务状态与证据状态一致。
2. reviewer tracker 中的 manuscript/evidence 链接可打开。
3. paper-facing 数值能够回溯到 experiment registry。
4. `dashboard.md` 没有已经关闭的 blocker。
