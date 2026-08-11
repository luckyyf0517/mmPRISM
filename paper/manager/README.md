# Revision Management Workspace

Status: `active_round0`
Last Updated: `2026-08-11`
Role: `navigation_and_control_plane`
Owner Scope: `paper/manager/*`

`paper/manager/` 是本轮 Nature Communications 返修的管理控制面。它参考 `mmExpert-Devel/paper/manager/` 的单一真值设计，但针对 mmPRISM 当前的核心问题扩展了代码架构、数据资产和实验 provenance 管理。

## 1. 本控制面回答的问题

1. 整体进行到哪一步：`dashboard.md`
2. 当前项目和论文主线是什么：`current/overview.md`
3. 哪些规则不能漂移：`current/core_rules.md`
4. 当前有哪些 blocker 和风险：`current/issues.md`
5. 旧代码和目标架构差距是什么：`current/architecture_status.md`
6. 数据当前在哪里、应该如何重建：`current/data_status.md`
7. 正文和 response letter 当前状态如何：`current/manuscript_status.md`
8. 全面路线图是什么：`current/roadmap.md`
9. 现在具体要做什么：`tasks/todo.md`
10. 每条 reviewer comment 如何闭环：`reviews/response_letter_tracker.md`
11. 每个论文结论如何追溯到数据和实验：`evidence/`
12. 每轮工作如何交接：`rounds/`

## 2. 快速接管顺序

新会话或新参与者按以下顺序阅读：

1. `dashboard.md`
2. `current/overview.md`
3. `current/issues.md`
4. `current/roadmap.md`
5. `tasks/todo.md`
6. 当前 round 的 `rounds/*/README.md`
7. `reviews/response_letter_tracker.md`
8. `current/operator_guide.md`

## 3. 目录结构

```text
paper/manager/
  README.md
  dashboard.md
  sync_map.md
  current/
  decisions/
  evidence/
  reviews/
  rounds/
  runbooks/
  tasks/
```

## 4. 单一真值规则

- 项目主线与边界：`current/overview.md`
- 不可漂移的工程和论文规则：`current/core_rules.md`
- 活跃风险：`current/issues.md`
- 架构现状与目标：`current/architecture_status.md`
- 数据现状与目标：`current/data_status.md`
- 阶段与里程碑：`current/roadmap.md`
- 当前优先级：`tasks/todo.md`
- reviewer comment 到证据/实验/正文的映射：`reviews/response_letter_tracker.md`
- 论文表格、图片和 claim 的 provenance：`evidence/paper_evidence_map.md`
- 已拍板的跨模块决定：`decisions/decision_log.md`

其他文档只能链接这些主归属文件，不应复制一份独立事实。

## 5. 状态与 ID 规范

统一状态：

```text
not_started / in_progress / blocked / evidence_ready /
writeback_ready / done / superseded
```

稳定 ID：

- `REV-*`：审稿意见和 response 工作
- `ARCH-*`：架构整理与重构
- `DATA-*`：数据定位、校验和重建
- `EXP-*`：复现、新增实验和评测
- `PAPER-*`：正文、图表和 response letter
- `OPS-*`：环境、存储、运行和发布

## 6. 维护原则

1. 每个任务必须有 owner、状态、依赖和验收条件。
2. 每个 paper-facing 数值必须绑定代码 commit、resolved config、数据 manifest、checkpoint 和 metrics artifact。
3. 旧结果在 provenance 未核对前只能标记为 `historical`，不能直接进入返修结论。
4. 数据重建和代码重构不能同时改变输入定义、模型逻辑和指标实现；每次只改变一层，并保留对照。
5. `dashboard.md` 只保留阶段、P0、blocker 和 next action，不承载完整细节。
