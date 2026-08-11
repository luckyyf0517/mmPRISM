# Revision Core Rules

Status: `active_guardrails`
Last Updated: `2026-08-11`
Role: `engineering_and_revision_guardrails`

## 1. 基线与重构

1. `src/mmprism/`、`configs/` 和 `tests/` 是新系统唯一主线；新代码不得导入 legacy 模块。
2. 不建立旧入口、旧配置或旧 checkpoint 的 compatibility shim；历史实现仅用于解释原投稿和提取科学契约。
3. 每个新模块必须先定义输入输出、数据 provenance 和 artifact contract，再进入正式训练。
4. Git 历史中的已删除代码只作为 forensic evidence，不恢复到 canonical package。
5. 在原投稿 evidence 审计结束前，不删除能解释历史实验的脚本、配置或 checkpoint 索引。
6. 从头训练不等于一次同时改变所有变量；data、radar、model、metric 仍按独立 gate 验证。

## 2. 数据

1. `raw/` 数据不可原地修改；修复结果写入版本化 `interim/` 或 `processed/`。
2. split 文件只保存稳定 `sample_id`/`group_id`，不保存机器相关绝对路径。
3. 禁止继续使用字符串 `.replace('poses', 'mmwave')` 推断关联模态路径；关联关系必须来自 manifest。
4. 所有派生数据必须记录 source、处理版本、配置、代码 commit、shape、dtype 和 checksum。
5. subject-independent / sequence-independent split 必须显式记录 group key，并做泄漏检查。
6. GFS 当前空间紧张，任何批量复制、格式转换和缓存生成前必须先做容量估算和 dry-run。

## 3. 实验

1. 每个正式 run 必须保存 resolved config、启动命令、git commit、环境摘要、manifest hash、seed 和 metrics。
2. paper-facing 指标必须保留 sample-level prediction，不能只保存均值。
3. 指标实现发生变化时必须创建新的 protocol version；新旧数值不能静默覆盖。
4. W&B/SwanLab 只能作为可视化副本，本地 artifact 才是论文 provenance 真值。
5. checkpoint 未绑定数据 manifest 和配置前只能标记为 `unverified_legacy`。

## 4. 论文返修

1. 不从二手 todo 推断 reviewer 原意；每轮必须回到原始 reviewer comment。
2. response letter 声称的每项改动必须在正文、图表或补充材料中真实落地。
3. 没有实验支持的强 claim 不得通过措辞包装进入正文。
4. reviewer comment、实验、证据和正文位置统一用稳定 ID 链接。
5. 数值、表号、图号或实验 protocol 变化后必须按 `sync_map.md` 做最小同步。

## 5. 运行与安全

1. 所有数据根目录、模型根目录和输出根目录通过配置或环境变量传入。
2. destructive 操作默认禁止；清理、移动或覆盖前必须生成清单并可恢复。
3. 不把 token、API key、个人绝对路径或大模型权重提交到 Git。
