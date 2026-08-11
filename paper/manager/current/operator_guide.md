# Revision Operator Guide

Status: `active`
Last Updated: `2026-08-11`
Role: `daily_takeover_and_execution`

## 1. 每次新会话启动

依次检查：

```bash
git status --short
git log --oneline -5
df -h /mnt/gfs/yanyifan
```

然后阅读：

1. `paper/manager/dashboard.md`
2. `paper/manager/current/issues.md`
3. `paper/manager/tasks/todo.md`
4. 当前 `paper/manager/rounds/*/README.md`

不要假设上一会话的挂载、数据、GPU job 或 manuscript 状态仍然有效。

## 2. 领取任务

1. 从 `tasks/todo.md` 选择最高优先级且依赖已满足的稳定 ID。
2. 将状态更新为 `in_progress`，记录 owner/session 和开始日期。
3. 阅读相应 runbook 与 source-of-truth 文件。
4. 先写验收条件，再开始改代码、生成数据或运行实验。

## 3. 数据操作

1. 先 inventory/dry-run，再创建或移动文件。
2. 原始数据只读；修复结果写新版本目录。
3. 大规模作业先输出预计文件数、读写量、峰值磁盘和临时空间。
4. 数据生成结束必须写 validation report 和 manifest hash。

### 3.1 CSL-News 下载作业

当前 transient user services：

```bash
systemctl --user status mmprism-csl-news-metadata.service
systemctl --user status mmprism-csl-news-archives.service
journalctl --user -u mmprism-csl-news-archives.service --since today
loginctl show-user "$USER" -p Linger
```

夜间无人登录时要求 `Linger=yes`；两个 service 均使用 `Restart=on-failure`。Transient unit 可跨
shell/SSH 退出运行，但主机重启后需要按 `scripts/download_csl_news.sh` 重新创建，现有 `.part`
文件会自动续传。

文件级进度：

```bash
find /mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/rgb_archives \
  -maxdepth 1 -type f -name 'archive_*.zip' | wc -l
du -sh /mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121
```

`.part` 表示可恢复的未完成文件。不要手工重命名、解压或运行 legacy cleanup/check 脚本。

明早 source-audit trial：

```bash
systemctl --user status mmprism-csl-news-source-trial.timer
systemctl --user list-timers mmprism-csl-news-source-trial.timer
journalctl --user -u mmprism-csl-news-source-trial.service
```

计划时间为 `2026-08-12 08:00 Asia/Shanghai`。它会选择一个已完成的 `.zip`，执行 SHA-256、
ZIP CRC、标签覆盖和 3 个视频完整解码，并写入
`/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1/`。这不是 RTMPose 标注或雷达仿真。

## 4. 实验操作

正式实验开始前记录：

- task/experiment ID
- git commit 和 dirty status
- resolved config
- environment lock/hash
- dataset manifest/split hash
- seed/GPU/precision
- command
- expected outputs and acceptance criteria

结束后把 artifact 路径和结果写入 `evidence/experiment_registry.md`。

## 5. 论文操作

1. 先从 reviewer 原文确认 ask。
2. 确认证据是否 ready，再修改 response 或正文。
3. 修改后更新 `current/manuscript_status.md` 和 tracker。
4. 每轮结束执行 manuscript/response compile、placeholder scan 和引用检查。

### 5.1 Overleaf Git 子模块

首次接管：

```bash
cp .env.example .env
# 仅在本机 .env 中填写私有 token
chmod 600 .env
scripts/overleaf_git.sh init
```

日常操作：

```bash
scripts/overleaf_git.sh status
scripts/overleaf_git.sh pull
# 修改并在 paper/manuscript 内提交后：
scripts/overleaf_git.sh push
```

规则：

1. 主仓库与论文子模块分别提交；主仓库只记录已审核的子模块 commit 指针。
2. 修改前先 `pull --ff-only`，避免静默合并 Overleaf 在线修改。
3. token 只能存在于 `.env`，不能进入 shell 命令、URL、Git config 或文档。
4. 当前主稿入口为 `paper/manuscript/sn-article.tex`；Overleaf Menu 的 Main document 设置需保持一致。
5. 当前机器无本地 TeX 工具链，因此每轮必须在 Overleaf 完成编译、引用和 warning 检查。

## 6. 收尾与交接

1. 按 `sync_map.md` 更新最小文档集。
2. 在当前 round README 记录已完成、未完成、blocker、artifact 和下一动作。
3. 不把未验证的结果标为 done；用 `blocked` 或 `evidence_ready` 表达真实状态。
