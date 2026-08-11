# mmPRISM Paper Revision Workspace

Status: `active_bootstrap_manuscript_linked`
Last Updated: `2026-08-11`

本目录用于管理 mmPRISM 的 Nature Communications 返修工作。它不替代代码仓库，也不把实验日志散落进论文正文；它负责把审稿意见、代码重构、数据重建、实验复现、论文证据和最终回写连接成一条可追踪链路。

## 目录职责

- `paper/manager/`：返修管理控制面，维护当前真值、任务、风险、轮次和证据索引。
- `paper/manuscript/`：Overleaf Git 子模块，保存当前论文正文；response letter 后续在同一稿件仓库维护。
- `paper/assets/`：预留给最终图表、场景照片和 submission package 资产；大文件不应直接提交到主仓库。

## 当前阶段

当前处于 `Round0 / revision bootstrap`：

1. 冻结旧代码基线并建立现状审计。
2. 定位 `/mnt/gfs/yanyifan` 下的原始数据、权重和历史实验资产。
3. 建立可复现的数据与实验契约。
4. 在不破坏历史结果的前提下启动渐进式重构。
5. 编辑决定与审稿意见已导入并编号；当前 Overleaf 稿件已接入，下一步绑定具体章节、表图和实验 evidence。

## Overleaf 稿件仓库

- 子模块路径：`paper/manuscript`
- 当前主稿入口：`paper/manuscript/sn-article.tex`
- 私有认证：根目录 `.env`，该文件被 Git 忽略且权限应为 `0600`
- 安全操作入口：`scripts/overleaf_git.sh`

```bash
cp .env.example .env
# 在 .env 中填写 OVERLEAF_GIT_TOKEN
scripts/overleaf_git.sh init
scripts/overleaf_git.sh status
scripts/overleaf_git.sh pull
scripts/overleaf_git.sh push
```

不得把 token 写入 clone URL、`.gitmodules`、Git remote 或命令历史。当前机器未安装 `latexmk`/`pdflatex`/`bibtex`，正式编译仍以 Overleaf 为准。

管理入口：`paper/manager/README.md`。
