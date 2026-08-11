# Revision Decision Log

Status: `active`
Last Updated: `2026-08-11`
Role: `cross_workstream_decisions`

| ID | Date | Decision | Rationale | Status | Consequence |
|---|---|---|---|---|---|
| `DEC-001` | 2026-08-11 | 使用 `paper/manager` 作为返修管理控制面 | 与参考项目一致，集中管理 reviewer、架构、数据和 evidence | accepted | 所有返修状态从本目录进入 |
| `DEC-002` | 2026-08-11 | 采用渐进式兼容重构，不直接推倒重写 | 初始假设是必须沿用历史数据与训练链 | superseded by `DEC-010` | 不再实施 compatibility shim |
| `DEC-003` | 2026-08-11 | 接受 `/mnt/gfs/yanyifan/mmPRISM` 为 canonical data root | 作者指定数据位于 `/mnt/gfs/yanyifan`；代码与大数据分离 | accepted | 上传先进入 versioned `incoming/`；容量与来源确认前不批量创建或解压数据 |
| `DEC-004` | 2026-08-11 | raw immutable，模态关系由 manifest 管理 | 消除路径替换和原地修改风险 | accepted | 数据重建必须 versioned |
| `DEC-005` | 2026-08-11 | 本地/挂载 artifact 是论文 provenance 真值 | 在线 logger 可能丢失或不可审计 | accepted | 每个 run 保存完整 metadata/predictions |
| `DEC-006` | 2026-08-11 | 原始邮件副本脱敏私人 MTS 链接，完整 `.eml` 只存 Git 忽略目录 | 投稿链接含私人访问令牌，不能进入代码仓库 | accepted | 科学/编辑文本完整保留，凭据不提交 |
| `DEC-007` | 2026-08-11 | 先完成 Stage 1 诊断，不在无结果时起草结果型 response | 避免编造实验、数值和 manuscript location | accepted | 当前只维护 tracker/plan，结果完成后进入 Stage 2 |
| `DEC-008` | 2026-08-11 | 将真实 stress matrix 作为信息密集型核心实验 | 同时回答方向、遮挡、新用户和 real-world generalization | proposed | 需作者确认采集/伦理/时间资源 |
| `DEC-009` | 2026-08-11 | 当前返修稿以 `paper/manuscript` 私有 Overleaf Git 子模块管理 | 保留 Overleaf 在线编译能力，同时让主仓库记录可审计稿件版本 | accepted | token 仅放根目录 `.env`；通过 `scripts/overleaf_git.sh` 同步，主入口为 `sn-article.tex` |
| `DEC-010` | 2026-08-11 | 采用 greenfield 全量重构，`src/mmprism` 是唯一新实现主线 | 作者确认数据和训练都将从头重建，无需保持旧训练链兼容 | accepted | 旧脚本/模块只读取证；新包禁止导入旧模块，不建立 compatibility shim |
| `DEC-011` | 2026-08-11 | 数据 intake 按“不可替代 raw/metadata 优先，公共资产下载，可再生派生物重建”执行 | GFS 仅余约 141 GB，无法无差别回传历史缓存 | accepted | 上传前必须 size/checksum preflight；metadata/calibration 和私人 raw 优先，pose/signal/feature 默认不上传 |
| `DEC-012` | 2026-08-11 | 在 provenance 关闭前不把 skeleton 仿真视为稿件 MANO mesh pipeline 的等价复现 | 稿件方法描述与当前可见 legacy code 不一致 | accepted | 必须恢复原 simulator/输入/配置或修订方法描述，并分别登记实验协议 |
| `DEC-013` | 2026-08-11 | CSL-News 使用官方 HF revision `3a060121` 从头下载，压缩包验证前不解压 | 官方源、935 GB 规模和 CC BY-NC 4.0 条款已确认；共享盘当前约余 3.6 TB | accepted | 436 archives 由可恢复 systemd service 下载，保留 1 TiB floor，完成后建立 source manifest |
| `DEC-014` | 2026-08-11 | CSL-News 下载从 16 路单连接 curl 切换为 4 worker x 8 连接 aria2 | HF 端点支持 Range；稳定后 60 秒有效写入 9.95 MB/s，高于切换前约 3.9–4.7 MB/s | accepted | 原 `.part` 原位续传，完成后仍原子重命名；curl 保留为脚本 fallback |
| `DEC-015` | 2026-08-11 | CSL-News pose 以 RTMW3D 原生输出加显式历史 2x24 transform 重建 | 保留可重解释的 133-joint 证据，避免将旧映射不可逆写入唯一数据 | accepted | 每个样本保存 native/transformed/canonical 数组、文本、checksum 和失败 sidecar |
| `DEC-016` | 2026-08-11 | 夜间 pose worker 可与其他任务共享 GPU，只以可用显存为门槛 | 操作者明确批准高利用率共卡；单 worker 稳定占用约 838 MiB | accepted | 默认门槛 2,048 MiB；允许繁忙卡和同卡多 worker，不干预其他进程；OOM/磁盘/source gate 仍触发停止 |
| `DEC-017` | 2026-08-11 | CSL-News source snapshot 使用唯一 JSON、portable ZIP URI、inline caption 和 clean-Git atomic finalize | JSON 722,711 个 key 唯一，CSV 有 4 条冲突重复；绝对路径和未绑定代码版本的 manifest 不可复现 | accepted | 每条 sample 绑定 archive/labels/config/commit；`zip://archive!/member` 由配置 root 解析；dirty worktree 不得生成正式 snapshot |
| `DEC-018` | 2026-08-11 | 只有完整逐 member CRC 通过的 CSL-News archive 才能进入标注、processed data 或训练 | central-directory 可读不代表压缩数据完整；首批 `005/008` 已复现 zlib failure | accepted | 原损坏 ZIP/partial artifact 保留；replacement 使用 versioned intake 并重新通过 checksum/CRC/coverage/decode gate |
| `DEC-019` | 2026-08-11 | CSL-News final ZIP promotion 必须通过 transfer、aria2 control 和完整解压/CRC 三重 gate | `archive_001` 在 aria2 HTTP 403 后被未传播退出码的旧子 shell 误晋升 | accepted | 下载器显式传播错误并在 `unzip -t` 后原子改名；已有坏 final 保留且不得自动覆盖 |
| `DEC-020` | 2026-08-11 | CSL-News 下载 promotion gate 与 annotation consumption gate 分离，消费端只读取 cumulative integrity registry 的 typed `passed` entry | final 文件名本身不足以证明 source 可用，手工 archive 清单会随下载进度漂移 | accepted | 5 分钟 clean-Git 增量扫描原子更新 registry；4 worker 按 archive ID 取模分片；每个 sidecar/marker 绑定实际 registry snapshot；failed/pending/stat-changed source 和其历史产物不得计入进度 |
| `DEC-021` | 2026-08-11 | CSL-News pose+caption 通过独立、不可变、integrity-gated manifest snapshot 暴露给后续数据链 | 持续写入的 sidecar 目录和失败 archive 历史产物不能直接作为稳定训练输入 | accepted | builder 冻结扫描开始时的 completed pair，只纳入 typed `passed` archive，校验 caption/shape/dtype/checksum 并保存 exact registry bytes；adapter 只解析 portable manifest，不通过路径替换关联数据 |

## 决策记录模板

```markdown
### DEC-XXX — Short title

- Date:
- Context:
- Options:
- Decision:
- Rationale:
- Consequences:
- Revisit Condition:
- Status: proposed / accepted / superseded
```
