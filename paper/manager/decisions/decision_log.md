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
