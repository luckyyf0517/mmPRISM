# Active Issues

Status: `active_risk_register`
Last Updated: `2026-08-11`
Role: `risk_register`

| ID | Severity | Issue | Impact | Status | Next Action |
|---|---|---|---|---|---|
| `BLOCK-DATA-ROOT` | P0 | CSL-News 官方源已开始下载，但私人 collected、CSL-Daily 和历史运行资产仍未定位 | 真实数据与原投稿结果仍无法验证 | blocked | 完成 CSL-News intake；定位旧服务器、备份、对象存储或其他来源 |
| `BLOCK-MANUSCRIPT` | P0 | 当前 Overleaf 稿件已接入，但原投稿定稿和 response 未导入，表图 provenance 未登记 | 无法完成新旧稿差异、全部 claim 和 paper-facing 数值审计 | in_progress | 继续执行 `PAPER-001B` |
| `BLOCK-PROVENANCE` | P0 | 论文数值与 checkpoint/split/metrics 未建立映射 | 结果不可审计 | not_started | 建立 data/experiment/paper evidence registry |
| `BLOCK-RUNTIME` | P1 | canonical UV/Python/CUDA 环境缺失 | 曾阻断 wheel、ML 和 GPU smoke | done | UV research profile、lockfile、wheel 和 A100 smoke 已验证；后续只通过 pyproject/uv.lock 变更 |
| `OPS-STORAGE-001` | P1 | `/mnt/gfs` 可用空间从约 141 GB 动态变化到约 3.6 TB | 共享容量可能在 935 GB 下载或解压期间再次变化 | in_progress | 下载保留 1 TiB floor；不并行解压，完成后重新预算 |
| `BLOCK-SIM-PROVENANCE` | P0 | 稿件声称 MANO mesh/ray tracing，当前可见 legacy 仿真主要使用 skeleton 插值 | 无法确认原投稿 synthetic data 方法或直接复现其结果 | blocked | 上传/定位原始 MANO/mesh/simulator 输入、配置、代码和运行证据 |
| `ARCH-STALE-001` | P1 | legacy 配置引用已删除的 `cubenet_rtm.py` 等模块 | 旧发布包不可运行，但不再阻断 canonical 新实现 | superseded | 仅纳入 forensic/release exclusion audit，不恢复到新包 |
| `ARCH-CONFIG-001` | P1 | legacy 配置、入口和代码中存在大量硬编码路径/GPU/dtype | 旧代码不可迁移；canonical release 必须彻底隔离 | in_progress | strict path/runtime/run-plan 已落地，继续完成所有新入口 |
| `DATA-PATH-001` | P1 | dataset 通过路径字符串替换关联多模态文件 | 易错且不可验证 | not_started | 改为 manifest-driven sample record |
| `DATA-CSLNEWS-META-001` | P1 | 官方 CSL-News CSV 比唯一 JSON 多 4 条冲突重复行 | CSV last-write-wins 会为 4 个视频静默选择错误译文 | mitigated | 固定 JSON 为 canonical source，CSV 只作审计；保留 profile 和四个 key，必要时反馈上游 |
| `DATA-CSLNEWS-INTEGRITY-001` | P0 | 已完成 ZIP 的完整 CRC 审计确认 `archive_005`、`archive_008` 损坏 | 3,251 个视频不可作为 source，原 11-archive snapshot 不能作为 integrity-verified 数据输入 | active | 保留原件和 partial artifacts；仅调度 9 个通过 archive；人工复核后 versioned 重下 `005/008` 并重新审计 |
| `EXP-TEST-001` | P1 | legacy 无自动化测试；canonical 仅有 foundation fixture/tests | radar/model/训练链仍缺保护 | in_progress | 扩展 shape、split、processor、model smoke tests |
| `DOC-DRIFT-001` | P2 | README/CLAUDE 描述多个当前不存在的模块和脚本 | 误导执行 | evidence_ready | legacy baseline 后统一重写文档 |
| `ARCH-LLM-001` | P2 | Phi-3 路径与当前 base API 不一致，现有配置主要只支持 MT5 | 模型支持边界不清 | not_started | 决定删除、修复或降级为 unsupported |
| `REV-ARCH-001` | P0 | 两阶段架构缺少 matched direct end-to-end baseline | 核心增益归因不充分 | blocked | `EXP-REV-001` |
| `REV-DA-001` | P0 | shallow adaptation 缺少 full/adversarial/MMD 横向比较 | “最优/高效”主张不充分 | blocked | `EXP-REV-002` |
| `REV-REAL-001` | P0 | 方向、遮挡、新用户和真实多样性证据不足 | 编辑明确要求 real-world generalization | blocked | `DATA-REV-002`, `EXP-REV-003` |
| `REV-SYNREAL-001` | P0 | 合成数据与真实数据 closeness 未直接衡量 | synthetic-trained 真实性依据不足 | blocked | `DATA-REV-003`, `EXP-REV-004` |
| `REV-ATTN-001` | P0 | spatial/channel/SE 缺少 leave-one-out | 模块堆叠可能被认为任意 | blocked | `EXP-REV-005` |
| `REV-XMODAL-001` | P1 | WiFi/声学 baseline 请求存在协议不可比风险 | 错误比较会产生新公平性问题 | not_started | 先做 `EXP-REV-007` feasibility audit |
| `REV-CODE-001` | P0 | Reviewer 已明确列出硬编码、SBERT、文档、LICENSE、Phi-3 问题 | code availability 可能直接阻断返修 | in_progress | `ARCH-REV-001`–`004`, `OPS-REV-002` |
| `BLOCK-REAL-COLLECTION` | P0 | 新增参与者、方向/遮挡采集与伦理条件未知 | P0 real-world evidence 无法排期 | blocked | 作者确认资源、伦理和可采集范围 |

## 风险关闭规则

风险只有在以下条件满足后才能标记 `done`：

1. 有可验证 artifact 或命令输出。
2. 对应 task 已完成并有验收记录。
3. 若影响论文，`paper_evidence_map.md` 和 reviewer tracker 已同步。
