# Revision Decision Log

Status: current
Owner: mmPRISM coordinator
Authority scope: Cross-workspace mmPRISM rules, contracts, architecture, operations, or decisions defined by this page.
Last reviewed: 2026-08-12

| ID | Date | Decision | Rationale | Status | Consequence |
|---|---|---|---|---|---|
| `DEC-001` | 2026-08-11 | 使用 `paper/manager` 作为返修管理控制面 | 与参考项目一致，集中管理 reviewer、架构、数据和 evidence | superseded by `DEC-039` | 旧路径只保留兼容入口，历史内容已迁入 Authority/Logs |
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
| `DEC-022` | 2026-08-11 | canonical split 绑定 exact manifest hash，并以匿名稳定 group ID 和 SHA-256 整数权重分配 | 文件顺序、浮点随机数和机器路径无法提供跨运行稳定且可审计的 split；原始 group value 也不应无必要暴露 | accepted | assignments 只保存 sample_id/group_id/split；builder 强制 coverage/group-disjoint/clean-Git/atomic/checksum gate；partial source 只能生成 partial split，sequence split 不得替代 subject-independent 证据 |
| `DEC-023` | 2026-08-11 | 所有 canonical formal run 先通过统一 writer 原子冻结配置、环境/Git、命令和具名输入哈希 | 仅保存 logger/checkpoint 或由各训练脚本自由落盘无法满足 paper evidence 审计 | accepted | `run-init` 只建立 provenance envelope；metrics 必须绑定 protocol/sample count 且为有限数值；完成状态要求 metrics；distributed prediction/checkpoint writer 单独实现 |
| `DEC-024` | 2026-08-11 | canonical radar v1 先冻结显式张量轴并只实现 NumPy range-Doppler，beamforming 等 provenance 恢复后再实现 | 稿件与 legacy 在 chirp 数、带宽、clutter 顺序、116/86 阵列和 `A^H` 共轭上冲突 | accepted | 新处理器不硬编码 64/32，保留复数精度；beamforming、物理轴和 simulation 保持 blocked，不能据此声称完整 4D cube 已复现 |
| `DEC-025` | 2026-08-11 | reviewer release 从 Git tracked allowlist 构建并自动审计，不直接复制开发仓库 | legacy、CLAUDE、manager、manuscript 和私有运维配置仍需取证但不得进入公开包 | accepted | 每个 selected file 绑定 hash；检查 required/forbidden path、绝对路径/凭据、entrypoint、missing/legacy import 和 cycle；LICENSE/model download/radar+mT5 examples 缺失时 release 必须失败 |
| `DEC-026` | 2026-08-11 | evaluator 模型通过 fixed HF commit、selected-file checksum manifest 和原子晋升管理 | moving `main` 与手写 wget 文件清单不能证明下载完整，历史脚本也遗漏 SBERT | accepted | canonical `scripts/download_models.sh` 同时准备 SimCSE/SBERT；坏 final 不覆盖，重复运行先复验；权重不进入 reviewer Git archive，按上游 license 下载 |
| `DEC-027` | 2026-08-11 | canonical caption-generation 只以 mT5 为重建目标，Phi-3 不列为 supported backend | legacy Phi-3 只有底层类，无 runnable config/train/eval/checkpoint evidence；greenfield 重构无需保留伪支持 | accepted | reviewer release 排除 legacy 类和 claim，并以 content gate 防止回流；只有完成 canonical contract、配置、训练/生成/评测与 provenance 后才可重新提案 |
| `DEC-028` | 2026-08-11 | OmniHand model-ready 输入使用 checksum-bound manifest，训练产物统一由 formal-run writer 登记 | 模型内解析路径或各训练脚本自由落盘会重新引入硬编码、模态错配和不可审计 checkpoint | accepted | model 只接收 tensor；adapter 严格校验相对 `.npy`、单位/坐标与 mask；checkpoint/prediction/config/history 由 orchestration 原子写入并绑定 run provenance |
| `DEC-029` | 2026-08-11 | 已发布 pose sidecar/NPZ identity conflict 只能通过 clean-run 全量审计绑定的显式排除处理 | 覆盖坏 sidecar 会销毁 incident evidence，直接放宽 manifest checksum gate 会掩盖其他损坏 | accepted | 原 pair、失败 snapshot 和 failure records 永久保留；排除项必须同时匹配 sample/archive、sidecar SHA、声明/实际 NPZ identity、audit report SHA 和 clean Git commit，证据复制进 snapshot `SHA256SUMS`；任一漂移硬失败 |
| `DEC-030` | 2026-08-11 | CSL-News source 恢复采用不可变 replacement overlay、registry 精确路径绑定和 source-versioned annotation artifact | 直接覆盖 primary ZIP 或按 archive ID 复用旧 pose 会销毁事故证据，并可能把不同 source bytes 静默混入同一训练集 | accepted | primary/旧产物永久保留；v2 entry 显式记录 `archive_path_relative/source_kind/SHA`；resume 绑定 archive+labels+member identity；不匹配产物用完整 source SHA 后缀共存；manifest 唯一选择 current-source sidecar 并输出 checksum-covered quarantine ledger |
| `DEC-031` | 2026-08-11 | CSL-News source manifest v2 只从一个冻结的 source-integrity v2 registry 构建 | glob primary root 无法表达 replacement，并会重新选中已知坏原件；live registry 更新也不能改变已发布 manifest | accepted | 配置必须注入 registry；逐 entry 校验 exact path/source kind/stat/SHA/video count/audit；snapshot 复制 registry 原始字节并以 `SHA256SUMS` 覆盖 registry/manifest/summary；v1 snapshot 仅保留为历史 linkage evidence |
| `DEC-032` | 2026-08-11 | formal train/evaluate 必须注册一个 canonical split assignment，并在写入前支持统一无副作用 preflight | 仅记录 manifest 或接受 CLI `--split` 标签无法证明样本真实属于声明 split，且失败到运行中才发现会浪费算力并留下无效目录 | accepted | `mmprism prepare` 在 clean Git 上验证配置、输入哈希、目标、manifest/split contract、全覆盖和无重叠且不写目录；OmniHand/WaveLLM 运行时重复 membership gate，并把 split 文件写入 `inputs.json` |
| `DEC-033` | 2026-08-11 | 已发布 canonical pose 冲突通过确定性的 full-source-SHA variant 恢复，原文件永久只读 | 覆盖或删除 canonical pair 会销毁 incident evidence，仅排除又会永久丢失可恢复样本 | accepted | recovery 只在 canonical 当前来源不完整/不一致时路由到 `--source_<archive-sha256>`；resume 和 status 只接受唯一有效 variant；manifest 必须继续验证并复制 canonical exclusion evidence，同时纳入恢复 record；多有效 variant 硬失败 |
| `DEC-034` | 2026-08-11 | prediction artifact 采用 rank-local immutable shard/receipt 与 rank-zero exact-coverage 聚合 | 多 rank 并发写最终 JSONL 或共享 `run.json` 会损坏 provenance；全量 pose payload 内存排序不可扩展 | accepted | rank 只做 no-clobber publish；rank 0 验证完整 rank/checksum/schema/ID coverage，以 SQLite 确定性排序并一次性登记 shard、receipt、index 和 final JSONL；原 shard 永久保留 |
| `DEC-035` | 2026-08-12 | canonical v1 训练恢复限定为 single-process completed-epoch exact resume | 把普通权重加载称为 resume 会丢失 optimizer/RNG/history；mid-epoch 与 DDP 状态需要不同的分片和 sampler 契约 | accepted | 每个完整 epoch 发布 immutable strict JSON/Safetensors 状态，精确绑定 Git/data/split/model/runtime 并恢复模型、AdamW、GradScaler、全部 RNG、loader、history/global step；仅训练目标可保持或增加，partial epoch 从上一完整边界重跑；DDP/checkpoint aggregation 继续独立实施 |
| `DEC-037` | 2026-08-12 | CSL-News RTMW3D 标注从 4 lane 扩展到 2 GPU 上的 8 个互斥 lane | GPU 5 空闲约 81 GiB；原 GPU 7 的 4 lane 健康但约 1.86k samples/hour。重新分片必须避免同时消费同一 archive | accepted | 先停止旧 `mod 4` pool，再启动 GPU 7 lane 0--3、GPU 5 lane 4--7 的 `archive_id % 8` pool；已发布 pair 由 source identity/config/member/artifact checksum 复验跳过；扩容不得中断下载/integrity 或清理任何数据。首段 157 pair/约 3 min、0 新 failure；稳定吞吐由后续 status report 确认 |
| `DEC-038` | 2026-08-12 | final training delivery 采用 immutable task-specific Parquet：每 row 一个 sample、每 part <=1,024 rows、每 split-homogeneous chunk <=64 parts | 作者要求统一 final training format，但当前 OmniHand 和 WaveLLM 模态/单位契约不同；现有 CSL-News visual pose 尚无 calibrated radar input | accepted | sidecar/NPZ 继续作为 interim forensic layer；frozen manifest/split 是唯一 materializer input；Pose Reconstruction 与 SLU 分立 schema，typed Arrow nested lists/ZSTD/no opaque blobs；delivery 需 capacity/inventory/checksum/reader parity/adapter smoke 后才可 formal train。CSL-News annotation 留在同一 repo，live run 后再模块化，不拆 workspace |
| `DEC-039` | 2026-08-12 | 项目 current truth 使用 project Authority 加五个轻量业务 workspace | 原控制面混合 current status、任务、运行证据和论文治理，日常同步成本过高；workspace 应表示业务流程而非 Python package | accepted | `docs/authority/00_INDEX.md` 管跨 workspace 真值；每个 workspace 只用一个 index 汇总当前状态；共享代码保留根目录；`paper/manager` 和 `docs/architecture` 旧文档改为兼容入口 |
| `DEC-040` | 2026-08-12 | 新建 semantic sign-language collection workspace，目标约 30 名可用参与者 | 旧自采数据全部是无语义手势，不能支撑连续手语翻译、数据集透明度或新参与者语义测试；新采集需要独立管理参考内容、同步录制和基本 QC | accepted | 旧数据不计入新语义 cohort；约 30 人完成核心有语义录制；边界场景采用紧凑子集；采集输出冻结后交给 data rebuild，不在采集 workspace 建共享代码；参与者构成由 `DEC-042` 补充 |
| `DEC-041` | 2026-08-12 | 新版真实语义数据的主要手语语种确定为中文手语（CSL） | 项目已有 CSL-News/CSL-Daily 语义资源，作者确认新采集也以 CSL 为主，避免继续把目标语种作为完全开放问题 | accepted | 冻结的参考视频、中文含义和数据统计均以 CSL 为主；不得把历史无语义动作或普通志愿者的自由动作重新标为 CSL；具体参考集在试采前冻结 |
| `DEC-042` | 2026-08-12 | 新版 CSL 采集采用少量专业人员加大量视频学习志愿者的轻量方案 | 约 30 名专业手语者难以招募，返修需要优先形成可执行的真实雷达数据；作者预计仅可能找到 3--4 名专业/熟练人员，且能否找到尚不确定 | accepted | 专业/熟练人员尽力用于示范、检查和少量参考采集但不作为开工门槛；其余参与者观看冻结 CSL 视频后复现；只统计实际有效录制，不管理报名漏斗；论文必须区分专业/熟练 signer 与 video-guided volunteer，不把后者表述为自然手语用户泛化 |

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
