# Revision Dashboard

Status: `round0_greenfield_foundation_active`
Last Updated: `2026-08-11`
Role: `control_panel`

本页只保留当前阶段、最高优先级、blocker 和下一步动作。

## 当前阶段

`Round0: Legacy Baseline, Asset Discovery, and Revision Bootstrap`

- 旧代码结构已完成首轮静态盘点。
- 返修管理控制面已建立。
- Major revision decision、编辑硬性要求和两位 reviewer comments 已保留原文并完成中英文结构化整理。
- Reviewer Stage 1 诊断、P0–P2 实验计划、response tracker 和 evidence ID 已建立。
- 当前返修稿已通过私有 Overleaf Git 子模块接入 `paper/manuscript`；主入口为 `sn-article.tex`。
- 稿件首轮静态扫描确认 Availability 正式章节缺失，且仍含编辑明确要求删除的夸张/首创式表达。
- clean snapshot `paper/manuscript@3242a40` 已完成 `PAPER-AUDIT-001`：主文 5 个 active TeX、
  6 figure/2 table、15 label、19 citation command 均完成结构检查；supplement 44 entries 全部通过 CRC。
- 审计定位 12 个 editor-prohibited 和 18 个 evidence-sensitive language hit；尚未在无证据时改写科学主张。
- `PAPER-AUDIT-001` v2 已从 19 个 LaTeX environment 识别 20 个 display item（主文 9、补充 11）
  并建立逐项 provenance/Source Data registry；Supplementary Tables S2-S6 明确为未验证占位数据。
- 作者已确认数据与训练从头重建；`src/mmprism` greenfield package、strict config、manifest contract、CLI 和 CPU 基础测试已开始落地。
- CSL-News 官方 RGB/labels 已锁定到 Hugging Face revision `3a060121`，935 GB compressed 下载已启动；
  其他 mmPRISM/CSL/OmniHand/WaveLLM 数据仍未到位。
- `archive_003` 的 SHA-256/CRC/722,711 条标签覆盖与视频解码审计通过；canonical RTMW3D
  worker 已在 GPU 7 持续运行，CPU/GPU/磁盘受限且当前健康。
- CSL-News metadata profile 已扫描全部 722,711 条 canonical JSON 记录；数据集类型、译文单元和
  长度统计已有部分证据，sign vocabulary、non-manual、subject/scene/split 仍待补齐。官方 CSV 的
  4 条冲突重复已隔离，不能覆盖唯一 JSON。
- 首个 CSL-News partial snapshot 的 18,095 条 schema/linkage 已验证；cumulative integrity registry
  在 `17:30Z` 覆盖 20 个 final ZIP，其中 `001/005/008` 失败并隔离，17 个 archive/27,975 videos 通过。
  5 分钟增量扫描和 4 个 registry-only dynamic annotation worker 已运行。
- clean commit `390093b` 已冻结首个 integrity-gated pose+caption partial snapshot：2,157 records、
  5 个 represented archive、15 个 failed-archive 历史 pair 明确排除；checksum/contract/adapter 验收通过。
- clean commit `eb5de64` 已为该 partial manifest 生成 sequence-disjoint split：1,701/219/237，
  2,157/2,157 coverage、0 cross-group leakage；缺少 signer，因此不作为 subject-independent 证据。
- `/mnt/gfs` 当前约余 3.4 TB，但属于共享动态容量；CSL-News 下载保留 1 TiB floor 且暂不解压。
- canonical UV 环境已锁定 Python 3.12/PyTorch 2.11 cu128；64 项测试、Ruff、Mypy、wheel、
  annotation 依赖 import、CUDA smoke 和真实 RTMW3D 视频 smoke 均通过。
- formal-run artifact writer 与 `run-init` 已实现：原子冻结 resolved config、Git/environment、命令和
  输入 SHA-256，metrics 强制显式 protocol/样本数/有限值；train/eval/prepare 服务仍待实现。
- 原投稿定稿与独立 response letter 尚未导入；当前子模块中已有主稿、章节、参考文献、图和 supplementary 压缩包。

## 当前最高优先级

1. 补充导入原投稿定稿，并按 20 项 display registry 补齐 dataset/split/run/checkpoint/metric/script/Source Data。
2. 按已建立的 sober-language 与 Availability 清单准备回写；不在新增实验完成前强化结果主张。
3. 监控 CSL-News 官方下载和夜间姿态标注，同时收集私人历史 archive/目录的名称、大小和可重下标记。
4. 确认可新增真实数据的人数、伦理边界、方向/遮挡采集条件和时间预算。
5. 在已锁定的 canonical 环境下，待真实 source 到位后完成 data/radar vertical slice 和端到端 GPU smoke。

## 当前 Blocker

- `DATA-CSLNEWS-INTEGRITY-001`：`005/008` member 解压失败，`001` 为 HTTP 403 后误晋升的
  incomplete final；promotion gate 已修复，原件保留并等待 versioned 重下验证。
- `BLOCK-DATA-ROOT`：CSL-News source 已进入下载，但私人 collected、CSL-Daily 和历史 run 仍未知。
- `BLOCK-SIM-PROVENANCE`：稿件 MANO mesh/ray-tracing 描述与当前可见 skeleton 仿真路径不一致。
- `BLOCK-RADAR-PROVENANCE`：稿件与 legacy 的 chirp、带宽、clutter、阵列规模和共轭约定冲突；
  canonical range-Doppler 已独立验收，但 beamforming/物理坐标必须等待 acquisition/calibration 证据。
- `BLOCK-MANUSCRIPT`：当前稿/supplement 的 19 个 environment/20 个 item 已登记，但原投稿定稿与
  response letter 缺失，科学 provenance 尚未闭合；Supplementary Tables S2-S6 必须由真实实验重建。
- `BLOCK-PROVENANCE`：历史 checkpoint、split 和 paper-facing 数值尚未建立对应关系。
- `BLOCK-REAL-COLLECTION`：新增参与者、方向/遮挡 protocol 和伦理/同意范围尚未确认。

## 下一步动作

1. 继续执行 `PAPER-001B`，登记当前 manuscript/supplement，并导入原投稿定稿。
2. 执行 `DATA-001-K/DATA-005-A`，下载 promotion 和标注前都做完整 ZIP gate；只对通过清单生成 pose。
3. 继续 `DATA-REV-001/002`：把 CSL-News profile 绑定到 frozen manifest，并确认真实采集的
   subject/non-manual/scene/split 与伦理资源。
4. 继续 `ARCH-001/002`，在已完成的 formal-run writer 上实现 canonical prepare/train/evaluate CLI。
5. 为 `ARCH-003` 收集 radar acquisition/channel/calibration 证据；绑定真实 manifest fixture 后实现
   beamforming 与完整 cube gate，禁止用 legacy 默认值填补未知项。

## Source Of Truth

- 总路线：`current/roadmap.md`
- 架构：`current/architecture_status.md`
- 数据：`current/data_status.md`
- 上传清单：`data_upload_checklist.md`
- 风险：`current/issues.md`
- 任务：`tasks/todo.md`
- 决策：`decisions/decision_log.md`
- 审稿原文：`reviews/review_en.md`
- 中文整理：`reviews/review_cn.md`
- 审稿诊断：`reviews/analysis.md`
