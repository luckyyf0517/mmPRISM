# Code Architecture Todo

Status: `greenfield_foundation_active`
Last Updated: `2026-08-11`
Role: `architecture_execution_tracker`

| ID | Priority | Task | Status | Acceptance |
|---|---|---|---|---|
| `ARCH-001-A` | P0 | 生成当前 import/config/entrypoint dependency graph | done | clean `c49f725` snapshot：33 modules/51 edges，missing/legacy/relative/cycle 0，entrypoint matched |
| `ARCH-001-B` | P0 | 冻结 legacy forensic inventory 与公开 release 排除边界 | in_progress | tracked allowlist/forbidden boundary 已实现；待最终 release artifact 和历史 evidence 完成后归档 legacy |
| `ARCH-001-C` | P0 | 定义 radar、pose、feature、caption 的 shape/dtype contract | evidence_ready | explicit axes/dtype/finite/value/unit/coordinate-frame validators 与 9 项 contract tests 通过 |
| `ARCH-002-A` | P1 | 建立 `pyproject.toml` 和版本锁定方案 | done | Python 3.12、UV lock、research extras、wheel 与 A100 smoke 通过 |
| `ARCH-002-B` | P1 | 建立 strict typed config 与 validation | evidence_ready | unknown key、类型、环境路径和 protocol version 启动前校验 |
| `ARCH-002-C` | P1 | 建立统一 path/runtime/run config | evidence_ready | path/runtime/run-plan 与原子 formal-run writer 已落地；resolved config、Git/environment、命令、输入 hash、metric protocol/status contract 通过 |
| `ARCH-002-D` | P1 | 建立统一 CLI 和 dry-run | in_progress | OmniHand train/evaluate 与既有 doctor/config/manifest/plan/run-init/split/release/model/CSL-News 命令已落地；prepare、WaveLLM formal train/eval 待实现 |
| `ARCH-003-A` | P1 | 实现 manifest-driven dataset adapters | in_progress | CSL-News pose+caption adapter 与 dependency-light SplitIndex 已通过真实 snapshot；其他数据族/radar adapter 待实现 |
| `ARCH-003-B` | P1 | 从明确契约实现 Processor/Simulation | in_progress | NumPy range-Doppler 与 17 项 analytic/error tests 通过；beamforming/physical axes/simulation 等 provenance 后实现 |
| `ARCH-004-A` | P1 | 实现 canonical OmniHand model/training/evaluation | evidence_ready | 模型 smoke 与 clean commit `81e9b89` A100/BF16 formal train/checkpoint/reload/evaluate、prediction replay 和独立审计均通过；待真实 manifest |
| `ARCH-004-B` | P1 | 实现 batch-first temporal processing | evidence_ready | 10-frame vectorized encoder、8-layer/16-head mask-aware transformer、single-frame path、padding counterfactual和显存/吞吐报告通过 |
| `ARCH-005-A` | P1 | 清理 model factory 与 MT5/Phi-3 支持边界 | evidence_ready | mT5-only boundary、pinned asset、canonical wrapper、两步 train/generate GPU smoke 和 unsupported-backend release gate 均通过；`EVID-CODE-MT5-SMOKE-V1` |
| `ARCH-005-B` | P1 | 实现 canonical WaveLLM modality/fusion/LLM components | in_progress | 双手 ST-GCN、radar projector、confidence-aware fusion 和 tiny/real-mT5 integration 已通过；待 production pose/feature/fused train-eval |
| `ARCH-006-A` | P2 | 统一 prediction/result artifact schema | in_progress | OmniHand single-rank streaming JSONL/atomic register 已通过；待 distributed-safe writer/aggregator 和 WaveLLM schema |
| `ARCH-006-B` | P2 | 建立 metric protocol versioning | in_progress | pose v1 count-weighted accumulator 与 summary 已统一；待 legacy/current 显式并存和 language metrics |
| `ARCH-007-A` | P2 | 添加 CI/static/test profiles | not_started | lint/unit/contracts on CPU |
| `ARCH-008-A` | P2 | 重写 README 与复现指南 | not_started | 文档命令通过验证 |
| `ARCH-REV-001` | P0 | 将审稿人点名脚本的本地路径改为 config/CLI | in_progress | clean-machine path smoke |
| `ARCH-REV-002` | P0 | 修复 `download_models.sh` 的 SBERT/SimCSE 完整准备 | evidence_ready | fixed HF commits、14-file checksum manifest、atomic/reuse gates 和两个真实 loader CPU smoke 通过；artifact `evaluation_models_smoke_v1.json` |
| `ARCH-REV-003` | P0 | 公共 release 文件清单、README 对齐、排除 `CLAUDE.md` 和 manager/private docs | in_progress | README 与 allowlist audit 已落地；model downloader/config/smoke 均已验证，待 LICENSE 和 radar example 后 reviewer profile 全绿 |
| `ARCH-REV-004` | P0 | Phi-3 支持范围决策与实现 | evidence_ready | `DEC-027` 选择移除 public support；66-file clean release 零命中，legacy 类排除且 content gate 回归测试通过 |
| `OPS-REV-002` | P0 | 经作者确认后添加许可证 | blocked | LICENSE 与 README/checklist 一致 |

## 重构顺序规则

严格按照 `contracts/config/runtime -> data -> radar -> OmniHand -> WaveLLM -> evaluation/artifacts -> legacy archive/release` 推进。新实现不得导入 legacy 模块；任何跨越顺序的修改都必须在 decision log 说明原因。
