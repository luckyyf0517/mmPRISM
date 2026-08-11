# Code Architecture Todo

Status: `greenfield_foundation_active`
Last Updated: `2026-08-11`
Role: `architecture_execution_tracker`

| ID | Priority | Task | Status | Acceptance |
|---|---|---|---|---|
| `ARCH-001-A` | P0 | 生成当前 import/config/entrypoint dependency graph | not_started | 缺失 target 和循环依赖清单 |
| `ARCH-001-B` | P0 | 冻结 legacy forensic inventory 与公开 release 排除边界 | in_progress | archive/retain/exclude 决策记录 |
| `ARCH-001-C` | P0 | 定义 radar、pose、feature、caption 的 shape/dtype contract | in_progress | sample manifest v1 已落地；其余 tensor contract 待实现 |
| `ARCH-002-A` | P1 | 建立 `pyproject.toml` 和版本锁定方案 | done | Python 3.12、UV lock、research extras、wheel 与 A100 smoke 通过 |
| `ARCH-002-B` | P1 | 建立 strict typed config 与 validation | evidence_ready | unknown key、类型、环境路径和 protocol version 启动前校验 |
| `ARCH-002-C` | P1 | 建立统一 path/runtime/run config | in_progress | path/runtime/run-plan 已落地；正式 artifact writer 待实现 |
| `ARCH-002-D` | P1 | 建立统一 CLI 和 dry-run | in_progress | doctor/config/manifest/plan、CSL-News audit/annotate/status 已落地；train/eval/prepare 待实现 |
| `ARCH-003-A` | P1 | 实现 manifest-driven dataset adapters | in_progress | 通用 manifest validator/fixture 已落地；真实数据 adapter 待实现 |
| `ARCH-003-B` | P1 | 从明确契约实现 Processor/Simulation | not_started | CPU/GPU shape + numerical tests |
| `ARCH-004-A` | P1 | 实现 canonical OmniHand model/training/evaluation | not_started | 2-batch smoke + metric artifact |
| `ARCH-004-B` | P1 | 实现 batch-first temporal processing | not_started | contract 正确、性能报告完整 |
| `ARCH-005-A` | P1 | 清理 model factory 与 MT5/Phi-3 支持边界 | not_started | supported model matrix + tests |
| `ARCH-005-B` | P1 | 实现 canonical WaveLLM modality/fusion/LLM components | not_started | pose-only/feature-only/fused tests |
| `ARCH-006-A` | P2 | 统一 prediction/result artifact schema | not_started | distributed-safe writer/aggregator |
| `ARCH-006-B` | P2 | 建立 metric protocol versioning | not_started | legacy/current protocol 并存可识别 |
| `ARCH-007-A` | P2 | 添加 CI/static/test profiles | not_started | lint/unit/contracts on CPU |
| `ARCH-008-A` | P2 | 重写 README 与复现指南 | not_started | 文档命令通过验证 |
| `ARCH-REV-001` | P0 | 将审稿人点名脚本的本地路径改为 config/CLI | in_progress | clean-machine path smoke |
| `ARCH-REV-002` | P0 | 修复 `download_models.sh` 的 SBERT/SimCSE 完整准备 | not_started | `run_evaluation.py` smoke 通过 |
| `ARCH-REV-003` | P0 | 公共 release 文件清单、README 对齐、排除 `CLAUDE.md` 和 manager/private docs | in_progress | archive audit 无不存在路径/内部文件 |
| `ARCH-REV-004` | P0 | Phi-3 支持范围决策与实现 | blocked | runnable config/script 或移除支持声明 |
| `OPS-REV-002` | P0 | 经作者确认后添加许可证 | blocked | LICENSE 与 README/checklist 一致 |

## 重构顺序规则

严格按照 `contracts/config/runtime -> data -> radar -> OmniHand -> WaveLLM -> evaluation/artifacts -> legacy archive/release` 推进。新实现不得导入 legacy 模块；任何跨越顺序的修改都必须在 decision log 说明原因。
