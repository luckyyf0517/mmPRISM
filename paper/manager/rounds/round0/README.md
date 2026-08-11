# Round0: Discovery and Revision Bootstrap

Status: `active`
Last Updated: `2026-08-11`
Role: `round0_control`

## Goal

在不修改科学结论、不大规模移动数据的前提下，建立返修工作的可信起点。

## Workstreams

| Workstream | IDs | Status | Output |
|---|---|---|---|
| Reviewer package intake | `PAPER-001A`, `REV-001` | done | original review + stable tracker + diagnosis |
| Manuscript intake | `PAPER-001B` | in_progress | current Overleaf manuscript linked; original submission and asset index pending |
| Asset discovery | `DATA-001` | blocked | data/checkpoint/result inventory |
| Runtime bootstrap | `OPS-001` | done | UV/Python 3.12/cu128 lock + wheel + A100 smoke |
| Greenfield foundation | `ARCH-001`, `ARCH-002` | in_progress | pyproject + strict config + contracts + CLI + CPU tests |
| Canonical smoke | `EXP-001` | blocked | new OmniHand + WaveLLM minimal artifacts |
| Management control plane | documentation bootstrap | done | `paper/manager/` |

## Known Findings

- 当前 repo 为 `master@518a402`，工作区在文档搭建前干净。
- 多个 RTM/temporal 文件可在 Git 历史找到，但当前已删除，配置仍有陈旧引用。
- 当前 README/CLAUDE 与实际源码不完全一致。
- canonical UV research profile 已通过 8 项测试、Ruff、Mypy、wheel、核心 ML import 和 A100 CUDA smoke。
- `/mnt/gfs/yanyifan` 未发现预期项目数据，且共享盘接近满载。
- Major revision decision 已收到；编辑明确要求替代架构、DA、dataset characterization 和 real-world generalization。
- 两位 reviewer 的意见已拆成 20 个科学/代码执行项，并建立 P0–P2 计划。
- 当前返修稿已作为 `paper/manuscript` Overleaf Git 子模块接入，接入快照为 `master@3242a40`，主入口为 `sn-article.tex`。
- 作者确认不保留旧训练链兼容；新实现以 `src/mmprism` 为唯一主线，legacy 代码只读取证。

## Exit Criteria

- [x] decision letter 和所有 reviewer comments 已导入并编号。
- [ ] 当前稿件与 supplementary 已接入；仍需导入原投稿定稿并完成资产登记。
- [ ] 数据、checkpoint、log、metrics 位置已确认。
- [x] UV 环境可重建且 CPU/GPU import smoke 通过。
- [ ] 至少一个 canonical OmniHand 与 WaveLLM 路径有 smoke artifact，或 blocker 有完整解释。
- [x] legacy 边界已决定为 forensic-only，不恢复到 canonical package。
- [ ] Round1 的数据 inventory 范围和存储预算已拍板。

## Handoff to Round1

Round1 只在 `BLOCK-DATA-ROOT` 关闭、数据 source inventory 可访问后开始。若 reviewer deadline 很紧，可并行启动纯文本 reviewer analysis，但不得在数据来源不明时承诺新增实验结果。
