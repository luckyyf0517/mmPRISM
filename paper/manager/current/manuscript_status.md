# Manuscript and Response Status

Status: `current_display_registry_active_original_submission_pending`
Last Updated: `2026-08-11`
Role: `manuscript_status_source_of_truth`

## 当前状态

| Asset | Expected Location | Status | Action |
|---|---|---|---|
| Original submitted manuscript | `paper/manager/original_submission/` | not_started | 导入 PDF/TeX 和提交日期 |
| Decision letter | `paper/manager/reviews/decision_letter.md` | done | 已提取决定和 mandatory requirements |
| Reviewer comments EN | `paper/manager/reviews/review_en.md` | done | 已按 R1/R2/R2-CODE 编号并保留原文 |
| Reviewer comments CN | `paper/manager/reviews/review_cn.md` | done | 已完成中文工作版 |
| Current revised manuscript | `paper/manuscript/` | evidence_ready_snapshot | Overleaf 子模块已接入并完成可重复静态 inventory；等待原投稿对照和正文回写 |
| Response letter | `paper/manuscript/response_letter.*` | not_started | evidence ready 后按 Direct Answer → Evidence → Revision 撰写 |
| Figures and tables | `paper/manuscript/pics/` | in_progress | 当前 19 个 environment/20 个 display item 已登记；科学 provenance 仍待补齐 |

## 当前 Overleaf 快照

| Item | Value | Verification |
|---|---|---|
| Submodule | `paper/manuscript` | `master@3242a40`（接入时快照） |
| Main document | `sn-article.tex` | 包含 `chapter/1_introduction.tex` 至 `chapter/4_discussion.tex` |
| Bibliography | `sn-bibliography.bib` | 由主稿引用 |
| Supplementary asset | `supplementary/Supplementary_Information.zip` | 44 个 entry 与内部依赖已登记；display provenance 待补 |
| Static audit | `PAPER-AUDIT-001` | v2 inventory 已生成并通过重复性/结构检查；20 个稳定 display ID 已建立 |
| Local compile | unavailable | 当前机器没有 `latexmk`, `pdflatex`, `bibtex` |
| Remote compile | Overleaf | 需在 Overleaf Menu 再确认 Main document 为 `sn-article.tex` |

远端访问凭据只保存在根目录 `.env`，不得进入本文件、`.gitmodules` 或 remote URL。

## 首轮稿件静态审计

接入后只做了只读扫描，尚未修改 Overleaf 正文：

1. `Data Availability` 和 `Code Availability` 目前只出现在注释模板中，正式章节缺失；处置和 closure gate 见 `availability_plan.md`。
2. 有效正文有 30 个语言审计命中：12 个编辑明确列举词、18 个证据敏感强表述；逐项位置和处理规则见 `editorial_language_audit.md`。
3. 两阶段必要性、真实鲁棒性和 synthetic-real 接近程度已有强结论性表述，但需要绑定本轮新增对照实验后再决定保留或降级。
4. 主文 Methods 已出现 4D cube 维度与符号；需对照 `R1-6` 检查单位、tensor layout 和首次出现位置，并在 response 中明确本轮修订。
5. 有效主稿包含 5 个 TeX source、12 个 section 层级、6 个 figure environment 和 2 个 table
   environment；其中一个 figure environment 有两个 caption，因此合计 7 个 figure item、2 个 table
   item。15 个 label、19 个 citation command、7 个图形和 23 个唯一 citation key 均解析成功。
6. `sn-article-bak.tex` 是 Springer Nature 模板备份，不是当前主稿；最终 submission/code archive 前应确认是否移除。
7. supplementary ZIP 的 44 个 entry/CRC/内部图形引用已通过静态审计；入口 `mian.tex` 疑似拼写错误，
   11 个 display item 的科学 provenance 尚未验证。Supplementary Tables S2-S6 还含明确“替换为真实数据”
   注释，当前数字统一为 `placeholder_unverified`。

完整 evidence：`evidence/manuscript_inventory.md`；机器可读 artifact：
`evidence/artifacts/manuscript_inventory_v2.json`；逐项控制表：`evidence/display_item_registry.md`。
主文静态审计没有发现 missing input/graphic、duplicate
label、unresolved ref 或 unresolved citation。

对应任务：`PAPER-001B`、`PAPER-REV-001`、`OPS-REV-001` 和 `PAPER-003`。

## 导入后首轮工作

1. 从原始审稿意见建立稳定 ID，例如 `AE-1`、`R1-1`。
2. 对每条意见标注 `text_logic / evidence_gap / asset_only / reproducibility / closed`。
3. 建立 original ask、planned action、experiment ID、evidence、manuscript location 和 response status。
4. 当前 9 个主文和 11 个 supplementary display item 已登记；逐项补齐 dataset、split、run、
   checkpoint、metric、生成脚本和 Source Data，原投稿导入后再做差异审计。
5. 对正文中的强 claim、数据规模、split、指标和模型命名做第一次 provenance scan。

## Manuscript Ready Gate

章节只有在以下条件满足时才能标记 `writeback_ready`：

- 相关 reviewer item 已有明确 action。
- 所需证据已 `evidence_ready`，或仅为诚实边界/文字修改。
- 数值和表图编号已绑定 evidence registry。
- 不包含未解释 placeholder 或旧架构表述。
