# Manuscript and Response Status

Status: `decision_reviews_and_current_manuscript_linked`
Last Updated: `2026-08-11`
Role: `manuscript_status_source_of_truth`

## 当前状态

| Asset | Expected Location | Status | Action |
|---|---|---|---|
| Original submitted manuscript | `paper/manager/original_submission/` | not_started | 导入 PDF/TeX 和提交日期 |
| Decision letter | `paper/manager/reviews/decision_letter.md` | done | 已提取决定和 mandatory requirements |
| Reviewer comments EN | `paper/manager/reviews/review_en.md` | done | 已按 R1/R2/R2-CODE 编号并保留原文 |
| Reviewer comments CN | `paper/manager/reviews/review_cn.md` | done | 已完成中文工作版 |
| Current revised manuscript | `paper/manuscript/` | in_progress | Overleaf 子模块已接入；继续做章节、表图和 claim inventory |
| Response letter | `paper/manuscript/response_letter.*` | not_started | evidence ready 后按 Direct Answer → Evidence → Revision 撰写 |
| Figures and tables | `paper/manuscript/pics/` | in_progress | 资产已接入，待建立 figure/table inventory 和 provenance |

## 当前 Overleaf 快照

| Item | Value | Verification |
|---|---|---|
| Submodule | `paper/manuscript` | `master@3242a40`（接入时快照） |
| Main document | `sn-article.tex` | 包含 `chapter/1_introduction.tex` 至 `chapter/4_discussion.tex` |
| Bibliography | `sn-bibliography.bib` | 由主稿引用 |
| Supplementary asset | `supplementary/Supplementary_Information.zip` | 已存在，内容尚未登记 |
| Local compile | unavailable | 当前机器没有 `latexmk`, `pdflatex`, `bibtex` |
| Remote compile | Overleaf | 需在 Overleaf Menu 再确认 Main document 为 `sn-article.tex` |

远端访问凭据只保存在根目录 `.env`，不得进入本文件、`.gitmodules` 或 remote URL。

## 首轮稿件静态审计

接入后只做了只读扫描，尚未修改 Overleaf 正文：

1. `Data Availability` 和 `Code Availability` 目前只出现在注释模板中，正式章节缺失；这是编辑明确的提交阻断项。
2. 摘要、Results 和 Discussion 仍含 `paving the way`、`remarkable`、`superior`、`paradigm` 等需按编辑要求收敛的表达。
3. 两阶段必要性、真实鲁棒性和 synthetic-real 接近程度已有强结论性表述，但需要绑定本轮新增对照实验后再决定保留或降级。
4. 主文 Methods 已出现 4D cube 维度与符号；需对照 `R1-6` 检查单位、tensor layout 和首次出现位置，并在 response 中明确本轮修订。
5. 主稿当前引用约 14 个图形、21 个 label 和 19 个 citation command；这些数字仅用于 intake，仍需建立逐项 inventory。
6. `sn-article-bak.tex` 是 Springer Nature 模板备份，不是当前主稿；最终 submission/code archive 前应确认是否移除。
7. `supplementary/Supplementary_Information.zip` 仍是未展开资产，尚未完成内容、版本和 provenance 审计。

对应任务：`PAPER-001B`、`PAPER-REV-001`、`OPS-REV-001` 和 `PAPER-003`。

## 导入后首轮工作

1. 从原始审稿意见建立稳定 ID，例如 `AE-1`、`R1-1`。
2. 对每条意见标注 `text_logic / evidence_gap / asset_only / reproducibility / closed`。
3. 建立 original ask、planned action、experiment ID、evidence、manuscript location 和 response status。
4. 把每张原投稿表/图登记到 `evidence/paper_evidence_map.md`。
5. 对正文中的强 claim、数据规模、split、指标和模型命名做第一次 provenance scan。

## Manuscript Ready Gate

章节只有在以下条件满足时才能标记 `writeback_ready`：

- 相关 reviewer item 已有明确 action。
- 所需证据已 `evidence_ready`，或仅为诚实边界/文字修改。
- 数值和表图编号已绑定 evidence registry。
- 不包含未解释 placeholder 或旧架构表述。
