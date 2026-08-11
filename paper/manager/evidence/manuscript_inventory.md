# Current Manuscript Inventory Evidence

Status: `evidence_ready_current_snapshot_only`
Last Updated: `2026-08-11`
Role: `current_manuscript_structure_and_asset_audit`
Audit ID: `PAPER-AUDIT-001`

## 证据边界

本页只证明当前 Overleaf 工作稿的文件结构和静态引用状态，不证明原投稿定稿与当前稿一致，也不证明
任何科学数值或结论。原投稿定稿仍未导入，因此 `PAPER-001B` 保持 `in_progress`。

| Item | Value |
|---|---|
| Manuscript submodule | `paper/manuscript@3242a40631ec5198e66fa8592763235c108513b2` |
| Main entry | `sn-article.tex` |
| Source fingerprint | `1f194d54b9981439a419f0716fa289e4e5b253856847cf958928d6f251a4c6a9` |
| Audit tool | `paper/manager/tools/audit_manuscript.py` |
| Tool SHA-256 | `a4bf0c903dcd97f6c9d796da6316eb4bc017419f268e6e9568233f234d19784a` |
| Machine-readable artifact | `evidence/artifacts/manuscript_inventory_v1.json` |
| Artifact SHA-256 | `823e93a3ad85eeec081001555167c76bc0bbe280db525d324e3a04c6bbd9fca9` |
| Audit schema | `mmprism.manuscript_audit.v1` |

复现命令：

```bash
uv run python paper/manager/tools/audit_manuscript.py \
  --output paper/manager/evidence/artifacts/manuscript_inventory_v1.json
```

审计器递归展开有效 `\input`/`\include`，忽略未转义 `%` 后的注释，并对 label/ref、citation/BibTeX、
图形文件和 supplementary ZIP 做结构化检查。JSON 不含时间戳或绝对路径，同一输入可产生稳定输出。

## 主稿结构

- 有效 TeX source：5 个，分别为主入口及 `chapter/1_introduction.tex` 至
  `chapter/4_discussion.tex`。
- 活跃 section 层级：12 个，包括 Introduction、Results、Methods、Discussion 和 8 个 subsection。
- 活跃 figure environment：6 个；活跃 table environment：2 个。
- 图形引用：7 个文件，全部存在；`pics/` 中另有 36 个未被当前主入口引用的历史/替代资产。
- label：15 个；ref command：7 个；无 duplicate label 或 unresolved ref。
- citation command：19 个，涉及 23 个唯一 BibTeX key；`sn-bibliography.bib` 有 32 个唯一条目，
  无 unresolved citation 或 duplicate key。
- `Data Availability` 和 `Code Availability` 均不存在于有效 section；模板注释不计入。
- 本轮发现 30 个 sober-language 命中：12 个编辑明确列举词，18 个证据敏感强表述。

### Figures

| Environment | Label | Asset | Source |
|---|---|---|---|
| Figure 1 | `fig:teaser` | `pics/ZZQ-25112636041-3780.jpg` | `chapter/1_introduction.tex:9` |
| Figure 2 | `fig:network` | `pics/network.png` | `chapter/2_results.tex:9` |
| Figure 3 | `fig:temporal` | `pics/temporal.png` | `chapter/2_results.tex:17` |
| Figure 4, composite | `fig:overall_comparison`, `fig:qualitative_demo` | `pics/overall_comparison.png`, `pics/rec_result.png` | `chapter/2_results.tex:50` |
| Figure 5 | `fig:training` | `pics/training.pdf` | `chapter/2_results.tex:135` |
| Figure 6 | `fig:llm` | `pics/llm.pdf` | `chapter/2_results.tex:154` |

这些编号是当前有效 environment 的顺序，不替代最终编译 PDF 中的编号。Figure 4 在同一 environment
中定义两个 label，必须在最终 PDF 和 source-data 映射时单独确认子图/面板关系。

### Tables

| Environment | Label | Source |
|---|---|---|
| Table 1 | `tab:related_work` | `chapter/1_introduction.tex:45` |
| Table 2 | `tab:llm_results` | `chapter/2_results.tex:163` |

每张图表仍需在 `paper_evidence_map.md` 中绑定生成脚本、数据、run、指标协议和最终文件 hash；
“文件存在”不代表 provenance 已关闭。

## Supplementary ZIP

| Item | Value |
|---|---|
| File | `paper/manuscript/supplementary/Supplementary_Information.zip` |
| SHA-256 | `f74f4eb0ac8c9e870a964e0cb5c72075cface6aa375d06ad81ba44d74d44abcd` |
| Size | 27,131,702 bytes |
| Entries | 44：1 TeX、30 PDF、12 PNG、1 JPG |
| CRC | 全条目通过，`bad_crc_entry=null` |
| Unsafe/duplicate/encrypted entries | 0 / 0 / 0 |
| Standalone entry | `mian.tex` |
| Parsed contents | 12 sections、5 figures、6 tables、5 referenced graphics |
| Missing input/graphics | 0 / 0 |
| Unreferenced media | 38 个，保留待 provenance/最终打包审计 |

`mian.tex` 很可能是 `main.tex` 的拼写错误。当前只登记，不在不知道 Overleaf/提交系统入口约定时改名。
ZIP 可完整读取并不证明其中 6 张表和 5 张图的数值来源已验证。

## 静态验收结论

以下检查通过：有效 input、图形路径、label/ref、citation/BibTeX、ZIP CRC 和 ZIP 内图形引用。

以下检查仍为 `attention_required`：

1. 增加正式 Data Availability 和 Code Availability section。
2. 按 `current/editorial_language_audit.md` 处理 30 个措辞命中。
3. 对 8 个主文 display environment 和 supplementary 11 个 display environment 建立逐项 provenance。
4. 确认 36 个主稿及 38 个 supplementary 未引用图形的 retain/remove 归属；当前不得批量删除。
5. 从 Overleaf 编译 PDF 做交叉引用、版面和颜色可访问性检查；本机仍无 TeX 工具链。
