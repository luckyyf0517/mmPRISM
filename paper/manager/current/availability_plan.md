# Data and Code Availability Plan

Status: `requirements_mapped_release_facts_pending`
Last Updated: `2026-08-11`
Role: `availability_and_source_data_control`

## 当前事实

- `PAPER-AUDIT-001` 确认当前有效稿件没有 Data Availability section。
- 当前有效稿件没有 Code Availability section。
- `sn-article.tex:256` 和 `:258` 的字符串仅位于注释模板，不能满足编辑要求。
- 最终文本必须位于 Methods 之后、References 之前。按当前结构，计划在 Discussion 后、
  `\backmatter`/`\bibliography` 前依次加入两个 unnumbered section；最终以编译 PDF 顺序复核。
- 目前没有可以诚实填写的 public data DOI、code release DOI/tag 或 source-data package，因此不向正文写入猜测链接。

## Data Availability 输入矩阵

| Data family | Required facts before drafting | Current status |
|---|---|---|
| CSL-News RGB/labels | canonical source、revision、license、引用、是否再分发或仅提供派生 manifest | HF revision 已固定；最终完整 manifest/处理结果与发布边界待定 |
| CSL-Daily/其他公共手语数据 | exact version、license、download URL/DOI、使用范围 | source 尚未导入 |
| 新增/历史真实 radar 数据 | participant consent、ethics approval/豁免、去标识化、可公开粒度、限制访问流程 | blocked，需作者和伦理边界确认 |
| Synthetic radar/pose | simulator/version、输入来源许可、生成配置、manifest/hash、可重建范围 | simulator provenance blocked |
| Processed pose/caption | source lineage、处理版本、checksum、许可继承、repository layout | CSL-News partial engineering snapshot 已有，final release 未冻结 |
| Paper Source Data | 每个主文/补充图表底层样本级数据，或公开 raw+code 可完全复现的豁免证据 | 20 个 display item 已登记；所有科学 provenance 待补，S2-S6 为 `placeholder_unverified` |

## Code Availability 输入矩阵

| Requirement | Evidence needed | Current status |
|---|---|---|
| Reviewer-accessible code | private review URL/archive、访问测试、固定 commit/tag | canonical rebuild in progress |
| Install/run instructions | clean-room UV bootstrap、example input、expected output | foundation 可安装；端到端路径未完成 |
| Model/evaluation assets | SBERT/SimCSE 下载、supported model matrix、checkpoint provenance | `ARCH-REV-002/004` pending |
| License | author-approved license 与第三方依赖/数据边界 | `OPS-REV-002` blocked on author decision |
| Persistent release | public repository + archived DOI/version after acceptance policy confirmation | not_started |
| Exclusions | token、private reviews、`paper/manager`、internal agent docs、未授权数据/weights | policy defined；release manifest pending |

## Draft Skeleton

以下只作为受控占位模板，不得原样进入正文：

```latex
\section*{Data Availability}
[PUBLIC DATASETS: repository/version/citation and access terms.]
[AUTHORS' RADAR DATA: public DOI or precise justified restriction and request process.]
[SOURCE DATA: exact Nature-required statement after the display-item decision.]

\section*{Code Availability}
[VERSIONED REPOSITORY/DOI and exact release tag.]
[INSTALLATION, example data/output, model weights and any justified restrictions.]
```

禁止填写 “available on request” 而不给理由、责任人、访问条件和时限。禁止在 release 未经独立访问测试前
声称代码或数据可用。

## Source Data 决策

Nature 要求为均值/表格、散点图和折线图提供底层数据；只有当所有 display item 已可由公开 raw data
和公开代码复现时，才可以采用无需单独 Source Data 文件的路径。当前采用以下 gate：

1. `PAPER-AUDIT-001` 已从 19 个 environment 识别并登记 20 个独立 display item：主文 9 个、
   supplementary 11 个，见 `evidence/display_item_registry.md`。
2. 按 registry 为每项补齐 raw/processed input、sample-level values、生成脚本、run ID、checkpoint、
   metric protocol 和 hash。Supplementary Tables S2-S6 在真实实验替换前不得提供或引用当前数字。
3. 决定单个 `Source Data.xlsx`（每图/表独立 sheet）或 labelled ZIP；不得只提供图片中的均值。
4. 在最终选择完成后同步 Data Availability 精确 statement 和 cover letter 描述。

## Closure Gate

- 两个 section 在有效 TeX 中存在，顺序为 Data 后 Code，且位于 bibliography 前。
- 所有 repository URL/DOI/tag 能从无作者凭据环境访问，受限数据流程另做真实测试。
- Data/code statements 与 README、checklists、license、release manifest 和 cover letter 一致。
- “Source Data are provided with this paper.” 只在确实提交对应 artifact 时出现。
- 审计 JSON 中 `missing_data_availability`、`missing_code_availability` 消失。
- Overleaf PDF 通过人工位置、链接、换行和修订标记检查。
