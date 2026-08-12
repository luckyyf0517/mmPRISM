# Nature Communications Compliance Todo

Status: historical
Owner: Paper revision lane
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12

| ID | Requirement | Deliverable | Owner | Status | Verification |
|---|---|---|---|---|---|
| `ED-COMP-1` | Code/software checklist | completed checklist PDF/doc | TBD | not_started | author sign-off |
| `ED-COMP-2` | Reviewer-assessment code | clean archive/repo, install/run README, example data/output | TBD | in_progress | clean-room smoke |
| `ED-COMP-3` | Machine learning checklist | completed ML checklist | TBD | not_started | author sign-off |
| `ED-COMP-4` | Colour-vision accessibility | audited and recoloured figures | TBD | not_started | palette + PDF visual audit |
| `ED-COMP-5` | Data Availability section | manuscript section after Methods | TBD | in_progress | `PAPER-AUDIT-001` 确认 active section 缺失；按 `current/availability_plan.md` 补 repository/restriction/source-data 事实后写入 |
| `ED-COMP-6` | Code Availability section | manuscript section after Data Availability | TBD | in_progress | `PAPER-AUDIT-001` 确认 active section 缺失；待 release URL/tag/license/clean-room smoke 后写入 |
| `ED-COMP-7` | Persistent data repository | repository/figshare decision, upload record, DOI when available | TBD | blocked | data inventory/license/consent |
| `ED-COMP-8` | Source Data | labelled Excel/zip or public-reproducibility exemption | TBD | in_progress | 19 个 environment/20 个 display item 已登记；逐项 artifact 待建，Supplementary Tables S2-S6 当前为 `placeholder_unverified` |
| `ED-COMP-9` | Source Data statement | exact statement when applicable | TBD | blocked | Data Availability review |
| `ED-COMP-10` | ORCID | corresponding-author account linkage | corresponding authors | not_started | account confirmation |
| `ED-COMP-11` | Author changes | signed form if author list changes | corresponding authors | conditional | final author-list comparison |
| `ED-COMP-12` | Current references | literature search and bibliography audit | TBD | not_started | final reference scan |
| `ED-SUBMIT-1` | Revised manuscript | tracked/colour-highlighted file | TBD | blocked | compile + visual check |
| `ED-SUBMIT-2` | Supplementary files | final supplement package | TBD | blocked | 当前 ZIP 44 entries/CRC/5 graphics 通过；入口命名、11 个 display provenance、compile/visual 仍待检查 |
| `ED-SUBMIT-3` | Point-by-point response | comments verbatim + evidence-grounded responses | TBD | blocked | closure audit |
| `ED-SUBMIT-4` | Cover letter | revision summary + Source Data description | TBD | not_started | author/editor review |

## Code Release Exclusions

Reviewer code archive 默认排除：

- `CLAUDE.md`
- `paper/manager/`
- 私密 manuscript/review materials
- logs/checkpoints not explicitly included
- author-home links、tokens、private paths

排除内部文件不能代替 README 修复；公开 archive 中的每个命令和路径仍必须实际存在并通过 clean-room test。
