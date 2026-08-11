# Editorial Decision and Mandatory Requirements

Status: `major_revision_invited`
Last Updated: `2026-08-11`
Role: `editorial_requirements_source_of_truth`

## 1. Submission Metadata

- Manuscript ID: `NCOMMS-26-006246-T`
- Title: `Geometry-guided Wireless Perception for Sign Language Understanding`
- Venue: `Nature Communications`
- Editor: `Dr Emanuele Poliani, Senior Editor (Engineering)`
- Decision received: `2026-08-11`
- Decision: `Major Revision`
- Planning target: `2026-11-11`（按邮件“within three months”推算；不是硬性拒收日期，如需显著延期应联系编辑）

完整原始邮件：`raw/decision_email_20260811_redacted.md`。

## 2. Verbatim Decision Core

> Thank you again for submitting your manuscript "Geometry-guided Wireless Perception for Sign Language Understanding" to Nature Communications. We have now received reports from 2 reviewers and, after careful consideration, we have decided to invite a major revision of the manuscript.

> As you will see from the reports copied below, the reviewers raise important concerns. We find that these concerns limit the strength of the study, and therefore we ask you to address them with additional work. Without substantial revisions, we will be unlikely to send the paper back to review. In particular, the reviewers request substantially stronger validation of the proposed framework, including comparisons against alternative architectures and domain adaptation strategies, clearer dataset characterization and evidence of real-world generalization.

## 3. Scientific and Writing Directives

| ID | Editorial Requirement | Status | Linked Reviewer Items |
|---|---|---|---|
| `ED-SCI-1` | substantially stronger validation of the proposed framework | not_started | `R1-3`, `R1-4`, `R1-5`, `R2-1`–`R2-6` |
| `ED-SCI-2` | comparisons against alternative architectures | not_started | `R2-1` |
| `ED-SCI-3` | comparisons against domain adaptation strategies | not_started | `R2-2` |
| `ED-SCI-4` | clearer dataset characterization | not_started | `R1-4`, `R2-4` |
| `ED-SCI-5` | evidence of real-world generalization | not_started | `R1-3`, `R1-4`, `R1-5`, `R2-3` |
| `ED-WRITE-1` | remove novelty/primacy language such as new, novel, first, unique, unprecedented | not_started | whole manuscript |
| `ED-WRITE-2` | remove exaggerated/subjective language such as superior, remarkable, pave the way/open new avenues | not_started | whole manuscript |
| `ED-WRITE-3` | point-by-point response must reproduce reviewer comments verbatim | not_started | response letter |
| `ED-WRITE-4` | manuscript changes must use track changes or colour highlighting | not_started | revised manuscript |
| `ED-WRITE-5` | requests not fulfilled must be explicitly justified | not_started | response letter |

## 4. Mandatory Policy and Submission Checklist

| ID | Requirement | Deliverable | Status |
|---|---|---|---|
| `ED-COMP-1` | Complete code/software checklist | completed checklist | not_started |
| `ED-COMP-2` | Make code available for reviewer assessment | reviewer-ready archive, install/run instructions, example data/output | in_progress |
| `ED-COMP-3` | Complete machine learning checklist | completed ML checklist | not_started |
| `ED-COMP-4` | Use colour-vision-accessible figures | visual audit and recoloured assets where needed | not_started |
| `ED-COMP-5` | Add Data Availability after Methods | manuscript section with restrictions/access conditions | not_started |
| `ED-COMP-6` | Add Code Availability after Data Availability | manuscript section with access conditions | not_started |
| `ED-COMP-7` | Deposit new data in persistent repository where feasible | repository/figshare plan and DOI when available | not_started |
| `ED-COMP-8` | Provide Source Data for displayed means/tables/plots, unless fully reproducible from public raw data/code | `Source Data` Excel/zip or documented exemption | not_started |
| `ED-COMP-9` | Include “Source Data are provided with this paper.” when applicable | Data Availability statement | not_started |
| `ED-COMP-10` | Corresponding authors link ORCID before acceptance | account confirmation | not_started |
| `ED-COMP-11` | Use author-list change form if authors change | signed approval form when applicable | conditional |
| `ED-COMP-12` | Keep references current and discuss relevant literature | bibliography audit | not_started |

## 5. Required Resubmission Files

- [ ] Revised manuscript
- [ ] Supplementary files
- [ ] Point-by-point response with comments reproduced verbatim
- [ ] Cover letter to the editor
- [ ] Code/software checklist
- [ ] Machine learning checklist
- [ ] Source Data or documented public-repository reproducibility route
- [ ] Other conditional forms, including author changes if applicable

## 6. Decision Interpretation

这是一次有明确返修机会、但需要实质性新增证据的 major revision。编辑没有指出方法存在致命正确性错误；决定性风险集中在：架构必要性、域适配对比、真实环境/新用户/方向/遮挡泛化、数据透明度、组件消融和代码可复现性。
