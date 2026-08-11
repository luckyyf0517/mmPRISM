# Final Submission Reviewer Audit Runbook

Status: `planned_final`
Last Updated: `2026-08-11`
Role: `reviewer_perspective_final_gate`

> 核心原则：最终 audit 从原始 reviewer comments 出发，而不是从作者 todo 出发。

## 1. 固定输入

1. `reviews/review_en.md`
2. `reviews/response_letter_tracker.md`
3. `reviews/reviewer_closure_matrix.md`
4. 当前 manuscript 和 response letter
5. `evidence/paper_evidence_map.md`
6. `current/issues.md`
7. `tasks/todo.md`

## 2. 每条 Comment 的固定问题

1. Original Ask 是否被准确理解？
2. Response 是否直接回答，而非只描述内部过程？
3. Manuscript 是否真实落实 response 声称的修改？
4. 数据/实验/图表证据是否可追溯？
5. Claim 是否过强、过弱或与限制冲突？
6. Reviewer 重读时最可能不满意什么？
7. 当前 action 是 `closed / text_logic / evidence_gap / asset_only / reproducibility` 中哪一类？

## 3. 工程与论文联合检查

- paper-facing config、split、checkpoint、metric artifact 可访问。
- manuscript 数值与 evidence map 一致。
- response letter 的表号、图号、章节位置准确。
- 无 `TBD/TODO/placeholder/draft` 等未解释标记。
- 无已 superseded 的模型、数据或 metric protocol 以当前时态出现。
- 主文、response 和 supplement 编译通过。
- 最终 PDF 做人工视觉检查。

## 4. Exit Gate

所有 P0/P1 reviewer item 必须为：

- `done`，或
- 经作者明确拍板的诚实边界，并在 response/limitations 中说明。

不能因为截止日期临近而把 `blocked` 直接改成 `done`。
