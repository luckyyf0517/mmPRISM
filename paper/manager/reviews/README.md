# Reviewer Materials

Status: `decision_and_reviews_imported`
Last Updated: `2026-08-11`

本目录保存审稿意见原文、分析和 reviewer-to-evidence tracker。

## 预期文件

```text
decision_letter.md
review_en.md
review_cn.md
response_letter_tracker.md
reviewer_closure_matrix.md
analysis.md
raw/
```

## 导入规则

1. 英文原文保持完整，不改写 reviewer wording。
2. 稳定 ID 使用 `AE-1`, `R1-1`, `R2-1` 等格式。
3. 如果一条 comment 含多个可独立验收的 ask，可拆成 `R1-2a`, `R1-2b`，但保留原 comment 链接。
4. 中文翻译只作辅助，不替代英文原文。
5. reviewer analysis 与正式 response draft 分离；tracker 记录状态，manuscript 目录保存正式回复正文。
6. 原始邮件中的私人 MTS 链接不得提交；完整 `.eml` 放入 Git 忽略的 `raw/private/`。
