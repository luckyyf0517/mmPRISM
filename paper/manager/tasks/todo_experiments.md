# Experiment and Evidence Todo

Status: `blocked_on_baseline_and_reviews`
Last Updated: `2026-08-11`
Role: `experiment_execution_tracker`

## A. Canonical Baseline Bootstrap

| ID | Experiment | Status | Required Artifact |
|---|---|---|---|
| `EXP-001-A` | Processor fixed-input golden run | blocked | input fixture, output tensor/hash, config |
| `EXP-001-B` | OmniHand single-frame smoke | passed_engineering | clean init/config、finite `[2,2,24,3]`、sample MPJPE；`EVID-CODE-OMNIHAND-SMOKE-V1` |
| `EXP-001-C` | OmniHand temporal smoke | passed_engineering | 10-frame shape、mask invariance、gradient/update、time/memory 与 deterministic replicate report |
| `EXP-001-D` | WaveLLM pose-only smoke | blocked | loss/generation/sample output |
| `EXP-001-E` | WaveLLM feature-only smoke | blocked | loss/generation/sample output |
| `EXP-001-F` | WaveLLM multimodal smoke | passed_engineering | clean `e31000b` formal train/adapter-checkpoint/reload/generation/character metric；`EVID-CODE-WAVELLM-FORMAL-V1` |

## B. Original Submission Reproduction

待导入原论文后，按每张表和图创建稳定 ID。每项至少记录：

- original reported value
- exact dataset/split
- preprocessing/model/metric protocol
- historical checkpoint/artifact
- rerun result and variance
- difference diagnosis
- paper-facing decision

## C. Reviewer-Driven Experiments

当前已注册：

| ID | Priority | Reviewer Items | Experiment | Status |
|---|---|---|---|---|
| `EXP-REV-001` | P0 | `ED-SCI-2`,`R2-1` | matched direct 4D-cube-to-LLM baseline | blocked |
| `EXP-REV-002` | P0 | `ED-SCI-3`,`R2-2` | shallow/full/adversarial/MMD equal-budget DA | blocked |
| `EXP-REV-003` | P0 | `R1-3`,`R1-4c/d`,`R1-5`,`R2-3` | real orientation/occlusion/new-user stress matrix | blocked |
| `EXP-REV-004` | P0 | `R1-4a` | synthetic-real fidelity analysis | blocked |
| `EXP-REV-005` | P0 | `R2-5` | attention leave-one-out ablation | blocked |
| `EXP-REV-006` | P0 | `R1-2` | standardized compute profile | blocked |
| `EXP-REV-007` | P1 | `R2-6` | cross-modal feasibility audit/baseline | not_started |

任何新实验开始前必须填写：

| Field | Requirement |
|---|---|
| Reviewer item | `AE-*` / `R*-*` |
| Hypothesis | 能够被结果证伪的一句话 |
| Protocol | dataset/split/baseline/metric/seeds |
| Acceptance | 什么结果足以回答 reviewer |
| Failure handling | 负结果如何解释和回写 |
| Paper target | section/table/figure/supplement |
| Budget | GPU-hours, storage, expected duration |

## D. Final Evidence Gates

- 原投稿关键数值已复现或解释 gap。
- 新结果包含多 seed/置信区间（适用时）。
- 所有指标保留 sample-level 输出。
- 主表、补充材料和 response letter 使用同一 protocol/version。
