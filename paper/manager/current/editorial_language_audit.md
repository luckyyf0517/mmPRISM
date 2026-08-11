# Editorial Language Audit

Status: `inventory_ready_writeback_blocked_by_evidence`
Last Updated: `2026-08-11`
Role: `sober_language_and_claim_strength_tracker`

## 范围

本清单绑定当前工作稿 `paper/manuscript@3242a40` 和 `PAPER-AUDIT-001`。扫描只覆盖
`sn-article.tex` 递归展开后的有效正文，注释模板、`sn-article-bak.tex` 和未被 include 的文件不计入。
机器可读逐行 context 见 `evidence/artifacts/manuscript_inventory_v2.json`。

编辑明确要求删除首创式、夸张和主观措辞。当前阶段只完成定位和处置分类；尚未修改正文，因为其中
多项比较与泛化表述必须等待复现实验或新增证据后才能决定“保留并量化”还是“降级/删除”。

## 编辑明确列举词

共 12 个命中。

| Term | Count | Active locations | Disposition |
|---|---:|---|---|
| `paving the way` | 1 | `sn-article.tex:202` | 必须改为对结果范围的中性陈述 |
| `remarkable` | 1 | `chapter/2_results.tex:40` | 必须删除，仅报告 MPJPE/PCK 与 uncertainty |
| `superior` | 2 | `chapter/2_results.tex:56,65` | 必须删除；若比较成立，改为具名 baseline 与量化差值 |
| `unique` | 1 | `chapter/2_results.tex:33` | 删除主观修饰，直接描述 micro-Doppler feature |
| `first` | 7 | Introduction `:27,33`; Results `:4,128`; Methods `:1,85`; Discussion `:1` | 均为步骤/顺序语义而非首创 claim；仍建议换成中性结构词，避免机械合规复核误判 |

当前有效正文没有命中 `novel`、`new`、`unprecedented`、`first-of-its-kind`、`extremely`、
`outstanding`、`excellent`、`ultra` 或 `fascinating`。这不代表最终修订稿自动合规；每次正文回写后必须重跑。

## 证据敏感强表述

共 18 个命中。

| Pattern | Count | Key locations | Required gate |
|---|---:|---|---|
| `high-fidelity` / `high fidelity` | 10 | Abstract `:201`; Introduction `:6,16,27,42`; Results `:1,36,128,158`; Discussion `:1` | 定义 fidelity、绑定 pose/translation protocol；无法量化时换为描述性表述 |
| `state-of-the-art` | 3 | Introduction `:6`; Results `:39,166` | 更新 literature、明确比较集合/日期/协议；不具备同协议比较则删除 |
| `optical-level` | 2 | Abstract `:201`; Results `:166` | `EVID-REV-ARCH` 加 vision oracle 协议；避免用单一相对百分比替代完整指标 |
| `generalizable paradigm` | 2 | Abstract `:202`; Discussion `:3` | 当前证据不足；默认降级，除非 `EVID-REV-REAL` 完整关闭 |
| `significantly outperforming` | 1 | Introduction figure caption `:16` | 需要统计检验、多 seed/样本级证据和 matched baseline，否则改为数值比较 |

## Claim Cluster 与证据

| Cluster | Current wording risk | Required evidence | Status |
|---|---|---|---|
| 两阶段必要性 | “more effective”, “structured bridge”, optical-level | `EVID-REV-ARCH`、matched compute/direct baseline | blocked |
| Attention 机制 | 模块协同与噪声抑制被写成已验证机制 | `EVID-REV-ATTN`、leave-one-out | blocked |
| Synthetic-real 对齐 | “physically consistent”, domain gap 已被克服 | `EVID-REV-SYNREAL`、matched fidelity set | blocked |
| 真实泛化/鲁棒性 | low light、multipath、dynamic、occlusion、practical deployment | `EVID-REV-REAL`、方向/遮挡/新用户 stress matrix | blocked |
| 翻译比较 | “significant advantage”, “optical-level” | 原结果复现、matched baseline、sample-level prediction | blocked |

## 回写规则

1. 先删除不影响科学含义的编辑禁用词；不要用同义夸张词替换。
2. 比较句统一写清对象、数据 split、指标、差值和 uncertainty，不使用孤立的 `superior`。
3. `significant` 只在预注册统计检验成立且报告 test/effect size 时使用。
4. `robust`、`generalizable`、`practical` 等范围词只能覆盖实际测试过的条件。
5. 摘要、图注、Results、Discussion 和 response letter 使用同一 claim strength。
6. 每次正文更改后重跑 `PAPER-AUDIT-001`；只有零个 editor-prohibited finding 且强表述均绑定 evidence，
   `ED-WRITE-1/2` 才能关闭。
