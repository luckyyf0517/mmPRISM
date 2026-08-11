# Paper Evidence Map

Status: `review_evidence_blocks_and_display_registry_active`
Last Updated: `2026-08-11`
Role: `paper_claim_to_artifact_map`

原稿导入后，每个主表、主图和关键 claim 都必须登记。

| Evidence ID | Paper Location | Claim / Question | Reviewer Items | Dataset/Split | Run IDs | Metric Protocol | Artifact | Provenance Status | Writeback Status |
|---|---|---|---|---|---|---|---|---|---|
| `EVID-REV-ARCH` | Introduction; Results; Discussion | explicit pose reconstruction vs direct radar-to-text | `ED-SCI-2`,`R2-1` | pending | `EXP-REV-001` | pose+translation protocol pending | pending | blocked | not_started |
| `EVID-REV-DA` | Results: reality gap; Methods: training details | shallow adaptation accuracy-efficiency trade-off | `ED-SCI-3`,`R2-2` | pending | `EXP-REV-002` | pending | pending | blocked | not_started |
| `EVID-REV-REAL` | Results: articulation recovery; Discussion | orientation/occlusion/new-user real-world generalization | `ED-SCI-5`,`R1-3`,`R1-4c/d`,`R1-5`,`R2-3` | pending | `EXP-REV-003` | condition-stratified pose+translation | pending | blocked | not_started |
| `EVID-REV-SYNREAL` | Results: reality gap; Methods: synthetic generation | synthetic-to-real fidelity and domain gap | `R1-4a` | pending | `EXP-REV-004` | pending | pending | blocked | not_started |
| `EVID-REV-ATTN` | Results: articulation recovery | necessity of spatial/channel/SE attention | `R2-5` | pending | `EXP-REV-005` | pose/downstream pending | pending | blocked | not_started |
| `EVID-REV-EFF` | Methods: model/training; new efficiency table TBD | training/inference efficiency | `R1-2` | pending | `EXP-REV-006` | standard compute profile | pending | blocked | not_started |
| `EVID-REV-XMODAL` | Introduction comparison table; Results: SLU | positioning against WiFi/acoustic continuous SLU | `R2-6` | pending | `EXP-REV-007` | comparability audit pending | pending | not_started | not_started |
| `EVID-REV-DATASET` | Methods: Datasets and synthetic data generation | dataset characterization and split transparency | `ED-SCI-4`,`R1-4b`,`R2-4` | pinned CSL-News metadata + partial pose manifest + partial sequence split; remaining datasets/signers/final splits pending | `DATA-REV-001-CSLNEWS-META-20260811` | metadata profile + `mmprism.sample.v1` + `mmprism.split_assignment.v1`; complete subject/split audit pending | `evidence/csl_news_metadata_profile.md`; `evidence/csl_news_pose_manifest.md`; `evidence/csl_news_pose_split.md`; partial hashes `4161593f...` / `133f32d5...` | in_progress | not_started |
| `EVID-PAPER-INVENTORY` | Current main entry + supplementary | exact active sections/display assets/references, Availability gaps, placeholder markers and sober-language locations | `ED-WRITE-1/2`,`ED-COMP-4/5/6/8` | n/a | `PAPER-AUDIT-001` | `mmprism.manuscript_audit.v2` | `evidence/manuscript_inventory.md`; JSON SHA-256 `db01f16e...` | evidence_ready_current_snapshot | inventory_only |
| `EVID-DISPLAY-REGISTRY` | 19 current display environments / 20 captioned items | per-item Source Data and dataset/split/run/checkpoint/metric/script requirements; S2-S6 placeholder exclusion | `ED-COMP-8`,`R1-2/3/4/5/6`,`R2-1/2/3/4/5/6` | item-specific; all scientific chains pending | item-specific; see registry | item-specific; see registry | `evidence/display_item_registry.md` | registered_provenance_pending | not_started |

## 推荐粒度

- 每张表至少一个 evidence block；如果表中不同 row 来自不同 protocol，应拆分。
- 每张主图记录生成脚本、输入数据和最终 PDF/PNG hash。
- 数据规模、split、公平性、泛化和鲁棒性等关键文字 claim 也应单独登记。
- response letter 中出现的新数值必须先进入本表，再进入正式回复。
