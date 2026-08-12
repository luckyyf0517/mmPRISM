# Documentation Migration Inventory

Baseline repository commit: `decae360c7e51497030183fcf1545a4fa5aaf3c7`

Baseline manuscript submodule: `3242a40631ec5198e66fa8592763235c108513b2`

Unrelated dirty-worktree paths isolated from this change:

```text
src/mmprism/evaluation/language.py
src/mmprism/evaluation/pose.py
src/mmprism/training/omnihand_run.py
src/mmprism/training/wavellm_run.py
src/mmprism/training/distributed.py
tests/integration/test_distributed_omnihand_run.py
```

No-touch paths:

```text
paper/manuscript/
paper/manager/evidence/artifacts/
config/
src/data/
src/eval/
src/fmcw/
src/model/
src/scripts/
src/utils/
root run_*.py
runtime data and artifacts outside Git
```

## Project Documents

| Source | Classification | Owner | Canonical target | Identity/artifact rule |
|---|---|---|---|---|
| `docs/architecture/README.md` | project Authority | coordinator | `docs/authority/30_ARCHITECTURE/SYSTEM_ARCHITECTURE.md` | preserve architecture claims |
| `docs/architecture/tensor_contracts.md` | project Authority | coordinator | `docs/authority/20_CONTRACTS/TENSOR_CONTRACTS.md` | preserve protocol names |
| `docs/architecture/data_splits.md` | project Authority | coordinator | `docs/authority/20_CONTRACTS/DATA_SPLITS.md` | preserve split schema IDs |
| `docs/architecture/run_artifacts.md` | project Authority | coordinator | `docs/authority/20_CONTRACTS/RUN_ARTIFACTS.md` | preserve artifact schemas |
| `docs/architecture/model_assets.md` | project Authority | coordinator | `docs/authority/20_CONTRACTS/MODEL_ASSETS.md` | preserve asset revisions |
| `docs/architecture/release_audit.md` | operation | coordinator | `docs/authority/40_OPERATIONS/RELEASE_AUDIT.md` | artifact JSON remains in place |
| `paper/manager/current/core_rules.md` | project Authority | coordinator | `docs/authority/20_CONTRACTS/ENGINEERING_RULES.md` | preserve engineering rules |
| `paper/manager/sync_map.md` | superseded Authority | coordinator | `docs/authority/20_CONTRACTS/DOCUMENT_GOVERNANCE.md` | replace multi-page sync rule |
| `paper/manager/decisions/decision_log.md` | project Authority | coordinator | `docs/authority/60_DECISIONS/DECISION_LOG.md` | preserve all `DEC-*` IDs |
| dashboard, overview, issues, roadmap, master/code todo, architecture status | Log snapshots | coordinator | `docs/logs/2026/08/` | preserve task/blocker IDs |
| formal preflight, model asset, and release evidence | Logs | coordinator | `docs/logs/2026/08/` | preserve hashes and evidence IDs |

## CSL-News Annotation Documents

| Source group | Classification | Canonical target | Identity/artifact rule |
|---|---|---|---|
| `docs/architecture/csl_news_data.md` | workspace Authority | `workspaces/csl_news_annotation/docs/authority/30_ARCHITECTURE/` | preserve source revision and contracts |
| CSL-News annotation runbook | operation | `workspaces/csl_news_annotation/docs/authority/40_OPERATIONS/` | retain repeatable operation |
| source integrity/manifest, pose manifest/split, metadata, legacy comparison | Logs | `workspaces/csl_news_annotation/docs/logs/2026/08/` | preserve `DATA-*`, split IDs, hashes, and incident identities |

## Data Rebuild Documents

| Source group | Classification | Canonical target | Identity/artifact rule |
|---|---|---|---|
| Parquet delivery | workspace Authority | `workspaces/data_rebuild/docs/authority/20_CONTRACTS/` | preserve `DEC-038` and delivery IDs |
| upload checklist and rebuild runbook | operations | `workspaces/data_rebuild/docs/authority/40_OPERATIONS/` | retain supported intake/rebuild operations |
| data registry | workspace Authority | `workspaces/data_rebuild/docs/authority/50_VALIDATION/` | preserve dataset/split/model/asset IDs |
| data status, data tasks, radar audit | Logs | `workspaces/data_rebuild/docs/logs/2026/08/` | preserve task/evidence IDs and hashes |

## Training Documents

| Source group | Classification | Canonical target | Identity/artifact rule |
|---|---|---|---|
| OmniHand architecture | workspace Authority | `workspaces/omnihand_training/docs/authority/30_ARCHITECTURE/` | preserve model/metric protocols |
| OmniHand smoke and formal run | Logs | `workspaces/omnihand_training/docs/logs/2026/08/` | preserve run/evidence IDs and JSON hashes |
| WaveLLM architecture and model support | workspace Authority | `workspaces/wavellm_training/docs/authority/` | preserve mT5-only boundary |
| mT5/WaveLLM smoke, formal run, support evidence | Logs | `workspaces/wavellm_training/docs/logs/2026/08/` | preserve run/evidence IDs and JSON hashes |

## Paper Revision Documents

| Source group | Classification | Canonical target | Identity/artifact rule |
|---|---|---|---|
| availability and display registry | workspace Authority | `workspaces/paper_revision/docs/authority/20_CONTRACTS/` | preserve `DISPLAY-*` and evidence IDs |
| operator, reproduction, submission audit, original intake | operations | `workspaces/paper_revision/docs/authority/40_OPERATIONS/` | retain supported named operations |
| manuscript, editorial, review, response, closure, paper evidence | workspace Authority | `workspaces/paper_revision/docs/authority/50_VALIDATION/` | preserve reviewer/task/evidence IDs |
| reviewer sources, manuscript inventory, round/task/registry snapshots | Logs | `workspaces/paper_revision/docs/logs/2026/08/` | preserve original text, IDs, and snapshot hashes |

## Compatibility And Inbound Links

Every source path above remains a compatibility entrypoint containing exactly one local link to its
canonical target. `scripts/audit_docs.py` validates target resolution and rejects independent metadata at
old paths. This makes each compatibility file its own inbound-link mapping; no second permanent redirect
registry is maintained after this change is archived.

## Immutable Artifact Baseline

| Artifact | SHA-256 |
|---|---|
| `evaluation_models_smoke_v1.json` | `e957ac79f620f0a982019befa4938c393357764f5d912b4b6a7c27996f789b39` |
| `manuscript_inventory_v1.json` | `823e93a3ad85eeec081001555167c76bc0bbe280db525d324e3a04c6bbd9fca9` |
| `manuscript_inventory_v2.json` | `db01f16e1fcfe7b22743eab8671820cbc428d816f55c0ed69af35c25ba0647d7` |
| `mt5_smoke_v1.json` | `57fd48b2028c8ee68b465b2aa2eaee2278596cb2905e46527d431b83c0b58df4` |
| `omnihand_formal_run_v1.json` | `cf8b306a169e8e7adea874502d25e007e2b99bb7ce38d958fe6307595a725862` |
| `omnihand_smoke_v1.json` | `0e49867864c36a65baf4c77fe838edd85f1969b6ade0c9012de104ac4126e389` |
| `release_audit_v1.json` | `dd14960f5f331e827a2ebc79ea55b7bf6be718e566c8fc2f816e2bb744fe8951` |
| `wavellm_formal_run_v1.json` | `71987e0c72277857697407c3ba9e78947d5957b7f2fba25d07d1a503f0e1e08c` |
