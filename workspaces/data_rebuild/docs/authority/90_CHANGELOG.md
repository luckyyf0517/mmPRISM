# Data Rebuild Changelog

Status: current
Owner: Data rebuild lane
Authority scope: Material changes to data intake, radar rebuild, split, and delivery boundaries.
Last reviewed: 2026-08-12

## 2026-08-12

- Consolidated data intake, radar provenance, split, and delivery under one business workspace.
- Deferred a radar/delivery workspace split until independent ownership and frozen handoffs exist.
- Implemented Parquet delivery v1: task-specific typed readers, deterministic split-isolated part/chunk planning,
  atomic materialization, copied frozen inputs, inventory/index/checksum validation, and portable build provenance.
- Kept the live CSL-News visual-pose lane outside final delivery: it remains intermediate evidence until metric
  radar/calibration and aligned feature contracts are available.
- Clarified that new semantic sign-language recruitment and acquisition belong to the collection workspace;
  data rebuild accepts only its frozen, validated session manifest.
- Registered the incoming historical WaveLLM bundle as transfer-in-progress and preservation-only. The recovered
  CSL-News-derived mT5-only export remains a smoke-verified fallback until a stable bundle receipt/audit resolves
  the historical asset's identity and usable role.
- Registered the old-project `dataset/`, `pretrained_models/`, and `log/` paths as a read-only legacy mirror, and
  documented the direct-to-final-volume CSL-Daily raw preservation upload to avoid a second 300-GB transfer.
- Archived CSL-News from the active rebuild: source/download/cache/source-manifest assets were removed and only
  completed pose artifacts plus frozen pose manifests/splits remain as checkpoint-side evidence.
- Added the CSL-Daily gated reproduction operation: direct-preservation receipt, immutable `annotation_v1`,
  candidate annotation QC, separately labelled skeleton simulation, and frozen handoffs. The legacy duplicated
  validation/test mapping is replay-only rather than a new independent evaluation split.
- Extended final translation Parquet delivery with an input-mode-bound `pose_only` profile: its schema and rows omit
  radar features and dimensions, while fusion preserves the existing non-null feature contract. Reader round trips
  and input-mode metadata tampering are covered by fixture validation. This incompatible schema change is published
  as delivery v2; v1 artifacts remain preserved historical products.
- Added a no-clobber, read-only source-receipt command for the direct CSL-Daily preservation upload. It compares
  time-separated inventories, hashes every source file, receipts legacy split identities, and publishes only under
  `interim/`; it remains deliberately blocked while the active rsync transfer changes the source tree.
