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
- Promoted the original-submission CSL-News-100 cam-pose WaveLLM checkpoint to a narrowly scoped P0 historical
  intake/audit asset, with an explicit handoff list and no automatic full-data retraining dependency.
