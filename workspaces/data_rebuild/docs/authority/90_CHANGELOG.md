# Data Rebuild Changelog

Status: current
Owner: Data rebuild lane
Authority scope: Material changes to data intake, radar rebuild, split, and delivery boundaries.
Last reviewed: 2026-08-13

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

## 2026-08-13

- Paused CSL-Daily execution for compute-server migration. The initialized `annotation_v2` queue remains paused and
  no v2 sample or manifest has been published. The shared migration runbook requires an isolated target smoke before
  any worker resume; cancelled batch benchmarks remain reusable scripts only.

- Recorded the then-provisional `DEC-048` pose-only ordering. It was superseded for camera-pose reconstruction by
  `DEC-051` after the v1 pilot demonstrated material contract failure; full-corpus `annotation_v2` is now mandatory.
  Feature/fusion remains explicitly a non-blocking later delivery.
- Applied `DEC-049`: CSL-Daily persistence ends at pre-beamforming synthetic FMCW. The power cube is derived by the
  versioned runtime processor; direct-cube materialization is retained only as a non-formal engineering prototype.
- Reclassified `DELIVERY-POSE-RECON-V1` and its direct-cube tensor/Parquet contract as fixture-only for CSL-Daily,
  and registered the pending raw-FMCW delivery and runtime-processing acceptance gates.
- Added a bounded CSL-Daily legacy-path replay runbook. It separates today's source-backed end-to-end diagnostic
  (camera pose -> temporary simulated cube -> OmniHand -> pose-only WaveLLM generation) from the formal
  raw-FMCW/Parquet reconstruction, defines source/asset gates and failure classification, and prohibits using
  the replay as paper evidence or an independent legacy test.
- Applied `DEC-050`: recorded the duplicate compressed CSL-Daily frame payload for removal, retained expanded
  frames/official metadata/review video as the source of record, and made full-corpus versioned cam-pose
  reconstruction with QC/eligibility coverage a P0 gate. The partial RTMW3D pilot remains diagnostic-only.
- Applied `DEC-051`: froze the partial v1 pilot because its completed poses retain NaNs and its sidecars lack native
  pose/scores, canonical confidence, and validity masks. A new contract-complete v2 implementation and resumable
  full-corpus build now precede all simulation and training work.
- Added v2 execution controls: finite interpolated canonical pose, native RTMW3D/scores plus confidence,
  validity, imputation and frame-mask audit payloads, per-sequence no-clobber restart validation, durable leases,
  cooperative pause/resume, error quarantine, and a lease-free full-coverage finalization gate.
- Removed full-source SHA-256 receipt from the P0 annotation gate. It remains an optional release/archive operation;
  source root/ID, frame list/count, configuration/model identity and Git state stay attached to formal outputs.
- Registered CE-CNSL as a P1 parallel intake under `DEC-052`, without creating another workspace. Added its
  independent dataset/split identities and a bounded source, label/signer, and pose-pilot runbook. Full-corpus
  annotation and simulation remain conditional and never block CSL-Daily P0.
- Applied `DEC-054`: CE-CNSL is now a paused follow-on source rather than an active parallel intake. Its assessment,
  identities, runbook, and OpenSpec remain; source download, adapter work, pilot, and GPU execution wait for an
  accepted CSL-Daily end-to-end stable loop and explicit coordinator reactivation.
- Applied `DEC-056`: explicit project-owner authorization during CSL-Daily's late stable phase may unlock CE-CNSL
  download and immutable receipt only. It does not activate label repair, adapter, pilot, processing, or GPU work.
