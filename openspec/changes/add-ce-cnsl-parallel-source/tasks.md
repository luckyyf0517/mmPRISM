## 0. Activation Gate

- [ ] 0.0 Optionally record project-owner authorization for source download and immutable receipt during CSL-Daily's
  late stable phase; keep all implementation and compute tasks inactive.
- [ ] 0.1 Confirm an accepted CSL-Daily `annotation_v2 -> synthetic FMCW -> OmniHand -> pose-only WaveLLM` stable
  loop with frozen run/evaluation evidence.
- [ ] 0.2 Record explicit reactivation after reviewing CSL-Daily's shared and dataset-specific interfaces.
- [ ] 0.3 Until 0.1 and 0.2 pass, permit only an explicitly authorized 0.0 download/receipt and keep tasks 1.2 onward
  plus all implementation and compute tasks inactive.

## 1. Source And Metadata

- [ ] 1.1 After optional authorization or full activation, freeze source URL/retrieval time, repository revision, full
  archive byte size and SHA-256 in immutable storage; do not extract or decode under download-only authorization.
- [ ] 1.2 After full activation, extract into a versioned destination and freeze the inventory and license status.
- [ ] 1.3 Validate exact video/CSV sample-number coverage and record codec/FPS/resolution/frame-count metadata.
- [ ] 1.4 Build and manually audit the signer repair table, especially H--L boundary samples.
- [ ] 1.5 Preserve spoken Chinese, raw Gloss, versioned normalized Gloss, and regional notes without destructive edits.

## 2. Shared Processing Boundary

- [ ] 2.1 Identify CSL-Daily annotation functions whose contracts are source-independent before extracting them.
- [ ] 2.2 Add a thin CE-CNSL source/label adapter using validated manifests and injected roots.
- [ ] 2.3 Reuse scheduler, native/canonical pose payload, confidence/validity, quarantine, and finalization behavior.
- [ ] 2.4 Add CPU-only adapter/manifest/label tests without importing optional training dependencies.

## 3. Pose Pilot And Promotion

- [ ] 3.1 Freeze a 120--240-sequence stratified pilot across signer/device/geometry/length/difficulty/label strata.
- [ ] 3.2 Run the pilot with aspect-ratio-preserving preprocessing and the CSL-Daily `annotation_v2` output contract.
- [ ] 3.3 Publish deterministic coverage/QC/overlay review and failure classification.
- [ ] 3.4 Record an explicit accept/reject decision; do not schedule full processing on a failed or incomplete pilot.

## 4. Full Processing And Experiments

- [ ] 4.1 After promotion, build an immutable full CE-CNSL pose manifest and eligible split-bound delivery.
- [ ] 4.2 Generate separately identified synthetic FMCW/OmniHand products using the accepted shared contracts.
- [ ] 4.3 Run `CSL-Daily -> CE-CNSL` sequential adaptation first.
- [ ] 4.4 Compare CE-CNSL-only and balanced joint/rehearsal training with per-dataset metrics and forgetting analysis.

## 5. Verification And Evidence

- [ ] 5.1 Bind every formal run to CE-CNSL source/manifest/split/label-transform hashes.
- [ ] 5.2 Keep CE-CNSL synthetic controls separate from real-radar and participant-disjoint claims.
- [ ] 5.3 Run unit -> contract -> integration -> GPU smoke -> paper evidence audit.
- [ ] 5.4 Run OpenSpec strict validation, documentation audit, and `git diff --check`.
