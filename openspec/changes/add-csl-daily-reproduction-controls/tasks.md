## 1. Contracts and Configuration

- [x] 1.1 Add strict WaveLLM modality-mode configuration and checkpoint compatibility metadata.
- [x] 1.2 Extend the JSONL+NPY translation manifest/profile so `pose_only` omits radar features by contract.
- [x] 1.3 Add CPU-only parser/contract tests for both modes and mismatched checkpoints/manifests.
- [x] 1.4 Extend the final Parquet translation delivery profile so `pose_only` omits radar features by contract.
- [ ] 1.5 Add a versioned CubeNet prediction/feature-export manifest contract with fold/checkpoint bindings.

## 2. Camera-Pose Baseline and Quality Gates

- [ ] 2.1 Record the retained CSL-Daily source identity and explicit absence audit for historical
  `poses`/signals/features/predicted poses; preserve legacy split maps as non-executable metadata and reject their
  absent old-machine targets before regeneration.
- [ ] 2.2 Preserve the 54-sample `annotation_v1` RTMW3D pilot as diagnostic evidence; do not extend, overwrite, or
  promote it because it contains material NaN/sidecar-contract failures.
- [ ] 2.3 Implement contract-complete CSL-Daily RTMW3D `annotation_v2` with raw native
  pose/score sidecars, finite canonical pose and explicit fill policy, confidence, joint/frame validity,
  resume/quarantine behavior, and injected roots.
- [ ] 2.4 Build deterministic full-corpus QC, stratified overlay-review outputs, and an all-source
  coverage/eligibility manifest for `annotation_v2` before simulation.
- [ ] 2.5 Add successor-versus-baseline comparison, blinded review ledger, and promotion decision gate for any
  annotation version after `annotation_v2`.

## 3. First-Loop Synthetic Reconstruction and Later Feature Export

- [ ] 3.1 Consume the accepted CSL-Daily simulation delivery to train a small CubeNet smoke, then freeze the
  canonical reconstruction protocol.
- [ ] 3.2 Implement checkpoint-bound predicted-pose export, with exact temporal alignment.
- [ ] 3.3 Implement fold assignment and cross-fitted predicted-pose materialization for training rows.
- [ ] 3.4 Add integration tests for one train/validation/test fixture and reject incomplete predicted-pose coverage.
- [ ] 3.5 Implement checkpoint-bound frame-feature export and cross-fitted materialization for the later fusion row.

## 4. WaveLLM Matrix and Historical Audit

- [ ] 4.1 Run a small `cam_pose` pose-only smoke using an accepted mT5 asset and production-shape data.
- [ ] 4.2 Run the `predicted_mmw_pose` pose-only smoke as the second first-loop control.
- [ ] 4.3 Run the later `predicted_mmw_pose_plus_cube_feature` fusion smoke after feature-export acceptance.
- [ ] 4.4 Freeze a production language metric protocol (BLEU, ROUGE, SBERT, SimCSE) and retain sample-level
  predictions/references.
- [ ] 4.5 Run the shortest receipt-bound direct historical replay using received pose/signal candidates; label any
  missing input/configuration linkage instead of filling it with a new product.
- [ ] 4.6 After transfer receipt, audit historical daily checkpoints/config/evaluation and classify each historical
  reported value using the paper reproduction taxonomy.

## 5. Documentation and Evidence

- [ ] 5.1 Publish lightweight source identity, annotation QC, and delivery records before formal training; perform a
  full source receipt only for release/archive needs.
- [ ] 5.2 Update data/model registries and formal run records for each frozen product.
- [ ] 5.3 Promote only accepted real-radar evidence to reviewer response/manuscript claims; retain CSL-Daily as
  explicitly labelled synthetic control or historical replay evidence.
- [ ] 5.4 Run unit -> contract -> integration -> GPU smoke -> paper evidence audit, plus documentation audit and
  `git diff --check`.
