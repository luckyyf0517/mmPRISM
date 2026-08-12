## 1. Contracts and Configuration

- [x] 1.1 Add strict WaveLLM modality-mode configuration and checkpoint compatibility metadata.
- [x] 1.2 Extend the JSONL+NPY translation manifest/profile so `pose_only` omits radar features by contract.
- [x] 1.3 Add CPU-only parser/contract tests for both modes and mismatched checkpoints/manifests.
- [x] 1.4 Extend the final Parquet translation delivery profile so `pose_only` omits radar features by contract.
- [ ] 1.5 Add a versioned CubeNet prediction/feature-export manifest contract with fold/checkpoint bindings.

## 2. Camera-Pose Baseline and Quality Gates

- [ ] 2.1 Receipt and audit uploaded historical CSL-Daily `poses`/signals/features and legacy split maps before
  regeneration; preserve their original relative locations and bind or explicitly reject source identity.
- [ ] 2.2 Implement CSL-Daily RTMW3D `annotation_v1` only after source receipt, with raw-output sidecars,
  resume/quarantine behavior, and injected roots.
- [ ] 2.3 Build deterministic baseline QC and stratified overlay-review outputs.
- [ ] 2.4 Implement `annotation_v2` only as a separate experiment/config after v1 is frozen.
- [ ] 2.5 Add candidate-versus-baseline comparison, blinded review ledger, and promotion decision gate.

## 3. Synthetic Reconstruction and Feature Export

- [ ] 3.1 Consume the accepted CSL-Daily simulation delivery to train a small CubeNet smoke, then freeze the
  canonical reconstruction protocol.
- [ ] 3.2 Implement checkpoint-bound predicted-pose and frame-feature export, with exact temporal alignment.
- [ ] 3.3 Implement fold assignment and cross-fitted prediction/feature materialization for training rows.
- [ ] 3.4 Add integration tests for one train/validation/test fixture and reject incomplete fold coverage.

## 4. WaveLLM Matrix and Historical Audit

- [ ] 4.1 Run a small `cam_pose` pose-only smoke using an accepted mT5 asset and production-shape data.
- [ ] 4.2 Run `predicted_mmw_pose` pose-only and `predicted_mmw_pose_plus_cube_feature` fusion smokes.
- [ ] 4.3 Freeze a production language metric protocol (BLEU, ROUGE, SBERT, SimCSE) and retain sample-level
  predictions/references.
- [ ] 4.4 Run the shortest receipt-bound direct historical replay using received pose/signal candidates; label any
  missing input/configuration linkage instead of filling it with a new product.
- [ ] 4.5 After transfer receipt, audit historical daily checkpoints/config/evaluation and classify each historical
  reported value using the paper reproduction taxonomy.

## 5. Documentation and Evidence

- [ ] 5.1 Publish source/legacy-pose receipt, annotation QC, and delivery receipts before formal training.
- [ ] 5.2 Update data/model registries and formal run records for each frozen product.
- [ ] 5.3 Promote only accepted real-radar evidence to reviewer response/manuscript claims; retain CSL-Daily as
  explicitly labelled synthetic control or historical replay evidence.
- [ ] 5.4 Run unit -> contract -> integration -> GPU smoke -> paper evidence audit, plus documentation audit and
  `git diff --check`.
