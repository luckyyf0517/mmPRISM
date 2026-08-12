## 1. Simulation Package (data-independent, first)

- [ ] 1.1 Add `src/mmprism/simulation/radar_config.py` with the IWR1843 configuration registry.
- [ ] 1.2 Add `simulator.py`, `processor.py`, `point_cloud.py` as faithful legacy ports without legacy imports.
- [ ] 1.3 Freeze a one-shot numerical-equivalence fixture against legacy `src/fmcw` output.
- [ ] 1.4 Add unit and tensor-contract tests; run ruff, mypy, targeted pytest.

## 2. CSL-Daily Intake And Annotation

- [ ] 2.1 Publish the operator upload specification (images, `csl2020ct_v2.pkl`, signer/split metadata, version,
      license, checksums) per DATA_INTAKE.
- [ ] 2.2 Implement `csl_daily_intake.py` (checksum/metadata gates, atomic no-clobber promotion to `raw/csl_daily/`).
- [ ] 2.3 Implement `csl_daily_annotation.py` (read-only pkl parse to typed records).
- [ ] 2.4 Add unit tests with small fixtures.

## 3. Pose Annotation

- [ ] 3.1 Implement `csl_daily_pose_annotation.py` on the `PoseEstimator` protocol, batch fixed to 1.
- [ ] 3.2 Add `configs/data/csl_daily_rtmw3d.yaml` with injected roots.
- [ ] 3.3 Add mocked unit tests; run GPU smoke on a small sequence subset.

## 4. Materialization And Delivery

- [ ] 4.1 Implement the pose-to-cube materialization pipeline and CLI, emitting `mmprism.pose_reconstruction.sample_v1`.
- [ ] 4.2 Add an end-to-end integration test over a small pose fixture.
- [ ] 4.3 Freeze splits (signer-aware if metadata allows, else official-split binding with recorded limitation).
- [ ] 4.4 Build and validate DELIVERY-POSE-RECON-V1 and DELIVERY-SLU-V1 parquet deliveries with reader parity.

## 5. Authority Closeout

- [ ] 5.1 Update DATA_REGISTRY, workspace changelogs/indexes, and the decision log (simulation protocol entry).
- [ ] 5.2 Run the full verification order including `scripts/audit_docs.py` and `git diff --check`.
