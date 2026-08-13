## 1. Simulation Package (data-independent, first)

- [x] 1.1 Add `src/mmprism/simulation/radar_config.py` with the IWR1843 configuration registry.
- [x] 1.2 Add `simulator.py`, `processor.py`, `point_cloud.py` as faithful legacy ports without legacy imports.
- [x] 1.3 Freeze a one-shot numerical-equivalence fixture against legacy `src/fmcw` output.
- [x] 1.4 Add unit and tensor-contract tests; run ruff, mypy, targeted pytest.

## 2. CSL-Daily Intake And Annotation

- [ ] 2.1 Publish the operator upload specification (images, `csl2020ct_v2.pkl`, signer/split metadata, version,
      license, checksums) per DATA_INTAKE.
- [x] 2.2 Implement `csl_daily_intake.py` (checksum/metadata gates, atomic no-clobber promotion to `raw/csl_daily/`).
- [x] 2.3 Implement `csl_daily_annotation.py` (read-only pkl parse to typed records).
- [x] 2.4 Add unit tests with small fixtures.

## 3. Pose Annotation

- [x] 3.1 Implement `csl_daily_pose_annotation.py` on the `PoseEstimator` protocol, batch fixed to 1.
- [x] 3.2 Add `configs/data/csl_daily_rtmw3d.yaml` with injected roots.
- [ ] 3.3 Preserve the partial RTMW3D `annotation_v1` as diagnostic-only. Freeze the source receipt, implement the
  contract-complete `annotation_v2`, complete its GPU smoke, then execute its resumable full-corpus run with validated
  native/score sidecars, finite canonical pose/confidence/validity/frame-mask payloads, QC/quarantine records, and an
  all-source coverage/eligibility manifest. It must not be promoted as training data before these gates pass.

## 4. Materialization And Delivery

- [x] 4.1 Implement a direct-cube engineering prototype and CLI with a numerical-equivalence fixture. It is retained
  only for comparison after `DEC-049`, not as a CSL-Daily formal product.
- [x] 4.2 Add an end-to-end direct-cube prototype integration test over a small pose fixture.
- [ ] 4.3 Implement the canonical pose-to-pre-beamforming-FMCW materializer and immutable raw-radar manifest;
  record signal representation/dtype/shape, simulator config and checksums.
- [ ] 4.4 Implement the runtime range/Doppler/beamforming adapter, injecting its processor contract outside the
  model; add CPU/GPU parity and no-persisted-cube tests.
- [ ] 4.5 Freeze splits (signer-aware if metadata allows, else official-split binding with recorded limitation).
- [ ] 4.6 Build and validate raw-radar pose-reconstruction and DELIVERY-SLU-V2 Parquet deliveries with reader
  parity, pilot capacity/throughput measurements, and no persisted `radar_cube`.

## 5. Authority Closeout

- [ ] 5.1 Update DATA_REGISTRY, workspace changelogs/indexes, and the decision log (simulation protocol entry).
- [ ] 5.2 Run the full verification order including `scripts/audit_docs.py` and `git diff --check`.
