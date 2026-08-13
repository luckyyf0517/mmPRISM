## Why

The revision-critical execution path requires recovering the CSL-Daily simulation / OmniHand second stage. The
new `src/mmprism/` package already provides an engineering radar-cube fixture path, but CSL-Daily requires a
different persistent data boundary: pre-beamforming simulated FMCW must be stored and the power cube must be derived
at runtime. The remaining upstream work includes intake, pose annotation, raw-radar materialization and a runtime
processor adapter. The legacy
implementations (`run_csl_daily_annotation.py`, `run_simulation.py`, `src/fmcw/`) are read-only forensic
references and cannot be imported.

## What Changes

- Add CSL-Daily raw-data intake (`src/mmprism/data/csl_daily_intake.py`): checksum/metadata-gated promotion of an
  operator-uploaded `incoming/` batch into an immutable `raw/csl_daily/` root.
- Add a read-only `csl2020ct_v2.pkl` annotation parser (`src/mmprism/data/csl_daily_annotation.py`) producing
  typed records.
- Add an RTMW3D pose-annotation adapter for CSL-Daily (`src/mmprism/data/csl_daily_pose_annotation.py`) on the
  existing `PoseEstimator` protocol, batch size fixed to 1 per the CSL-News numerical-equivalence finding.
- Add a new `src/mmprism/simulation/` package: IWR1843 radar configuration registry, point-reflector FMCW
  simulator (faithful port of the legacy skeleton simulator), synthetic-array processor producing
  `[T, 64, 32, 32, 32]` power cubes, and pose point-cloud densification/preprocessing. No legacy imports; a
  one-shot numerical-equivalence fixture is frozen against the legacy output.
- Add a config-driven pose-to-pre-beamforming-FMCW materialization pipeline emitting a raw-radar reconstruction
  manifest bound to a `radar_config_id`, a processor fingerprint and the frozen pose manifest hash. The runtime
  adapter derives CubeNet power cubes on device and no formal CSL-Daily delivery persists `radar_cube`.
- Register the rebuilt simulation as a separately labeled experiment protocol per `DEC-012`: it is a
  code-faithful re-implementation of the original-submission skeleton simulator and is **not** registered as a
  reproduction of the manuscript-described MANO mesh / ray-tracing pipeline.
- Produce frozen splits and raw-radar pose-reconstruction / DELIVERY-SLU-V2 Parquet deliveries. The direct-cube
  Parquet prototype remains a numerical fixture, not a CSL-Daily formal product.

## Non-Goals

- Resolving the MANO-versus-skeleton provenance question (`ASSET-SIM-PROVENANCE` / `DATA-001-J` stay blocked;
  no MANO-equivalence claim is made or implied).
- CSL-News reconstruction or retraining (archived by `DEC-043`/`DEC-044`).
- OmniHand or WaveLLM training runs, checkpoint intake/audit, and WaveLLM `radar_feature` extraction (owned by
  the training workspaces; this change delivers model-ready data only).
- Real-radar beamforming, physical axes, or calibration evidence (`DEC-024` scope is untouched; simulated cubes
  use the simulator's own synthetic array geometry).
- Modifying legacy forensic code or moving shared code into workspaces.

## Impact

- New raw-radar data adapter/reader under `src/mmprism/data/`, runtime processor orchestration, versioned configs
  under `configs/data/`, and unit/contract/integration tests. Existing direct-cube code is retained for comparison
  but not expanded for CSL-Daily formal delivery.
- Raw CSL-Daily data lands under the data root as an immutable intake once the operator uploads it; nothing is
  downloaded or fetched automatically.
- Data rebuild Authority (registry, changelog, index) records each stage's status.
