## Context

The data rebuild workspace owns public data intake, simulation recovery, splits, and model-ready delivery. The
OmniHand workspace boundary starts at a validated radar-cube/metric-pose manifest; the WaveLLM workspace expects
aligned metric pose, confidence, radar features, and captions. The original-submission CSL-Daily pipeline ran:
RTMW3D-L pose annotation over sentence images (`[T, 2, 24, 3]`), skeleton point-reflector FMCW simulation
(`config/radar/iwr1843.py`, 64 chirps / 256 ADC samples / 86-element virtual array), then CubeNet training. The
legacy code implementing this is read-only forensic reference under the AGENTS.md legacy boundary.

Raw CSL-Daily is operator-uploaded to `/mnt/gfs/yanyifan/mmPRISM/incoming/<batch>/` per
`workspaces/data_rebuild/docs/authority/40_OPERATIONS/DATA_INTAKE.md`; intake promotion is a gated, check-summed,
no-clobber operation.

## Decisions

### Simulation fidelity and labeling

The new `mmprism.simulation` package is a line-faithful re-implementation of the legacy skeleton point-reflector
simulator (path amplitude `1/(d/2)^2`, baseband phase model, synthetic 86-element array, radar at
`[0, 0, -0.80]`; Hann range FFT to 32 bins, mean-removed Hann Doppler FFT with fftshift to 64 bins, synthetic
steering-vector beamforming; point-cloud densification with z scaled by 0.6, Gaussian smoothing sigma=1,
finite-difference velocities, 30-to-10 fps decimation). Equivalence is proven once against the legacy module on
fixed seeded inputs and frozen as a test fixture; runtime code never imports legacy modules. Per `DEC-012` the
output is registered as its own experiment protocol (`csl_daily_skeleton_sim_v1`), never as MANO-pipeline
reproduction.

### Radar configuration identity

Every simulated sample binds a `radar_config_id`. The only registered configuration is the IWR1843 simulation
configuration (77 GHz, slope 70e12 Hz/s, 256 ADC samples at 5.209 MHz, 64 chirps, 355 us chirp period, 50 ms
frame period). The 64-chirp-sim versus 128x86x256-real mismatch documented in DATA_INTAKE remains an open
evidence conflict and is out of scope.

### Pose annotation

The CSL-Daily adapter reuses the `PoseEstimator` protocol from the CSL-News annotation module with the RTMW3D-L
checkpoint already on the NAS. Historical per-frame logic is preserved: scores below 0.5 become NaN, depth is
re-centered on joints 6/7, 17 body + 42 hand keypoints reduce to `[T, 2, 24, 3]`, sequences with NaN arms or
all-NaN hands are skipped and recorded as QC failures. Batch size is fixed at 1 because larger batches were
shown to be non-equivalent in the CSL-News lineage.

### Delivery boundary

This change ends at validated Parquet deliveries (DELIVERY-POSE-RECON-V1 and DELIVERY-SLU-V2) plus frozen
manifests/splits. WaveLLM `radar_feature [T, F]` requires a trained OmniHand encoder and is explicitly handed
off to the training workspaces. If signer metadata arrives with intake, splits are signer-aware
`sha256_mod_weight_v1` group-disjoint; otherwise the official CSL-Daily split is bound and the limitation is
recorded in the registry.

## Risks

- Raw data upload timing gates every data-dependent stage; code work is sequenced so the simulation package and
  all fixture-driven tests land first.
- Numerical-equivalence fixtures must be generated in a controlled one-shot step and hashed; any later drift
  fails CI rather than silently changing the protocol.
- Full-corpus annotation (~16k sequences) and simulation need GPU time; scheduling is an operational concern
  recorded in the workspace index, not this change.
