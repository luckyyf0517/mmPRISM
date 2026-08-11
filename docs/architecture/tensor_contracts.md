# Canonical Tensor and Range-Doppler Contracts

Status: `range_doppler_v1_implemented_beamforming_blocked`
Last Updated: `2026-08-11`

## Scope

This document fixes the in-memory axis, dtype, unit, and processing conventions used by the rebuilt
package. It does not assert that the original paper pipeline has been reproduced. In particular, 2D
beamforming remains blocked until the virtual-array mapping, channel order, calibration, and complex
conjugation convention are recovered from acquisition evidence.

The dependency-light validators live in `mmprism.contracts.tensors`. The NumPy implementation lives in
`mmprism.radar.processing`; neither module imports PyTorch, Lightning, Transformers, or legacy code.

## Tensor Contracts

| Artifact | Canonical trailing axes | Dtype | Value/metadata requirements |
|---|---|---|---|
| raw ADC frame | `[chirp, antenna, sample]` | complex floating | finite; optional leading `batch`, `time` |
| range-Doppler spectrum | `[doppler, antenna, range]` | complex floating | finite; Doppler axis is FFT-shifted |
| 4D radar cube | `[doppler, range, azimuth, elevation]` | real floating | finite, non-negative power |
| dual-hand pose | `[hand, joint, coordinate] = [2,24,3]` | real floating | finite, metres, explicit coordinate-frame ID |
| feature sequence | `[time, feature]` | real floating | finite; optional leading `batch` |
| caption | scalar Unicode string | text | non-empty, no NUL, explicit language tag |

For dual-hand pose, hand index `0` is left and `1` is right. Joint order is arm shoulder, arm elbow,
arm wrist, then the 21-point hand order (hand wrist followed by thumb, index, middle, ring, and little
finger joints from base to tip). Coordinate components are `[x,y,z]`; the meaning and orientation of
those axes comes from the mandatory coordinate-frame ID.

The current CSL-News RTMW3D-derived `canonical_pose` has this shape and ordering, but its physical unit
and coordinate frame have not yet been established as metric radar coordinates. It must not be passed as
a conforming dual-hand metric pose, or used for millimetre-error claims, until that calibration evidence is
recovered or a documented conversion is implemented.

### Model-ready pose reconstruction manifest

The dependency-light adapter `mmprism.data.PoseReconstructionManifest` accepts only local relative
`.npy` modality references in `mmprism.sample.v1` JSONL records. Each modality must carry its exact
shape, dtype, and SHA-256. Required modalities are:

- `radar_cube`: `[T,D,R,A,E]` `float32` non-negative power under
  `mmprism.radar_cube.power_v1`;
- `pose_gt`: `[2,24,3]` `float32` metres with an explicit coordinate-frame ID.

Optional `frame_mask: [T] bool` and `pose_valid: [2,24] bool` arrays default to all-valid. A batch may
contain different sequence lengths, but all samples must share the same spatial cube shape and pose
coordinate frame. Collation pads time with zero power and marks padding invalid; invalid source frames are
also zeroed. The sample acquisition metadata must declare
`sample_protocol: mmprism.pose_reconstruction.sample_v1`. This contract does not make the current
CSL-News image-derived pose metric or establish radar-camera calibration.

Raw storage order intentionally differs from the notation order `S_raw(t,n,m)` used in the manuscript:
`t -> sample`, `n -> chirp`, and `m -> antenna`. The storage mapping is explicit at every validator call;
dimensions are never inferred from size alone.

## Range-Doppler Protocol V1

`range_doppler_transform` performs the following operations without mutating its input:

1. Validate finite complex input and explicit leading axes.
2. Apply either a rectangular window or periodic Hann window along fast time.
3. Apply the range FFT along `sample`, optionally with zero padding.
4. Optionally subtract the complex mean across slow time before the Doppler FFT.
5. Apply the configured window along `chirp`.
6. Apply the Doppler FFT and `fftshift` along its output axis.
7. Select an explicit contiguous range-bin interval and validate the output contract.

The periodic Hann is `numpy.hanning(N + 1)[:-1]`. FFT sizes default to the corresponding input size,
may be increased for zero padding, and may not silently truncate input. No normalization, power
conversion, physical range/velocity calibration, antenna selection, or beamforming is applied.

Rectangular-window analytic complex tones are the golden reference: a tone placed at integer fast- and
slow-time frequencies must peak at its exact range bin and FFT-shifted Doppler bin. Static slow-time
signals must vanish when mean subtraction is enabled.

## Evidence Conflicts Kept Open

| Topic | Manuscript | Visible legacy implementation | Canonical action |
|---|---|---|---|
| raw notation/storage | `S_raw(t,n,m)` | arrays documented as `[B, chirp, antenna, sample]` | map names explicitly; canonical storage uses `[chirp, antenna, sample]` |
| observed shape | dimensions not fully stated | collected fallback `[128,86,256]`; simulator config `64` chirps and `256` samples | no hard-coded dimension sizes |
| bandwidth | `3.85 GHz`, about `3.89 cm` | `70e12 Hz/s` over `5-55 us` gives `3.5 GHz`, about `4.29 cm`; 256/5.209 MHz samples span about `49.15 us` | require per-sequence acquisition config before physical axes |
| clutter removal | manuscript equation subtracts mean after range-Doppler FFT over Doppler bins | subtracts slow-time mean before Doppler FFT | expose and record pre-Doppler choice; do not claim equivalence |
| beamforming | `Y=A^H X`, 192 virtual elements reduced to 86 horizontal/4 vertical | `einsum(..., A)` has no explicit conjugation; default coordinate list has 116 elements | defer canonical beamforming |
| simulation | MANO mesh and ray tracing | visible simulator uses skeleton points, calls undefined `get_index`, and casts complex echo to real default dtype | do not reuse as canonical simulation |

The numerical and provenance audit is tracked in
`paper/manager/evidence/radar_contract_audit.md`.

## Required Evidence Before Beamforming

- exact hardware profile and chirp/frame configuration for every collected data version;
- Tx/Rx order, TDM schedule, virtual-channel order, removed/duplicated channels, and bad-channel mask;
- virtual-array coordinates and units, phase/range calibration, and radar-camera extrinsics;
- definition and orientation of azimuth/elevation grids;
- whether stored arrays are pre-range-FFT or raw ADC and how complex values are serialized;
- original beamforming checkpoint/config and an input/output fixture with checksum;
- a confirmed conjugation convention consistent across steering-vector construction and multiplication.

Until these are available, the canonical processor stops at the complex range-Doppler spectrum.
