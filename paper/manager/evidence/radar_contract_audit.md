# Radar Tensor and Processing Contract Audit

Status: `range_doppler_evidence_ready_beamforming_blocked`
Last Updated: `2026-08-11`
Role: `ARCH-001-C_ARCH-003-B_radar_contract_evidence`

## 1. Audited Sources

- current manuscript `paper/manuscript/chapter/3_methods.tex`, especially lines 3-22;
- legacy acquisition/simulator constants in `config/radar/iwr1843.py`;
- legacy range, Doppler, beamforming, simulation, and array code in `src/fmcw/simulator.py`;
- legacy collected-data fallbacks in `src/data/dataset.py`;
- canonical implementation in `src/mmprism/contracts/tensors.py` and
  `src/mmprism/radar/processing.py`.

The legacy files remain forensic references and were not modified or imported by canonical code.

## 2. Confirmed Canonical Boundary

The rebuilt package now validates the following explicit contracts:

```text
raw ADC:       complex [..., chirp, antenna, sample]
range-Doppler: complex [..., doppler, antenna, range]
radar cube:    real non-negative [..., doppler, range, azimuth, elevation]
dual pose:     float [..., 2, 24, 3], metres, named coordinate frame
features:      float [..., time, feature]
caption:       non-empty Unicode plus language tag
```

The metric pose contract fixes hand order to left/right, the 24-joint order to three arm joints plus the
standard 21 hand joints, and coordinates to `[x,y,z]`. Existing CSL-News RTMW3D-derived arrays only have
confirmed shape/order compatibility; their units and coordinate frame are not yet proven to satisfy this
contract.

The canonical NumPy processor implements range FFT, optional pre-Doppler slow-time mean subtraction,
Doppler FFT/shift, periodic-Hann or rectangular windows, zero padding, and explicit range selection.
It preserves complex64/complex128 precision and does not mutate input.

Analytic tests establish exact integer-bin range/Doppler peaks, static-clutter removal, leading-axis
behavior, range selection, dtype preservation, non-mutation, and invalid input/config rejection. These
tests prove the stated processor contract; they do not prove equivalence to an unrecovered historical run.

## 3. Material Discrepancies

| ID | Finding | Evidence and impact | Status |
|---|---|---|---|
| `RADAR-AUDIT-001` | chirp-count conflict | collected fallbacks use 128 chirps; legacy radar config uses 64 | unresolved |
| `RADAR-AUDIT-002` | bandwidth conflict | manuscript states 3.85 GHz/3.89 cm; legacy `70e12*(55-5) us` gives 3.5 GHz/4.29 cm | unresolved |
| `RADAR-AUDIT-003` | sample-span nuance | 256 samples at 5.209 MHz span about 49.15 us, so endpoint/usable-bandwidth convention must be recovered | unresolved |
| `RADAR-AUDIT-004` | clutter-removal order | manuscript subtracts a mean over Doppler bins after 2D FFT; legacy subtracts slow-time mean before Doppler FFT | unresolved |
| `RADAR-AUDIT-005` | antenna geometry conflict | manuscript states 86 horizontal and 4 vertical; legacy default array list has 116 unique 2D coordinates | unresolved |
| `RADAR-AUDIT-006` | steering conjugation conflict | manuscript specifies `A^H X`; legacy `einsum('bdar,aw->bdrw', X, A)` does not visibly conjugate `A` | unresolved |
| `RADAR-AUDIT-007` | visible simulator is not executable evidence | `mmSimulator.init()` calls undefined `get_index`; `Simulation` lacks the `array_size` argument passed by the model | unresolved |
| `RADAR-AUDIT-008` | simulated complex echo loss | legacy echo is complex before `.to(self.dtype)`, whose default is float32 | unresolved |

## 4. Evidence Gate

Canonical beamforming, physical range/velocity axes, and simulation remain blocked until the project
receives acquisition configurations, channel mapping, array/calibration data, coordinate frames, and a
verified historical fixture. A new implementation must then be tested against both analytic array signals
and the recovered fixture, with the conjugation convention stated explicitly.

No manuscript number or method claim is promoted from this audit. In particular, the existence of a
tested range-Doppler transform is not evidence that the reported 4D cubes or paper metrics were reproduced.
