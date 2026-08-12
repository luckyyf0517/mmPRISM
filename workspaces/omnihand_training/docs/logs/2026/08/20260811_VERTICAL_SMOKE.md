# OmniHand CubeNet Vertical Smoke Evidence

Status: historical
Owner: OmniHand training lane
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12
Legacy evidence role: `ARCH-004_EXP-001_engineering_evidence`
Evidence ID: `EVID-CODE-OMNIHAND-SMOKE-V1`

## 1. Scope

This evidence verifies the executable canonical CubeNet/OmniHand reconstruction boundary. It does not
reproduce a paper result. The run starts from deterministic random initialization, consumes synthetic
non-negative radar-cube power, and performs two optimizer steps against synthetic metric-pose targets.

## 2. Reproduction

```bash
export MMPRISM_ARTIFACT_ROOT=/path/to/mmprism-runs
MMPRISM_DEVICE=cuda:0 scripts/run_omnihand_smoke.sh
```

The smoke requires a clean Git worktree. The versioned config contains no machine path and fixes a
10-frame, 8-layer, 16-head temporal model. Artifact root and device remain runtime inputs.

## 3. Verified Identity

```text
implementation commit: 688d44d18d7441c0c79706546e97683db2713ce9 (clean)
config: configs/examples/omnihand_smoke.yaml
config fingerprint: 9111d597d37b2eab5662b2c492a5ce09caef21733afbac2c0540d419df39c13b
seed: 20260811
Python / Torch / CUDA: 3.12.13 / 2.11.0+cu128 / 12.8
device / precision: NVIDIA A100-SXM4-80GB GPU 5 / bfloat16 autocast
artifact: paper/manager/evidence/artifacts/omnihand_smoke_v1.json
artifact SHA-256: 0e49867864c36a65baf4c77fe838edd85f1969b6ade0c9012de104ac4126e389
status: passed
```

The tracked artifact is a strict-JSON copy with the machine-specific command path removed. The original
and independent replicate remain in the mounted artifact root.

## 4. Verified Result

| Gate | Result |
|---|---:|
| model parameters | 679,097 trainable / 0 frozen |
| input / output | `[2,10,8,16,12,8]` / `[2,2,24,3]` |
| finite masked L1 steps | `0.501012`, `0.448130` metres |
| spatial gradient norms | `0.728973`, `0.445822` |
| temporal gradient norms | `0.571176`, `0.414951` |
| pose-head gradient norms | `0.646890`, `0.646585` |
| tracked parameter max deltas | all three paths approximately `0.002001` |
| single-frame output | finite `[2,2,24,3]` |
| masked-padding counterfactual | maximum absolute prediction difference `0.0` |
| metric protocol | `mmprism.pose_metric.dual_hand_metric_v1` |
| sample-level metric records | 2/2 present and finite |
| peak allocated / reserved CUDA memory | 48,904,704 / 77,594,624 B |
| two-step elapsed time | 1.369889 s |

Channel, spatial, and squeeze-excitation attention were enabled in the smoke and are independent config
axes with leave-one-out integration tests. PAFPN odd-shape behavior, temporal mask invariance, vectorized
frame processing, forward/backward, and hand-wrist-root metric semantics are covered by the automated
suite. The complete suite passed 135 tests with Ruff, strict Mypy, sdist, and wheel checks.

An independent rerun at the same commit and seed produced an identical normalized JSON after removing
only command and wall-clock throughput fields. Both normalized payloads have SHA-256
`282bd16e72ea34dfc9ae8882c3a5a1ebb9a5078a54f4e08b7ced213730c223a1`.

## 5. Failure Trace

The first clean run at `e082e8138b8f347a46c3f56037f1323edd85d0f5` completed training but exposed
a missing BF16 autocast context in post-training inference. It failed before atomic report promotion and
left no JSON artifact. Commit `688d44d` changed synthetic input storage to FP32 and applied the configured
autocast consistently to training and inference; two subsequent runs passed deterministically.

## 6. Evidence Boundary

- The finite synthetic MPJPE/PCK values only test the metric path and are not scientific performance.
- No real or simulated paper dataset manifest, split, coordinate frame, calibration, or beamformed cube
  is attached.
- No checkpoint, distributed prediction writer, confidence estimator, or production scheduler is tested.
- `ARCH-004` is engineering-evidence ready; full training and paper reproduction remain gated by
  `BLOCK-RADAR-PROVENANCE`, `BLOCK-DATA-ROOT`, and `BLOCK-PROVENANCE`.
