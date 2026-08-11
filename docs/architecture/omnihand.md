# Canonical OmniHand Reconstruction

Status: `engineering_vertical_slice`
Last Updated: `2026-08-11`

## Boundary

The canonical model starts after construction of a physically valid radar cube. It does not resolve
data paths, infer acquisition parameters, perform beamforming, write checkpoints, or parse CLI input.

```text
input   [batch,time,doppler,range,azimuth,elevation]
        finite non-negative radar-cube power
output  [batch,2,24,3]
        left/right dual-hand pose in metres
```

The first three joints are arm shoulder, arm elbow, and arm wrist. Joint `3` is the hand wrist; joints
`4..23` are the five four-joint fingers. Coordinate-frame identity belongs to the manifest and adapter
boundary and must be explicit before a real dataset is accepted.

## Model

- Frame encoder: a 3D stem followed by depthwise-separable residual stages.
- Attention: channel, spatial, and squeeze-excitation modules are separate Boolean configuration axes
  so each module can be removed independently in `EXP-REV-005`.
- Neck: optional shape-safe 3D PAFPN with explicit interpolation to each target level.
- Temporal encoder: a mask-aware transformer with learned position embeddings.
- Aggregation: a learned mixture of CLS, valid-frame mean, and temporal-attention summaries.
- Head: layer normalization and linear regression to two 24-joint hands.

The versioned engineering config uses 10 frames, 8 transformer layers, and 16 heads to match the
current manuscript description. Spatial dimensions and channel widths are deliberately compact because
the config is a two-step integration smoke rather than a production training recipe.

## Objective And Metrics

The smoke objective is masked coordinate L1 in metres. The metric protocol
`mmprism.pose_metric.dual_hand_metric_v1` reports:

- absolute MPJPE in millimetres over valid joints;
- wrist-relative MPJPE over finger joints;
- wrist-relative PCK at an explicitly configured millimetre threshold;
- sample-level absolute MPJPE.

Wrist-relative metrics use joint `3` as the root and exclude the wrist itself. This corrects the legacy
evaluation path that labelled the mean of hand joints as a wrist.

## Engineering Smoke

`mmprism omnihand-smoke` requires a clean Git commit and records the resolved config fingerprint,
command, Git state, Python/PyTorch/CUDA runtime, deterministic seed, input hashes, per-step losses and
metrics, gradients, parameter deltas, sample metrics, timing, and peak CUDA memory. It also checks a
single-frame path and proves that changing masked padding cannot change the final pose.

This artifact can establish that the canonical model is runnable. Synthetic smoke metrics cannot be
used as paper evidence, and the smoke does not close the beamforming, calibration, dataset, split,
checkpoint, or real-world generalization blockers.
