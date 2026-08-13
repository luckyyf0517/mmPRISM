# Historical CSL-Daily Checkpoint Namespace Inspection

Status: preliminary read-only inspection
Owner: WaveLLM training lane
Evidence scope: Modal namespaces observed in three inbound historical CSL-Daily checkpoints; no completeness,
configuration, loading, metric, or manuscript-linkage claim.
Recorded: 2026-08-13

## Purpose

The first CSL-Daily reconstruction loop needs to choose whether checkpoint-bound CubeNet features are a blocking
input. This inspection checks only whether the received daily WaveLLM state archives visibly contain the legacy
feature/fusion module namespaces. It supports execution priority `DEC-048`; it does not accept an inbound checkpoint
under `DEC-046`.

## Method

For each immutable input below, inspect the ZIP central directory and read only the serialized
`mp_rank_00_model_states/data.pkl` bytes. Search its state-key namespace strings for the legacy module prefixes:

```text
hand_pose_encoder.
feature_projection.
fusion_module.
```

No input was copied, loaded through legacy code, converted, or modified. A safe PyTorch `weights_only=True`
metadata load was attempted but rejected the historical `easydict.EasyDict` serialization representation. The result
below therefore remains a namespace inspection, not a tensor deserialization or controlled load.

## Observations

| Inbound run directory | `hand_pose_encoder.` observed | `feature_projection.` observed | `fusion_module.` observed | Interpretation boundary |
|---|---:|---:|---:|---|
| `wavellm_mt5_daily_0612` | yes | no | no | Consistent with legacy pose-only module construction. |
| `wavellm_mt5_daily_0702_gt` | yes | no | no | Consistent with legacy pose-only module construction. |
| `wavellm_mt5_daily_0826` | yes | no | no | Consistent with legacy pose-only module construction. |

In the preserved legacy implementation, enabling features constructs `feature_projection`; enabling both predicted
pose and features also constructs `fusion_module`. See [legacy trainer](../../../../../../legacy/src/model/trainer.py)
at the feature construction and fusion branch. The observed namespace pattern is therefore sufficient to prioritize
the pose-only route, but not to reconstruct each run's resolved configuration or data inputs.

## Consequences And Limits

- The first canonical CSL-Daily loop is `annotation_v1` -> skeleton simulation -> OmniHand -> pose-only WaveLLM,
  with camera-pose and cross-fitted predicted-mmWave-pose controls.
- Checkpoint-bound feature/fusion remains a separate third-stage experiment. It is not removed and must still meet
  its manifest, cross-fitting, and exact-alignment contract before use.
- This inspection does **not** prove model state completeness, DeepSpeed world size, tensor compatibility, input
  data mode, historical split, evaluation protocol, metric validity, original-submission linkage, or loadability.
- The archive remains preservation-only until its stable receipt, format/world-size audit, metadata/tensor audit,
  and controlled load are completed under `DEC-046`.
