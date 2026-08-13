# OmniHand Training Changelog

Status: current
Owner: OmniHand training lane
Authority scope: Material changes to the OmniHand training and evaluation workflow.
Last reviewed: 2026-08-13

## 2026-08-12

- Established a focused training workspace without moving canonical model or training code.
- Classified existing synthetic smoke and formal-run evidence as historical engineering Logs.
- Added the shared distributed formal-run lifecycle and accepted a two-process CPU/Gloo OmniHand integration test;
  DDP resume and multi-GPU NCCL remain open.
- Defined the CSL-Daily synthetic-control consumption and the checkpoint-bound cross-fitted prediction/feature
  export required before WaveLLM can evaluate reconstructed-pose and fusion inputs.

## 2026-08-13

- Applied `DEC-048`: cross-fitted predicted poses are now the first CSL-Daily WaveLLM handoff. Checkpoint-bound
  frame features remain required only for the later fusion comparison and do not block pose-only reconstruction or
  translation training.
- Applied `DEC-049`: formal CSL-Daily OmniHand inputs are persistent pre-beamforming synthetic FMCW; the runtime
  adapter, not the delivery, produces the transient CubeNet power cube.
