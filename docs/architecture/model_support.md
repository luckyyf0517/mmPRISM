# Language Model Support Boundary

Status: `mt5_formal_single_device_resume_ready_phi3_excluded`
Last Updated: `2026-08-11`

## Decision

The canonical rebuild targets mT5 only. Phi-3 is not a supported mmPRISM backend and is excluded from
the reviewer release. The legacy class remains read-only forensic material until the original submission
audit is complete; it does not justify a public support claim.

This choice follows the greenfield policy: an incomplete backend with no runnable config, training path,
evaluation path, or checkpoint provenance is removed from the supported surface instead of receiving a
compatibility shim. Reintroducing Phi-3 would require all of the following:

1. a canonical `mmprism.models` implementation with the same explicit tensor contract as the mT5 path;
2. strict versioned configuration and pinned tokenizer/base-model assets;
3. two-batch train/generate/evaluate integration tests and a real checkpoint smoke;
4. artifact provenance, sample-level predictions, metric protocol and compute profile;
5. reviewer README commands and a clean release audit.

## Enforced Surface

- Canonical README: mT5 is the only supported generation backend and has a bounded engineering smoke.
- Reviewer inventory: legacy model namespaces, legacy config, CLAUDE and root entrypoints are excluded.
- Release content gate: the encoded `unsupported_language_backend` rule rejects a Phi-3 name in any
  selected README, config, script, canonical source or test.
- Release deliverable gate: `configs/examples/mt5_smoke.yaml`, the pinned asset config, downloader and
  runner are selected and hashed; the clean release audit no longer reports an mT5 missing path.

## Verified mT5 Boundary

The canonical path implements a dual-hand ST-GCN pose encoder, 1024-D radar feature projector,
confidence-aware fusion and mT5 wrapper. A clean-commit A100 smoke completed two finite cross-entropy
steps, verified nonzero gradients and parameter changes in all three adapter paths, forced the pose gate
to zero for zero-confidence input, and generated two sample-level beam predictions.

The smoke freezes all 582,401,280 language-model parameters and updates 2,170,432 adapter parameters.
Clean commit `e31000b` additionally completed manifest-driven single-device train, adapter-only
Safetensors, reload, sample prediction, Unicode character metric, and standalone evaluation. That formal
run is still a synthetic engineering fixture, not a real dataset result, production metric protocol, or
paper-facing value. Single-process completed-epoch resume is implemented for the same adapter-only model
scope; real-data and distributed model/checkpoint requirements remain open under `ARCH-003`, `ARCH-006`,
and the experiment registry.

SBERT and SimCSE are evaluator assets, not caption-generation backends. Their loader readiness is tracked
separately under `EVID-CODE-MODELS-V1`.
