# Language Model Support Boundary

Status: `mt5_target_phi3_excluded`
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

- Canonical README: mT5 is described only as the rebuild target, not as runnable yet.
- Reviewer inventory: legacy model namespaces, legacy config, CLAUDE and root entrypoints are excluded.
- Release content gate: the encoded `unsupported_language_backend` rule rejects a Phi-3 name in any
  selected README, config, script, canonical source or test.
- Release deliverable gate: `configs/examples/mt5_smoke.yaml` remains required and missing until the
  actual mT5 vertical slice passes.

SBERT and SimCSE are evaluator assets, not caption-generation backends. Their loader readiness is tracked
separately under `EVID-CODE-MODELS-V1`.
