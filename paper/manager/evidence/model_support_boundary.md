# Language Model Support Boundary Evidence

Status: `evidence_ready`
Last Updated: `2026-08-11`
Role: `ARCH-REV-004_R2-CODE-5_evidence`
Evidence ID: `EVID-CODE-MODEL-SUPPORT-V1`

## Direct Decision

The revision release does not claim or ship Phi-3 support. mT5 is the sole caption-generation rebuild
target, and its bounded portable engineering smoke now passes. Legacy Phi-3 source stays in the internal forensic repository only; it is not a canonical module,
configuration, entrypoint, example, or README feature.

This resolves the reviewer's reproducibility concern by removing an unsupported public claim rather than
adding an unverified example around a legacy class.

## Enforced Evidence

```text
decision: DEC-027
architecture: docs/architecture/model_support.md
release profile: configs/release/reviewer_release_v1.yaml
release builder commit: 79b45b58d803b3b07a8b7476f87c208e6f17399d (clean)
release config fingerprint: 1c39365a3109b2ca34a6597b3ee25bc22d674dd6d78e270a3baae8d60e468abf
release artifact: paper/manager/evidence/artifacts/release_audit_v1.json
release artifact SHA-256: 5bea122691306153bbadc9ba2cd5f3bdefca0353c9dba0394372e331acee89ba
selected files: 76
FORBIDDEN_CONTENT findings: 0
other unexpected findings: 0
```

The release profile encodes the unsupported backend name as a regex that does not match its own config
text. Any selected README, config, script, canonical source, or test that reintroduces the name produces a
`FORBIDDEN_CONTENT` finding. `test_release_audit_rejects_an_unsupported_model_claim` verifies file and line
reporting for this gate.

The updated release report explicitly lists legacy `src/model/llm/phi3_model.py`, `CLAUDE.md`, legacy configs,
and root entrypoints under tracked-but-excluded internal material. Static dependency analysis covers all
39 selected canonical modules and reports zero legacy imports, missing targets, relative imports, or cycles.

Canonical mT5 execution evidence is recorded separately as `EVID-CODE-MT5-SMOKE-V1`. It binds the
fixed upstream revision, checksum manifests, clean Git commit, two finite adapter updates, confidence
counterfactual and sample-level beam outputs. This evidence does not claim full paper-training readiness.

## Remaining Boundary

- The mT5 required path is present and verified; release audit no longer reports it as a blocker.
- No statement here claims that the production WaveLLM/mT5 training and evaluation path is complete.
- Reintroducing Phi-3 requires a new decision plus canonical contract, pinned assets, runnable config,
  training/generation/evaluation integration, checkpoint provenance and clean reviewer smoke.
- Final closure of `R2-CODE-5` still requires response-letter text and final reviewer archive verification.
