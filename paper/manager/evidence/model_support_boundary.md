# Language Model Support Boundary Evidence

Status: `evidence_ready`
Last Updated: `2026-08-11`
Role: `ARCH-REV-004_R2-CODE-5_evidence`
Evidence ID: `EVID-CODE-MODEL-SUPPORT-V1`

## Direct Decision

The revision release does not claim or ship Phi-3 support. mT5 is the sole caption-generation rebuild
target, and it remains explicitly non-runnable until the required portable train/generate/evaluate smoke
exists. Legacy Phi-3 source stays in the internal forensic repository only; it is not a canonical module,
configuration, entrypoint, example, or README feature.

This resolves the reviewer's reproducibility concern by removing an unsupported public claim rather than
adding an unverified example around a legacy class.

## Enforced Evidence

```text
decision: DEC-027
architecture: docs/architecture/model_support.md
release profile: configs/release/reviewer_release_v1.yaml
release builder commit: 812c11716ecd65195d3c1d91933427b3b09af064 (clean)
release config fingerprint: 9f6fe0a0d5a5dd676c1819734edd8ef06b6aeaaa43cc84b3b89140370688a99f
release artifact: paper/manager/evidence/artifacts/release_audit_v1.json
release artifact SHA-256: e2742e31032c9378cbc44106cd405cf6f90107f20e74a590ac09458233d251ff
selected files: 66
FORBIDDEN_CONTENT findings: 0
other unexpected findings: 0
```

The release profile encodes the unsupported backend name as a regex that does not match its own config
text. Any selected README, config, script, canonical source, or test that reintroduces the name produces a
`FORBIDDEN_CONTENT` finding. `test_release_audit_rejects_an_unsupported_model_claim` verifies file and line
reporting for this gate.

The same release report explicitly lists legacy `src/model/llm/phi3_model.py`, `CLAUDE.md`, legacy configs,
and root entrypoints under tracked-but-excluded internal material. Static dependency analysis covers all
35 selected canonical modules and reports zero legacy imports, missing targets, relative imports, or cycles.

## Remaining Boundary

- `configs/examples/mt5_smoke.yaml` is still missing by design and remains one of three release blockers.
- No statement here claims that the canonical WaveLLM/mT5 vertical slice is complete.
- Reintroducing Phi-3 requires a new decision plus canonical contract, pinned assets, runnable config,
  training/generation/evaluation integration, checkpoint provenance and clean reviewer smoke.
- Final closure of `R2-CODE-5` still requires response-letter text and final reviewer archive verification.
