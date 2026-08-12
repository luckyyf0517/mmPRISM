# Legacy Forensic Reference

This directory contains the original-submission codebase, preserved read-only as forensic
evidence for the manuscript evidence audit. It was relocated here from the repository root
(`run_*.py`, `config/`, `src/{data,fmcw,model,eval,scripts,utils}`, `scripts/{omnihand,wavellm}`)
on 2026-08-12. Contents are byte-identical to the pre-move state; only the location changed.

## Boundary (mirrors AGENTS.md "Legacy Boundary" and DEC-010)

- Do not add new features to legacy modules.
- Do not import legacy modules from `src/mmprism/`.
- Do not create compatibility shims unless a documented evidence-recovery task requires one.
- Preserve historical code until the original manuscript evidence audit is complete.

## Layout

- `run_*.py`, `convert_npy.py`, `view_mmwave_cube.py` — historical entry points (annotation,
  simulation, training, feature extraction, evaluation).
- `config/` — historical YAML/Python configuration (`omnihand/`, `radar/`, `wavellm/`).
- `src/` — historical modules: `data` (datasets), `fmcw` (radar simulator/processor/beamformer),
  `model` (OmniHand, CubeNet, WaveLLM), `eval` (metrics), `scripts`, `utils`.
- `scripts/` — historical torchrun/deepspeed wrappers (`omnihand/`, `wavellm/`).
- `kill.sh`, `download_models.sh` — historical ops helpers, superseded by `scripts/` tooling.

## Import mechanics

Legacy modules import each other via absolute names rooted at the old repository root
(`from src.fmcw.simulator import ...`, `from config.radar import ...`). To execute any legacy
entry point for forensic purposes, put this `legacy/` directory on `sys.path` (namespace
packages resolve `src.*` and `config.*` from there). The only maintained consumer is
`scripts/simulation/freeze_legacy_equivalence_fixture.py`, used to regenerate the frozen
numerical-equivalence fixture for `mmprism.simulation`.

## Known documented discrepancies (do not "fix")

- The manuscript describes MANO mesh + ray tracing; the code implements a skeleton
  point-reflector simulator (DEC-012; no MANO-equivalence claim is permitted).
- `src/fmcw/simulator.py` casts the complex echo to float32 (discards the imaginary part) and
  calls an undefined `get_index()`. Both are preserved verbatim and mirrored deliberately in
  `src/mmprism/simulation/`.
