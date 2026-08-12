## Why

Legacy forensic code (original-submission reference) was scattered across the repository root
(`run_*.py`, `convert_npy.py`, `view_mmwave_cube.py`, `kill.sh`, `download_models.sh`), `config/`,
`src/{data,fmcw,model,eval,scripts,utils}`, and `scripts/{omnihand,wavellm}`. The scattered layout
made the legacy boundary hard to see, and `pyproject.toml`'s `packages.find where=["src"]` was
picking the legacy `src/` modules up as installable top-level packages (`data`, `fmcw`, `model`,
...) in the built distribution.

## What Changes

- Relocate all AGENTS.md-defined legacy forensic code into an explicit top-level `legacy/`
  directory, preserving content byte-identically (move only, nothing deleted):
  - root entry points and helpers -> `legacy/`
  - `config/` -> `legacy/config/`
  - `src/{data,fmcw,model,eval,scripts,utils}` -> `legacy/src/`
  - `scripts/{omnihand,wavellm}` -> `legacy/scripts/`
- Add `legacy/README.md` documenting forensic status, the boundary rules, and import mechanics.
- Repoint the one maintained legacy consumer,
  `scripts/simulation/freeze_legacy_equivalence_fixture.py`, at `legacy/`.
- Update the release-audit allowlist rules (`configs/release/reviewer_release_v1.yaml`) so the
  forbidden-path/import rules cover the new location (`legacy/**`, `legacy` import prefix).
- Update AGENTS.md / CLAUDE.md legacy-boundary wording and current Authority path references.
  Dated logs and frozen evidence keep their historical paths verbatim.

## Non-Goals

- Deleting, rewriting, or "fixing" any legacy file (documented discrepancies stay verbatim).
- Changing `src/mmprism/`, tests, or canonical configs behavior.
- Making legacy entry points runnable as a supported workflow; they remain read-only evidence.

## Impact

- `src/` contains only `mmprism` packages; built wheels/sdists no longer ship legacy modules.
- Legacy scripts are no longer importable from the repository root; forensic execution requires
  putting `legacy/` on `sys.path` (handled by the freeze-fixture script).
