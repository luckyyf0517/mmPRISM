## 1. Relocation

- [x] 1.1 Verify the boundary file list (legacy-only vs current tooling).
- [x] 1.2 Move legacy code into `legacy/` preserving content (no deletion).
- [x] 1.3 Add `legacy/README.md` with forensic status and boundary rules.

## 2. Reference Updates

- [x] 2.1 Repoint `scripts/simulation/freeze_legacy_equivalence_fixture.py` at `legacy/`.
- [x] 2.2 Update release-audit forbidden paths/import prefixes and `configs/README.md`.
- [ ] 2.3 Update AGENTS.md, CLAUDE.md, and current Authority path references; add changelog and
      decision-log entries.

## 3. Verification

- [ ] 3.1 Run pytest, ruff, mypy, release audit, docs audit, and `git diff --check`.
- [ ] 3.2 Confirm legacy top-level packages no longer import from a clean interpreter.
