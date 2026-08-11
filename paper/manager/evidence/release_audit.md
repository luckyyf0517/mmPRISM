# Reviewer Release Audit Evidence

Status: `boundary_verified_required_deliverables_blocked`
Last Updated: `2026-08-11`
Role: `ARCH-001-A_ARCH-001-B_ARCH-REV-003_R2-CODE-3_evidence`
Evidence ID: `EVID-CODE-RELEASE-V1`

## 1. Snapshot Identity

```text
schema: mmprism.release_audit_report.v1
release profile: reviewer_release_v1
builder commit: c49f7252d53f0e10c17c91b5795f4a336168f695
builder Git state: clean
config: configs/release/reviewer_release_v1.yaml
config fingerprint: bbc4924e64f597d171889ae689c8cbe86d8e37a67af1e1f28f1d6821df234af0
artifact: paper/manager/evidence/artifacts/release_audit_v1.json
artifact SHA-256: 57ad568e8223b26a8d2b4df3b7ec5325250ab3326e130cf008fcf2bcd2b48c9c
status: failed (expected blockers retained)
```

Reproduction command:

```bash
uv run mmprism release-audit configs/release/reviewer_release_v1.yaml \
  --output paper/manager/evidence/artifacts/release_audit_v1.json
```

Exit code `1` means the structured audit completed but the release gate did not pass. Configuration or
execution failure would return `2`.

## 2. Verified Boundary

| Gate | Result |
|---|---:|
| Git tracked files inspected | 240 |
| release-selected files | 61 |
| selected bytes | 602,934 |
| selected files with size + SHA-256 | 61/61 |
| tracked internal/legacy paths explicitly excluded | 145 |
| canonical Python modules | 33 |
| internal dependency edges | 51 |
| missing canonical import targets | 0 |
| canonical imports of forbidden legacy namespaces | 0 |
| relative canonical imports | 0 |
| import cycles | 0 |
| forbidden selected paths | 0 |
| local absolute path/token content hits | 0 |
| expected `mmprism = mmprism.cli:main` entrypoint | matched |

The 145 excluded tracked paths include `CLAUDE.md`, `AGENTS.md`, the manuscript/revision area, root
legacy entrypoints/configuration, legacy `src/*` namespaces, and internal operational scripts. They remain
in the development repository for evidence recovery and are absent from the release selection.

## 3. Open Release Blockers

The profile failed on exactly four `REQUIRED_PATH_MISSING` findings:

1. `LICENSE`: author approval is still required (`OPS-REV-002`, `R2-CODE-4`).
2. `scripts/download_models.sh`: supported evaluator/model acquisition is incomplete (`ARCH-REV-002`,
   `R2-CODE-2`).
3. `configs/examples/radar_smoke.yaml`: complete portable radar/cube path remains blocked on acquisition
   and calibration provenance.
4. `configs/examples/mt5_smoke.yaml`: canonical mT5 train/evaluate vertical slice is not implemented.

These are release requirements, not placeholders to satisfy by creating empty files. Each path must become
runnable and pass a clean-environment smoke before the report may turn green.

## 4. Evidence Boundary

- This report verifies selection, hashes, text/path safety, entrypoints, and static dependency structure.
- It does not create a ZIP, test installation inside a clean container, download models, or execute
  prepare/train/evaluate.
- It does not close `R2-CODE-3` until the four blockers are resolved and a final reviewer archive passes the
  same audit plus clean-environment execution.
- Phi-3 is not selected or advertised by the canonical README. A final support/removal decision remains
  separately tracked under `ARCH-REV-004`.
