# Reviewer Release Audit Evidence

Status: `boundary_verified_three_required_deliverables_blocked`
Last Updated: `2026-08-11`
Role: `ARCH-001-A_ARCH-001-B_ARCH-REV-003_R2-CODE-3_evidence`
Evidence ID: `EVID-CODE-RELEASE-V1`

## 1. Snapshot Identity

```text
schema: mmprism.release_audit_report.v1
release profile: reviewer_release_v1
builder commit: 288955e70d3afca0c4eb3ca2ce8bcb1f250b3e46
builder Git state: clean
config: configs/release/reviewer_release_v1.yaml
config fingerprint: 2c8d073ea0e32566f17614e006bff7aec7a1a25d807407f83f944d31a687a71a
artifact: paper/manager/evidence/artifacts/release_audit_v1.json
artifact SHA-256: a11cbf0aa8f0cadab07fc25c57bfde680c3438bec6d4399b4f0c76265cda8a26
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
| Git tracked files inspected | 250 |
| release-selected files | 66 |
| selected bytes | 651,268 |
| selected files with size + SHA-256 | 66/66 |
| tracked internal/legacy paths explicitly excluded | 149 |
| canonical Python modules | 35 |
| internal dependency edges | 54 |
| missing canonical import targets | 0 |
| canonical imports of forbidden legacy namespaces | 0 |
| relative canonical imports | 0 |
| import cycles | 0 |
| forbidden selected paths | 0 |
| local absolute path/token content hits | 0 |
| expected `mmprism = mmprism.cli:main` entrypoint | matched |

The 149 excluded tracked paths include `CLAUDE.md`, `AGENTS.md`, the manuscript/revision area, root
legacy entrypoints/configuration, legacy `src/*` namespaces, and internal operational scripts. They remain
in the development repository for evidence recovery and are absent from the release selection.

## 3. Open Release Blockers

The profile failed on exactly three `REQUIRED_PATH_MISSING` findings:

1. `LICENSE`: author approval is still required (`OPS-REV-002`, `R2-CODE-4`).
2. `configs/examples/radar_smoke.yaml`: complete portable radar/cube path remains blocked on acquisition
   and calibration provenance.
3. `configs/examples/mt5_smoke.yaml`: canonical mT5 train/evaluate vertical slice is not implemented.

These are release requirements, not placeholders to satisfy by creating empty files. Each path must become
runnable and pass a clean-environment smoke before the report may turn green.

`scripts/download_models.sh` and `configs/models/evaluation_models_v1.yaml` are now selected and hashed
by the release profile. Their fixed-revision download, checksum and two-loader execution evidence is
recorded separately as `EVID-CODE-MODELS-V1`; therefore `ARCH-REV-002` is no longer a release blocker.

## 4. Evidence Boundary

- This report verifies selection, hashes, text/path safety, entrypoints, and static dependency structure.
- This audit does not itself create a ZIP, test installation inside a clean container, download models, or execute
  prepare/train/evaluate.
- It does not close `R2-CODE-3` until the four blockers are resolved and a final reviewer archive passes the
  same audit plus clean-environment execution.
- Phi-3 is not selected or advertised by the canonical README. A final support/removal decision remains
  separately tracked under `ARCH-REV-004`.
