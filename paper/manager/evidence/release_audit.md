# Reviewer Release Audit Evidence

Status: `boundary_verified_two_required_deliverables_blocked`
Last Updated: `2026-08-11`
Role: `ARCH-001-A_ARCH-001-B_ARCH-REV-003_R2-CODE-3_evidence`
Evidence ID: `EVID-CODE-RELEASE-V1`

## 1. Snapshot Identity

```text
schema: mmprism.release_audit_report.v1
release profile: reviewer_release_v1
builder commit: 81e9b89896a25bc26eece5f789b9a842004a4d4a
builder Git state: clean
config: configs/release/reviewer_release_v1.yaml
config fingerprint: 1126cd2a33a6ef0a18c6271e3a12cd29888a724d8df2074bf52b040beb4a699a
artifact: paper/manager/evidence/artifacts/release_audit_v1.json
artifact SHA-256: f569badc56555ac2dc936c7f74881fcefaf442eb44bd89a49f662fe6ea533a17
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
| Git tracked files inspected | 285 |
| release-selected files | 94 |
| selected bytes | 881,507 |
| selected files with size + SHA-256 | 94/94 |
| tracked internal/legacy paths explicitly excluded | 154 |
| canonical Python modules | 46 |
| internal dependency edges | 86 |
| missing canonical import targets | 0 |
| canonical imports of forbidden legacy namespaces | 0 |
| relative canonical imports | 0 |
| import cycles | 0 |
| forbidden selected paths | 0 |
| local absolute path/token content hits | 0 |
| unsupported backend content hits | 0 |
| expected `mmprism = mmprism.cli:main` entrypoint | matched |

The 154 excluded tracked paths include `CLAUDE.md`, `AGENTS.md`, the manuscript/revision area, root
legacy entrypoints/configuration, legacy `src/*` namespaces, and internal operational scripts. They remain
in the development repository for evidence recovery and are absent from the release selection.

## 3. Open Release Blockers

The profile failed on exactly two `REQUIRED_PATH_MISSING` findings:

1. `LICENSE`: author approval is still required (`OPS-REV-002`, `R2-CODE-4`).
2. `configs/examples/radar_smoke.yaml`: complete portable radar/cube path remains blocked on acquisition
   and calibration provenance.
These are release requirements, not placeholders to satisfy by creating empty files. Each path must become
runnable and pass a clean-environment smoke before the report may turn green.

The former mT5 missing-path blocker is closed. The profile now selects and hashes
`configs/examples/mt5_smoke.yaml`, `configs/models/mt5_base_v1.yaml`, `scripts/download_mt5.sh`,
`scripts/run_mt5_smoke.sh`, canonical model/training modules and their tests. Fixed-revision asset and
A100 execution evidence is recorded as `EVID-CODE-MT5-SMOKE-V1`.

The profile also selects `configs/examples/omnihand_smoke.yaml`, `scripts/run_omnihand_smoke.sh`,
canonical CubeNet/config/metric/smoke modules and their tests. Clean-commit A100 execution evidence is
recorded as `EVID-CODE-OMNIHAND-SMOKE-V1`.

The current profile additionally selects the strict model-ready pose adapter, OmniHand formal run config,
single-device train/evaluate orchestration, Safetensors support, and their tests. Clean-commit A100/BF16
train/checkpoint/reload/evaluate evidence is recorded as `EVID-CODE-OMNIHAND-FORMAL-V1`.

`scripts/download_models.sh` and `configs/models/evaluation_models_v1.yaml` are now selected and hashed
by the release profile. Their fixed-revision download, checksum and two-loader execution evidence is
recorded separately as `EVID-CODE-MODELS-V1`; therefore `ARCH-REV-002` is no longer a release blocker.

## 4. Evidence Boundary

- This report verifies selection, hashes, text/path safety, entrypoints, and static dependency structure.
- This audit does not itself create a ZIP, test installation inside a clean container, download models, or execute
  prepare/train/evaluate.
- It does not close `R2-CODE-3` until the two blockers are resolved and a final reviewer archive passes the
  same audit plus clean-environment execution.
- Phi-3 is intentionally unsupported, excluded from the selected surface and protected by a zero-hit
  content regression gate. See `EVID-CODE-MODEL-SUPPORT-V1`.
