# Reviewer Release Audit Evidence

Status: historical
Owner: mmPRISM coordinator
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12
Legacy evidence role: `ARCH-001-A_ARCH-001-B_ARCH-REV-003_R2-CODE-3_evidence`
Evidence ID: `EVID-CODE-RELEASE-V1`

## 1. Snapshot Identity

```text
schema: mmprism.release_audit_report.v1
release profile: reviewer_release_v1
builder commit: 4e5e3abd94d803d533e51d7d6df58e4590aecf97
builder Git state: clean
config: configs/release/reviewer_release_v1.yaml
config fingerprint: 0c3ff43885d441ef1af918b1f57d38be8677e8b1919d620978a2775d2cd3915d
artifact: paper/manager/evidence/artifacts/release_audit_v1.json
artifact SHA-256: dd14960f5f331e827a2ebc79ea55b7bf6be718e566c8fc2f816e2bb744fe8951
status: failed (expected blockers retained)
```

Reproduction command:

```bash
uv run --frozen --extra train mmprism release-audit \
  configs/release/reviewer_release_v1.yaml \
  --project-root "$PWD" \
  --output /tmp/mmprism-release-audit.json
```

Exit code `1` means the structured audit completed but the release gate did not pass. Configuration or
execution failure would return `2`.

## 2. Verified Boundary

| Gate | Result |
|---|---:|
| Git tracked files inspected | 304 |
| release-selected files | 107 |
| selected bytes | 1,053,234 |
| selected files with size + SHA-256 | 107/107 |
| tracked internal/legacy paths explicitly excluded | 158 |
| canonical Python modules | 51 |
| internal dependency edges | 105 |
| missing canonical import targets | 0 |
| canonical imports of forbidden legacy namespaces | 0 |
| relative canonical imports | 0 |
| import cycles | 0 |
| forbidden selected paths | 0 |
| local absolute path/token content hits | 0 |
| unsupported backend content hits | 0 |
| expected `mmprism = mmprism.cli:main` entrypoint | matched |

The 158 excluded tracked paths include `CLAUDE.md`, `AGENTS.md`, the manuscript/revision area, root
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

The profile now includes the CPU-only CSL-News annotation identity audit module and its tests. The audit
freezes the published sidecar set, streams artifact hashes, records clean Git provenance and reports every
invalid pair; the pose-manifest builder consumes only checksum-bound exclusion evidence and does not weaken
ordinary checksum validation.

The profile now also selects strict sign-language-translation manifests, WaveLLM formal configs and
launcher, the single-device train/evaluate implementation, character-metric protocol, and their tests.
Clean-commit A100/BF16 adapter-checkpoint/reload/evaluate evidence is recorded as
`EVID-CODE-WAVELLM-FORMAL-V1`.

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
