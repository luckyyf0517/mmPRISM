# Formal Run Preflight Evidence

Status: historical
Owner: mmPRISM coordinator
Evidence scope: Immutable migration snapshot or dated evidence retained by this log.
Recorded: 2026-08-12
Legacy evidence role: `ARCH-002-D_formal_preflight_evidence`
Evidence ID: `EVID-CODE-FORMAL-PREFLIGHT-V1`

## 1. Scope

This evidence verifies the dependency-light, side-effect-free gate used before canonical formal runs and
the split-provenance enforcement repeated by OmniHand and WaveLLM train/evaluate services. It proves an
engineering orchestration contract, not a model result, dataset-completeness claim, or paper metric.

## 2. Identity

```text
implementation commit: 766453bcd51b2f72b440c5530f0dc72669fe7ea1 (clean)
report schema: mmprism.prepare_report.v1
experiment config SHA-256: a292312fa29b82fbd0e61623c008e68ecf2c831fb1b56795e8586c42ad531888
manifest SHA-256: 9abd9779e7924e553174d24b26f1805c01cf2d40176471ba1c1a75731e3b71f0
split assignment SHA-256: 6386d3efc00879036035f420a588f8169e4839d3ecb40849e17e8689edfc76d5
status: passed
```

## 3. Clean-Commit CLI Check

```bash
MMPRISM_DATA_ROOT=/home/yanyifan/mmPRISM \
MMPRISM_ARTIFACT_ROOT=/tmp/mmprism-prepare-artifacts \
MMPRISM_CACHE_ROOT=/tmp/mmprism-prepare-cache \
uv run --frozen mmprism prepare configs/examples/pose_smoke.yaml \
  --input manifest:data_manifest=tests/fixtures/manifests/pose_smoke.jsonl \
  --input split:split_assignments=tests/fixtures/splits/pose_smoke.jsonl \
  --split-binding data_manifest=train
```

The report recorded `git.dirty=false`, one valid manifest record, one valid split assignment, one bound
sample, zero unbound assignments, and an absent planned run directory. Both `/tmp` destinations were absent
before and after the command; the report identified `/tmp` as their writable ancestor without creating
either destination.

## 4. Automated Verification

```text
Ruff: passed
Mypy: 52 source files, no issues
Pytest: 187 passed
git diff --check: passed
```

The tests include successful and failed split membership, dirty-Git rejection, incomplete manifest-binding
coverage, cross-manifest sample overlap, side-effect checks, dependency-light artifact import, and assertions
that both training and evaluation `inputs.json` files register `split_assignments`.

## 5. Evidence Boundary

- The fixture has one synthetic manifest record and is not a training dataset.
- `prepare` emits a report to stdout; it does not create or complete a formal run.
- Formal train/evaluate services repeat the relevant membership validation because repository state or input
  bytes can change after preflight.
- No value from this check is eligible for the manuscript, response letter, figures, or Source Data.
