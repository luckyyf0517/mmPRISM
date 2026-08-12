## 1. Control Plane

- [x] 1.1 Implement atomic scheduler control, lease claim/release, stale-lease
  recovery, and status reporting without optional ML dependencies.
- [x] 1.2 Add deterministic unit coverage for pause/resume, mutual exclusion,
  heartbeat, and stale-lease recovery.

## 2. Annotation Integration

- [x] 2.1 Add cooperative stop checks at sample boundaries to the annotation
  runner without altering its output contract.
- [x] 2.2 Add an elastic archive worker that consumes only registry-passed,
  source-incomplete candidates.
- [x] 2.3 Add thin CLI commands and the GPU-selection dynamic-worker wrapper.

## 3. Operations

- [x] 3.1 Update the runbook, architecture contract, workspace index, and
  changelog with pause/resume, elastic capacity, and recovery procedures.
- [x] 3.2 Run targeted CPU-only tests, static checks, documentation audit, and
  whitespace validation.
