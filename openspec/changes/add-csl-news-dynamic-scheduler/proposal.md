## Why

The current CSL-News annotation operation assigns archives through
`archive_id % worker_count`. That makes the worker count part of placement,
so changing capacity requires a coordinated restart and redistributes live
work. It also has no durable operator control for pausing a long rebuild.

## What Changes

- Add a filesystem-backed, source-agnostic archive lease scheduler for
  CSL-News annotation.
- Make worker capacity elastic: each worker claims one eligible archive at a
  time, independently of a fixed worker index/count.
- Add explicit `paused`/`running` control and cooperative pausing at the
  boundary between video samples.
- Expose scheduler initialization, control, status, and dynamic-worker CLI
  commands, plus a GPU-selection worker wrapper.
- Update the CSL-News operation runbook and current workspace state.

## Non-Goals

- Do not modify raw ZIPs, `.part` download checkpoints, scratch videos,
  completed artifacts, failure records, or historic conflict variants.
- Do not create a final training dataset, change visual-pose output schemas,
  or alter the source-integrity gate.
- Do not introduce a central service, database daemon, or GPU resource
  scheduler. Existing process supervisors remain responsible for process
  lifetime and GPU placement.

## Safety Boundary

The scheduler only controls which integrity-passed archive a worker may
attempt. Existing per-sample atomic NPZ/sidecar publication remains the
durable completion authority. A worker owns a time-bounded archive lease;
on an unclean exit, the lease is reclaimed only after expiry and completed
samples are skipped by the existing source-bound output validation.
