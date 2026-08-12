## Decision

Use a small filesystem control plane rooted below the existing annotation
output root. It contains an atomically written control document, a short-held
POSIX advisory claim lock, and one lease document per active archive. It does
not persist a completed-task table: a valid, source-bound annotation archive
marker and the per-sample output contracts already provide that evidence.

## Queue Semantics

1. A worker reads the exact integrity registry and offers only passed archives
   whose current-source annotation marker is incomplete.
2. Under the claim lock, it rejects active leases, archives stale leases, and
   creates one durable lease for the selected archive.
3. The worker renews the lease before each video and checks the control state.
   `paused` prevents the next sample from starting; the current sample is
   allowed to finish and remains atomically published.
4. The worker releases the archive lease after an archive attempt. A later
   worker may resume it; completed source-bound samples are skipped.
5. A dead worker's lease becomes eligible only after `lease_seconds` with no
   heartbeat. The default is deliberately much longer than normal per-video
   inference.

The candidate sort is deterministic (`video_count` descending, then archive
ID), while concurrent claims are serialized only for the brief control-plane
operation. Workers do not own an index or a static shard. Starting an
additional worker therefore adds capacity without changing any running worker.

## Interfaces

- `csl-news-scheduler-init`: creates a paused control plane without
  overwriting an existing one.
- `csl-news-scheduler-pause` / `csl-news-scheduler-resume`: atomically change
  control state.
- `csl-news-scheduler-status`: reports control state and active/stale leases.
- `csl-news-annotate-scheduled`: runs one elastic worker. It exits cleanly
  while paused and waits only while running with no eligible archive.

All control records contain source/config identity and the worker identity.
The existing static `csl-news-annotate` command remains available for
forensic, targeted, and one-archive recovery operations.

## Failure Handling

- Control and lease writes are fsynced and atomically promoted.
- A stale lease is retained under `leases/expired/` as operational evidence;
  it is never silently overwritten.
- The worker never processes a ZIP absent from the integrity registry, and it
  reuses the existing source-bound marker/output checks.
- `SIGTERM` is handled by normal process termination. The existing atomic
  sample writer guarantees that the output is either absent or complete; the
  lease expires for later recovery.
