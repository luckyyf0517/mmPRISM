# Historical WaveLLM Bundle Staging Handoff

Status: preserved, receipt/audit pending
Owner: WaveLLM training lane
Evidence scope: Read-only staging observation for the historical WaveLLM bundle before compute-server migration.
Recorded: 2026-08-13

## Observed State

The author confirmed upload completion for the historical WaveLLM archive at:

```text
/mnt/gfs/yanyifan/mmPRISM/log/archived/
```

At the migration handoff, this root contained four named run directories:

```text
wavellm_mt5_daily_0612/
wavellm_mt5_daily_0702_gt/
wavellm_mt5_daily_0826/
wavellm_mt5_news_0523_gt/
```

Read-only inventory observed 30 non-`.DS_Store` files totaling 32,819,906,079 bytes. Each run has a directory-form
DeepSpeed checkpoint layout with a model-state file and two BF16 ZeRO optimizer rank files. The earlier namespace
inspection is recorded separately in
[20260813_HISTORICAL_DAILY_CHECKPOINT_NAMESPACE_INSPECTION.md](20260813_HISTORICAL_DAILY_CHECKPOINT_NAMESPACE_INSPECTION.md).

## Boundary

This is a staging observation, not a receipt. It does not validate source equivalence, file checksums, checkpoint
world-size completeness, serialized metadata, model compatibility, data split, metrics, original-submission linkage,
or suitability for training. The source remains immutable and preservation-only under `DEC-046`.

The compute-server migration must not copy, load, convert, remove, or reclassify these files by default. A future
controlled audit may copy a selected checksum-bound input to a separate derived location, while leaving this root
unchanged.
