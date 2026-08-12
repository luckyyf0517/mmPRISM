# Historical WaveLLM Bundle Transfer Record

Status: transfer in progress
Owner: WaveLLM training lane
Evidence scope: Incoming historical artifacts; preservation and receipt status only.
Recorded: 2026-08-12

## Incoming Source

The author identified historical model artifacts and began uploading them directly to the project mirror:

```text
/mnt/gfs/yanyifan/mmPRISM/log/archived/
```

This path is an inbound historical-original location, not a canonical model-asset destination and not an
active training output. It is ignored by Git. The historical files are valuable evidence and must remain
unchanged throughout receipt and audit.

At discovery, the transfer had created the following candidate run directories. Names are unverified
historical labels and do not by themselves establish data scope, model role, metric, or manuscript linkage.

```text
wavellm_mt5_news_0523_gt/
wavellm_mt5_daily_0612/
wavellm_mt5_daily_0702_gt/
wavellm_mt5_daily_0826/
```

The first partially transferred candidate, `wavellm_mt5_news_0523_gt/last.ckpt`, has a directory-form
DeepSpeed-style layout including `checkpoint/mp_rank_00_model_states.pt`, `zero_to_fp32.py`, `latest`, and
rank-local evaluation JSON files. It must not be treated as a complete single-file Lightning checkpoint.

## Preservation Rule

Until the uploader explicitly confirms completion and a stable snapshot is recorded:

- do not move, rename, delete, archive, chmod, hard-link, or write anywhere under `log/`;
- do not launch training, evaluation, checkpoint conversion, DeepSpeed recovery, or `torch.load` on a partial file;
- do not calculate final SHA-256 values or infer model contents from partial bytes;
- do not replace the existing local mT5 receipt plan or alter formal experiment initialization based on directory
  names alone.

This does not prohibit read-only directory and size monitoring. Monitoring results are provisional and are not a
receipt.

## Post-Transfer Acceptance Gate

After the author confirms transfer completion, intake proceeds without modifying the inbound source:

1. Record two identical, time-separated recursive inventories to establish a stable file set.
2. Capture relative path, byte size, modification time, SHA-256, and file count in a tracked receipt; retain the
   source path unchanged.
3. Identify checkpoint format and world-size completeness from metadata and rank-local files; report missing shards
   as a receipt failure, not a reason to synthesize them.
4. Inspect checkpoint metadata and tensor-key namespaces with a CPU-only, read-only load appropriate to the confirmed
   format. Record model/config/data/split/evaluation references as observed facts or `unknown`.
5. Run a controlled load only against a copied, checksum-bound derived asset or an explicitly read-only adapter.
   Report missing and unexpected tensors before using any weights for a formal run.
6. Update the model registry, WaveLLM architecture boundary, data registry, and paper evidence map with the accepted
   role. Historical checkpoint recovery does not validate historical metrics, splits, or paper claims by itself.

## Current Boundary

The earlier mT5-only export remains a load-smoke-verified fallback initialization. It is not replaced by the
incoming bundle until the gate above establishes a complete, identified asset. New CSL-Daily engineering work may
continue against canonical contracts, but no paper-facing comparison may claim reuse or reproduction of an incoming
historical model before receipt and audit complete.
