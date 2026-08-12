# CSL-Daily Reproduction Operation

Status: current
Owner: Data rebuild lane
Authority scope: CSL-Daily source receipt, camera-pose quality control, synthetic reconstruction delivery, and handoff boundaries.
Last reviewed: 2026-08-12

## Purpose And Boundary

CSL-Daily is the active practical control dataset for rebuilding the camera-pose -> skeleton-simulated cube ->
mmWave-pose route.  It can establish a controlled synthetic-data baseline and help diagnose the former training
pipeline.  It cannot by itself answer the reviewer requests about real radar, new users, `30`/`60` degree
orientation, or real occlusion; those remain separate real-data evidence.

The cross-workspace meanings of cam-pose, synthetic radar, mmw-pose and radar feature are fixed by the
[research execution model](../../../../../docs/authority/30_ARCHITECTURE/RESEARCH_EXECUTION_MODEL.md). This page
owns only the Data Rebuild portion of that model.

This operation owns source receipt through frozen reconstruction/translation inputs.  CubeNet training, feature
export, WaveLLM training, historical checkpoint audit, and paper-claim promotion remain owned by their respective
workspaces.  The planned cross-workspace controls are specified by OpenSpec
`add-csl-daily-reproduction-controls`.

## Transfer Gate

The raw preservation candidate is currently written by rsync at:

```text
/mnt/gfs/yanyifan/mmPRISM/external/csl_daily/csl_daily_original_20260812/
```

Its direct final-volume placement avoids a second approximately 300 GB copy.  It remains unaccepted while rsync
is running.  Do not hash the moving tree, extract archives, write poses, or train from it.  At completion:

1. Record two time-separated recursive relative inventories with equal paths, counts, sizes, and mtimes.
2. Record source/version/license/download metadata, input annotation identity, and any source-provided checksums.
3. Read the annotation and a deterministic image subset without modifying source bytes.
4. Bind legacy `dataset/csl-daily/{train,val,test,all}.json` by checksum as historical evidence only; do not use
   their absolute pose paths.
5. Inventory any uploaded `sentence/poses`, synthetic signals/cubes, and features as historical derived candidates:
   preserve relative names/bytes, record count/shape/dtype/checksum, bind each to source/caption records where
   evidence permits, and record missing producer/configuration evidence explicitly.
6. Publish a receipt and create an immutable source manifest. No canonical output writes to `external/`.

The read-only receipt command is ready but must not be run while rsync remains active:

```bash
uv run mmprism csl-daily-source-receipt \
  --source-root /mnt/gfs/yanyifan/mmPRISM/external/csl_daily/csl_daily_original_20260812 \
  --receipt-root /mnt/gfs/yanyifan/mmPRISM/interim/csl_daily/source_receipts \
  --source-id DATASET-CSL-DAILY \
  --legacy-split-root /mnt/gfs/yanyifan/mmPRISM/dataset/csl-daily \
  --stability-wait-seconds 60
```

It records two time-separated metadata inventories, hashes every regular source file, takes a post-hash metadata
inventory, and publishes a no-clobber receipt under `interim/`; it never writes under `external/`. Validate the
published artifact with `uv run mmprism csl-daily-source-receipt-validate RECEIPT_ROOT` before source-manifest or
annotation work.

The existing `val.json` and `test.json` are byte-identical.  This permits only a labelled historical replay of
the former validation-as-test convention.  New controls require an explicit distinct holdout assignment or omit
an independent test summary.

## Camera-Pose Lane

### Historical Direct Replay Candidate

Existing annotated CSL-Daily poses are valuable because they may be the exact old input to CubeNet and WaveLLM.
After their receipt, start the shortest direct replay first: accepted received cam-pose, with a received synthetic
signal/cube only when its producer/source binding passes, or otherwise a controlled regeneration from that same
received cam-pose. It is always `historical_replay`; the legacy duplicated validation/test mapping is explicitly
`legacy_validation_as_test`. This lane must never overwrite the historical asset or silently substitute a new pose.

### Canonical Baseline: `annotation_v1`

In parallel with historical receipt, reproduce the old transform in a new versioned interim root and through
canonical code:

```text
raw frame
-> full-image single-person RTMW3D inference, batch=1
-> preserve native pose/scores in sidecar
-> confidence threshold
-> sequence depth centering
-> 17 body + 42 hand selection
-> dual-hand [T,2,24,3] output and confidence/validity
-> quarantine failures and publish QC
```

`batch=1` is a historical numerical-equivalence baseline, not a claim that batching is universally invalid. This
new canonical baseline need not be byte-identical to a transferred historical pose product; the comparison report
states any implementation/model/environment difference. The baseline output is immutable once frozen; no later
quality improvement overwrites it.

### QC Before Full Promotion

Before selecting any all-corpus annotation for simulation, run deterministic stratified review over raw frames,
overlay renders, and sidecars.  The report must include:

- source and annotation coverage; skip, failure, and quarantine reasons;
- NaN/finite rates, arm/hand visibility, confidence distributions, and sequence lengths;
- frame speed/acceleration, jump rate, bone-length stability, and left/right identity-swap alerts;
- strata by length, confidence, motion/jitter, overlap/occlusion proxy, crop condition, and any available official
  signer/recording metadata;
- blinded manual labels for tracking, left/right identity, gross 3D plausibility, and failure category.

### Candidate: `annotation_v2`

Quality improvements such as person tracking, crop policy, temporal association, hand identity rules, or post-
processing are permitted only under a separate annotation version/config.  Promotion requires a v1-v2 comparison
and an explicit decision record.  Reduced jitter alone does not establish a better target pose.

## Synthetic Reconstruction Delivery

After the selected annotation is frozen, the in-progress `rebuild-csl-daily-simulation` change produces the
separately labelled `csl_daily_skeleton_sim_v1` route:

```text
selected camera-pose manifest
-> point-cloud preprocessing
-> IWR1843 skeleton point-reflector FMCW simulation
-> [T,64,32,32,32] synthetic power cube
-> frozen pose-reconstruction manifest/split
-> split-isolated Parquet delivery
```

It is a controlled reconstruction of the legacy skeleton simulator, not a direct reproduction of the manuscript-
described MANO mesh/ray-tracing method.  Every sample binds annotation version, simulation configuration, source
manifest hash, and split identity.

## Required Handoffs

### To OmniHand Training

Deliver one immutable `DELIVERY-POSE-RECON-V1` build with producer commit, source/eligibility manifest hash,
split assignment hash, Parquet inventory/checksums, validation report, capacity report, coordinate frame/units,
and simulation protocol.  A CubeNet formal run must not point to live sidecars.

### To WaveLLM Training

The first handoff can support camera-pose semantic controls: pose, confidence, mask, caption, manifest and split.
It does not contain a radar feature.  A fusion handoff waits for an OmniHand checkpoint-bound feature export.

## Do Not Do

- Do not alter raw CSL-Daily bytes or write derived data under `external/`.
- Do not use the duplicated historical test file as a new independent test set.
- Do not call a CSL-Daily synthetic result real-radar, sim-to-real, or reviewer generalization evidence.
- Do not use incoming `log/` checkpoints until their separate stable receipt/audit completes.
- Do not infer file relationships by legacy path replacement.

## Validation Order

```text
source receipt -> annotation unit/GPU smoke -> baseline QC -> selected annotation manifest
-> simulation contract/integration -> frozen split -> Parquet validator/reader parity
-> one-batch adapter smoke -> formal OmniHand run
```
