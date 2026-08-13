# CSL-Daily Reproduction Operation

Status: current
Owner: Data rebuild lane
Authority scope: CSL-Daily source identity, camera-pose quality control, synthetic reconstruction delivery, and handoff boundaries.
Last reviewed: 2026-08-13

## Purpose And Boundary

CSL-Daily is the active practical control dataset for rebuilding the camera-pose -> skeleton-simulated radar ->
mmWave-pose route. A full, quality-controlled camera-pose rebuild is P0 because no historical CSL-Daily annotation
payload survived transfer; the retained legacy JSON files are stale absolute-path maps only. This operation establishes
the camera-pose input needed to simulate radar and reconstruct mmw-pose. It cannot by itself answer reviewer requests
about real radar, new users, `30`/`60` degree orientation, or real occlusion; those remain separate real-data evidence.

The cross-workspace meanings of cam-pose, synthetic radar, mmw-pose and radar feature are fixed by the
[research execution model](../../../../../docs/authority/30_ARCHITECTURE/RESEARCH_EXECUTION_MODEL.md). This page
owns only the Data Rebuild portion of that model.

This operation owns source identity through frozen reconstruction/translation inputs. CubeNet training, feature
export, WaveLLM training, historical checkpoint audit, and paper-claim promotion remain owned by their respective
workspaces.  The planned cross-workspace controls are specified by OpenSpec
`add-csl-daily-reproduction-controls`.

## Transfer Gate

The raw source arrived at:

```text
/mnt/gfs/yanyifan/mmPRISM/external/csl_daily/csl_daily_original_20260812/
```

Its direct final-volume placement avoids a second approximately 300 GB copy. The upload is complete. The source tree
originally contained one expanded `frames_512x512/` tree plus a full archive and ten transfer splits of the same
archive. The redundant compressed payloads were removed on 2026-08-13 after a checksum-bound retention receipt at
`/mnt/gfs/yanyifan/mmPRISM/interim/csl_daily/source_retention/20260813_redundant_archive_cleanup/`; expanded frames,
official labels/splits, and review MP4s are retained. Do not modify retained source bytes. The annotation run records
the source root/ID, per-sequence frame list and count, configuration fingerprint, model checkpoint hash, Git state,
and QC outcomes. After cleanup:

1. Record source/version/license/download metadata, input annotation identity, and any source-provided checksums.
2. Read the annotation and a deterministic image subset without modifying source bytes.
3. Bind legacy `dataset/csl-daily/{train,val,test,all}.json` by checksum as historical evidence only; do not use
   their absolute pose paths.
4. Record the completed search for historical `sentence/poses`, synthetic signals/cubes, predicted poses, and features:
   none were uploaded. Preserve the legacy JSON mapping files as non-executable historical metadata and explicitly
   record their absent old-machine targets; do not invent or path-rewrite a historical derived product.
5. No canonical output writes to `external/`.

The full checksum receipt command remains available for a later release or public archive, but is not on the P0
annotation critical path:

```bash
uv run mmprism csl-daily-source-receipt \
  --source-root /mnt/gfs/yanyifan/mmPRISM/external/csl_daily/csl_daily_original_20260812 \
  --receipt-root /mnt/gfs/yanyifan/mmPRISM/interim/csl_daily/source_receipts \
  --source-id DATASET-CSL-DAILY \
  --legacy-split-root /mnt/gfs/yanyifan/mmPRISM/dataset/csl-daily \
  --stability-wait-seconds 60
```

It records two time-separated metadata inventories, hashes every regular source file, and publishes a no-clobber
receipt under `interim/`; it never writes under `external/`. It must not block reconstruction, and should be run only
when the stronger release/archive evidence is needed.

The existing `val.json` and `test.json` are byte-identical.  This permits only a labelled historical replay of
the former validation-as-test convention.  New controls require an explicit distinct holdout assignment or omit
an independent test summary.

## Camera-Pose Lane

### Historical Direct Replay Status

No historical full CSL-Daily cam-pose, synthetic signal/cube, predicted-pose, or feature asset is available locally.
The received `dataset/csl-daily/{train,val,test,all}.json` files name paths on the former machine and all targets are
absent; `val.json` and `test.json` are byte-identical. Historical direct replay is closed unless an independently
received, immutable asset and producer receipt later establish a valid input. It must never be simulated by relabelling
the new reconstruction as historical replay.

### Frozen Diagnostic: `annotation_v1`

`rtmw3d_l_794dbc78_v1` is a 105-sequence pilot only: 54 arrays completed and 51 sequences skipped. Its completed
arrays can contain NaN hand coordinates, and its sidecars do not retain native 133-joint poses/scores, canonical
confidence, or joint/frame validity. It is preserved for failure analysis and must never be overwritten, extended,
or treated as a model-ready cam-pose build.

### Canonical P0 Baseline: `annotation_v2`

Implement and rebuild the full accepted source through canonical code into a new versioned interim root:

```text
raw frame
-> full-image single-person RTMW3D inference, batch=1
-> preserve native pose/scores in sidecar
-> confidence threshold
-> sequence depth centering
-> 17 body + 42 hand selection
-> finite dual-hand [T,2,24,3] output with explicit fill policy, confidence, joint validity and frame mask
-> quarantine failures and publish QC
```

`batch=1` is a historical numerical-equivalence baseline, not a claim that batching is universally invalid. This
new canonical baseline need not be byte-identical to a transferred historical pose product; the comparison report
states any implementation/model/environment difference. The baseline output is immutable once frozen; no later
quality improvement overwrites it.

`annotation_v2` is mandatory P0 and the first training target. It is required because the frozen v1 pilot's observed
contract failures meet the pre-existing material-failure condition for a successor annotation version. It uses a
lease-controlled queue: workers atomically publish only their one sequence's payload and sidecar;
separate finalization creates the global manifest only when the queue is paused and lease-free. Every source sequence
must end as completed, QC-skipped, or an immutable error quarantine. Finalization hard-fails on any unprocessed
sequence and never publishes a partial training manifest. Run its contract/GPU smoke, deterministic QC, and full
coverage/eligibility manifest before simulation, OmniHand, or pose-only WaveLLM.

### Controlled Execution

After the one-sequence GPU smoke passes visual review, use the following lifecycle. Direct `csl-daily-annotate` is
deliberately rejected for v2 to prevent bypassing the queue.

```bash
export MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM
CONFIG=configs/data/csl_daily_rtmw3d_v2.yaml

uv run mmprism csl-daily-scheduler-init "$CONFIG" --lease-seconds 1800
uv run mmprism csl-daily-scheduler-resume "$CONFIG" --reason "P0 annotation v2"
```

Submit one or more Slurm workers using the v2 worker script. Workers acquire exclusive sequence leases, can be added
or stopped independently, and only load RTMW3D after a sequence is claimed. To stop cleanly:

```bash
uv run mmprism csl-daily-scheduler-pause "$CONFIG" --reason "operator pause"
uv run mmprism csl-daily-scheduler-status "$CONFIG"
uv run mmprism csl-daily-annotation-finalize "$CONFIG"
```

Finalization writes `coverage.json`, `pose_qc.jsonl`, and the eligible `pose_manifest.jsonl`; inspect the quarantine
ledger and deterministic QC/review before declaring the artifact accepted. Resume is the same `scheduler-resume`
command; existing completed, QC-skipped, and quarantined sequences are not silently recomputed.

### QC Before Full Promotion

Before selecting any all-corpus annotation for simulation, run deterministic stratified review over raw frames,
overlay renders, and sidecars.  The report must include:

- source and annotation coverage; skip, failure, and quarantine reasons;
- NaN/finite rates, arm/hand visibility, confidence distributions, and sequence lengths;
- frame speed/acceleration, jump rate, bone-length stability, and left/right identity-swap alerts;
- strata by length, confidence, motion/jitter, overlap/occlusion proxy, crop condition, and any available official
  signer/recording metadata;
- blinded manual labels for tracking, left/right identity, gross 3D plausibility, and failure category.

Future annotation improvements after the canonical P0 build require a distinct successor version/config and a
comparison/promotion decision. Reduced jitter alone does not establish a better target pose.

## Synthetic Reconstruction Delivery

After the selected annotation is frozen, the in-progress `rebuild-csl-daily-simulation` change produces the
separately labelled `csl_daily_skeleton_sim_v1` route. Its persistent radar boundary is the pre-beamforming FMCW
signal, not the CubeNet power cube:

```text
selected camera-pose manifest
-> point-cloud preprocessing
-> IWR1843 skeleton point-reflector FMCW simulation
-> checksum-bound synthetic FMCW [T,C,A,S] sidecar and typed Parquet payload
-> frozen raw-radar pose-reconstruction manifest/split
-> runtime range/Doppler/beamforming -> transient [T,D,R,A,E] power cube
-> OmniHand
```

It is a controlled reconstruction of the legacy skeleton simulator, not a direct reproduction of the manuscript-
described MANO mesh/ray-tracing method. Every sample binds annotation version, simulation configuration, source
manifest hash, split identity, signal representation/dtype/shape, and processor fingerprint. The pilot must measure
the raw-signal storage and runtime processing cost before the full build; raw FMCW is the correct interface boundary
but is not assumed to be smaller than a power cube.

The currently implemented direct-cube materializer is retained only as an engineering prototype and numerical
comparison fixture. It must not create a formal CSL-Daily delivery after `DEC-049`.

## Required Handoffs

### To OmniHand Training

Deliver one immutable raw-radar pose-reconstruction build with producer commit, source/eligibility manifest hash,
split assignment hash, Parquet inventory/checksums, validation report, pilot capacity/throughput report, coordinate
frame/units, simulator contract, and processor fingerprint. A CubeNet formal run must not point to live sidecars or
persisted power cubes.

### To WaveLLM Training

The first handoff supports the camera-pose semantic control: pose, confidence, mask, caption, manifest and split.
After OmniHand, a second pose-only handoff contains cross-fitted predicted mmw-pose with the same bindings. Neither
first-loop handoff contains a radar feature. A fusion handoff waits for a checkpoint-bound OmniHand feature export
and is not a dependency of the pose-only runs.

## Do Not Do

- Do not alter raw CSL-Daily bytes or write derived data under `external/`.
- Do not use the duplicated historical test file as a new independent test set.
- Do not call a CSL-Daily synthetic result real-radar, sim-to-real, or reviewer generalization evidence.
- Do not use incoming `log/` checkpoints until their separate stable receipt/audit completes.
- Do not infer file relationships by legacy path replacement.

## Validation Order

```text
source location and lightweight identity checks -> annotation unit/GPU smoke -> baseline QC -> selected annotation manifest
-> simulation contract/integration -> frozen split -> Parquet validator/reader parity
-> one-batch adapter smoke -> formal OmniHand run
```
