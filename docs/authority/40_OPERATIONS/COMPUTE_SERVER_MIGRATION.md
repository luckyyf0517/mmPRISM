# Compute Server Migration And Handoff

Status: current
Owner: mmPRISM coordinator
Authority scope: Controlled transition of compute execution to a replacement GPU server without changing data or evidence state.
Last reviewed: 2026-08-13

## Purpose

The shared Slurm node is saturated and no longer provides predictable turnaround for the revision-critical
CSL-Daily reconstruction and training loop. This runbook prepares a replacement compute server. It is a runtime
relocation only: no data product, checkpoint, experiment result, or paper claim is promoted by moving machines.

Current local execution is intentionally stopped. The two pending CSL-Daily batch-64 benchmark jobs (`116`, `117`)
were cancelled before they started. The `annotation_v2` scheduler was initialized at
`interim/csl_daily/pose_annotation/rtmw3d_l_794dbc78_v2/` and remains `paused`; it has no published v2 sample,
global pose manifest, or accepted training input.

## Source Of Truth And Freeze Boundary

The GFS project mirror remains the source of truth:

```text
/mnt/gfs/yanyifan/mmPRISM/
  external/csl_daily/csl_daily_original_20260812/  immutable CSL-Daily source
  interim/csl_daily/pose_annotation/..._v1/        frozen diagnostic only
  interim/csl_daily/pose_annotation/..._v2/        paused scheduler control state
  log/archived/                                    preserved historical WaveLLM bundle
  dataset/, pretrained_models/                     read-only historical-layout evidence
```

Do not use the migration to modify, delete, rename, rehash in place, convert, or train directly from `dataset/`,
`pretrained_models/`, or `log/archived/`. The historical WaveLLM bundle is staged but remains preservation-only:
the separate receipt, format/world-size audit, metadata/tensor audit, and controlled-load decision are still
pending. Its staging state is recorded in the
[historical-bundle handoff log](../../../workspaces/wavellm_training/docs/logs/2026/08/20260813_HISTORICAL_WAVELLM_STAGING_HANDOFF.md).

Raw CSL-Daily input stays immutable. A replacement server may either mount GFS read-only or receive a resumable
read-only copy of the minimum source/model inputs. Derived outputs must use a new versioned target root or a
deliberately accepted GFS output root; they must never be written under `external/`.

## Target Server Requirements

| Area | Requirement |
|---|---|
| Platform | Linux x86_64, Python 3.12, `uv >=0.11,<0.12`, sufficient local SSD for environment, model cache and active shards. |
| GPU | At least one CUDA-capable GPU for smoke; four 24/32 GB GPUs are appropriate for distributed training. A Blackwell 5090/RTX PRO 6000D requires a driver and PyTorch build that recognize its CUDA capability. |
| CUDA stack | Use the committed PyTorch CUDA 12.8 lockfile (`torch 2.11.0+cu128` on this host). Do not copy old virtual environments, PyTorch wheels, DeepSpeed builds, FlashAttention, xFormers, MMCV, or MMPose extension binaries. |
| Distributed path | For four GPUs, install the `distributed` profile and validate NCCL locally. Network connectivity is adequate for DDP/ZeRO-2; NVLink is optional for the current mT5-base control. |
| Storage | Keep active caches, temporary Parquet parts, checkpoints and logs on local SSD where possible. Copy only frozen manifests, selected data shards and required model assets; sync formal artifacts back atomically. |
| Access | The target may read GFS/NAS data but must not require project tokens in the repository. Machine-specific credentials remain in an untracked target-local `.env`. |

`4 x 4090/5090` is expected to be the most efficient first training target. The historical WaveLLM script used
two ranks with ZeRO-2; four ranks reduce its old global batch of 64 to a micro-batch of 16 per GPU. A single
96 GB GPU is sufficient for smoke/debugging but does not replace the throughput of four independent GPUs.

## Transfer Procedure

1. Freeze this repository commit and clone it on the target. Do not copy `.venv`, `.cache`, `logs/`, output roots,
   or any untracked local state.
2. On the target, create an untracked `.env` containing only machine-local roots, for example
   `MMPRISM_DATA_ROOT`, `MMPRISM_ARTIFACT_ROOT`, `MMPRISM_CACHE_ROOT`, and `MMPRISM_MODEL_ROOT`. Keep all secrets
   and remote credentials out of Git and command histories.
3. Mount GFS read-only for source/evidence inspection. If it is unavailable or too slow, create a resumable copy of
   the required frozen subset with a manifest and checksum inventory. Do not transfer the whole historical mirror by
   default. Start with the code, CSL-Daily source/model subset, pinned mT5 asset, and only the checkpoint required
   for a named controlled-load audit.
4. Create the canonical environment from the committed lock:

   ```bash
   cd /path/to/mmPRISM
   scripts/bootstrap_env.sh distributed
   ```

   This creates a fresh environment with CUDA 12.8 PyTorch and DeepSpeed. Any package that compiles CUDA code must
   be installed or rebuilt on the target; cached binaries from the A100 host are invalid for a different GPU class.
5. Record the target's driver, GPU names, CUDA runtime, PyTorch version, CUDA capability, NCCL version, free disk,
   Git commit and resolved environment in the first smoke run. Do not treat `nvidia-smi` output as sufficient proof
   that the Python stack supports the GPU.
6. Complete the acceptance gates below. Only after all required gates pass may the coordinator explicitly resume
   `annotation_v2` or submit a training job.

## Target Acceptance Gates

Run in order, retaining terminal output in a dated target-side handoff record:

```bash
nvidia-smi
uv --version
uv run mmprism doctor
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
uv run ruff check src/mmprism tests
uv run pytest tests/unit/test_csl_daily_pose_annotation.py \
  tests/unit/test_csl_daily_annotation_scheduler.py
```

Then perform three isolated GPU gates in order:

1. **Core models:** run `scripts/run_omnihand_smoke.sh` and `scripts/run_mt5_smoke.sh` with target-local artifact
   roots and the pinned mT5 asset. These are engineering checks only.
2. **Distributed runtime:** on a four-GPU target, execute the existing multi-GPU NCCL integration smoke before
   using DeepSpeed for a formal run. For WaveLLM, begin with BF16 and ZeRO-2; do not jump to ZeRO-3 unless a measured
   micro-batch OOM requires it.
3. **Annotation runtime:** with a read-only CSL-Daily input subset and a separate disposable output root, run one
   RTMW3D sequence through the v2 transform and visually inspect native/canonical output. Do not resume the canonical
   scheduler merely to test the target environment.

On Blackwell GPUs, a capability/import/kernel failure is a target-environment failure. Rebuild the relevant package
against the target's PyTorch/CUDA combination; do not downgrade the repository lock or patch canonical code solely
to preserve an obsolete wheel.

## Resume Order After Acceptance

1. Review the target handoff record and explicitly select the target artifact root.
2. Run the one-sequence `annotation_v2` quality smoke and review its overlay/sidecar.
3. Resume the existing `annotation_v2` scheduler, submit only the approved number of elastic Slurm-equivalent
   workers, and preserve the pause/finalize rules in the
   [CSL-Daily reproduction operation](../../../workspaces/data_rebuild/docs/authority/40_OPERATIONS/CSL_DAILY_REPRODUCTION.md).
4. After a frozen full-corpus annotation and synthetic-FMCW delivery, run OmniHand before the two pose-only WaveLLM
   controls. The feature/fusion comparison remains later work.
5. Keep the historical WaveLLM bundle audit separate from new training. It is not a prerequisite for the first
   controlled loop and must not silently become an initialization source.

## Handoff Record

The operator records only the following at migration completion:

```text
target hostname and operator
repository commit and clean/dirty status
GPU/driver/PyTorch/CUDA/NCCL versions
environment profile and lockfile hash
source mount or copied-subset manifest/checksum identity
artifact root and free-capacity report
each acceptance gate command, result, and artifact location
explicit scheduler state and submitted-job IDs (if any)
known blocker and next authorized action
```

No data or model becomes accepted merely because it was copied to the target. Formal runs retain the normal resolved
configuration, Git state, manifest hash, split hash, environment, seed and metric artifact requirements.
