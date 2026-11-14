# mmPRISM

mmPRISM (the paper's system name for radar pose reconstruction and sign-language
understanding) reconstructs articulated pose from complex millimetre-wave radar
and translates the reconstructed pose into Chinese. Its reconstruction model
combines a complex signal front end, a pretrained visual encoder, a spectral
residual branch and two supervised decoders. The pose-to-text model is called the
**reader** throughout this package.

This release provides the complete inference and evaluation workflow for nine
signers: zero-shot reconstruction, personal adaptation of reconstruction and the
reader using fifty segments per signer, and translation with both models adapted.
The workflow starts from the supplied pretrained checkpoints. Full training
sources are included for reading the Methods recipe. The data pack released with
this package is the evaluation set: the nine held-out signers, 1,416 evaluation
segments and 450 calibration segments. The training pool is not public and is
available from the corresponding author under the access described in the paper.

The experimental sequence is:

```text
radar -> zero-shot pose -> evaluation before adaptation
      -> adapt reconstruction on 50 segments -> adapted pose -> evaluation after adaptation
      -> adapt the reader on the same 50 segments -> Chinese text -> translation scores
```

## Reproduce in one pass

A complete run takes three downloads, one configuration edit and the commands below.

```bash
# 1. the three downloads (Hugging Face account not required for public files)
pip install -U huggingface_hub
hf download luckyyf/mmprism-data --repo-type dataset --local-dir mmprism-data
hf download luckyyf/mmprism-checkpoints --local-dir weights-download
hf download luckyyf/mmprism-eval-models --local-dir assets

# 2. put the checkpoints on the relative paths the layout expects
mkdir -p weights/artifacts/retained/mainline_20261106/checkpoints weights/artifacts/retained/stage2_reader_20261106
mv weights-download/producer_mixed100k_e2e_dual50_step_20000.pt weights/artifacts/retained/mainline_20261106/checkpoints/
mv weights-download/stage2_reader_20261106_last.ckpt weights/artifacts/retained/stage2_reader_20261106/last.ckpt
rmdir weights-download

# 3. point the configuration at the three roots, then follow steps 3 to 6 below
cp configs/paths.example.toml configs/paths.toml
# data_root = "<this directory>/mmprism-data"
# weights_root = "<this directory>/weights"
# assets_root = "<this directory>/assets"
```

Resources: one NVIDIA GPU with at least 32 GB of memory, about 150 GB of disk
for the inputs and the generated checkpoints, and roughly two to three hours for
the nine signers on one card, or under an hour with four cards running one signer
each.

The run writes, below `output_root`: reconstructed poses, the personal
reconstruction and reader checkpoints with their training reports, the text
predictions, and the pose and translation reports. Step 3 verifies that the data
and the checkpoints match their recorded digests before anything is computed.

## 1. Install the environment

Use Linux, Python 3.12 and [uv](https://docs.astral.sh/uv/) (the Python package and
environment manager). The lock file pins PyTorch and its CUDA (Compute Unified
Device Architecture, NVIDIA's accelerator runtime) 12.8 build. Inference and
adaptation were validated on NVIDIA RTX 5090 graphics cards; use a compatible
NVIDIA driver.

Run these commands from the extracted code directory:

```bash
mkdir -p .tmp .cache/uv
export TMPDIR="$PWD/.tmp"
export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

uv sync --frozen --python 3.12
cp configs/paths.example.toml configs/paths.toml
```

Keep this directory on a volume with space for the data, environment and generated
checkpoints. The commands below use one visible accelerator and process signers
in sequence:

```bash
export CUDA_VISIBLE_DEVICES=0
```

The complete nine-signer validation took about 49 minutes with signers distributed
across four cards, excluding installation. Running sequentially on one card takes
longer.

## 2. Configure data, weights and assets

Edit `configs/paths.toml` to point to your resources:

```toml
data_root = "/path/to/mmprism-data"
weights_root = "/path/to/checkpoint-root"
assets_root = "/path/to/body-rig-and-scoring-assets"
output_root = "/path/to/mmprism-outputs"
```

| Root | Contents |
|---|---|
| `data_root` | The released evaluation set, nine held-out signers, including splits, pose supervision and reader tokenizer/configuration files. |
| `weights_root` | The original reconstruction and reader checkpoints, at the relative locations declared under `[weights]` in the layout. |
| `assets_root` | The body rig, forward-kinematics constants and two sentence encoders listed below. |
| `output_root` | Generated poses, personal checkpoints, text predictions and evaluation reports. Use a new directory to retain a separate run. |

Stage one of the reconstruction model is a two-phase schedule: `configs/stage1_real.example.yaml`
runs updates 0-80000 on measured windows and stops there, and `configs/stage1_mixed.example.yaml`
continues for the last 20000, with one simulated window per card. Step counts in the second file
and in checkpoint names are local to that continuation, so the shipped
`producer_mixed100k_e2e_dual50_step_20000.pt` is update 100000 of Supplementary Note 7.

[configs/data_layout.toml](configs/data_layout.toml) maps each resource to a path
relative to its root. If your storage uses a different directory structure, edit
the corresponding relative entries. Absolute paths belong in `paths.toml`;
layout entries must stay relative and contain no parent-directory traversal.
Original checkpoints can be referenced in place.

You can also set `MMPRISM_DATA_ROOT`, `MMPRISM_WEIGHTS_ROOT`,
`MMPRISM_ASSETS_ROOT` and `MMPRISM_OUTPUT_ROOT`. These environment variables
override `paths.toml`. Every command accepts `--data-root` for a further override
of the data root, and `--config-dir` to select an alternative configuration
directory containing `paths.toml` and `data_layout.toml`.

### Data pack

The pack contains **1,416 evaluation segments / 78,801 evaluation frames**, plus
**450 calibration segments: exactly fifty for each signer**. Radar files occupy
98.3 gigabytes. The signer identifiers are `p01`, `p03`, `p06`, `p10`, `p11`,
`p15`, `p17`, `p36` and `p37`.

```text
mmprism-data/
  manifest.json
  radar/{signer}/{segment}/mmwave.fp16.npy
  pool/{signer}/
    manifest.jsonl
    pose_camera_m.npy
    valid_mask.npy
  labels3d/{signer}/
  annotations/camera_intrinsics.json
  splits/
    cells.json
    calibration_selection.json
    bigsplit.json
  daily/ce_daily_gtpose_paired/
  assets/reader_base/
```

`pool/` supplies full-joint reference poses and frame-validity masks for the
released segments; `labels3d/` contains the available matching episode-table
subset. `daily/` supplies the public-corpus pose annotations. Images and videos
are excluded. Each radar file stores half-precision real/imaginary values with
shape `(frames, 64, 116, 32, 2)`.

The files in `splits/` define **our repartition of the source corpus**.
`cells.json` separates calibration membership (`C`) from evaluation membership
(`D`); `calibration_selection.json` fixes the fifty calibration segments and their
order for each signer. `bigsplit.json` records the sentence split used by the
reader recipe. The commands consume these supplied memberships directly.

### Where the supplied files are

| What | Where | Size |
|---|---|---|
| this code | this repository | 1.7 MB |
| reconstruction and translation checkpoints | https://huggingface.co/luckyyf/mmprism-checkpoints | 8.6 GB |
| evaluation set, nine held-out signers | https://huggingface.co/datasets/luckyyf/mmprism-data | 93 GB |
| body rig, forward-kinematics constants, two sentence encoders | https://huggingface.co/luckyyf/mmprism-eval-models | 1.5 GB |

[manifests/weights_manifest.json](manifests/weights_manifest.json) records the size
and SHA-256 digest of both checkpoints, so the downloaded copies can be checked
against it. The reader loads its tokenizer and language-model configuration from
`data_root/assets/reader_base/`, then loads learned parameters from the reader
checkpoint.

| Layout key | Default location below `assets_root` | Purpose |
|---|---|---|
| `assets.mhr_rig` | `mhr_model.pt` | MHR (Momentum Human Rig, the articulated geometry used by the auxiliary reconstruction decoder). |
| `assets.mhr_fk_constants` | `mhr_fk_constants.npz` | Forward-kinematics constants for reconstruction adaptation. |
| `assets.sbert` | `sbert/` | SBERT (Sentence-BERT, the sentence encoder used for semantic translation similarity): `shibing624/text2vec-base-chinese` at revision `183bb99aa7af74355fb58d16edf8c13ae7c5433e`. |
| `assets.simcse` | `simcse/` | SimCSE (Simple Contrastive Learning of Sentence Embeddings, the second semantic translation scorer): `cyclone/simcse-chinese-roberta-wwm-ext` at revision `871d7039a3fccd4869d545a25b63c545341ca7f4`. |

The two scoring models sit in the evaluation asset repository at the revisions
named in the table and are placed below `assets_root` as `sbert/` and `simcse/`,
next to `mhr_model.pt` and `mhr_fk_constants.npz`. Nothing is downloaded at run
time: the commands load the reader tokenizer, the rig constants and both sentence
encoders from local files.

## 3. Verify the inputs

```bash
uv run --frozen mmprism-verify-data
uv run --frozen mmprism-verify-weights
```

Data verification checks the declared resources, radar file counts, array headers,
file sizes and the manifest's sampled digests. Weight verification checks file
existence and the original file digests without loading or rewriting checkpoints.
Both commands print their findings and exit with status zero on success.

## 4. Run zero-shot reconstruction

Export pose predictions from the supplied reconstruction model, then evaluate
them:

```bash
uv run --frozen mmprism-reconstruct --mode zero-shot --device cuda:0
uv run --frozen mmprism-evaluate --mode zero-shot
```

Reconstruction exports both the calibration and evaluation segments. Evaluation
scores only the evaluation membership. The pooled report is
`output_root/evaluation/pose_zero-shot.json`.

## 5. Adapt reconstruction and evaluate it

Adapt a separate reconstruction model for each signer using that signer's fixed
fifty calibration segments, export its predictions, and evaluate them:

```bash
uv run --frozen mmprism-adapt --side reconstruction --device cuda:0
uv run --frozen mmprism-reconstruct --mode adapted --device cuda:0
uv run --frozen mmprism-evaluate --mode adapted
```

The adaptation command saves each personal checkpoint under
`adaptation/reconstruction/{signer}/model.pt`. The pooled evaluation report is
`evaluation/pose_adapted.json`, below `output_root`.

The fixed adaptation settings are in
[configs/adaptation.yaml](configs/adaptation.yaml):

| Setting | Reconstruction | Reader |
|---|---:|---:|
| Calibration segments per signer | 50 | 50 |
| Random seed | `20261105` | `20261008` |
| Optimizer updates | 250 | 500 |
| Peak learning rate | `1e-4` | `5e-5` |
| Schedule | 20-update warmup, then cosine decay to `1e-5` | Cosine decay |
| Batch per optimizer update | Eight windows, accumulated as four batches of two | Eight segments, with a smaller final batch in each pass |

### What the evaluation writes

Geometric errors are reported in millimetres, and hand accuracy is the percentage
of evaluated hand joints whose wrist-relative error is below forty millimetres.
Poses are scored from overlapping 32-frame windows with stride 24: each frame
takes the prediction from the window in which it lies furthest from an edge.
Evaluation uses valid reference frames and joints projected inside the camera
image, and pooled values aggregate joint sums and counts across the signers that
were run.

## 6. Adapt the reader and evaluate translation

Run this step after exporting adapted reconstruction in step 5. Reader adaptation
uses the reconstructed poses and text labels of the same fifty calibration
segments, then translation scores the separate evaluation segments:

```bash
uv run --frozen mmprism-adapt --side reader --mode adapted --device cuda:0
uv run --frozen mmprism-translate --mode adapted --reader adapted --device cuda:0
```

The two translation options select different parts of the pipeline:

- `--mode zero-shot` or `--mode adapted` selects the reconstruction outputs.
- `--reader base` selects the supplied pretrained reader; `--reader adapted`
  selects the personal reader trained on outputs from the selected mode.

For example, to score the reader before its personal adaptation while keeping
the adapted reconstruction fixed, run:

```bash
uv run --frozen mmprism-translate --mode adapted --reader base --device cuda:0
```

The main two-command sequence above reproduces the paper's **both adapted**
setting. It writes text predictions and references to
`translation/adapted/adapted/{signer}.jsonl` and the pooled scores to
`evaluation/translation_adapted_adapted.json`, below `output_root`.

### What the translation evaluation writes

Translation scores use BLEU-4 (Bilingual Evaluation Understudy, overlap of up to
four consecutive characters), ROUGE-L (Recall-Oriented Understudy for Gisting
Evaluation–Longest Common Subsequence, sequence overlap), and the two sentence
encoders described above. Scores are computed per signer and pooled by
evaluation-segment count. Relative performance is the mean of the four metric ratios to the
paper's camera-pose reference, multiplied by one hundred; those denominators are
stored in [configs/translation_reference.json](configs/translation_reference.json).

## Run one signer or inspect the outputs

Every experiment command accepts `--signer p01`; repeat the option to select
multiple signers. For a complete single-signer trial, add `--signer p01` to each
command in steps 3–6. Keep the same selection through adaptation, reconstruction,
evaluation and translation. Omit `--signer` to run all nine signers and
obtain the pooled report.

Subset runs save per-signer reports and print scores for the selected signers.
The pooled report files are written when all nine signers are selected. All
locations below are relative to `output_root`:

| Output | Location |
|---|---|
| Per-segment poses and features | `reconstruction/{mode}/{signer}/{segment}.npz` |
| Personal reconstruction checkpoint and training report | `adaptation/reconstruction/{signer}/model.pt` and `report.json` |
| Personal reader checkpoint and training report | `adaptation/reader/{mode}/{signer}/model.pt` and `report.json` |
| Pooled geometric evaluation | `evaluation/pose_{mode}.json` |
| Per-signer geometric evaluation | `evaluation/pose/{mode}/{signer}.json` |
| Predicted and reference text | `translation/{mode}/{reader}/{signer}.jsonl` |
| Pooled translation evaluation | `evaluation/translation_{mode}_{reader}.json` |
| Per-signer translation evaluation | `evaluation/translation/{mode}/{reader}/{signer}.json` |

The geometric reports expose `full24_mm`, `wrist_relative_hand20_mm` and
`pck40_wrist_relative` for full-pose error, wrist-relative hand error and hand
accuracy, respectively. Complete reports contain `pooled` and `per_person`
results. Pose archives contain `pose` in metres, `feature`, `error`,
`error_hand_rel` and `window_start`.

Run `uv run --frozen mmprism-reconstruct --help`, or replace the command name
with any of the other five names, to inspect its options. A missing adapted
checkpoint means the corresponding adaptation step must finish first. A missing
pose export means reconstruction must run for the same mode and signer before
evaluation or reader adaptation.

## Read the implementation and training recipe

The main execution path is
[reconstruction inference](src/mmprism/release/inference.py),
[personal reconstruction adaptation](src/mmprism/release/adaptation.py),
[pose scoring](src/mmprism/release/scoring.py),
[reader adaptation and decoding](src/mmprism/release/reader.py), and
[translation scoring](src/mmprism/release/translation_metrics.py).
[configs/reconstruction.yaml](configs/reconstruction.yaml) and
[configs/reader.yaml](configs/reader.yaml) define the two models.
The [compact radar reader](src/mmprism/data/compact_radar.py) supplies validated
radar windows to inference and training.

The reconstruction training sources implement 80,000 measured-data updates
followed by 20,000 updates mixing measured and freshly simulated windows. Start
with the [trainer](src/mmprism/training/reconstruction/trainer.py),
[window dataset](src/mmprism/training/reconstruction/dataset.py),
[mixed stream](src/mmprism/training/reconstruction/mixed_stream.py) and
[radar renderer](src/mmprism/simulation/renderer.py). Example settings are in
[stage1_real.example.yaml](configs/stage1_real.example.yaml) and
[stage1_mixed.example.yaml](configs/stage1_mixed.example.yaml).

The reader is trained in three stages. Stage one reads public camera poses paired
with text for 3,600 updates at 8e-4; stage two reads the poses reconstructed from
the same sentences for 3,000 updates at 1e-4 after a 100-update warm-up; stage
three adapts the reader to a signer, on the same fifty segments as the
reconstruction stage, for 500 updates at 5e-5. The first two stages carry 512
sequences per update on four cards and apply joint-style and tempo augmentation;
stage three runs on one card with eight segments per update. The
[reader training recipe](src/mmprism/training/translation/recipe.py) holds the
augmentation, the conditional-discrimination term and the fixed training
endpoint, and [reader_stage1.example.yaml](configs/reader_stage1.example.yaml)
and [reader_stage2.example.yaml](configs/reader_stage2.example.yaml) record the
two pretraining recipes. Stage three is the executable adaptation step described
above. Pretraining needs the full corpus and the initialization assets, which
extend beyond the released data pack; these recipes are supplied as readable
code and configuration.

## Licence

This package is released under the MIT licence; the full text is in
[LICENSE](LICENSE). The upstream components listed in [NOTICE](NOTICE) keep their
own licences and are obtained separately.
