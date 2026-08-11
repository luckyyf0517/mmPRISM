# WaveLLM Formal GPU Run Evidence

Status: `engineering_evidence_ready`
Last Updated: `2026-08-11`
Role: `ARCH-005_EXP-001_formal_run_evidence`
Evidence ID: `EVID-CODE-WAVELLM-FORMAL-V1`

## 1. Scope

This evidence verifies the canonical single-device WaveLLM train, adapter checkpoint, reload,
prediction, and evaluation lifecycle on a clean commit and an A100 GPU. The run uses the pinned
mT5-base asset and a deterministic synthetic model-ready manifest fixture. It proves the formal-run,
multimodal adapter, checkpoint, and language-metric boundaries; it is not a paper result or evidence of
real sign-language generalization.

## 2. Identity

```text
implementation commit: e31000b3f55718d36df15e2013d80e18f7b690e1 (clean)
artifact root: /mnt/gfs/yanyifan/mmPRISM/engineering/wavellm_formal_smoke_e31000b
train manifest: 4 records / SHA-256 3b1e68bf597eb3f30fa209f4f5b859d6e595d4ee9b1fd01a5b1ee1895286c288
validation manifest: 2 records / SHA-256 193ae2dde940e7a1b1e4586237c5ff40b785dfa6a8483e497d3bad6ee96019b7
model asset: google/mt5-base@2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f
device: NVIDIA A100-SXM4-80GB physical GPU 5, exposed as cuda:0
precision: bf16-mixed
seed: 20260811
Python / Torch / CUDA: 3.12.13 / 2.11.0+cu128 / 12.8
model parameters: 584,571,712 total / 2,170,432 trainable / 582,401,280 frozen
audit: audit.json / SHA-256 ea2d074092c865615077242b156d96d45c01b933577a25d4d62757ef4ead6458
tracked summary: artifacts/wavellm_formal_run_v1.json
status: passed
```

GPU selection followed the approved shared-GPU policy: free memory was the only resource gate and GPU
utilization was not a gate. Physical GPU 5 had 81,154 MiB free before launch. No other process was
stopped, changed, or migrated.

## 3. Executed Lifecycle

Training used `mmprism wavellm-train` with
`configs/examples/wavellm_train_smoke_experiment.yaml`,
`configs/examples/wavellm_train_smoke.yaml`, and `configs/models/mt5_base_v1.yaml`. Standalone
evaluation then loaded the emitted adapter-only Safetensors checkpoint and metadata through
`mmprism wavellm-evaluate`. Each run froze the resolved configuration, command, Git/environment state,
model-asset manifests, data manifests, and their SHA-256 identities.

| Artifact | Identity |
|---|---|
| train run | `wavellm-train-smoke__20260811T205116Z__90a769db` |
| evaluate run | `wavellm-train-smoke__20260811T205146Z__90a769db` |
| checkpoint | `e4aab4edcc00f0ed51e290a3bb841e8549732b4c542daeb4d6b77d32229f5f44` |
| adapter inventory | 62 tensors; zero `language_model.*` keys |
| train predictions | `5f95172f06efdce35b318ad91ccc2e7aa29098978b46b736140c7679ac90ca03` |
| evaluate predictions | `5f95172f06efdce35b318ad91ccc2e7aa29098978b46b736140c7679ac90ca03` |
| metric protocol | `mmprism.language_metric.character_v1` |

The strict model-ready records bind `[T,2,24,3]` float32 metric poses, `[T,2,24]` confidence,
`[T,1024]` radar features, optional frame masks, inline captions, units, coordinate frame, and array
checksums. Training rejects sample or sequence overlap between train and validation. Evaluation rejects
checkpoint weight, model/task config, model-asset, unit, or coordinate-frame drift before loading.

## 4. Independent Audit

An external auditor that does not import the mmPRISM training/evaluation services passed 250 gates:

- all declared input and artifact sizes and SHA-256 values matched;
- both runs completed on the same clean implementation commit;
- all 24 fixture arrays matched manifest shape, dtype, checksum, and finite-value contracts;
- train/validation sample and sequence IDs were disjoint;
- the checkpoint was valid adapter-only Safetensors with finite tensors and no language-model keys;
- both validation sample IDs appeared exactly once;
- standalone evaluation reproduced byte-identical predictions;
- Unicode code-point edit counts and count-weighted summary values were independently recomputed;
- A100, BF16, deterministic seed, performance, and CUDA peak-memory records were present;
- no hidden or temporary files remained in completed run directories.

The auditor source is retained at `audit_run.py` with SHA-256
`a61b4a26756037980eb3aaaa8fbcce14f1fbe05cf8864bd5242ab3b09f8791ec`.

## 5. Performance Record

| Mode | Core work | End-to-end | Peak allocated / reserved | Throughput |
|---|---:|---:|---:|---:|
| train | 2 optimizer steps / 1.5687 s | 17.3319 s | 1,291,147,776 / 1,312,817,152 B | 1.2749 steps/s |
| evaluate | 2 predictions / 1.0769 s | 16.2700 s | 1,210,208,256 / 1,220,542,464 B | 1.8571 samples/s |

These timings characterize only the tiny engineering fixture and include pinned mT5 loading in the
end-to-end column. Reviewer-facing efficiency evidence still requires the final architecture, real input
length distribution, warm-up/repetition protocol, and matched baselines.

## 6. Evidence Boundary

- Input pose and radar features are deterministic random tensors in
  `synthetic_radar_cartesian_v1`; captions are fixture text.
- No real radar acquisition, reconstructed OmniHand output, scientific split, subject, environment, or
  manuscript checkpoint is attached.
- Character-metric values from this run are deliberately omitted from tracked evidence and must not be
  copied into the manuscript, response letter, figures, tables, or Source Data.
- The run does not close real-data training, resume, distributed checkpoint/prediction aggregation,
  production BLEU/ROUGE/semantic metrics, or reviewer experiments.
