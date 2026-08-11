# OmniHand Formal GPU Run Evidence

Status: `engineering_evidence_ready`
Last Updated: `2026-08-11`
Role: `ARCH-004_EXP-001_formal_run_evidence`
Evidence ID: `EVID-CODE-OMNIHAND-FORMAL-V1`

## 1. Scope

This evidence verifies the canonical single-device OmniHand train, checkpoint, reload, prediction, and
evaluation lifecycle on a clean commit and an A100 GPU. The run uses a deterministic synthetic
model-ready manifest fixture. It proves the formal-run and model orchestration boundary, not a paper
result or real-data generalization claim.

## 2. Identity

```text
implementation commit: 81e9b89896a25bc26eece5f789b9a842004a4d4a (clean)
artifact root: /mnt/gfs/yanyifan/mmPRISM/engineering/omnihand_formal_smoke_81e9b89
train manifest: 4 records / SHA-256 0020607a88dc6c92b3e4067180a40bb294fe6d29501199f33ea0c1cfb34897e7
validation manifest: 2 records / SHA-256 48be61e04275f624e075641a8f4127c67bc067d6021e8cc4c15246da39f2bc46
device: NVIDIA A100-SXM4-80GB physical GPU 5, exposed as cuda:0
precision: bf16-mixed
seed: 20260811
Python / Torch / CUDA: 3.12.13 / 2.11.0+cu128 / 12.8
model parameters: 679,097
audit: audit.json / SHA-256 4450c5be6684dc51a1cee43a70361c721707165250f9a4a1709642648b3ea4d4
tracked summary: artifacts/omnihand_formal_run_v1.json / SHA-256 cf8b306a169e8e7adea874502d25e007e2b99bb7ce38d958fe6307595a725862
status: passed
```

The tracked summary deliberately omits synthetic pose metric values. The complete finite values remain in
the checksum-bound mounted audit, but they are not eligible for manuscript use.

## 3. Executed Lifecycle

Training used `mmprism omnihand-train` with
`configs/examples/omnihand_train_smoke_experiment.yaml` and
`configs/examples/omnihand_train_smoke.yaml`, followed by `mmprism omnihand-evaluate` against the emitted
Safetensors checkpoint and metadata. The resolved commands, absolute runtime inputs, and their hashes are
frozen in each run's `run.json` and `inputs.json`.

| Artifact | Identity |
|---|---|
| train run | `omnihand-train-smoke__20260811T201542Z__eb99098e` |
| evaluate run | `omnihand-train-smoke__20260811T201630Z__eb99098e` |
| checkpoint | `18b941a3161a10978ca91033ed670a9881a09339b797c80fc4aed13e9c9b8010` |
| train predictions | `c80928b022877d4857b87940d109ebf171e7edbbac449daaf87844c491ee0f6c` |
| evaluate predictions | `c80928b022877d4857b87940d109ebf171e7edbbac449daaf87844c491ee0f6c` |
| metric protocol | `mmprism.pose_metric.dual_hand_metric_v1` |

## 4. Independent Audit

All thirteen audit gates passed:

- train and evaluate runs completed on the same clean commit;
- all declared inputs and produced artifacts matched their SHA-256 records;
- the checkpoint metadata referenced the exact Safetensors payload and input identities;
- both validation sample IDs had finite, count-weighted metric records;
- standalone evaluation reproduced byte-identical sample predictions and identical metric summary values;
- runtime metadata confirmed deterministic BF16 execution on an A100;
- performance artifacts and CUDA peak-memory counters were present;
- no hidden temporary files remained in either completed run.

## 5. Performance Record

| Mode | Core work | End-to-end | Peak allocated / reserved | Throughput |
|---|---:|---:|---:|---:|
| train | 2 optimizer steps / 1.289212 s | 4.064847 s | 48,897,536 / 79,691,776 B | 1.5513 steps/s |
| evaluate | 2 predictions / 0.864685 s | 3.297752 s | 18,224,640 / 41,943,040 B | 2.3130 samples/s |

These timings characterize only the tiny engineering fixture and include process/model setup in the
end-to-end column. Reviewer-facing efficiency evidence still requires the final architecture, real input
length distribution, matched baselines, warm-up policy, repetitions, and fixed hardware protocol.

## 6. Evidence Boundary

- The input arrays are deterministic synthetic tensors in `synthetic_radar_cartesian_v1` coordinates.
- No real or manuscript synthetic dataset, subject split, calibrated acquisition, beamformed cube, or
  paper checkpoint is attached.
- The run does not close resume, distributed checkpoint/prediction aggregation, WaveLLM, or real-data
  training requirements.
- Synthetic reconstruction metrics must not be copied into the manuscript, response letter, figures, or
  Source Data package.
