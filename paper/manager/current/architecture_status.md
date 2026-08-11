# Architecture Status and Target Design

Status: `greenfield_foundation_active`
Last Updated: `2026-08-11`
Role: `architecture_source_of_truth`

## 1. 当前代码结构

```text
run_model.py                 OmniHand Lightning 训练/测试入口
run_peft.py                  WaveLLM + DeepSpeed 训练/测试入口
run_*annotation.py           RTMPose3D 数据标注脚本
run_simulation.py            pose -> FMCW radar 仿真
run_extract_feature.py       OmniHand 特征/预测姿态导出
run_evaluation.py            文本指标与语义相似度
src/data/                    dataset 与 LightningDataModule
src/fmcw/                    radar simulation / processing / beamforming
src/model/omnihand.py        姿态估计 LightningModule
src/model/trainer.py         WaveLLM LightningModule
src/model/encoder/           CubeNet/MMHand/temporal transformer/pose GCN
src/model/llm/               MT5/Phi-3 wrapper
src/eval/                    pose/text metrics
config/                      实验 YAML
```

当前约 62 个 Python 文件、约 12.4K 行自有与 vendored Python 代码；没有测试目录、CI、package metadata 或锁定环境。

上述内容现统一标记为 `legacy_forensic_reference`。新实现已建立：

```text
pyproject.toml
AGENTS.md
configs/
docs/architecture/
src/mmprism/
tests/
```

当前 foundation 已包含 strict config、环境变量展开、manifest v1、无副作用 run plan、runtime
provenance、原子 formal-run artifact writer、统一 CLI 和 dependency-light unit tests。正式 run 现可
冻结 resolved config、环境/Git、命令以及 manifest/split/checkpoint/model 输入 SHA-256，并以显式
metric protocol 写入有限数值；domain writer 还可原子写入 strict JSON/JSONL，并登记外部原子完成的
checkpoint 等顶层 artifact。Model-ready pose adapter 现严格读取 manifest 绑定的 radar-cube/metric-pose
`.npy`、校验 shape/dtype/checksum/单位/坐标系，并对变长时间序列执行零填充和 mask。CSL-News RTMW3D 标注已具有独立 strict
config、原子 artifact、resume/failure contract、GPU smoke、portable pose+caption manifest builder、
无训练依赖的随机访问 adapter 和 deterministic group split。Radar 已冻结 raw/range-Doppler/cube、
pose、feature 和 caption contract，并完成 NumPy range-Doppler；beamforming/physical axes/simulation
因 provenance 冲突保持 blocked。Canonical OmniHand 工程切片现包含 depthwise CubeNet、3D PAFPN、
独立 channel/spatial/SE attention、vectorized mask-aware temporal transformer、single-frame path 和
versioned pose metric；clean commit `688d44d` 上两次 A100 运行确定性通过。其 single-device formal
service 已在 CPU fixture 上闭合 strict task config、clean-Git/input hash、train/validation manifest、
mixed-precision orchestration、Safetensors checkpoint、history、streaming sample prediction、checkpoint
reload/evaluate、tamper rejection 和独立耗时/吞吐/CUDA peak-memory artifact。clean commit `81e9b89`
已在 A100/BF16 上完成 synthetic model-ready manifest 的 formal train/checkpoint/reload/evaluate，
独立 13-gate 审计验证输入/产物哈希、prediction/summary replay 和性能记录；真实数据仍待执行。Canonical WaveLLM
现包含 strict pose/confidence/radar-feature/caption adapter、变长序列 mask、双手 ST-GCN、radar projector、
confidence-aware fusion、真实 mT5、单卡 mixed-precision train/evaluate、adapter/full Safetensors、sample-level
prediction 和 Unicode character metric。clean commit `e31000b` 已在 A100/BF16 上完成 4/2-record synthetic
manifest 的 train/checkpoint/reload/evaluate，独立 250-gate 审计验证全部输入/数组/产物哈希、sequence split、
adapter inventory、prediction/metric replay 和性能记录；真实 pose/radar feature 数据、production metrics、
resume 和 distributed aggregation 仍待实现。公开代码边界现由 Git allowlist release audit 管理，可生成逐文件 SHA-256 inventory、
canonical dependency graph，并自动拒绝 legacy/internal path、硬编码本地路径、缺失 entrypoint 和 import cycle。
外部 evaluator 资产现由 `mmprism.assets` 管理：strict config 固定 Hugging Face commit 和相对目标，
下载使用可恢复 cache、逐文件 SHA-256 与同文件系统原子晋升，现有目录只有在完整复验后才复用；
`models-smoke` 延迟导入 ML 依赖并已真实加载 SimCSE/SBERT。mT5-base 同样固定 revision、逐文件
校验并原子晋升，clean commit `79b45b5` 上的 A100 两步 adapter smoke 已通过。

Foundation and environment verification (`2026-08-11`)：

```text
UV 0.11.23 / Python 3.12.13 / uv.lock
163 tests passed（含 OmniHand/WaveLLM formal train/reload/evaluate、CubeNet 和 mT5 integration）
doctor/config/plan/manifest/split CLI passed
Ruff and strict Mypy passed
sdist and wheel build passed
canonical-to-legacy import scan passed
PyTorch 2.11.0+cu128 / CUDA runtime 12.8
8 x NVIDIA A100-SXM4-80GB detected
CUDA matrix multiplication smoke passed
MMPose 1.3.2 / MMEngine 0.10.7 / MMCV-lite 2.1.0 imports passed
RTMW3D-L real-video smoke passed: 125 frames, native [T,133,3], canonical [T,2,24,3]
CPU-only annotation status report passed against a live writer and ignores atomic temp files
Deterministic annotation QC passed on 100 live samples / 24,628 frames with no warnings
Full 722,711-record CSL-News JSON/CSV metadata profiler completed with explicit limitations
Portable CSL-News source-manifest builder passed atomic/clean-Git/coverage contract tests
CSL-News download promotion requires transfer success, no residual aria2 control file, and full ZIP CRC
Unreadable and member-corrupt final ZIPs produce structured audit evidence and remain excluded
Cumulative integrity registry uses clean-Git provenance, flock, per-archive audit and atomic updates
Four annotation workers consume only registry-passed archives with stable modulo sharding
Registry-aware status excludes quarantined outputs from progress and reports them separately
Each sample/failure sidecar and archive marker binds the registry snapshot and archive audit provenance
Integrity-gated pose+caption snapshot passed 2,157-record contract/checksum/portable-path validation
Pose manifest adapter loaded first/middle/last native and canonical arrays without training imports
Partial sequence split passed 2,157/2,157 coverage and zero cross-split group leakage
All 2,157 group IDs and assignment buckets independently recomputed and matched
Formal run initialization atomically writes config/environment/input hashes and refuses collisions
Metrics require a versioned protocol, sample count and finite values; completed runs require metrics
Pinned SimCSE/SBERT acquisition passed 14-file checksum validation and real CPU `[2,768]` loader smoke
Pinned mT5-base acquisition passed 6-file checksum validation at immutable revision `2eb15465...`
Clean mT5 A100 smoke at `79b45b5`: two finite adapter updates, confidence counterfactual and beam generation passed
Clean OmniHand A100 smoke at `688d44d`: two finite updates, single/temporal path, mask counterfactual and deterministic replicate passed
Clean OmniHand formal A100 run at `81e9b89`: train/checkpoint/reload/evaluate and 13-gate replay/hash audit passed
Clean WaveLLM formal A100 run at `e31000b`: adapter checkpoint/reload/evaluate and 250-gate replay/hash audit passed
Clean release audit after WaveLLM formal implementation: 105 hashed files selected; 156 internal/legacy paths excluded
Canonical dependency audit: 50 modules / 102 edges / 0 missing targets / 0 legacy imports / 0 cycles
Reviewer release remains blocked only on LICENSE and the provenance-gated radar example
Caption-generation support is mT5-only by policy; the unsupported legacy backend is excluded and guarded by a release content test
```

Research profile 已安装 Lightning 2.6.5、Transformers 4.57.6、PEFT 0.20.0、SciPy/HDF5、sentence-transformers、OpenCV、W&B 等核心依赖。DeepSpeed 作为 `distributed` profile 按需安装，不进入默认研究环境。

## 2. 当前有效主链

### OmniHand

```text
non-negative radar-cube power [B,T,Doppler,Range,Azimuth,Elevation]
  -> vectorized depthwise CubeNet + optional 3D PAFPN
  -> independently configurable channel/spatial/SE attention
  -> mask-aware temporal transformer + learned CLS/mean/attention mixture
  -> metric pose regressor [B,2,24,3]
```

该边界已在 synthetic tensor 上完成工程验证；raw ADC 到物理 radar cube 的 beamforming 和坐标校准
仍由 `BLOCK-RADAR-PROVENANCE` 阻塞，不能用本 smoke 代替。

### WaveLLM

```text
predicted pose [B,T,2,24,3] and/or feature [B,T,D]
  -> pose GCN / feature projection
  -> dynamic multimodal fusion
  -> concatenate with MT5 encoder embeddings
  -> caption generation
```

该链已在 synthetic model-ready manifest 上完成 clean-commit A100/BF16 正式 train/evaluate 闭环；
`mmprism.sign_language_translation.sample_v1`、adapter-only checkpoint、checkpoint/task/model-asset
绑定和 `mmprism.language_metric.character_v1` 已通过独立审计。真实 pose/radar feature 仍由上游
`BLOCK-RADAR-PROVENANCE` 和真实数据到达情况阻塞，不能用本工程 smoke 代替论文结果。

## 3. 已确认的架构问题

| ID | 问题 | 影响 | 迁移原则 |
|---|---|---|---|
| `ARCH-001` | 入口、配置和工具职责混杂 | 难以测试和组合 | CLI 只解析参数，业务逻辑进入 package service |
| `ARCH-002` | YAML 使用动态字符串 import 和可变 EasyDict | 无 schema、错误延迟到运行期 | structured config + config validation |
| `ARCH-003` | 多处绝对路径、GPU、bf16、rank 假设 | 无法跨机器运行 | RuntimeConfig/PathConfig 统一注入 |
| `ARCH-004` | dataset 用字符串替换推断模态路径 | 路径脆弱、难做 provenance | manifest record 显式列出每个 modality |
| `ARCH-005` | `cubenet_rtm.py` 等模块已删除但配置仍引用 | 配置启动失败 | 先 forensic audit，再归档/恢复/替换 |
| `ARCH-006` | README、CLAUDE、脚本与当前源码漂移 | 新执行者容易跑错 | 文档由 validated commands 反向生成/维护 |
| `ARCH-007` | legacy model factory 重复定义，Phi-3 API 与 base 不一致 | 若公开会造成伪支持 | `DEC-027` 固定 canonical mT5-only；legacy Phi-3 排除并由 release content gate 防回流 |
| `ARCH-008` | temporal frame processing 使用 Python 循环 | 性能和显存低效 | 正确性冻结后再 batch flatten/vectorize |
| `ARCH-009` | 测试期间逐 batch 读写同一 JSON | 慢且易损坏 | rank-local append/aggregate artifact writer |
| `ARCH-010` | 指标、模型、运行脚本没有 protocol version | 数值漂移难解释 | metric/data/model protocol 显式版本化 |
| `ARCH-011` | 稿件与 legacy 的带宽、chirp、阵列、clutter 和 steering 共轭不一致 | 直接照搬将生成不可解释的 4D cube | range-Doppler 独立验收；beamforming 等 acquisition/calibration evidence |
| `ARCH-012` | 开发仓库必须保留 legacy/internal 证据，但 reviewer archive 必须排除 | 直接复制仓库会再次提交 CLAUDE、硬编码 legacy 和私人材料 | tracked allowlist + required/forbidden/content/import/cycle audit |

## 4. Canonical Package

新系统已开始采用标准 `src` layout：

```text
pyproject.toml
configs/
  data/
  radar/
  model/
  experiment/
src/mmprism/
  contracts/
  config/
  data/
    schemas.py
    manifests.py
    datasets/
    transforms/
    validation/
  radar/
    processing/
    simulation/
  models/
    omnihand/
    wavellm/
    encoders/
    fusion/
  training/
  evaluation/
  artifacts/
  release/
  runtime/
  cli/
tests/
  unit/
  contracts/
  integration/
  fixtures/
```

旧 `src.data.*`、`src.fmcw.*`、`src.model.*` 仅作为 forensic reference；canonical package 不导入旧模块，也不提供 re-export wrapper。

## 5. 模块边界

- `data`：只负责 sample record、读取、变换、collate 和 split；不包含模型逻辑。
- `radar`：纯输入输出明确的信号处理和仿真；不依赖 Lightning。
- `models`：纯 `nn.Module`；不读路径、不写文件、不创建 logger。
- `training`：Lightning/DeepSpeed 适配、优化器、checkpoint 和 distributed 行为。
- `evaluation`：版本化 metric protocol 和 sample-level result schema。
- `artifacts`：resolved config、run metadata、prediction 和 summary 的统一写入。
- `cli`：薄入口，仅组合 config 与 service。

## 6. 实现顺序

1. `OPS/Config`：环境、路径、配置校验、run metadata。
2. `Data Contract`：manifest、split、fixture、validator。
3. `Radar Processor`：从明确数组契约实现仿真与处理，并建立数值测试。
4. `OmniHand`：encoder/regressor/training/evaluation 解耦。
5. `WaveLLM`：modality encoder/fusion/LLM wrapper/training 解耦。
6. `Evaluation`：统一 prediction schema 和 metric protocol。
7. `Legacy Archive/Release`：paper evidence 锁定后归档旧实现，公开 release 只保留验证过的新入口。

## 7. 测试门槛

- Unit：路径解析、manifest schema、normalization、split、metric。
- Contract：每个 dataset adapter 的 shape/dtype/key；雷达 processor 输出契约。
- Golden：固定小样本上的 legacy/new 数值容差。
- Integration：各训练入口 2 batch train/val/test。
- GPU smoke：单卡 bf16/fp32、DDP 两卡、DeepSpeed 最小步数。
- Reproduction：至少一个 OmniHand checkpoint 和一个 WaveLLM checkpoint 的完整评测。
