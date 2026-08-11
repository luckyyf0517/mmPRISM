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
provenance、统一 CLI 和 dependency-light unit tests。CSL-News RTMW3D 标注已具有独立 strict
config、原子 artifact、resume/failure contract 和 GPU smoke；Radar、OmniHand、WaveLLM 与通用
训练 artifact writer 尚未实现。

Foundation and environment verification (`2026-08-11`)：

```text
UV 0.11.23 / Python 3.12.13 / uv.lock
24 unit tests passed
doctor/config/plan/manifest CLI passed
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
```

Research profile 已安装 Lightning 2.6.5、Transformers 4.57.6、PEFT 0.20.0、SciPy/HDF5、sentence-transformers、OpenCV、W&B 等核心依赖。DeepSpeed 作为 `distributed` profile 按需安装，不进入默认研究环境。

## 2. 当前有效主链

### OmniHand

```text
raw complex radar or simulated pose/velocity
  -> Processor(range optional, Doppler FFT, beamforming)
  -> [B, 64, 32, 32, 32] cube
  -> CubeNet / MMHandEncoder / CubeNetTransformer
  -> linear regressor
  -> [B, 2, 24, 3] joints
```

### WaveLLM

```text
predicted pose [B,T,2,24,3] and/or feature [B,T,D]
  -> pose GCN / feature projection
  -> dynamic multimodal fusion
  -> concatenate with MT5 encoder embeddings
  -> caption generation
```

## 3. 已确认的架构问题

| ID | 问题 | 影响 | 迁移原则 |
|---|---|---|---|
| `ARCH-001` | 入口、配置和工具职责混杂 | 难以测试和组合 | CLI 只解析参数，业务逻辑进入 package service |
| `ARCH-002` | YAML 使用动态字符串 import 和可变 EasyDict | 无 schema、错误延迟到运行期 | structured config + config validation |
| `ARCH-003` | 多处绝对路径、GPU、bf16、rank 假设 | 无法跨机器运行 | RuntimeConfig/PathConfig 统一注入 |
| `ARCH-004` | dataset 用字符串替换推断模态路径 | 路径脆弱、难做 provenance | manifest record 显式列出每个 modality |
| `ARCH-005` | `cubenet_rtm.py` 等模块已删除但配置仍引用 | 配置启动失败 | 先 forensic audit，再归档/恢复/替换 |
| `ARCH-006` | README、CLAUDE、脚本与当前源码漂移 | 新执行者容易跑错 | 文档由 validated commands 反向生成/维护 |
| `ARCH-007` | model factory 重复定义，Phi-3 API 与 base 不一致 | 支持范围不清 | 明确 MT5 baseline；Phi-3 单独验收或移除 |
| `ARCH-008` | temporal frame processing 使用 Python 循环 | 性能和显存低效 | 正确性冻结后再 batch flatten/vectorize |
| `ARCH-009` | 测试期间逐 batch 读写同一 JSON | 慢且易损坏 | rank-local append/aggregate artifact writer |
| `ARCH-010` | 指标、模型、运行脚本没有 protocol version | 数值漂移难解释 | metric/data/model protocol 显式版本化 |

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
