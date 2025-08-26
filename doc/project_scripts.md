# 项目Python脚本功能说明文档

## 根目录运行脚本

### 1. run_model.py
**功能**: OmniHand模型训练和测试的主入口脚本
- 支持分布式训练（DDP）
- 配置文件驱动的训练流程
- 支持训练和测试模式切换
- 集成Wandb和SwanLab日志记录
- 支持模型检查点保存和恢复

### 2. run_peft.py
**功能**: WaveLLM模型的PEFT（参数高效微调）训练脚本
- 集成DeepSpeed进行大规模模型训练优化
- 支持LoRA微调配置
- 多GPU分布式训练
- 混合精度训练（BF16/FP16）
- 支持梯度累积和学习率调度

### 3. run_evaluation.py
**功能**: 模型评估和指标计算脚本
- 计算传统NLP指标（BLEU, ROUGE-L）
- 计算语义相似度指标（SBERT, SimCSE）
- 批量处理评估结果
- 支持多GPU语义相似度计算
- 结果保存为JSON格式

### 4. run_simulation.py
**功能**: mmWave信号仿真脚本
- 从3D姿态数据生成雷达信号
- 点云数据处理和插值
- 雷达信号模拟和保存
- 支持批量处理多个序列
- 多GPU并行处理支持

### 5. run_extract_feature.py
**功能**: 特征提取和姿态预测脚本
- 使用训练好的OmniHand模型提取特征
- 生成预测姿态数据
- 批量处理数据集
- 支持分布式特征提取
- MPJPE指标实时计算

### 6. run_csl_daily_annotation.py / run_csl_news_annotation.py / run_collected_annotation.py
**功能**: 数据集标注处理脚本
- 处理CSL-Daily、CSL-News和Collected数据集的标注
- 标注格式转换和验证
- 数据集分割和组织

### 7. run_evaluate_each.py
**功能**: 单样本评估脚本
- 对单个样本进行详细评估
- 生成评估报告和可视化结果

## src/scripts/ 目录脚本

### 1. extract.py
**功能**: 数据提取和组织脚本
- 从原始收集数据中提取不同类型文件
- 按文件类型组织数据结构
- 支持mmWave数据帧分割
- 自动创建输出目录结构

### 2. split.py
**功能**: 数据集分割脚本
- 基于哈希的确定性数据分割
- 支持训练集/验证集划分
- 生成JSON格式的分割文件
- 可配置验证集比例

### 3. check.py
**功能**: 数据质量检查脚本
- 验证姿态数据文件格式和维度
- 检测并报告无效文件
- 提供详细的检查日志
- 支持批量数据验证

### 4. stat.py
**功能**: 数据统计脚本
- 计算数据集统计信息
- 分析数据分布和特征
- 生成统计报告

## src/utils/ 目录工具脚本

### 1. tools.py
**功能**: 通用工具函数
- 模型实例化工具（instantiate_from_config）
- 动态类导入工具（get_obj_from_str）
- 网络参数梯度控制工具

### 2. io.py
**功能**: 输入输出工具
- YAML配置文件加载（load_yaml）
- 文件列表加载（load_file_list）

### 3. plot.py
**功能**: 数据可视化工具
- 训练曲线绘制
- 结果可视化
- 图表生成和保存

### 4. deepspeed_utils.py
**功能**: DeepSpeed训练工具
- DeepSpeed配置生成
- 训练参数配置
- 分布式训练支持

## src/data/ 目录数据处理脚本

### 1. data_interface.py
**功能**: 数据接口模块
- PyTorch Lightning数据模块实现
- 数据加载器配置和管理
- 训练/验证/测试数据集划分

### 2. dataset.py
**功能**: 数据集实现
- 多种数据集类实现（SingleFrameDataset, SequenceBaseDataset等）
- 数据预处理和增强
- 不同模态数据支持（姿态、特征、mmWave信号）

## src/model/ 目录模型脚本

### 1. omnihand.py
**功能**: OmniHand手部姿态估计模型
- 3D手部姿态估计主模型
- GAN训练支持
- 损失函数计算和优化

### 2. trainer.py
**功能**: WaveLLM训练器
- LLM模型训练和微调
- 多模态数据融合
- PEFT/LoRA支持

### 3. encoder/ 目录编码器
- **cubenet_rtm.py**: 基础CubeNet RTM编码器
- **cubenet_rtm_temporal.py**: 时序处理编码器（TVAN-inspired）
- **cubenet_rtm_transformer.py**: Transformer时序编码器
- **cubenet_rtm_lstm.py**: LSTM时序编码器
- **pose_encoder.py**: 姿态编码器（GCN网络）
- **mmhand_encoder.py**: MMHand编码器

### 4. llm/ 目录语言模型
- **model_factory.py**: 模型工厂模式实现
- **mt5_model.py**: MT5模型实现
- **phi3_model.py**: Phi-3模型实现

### 5. discriminator.py
**功能**: 判别器模型
- 关键点判别器实现
- GAN训练支持

## fmcw/ 目录雷达信号处理脚本

### 1. simulator.py
**功能**: 雷达信号仿真器
- mmWave信号处理和仿真
- 点云数据处理
- 雷达帧生成

### 2. beamformer.py
**功能**: 波束成形处理
- 天线阵列波束成形计算
- 方向向量构建

### 3. fmcw_radar.py
**功能**: FMCW雷达配置
- 雷达参数配置和管理

## 使用说明

### 训练流程
1. **数据准备**: 使用src/scripts/extract.py组织数据
2. **数据分割**: 使用src/scripts/split.py划分数据集
3. **模型训练**: 使用run_model.py或run_peft.py进行训练
4. **模型评估**: 使用run_evaluation.py进行评估

### 特征提取流程
1. **预训练模型**: 准备预训练的OmniHand模型
2. **特征提取**: 运行run_extract_feature.py提取特征
3. **结果验证**: 检查生成的特征和姿态文件

### 信号仿真流程
1. **姿态数据**: 准备3D姿态数据
2. **信号生成**: 运行run_simulation.py生成雷达信号
3. **结果保存**: 保存生成的信号文件