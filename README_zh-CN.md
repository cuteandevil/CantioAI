# CantioAI 完整系统
一个生产就绪、集成的语音转换AI系统，它结合了传统信号处理（WORLD vocoder）和现代深度学习（基于Transformer的架构）来提供高质量、可靠的唱唱转语音转换，具有系统级优势。

## 🎯 项目目的
CantioAI通过将语音转换视为一个**多任务学习问题**来解决高质量语音转换挑战，系统同时学习：
- 唱唱转唱唱语音转换（F0、谱带、非周期性）
- 语音转语音转换（F0、谱带）
- 噪声鲁棒性（去噪、干净特征估计、信噪比预测）
同时保持与传统音频合成管道的兼容性。

## 🏗️ 系统架构

### 核心处理流水线
```mermaid
graph LR
A[输入音频] --> B[WORLD分析]
B --> C[特征提取]
C --> D[多任务混合模型]
D --> E[任务特定预测]
E --> F[WORLD合成]
F --> G[后处理]
G --> H[输出音频]
```

### 关键技术创新
1. **基于Transformer的骨干网络**：用层次Transformer替换CNN+BiLSTM以获得更好的多尺度时间建模
2. **多任务学习框架**：共享编码器配合任务特定头实现联合优化
3. **可微分音高量化**：使用直通估计器实现音乐精准的F0预测
4. **任务条件AdaIN**：自适应实例归一化实现说话者特征变换
5. **集成配置系统**：统一的YAML配置带引用解析

## 📁 项目结构
```
CantioAI/
├── cantioai/                 # 主源代码
│   ├── __init__.py
│   ├── data/                 # 数据加载 & 预处理
│   │   ├── dataset.py        # PyTorch数据集 for .npz特征
│   │   └── utils/            # 音频处理工具（WORLD、归一化等）
│   ├── models/               # 神经网络架构
│   │   ├── hybrid_predictor.py   # 基于Transformer的光谱预测器
│   │   ├── pitch_quantizer.py    # 可微分F0量化
│   │   └── hybrid_svc.py         # 多任务语音转换系统
│   ├── inference/            # 音频合成 & 推理
│   │   └── synthesizer.py        # 基于WORLD的音频合成
│   ├── training/             # 训练循环 & 优化
│   │   ├── trainer.py          # 主训练循环
│   │   └── losses.py           # 损失计算函数
│   ├── utils/                # 系统工具
│   │   ├── config_integrated.py    # 统一配置加载器
│   │   ├── system_initializer.py   # 系统启动/关闭
│   │   └── system_monitor.py       # 健康监控 & 警报
│   └── main.py               # 系统入口点
├── multitask/                # 多任务学习组件
│   ├── shared_encoder.py     # 跨任务特征共享
│   ├── task_heads.py         # 任务特定预测头
│   ├── adaptive_norm.py      # 任务条件AdaIN
│   ├── loss_design.py        # 多任务损失平衡
│   ├── training_strategies.py # 动态任务调度
│   └── evaluation_framework.py # 全面评估指标
├── scripts/                  # 入口点脚本
│   ├── train.py              # 模型训练
│   ├── infer.py              # 音频合成/推理
│   └── evaluate.py           # 系统评估
├── configs/                  # 配置文件
│   └── integrated/           # 集成配置
│       └── cantioai.yaml     # 基础系统配置
├── data/                     # 数据目录
│   ├── raw/                  # 原始音频文件
│   ├── processed/            # 预处理特征(.npz)
│   └── datasets/             # 精选数据集
├── results/                  # 输出目录
├── logs/                     # 日志文件
├── checkpoints/              # 模型检查点
└── notebooks/                # Jupyter笔记本
    └── 01_quickstart.ipynb # 入门指南
```

## 🔧 关键特性

### 系统级可靠性
- **生产级错误处理**：优雅降级、详细错误报告、自动恢复
- **故障隔离**：组件级失败容纳防止系统范围崩溃
- **健康监控**：实时指标收集（延迟、吞吐量、内存使用）配合阈值告警
- **资源管理**：自动清理、内存泄漏预防、GPU利用优化

### 增强可观测性
- **结构化日志**：带时间戳、级别和上下文信息的一致格式
- **性能指标**：推理延迟、训练吞吐量、收敛跟踪
- **调试接口**：中间表示检查、梯度流分析
- **实验跟踪**：与TensorBoard和Weights & Biases集成

### 简化可维护性
- **统一配置**：单个YAML文件带跨阶段引用解析
- **分层架构**：显式关注点分离（数据、模型、训练、推理）
- **一致接口**：标准化数据形状、可预测张量维度
- **文档优先方法**：清晰的模块文档字符串、使用示例、类型提示

### 灵活部署
- **4种启动模式**：
  * 完整系统：训练 + 推理 + 监控
  * 仅后端：训练 + 监控（无音频I/O）
  * 仅前端：推理 + 监控（使用预训练模型）
  * 全栈：完整系统带Web界面
- **跨平台支持**：Windows/Linux兼容性，CPU/GPU加速
- **容器就绪**：支持Docker实现可重现部署

### 生产就绪
- **Windows兼容性**：在Windows 10/11上测试通过CUDA支持
- **详细日志**：全面审计轨迹用于调试和合规
- **系统就绪检查**：启动前依赖、配置、资源预验证
- **确定性行为**： seeded随机数以获得可重现结果

## ⚙️ 配置系统

统一配置系统（`config.yaml`）支持：

### 数据配置
```yaml
data:
  raw_dir: data/raw/
  processed_dir: data/processed/
  datasets_dir: data/datasets/
  train_dataset: train_features.npz
  val_dataset: val_features.npz
  test_dataset: test_features.npz
  audio_extensions: [".wav", ".flac", ".mp3"]
```

### 特征提取（WORLD分析）
```yaml
feature:
  frame_period: 5.0        # 毫秒（帧移位）
  fft_size: 1024
  f0_floor: 71.0             # 赫兹（最小F0）
  f0_ceil: 800.0             # 赫兹（最大F0）
  num_mcep: 60               # 梅尔倒谱系数维度
  mcep_alpha: 0.41           # 全通常数
  normalize_features: true
  f0_norm_method: "log"      # 选项："log", "standard", "minmax"
  sp_norm_method: "standard" # 选项："standard", "minmax"
  silence_threshold: 0.03    # 振幅阈值用于静音检测
  f0_interpolation: "linear" # 选项："linear", "none"
```

### 模型架构
```yaml
model:
  # 基于Transformer的HybridSpectralPredictor（替换CNN+BiLSTM）
  phoneme_feature_dim: 32     # D_ph - 音素特征维度
  spectral_envelope_dim: 60    # D_sp - 输出光谱包络维度
  speaker_embed_dim: 128       # D_spk - 说话者嵌入维度
  n_speakers: 100              # 说话者总数
  use_pitch_quantizer: true    # 启用可微分音高量化
  pitch_quantizer:
    ref_freq: 440.0            # A4参考频率
    ref_midi: 69               # A4的MIDI音符
    octaves: 10                # 八度范围覆盖
    use_ste: true              # 使用直通估计器
  transformer:
    type: "hierarchical"       # 分层Transformer用于多尺度处理
    hidden_dim: 512
    num_heads: 8
    num_layers: 6
    ff_dim: 2048
    dropout: 0.1
    max_seq_len: 5000
    positional_encoding:
      type: "relative_bias"    # 相对位置编码
    hierarchical:
      local_window: 32
      medium_window: 128
      global_window: "full"
      downsampling_factors: [2, 4]
      streaming:
        causal: true
```

### 训练参数
```yaml
training:
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 1e-5
  epochs: 100
  validation_interval: 1          # 每N个epoch验证一次
  save_interval: 10             # 每10个epoch保存一次检查点
  optimizer: "adam"             # 选项："adam", "adamw", "sgd"
  lr_scheduler: "step"         # 选项："step", "cosine", "plateau", "none"
  lr_step_size: 30              # 用于StepLR
  lr_gamma: 0.1                 # 用于StepLR
  grad_clip: 1.0                # 梯度裁剪最大范数（None禁用）
  use_amp: false                # 自动混合精度
  device: "auto"                # 选项："auto", "cpu", "cuda", "cuda:0"等
  seed: 42                      # 随机种子确保可重现
```

### 损失函数权重
```yaml
loss:
  sp_loss_weight: 1.0          # 光谱包络损失
  sp_loss_type: "l1"           # 选项："l1", "l2", "huber"
  f0_loss_weight: 0.1          # F0损失（如果使用音高量化器）
  f0_loss_type: "l1"
  adv_loss_weight: 0.0         # 对抗损失（未来扩展）
  huber_delta: 1.0             # Huber损失delta（如果使用）
```

### 实验设置
```yaml
experiment:
  name: "cantioai_base"
  description: "基线混合源滤波器+神经vocoder"
  log_interval: 100           # 每N个batch记录训练状态
  use_tensorboard: true
  tensorboard_dir: "logs/tensorboard"
  checkpoint_dir: "checkpoints"
  resume_from: ""               # 从检查点恢复训练的路径
  val_split: 0.1                # 训练数据用于验证的比例
  test_batch_size: 32
```

### 推理设置
```yaml
inference:
  synthesize_batch_size: 4
  default_f0_hz: 220.0        # 零射合成的默认F0
  # WORLD合成参数
  synth_frame_period: 5.0       # 毫秒
  apply_preemphasis: true
  preemphasis_coeff: 0.97
  normalize_output: true
  output_dir: "results/"
  output_format: "wav"         # 选项："wav", "flac"
```

### 扩散模型（可选增强）
```yaml
diffusion:
  enabled: true
  mode: "postprocess"         # postprocess, joint, direct
  process:
    type: "waveform"          # waveform, mel, spec
    scheduler: "cosine"       # linear, cosine, sqrt
    timesteps: 1000
    beta_start: 0.0001
    beta_end: 0.02
  denoiser:
    type: "diffwave"          # diffwave, wavenet, unet
    hidden_dim: 128
    num_layers: 30
    num_cycles: 10
    condition_projection: true
  conditioning:
    f0_method: "cross_attention" # 交叉注意力, adain, concat
    sp_method: "adain"
    ap_method: "concat"
    hubert_method: "cross_attention"
  training:
    strategy: "two_stage"     # two_stage, joint
    loss_type: "simple"         # simple, v, spec
    learning_rate: 1e-4
    weight_decay: 0.0
  sampling:
    sampler: "ddim"             # ddim, dpm, plms
    steps: 50
    guidance_scale: 3.0
    temperature: 1.0
  efficiency:
    use_checkpoint: true
    gradient_accumulation: 1
    mixed_precision: true
```

## 🚀 安装与使用

### 安装
```bash
# 克隆仓库
git clone https://github.com/cuteandevil/CantioAI.git
cd CantioAI

# 安装开发模式
pip install -e .

# 验证安装
python -c "import cantioai; print('CantioAI导入成功')"
```

### 训练
```bash
# 使用默认配置训练
python scripts/train.py --config configs/integrated/cantioai.yaml --data-dir data/processed/

# 自定义配置
python scripts/train.py --config my_config.yaml --data-dir /path/to/features --batch-size 32
```

### 推理
```bash
# 从特征合成音频
python scripts/infer.py --config configs/integrated/cantioai.yaml \
  --model-path checkpoints/best_model.pt \
  --input-path data/processed/test_features.npz \
  --output-path results/converted_audio.wav

# 批量合成
python scripts/infer.py --config configs/integrated/cantioai.yaml \
  --model-path checkpoints/latest.pt \
  --input-path data/processed/ \
  --output-path results/ \
  --batch-size 8
```

### 评估
```bash
# 运行基础功能测试
python scripts/evaluate.py
```

## 📊 评估框架

综合评估系统包括：

### 客观指标
- **梅尔倒谱失真（MCD）**：光谱相似度测量
- **基频均方根误差**：F0预测精度（赫兹和音度）
- **语音转换错误（VCE）**：总体转换质量
- **信噪比改善**：去噪效果

### 主观指标
- **平均意见分（MOS）**：人类感知评估
- **说话者相似度**：目标说话者匹配准确度
- **自然性**：转换语音的感知自然度
- **偏好测试**：与基线方法的A/B比较

### 鲁棒性指标
- **噪声鲁棒性**：各种噪声条件下的性能
- **说话者可变性**：不同说话者对之间的一致性
- **跨语言**：不同语言的性能
- **实时因子**：推理延迟测量

## 🔬 研究能力

### 支持的消融研究
- Transformer与CNN+BiLSTM骨干网络比较
- 共享编码器有效性分析
- 任务条件AdaIN消融
- 可微分音高量化影响
- 多任务vs单任务学习比较

### 扩展点
- 附加任务头（情感转换、语言适应）
- 替代vocoder（NeRAF、STRAIGHT）
- 不同激发源（SWIPE、YIN）
- 新型损失函数（感知、对抗）

## 📄 许可证

本项目采用Apache许可证2.0 - 请参看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

建立在以下研究之上：
- WORLD vocoder框架（https://github.com/mmorise/World）
- Transformer架构（Vaswani等，2017）
- 多任务学习原则（Ruder，2017）
- 神经vocoder进展（https://github.com/r9y9/wavenet_vocoder）

## 引用

如果您在研究中使用CantioAI，请引用：

```bibtex
@software{CantioAI,
  author = {CantioAI Contributors},
  title = {CantioAI: 完整语音转换系统},
  year = {2026},
  url = {https://github.com/cuteandevil/CantioAI}
}
```

---
*最后更新：2026年3月26日*
*版本：1.0.0（第9阶段 - 完整系统）