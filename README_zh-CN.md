# CantioAI 完整系统
一个具有**系统级优势**的生产就绪、集成的AI系统，这些优势使得单个AI模型更加可靠、可观测和可维护。

## 项目结构
```
cantioai/
├── README.md
├── README_zh-CN.md          # 本文件
├── requirements.txt
├── setup.py
├── .gitignore
├── cantioai/                # 前端应用
├── configs/
│   └── integrated/          # 统一配置文件
│       ├── cantioai.yaml            # 主配置
│       ├── hybrid.yaml              # 混合架构配置
│       ├── adversarial.yaml         # 对抗训练配置
│       ├── multitask.yaml           # 多任务学习配置
│       ├── transformer.yaml         # Transformer 骨干网络配置
│       ├── hubert.yaml              # HuBERT 特征提取配置
│       ├── diffusion.yaml           # 扩散模型增强配置
│       └── optimization.yaml        # 实时推理优化配置
├── data/
│   ├── raw/
    │   ├── processed/
    │   └── datasets/
├── logs/                    # 系统日志
├── models/                  # 训练好的模型
├── results/                 # 实验结果
├── src/
│   ├── __init__.py
│   ├── data/                # 数据处理层
│   │   ├── __init__.py
│   │   ├── data/
    │   │   │   ├── __init__.py
│   │   │   ├── dataset.py
    │   │   │   ├── preprocess.py
    │   │   │   └── utils.py
│   ├── models/              # 模型层
│   │   ├── __init__.py
│   │   │   ├── models/
    │   │   │   │   ├── __init__.py
│   │   │   │   ├── sentiment_transformer.py    # 情感分析Transformer
    │   │   │   │   ├── multitask_head.py           # 多任务输出头
    │   │   │   │   └── utils.py
│   ├── training/                 # 训练层
│   │   │   │   ├── __init__.py
│   │   │   │   ├── trainer.py
│   │   │   │   ├── loss_functions.py
│   │   │   │   └── utils.py
│   ├── inference/                # 推理层
│   │   │   │   ├── __init__.py
│   │   │   │   ├── predictor.py
│   │   │   │   ├── batch_processor.py
│   │   │   │   │   └── utils.py
│   ├── service/                  # 服务层
│   │   │   │   ├── __init__.py
│   │   │   │   │   ├── api_server.py         # REST API服务
│   │   │   │   │   ├── websocket_handler.py  # WebSocket实时服务
│   │   │   │   │   └── utils.py
│   └── utils/                    # 工具层
│       ├── config_integrated.py    # 统一配置加载器
│       ├── system_initializer.py   # 系统初始化器
│       ├── system_monitor.py       # 系统健康监控器
│       └── ...                         # 其他实用工具
├── scripts/
│   ├── start_cantioai.py         # 统一启动脚本
│   ├── preprocess.py
│   ├── train.py
│   ├── infer.py
│   └── evaluate.py
├── tests/
│   └── test_basic.py
└── notebooks/
    └── 01_quickstart.ipynb
```

## 核心技术
- **激励声源**：WORLD (f0, ap)
- **神经网络**：预测谱包络 (sp)
- **架构**：混合源-滤波 + 神经声码器

## 主要特性
- **系统级可靠性**：工作级错误处理、优雅关机机制和故障隔离
- **增强的可观测性**：实时健康监控、性能指标收集和阈值告警
- **简化的可维护性**：统一配置管理、清晰的分层架构和一致的开发者约定
- **灵活的部署**：4种启动模式（完整系统、仅后端、仅前端、全栈）支持从开发到生产的完整生命周期
- **生产就绪**：Windows 兼容性、详细日志和系统就绪检查
- **真正集成**：所有 8 个前序阶段通过统一框架真正集成为一个协同工作的系统

## 第9阶段实现的主要特性
- **统一配置系统**：单一事实来源，支持跨阶段引用解析
- **分层初始化**：10阶段明确初始化顺序，正确的依赖管理
- **健康监控**：实时指标收集、阈值告警、系统健康评估
- **灵活部署**：4种启动模式（完整系统、仅后端、仅前端、全栈）支持从开发到生产的完整生命周期
- **生产就绪**：Windows 兼容性、错误处理、优雅关机
- **完全集成**：所有 8 个前序阶段统一为一个系统

## 安装
```bash
pip install -e .
```

## 使用方法
参考 `scripts/train.py` 进行训练和 `scripts/infer.py` 进行推理。

## 许可证
Apache License 2.0
```

## 致谢
基于 [HybridSpectralPredictor](https://github.com/yourusername/HybridSpectralPredictor) 和 [DifferentiablePitchQuantizer](https://github.com/yourusername/DifferentiablePitchQuantizer) 实现构建。