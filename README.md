# CantioAI Complete System
A production-ready, integrated AI system with **system-level advantages** that make individual AI models more reliable, observable, and maintainable.

## Project Structure

```
cantioai/
├── README.md
├── requirements.txt
├── config.yaml
├── setup.py
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hybrid_predictor.py
│   │   ├── pitch_quantizer.py
│   │   └── hybrid_svc.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── synthesizer.py
│   │   └── vocoder.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── infer.py
│   └── evaluate.py
├── tests/
│   └── test_basic.py
└── notebooks/
    └── 01_quickstart.ipynb
```

## Core Technologies

- **Excitation Source**: WORLD (f0, ap)
- **Neural Network**: Predicts spectral envelope (sp)
- **Architecture**: Hybrid Source-Filter + Neural Vocoder

## Key Features

- **System-Level Reliability**: 工作级错误处理、优雅关机机制和故障隔离
- **Enhanced Observability**: 实时健康监控、性能指标收集和阈值告警
- **Simplified Maintainability**: 统一配置管理、清晰的分层架构和一致的开发者约定
- **Flexible Deployment**: 4种启动模式（完整系统、仅后端、仅前端、全栈）支持从开发到生产的完整生命周期
- **Production Readiness**: Windows 兼容性、详细日志和系统就绪检查

## Installation

```bash
pip install -e .
```

## Usage

See `scripts/train.py` for training and `scripts/infer.py` for inference.

## Key Features of Stage 9 Implementation

- **统一配置系统**: 单一事实来源，支持跨阶段引用解析
- **分层初始化**：10阶段明确初始化顺序，正确的依赖管理
- **健康监控**：实时指标收集、阈值告警、系统健康评估
- **灵活部署**：4种启动模式（完整系统、仅后端、仅前端、全栈）
- **生产就绪**：Windows 兼容性、错误处理、优雅关机
- **完全集成**：所有 8 个前序阶段统一为一个系统

## License

Apache License 2.0

.