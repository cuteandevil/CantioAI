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

- **System-Level Reliability**: Production-grade error handling, graceful shutdown mechanisms, and fault isolation
- **Enhanced Observability**: Real-time health monitoring, performance metrics collection, and threshold-based alerting
- **Simplified Maintainability**: Unified configuration management, clear layered architecture, and consistent developer conventions
- **Flexible Deployment**: 4 startup modes (full system, backend-only, frontend-only, full-stack) supporting complete lifecycle from development to production
- **Production Readiness**: Windows compatibility, detailed logging, and system readiness checks

## Installation

```bash
pip install -e .
```

## Usage

See `scripts/train.py` for training and `scripts/infer.py` for inference.

## Key Features of Stage 9 Implementation

- **Unified Configuration System**: Single source of truth with cross-stage reference resolution support
- **Layered Initialization**: 10-stage explicit initialization order with proper dependency management
- **Health Monitoring**: Real-time metrics collection, threshold-based alerting, and system health assessment
- **Flexible Deployment**: 4 startup modes (full system, backend-only, frontend-only, full-stack)
- **Production Ready**: Windows compatibility, error handling, and graceful shutdown
- **Fully Integrated**: All 8 preceding stages unified into a single system

## License

Apache License 2.0

.