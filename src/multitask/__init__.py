"""
Multi-task Learning Components
Updated to support Transformer-based Hybrid Spectral Predictor
"""
from .shared_encoder import MultiTaskSharedEncoder
from .task_heads import (
    SingingConversionHead,
    SpeechConversionHead,
    NoiseRobustnessHead,
    create_task_head
)
from .training_strategies import (
    MultiTaskTrainer,
    create_progressive_multi_task_trainer,
    create_curriculum_learning_trainer
)
from ..dataloader import create_multitask_dataloader

__all__ = [
    "MultiTaskSharedEncoder",
    "SingingConversionHead",
    "SpeechConversionHead",
    "NoiseRobustnessHead",
    "create_task_head",
    "MultiTaskTrainer",
    "create_progressive_multi_task_trainer",
    "create_curriculum_learning_trainer",
    "create_multitask_dataloader"
]