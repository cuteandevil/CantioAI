"""
CantioAI 对抗训练模块
"""

from .adversarial_manager import AdversarialManager
from .discriminators.discriminator_factory import DiscriminatorFactory
from .losses.loss_manager import LossManager

__all__ = [
    "AdversarialManager",
    "DiscriminatorFactory",
    "LossManager"
]