"""
训练器模块
"""

# 尝试导入训练器实现
try:
    from .progressive_trainer import ProgressiveAdversarialTrainer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from cantioai.adversarial.trainers.progressive_trainer import ProgressiveAdversarialTrainer
    except ImportError:
        ProgressiveAdversarialTrainer = None

__all__ = [
    "ProgressiveAdversarialTrainer"
]