"""
损失函数模块
"""

from .loss_manager import LossManager

# 尝试导入损失函数实现
try:
    from ..adversarial_loss import AdversarialLoss
    from ..feature_matching_loss import FeatureMatchingLoss
    from ..control_consistency_loss import ControlConsistencyLoss
    from .enhanced_adversarial_loss import EnhancedAdversarialLoss
    from .enhanced_feature_matching_loss import EnhancedFeatureMatchingLoss
    from .detailed_consistency_loss import DetailedConsistencyLoss
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from cantioai.adversarial_loss import AdversarialLoss
        from cantioai.feature_matching_loss import FeatureMatchingLoss
        from cantioai.control_consistency_loss import ControlConsistencyLoss
        from cantioai.discriminators.losses.enhanced_adversarial_loss import EnhancedAdversarialLoss
        from cantioai.discriminators.losses.enhanced_feature_matching_loss import EnhancedFeatureMatchingLoss
        from cantioai.discriminators.losses.detailed_consistency_loss import DetailedConsistencyLoss
    except ImportError:
        AdversarialLoss = FeatureMatchingLoss = ControlConsistencyLoss = None
        EnhancedAdversarialLoss = EnhancedFeatureMatchingLoss = DetailedConsistencyLoss = None

__all__ = [
    "LossManager",
    "AdversarialLoss",
    "EnhancedAdversarialLoss",
    "FeatureMatchingLoss",
    "EnhancedFeatureMatchingLoss",
    "ControlConsistencyLoss",
    "DetailedConsistencyLoss"
]