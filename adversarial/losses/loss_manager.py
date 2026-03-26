"""
损失管理器：
负责管理和计算不同类型的损失函数
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import loss classes
try:
    from ..adversarial_loss import AdversarialLoss
    from ..feature_matching_loss import FeatureMatchingLoss
    from ..control_consistency_loss import ControlConsistencyLoss
    from .enhanced_adversarial_loss import EnhancedAdversarialLoss
    from .enhanced_feature_matching_loss import EnhancedFeatureMatchingLoss
    from .detailed_consistency_loss import DetailedConsistencyLoss
except ImportError:
    # Fallback for when modules are not in expected location
    AdversarialLoss = FeatureMatchingLoss = ControlConsistencyLoss = None
    EnhancedAdversarialLoss = EnhancedFeatureMatchingLoss = DetailedConsistencyLoss = None

logger = logging.getLogger(__name__)


class LossManager:
    """
    损失管理器：
    负责管理和计算不同类型的损失函数
    """

    def __init__(self, config: Dict):
        """
        初始化损失管理器

        参数:
            config: 损失函数配置
        """
        self.config = config
        self.losses = {}
        self._init_losses()

    def _init_losses(self):
        """初始化损失函数"""
        loss_config = self.config.get("losses", {})

        # 对抗损失
        if loss_config.get("adversarial", {}).get("enabled", False):
            adv_type = loss_config.get("adversarial", {}).get("type", "hinge")
            if adv_type in ["gan", "wgan", "wgan_gp", "hinge"] and EnhancedAdversarialLoss is not None:
                self.losses["adversarial"] = EnhancedAdversarialLoss(loss_config.get("adversarial", {}))
                logger.info(f"初始化增强对抗损失: {adv_type}")
            elif AdversarialLoss is not None:
                self.losses["adversarial"] = AdversarialLoss(loss_config.get("adversarial", {}))
                logger.info("初始化标准对抗损失")
            else:
                logger.warning("对抗损失初始化失败")

        # 特征匹配损失
        if loss_config.get("feature_matching", {}).get("enabled", False):
            if EnhancedFeatureMatchingLoss is not None:
                self.losses["feature_matching"] = EnhancedFeatureMatchingLoss(loss_config.get("feature_matching", {}))
                logger.info("初始化增强特征匹配损失")
            elif FeatureMatchingLoss is not None:
                self.losses["feature_matching"] = FeatureMatchingLoss(loss_config.get("feature_matching", {}))
                logger.info("初始化标准特征匹配损失")
            else:
                logger.warning("特征匹配损失初始化失败")

        # 一致性损失
        if loss_config.get("consistency", {}).get("enabled", False):
            if DetailedConsistencyLoss is not None:
                self.losses["consistency"] = DetailedConsistencyLoss(loss_config.get("consistency", {}))
                logger.info("初始化详细一致性损失")
            elif ControlConsistencyLoss is not None:
                self.losses["consistency"] = ControlConsistencyLoss(loss_config.get("consistency", {}))
                logger.info("初始化标准一致性损失")
            else:
                logger.warning("一致性损失初始化失败")

    def compute_losses(self,
                      real_outputs: Dict[str, torch.Tensor],
                      fake_outputs: Dict[str, torch.Tensor],
                      real_data: Optional[torch.Tensor] = None,
                      fake_data: Optional[torch.Tensor] = None,
                      features_real: Optional[Dict] = None,
                      features_fake: Optional[Dict] = None,
                      discriminator_type: str = "unknown") -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict]]:
        """
        计算所有损失

        参数:
            real_outputs: 判别器对真实数据的输出
            fake_outputs: 判别器对假数据的输出
            real_data: 真实数据（用于梯度惩罚等）
            fake_data: 假数据
            features_real: 真实数据的中间特征
            features_fake: 假数据的中间特征
            discriminator_type: 判别器类型

        返回:
            (总损失字典, 详细损失信息字典)
        """
        total_losses = {
            "generator": 0.0,
            "discriminator": 0.0
        }
        loss_details = {}

        for loss_name, loss_fn in self.losses.items():
            try:
                if loss_name == "adversarial":
                    # 对抗损失需要分别计算生成器和判别器损失
                    gen_loss, gen_details = loss_fn.generator_loss(fake_outputs, discriminator_type)
                    disc_loss, disc_details = loss_fn.discriminator_loss(
                        real_outputs, fake_outputs, discriminator_type, real_data
                    )

                    total_losses["generator"] += gen_loss
                    total_losses["discriminator"] += disc_loss

                    loss_details[f"{loss_name}_generator"] = gen_details
                    loss_details[f"{loss_name}_discriminator"] = disc_details

                elif loss_name == "feature_matching":
                    # 特征匹配损失
                    if features_real is not None and features_fake is not None:
                        loss, details = loss_fn(features_real, features_fake)
                        total_losses["generator"] += loss  # 特征匹配通常只用于生成器
                        loss_details[loss_name] = details
                    else:
                        logger.warning("特征匹配损失需要特征输入")

                elif loss_name == "consistency":
                    # 一致性损失
                    if fake_data is not None:
                        # 一致性损失通常需要生成的音频和控制参数
                        loss, details = loss_fn.compute(
                            fake_audio=fake_data,
                            # 这里需要根据实际情况传递控制参数
                        )
                        total_losses["generator"] += loss
                        loss_details[loss_name] = details
                    else:
                        logger.warning("一致性损失需要音频输入")

            except Exception as e:
                logger.error(f"计算损失 {loss_name} 时出错: {e}")

        return total_losses, loss_details

    def get_loss_weights(self) -> Dict[str, float]:
        """
        获取损失权重

        返回:
            损失权重字典
        """
        weights = {}
        loss_config = self.config.get("losses", {})

        for loss_name in self.losses.keys():
            weight_config = loss_config.get(loss_name, {}).get("weight", 1.0)
            weights[loss_name] = weight_config

        return weights

    def get_available_losses(self) -> List[str]:
        """
        获取可用的损失函数

        返回:
            可用损失函数列表
        """
        return list(self.losses.keys())