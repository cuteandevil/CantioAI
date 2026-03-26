"""
判别器工厂：
负责创建和管理不同类型的判别器
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

# Import discriminator classes
try:
    from ..control_aware_mpd import ControlAwareMPD
    from ..control_aware_msd import ControlAwareMSD
    from ..discriminators.specialized.source_filter_discriminator import SourceFilterDiscriminator
    from ..discriminators.specialized.control_consistency_disc import ControlConsistencyDiscriminator
except ImportError:
    # Fallback for when modules are not in expected location
    ControlAwareMPD = ControlAwareMSD = SourceFilterDiscriminator = ControlConsistencyDiscriminator = None

logger = logging.getLogger(__name__)


class DiscriminatorFactory:
    """
    判别器工厂：
    负责创建和管理不同类型的判别器
    """

    @staticmethod
    def create_discriminator(disc_type: str, config: Dict, device: torch.device) -> Optional[nn.Module]:
        """
        创建单个判别器

        参数:
            disc_type: 判别器类型 ("mpd", "msd", "source_filter", "control_consistency")
            config: 判别器配置
            device: 设备

        返回:
            判别器实例
        """
        if disc_type == "mpd":
            if ControlAwareMPD is None:
                logger.error("ControlAwareMPD not available")
                return None
            return ControlAwareMPD(config).to(device)

        elif disc_type == "msd":
            if ControlAwareMSD is None:
                logger.error("ControlAwareMSD not available")
                return None
            return ControlAwareMSD(config).to(device)

        elif disc_type == "source_filter":
            if SourceFilterDiscriminator is None:
                logger.error("SourceFilterDiscriminator not available")
                return None
            return SourceFilterDiscriminator(config).to(device)

        elif disc_type == "control_consistency":
            if ControlConsistencyDiscriminator is None:
                logger.error("ControlConsistencyDiscriminator not available")
                return None
            return ControlConsistencyDiscriminator(config).to(device)

        else:
            logger.error(f"未知的判别器类型: {disc_type}")
            return None

    @staticmethod
    def create_discriminators(config: Dict, disc_types: List[str], device: torch.device) -> Dict[str, nn.Module]:
        """
        创建多个判别器

        参数:
            config: 判别器配置
            disc_types: 判别器类型列表
            device: 设备

        返回:
            判别器字典 {disc_type: discriminator_instance}
        """
        discriminators = {}

        for disc_type in disc_types:
            # 获取特定判别器的配置
            disc_config = config.get(disc_type, {})

            # 创建判别器
            discriminator = DiscriminatorFactory.create_discriminator(disc_type, disc_config, device)

            if discriminator is not None:
                discriminators[disc_type] = discriminator
                logger.info(f"成功创建判别器: {disc_type}")
            else:
                logger.warning(f"创建判别器失败: {disc_type}")

        return discriminators

    @staticmethod
    def get_available_discriminators() -> List[str]:
        """
        获取可用的判别器类型

        返回:
            可用判别器类型列表
        """
        available = []
        if ControlAwareMPD is not None:
            available.append("mpd")
        if ControlAwareMSD is not None:
            available.append("msd")
        if SourceFilterDiscriminator is not None:
            available.append("source_filter")
        if ControlConsistencyDiscriminator is not None:
            available.append("control_consistency")
        return available