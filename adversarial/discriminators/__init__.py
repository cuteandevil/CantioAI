"""
判别器模块
"""

from .discriminator_factory import DiscriminatorFactory

# 尝试导入判别器实现
try:
    from ..control_aware_mpd import ControlAwareMPD
    from ..control_aware_msd import ControlAwareMSD
    from ..discriminators.specialized.source_filter_discriminator import SourceFilterDiscriminator
    from ..discriminators.specialized.control_consistency_disc import ControlConsistencyDiscriminator
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from cantioai.control_aware_mpd import ControlAwareMPD
        from cantioai.control_aware_msd import ControlAwareMSD
        from cantioai.discriminators.specialized.source_filter_discriminator import SourceFilterDiscriminator
        from cantioai.discriminators.specialized.control_consistency_disc import ControlConsistencyDiscriminator
    except ImportError:
        ControlAwareMPD = ControlAwareMSD = SourceFilterDiscriminator = ControlConsistencyDiscriminator = None

__all__ = [
    "DiscriminatorFactory",
    "ControlAwareMPD",
    "ControlAwareMSD",
    "SourceFilterDiscriminator",
    "ControlConsistencyDiscriminator"
]