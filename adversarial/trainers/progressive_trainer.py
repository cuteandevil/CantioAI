"""
渐进式对抗训练器：
实现渐进式训练策略的训练器
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from pathlib import Path

# Try to import torch.distributed for distributed training
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Import required modules
try:
    from ..adversarial_manager import AdversarialManager
    from ..discriminators.discriminator_factory import DiscriminatorFactory
    from ..losses.loss_manager import LossManager
except ImportError:
    # Fallback for when modules are not in expected location
    AdversarialManager = DiscriminatorFactory = LossManager = None

logger = logging.getLogger(__name__)


class ProgressiveAdversarialTrainer:
    """
    渐进式对抗训练器：
    实现渐进式训练策略的训练器
    """

    def __init__(self, generator: nn.Module, discriminators: Dict[str, nn.Module],
                 config: Dict, device: torch.device, experiment_dir: Optional[str] = None):
        """
        初始化渐进式对抗训练器

        参数:
            generator: 生成器模型
            discriminators: 判别器字典 {disc_type: discriminator_instance}
            config: 训练配置
            device: 设备
            experiment_dir: 实验目录（如果不指定，则自动创建）
        """
        self.generator = generator.to(device)
        self.discriminators = {name: disc.to(device) for name, disc in discriminators.items()}
        self.config = config
        self.device = device
        # 分布式训练标志
        self.distributed = self.config.get("distributed", {}).get("enabled", False)
        if self.distributed:
            # We assume that the process group has been initialized by the AdversarialManager
            self.world_size = dist.get_world_size() if DISTRIBUTED_AVAILABLE else 1
        else:
            self.world_size = 1

        # 设置实验目录
        if experiment_dir is None:
            import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = f"experiments/{config['experiment']['name']}_{timestamp}"

        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 初始化训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # 设置优化器
        self._init_optimizers()

        # 设置学习率调度器
        self._init_schedulers()

        self.logger.info(f"渐进式对抗训练器初始化完成，设备: {self.device}")

    def _reduce_losses(self, loss_dict):
        """在分布式训练中减少损失字典"""
        if not self.distributed:
            return loss_dict
        reduced_dict = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                # We assume it's a scalar tensor
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value = value / self.world_size
                reduced_dict[key] = value
            else:
                reduced_dict[key] = value
        return reduced_dict

    def _init_optimizers(self):
        """初始化优化器"""
        opt_config = self.config.get("optimization", {})

        # 生成器优化器
        g_params = self.generator.parameters()
        self.optimizer_g = torch.optim.Adam(g_params, lr=self.config.get("training", {}).get("learning_rate", {}).get("generator", 0.0001))

        # 判别器优化器
        d_params = []
        for disc in self.discriminators.values():
            d_params.extend(disc.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=self.config.get("training", {}).get("learning_rate", {}).get("discriminator", 0.0001))

        self.logger.info("优化器初始化完成")

    def _init_schedulers(self):
        """初始化学习率调度器"""
        scheduler_config = self.config.get("schedulers", {})

        if "generator" in scheduler_config:
            self.scheduler_g = torch.optim.lr_scheduler.StepLR(
                self.optimizer_g,
                step_size=scheduler_config.get("generator", {}).get("step_size", 50),
                gamma=scheduler_config.get("generator", {}).get("gamma", 0.5)
            )

        if "discriminators" in scheduler_config:
            self.scheduler_d = torch.optim.lr_scheduler.StepLR(
                self.optimizer_d,
                step_size=scheduler_config.get("discriminators", {}).get("step_size", 50),
                gamma=scheduler_config.get("discriminators", {}).get("gamma", 0.5)
            )

        self.logger.info("学习率调度器初始化完成")

    def train_epoch(self, train_loader, epoch: int, val_loader=None) -> Dict[str, float]:
        """
        训练一个epoch

        参数:
            train_loader: 训练数据加载器
            epoch: 当前epoch数
            val_loader: 验证数据加载器

        返回:
            训练指标字典
        """
        self.generator.train()
        for disc in self.discriminators.values():
            disc.train()

        epoch_metrics = {
            "generator_loss": 0.0,
            "discriminator_loss": 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for batch in train_loader:
                # 准备数据
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 生成音频
                fake_audio = self.generator(batch)

                # 判别器前向传播
                disc_outputs = {}
                for name, disc in self.discriminators.items():
                    disc_outputs[name] = disc(fake_audio)

                # 计算损失
                total_losses = {
                    "generator": 0.0,
                    "discriminator": 0.0
                }
                loss_details = {}

                for loss_name, loss_fn in self.losses.items():
                    try:
                        if loss_name == "adversarial":
                            # 对抗损失
                            gen_loss, gen_details = loss_fn.generator_loss(list(disc_outputs.values()), "generator")
                            disc_loss, disc_details = loss_fn.discriminator_loss(
                                list(disc_outputs.values()), list(disc_outputs.values()), "discriminator", None
                            )

                            total_losses["generator"] += gen_loss
                            total_losses["discriminator"] += disc_loss

                            loss_details[f"{loss_name}_generator"] = gen_details
                            loss_details[f"{loss_name}_discriminator"] = disc_details
                    except Exception as e:
                        self.logger.warning(f"计算损失 {loss_name} 时出错: {e}")

                # 累积损失
                num_batches += 1
                for key in total_losses.keys():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += total_losses[key]

        # 计算平均损失
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

        # 更新当前epoch
        self.current_epoch = epoch + 1
        # 在分布式训练中减少损失
        epoch_metrics = self._reduce_losses(epoch_metrics)
        return epoch_metrics

    def evaluate(self, val_loader) -> Dict[str, float]:
        """
        评估模型

        参数:
            val_loader: 验证数据加载器

        返回:
            评估指标字典
        """
        if val_loader is None:
            return {}

        self.generator.eval()
        for disc in self.discriminators.values():
            disc.eval()

        metrics = {
            "generator_loss": 0.0,
            "discriminator_loss": 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # 准备数据
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 生成音频
                fake_audio = self.generator(batch)

                # 判别器前向传播
                disc_outputs = {}
                for name, disc in self.discriminators.items():
                    disc_outputs[name] = disc(fake_audio)

                # 计算损失
                total_losses = {
                    "generator": 0.0,
                    "discriminator": 0.0
                }
                loss_details = {}

                for loss_name, loss_fn in self.losses.items():
                    try:
                        if loss_name == "adversarial":
                            # 对抗损失
                            gen_loss, gen_details = loss_fn.generator_loss(list(disc_outputs.values()), "generator")
                            disc_loss, disc_details = loss_fn.discriminator_loss(
                                list(disc_outputs.values()), list(disc_outputs.values()), "discriminator", None
                            )

                            metrics["generator_loss"] += gen_loss
                            metrics["discriminator_loss"] += disc_loss
                    except Exception as e:
                        self.logger.warning(f"计算损失 {loss_name} 时出错: {e}")

                num_batches += 1

        # 计算平均损失
        if num_batches > 0:
            for key in metrics:
                metrics[key] /= num_batches

        # 在分布式训练中减少损失
        metrics = self._reduce_losses(metrics)
        return metrics

    def update_schedulers(self):
        """更新学习率"""
        if hasattr(self, 'scheduler_g'):
            self.scheduler_g.step()
        if hasattr(self, 'scheduler_d'):
            self.scheduler_d.step()

        self.logger.info("学习率已更新")

    def get_state(self) -> Dict[str, Any]:
        """获取训练器状态"""
        return {
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "optimizer_g_state": self.optimizer_g.state_dict(),
            "optimizer_d_state": self.optimizer_d.state_dict(),
            "scheduler_g_state": self.scheduler_g.state_dict() if hasattr(self, 'scheduler_g') else None,
            "scheduler_d_state": self.scheduler_d.state_dict() if hasattr(self, 'scheduler_d') else None
        }

    def load_state(self, state: Dict[str, Any]):
        """加载训练器状态"""
        if "current_epoch" in state:
            self.current_epoch = state["current_epoch"]
        if "best_val_loss" in state:
            self.best_val_loss = state["best_val_loss"]
        if "optimizer_g_state" in state and self.optimizer_g is not None:
            self.optimizer_g.load_state(state["optimizer_g_state"])
        if "optimizer_d_state" in state and self.optimizer_d is not None:
            self.optimizer_d.load_state(state["optimizer_d_state"])
        if "scheduler_g_state" in state and hasattr(self, 'scheduler_g'):
            self.scheduler_g.load_state(state["scheduler_g_state"])
        if "scheduler_d_state" in state and hasattr(self, 'scheduler_d'):
            self.scheduler_d.load_state(state["scheduler_d_state"])