"""
对抗训练管理器：
负责初始化、管理、训练和评估整个对抗训练系统
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import sys
from datetime import datetime

# Try to import yaml, but handle gracefully if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # We'll handle this in the _load_configs method

# Try to import torch.distributed for distributed training
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# These imports will work once the respective modules are created
try:
    from .discriminators import (
        ControlAwareMPD,
        ControlAwareMSD,
        SourceFilterDiscriminator,
        ControlConsistencyDiscriminator,
        DiscriminatorFactory
    )
    from .losses import LossManager
    from .trainers.progressive_trainer import ProgressiveAdversarialTrainer
except ImportError:
    # Fallback for when modules are not yet created
    ControlAwareMPD = ControlAwareMSD = SourceFilterDiscriminator = ControlConsistencyDiscriminator = None
    DiscriminatorFactory = LossManager = ProgressiveAdversarialTrainer = None

try:
    from ..utils.config import load_config, merge_configs
except ImportError:
    # Fallback implementations
    def load_config(path):
        if not YAML_AVAILABLE:
            raise ImportError("YAML is not available. Please install PyYAML to use configuration files.")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def merge_configs(base, override):
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        return result


class AdversarialManager:
    """
    对抗训练管理器：
    负责初始化、管理、训练和评估整个对抗训练系统
    """

    def __init__(self, config_path: str, experiment_dir: Optional[str] = None):
        """
        初始化对抗训练管理器

        参数:
            config_path: 配置文件路径
            experiment_dir: 实验目录（如果不指定，则自动创建）
        """
        # 加载配置
        self.config = self._load_configs(config_path)
        # 初始化分布式训练环境
        self._init_distributed()

        # 设置实验目录
        if experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = f"experiments/{self.config['experiment']['name']}_{timestamp}"

        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 设置设备
        if self.config.get("distributed", {}).get("enabled", False):
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # 设置随机种子
        self._set_seed()

        # 初始化组件
        self.model = None
        self.discriminators = {}
        self.trainer = None
        self.data_loaders = None

        # 日志记录器
        self.writer = None
        self.logger = logging.getLogger(__name__)

        # 实验跟踪
        self.use_wandb = self.config["logging"].get("use_wandb", False)
        if self.use_wandb:
            self._init_wandb()

        self.use_tensorboard = self.config["logging"].get("use_tensorboard", True)
        if self.use_tensorboard:
            self._init_tensorboard()

        self.logger.info(f"对抗训练管理器初始化完成，设备: {self.device}")

    def _init_distributed(self):
        """初始化分布式训练环境"""
        if not DISTRIBUTED_AVAILABLE:
            self.logger.warning("torch.distributed not available, skipping distributed training initialization")
            return
        # Check if distributed training is enabled in config
        dist_config = self.config.get("distributed", {})
        if not dist_config.get("enabled", False):
            return
        # Initialize the process group
        dist.init_process_group(
            backend=dist_config.get("backend", "nccl"),
            init_method=dist_config.get("init_method", "env://")
        )
        # Set the device to local rank
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # Set the device
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.logger.info(f"分布式训练环境初始化完成，本地排名: {self.local_rank}, 世界大小: {self.world_size}")

    def _load_configs(self, config_path: str) -> Dict:
        """加载配置文件"""
        if not YAML_AVAILABLE:
            raise ImportError("YAML is not available. Please install PyYAML to use configuration files.")

        base_config = load_config(config_path)

        # 加载判别器配置
        disc_config_path = Path(config_path).parent / "discriminators.yaml"
        if disc_config_path.exists():
            disc_config = load_config(disc_config_path)
            base_config = merge_configs(base_config, disc_config)

        # 加载损失配置
        loss_config_path = Path(config_path).parent / "losses.yaml"
        if loss_config_path.exists():
            loss_config = load_config(loss_config_path)
            base_config = merge_configs(base_config, loss_config)

        # 加载训练配置
        train_config_path = Path(config_path).parent / "training.yaml"
        if train_config_path.exists():
            train_config = load_config(train_config_path)
            base_config = merge_configs(base_config, train_config)

        return base_config

    def _setup_logging(self):
        """设置日志系统"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # 配置文件日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )

        # 保存配置
        config_save_path = self.experiment_dir / "config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _set_seed(self):
        """设置随机种子"""
        seed = self.config["experiment"].get("seed", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_wandb(self):
        """初始化Weights & Biases"""
        try:
            import wandb
            wandb.init(
                project=self.config["logging"]["wandb_project"],
                name=self.config["experiment"]["name"],
                config=self.config,
                dir=self.experiment_dir
            )
            wandb.define_metric("epoch")
            wandb.define_metric("step")
        except ImportError:
            self.logger.warning("Weights & Biases not installed, skipping wandb initialization")

    def _init_tensorboard(self):
        """初始化TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.experiment_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            self.logger.warning("TensorBoard not installed, skipping tensorboard initialization")

    def setup_training(self, model, train_loader, val_loader=None, test_loader=None):
        """
        设置训练环境

        参数:
            model: 基础生成器模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        self.model = model.to(self.device)
        # 如果启用了分布式训练，则用DDP包装模型
        if self.config.get("distributed", {}).get("enabled", False):
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )
        self.data_loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

        # 初始化判别器
        self._init_discriminators()

        # 初始化训练器
        self._init_trainer()

        self.logger.info("训练环境设置完成")
        self.logger.info(f"训练数据: {len(train_loader)} 批次")
        if val_loader:
            self.logger.info(f"验证数据: {len(val_loader)} 批次")
        if test_loader:
            self.logger.info(f"测试数据: {len(test_loader)} 批次")

    def _init_discriminators(self):
        """初始化判别器"""
        if DiscriminatorFactory is None:
            self.logger.warning("DiscriminatorFactory not available, skipping discriminator initialization")
            return

        disc_config = self.config.get("discriminators", {})
        enabled_types = disc_config.get("types", [])
        disc_weights = disc_config.get("weights", {})

        for disc_type in enabled_types:
            if disc_type not in disc_weights:
                self.logger.warning(f"未找到判别器 {disc_type} 的权重，使用默认值 1.0")
                disc_weights[disc_type] = 1.0

        # 通过工厂创建判别器
        self.discriminators = DiscriminatorFactory.create_discriminators(
            disc_config, enabled_types, device=self.device
        )
        # 如果启用了分布式训练，则用DDP包装判别器
        if self.config.get("distributed", {}).get("enabled", False):
            for name, disc in self.discriminators.items():
                if disc is not None:
                    disc = disc.to(self.device)
                    disc = torch.nn.parallel.DistributedDataParallel(
                        disc, device_ids=[self.local_rank], output_device=self.local_rank
                    )
                    self.discriminators[name] = disc

        # 打印判别器信息
        total_params = 0
        for name, disc in self.discriminators.items():
            if disc is not None:
                params = sum(p.numel() for p in disc.parameters())
                total_params += params
                self.logger.info(f"判别器 {name}: {params:,} 参数, 权重: {disc_weights.get(name, 1.0)}")

        self.logger.info(f"总判别器参数: {total_params:,}")

    def _init_trainer(self):
        """初始化训练器"""
        if ProgressiveAdversarialTrainer is None:
            self.logger.warning("ProgressiveAdversarialTrainer not available, skipping trainer initialization")
            return

        trainer_config = self.config.get("training", {})

        self.trainer = ProgressiveAdversarialTrainer(
            generator=self.model,
            discriminators=self.discriminators,
            config=self.config,
            device=self.device,
            experiment_dir=self.experiment_dir
        )

        # 设置优化器
        self._init_optimizers()

        # 设置学习率调度器
        self._init_schedulers()

        self.logger.info("训练器初始化完成")

    def _init_optimizers(self):
        """初始化优化器"""
        if self.trainer is None:
            return

        opt_config = self.config.get("optimization", {})

        # 生成器优化器
        g_params = self.model.parameters()
        self.trainer.init_optimizer_g(g_params, opt_config.get("generator", {}))

        # 判别器优化器
        d_params = []
        for disc in self.discriminators.values():
            if disc is not None:
                d_params.extend(disc.parameters())
        self.trainer.init_optimizer_d(d_params, opt_config.get("discriminator", {}))

        self.logger.info("优化器初始化完成")

    def _init_schedulers(self):
        """初始化学习率调度器"""
        if self.trainer is None:
            return

        scheduler_config = self.config.get("schedulers", {})

        if "generator" in scheduler_config:
            self.trainer.init_scheduler_g(scheduler_config["generator"])

        if "discriminators" in scheduler_config:
            self.trainer.init_scheduler_d(scheduler_config["discriminators"])

        self.logger.info("学习率调度器初始化完成")

    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        开始训练

        参数:
            num_epochs: 训练轮数
            start_epoch: 开始轮数（用于恢复训练）
        """
        if self.trainer is None:
            self.logger.error("Trainer not initialized, cannot start training")
            return

        # 检查超参数搜索是否启用
        if self.enabled:
            self.logger.info(f"超参数搜索已启用，开始超参数搜索")
            # 运行超参数搜索以获取最佳配置
            best_trial = self.run_search()
            if best_trial is not None:
                # 获取最佳配置
                best_config = self.get_best_config()
                if best_config is not None:
                    # 用最佳配置更新基本配置
                    self.base_config = best_config
                    self.config = best_config  # 同时更新self.config以确保trainer使用最新配置
                    self.logger.info(f"使用超参数搜索找到的最佳配置更新基本配置")
                else:
                    self.logger.warning("超参数搜索未返回有效的最佳试验")
            else:
                self.logger.warning("超参数搜索未找到有效的最佳试验")
        # 如果启用了超参数搜索且找到了最佳配置，则重新初始化训练器以使用更新后的配置
        if self.enabled and 'best_config' in locals() and best_config is not None:
            self.logger.info("重新初始化训练器以使用超参数搜索后的配置")
            self._init_trainer()
        self.logger.info(f"开始对抗训练，总轮数: {num_epochs}")

        try:
            for epoch in range(start_epoch, num_epochs):
                # 训练一个epoch
                train_metrics = self.trainer.train_epoch(
                    self.data_loaders["train"],
                    epoch,
                    self.data_loaders.get("val")
                )

                # 记录训练指标
                self._log_metrics(train_metrics, epoch, "train")

                # 验证
                if self.data_loaders.get("val") and epoch % self.config["logging"].get("eval_interval", 5) == 0:
                    val_metrics = self.evaluate("val")
                    self._log_metrics(val_metrics, epoch, "val")

                    # 保存最佳模型
                    if val_metrics.get("total_loss", float('inf')) < self.trainer.best_val_loss:
                        self.trainer.best_val_loss = val_metrics["total_loss"]
                        self.save_checkpoint(f"best_model_epoch_{epoch}.pt", epoch)

                # 保存检查点
                if epoch % self.config["logging"].get("save_checkpoint_interval", 10) == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch)

                # 更新学习率
                self.trainer.update_schedulers()

            # 训练完成
            self.logger.info("训练完成！")

            # 最终评估
            if self.data_loaders.get("test"):
                test_metrics = self.evaluate("test")
                self._log_metrics(test_metrics, num_epochs, "test")
                self.logger.info(f"测试集结果: {test_metrics}")

        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练出错: {e}", exc_info=True)
            raise

    def evaluate(self, split: str = "val") -> Dict[str, float]:
        """
        评估模型

        参数:
            split: 评估数据集（val/test）

        返回:
            评估指标字典
        """
        if split not in self.data_loaders or self.data_loaders[split] is None:
            self.logger.warning(f"数据集 {split} 不存在，跳过评估")
            return {}

        self.logger.info(f"在 {split} 数据集上评估模型")

        # 设置模型为评估模式
        self.model.eval()
        for disc in self.discriminators.values():
            if disc is not None:
                disc.eval()

        # 评估
        metrics = self.trainer.evaluate(
            self.data_loaders[split],
            split=split
        )

        # 设置回训练模式
        self.model.train()
        for disc in self.discriminators.values():
            if disc is not None:
                disc.train()

        return metrics

    def _log_metrics(self, metrics: Dict[str, float], epoch: int, prefix: str = ""):
        """记录指标"""
        if prefix:
            prefix = f"{prefix}/"

        # 记录到TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}{key}", value, epoch)

        # 记录到WandB
        if self.use_wandb:
            try:
                import wandb
                wandb_metrics = {f"{prefix}{key}": value for key, value in metrics.items()
                                if isinstance(value, (int, float))}
                wandb_metrics["epoch"] = epoch
                wandb_metrics["step"] = epoch * len(self.data_loaders["train"])
                wandb.log(wandb_metrics)
            except ImportError:
                pass

        # 记录到日志
        self.logger.info(f"Epoch {epoch} - {prefix}指标: {metrics}")

    def save_checkpoint(self, filename: str, epoch: int):
        """保存检查点"""
        checkpoint_path = self.experiment_dir / "checkpoints" / filename
        checkpoint_path.parent.mkdir(exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "discriminators_state_dict": {
                name: disc.state_dict() for name, disc in self.discriminators.items() if disc is not None
            },
            "trainer_state": self.trainer.get_state() if self.trainer else {},
            "config": self.config
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 加载判别器
        for name, disc in self.discriminators.items():
            if disc is not None and name in checkpoint["discriminators_state_dict"]:
                disc.load_state_dict(checkpoint["discriminators_state_dict"][name])

        # 加载训练器状态
        if "trainer_state" in checkpoint and self.trainer is not None:
            self.trainer.load_state(checkpoint["trainer_state"])

        self.logger.info(f"检查点已加载: {checkpoint_path}")

    def generate_examples(self, num_examples: int = 3, save_dir: Optional[str] = None):
        """
        生成示例音频

        参数:
            num_examples: 生成示例数量
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = self.experiment_dir / "examples"
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(exist_ok=True)

        self.logger.info(f"生成 {num_examples} 个示例音频")

        # 设置模型为评估模式
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(self.data_loaders["val"]):
                if i >= num_examples:
                    break

                # 准备数据
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 生成音频
                output = self.model.forward(batch)

                # 保存音频
                try:
                    import soundfile as sf
                    for j in range(min(output["audio"].shape[0], num_examples)):
                        audio_real = batch["audio"][j].cpu().numpy()
                        audio_gen = output["audio"][j].cpu().numpy()

                        # 保存真实和生成音频
                        sf.write(save_dir / f"example_{i}_{j}_real.wav", audio_real, 24000)
                        sf.write(save_dir / f"example_{i}_{j}_gen.wav", audio_gen, 24000)
                except ImportError:
                    self.logger.warning("soundfile not installed, skipping audio file generation")

        # 设置回训练模式
        self.model.train()

        self.logger.info(f"示例音频已保存到: {save_dir}")

    def close(self):
        """关闭资源"""
        if self.writer:
            self.writer.close()

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass

        self.logger.info("对抗训练管理器已关闭")