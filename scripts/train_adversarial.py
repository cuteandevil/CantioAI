#!/usr/bin/env python3
"""
对抗训练入口脚本
"""

import argparse
import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adversarial.adversarial_manager import AdversarialManager
from utils.config import load_config
from models.hybrid_vocoder import HybridVocoderSystem
from data.dataset import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description="CantioAI 对抗训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/adversarial/base.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="实验目录（可选）"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（覆盖配置文件）"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 创建对抗训练管理器
    manager = AdversarialManager(args.config, args.experiment_dir)

    # 创建模型
    model = HybridVocoderSystem(config.get("model", {}))

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(config.get("data", {}))

    # 设置训练环境
    manager.setup_training(model, train_loader, val_loader, test_loader)

    # 如果指定了恢复训练
    if args.resume:
        manager.load_checkpoint(args.resume)
        start_epoch = manager.trainer.current_epoch if hasattr(manager.trainer, 'current_epoch') else 0
        print(f"从检查点恢复训练，开始轮数: {start_epoch}")
    else:
        start_epoch = 0

    # 设置训练轮数
    num_epochs = args.epochs if args.epochs is not None else config.get("training", {}).get("epochs", 100)

    # 开始训练
    try:
        manager.train(num_epochs, start_epoch)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    finally:
        # 生成一些示例
        manager.generate_examples(num_examples=3)
        # 关闭资源
        manager.close()


if __name__ == "__main__":
    main()