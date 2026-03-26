#!/usr/bin/env python3
"""
对抗模型评估脚本
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
from evaluation.metrics.audio_metrics import AudioQualityMetrics


def main():
    parser = argparse.ArgumentParser(description="CantioAI 对抗模型评估")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/adversarial/base.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="评估数据集"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="结果保存目录"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 创建对抗训练管理器
    manager = AdversarialManager(args.config)

    # 创建模型
    model = HybridVocoderSystem(config.get("model", {}))

    # 创建数据加载器
    _, val_loader, test_loader = create_dataloaders(config.get("data", {}))

    # 设置评估环境
    data_loaders = {
        "val": val_loader,
        "test": test_loader
    }
    manager.model = model.to(manager.device)
    manager.data_loaders = data_loaders

    # 加载检查点
    print(f"从检查点加载模型: {args.checkpoint}")
    manager.load_checkpoint(args.checkpoint)

    # 运行评估
    print(f"在 {args.split} 数据集上进行评估...")
    metrics = manager.evaluate(args.split)

    # 打印结果
    print("\n评估结果:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # 保存结果
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(save_dir / f"evaluation_{args.split}.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n结果已保存到: {save_dir}")

    # 关闭资源
    manager.close()


if __name__ == "__main__":
    main()