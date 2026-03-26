#!/usr/bin/env python3
"""
模型对比脚本
"""

import argparse
import torch
import sys
import os
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adversarial.adversarial_manager import AdversarialManager
from utils.config import load_config
from models.hybrid_vocoder import HybridVocoderSystem
from data.dataset import create_dataloaders
from evaluation.metrics.audio_metrics import AudioQualityMetrics


def load_model_from_checkpoint(checkpoint_path, config, device):
    """从检查点加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HybridVocoderSystem(config.get("model", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def evaluate_model(model, data_loader, device, split_name="test"):
    """评估单个模型"""
    model.eval()
    metrics_calculator = AudioQualityMetrics(sample_rate=24000, device=device)

    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            # 准备数据
            batch = {k: v.to(device) for k, v in batch.items()}

            # 生成音频
            output = model.forward(batch)

            # 计算指标
            real_audio = batch["audio"]
            gen_audio = output["audio"]

            batch_metrics = metrics_calculator.compute_all_metrics(
                real_audio, gen_audio, compute_pesq_stoi=False
            )

            # 累积指标
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value

            num_batches += 1

    # 计算平均值
    avg_metrics = {}
    for key, value in total_metrics.items():
        avg_metrics[key] = value / num_batches if num_batches > 0 else 0.0

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="CantioAI 模型对比")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/adversarial/base.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="模型检查点路径列表"
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="模型名称列表（与checkpoints对应）"
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

    # 设置模型名称
    if args.names is None:
        args.names = [f"Model_{i+1}" for i in range(len(args.checkpoints))]
    elif len(args.names) != len(args.checkpoints):
        raise ValueError("模型名称数量必须与检查点数量匹配")

    # 创建数据加载器
    _, val_loader, test_loader = create_dataloaders(config.get("data", {}))
    data_loader = val_loader if args.split == "val" else test_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 评估所有模型
    results = {}
    for name, checkpoint_path in zip(args.names, args.checkpoints):
        print(f"评估模型: {name}")
        print(f"检查点路径: {checkpoint_path}")

        try:
            model = load_model_from_checkpoint(checkpoint_path, config, device)
            metrics = evaluate_model(model, data_loader, device, args.split)
            results[name] = metrics

            print(f"  评估完成!")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.6f}")
        except Exception as e:
            print(f"  评估失败: {e}")
            results[name] = {"error": str(e)}

    # 打印对比结果
    print("\n" + "=" * 80)
    print("模型对比结果")
    print("=" * 80)

    # 获取所有指标键
    all_keys = set()
    for metrics in results.values():
        if isinstance(metrics, dict) and "error" not in metrics:
            all_keys.update(metrics.keys())

    # 打印表头
    print(f"{'模型名称':<20}", end="")
    for key in sorted(all_keys):
        print(f"{key:<15}", end="")
    print()
    print("-" * 80)

    # 打印每个模型的结果
    for name, metrics in results.items():
        print(f"{name:<20}", end="")
        if isinstance(metrics, dict) and "error" not in metrics:
            for key in sorted(all_keys):
                value = metrics.get(key, 0.0)
                if isinstance(value, float):
                    print(f"{value:<15.6f}", end="")
                else:
                    print(f"{str(value):<15}", end="")
        else:
            for _ in all_keys:
                print(f"{'ERROR':<15}", end="")
        print()

    # 保存结果
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存详细结果
        with open(save_dir / "comparison_detailed.json", 'w') as f:
            json.dump(results, f, indent=2)

        # 保存CSV格式的对比结果
        import csv
        with open(save_dir / "comparison.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + sorted(all_keys))
            for name, metrics in results.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    row = [name] + [metrics.get(key, "") for key in sorted(all_keys)]
                else:
                    row = [name] + ["ERROR"] * len(all_keys)
                writer.writerow(row)

        print(f"\n对比结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()