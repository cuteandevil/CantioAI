"""
Training script for CantioAI model.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from cantioai.src.models.hybrid_svc import HybridSVC
from cantioai.src.data.dataset import CantioAIDataset, create_data_loaders
from cantioai.src.training.trainer import CantioAITrainer
from cantioai.src.utils.logging import setup_logging
from cantioai.src.utils.config import load_config

logger = logging.getLogger(__name__)


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train CantioAI hybrid source-filter + neural vocoder model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed features"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (None for stdout only)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )

    logger.info("Starting CantioAI model training")
    logger.info(f"Arguments: {args}")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Setup device
        device_str = config.get("training", {}).get("device", "auto")
        if device_str == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_str
        logger.info(f"Using device: {device}")

        # Setup data loaders
        data_config = config.get("data", {})
        dataset_config = {
            "phoneme_feature_dim": config.get("model", {}).get("phoneme_feature_dim", 32),
            "spectral_envelope_dim": config.get("model", {}).get("spectral_envelope_dim", 60),
            "augment": config.get("training", {}).get("augment", False)
        }

        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=config.get("training", {}).get("batch_size", 16),
            num_workers=config.get("training", {}).get("num_workers", 4),
            pin_memory=config.get("training", {}).get("pin_memory", True),
            **dataset_config
        )
        logger.info(
            f"Data loaders created: train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
        )

        # Initialize model
        model_config = config.get("model", {})
        model = HybridSVC(
            phoneme_feature_dim=model_config.get("phoneme_feature_dim", 32),
            spectral_envelope_dim=model_config.get("spectral_envelope_dim", 60),
            speaker_embed_dim=model_config.get("speaker_embed_dim", 128),
            n_speakers=model_config.get("n_speakers", 100),
            use_pitch_quantizer=model_config.get("use_pitch_quantizer", True),
            pitch_quantizer_config=model_config.get("pitch_quantizer", {}),
            **{
                k: v for k, v in model_config.items()
                if k not in ["phoneme_feature_dim", "spectral_envelope_dim",
                           "speaker_embed_dim", "n_speakers", "use_pitch_quantizer",
                           "pitch_quantizer"]
            }
        )
        logger.info(f"Model initialized: {model.__class__.__name__}")

        # Setup loggers
        wandb_logger = None
        tensorboard_writer = None

        if not args.no_wandb:
            try:
                import wandb
                wandb_config = config.get("experiment", {}).get("wandb", {})
                if wandb_config.get("enabled", False):
                    wandb.init(
                        project=wandb_config.get("project", "cantioai"),
                        name=config.get("experiment", {}).get("name", "cantioai_base"),
                        config=config,
                        **wandb_config.get("init_args", {})
                    )
                    wandb_logger = wandb
                    logger.info("Weights & Biases logging enabled")
            except ImportError:
                logger.warning("Weights & Biases not installed, skipping")

        if not args.no_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(
                    config.get("experiment", {}).get("tensorboard_dir", "logs/tensorboard")
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
                logger.info(f"TensorBoard logging enabled: {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not installed, skipping")

        # Initialize trainer
        trainer = CantioAITrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            wandb_logger=wandb_logger,
            tensorboard_writer=tensorboard_writer
        )

        # Resume from checkpoint if specified
        start_epoch = 1
        if args.resume_from:
            logger.info(f"Resuming training from checkpoint: {args.resume_from}")
            start_epoch = trainer.load_checkpoint(args.resume_from, load_optimizer=True)
            start_epoch += 1  # Start from next epoch
            logger.info(f"Resuming from epoch {start_epoch}")

        # Train
        trainer.train()

        # Close loggers
        if wandb_logger is not None:
            try:
                wandb.finish()
            except:
                pass
        if tensorboard_writer is not None:
            try:
                tensorboard_writer.close()
            except:
                pass

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()