"""
Training loop for CantioAI model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any, Callable
import logging
import time
import os
from pathlib import Path

from ..training.losses import compute_total_loss
from ..models.hybrid_svc import HybridSVC

logger = logging.getLogger(__name__)


class CantioAITrainer:
    """
    Trainer for HybridSVC model.
    """

    def __init__(
        self,
        model: HybridSVC,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[str] = None,
        wandb_logger: Optional[Any] = None,
        tensorboard_writer: Optional[Any] = None
    ):
        """
        Initialize trainer.

        Args:
            model: HybridSVC model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary (from config.yaml)
            device: Device to use ("cpu", "cuda", etc.)
            wandb_logger: Weights & Biases logger (optional)
            tensorboard_writer: TensorBoard writer (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.wandb_logger = wandb_logger
        self.tensorboard_writer = tensorboard_writer

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Training configuration
        train_config = config.get("training", {})
        self.learning_rate = train_config.get("learning_rate", 0.001)
        self.weight_decay = train_config.get("weight_decay", 1e-5)
        self.epochs = train_config.get("epochs", 100)
        self.grad_clip = train_config.get("grad_clip", 1.0)
        self.use_amp = train_config.get("use_amp", False)
        self.validation_interval = train_config.get("validation_interval", 1)
        self.save_interval = train_config.get("save_interval", 10)

        # Optimizer
        optimizer_name = train_config.get("optimizer", "adam").lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=tuple(train_config.get("betas", [0.9, 0.999]))
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=tuple(train_config.get("betas", [0.9, 0.999]))
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=train_config.get("momentum", 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Learning rate scheduler
        lr_scheduler_name = train_config.get("lr_scheduler", "step").lower()
        if lr_scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get("lr_step_size", 30),
                gamma=train_config.get("lr_gamma", 0.1)
            )
        elif lr_scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.get("epochs", 100),
                eta_min=train_config.get("lr_min", 1e-6)
            )
        elif lr_scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=train_config.get("lr_factor", 0.5),
                patience=train_config.get("lr_patience", 10),
                verbose=False
            )
        elif lr_scheduler_name == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported LR scheduler: {lr_scheduler_name}")

        # Mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Checkpoint directory
        self.checkpoint_dir = Path(
            config.get("experiment", {}).get("checkpoint_dir", "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_interval = config.get("experiment", {}).get("log_interval", 100)
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info(
            f"Initialized CantioAITrainer:\n"
            f"  Device: {self.device}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Learning rate: {self.learning_rate}\n"
            f"  Weight decay: {self.weight_decay}\n"
            f"  Gradient clip: {self.grad_clip}\n"
            f"  Use AMP: {self.use_amp}\n"
            f"  Optimizer: {optimizer_name}\n"
            f"  LR Scheduler: {lr_scheduler_name}"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {}
        epoch_loss_count = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                outputs = self._forward_pass(batch)
                loss, loss_dict = compute_total_loss(
                    outputs,
                    batch,
                    **self.config.get("loss", {})
                )

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Update losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            epoch_loss_count += 1

            # Logging
            self.global_step += 1
            if batch_idx % self.log_interval == 0:
                self._log_metrics(loss_dict, self.global_step, "train")

            # Step LR scheduler (if per-iteration)
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau is updated per epoch with validation loss
                pass
            elif self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

        # Epoch timing
        epoch_time = time.time() - start_time
        avg_losses = {k: v / epoch_loss_count for k, v in epoch_losses.items()}

        logger.info(
            f"Epoch {epoch}/{self.epochs} completed in {epoch_time:.2f}s - "
            f"Loss: {avg_losses.get('total_loss', 0.0):.6f}"
        )

        # Log epoch metrics
        self._log_metrics(avg_losses, epoch, "train_epoch")

        # Step LR scheduler for epoch-based schedulers
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # Will be updated with validation loss
            pass
        elif self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        return avg_losses

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.eval()
        epoch_losses = {}
        epoch_loss_count = 0

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Forward pass
                with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                    outputs = self._forward_pass(batch)
                    loss, loss_dict = compute_total_loss(
                        outputs,
                        batch,
                        **self.config.get("loss", {})
                    )

                # Update losses
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                epoch_loss_count += 1

        # Epoch timing
        epoch_time = time.time() - start_time
        avg_losses = {k: v / epoch_loss_count for k, v in epoch_losses.items()}

        logger.info(
            f"Validation Epoch {epoch} completed in {epoch_time:.2f}s - "
            f"Loss: {avg_losses.get('total_loss', 0.0):.6f}"
        )

        # Log epoch metrics
        self._log_metrics(avg_losses, epoch, "val_epoch")

        return avg_losses

    def train(self) -> None:
        """
        Main training loop.
        """
        logger.info("Starting training...")
        start_time = time.time()

        try:
            for epoch in range(1, self.epochs + 1):
                # Train
                train_losses = self.train_epoch(epoch)

                # Validate
                if epoch % self.validation_interval == 0:
                    val_losses = self.validate_epoch(epoch)

                    # Update LR scheduler for ReduceLROnPlateau
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        val_loss = val_losses.get("total_loss", 0.0)
                        self.scheduler.step(val_loss)

                    # Save best model
                    val_loss = val_losses.get("total_loss", 0.0)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(
                            epoch,
                            is_best=True,
                            extra_info={"val_loss": val_loss}
                        )

                # Save regular checkpoint
                if epoch % self.save_interval == 0:
                    self.save_checkpoint(epoch, is_best=False)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Final timing
            total_time = time.time() - start_time
            logger.info(
                f"Training completed in {total_time/3600:.2f} hours. "
                f"Best validation loss: {self.best_val_loss:.6f}"
            )

            # Save final model
            self.save_checkpoint(self.epochs, is_best=False, extra_info={"final": True})

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary of model outputs
        """
        phoneme_features = batch["phoneme_features"]
        f0 = batch["f0"]
        spk_id = batch["spk_id"]

        # Determine if F0 is in Hz (assume yes for training data)
        f0_is_hz = True  # Training data should have F0 in Hz

        sp_pred, f0_quant, extras = self.model(
            phoneme_features=phoneme_features,
            f0=f0,
            spk_id=spk_id,
            f0_is_hz=f0_is_hz,
            return_quantized_f0=self.config.get("model", {}).get("use_pitch_quantizer", True)
        )

        outputs = {
            "sp_pred": sp_pred,
        }
        if f0_quant is not None:
            outputs["f0_pred"] = f0_quant  # For loss computation
            outputs["f0_quant"] = f0_quant
        outputs.update(extras)

        return outputs

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move batch dictionary to device.

        Args:
            batch: Input batch dictionary

        Returns:
            Batch dictionary on device
        """
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to configured loggers.

        Args:
            metrics: Dictionary of metric values
            step: Current step/epoch
            prefix: Prefix for metric names (e.g., "train", "val")
        """
        # Format prefix
        if prefix:
            prefix = f"{prefix}/"

        # Log to console (handled by logger in calling functions)

        # Log to TensorBoard
        if self.tensorboard_writer is not None:
            try:
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(
                        f"{prefix}{key}", value, step
                    )
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")

        # Log to Weights & Biases
        if self.wandb_logger is not None:
            try:
                log_dict = {f"{prefix}{key}": value for key, value in metrics.items()}
                log_dict["step"] = step
                self.wandb_logger.log(log_dict)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            extra_info: Additional information to save in checkpoint
        """
        if extra_info is None:
            extra_info = {}

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "extra_info": extra_info,
        }

        # Add scaler state if using AMP
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Determine filename
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        logger.info(f"Saved checkpoint: {filepath}")

        # Also save as latest for easy resuming
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True
    ) -> int:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer and scheduler state

        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load extra info
        epoch = checkpoint.get("epoch", 0)
        self.config = checkpoint.get("config", self.config)
        extra_info = checkpoint.get("extra_info", {})

        logger.info(
            f"Loaded checkpoint from epoch {epoch}. "
            f"Extra info: {extra_info}"
        )

        return epoch