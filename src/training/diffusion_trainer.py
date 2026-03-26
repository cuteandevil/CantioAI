"""
Diffusion model training pipeline for CantioAI.
Implements two-stage training, joint training, and efficient training techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any, Callable, Union
import logging
import time
import os
from pathlib import Path

from ..models.diffusion import ConditionalDiffusionModel, DiffusionPostProcessor
from ..models.hybrid_svc import HybridSVC

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """
    Trainer for Conditional Diffusion Model with support for:
    - Two-stage training: train on real data, then fine-tune on generated data
    - Joint training: train diffusion model together with base model
    - Efficient training: gradient checkpointing, mixed precision, gradient accumulation
    """

    def __init__(
        self,
        diffusion_model: ConditionalDiffusionModel,
        base_model: Optional[HybridSVC] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        wandb_logger: Optional[Any] = None,
        tensorboard_writer: Optional[Any] = None
    ):
        """
        Initialize diffusion trainer.

        Args:
            diffusion_model: ConditionalDiffusionModel to train
            base_model: Optional HybridSVC model for joint training
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary (from config.yaml)
            device: Device to use ("cpu", "cuda", etc.)
            wandb_logger: Weights & Biases logger (optional)
            tensorboard_writer: TensorBoard writer (optional)
        """
        self.diffusion_model = diffusion_model
        self.base_model = base_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.wandb_logger = wandb_logger
        self.tensorboard_writer = tensorboard_writer

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.diffusion_model.to(self.device)
        if self.base_model is not None:
            self.base_model.to(self.device)

        # Training configuration
        diffusion_config = self.config.get("diffusion", {}).get("training", {})
        self.learning_rate = diffusion_config.get("learning_rate", 1e-4)
        self.weight_decay = diffusion_config.get("weight_decay", 0.0)
        self.epochs = diffusion_config.get("epochs", 100)
        self.grad_clip = diffusion_config.get("grad_clip", 1.0)
        self.use_amp = diffusion_config.get("mixed_precision", False)
        self.gradient_accumulation_steps = diffusion_config.get("gradient_accumulation", 1)
        self.use_checkpoint = diffusion_config.get("use_checkpoint", True)

        # Optimization settings
        optimizer_name = diffusion_config.get("optimizer", "adam").lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.diffusion_model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.diffusion_model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Learning rate scheduler
        lr_scheduler_name = diffusion_config.get("lr_scheduler", "step").lower()
        if lr_scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=diffusion_config.get("lr_step_size", 30),
                gamma=diffusion_config.get("lr_gamma", 0.1)
            )
        elif lr_scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=diffusion_config.get("epochs", 100),
                eta_min=diffusion_config.get("lr_min", 1e-6)
            )
        elif lr_scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=diffusion_config.get("lr_factor", 0.5),
                patience=diffusion_config.get("lr_patience", 10),
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

        # Gradient checkpointing
        if self.use_checkpoint:
            self._enable_gradient_checkpointing()

        # Checkpoint directory
        self.checkpoint_dir = Path(
            self.config.get("experiment", {}).get("checkpoint_dir", "checkpoints")
        ) / "diffusion"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_interval = self.config.get("experiment", {}).get("log_interval", 100)
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info(
            f"Initialized DiffusionTrainer:\n"
            f"  Device: {self.device}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Learning rate: {self.learning_rate}\n"
            f"  Weight decay: {self.weight_decay}\n"
            f"  Gradient clip: {self.grad_clip}\n"
            f"  Gradient accumulation steps: {self.gradient_accumulation_steps}\n"
            f"  Use AMP: {self.use_amp}\n"
            f"  Use gradient checkpointing: {self.use_checkpoint}\n"
            f"  Optimizer: {optimizer_name}\n"
            f"  LR Scheduler: {lr_scheduler_name}"
        )

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        def checkpoint_forward(module, *args):
            if hasattr(module, 'forward'):
                return torch.utils.checkpoint.checkpoint(module.forward, *args)
            else:
                return module(*args)

        # Apply checkpointing to residual layers if they support it
        for i, layer in enumerate(self.diffusion_model.residual_layers):
            if hasattr(layer, 'forward'):
                # Wrap the forward method with checkpointing
                original_forward = layer.forward
                layer.forward = lambda x, condition, diffusion_step: torch.utils.checkpoint.checkpoint(
                    original_forward, x, condition, diffusion_step
                )

    def train_two_stage(
        self,
        real_data_loader: DataLoader,
        generated_data_loader: Optional[DataLoader] = None,
        stage1_epochs: Optional[int] = None,
        stage2_epochs: Optional[int] = None
    ) -> None:
        """
        Two-stage training strategy:
        Stage 1: Train on real data with true conditioning
        Stage 2: Fine-tune on generated data with model-predicted conditioning

        Args:
            real_data_loader: DataLoader with real audio and conditioning
            generated_data_loader: DataLoader with generated audio and conditioning
            stage1_epochs: Number of epochs for stage 1 (if None, uses self.epochs // 2)
            stage2_epochs: Number of epochs for stage 2 (if None, uses self.epochs // 2)
        """
        if stage1_epochs is None:
            stage1_epochs = self.epochs // 2
        if stage2_epochs is None:
            stage2_epochs = self.epochs - stage1_epochs

        logger.info(f"Starting two-stage training: {stage1_epochs} epochs (stage 1) + {stage2_epochs} epochs (stage 2)")

        # Stage 1: Train on real data
        logger.info("=== Stage 1: Training on real data ===")
        self._train_stage(
            data_loader=real_data_loader,
            epochs=stage1_epochs,
            stage_name="stage1_real",
            use_generated_conditioning=False
        )

        # Stage 2: Fine-tune on generated data
        if generated_data_loader is not None and stage2_epochs > 0:
            logger.info("=== Stage 2: Fine-tuning on generated data ===")
            self._train_stage(
                data_loader=generated_data_loader,
                epochs=stage2_epochs,
                stage_name="stage2_generated",
                use_generated_conditioning=True
            )
        elif stage2_epochs > 0:
            logger.warning("No generated data loader provided for stage 2, continuing with real data")
            self._train_stage(
                data_loader=real_data_loader,
                epochs=stage2_epochs,
                stage_name="stage2_real",
                use_generated_conditioning=False
            )

    def train_joint(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> None:
        """
        Joint training strategy: train diffusion model together with base model.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs (if None, uses self.epochs)
        """
        if epochs is None:
            epochs = self.epochs

        if self.base_model is None:
            raise ValueError("Base model required for joint training")

        logger.info(f"Starting joint training for {epochs} epochs")

        # Create optimizers for both models
        base_optimizer = optim.Adam(
            self.base_model.parameters(),
            lr=self.config.get("training", {}).get("learning_rate", 0.001),
            weight_decay=self.config.get("training", {}).get("weight_decay", 1e-5)
        )

        # Training loop
        self.base_model.train()
        self.diffusion_model.train()

        for epoch in range(1, epochs + 1):
            epoch_losses = {}
            epoch_loss_count = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Zero gradients
                self.optimizer.zero_grad()
                base_optimizer.zero_grad()

                # Forward pass through base model
                with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                    base_outputs = self._forward_base_pass(batch)

                    # Generate conditioning for diffusion model
                    f0_cond, sp_cond, ap_cond, hubert_cond = self._extract_conditions(
                        base_outputs, batch
                    )

                    # Forward pass through diffusion model
                    diffusion_loss = self.diffusion_model(
                        x=batch["audio"],  # Target clean audio
                        f0_cond=f0_cond,
                        sp_cond=sp_cond,
                        ap_cond=ap_cond,
                        hbert_cond=hubert_cond
                    )

                    # Base model loss (reconstruction loss)
                    base_loss, base_loss_dict = self._compute_base_loss(
                        base_outputs, batch
                    )

                    # Combined loss
                    diffusion_weight = self.config.get("diffusion", {}).get("training", {}).get("loss_weight", 1.0)
                    total_loss = base_loss + diffusion_weight * diffusion_loss

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        self.scaler.unscale_(base_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.diffusion_model.parameters()) + list(self.base_model.parameters()),
                            self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.step(base_optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.diffusion_model.parameters()) + list(self.base_model.parameters()),
                            self.grad_clip
                        )
                    self.optimizer.step()
                    base_optimizer.step()

                # Update losses
                loss_dict = {
                    "diffusion_loss": diffusion_loss.item(),
                    "base_loss": base_loss.item(),
                    "total_loss": total_loss.item()
                }
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                epoch_loss_count += 1

                # Logging
                self.global_step += 1
                if batch_idx % self.log_interval == 0:
                    self._log_metrics(loss_dict, self.global_step, "train_joint")

            # Epoch timing
            epoch_time = time.time() - start_time
            avg_losses = {k: v / epoch_loss_count for k, v in epoch_losses.items()}

            logger.info(
                f"Joint Training Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {avg_losses.get('total_loss', 0.0):.6f}"
            )

            # Log epoch metrics
            self._log_metrics(avg_losses, epoch, "train_joint_epoch")

            # Validation
            if val_loader is not None and epoch % self.config.get("training", {}).get("validation_interval", 1) == 0:
                val_losses = self.validate_joint(val_loader)
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
                        extra_info={"val_loss": val_loss, "stage": "joint"}
                    )

            # Save regular checkpoint
            if epoch % self.config.get("training", {}).get("save_interval", 10) == 0:
                self.save_checkpoint(epoch, is_best=False, extra_info={"stage": "joint"})

            # Step LR scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Will be updated with validation loss
                pass
            elif self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

        # Save final model
        self.save_checkpoint(epochs, is_best=False, extra_info={"final": True, "stage": "joint"})

    def _train_stage(
        self,
        data_loader: DataLoader,
        epochs: int,
        stage_name: str,
        use_generated_conditioning: bool = False
    ) -> None:
        """
        Train for a specific stage.

        Args:
            data_loader: DataLoader for training data
            epochs: Number of epochs to train
            stage_name: Name of the stage for logging
            use_generated_conditioning: Whether to use model-generated conditioning
        """
        self.diffusion_model.train()

        for epoch in range(1, epochs + 1):
            epoch_losses = {}
            epoch_loss_count = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                    # Extract conditioning
                    if use_generated_conditioning and self.base_model is not None:
                        # Use base model to generate conditioning
                        with torch.no_grad():
                            base_outputs = self._forward_base_pass(batch)
                            f0_cond, sp_cond, ap_cond, hubert_cond = self._extract_conditions(
                                base_outputs, batch
                            )
                    else:
                        # Use provided conditioning from batch
                        f0_cond = batch.get("f0")
                        sp_cond = batch.get("sp")
                        ap_cond = batch.get("ap")
                        hubert_cond = batch.get("hubert")

                        # Ensure proper shapes
                        if f0_cond is not None and f0_cond.dim() == 2:
                            f0_cond = f0_cond.unsqueeze(-1)
                        if sp_cond is not None and sp_cond.dim() == 2:
                            sp_cond = sp_cond.unsqueeze(-1)
                        if ap_cond is not None and ap_cond.dim() == 2:
                            ap_cond = ap_cond.unsqueeze(-1)
                        if hubert_cond is not None and hubert_cond.dim() == 2:
                            # Expand to match audio length
                            hubert_cond = hubert_cond.unsqueeze(1).expand(
                                -1, batch["audio"].shape[-1], -1
                            )

                    # Compute diffusion loss
                    loss = self.diffusion_model(
                        x=batch["audio"],
                        f0_cond=f0_cond,
                        sp_cond=sp_cond,
                        ap_cond=ap_cond,
                        hbert_cond=hubert_cond
                    )

                # Backward pass with gradient accumulation
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.diffusion_model.parameters(), self.grad_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.diffusion_model.parameters(), self.grad_clip
                            )
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update losses
                epoch_losses["diffusion_loss"] = epoch_losses.get("diffusion_loss", 0.0) + loss.item()
                epoch_loss_count += 1

                # Logging
                self.global_step += 1
                if batch_idx % self.log_interval == 0:
                    self._log_metrics({"diffusion_loss": loss.item()}, self.global_step, f"train_{stage_name}")

            # Handle remaining gradients if not divisible by accumulation steps
            if (epoch_loss_count % self.gradient_accumulation_steps) != 0:
                if self.use_amp:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.diffusion_model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.diffusion_model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

            # Epoch timing
            epoch_time = time.time() - start_time
            avg_loss = epoch_losses["diffusion_loss"] / max(epoch_loss_count, 1)

            logger.info(
                f"{stage_name.capitalize()} Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {avg_loss:.6f}"
            )

            # Log epoch metrics
            self._log_metrics({"diffusion_loss": avg_loss}, epoch, f"train_{stage_name}_epoch")

            # Validation
            if self.val_loader is not None and epoch % self.config.get("training", {}).get("validation_interval", 1) == 0:
                val_losses = self.validate_stage(self.val_loader, use_generated_conditioning)
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = val_losses.get("diffusion_loss", 0.0)
                    self.scheduler.step(val_loss)

                # Save best model
                val_loss = val_losses.get("diffusion_loss", 0.0)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        epoch,
                        is_best=True,
                        extra_info={"val_loss": val_loss, "stage": stage_name}
                    )

            # Save regular checkpoint
            if epoch % self.config.get("training", {}).get("save_interval", 10) == 0:
                self.save_checkpoint(epoch, is_best=False, extra_info={"stage": stage_name})

            # Step LR scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Will be updated with validation loss
                pass
            elif self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

    def validate_stage(
        self,
        val_loader: DataLoader,
        use_generated_conditioning: bool = False
    ) -> Dict[str, float]:
        """
        Validate diffusion model for a specific stage.

        Args:
            val_loader: DataLoader for validation data
            use_generated_conditioning: Whether to use model-generated conditioning

        Returns:
            Dictionary of average losses for the epoch
        """
        self.diffusion_model.eval()
        epoch_losses = {}
        epoch_loss_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Extract conditioning
                if use_generated_conditioning and self.base_model is not None:
                    # Use base model to generate conditioning
                    base_outputs = self._forward_base_pass(batch)
                    f0_cond, sp_cond, ap_cond, hubert_cond = self._extract_conditions(
                        base_outputs, batch
                    )
                else:
                    # Use provided conditioning from batch
                    f0_cond = batch.get("f0")
                    sp_cond = batch.get("sp")
                    ap_cond = batch.get("ap")
                    hubert_cond = batch.get("hubert")

                    # Ensure proper shapes
                    if f0_cond is not None and f0_cond.dim() == 2:
                        f0_cond = f0_cond.unsqueeze(-1)
                    if sp_cond is not None and sp_cond.dim() == 2:
                        sp_cond = sp_cond.unsqueeze(-1)
                    if ap_cond is not None and ap_cond.dim() == 2:
                        ap_cond = ap_cond.unsqueeze(-1)
                    if hubert_cond is not None and hubert_cond.dim() == 2:
                        # Expand to match audio length
                        hubert_cond = hubert_cond.unsqueeze(1).expand(
                            -1, batch["audio"].shape[-1], -1
                        )

                # Compute diffusion loss
                loss = self.diffusion_model(
                    x=batch["audio"],
                    f0_cond=f0_cond,
                    sp_cond=sp_cond,
                    ap_cond=ap_cond,
                    hbert_cond=hubert_cond
                )

                # Update losses
                epoch_losses["diffusion_loss"] = epoch_losses.get("diffusion_loss", 0.0) + loss.item()
                epoch_loss_count += 1

        avg_losses = {k: v / max(epoch_loss_count, 1) for k, v in epoch_losses.items()}
        return avg_losses

    def validate_joint(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate joint training setup.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary of average losses for the epoch
        """
        self.diffusion_model.eval()
        self.base_model.eval()
        epoch_losses = {}
        epoch_loss_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Forward pass through base model
                base_outputs = self._forward_base_pass(batch)

                # Generate conditioning for diffusion model
                f0_cond, sp_cond, ap_cond, hubert_cond = self._extract_conditions(
                    base_outputs, batch
                )

                # Forward pass through diffusion model
                diffusion_loss = self.diffusion_model(
                    x=batch["audio"],
                    f0_cond=f0_cond,
                    sp_cond=sp_cond,
                    ap_cond=ap_cond,
                    hbert_cond=hubert_cond
                )

                # Base model loss
                base_loss, _ = self._compute_base_loss(base_outputs, batch)

                # Combined loss
                diffusion_weight = self.config.get("diffusion", {}).get("training", {}).get("loss_weight", 1.0)
                total_loss = base_loss + diffusion_weight * diffusion_loss

                # Update losses
                epoch_losses["diffusion_loss"] = epoch_losses.get("diffusion_loss", 0.0) + diffusion_loss.item()
                epoch_losses["base_loss"] = epoch_losses.get("base_loss", 0.0) + base_loss.item()
                epoch_losses["total_loss"] = epoch_losses.get("total_loss", 0.0) + total_loss.item()
                epoch_loss_count += 1

        avg_losses = {k: v / max(epoch_loss_count, 1) for k, v in epoch_losses.items()}
        return avg_losses

    def _forward_base_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the base model.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary of model outputs
        """
        phoneme_features = batch["phoneme_features"]
        f0 = batch["f0"]
        spk_id = batch["spk_id"]

        # Determine if F0 is in Hz (assume yes for training data)
        f0_is_hz = True

        sp_pred, f0_quant, extras = self.base_model(
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
            outputs["f0_pred"] = f0_quant
            outputs["f0_quant"] = f0_quant
        outputs.update(extras)

        return outputs

    def _extract_conditions(
        self,
        base_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract conditioning information from base model outputs and batch.

        Args:
            base_outputs: Outputs from base model
            batch: Input batch

        Returns:
            Tuple of (f0_cond, sp_cond, ap_cond, hubert_cond)
        """
        # Extract F0 conditioning
        if "f0_pred" in base_outputs:
            f0_cond = base_outputs["f0_pred"]
        elif "f0" in batch:
            f0_cond = batch["f0"]
        else:
            # Create dummy F0 conditioning
            batch_size = batch["audio"].shape[0]
            seq_len = batch["audio"].shape[-1]
            f0_cond = torch.zeros(batch_size, seq_len, 1, device=self.device)

        # Extract SP conditioning
        if "sp_pred" in base_outputs:
            sp_cond = base_outputs["sp_pred"]
        elif "sp" in batch:
            sp_cond = batch["sp"]
        else:
            # Create dummy SP conditioning
            batch_size = batch["audio"].shape[0]
            seq_len = batch["audio"].shape[-1]
            sp_dim = self.config.get("model", {}).get("spectral_envelope_dim", 60)
            sp_cond = torch.zeros(batch_size, seq_len, sp_dim, device=self.device)

        # Extract AP conditioning (use dummy if not available)
        if "ap" in batch:
            ap_cond = batch["ap"]
        else:
            # Create dummy AP conditioning
            batch_size = batch["audio"].shape[0]
            seq_len = batch["audio"].shape[-1]
            ap_cond = torch.zeros(batch_size, seq_len, 1, device=self.device)

        # Extract HuBERT conditioning (use dummy if not available)
        if "hubert" in batch:
            hubert_cond = batch["hubert"]
        else:
            # Create dummy HuBERT conditioning
            batch_size = batch["audio"].shape[0]
            seq_len = batch["audio"].shape[-1]
            hubert_dim = self.config.get("model", {}).get("phoneme_feature_dim", 32)
            hubert_cond = torch.zeros(batch_size, seq_len, hubert_dim, device=self.device)

        return f0_cond, sp_cond, ap_cond, hubert_cond

    def _compute_base_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute base model loss using existing loss functions.

        Args:
            outputs: Base model outputs
            batch: Input batch

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        from ..training.losses import compute_total_loss

        # Prepare targets
        targets = {}
        if "sp" in batch:
            targets["sp"] = batch["sp"]
        if "f0" in batch:
            targets["f0"] = batch["f0"]
        if "f0_quant" in batch:
            targets["f0_quant"] = batch["f0_quant"]

        # Compute loss
        loss, loss_dict = compute_total_loss(
            outputs,
            targets,
            **self.config.get("loss", {})
        )

        return loss, loss_dict

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
            "diffusion_model_state_dict": self.diffusion_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "extra_info": extra_info,
        }

        # Add base model state if joint training
        if self.base_model is not None:
            checkpoint["base_model_state_dict"] = self.base_model.state_dict()

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

        # Load diffusion model state
        self.diffusion_model.load_state_dict(checkpoint["diffusion_model_state_dict"])

        # Load base model state if present
        if self.base_model is not None and "base_model_state_dict" in checkpoint:
            self.base_model.load_state_dict(checkpoint["base_model_state_dict"])

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