"""
Progressive Adversarial Trainer for CantioAI.
Implements staged, configurable adversarial training strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any, List, Union
import logging
import time
import os
from pathlib import Path
from typing import Union

from ..models.hybrid_svc import HybridSVC
from ..discriminators import HybridDiscriminatorSystem
from ..discriminators.specialized.source_filter_discriminator import SourceFilterDiscriminator
from ..discriminators.specialized.control_consistency_disc import ControlConsistencyDiscriminator
from ..discriminators.losses.enhanced_feature_matching_loss import EnhancedFeatureMatchingLoss
from ..discriminators.losses.detailed_consistency_loss import DetailedConsistencyLoss

logger = logging.getLogger(__name__)


class GANTrainingStrategies:
    """GAN training strategies implementation."""

    def __init__(self, config: dict):
        self.config = config
        self.r1_regularization = config.get("r1_regularization", 10.0)
        self.gradient_penalty = config.get("gradient_penalty", True)
        self.penalty_weight = config.get("penalty_weight", 10.0)
        self.spectral_norm = config.get("spectral_norm", True)
        self.ema_generator = config.get("ema_generator", True)
        self.ema_decay = config.get("ema_decay", 0.999)

        # EMA generator if enabled
        if self.ema_generator:
            self.ema_generator = None  # Will be initialized when needed
        else:
            self.ema_generator = None

    def set_phase_weights(self, weights: dict):
        """Set loss weights for current training phase."""
        self.phase_weights = weights or {}

    def get_weight(self, key: str, default: float = 1.0) -> float:
        """Get weight for a loss component."""
        return self.phase_weights.get(key, default)

    def get_discriminator_weight(self, disc_name: str) -> float:
        """Get weight for a specific discriminator."""
        disc_weights = self.config.get("discriminator_weights", {
            "mpd": 1.0,
            "msd": 0.8,
            "source_filter": 0.5,
            "control_consistency": 0.3
        })
        return disc_weights.get(disc_name, 1.0)

    def get_feature_matching_weight(self, disc_name: str) -> float:
        """Get weight for feature matching for a specific discriminator."""
        fm_weights = self.config.get("feature_matching_weights", {
            "mpd": 1.0,
            "msd": 0.8,
            "source_filter": 0.5,
            "control_consistency": 0.3
        })
        return fm_weights.get(disc_name, 1.0)

    def compute_r1_penalty(self, real_data, discriminator_output) -> torch.Tensor:
        """Compute R1 regularization penalty."""
        if not self.r1_regularization > 0:
            return torch.tensor(0.0, device=real_data.device)

        # Gradient of discriminator output w.r.t. real data
        grad_real = torch.autograd.grad(
            outputs=discriminator_output.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute penalty
        grad_penalty = grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        r1_penalty = (self.r1_regularization / 2) * grad_penalty.mean()
        return r1_penalty

    def compute_gradient_penalty(self, real_data, fake_data, discriminator) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        if not self.gradient_penalty:
            return torch.tensor(0.0, device=real_data.device)

        batch_size = real_data.size(0)
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Get discriminator output for interpolated samples
        if hasattr(discriminator, '__call__'):
            # Handle different discriminator types
            try:
                # Try standard audio discriminator call
                d_interpolated = discriminator(interpolated)
                if isinstance(d_interpolated, dict):
                    # For complex discriminators, use main output
                    d_interpolated = d_interpolated.get("discriminator_output",
                                                      d_interpolated.get("output", list(d_interpolated.values())[0]))
            except:
                # Fallback
                d_interpolated = discriminator(interpolated)
        else:
            d_interpolated = discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated.sum(),
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return self.penalty_weight * gradient_penalty

    def update_ema(self, generator):
        """Update EMA generator weights."""
        if not self.ema_generator or not self.ema_generator:
            # Initialize EMA generator
            self.ema_generator = type(generator)(
                *[getattr(generator, attr) for attr in dir(generator)
                  if not attr.startswith('_') and not callable(getattr(generator, attr))]
            )
            self.ema_generator.load_state_dict(generator.state_dict())
            for param in self.ema_generator.parameters():
                param.requires_grad_(False)
        else:
            # Update EMA weights
            for ema_param, param in zip(self.ema_generator.parameters(), generator.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)


class ProgressiveAdversarialTrainer:
    """
    Progressive Adversarial Trainer:
    Implements staged, configurable adversarial training strategies.

    Design points:
    1. Staged training: Basic training → Partial adversarial → Full adversarial
    2. Dynamic scheduling: Automatically adjust training strategy
    3. Multiple GAN variants: Supports various GAN training techniques
    4. Integration with existing trainer: Builds upon CantioAITrainer
    """

    def __init__(self, generator: HybridSVC, discriminators: dict, config: dict):
        """
        Initialize the progressive adversarial trainer.

        Args:
            generator: HybridSVC model (generator)
            discriminators: Dictionary of discriminators
            config: Training configuration
        """
        self.generator = generator
        self.discriminators = discriminators
        self.config = config

        # Initialize optimizers
        self.optimizer_g = self._create_optimizer(
            generator.parameters(),
            config.get("optimizer", {}).get("generator", {})
        )

        self.optimizer_d = {}
        for name, disc in discriminators.items():
            self.optimizer_d[name] = self._create_optimizer(
                disc.parameters(),
                config.get("optimizer", {}).get("discriminators", {}).get(name, {})
            )

        # Initialize loss functions
        self.loss_functions = self._create_loss_functions(config.get("losses", {}))

        # Initialize GAN strategies
        self.gan_strategies = GANTrainingStrategies(config.get("gan_strategies", {}))

        # Training state
        self.current_phase = "phase1"  # phase1, phase2, phase3
        self.current_epoch = 0
        self.global_step = 0

        # Training history
        self.training_history = {
            "generator_losses": [],
            "discriminator_losses": {name: [] for name in discriminators.keys()},
            "feature_matching_losses": [],
            "consistency_losses": []
        }

        # Learning rate schedulers
        self.schedulers = self._create_schedulers(config.get("schedulers", {}))

        # Set initial training phase
        self.set_training_phase("phase1")

        # Device
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.generator.to(self.device)
        for disc in self.discriminators.values():
            disc.to(self.device)

        # Mixed precision
        self.use_amp = config.get("training", {}).get("use_amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        logger.info(f"Initialized ProgressiveAdversarialTrainer in {self.current_phase}")

    def _create_optimizer(self, parameters, config):
        """Create optimizer from config."""
        optimizer_type = config.get("type", "adam")
        lr = config.get("lr", 0.0001)
        betas = tuple(config.get("betas", [0.5, 0.999]))
        weight_decay = config.get("weight_decay", 0.0)

        if optimizer_type == "adam":
            return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            return optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            momentum = config.get("momentum", 0.9)
            return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _create_loss_functions(self, loss_config):
        """Create loss functions from config."""
        losses = {}

        # Base loss (reconstruction)
        if "base" in loss_config:
            from ..training.losses import compute_total_loss  # Use existing loss function
            losses["base"] = lambda pred, target: nn.L1Loss()(pred, target)  # Simplified

        # Adversarial loss
        if "adversarial" in loss_config:
            losses["adversarial"] = EnhancedAdversarialLoss(loss_config.adversarial)

        # Feature matching loss
        if "feature_matching" in loss_config:
            losses["feature_matching"] = EnhancedFeatureMatchingLoss(loss_config.feature_matching)

        # Consistency loss
        if "consistency" in loss_config:
            losses["consistency"] = DetailedConsistencyLoss(loss_config.consistency)

        return losses

    def _create_schedulers(self, scheduler_config):
        """Create learning rate schedulers from config."""
        schedulers = {}

        # Generator scheduler
        if "generator" in scheduler_config:
            schedulers["generator"] = self._create_scheduler(
                self.optimizer_g,
                scheduler_config.generator
            )

        # Discriminator schedulers
        for name in self.discriminators.keys():
            if name in scheduler_config:
                schedulers[name] = self._create_scheduler(
                    self.optimizer_d[name],
                    scheduler_config[name]
                )

        return schedulers

    def _create_scheduler(self, optimizer, config):
        """Create a single learning rate scheduler."""
        scheduler_type = config.get("type", "step")

        if scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("step_size", 10),
                gamma=config.get("gamma", 0.5)
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get("T_max", 100)
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get("mode", "min"),
                factor=config.get("factor", 0.5),
                patience=config.get("patience", 5)
            )
        else:
            return None

    def set_training_phase(self, phase: str):
        """
        Set the current training phase.

        Args:
            phase: Training phase ("phase1", "phase2", "phase3")
        """
        self.current_phase = phase
        phase_config = self.config.get("phases", {}).get(phase, {})

        # Set discriminator update frequency
        self.d_update_frequency = phase_config.get("d_update_frequency", 1)

        # Set loss weights for GAN strategies
        self.gan_strategies.set_phase_weights(phase_config.get("loss_weights", {}))

        # Freeze/unfreeze discriminators based on phase
        if phase == "phase1":
            self._freeze_discriminators()
            logger.info("Set to phase1: Generator only training")
        else:
            self._unfreeze_discriminators()
            logger.info(f"Set to {phase}: Adversarial training enabled")

        logger.info(f"Discriminator update frequency: {self.d_update_frequency}")

    def _freeze_discriminators(self):
        """Freeze all discriminator parameters."""
        for name, disc in self.discriminators.items():
            for param in disc.parameters():
                param.requires_grad = False
            logger.debug(f"Frozen discriminator: {name}")

    def _unfreeze_discriminators(self):
        """Unfreeze all discriminator parameters."""
        for name, disc in self.discriminators.items():
            for param in disc.parameters():
                param.requires_grad = True
            logger.debug(f"Unfrozen discriminator: {name}")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Input batch dictionary
            batch_idx: Current batch index

        Returns:
            Dictionary of losses for the step
        """
        # Move batch to device
        batch = self._move_to_device(batch)

        # Extract batch components
        phoneme_features = batch["phoneme_features"]
        f0 = batch["f0"]
        sp = batch["sp"]
        ap = batch["ap"]
        spk_id = batch["spk_id"]
        audio = batch["audio"]

        losses = {}

        # ----------------------
        # Generator Forward Pass
        # ----------------------
        self.generator.train()
        sp_pred, f0_quant, extras = self.generator(
            phoneme_features=phoneme_features,
            f0=f0,
            spk_id=spk_id,
            f0_is_hz=True,
            return_quantized_f0=True
        )

        # Synthesize audio using vocoder (simplified - would use actual vocoder)
        # For now, we'll use a placeholder - in practice this would call the vocoder
        pred_audio = self._synthesize_audio(sp_pred, f0, ap)

        # ----------------------
        # Discriminator Training
        # ----------------------
        d_losses = {}
        if self.current_phase != "phase1" and batch_idx % self.d_update_frequency == 0:
            for name, discriminator in self.discriminators.items():
                self.optimizer_d[name].zero_grad()

                # Get discriminator outputs
                if name == "source_filter":
                    # Source filter discriminator takes parameters directly
                    real_output = discriminator(f0, sp, ap, return_features=True)
                    fake_output = discriminator(f0, sp.detach(), ap.detach(), return_features=False)
                elif name == "control_consistency":
                    # Control consistency discriminator needs audio features
                    # Extract features from real and fake audio
                    real_audio_features = self._extract_audio_features(audio)
                    fake_audio_features = self._extract_audio_features(pred_audio.detach())
                    real_output = discriminator(real_audio_features, f0, sp, ap, return_features=True)
                    fake_output = discriminator(fake_audio_features, f0, sp.detach(), ap.detach(), return_features=False)
                else:
                    # Standard audio discriminators (MPD, MSD, etc.)
                    real_output = discriminator(audio, f0, sp, ap, return_features=True)
                    fake_output = discriminator(pred_audio.detach(), f0, sp, ap, return_features=False)

                # Compute discriminator loss
                d_loss, d_loss_dict = self.loss_functions["adversarial"].discriminator_loss(
                    real_output, fake_output, discriminator_type=name
                )

                # Add regularization if applicable
                if name in ["mpd", "msd"]:  # Apply R1 to standard discriminators
                    r1_penalty = self.gan_strategies.compute_r1_penalty(audio, real_output)
                    d_loss += r1_penalty
                    d_loss_dict["r1_penalty"] = r1_penalty.item()

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(d_loss).backward()
                    if self.config.get("training", {}).get("grad_clip", 0) > 0:
                        self.scaler.unscale_(self.optimizer_d[name])
                        torch.nn.utils.clip_grad_norm_(
                            discriminator.parameters(),
                            self.config["training"]["grad_clip"]
                        )
                    self.scaler.step(self.optimizer_d[name])
                    self.scaler.update()
                else:
                    d_loss.backward()
                    if self.config.get("training", {}).get("grad_clip", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            discriminator.parameters(),
                            self.config["training"]["grad_clip"]
                        )
                    self.optimizer_d[name].step()

                # Record losses
                d_losses[f"d_{name}_loss"] = d_loss.item()
                for key, value in d_loss_dict.items():
                    d_losses[f"d_{name}_{key}"] = value

                # Update history
                self.training_history["discriminator_losses"][name].append(d_loss.item())

        # ----------------------
        # Generator Training
        # ----------------------
        self.optimizer_g.zero_grad()

        # Base loss (reconstruction)
        base_loss = self.loss_functions["base"](sp_pred, sp)
        losses["base_loss"] = base_loss.item()

        # Initialize adversarial losses
        adv_loss = 0.0
        adv_loss_dict = {}
        fm_loss = 0.0
        fm_loss_dict = {}
        consistency_loss = 0.0
        consistency_loss_dict = {}

        if self.current_phase != "phase1":
            # Adversarial loss
            adv_loss, adv_loss_dict = self._compute_generator_adversarial_loss(
                pred_audio, f0, sp, ap, sp_pred
            )
            losses["adv_loss"] = adv_loss.item()
            for key, value in adv_loss_dict.items():
                losses[f"adv_{key}"] = value

            # Feature matching loss
            if "feature_matching" in self.loss_functions:
                fm_loss, fm_loss_dict = self._compute_feature_matching_loss(
                    audio, pred_audio, f0, sp, ap
                )
                losses["fm_loss"] = fm_loss.item()
                for key, value in fm_loss_dict.items():
                    losses[f"fm_{key}"] = value

            # Consistency loss
            if "consistency" in self.loss_functions:
                consistency_loss, consistency_loss_dict = self._compute_consistency_loss(
                    audio, pred_audio, f0, sp, ap, sp_pred
                )
                losses["consistency_loss"] = consistency_loss.item()
                for key, value in consistency_loss_dict.items():
                    losses[f"consistency_{key}"] = value

        # Compute total generator loss
        total_g_loss = (
            self.gan_strategies.get_weight("base", 1.0) * base_loss +
            self.gan_strategies.get_weight("adversarial", 0.0) * adv_loss +
            self.gan_strategies.get_weight("feature_matching", 0.0) * fm_loss +
            self.gan_strategies.get_weight("consistency", 0.0) * consistency_loss
        )

        losses["total_g_loss"] = total_g_loss.item()

        # Backward pass for generator
        if self.use_amp:
            self.scaler.scale(total_g_loss).backward()
            if self.config.get("training", {}).get("grad_clip", 0) > 0:
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config["training"]["grad_clip"])
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            total_g_loss.backward()
            if self.config.get("training", {}).get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config["training"]["grad_clip"])
            self.optimizer_g.step()

        # Update EMA generator if enabled
        if self.gan_strategies.ema_generator:
            self.gan_strategies.update_ema(self.generator)

        # Update learning rate schedulers
        self._update_schedulers()

        # Update global step
        self.global_step += 1

        # Record losses in history
        self.training_history["generator_losses"].append(total_g_loss.item())
        self.training_history["feature_matching_losses"].append(fm_loss)
        self.training_history["consistency_losses"].append(consistency_loss)

        # Log metrics periodically
        if self.global_step % 100 == 0:
            logger.info(
                f"Step {self.global_step} - "
                f"G Loss: {total_g_loss.item():.6f}, "
                f"Base: {base_loss.item():.6f}, "
                f"Adv: {adv_loss:.6f}, "
                f"FM: {fm_loss:.6f}, "
                f"Cons: {consistency_loss:.6f}"
            )

        return losses

    def _compute_generator_adversarial_loss(self, pred_audio, f0, sp, ap, pred_sp):
        """Compute adversarial loss for generator."""
        adv_loss = 0.0
        adv_loss_dict = {}

        for name, discriminator in self.discriminators.items():
            weight = self.gan_strategies.get_discriminator_weight(name)

            if name == "source_filter":
                # Source filter discriminator: want generated params to be classified as real
                fake_output = discriminator(f0, pred_sp, ap, return_features=True)
                # Get the main output (adjust based on actual discriminator output structure)
                if isinstance(fake_output, dict):
                    pred_scores = fake_output.get("discriminator_output",
                                              fake_output.get("output", list(fake_output.values())[0]))
                else:
                    pred_scores = fake_output

                # Want discriminator to output ~1 (real) for generated data
                target = torch.ones_like(pred_scores)
                loss = F.binary_cross_entropy(pred_scores, target)

            elif name == "control_consistency":
                # Control consistency discriminator: want high consistency score
                pred_audio_features = self._extract_audio_features(pred_audio)
                fake_output = discriminator(pred_audio_features, f0, pred_sp, ap, return_features=True)
                if isinstance(fake_output, dict):
                    consistency_score = fake_output.get("consistency_score",
                                                      fake_output.get("score", list(fake_output.values())[0]))
                else:
                    consistency_score = fake_output

                # Want consistency score to be ~1 (consistent)
                target = torch.ones_like(consistency_score)
                loss = F.binary_cross_entropy(consistency_score, target)

            else:
                # Standard audio discriminators
                fake_output = discriminator(pred_audio, f0, sp, ap, return_features=True)
                if isinstance(fake_output, dict):
                    pred_scores = fake_output.get("discriminator_output",
                                              fake_output.get("output", list(fake_output.values())[0]))
                else:
                    pred_scores = fake_output

                # Standard adversarial loss: generator wants discriminator to think fake is real
                target = torch.ones_like(pred_scores)
                loss = F.binary_cross_entropy(pred_scores, target)

            weighted_loss = loss * weight
            adv_loss += weighted_loss
            adv_loss_dict[f"adv_{name}_loss"] = loss.item()
            adv_loss_dict[f"adv_{name}_weighted"] = weighted_loss.item()

        return adv_loss, adv_loss_dict

    def _compute_feature_matching_loss(self, real_audio, fake_audio, f0, sp, ap):
        """Compute feature matching loss."""
        fm_loss = 0.0
        fm_loss_dict = {}

        for name, discriminator in self.discriminators.items():
            weight = self.gan_strategies.get_feature_matching_weight(name)

            if name == "source_filter":
                # Feature matching for source filter discriminator
                real_features = discriminator(f0, sp, ap, return_features=True)
                fake_features = discriminator(f0, sp, ap, return_features=True)  # Same input for params
            elif name == "control_consistency":
                # Feature matching for control consistency discriminator
                real_audio_features = self._extract_audio_features(real_audio)
                fake_audio_features = self._extract_audio_features(fake_audio)
                real_features = discriminator(real_audio_features, f0, sp, ap, return_features=True)
                fake_features = discriminator(fake_audio_features, f0, sp, ap, return_features=True)
            else:
                # Standard audio discriminators
                real_features = discriminator(real_audio, f0, sp, ap, return_features=True)
                fake_features = discriminator(fake_audio, f0, sp, ap, return_features=True)

            # Compute feature matching loss
            loss, loss_details = self.loss_functions["feature_matching"](
                real_features, fake_features
            )

            weighted_loss = loss * weight
            fm_loss += weighted_loss
            fm_loss_dict[f"fm_{name}_loss"] = loss.item()
            fm_loss_dict[f"fm_{name}_weighted"] = weighted_loss.item()
            for key, value in loss_details.items():
                fm_loss_dict[f"fm_{name}_{key}"] = value

        return fm_loss, fm_loss_dict

    def _compute_consistency_loss(self, real_audio, fake_audio, f0, sp, ap, pred_sp):
        """Compute consistency loss."""
        consistency_loss = 0.0
        consistency_loss_dict = {}

        # Only compute if we have the control consistency discriminator
        if "control_consistency" in self.discriminators:
            discriminator = self.discriminators["control_consistency"]

            # Get detailed consistency for real audio
            real_audio_features = self._extract_audio_features(real_audio)
            real_consistency = discriminator.compute_detailed_consistency(
                real_audio_features, f0, sp, ap
            )

            # Get detailed consistency for generated audio
            fake_audio_features = self._extract_audio_features(fake_audio)
            fake_consistency = discriminator.compute_detailed_consistency(
                fake_audio_features, f0, pred_sp, ap
            )

            # Compute loss between real and generated consistency metrics
            loss_components = []
            for key in real_consistency.keys():
                if key in fake_consistency:
                    if isinstance(real_consistency[key], torch.Tensor) and isinstance(fake_consistency[key], torch.Tensor):
                        if self.loss_type == "l1":
                            loss = F.l1_loss(fake_consistency[key], real_consistency[key])
                        else:
                            loss = F.mse_loss(fake_consistency[key], real_consistency[key])
                    else:
                        # Handle scalar values
                        loss = abs(fake_consistency[key] - real_consistency[key])
                    loss_components.append(loss)

                    consistency_loss_dict[f"consistency_{key}"] = loss.item()

            if loss_components:
                consistency_loss = sum(loss_components) / len(loss_components)

        return consistency_loss, consistency_loss_dict

    def _extract_audio_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from audio for consistency computation."""
        # Simple placeholder - in practice would use mel-spectrogram or learned encoder
        B, _, T = audio.shape
        feature_dim = 80
        time_steps = max(1, T // 16)  # Typical downsampling
        # Create features using simple operations (placeholder)
        features = torch.randn(B, feature_dim, time_steps, device=audio.device)
        return features

    def _synthesize_audio(self, sp_pred: torch.Tensor, f0: torch.Tensor, ap: torch.Tensor) -> torch.Tensor:
        """
        Synthesize audio from predicted parameters.
        This is a simplified placeholder - in practice would use the actual vocoder.
        """
        # For now, create a dummy audio tensor of appropriate shape
        # In practice, this would call the neural vocoder or WORLD synthesizer
        B, _, T_sp = sp_pred.shape
        # Audio is typically longer than feature frames
        T_audio = T_sp * 16  # Typical upsampling factor
        audio = torch.randn(B, 1, T_audio, device=sp_pred.device)
        return audio

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _update_schedulers(self):
        """Update learning rate schedulers."""
        # Update generator scheduler
        if "generator" in self.schedulers and self.schedulers["generator"] is not None:
            scheduler = self.schedulers["generator"]
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Will be updated with validation loss
                pass
            else:
                scheduler.step()

        # Update discriminator schedulers
        for name, scheduler in self.schedulers.items():
            if name != "generator" and scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Will be updated with validation loss
                    pass
                else:
                    scheduler.step()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average losses for the epoch
        """
        self.generator.train()
        for disc in self.discriminators.values():
            disc.train()

        epoch_losses = {}
        epoch_loss_count = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            batch_losses = self.training_step(batch, batch_idx)

            # Accumulate losses
            for key, value in batch_losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            epoch_loss_count += 1

        # Calculate averages
        avg_losses = {k: v / epoch_loss_count for k, v in epoch_losses.items()}

        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s - "
            f"G Loss: {avg_losses.get('total_g_loss', 0.0):.6f}"
        )

        return avg_losses

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average losses for the epoch
        """
        self.generator.eval()
        for disc in self.discriminators.values():
            disc.eval()

        epoch_losses = {}
        epoch_loss_count = 0

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Extract components
                phoneme_features = batch["phoneme_features"]
                f0 = batch["f0"]
                sp = batch["sp"]
                ap = batch["ap"]
                spk_id = batch["spk_id"]
                audio = batch["audio"]

                # Generator forward pass
                sp_pred, f0_quant, extras = self.generator(
                    phoneme_features=phoneme_features,
                    f0=f0,
                    spk_id=spk_id,
                    f0_is_hz=True,
                    return_quantized_f0=True
                )

                # Synthesize audio
                pred_audio = self._synthesize_audio(sp_pred, f0, ap)

                # Compute validation losses (similar to training but without backward pass)
                batch_losses = {}

                # Base loss
                base_loss = nn.L1Loss()(sp_pred, sp)
                batch_losses["base_loss"] = base_loss.item()

                # Adversarial loss (if not phase1)
                if self.current_phase != "phase1":
                    adv_loss, _ = self._compute_generator_adversarial_loss(
                        pred_audio, f0, sp, ap, sp_pred
                    )
                    batch_losses["adv_loss"] = adv_loss.item()

                    # Feature matching loss
                    if "feature_matching" in self.loss_functions:
                        fm_loss, _ = self._compute_feature_matching_loss(
                            audio, pred_audio, f0, sp, ap
                        )
                        batch_losses["fm_loss"] = fm_loss.item()

                    # Consistency loss
                    if "consistency" in self.loss_functions:
                        consistency_loss, _ = self._compute_consistency_loss(
                            audio, pred_audio, f0, sp, ap, sp_pred
                        )
                        batch_losses["consistency_loss"] = consistency_loss.item()

                # Total loss
                total_loss = (
                    self.gan_strategies.get_weight("base", 1.0) * base_loss +
                    self.gan_strategies.get_weight("adversarial", 0.0) * adv_loss +
                    self.gan_strategies.get_weight("feature_matching", 0.0) * fm_loss +
                    self.gan_strategies.get_weight("consistency", 0.0) * consistency_loss
                )
                batch_losses["total_loss"] = total_loss.item()

                # Accumulate losses
                for key, value in batch_losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                epoch_loss_count += 1

        # Calculate averages
        avg_losses = {k: v / epoch_loss_count for k, v in epoch_losses.items()}

        epoch_time = time.time() - start_time
        logger.info(
            f"Validation Epoch {epoch} completed in {epoch_time:.2f}s - "
            f"Loss: {avg_losses.get('total_loss', 0.0):.6f}"
        )

        return avg_losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Main training loop with progressive stages.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting progressive adversarial training...")
        start_time = time.time()

        try:
            # Get phase configuration
            phases = self.config.get("phases", {})
            phase_order = ["phase1", "phase2", "phase3"]

            for phase in phase_order:
                if phase not in phases:
                    continue

                phase_config = phases[phase]
                max_epochs = phase_config.get("max_epochs", 50)

                logger.info(f"Starting {phase} for {max_epochs} epochs")
                self.set_training_phase(phase)

                # Train for this phase's epochs
                for epoch in range(self.current_epoch + 1, self.current_epoch + max_epochs + 1):
                    # Train
                    train_losses = self.train_epoch(train_loader, epoch)

                    # Validate
                    if epoch % self.config.get("training", {}).get("validation_interval", 1) == 0:
                        val_losses = self.validate_epoch(val_loader, epoch)

                        # Update ReduceLROnPlateau schedulers with validation loss
                        if "generator" in self.schedulers:
                            sched = self.schedulers["generator"]
                            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                                sched.step(val_losses.get("total_loss", 0.0))

                        for name in self.discriminators.keys():
                            if name in self.schedulers:
                                sched = self.schedulers[name]
                                if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                                    sched.step(val_losses.get("total_loss", 0.0))

                    self.current_epoch = epoch

                    # Save checkpoint
                    if epoch % self.config.get("training", {}).get("save_checkpoint_freq", 10) == 0:
                        self.save_checkpoint(epoch)

                logger.info(f"Completed {phase}")

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
                f"Final epoch: {self.current_epoch}"
            )

            # Save final model
            self.save_checkpoint(self.current_epoch, extra_info={"final": True})

    def save_checkpoint(self, epoch: int, path: Optional[Union[str, Path]] = None,
                       extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch number
            path: Path to save checkpoint (optional)
            extra_info: Additional information to save
        """
        if extra_info is None:
            extra_info = {}

        if path is None:
            path = f"checkpoints/progressive_adversarial_phase_{self.current_phase}_epoch_{epoch:04d}.pt"

        checkpoint = {
            "epoch": epoch,
            "current_phase": self.current_phase,
            "global_step": self.global_step,
            "generator_state_dict": self.generator.state_dict(),
            "discriminators_state_dict": {
                name: disc.state_dict() for name, disc in self.discriminators.items()
            },
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": {
                name: opt.state_dict() for name, opt in self.optimizer_d.items()
            },
            "schedulers_state_dict": {
                name: sched.state_dict() if sched is not None else None
                for name, sched in self.schedulers.items()
            },
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "training_history": self.training_history,
            "config": self.config,
            "extra_info": extra_info,
            "gan_strategies_state": {
                "ema_generator": self.gan_strategies.ema_generator.state_dict()
                if self.gan_strategies.ema_generator and self.gan_strategies.ema_generator else None
            }
        }

        if path:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, filepath)
            logger.info(f"Saved checkpoint: {filepath}")

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Epoch number of the loaded checkpoint
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # Load model states
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        for name, disc in self.discriminators.items():
            if name in checkpoint["discriminators_state_dict"]:
                disc.load_state_dict(checkpoint["discriminators_state_dict"][name])

        # Load optimizer states
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        for name, opt in self.optimizer_d.items():
            if name in checkpoint["optimizer_d_state_dict"]:
                opt.load_state_dict(checkpoint["optimizer_d_state_dict"][name])

        # Load scheduler states
        for name, sched in self.schedulers.items():
            if name in checkpoint["schedulers_state_dict"] and checkpoint["schedulers_state_dict"][name] is not None:
                sched.load_state_dict(checkpoint["schedulers_state_dict"][name])

        # Load scaler state
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.current_phase = checkpoint.get("current_phase", "phase1")
        self.global_step = checkpoint.get("global_step", 0)
        self.training_history = checkpoint.get("training_history", {
            "generator_losses": [],
            "discriminator_losses": {name: [] for name in self.discriminators.keys()},
            "feature_matching_losses": [],
            "consistency_losses": []
        })
        self.config = checkpoint.get("config", self.config)

        # Load GAN strategies
        ema_state = checkpoint.get("gan_strategies_state", {}).get("ema_generator")
        if ema_state and self.gan_strategies.ema_generator:
            self.gan_strategies.ema_generator.load_state_dict(ema_state)

        logger.info(
            f"Loaded checkpoint from epoch {self.current_epoch}, "
            f"phase: {self.current_phase}, step: {self.global_step}"
        )

        return self.current_epoch