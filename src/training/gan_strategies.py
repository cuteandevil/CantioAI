"""
GAN Training Strategies for Progressive Adversarial Training.
Implements various GAN training techniques and strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class GANTrainingStrategies:
    """
    GAN Training Strategies:
    Implements various GAN training techniques including R1 regularization,
    gradient penalty, spectral normalization, and EMA.
    """

    def __init__(self, config: dict):
        """
        Initialize GAN training strategies.

        Args:
            config: Configuration dictionary for GAN strategies
        """
        self.config = config

        # R1 regularization
        self.r1_regularization = config.get("r1_regularization", 10.0)

        # Gradient penalty (WGAN-GP)
        self.gradient_penalty = config.get("gradient_penalty", True)
        self.penalty_weight = config.get("penalty_weight", 10.0)

        # Spectral normalization
        self.spectral_norm = config.get("spectral_norm", True)

        # Exponential Moving Average (EMA) for generator
        self.ema_generator = config.get("ema_generator", True)
        self.ema_decay = config.get("ema_decay", 0.999)
        self.ema_generator_obj = None  # Will be initialized when needed

        # Loss weights (will be set per phase)
        self.phase_weights = {}

        logger.info(f"Initialized GANTrainingStrategies with config: {config}")

    def set_phase_weights(self, weights: dict):
        """
        Set loss weights for the current training phase.

        Args:
            weights: Dictionary of loss weights for current phase
        """
        self.phase_weights = weights or {}

    def get_weight(self, key: str, default: float = 1.0) -> float:
        """
        Get weight for a loss component.

        Args:
            key: Loss component key
            default: Default weight if key not found

        Returns:
            Weight for the loss component
        """
        return self.phase_weights.get(key, default)

    def get_discriminator_weight(self, disc_name: str) -> float:
        """
        Get weight for a specific discriminator.

        Args:
            disc_name: Name of the discriminator

        Returns:
            Weight for the discriminator
        """
        disc_weights = self.config.get("discriminator_weights", {
            "mpd": 1.0,
            "msd": 0.8,
            "source_filter": 0.5,
            "control_consistency": 0.3
        })
        return disc_weights.get(disc_name, 1.0)

    def get_feature_matching_weight(self, disc_name: str) -> float:
        """
        Get weight for feature matching for a specific discriminator.

        Args:
            disc_name: Name of the discriminator

        Returns:
            Weight for feature matching
        """
        fm_weights = self.config.get("feature_matching_weights", {
            "mpd": 1.0,
            "msd": 0.8,
            "source_filter": 0.5,
            "control_consistency": 0.3
        })
        return fm_weights.get(disc_name, 1.0)

    def compute_r1_penalty(self, real_data, discriminator_output) -> torch.Tensor:
        """
        Compute R1 regularization penalty for stabilizing GAN training.

        Args:
            real_data: Real data tensor
            discriminator_output: Discriminator output for real data

        Returns:
            R1 penalty tensor
        """
        if not self.r1_regularization > 0:
            return torch.tensor(0.0, device=real_data.device)

        # Compute gradients of discriminator output w.r.t. real data
        grad_real = torch.autograd.grad(
            outputs=discriminator_output.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute R1 penalty: ||∇D(x)||²
        grad_penalty = grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        r1_penalty = (self.r1_regularization / 2) * grad_penalty.mean()
        return r1_penalty

    def compute_gradient_penalty(self, real_data, fake_data, discriminator) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.

        Args:
            real_data: Real data tensor
            fake_data: Fake data tensor
            discriminator: Discriminator model

        Returns:
            Gradient penalty tensor
        """
        if not self.gradient_penalty:
            return torch.tensor(0.0, device=real_data.device)

        batch_size = real_data.size(0)
        # Random interpolation between real and fake data
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Get discriminator output for interpolated samples
        try:
            # Try to handle different discriminator output formats
            d_interpolated = discriminator(interpolated)
            if isinstance(d_interpolated, dict):
                # Extract main output from dict
                d_interpolated = d_interpolated.get(
                    "discriminator_output",
                    d_interpolated.get("output", list(d_interpolated.values())[0])
                )
        except Exception as e:
            # Fallback: treat as direct tensor output
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

        # Compute gradient penalty: ||∇D(x̂)|| - 1)²
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return self.penalty_weight * gradient_penalty

    def apply_spectral_norm(self, module: nn.Module) -> nn.Module:
        """
        Apply spectral normalization to a module.

        Args:
            module: PyTorch module to apply spectral norm to

        Returns:
            Module with spectral normalization applied
        """
        if not self.spectral_norm:
            return module

        # Apply spectral norm to all applicable layers
        for name, child in module.named_children():
            if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                setattr(module, name, nn.utils.spectral_norm(child))
            else:
                self.apply_spectral_norm(child)
        return module

    def update_ema(self, generator: nn.Module):
        """
        Update Exponential Moving Average (EMA) of generator weights.

        Args:
            generator: Generator model to update EMA for
        """
        if not self.ema_generator:
            return

        if self.ema_generator_obj is None:
            # Initialize EMA generator
            self.ema_generator_obj = type(generator)(
                *self._get_generator_init_args(generator)
            )
            self.ema_generator_obj.load_state_dict(generator.state_dict())
            # Disable gradients for EMA parameters
            for param in self.ema_generator_obj.parameters():
                param.requires_grad_(False)
        else:
            # Update EMA weights
            for ema_param, param in zip(self.ema_generator_obj.parameters(), generator.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _get_generator_init_args(self, generator: nn.Module) -> list:
        """
        Extract initialization arguments from generator.
        This is a simplified implementation - in practice would be more robust.

        Args:
            generator: Generator model

        Returns:
            List of initialization arguments
        """
        # This is a placeholder implementation
        # In practice, you'd need to store the init args when creating the generator
        return []

    def get_ema_generator(self) -> Optional[nn.Module]:
        """
        Get the EMA generator if enabled.

        Returns:
            EMA generator model or None if not enabled
        """
        return self.ema_generator_obj if self.ema_generator else None

    def compute_adaptive_loss_weights(self, losses: Dict[str, float],
                                    epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Compute adaptive loss weights based on training progress.

        Args:
            losses: Current loss values
            epoch: Current epoch
            total_epochs: Total number of epochs

        Returns:
            Adaptate loss weights
        """
        # Simple adaptive weighting: increase weights for losses that are not decreasing
        progress = epoch / max(total_epochs, 1)
        adaptive_weights = {}

        for loss_name, loss_value in losses.items():
            # Base weight
            base_weight = 1.0

            # Adjust based on loss magnitude (higher loss -> higher weight)
            # This is a simple heuristic - more sophisticated methods could be used
            if loss_value > 0.1:  # Arbitrary threshold
                adaptive_weight = base_weight * (1.0 + loss_value)
            else:
                adaptive_weight = base_weight

            # Apply progress-based scaling
            # Early training: focus on reconstruction
            # Later training: focus on adversarial aspects
            if "adv" in loss_name or "fm" in loss_name or "consistency" in loss_name:
                # Adversarial components: increase weight over time
                progress_factor = 0.5 + 0.5 * progress  # 0.5 to 1.0
                adaptive_weight *= progress_factor
            elif "base" in loss_name:
                # Base component: decrease weight over time
                progress_factor = 1.0 - 0.5 * progress  # 1.0 to 0.5
                adaptive_weight *= progress_factor

            adaptive_weights[loss_name] = adaptive_weight

        return adaptive_weights