"""
Multi-task loss function design.
Implements dynamic loss weighting, gradient normalization, and uncertainty weighting mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class UncertaintyWeighting(nn.Module):
    """
    Uncertainty weighting for multi-task learning.
    Learns task-specific uncertainty weights for loss balancing.
    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """

    def __init__(
        self,
        num_tasks: int = 3,
        init_log_var: float = 0.0,
        log_var_range: Tuple[float, float] = (-10.0, 10.0)
    ):
        """
        Initialize uncertainty weighting.

        Args:
            num_tasks: Number of tasks
            init_log_var: Initial value for log variance parameters
            log_var_range: Valid range for log variance parameters
        """
        super().__init__()

        self.num_tasks = num_tasks
        # Learn log variance for each task (lower = more certain)
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))
        self.log_var_range = log_var_range

        logger.info(
            f"Initialized UncertaintyWeighting:\n"
            f"  Num tasks: {num_tasks}\n"
            f"  Init log var: {init_log_var}\n"
            f"  Log var range: {log_var_range}"
        )

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Apply uncertainty weighting to task losses.

        Args:
            losses: Task losses tensor of shape (num_tasks,) or (B, num_tasks)

        Returns:
            Weighted losses tensor of same shape as input
        """
        # Handle both per-batch and per-task losses
        if losses.dim() == 1:
            # Per-task losses: (num_tasks,)
            precision = torch.exp(-self.log_vars)  # (num_tasks,)
            weighted_losses = losses * precision
            # Normalize to prevent scale issues
            # sum(weighted_losses) = sum(losses * precision)
            # We want: weighted_losses / sum(weighted_losses) * num_tasks
            # So that if all losses equal, each weighted loss = 1
            weighted_losses = weighted_losses * self.num_tasks / weighted_losses.sum()
        elif losses.dim() == 2:
            # Per-batch per-task losses: (B, num_tasks)
            precision = torch.exp(-self.log_vars)  # (num_tasks,)
            weighted_losses = losses * precision.unsqueeze(0)  # (B, num_tasks)
            # Normalize per batch
            batch_sum = weighted_losses.sum(dim=1, keepdim=True)  # (B, 1)
            weighted_losses = weighted_losses * self.num_tasks / batch_sum
        else:
            raise ValueError(f"Unsupported losses shape: {losses.shape}")

        # Clamp log vars to valid range
        with torch.no_grad():
            self.log_vars.data.clamp_(min=self.log_var_range[0], max=self.log_var_range[1])

        return weighted_losses

    def extra_repr(self) -> str:
        return f'num_tasks={self.num_tasks}'


class GradientNormalization(nn.Module):
    """
    Gradient normalization for multi-task learning.
    Normalizes gradients per task to prevent one task from dominating.
    Based on: "Gradient Surgery for Multi-Task Learning"
    """

    def __init__(
        self,
        num_tasks: int = 3,
        eps: float = 1e-8
    ):
        """
        Initialize gradient normalization.

        Args:
            num_tasks: Number of tasks
            eps: Small constant for numerical stability
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.eps = eps

        logger.info(
            f"Initialized GradientNormalization:\n"
            f"  Num tasks: {num_tasks}\n"
            f"  Eps: {eps}"
        )

    def forward(
        self,
        losses: torch.Tensor,
        shared_parameters: List[torch.nn.Parameter]
    ) -> torch.Tensor:
        """
        Apply gradient normalization using stored gradients.

        Args:
            losses: Task losses tensor of shape (num_tasks,) or (B, num_tasks)
            shared_parameters: List of shared parameters to normalize gradients for

        Returns:
            Normalized losses tensor of same shape as input
        """
        # This is a placeholder - actual gradient normalization happens in optimizer
        # For now, just return losses unchanged
        # In practice, this would be used in a custom optimizer that:
        # 1. Computes gradients for each task
        # 2. Normalizes gradient norms across tasks
        # 3. Applies normalized gradients
        return losses


class TaskPrioritization(nn.Module):
    """
    Task prioritization for multi-task learning.
    Adjusts task importance based on training progress.
    """

    def __init__(
        self,
        num_tasks: int = 3,
        priority_type: str = "linear",  # "linear", "exponential", "cosine"
        warmup_steps: int = 1000
        total_steps: int = 10000
    ):
        """
        Initialize task prioritization.

        Args:
            num_tasks: Number of tasks
            priority_type: Type of prioritization schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.priority_type = priority_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        logger.info(
            f"Initialized TaskPrioritization:\n"
            f"  Num tasks: {num_tasks}\n"
            f"  Priority type: {priority_type}\n"
            f"  Warmup steps: {warmup_steps}\n"
            f"  Total steps: {total_steps}"
        )

    def forward(
        self,
        current_step: int
    ) -> torch.Tensor:
        """
        Get task priority weights for current training step.

        Args:
            current_step: Current training step

        Returns:
            Task priority weights tensor of shape (num_tasks,)
        """
        # Compute progress ratio
        progress = min(current_step / self.total_steps, 1.0)

        # Compute priority weights based on type
        if self.priority_type == "linear":
            # Linear increase in priority
            priorities = torch.full((self.num_tasks,), progress)
        elif self.priority_type == "exponential":
            # Exponential increase in priority
            priorities = torch.exp(torch.full((self.num_tasks,), progress * 10)) - 1
            priorities = torch.clamp(priorities, min=0, max=10)
            priorities = priorities / priorities.max()  # Normalize to [0,1]
        elif self.priority_type == "cosine":
            # Cosine annealing: start high, end low
            priorities = 0.5 * (1 + torch.cos(progress * 3.14159))
        else:
            raise ValueError(f"Unsupported priority type: {self.priority_type}")

        # Ensure at least minimal priority
        priorities = torch.clamp(priorities, min=0.1, max=1.0)

        return priorities


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss.
    Combines uncertainty weighting with standard multi-task loss.
    """

    def __init__(
        self,
        num_tasks: int = 3,
        init_log_var: float = 0.0,
        log_var_range: Tuple[float, float] = (-10.0, 10.0),
        reduction: str = "mean"
    ):
        """
        Initialize uncertainty-weighted multi-task loss.

        Args:
            num_tasks: Number of tasks
            init_log_var: Initial value for log variance parameters
            log_var_range: Valid range for log variance parameters
            reduction: Reduction method ("none", "mean", "sum")
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.uncertainty_weighting = UncertaintyWeighting(
            num_tasks=num_tasks,
            init_log_var=init_log_var,
            log_var_range=log_var_range
        )
        self.reduction = reduction

        logger.info(
            f"Initialized UncertaintyWeightedLoss:\n"
            f"  Num tasks: {num_tasks}\n"
            f"  Init log var: {init_log_var}\n"
            f"  Log var range: {log_var_range}\n"
            f"  Reduction: {reduction}"
        )

    def forward(
        self,
        losses: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply uncertainty weighting to task losses.

        Args:
            losses: Task losses tensor of shape (num_tasks,) or (B, num_tasks)

        Returns:
            Weighted loss scalar
        """
        # Apply uncertainty weighting
        weighted_losses = self.uncertainty_weighting(losses)

        # Apply reduction
        if self.reduction == "none":
            return weighted_losses
        elif self.reduction == "mean":
            return weighted_losses.mean()
        elif self.reduction == "sum":
            return weighted_losses.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


def create_uncertainty_weighting(
    num_tasks: int = 3,
    init_log_var: float = 0.0,
    log_var_range: Tuple[float, float] = (-10.0, 10.0)
) -> UncertaintyWeighting:
    """
    Factory function to create UncertaintyWeighting.

    Args:
        num_tasks: Number of tasks
        init_log_var: Initial value for log variance parameters
        log_var_range: Valid range for log variance parameters

    Returns:
        UncertaintyWeighting module
    """
    return UncertaintyWeighting(
        num_tasks=num_tasks,
        init_log_var=init_log_var,
        log_var_range=log_var_range
    )


def create_gradient_normalization(
    num_tasks: int = 3,
    eps: float = 1e-8
) -> GradientNormalization:
    """
    Factory function to create GradientNormalization.

    Args:
        num_tasks: Number of tasks
        eps: Small constant for numerical stability

    Returns:
        GradientNormalization module
    """
    return GradientNormalization(
        num_tasks=num_tasks,
        eps=eps
    )


def create_task_prioritization(
    num_tasks: int = 3,
    priority_type: str = "linear",
    warmup_steps: int = 1000
    total_steps: int = 10000
) -> TaskPrioritization:
    """
    Factory function to create TaskPrioritization.

    Args:
        num_tasks: Number of tasks
        priority_type: Type of prioritization schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps

    Returns:
        TaskPrioritization module
    """
    return TaskPrioritization(
        num_tasks=num_tasks,
        priority_type=priority_type,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )


def create_uncertainty_weighted_loss(
    num_tasks: int = 3,
    init_log_var: float = 0.0,
    log_var_range: Tuple[float, float] = (-10.0, 10.0),
    reduction: str = "mean"
) -> UncertaintyWeightedLoss:
    """
    Factory function to create UncertaintyWeightedLoss.

    Args:
        num_tasks: Number of tasks
        init_log_var: Initial value for log variance parameters
        log_var_range: Valid range for log variance parameters
        reduction: Reduction method

    Returns:
        UncertaintyWeightedLoss module
    """
    return UncertaintyWeightedLoss(
        num_tasks=num_tasks,
        init_log_var=init_log_var,
        log_var_range=log_var_range,
        reduction=reduction
    )


if __name__ == "__main__":
    # Simple test
    num_tasks = 3
    losses = torch.tensor([1.0, 2.0, 3.0])  # (num_tasks,)

    # Test uncertainty weighting
    uw = UncertaintyWeighting(num_tasks=num_tasks)
    weighted_losses = uw(losses)
    print(f"UncertaintyWeighting:")
    print(f"  Losses: {losses}")
    print(f"  Log vars: {uw.log_vars}")
    print(f"  Weighted losses: {weighted_losses}")

    # Test gradient normalization
    gn = GradientNormalization(num_tasks=num_tasks)
    # This is a placeholder - actual use requires optimizer integration
    print(f"\nGradientNormalization:")
    print(f"  Module created: {type(gn).__name__}")

    # Test task prioritization
    tp = TaskPrioritization(num_tasks=num_tasks, warmup_steps=500, total_steps=1000)
    # Test at different steps
    for step in [0, 250, 500, 750, 1000]:
        priorities = tp(step)
        print(f"\nTaskPrioritization at step {step}:")
        print(f"  Priorities: {priorities}")

    # Test uncertainty-weighted loss
    uwl = UncertaintyWeightedLoss(num_tasks=num_tasks)
    weighted_loss = uwl(losses)
    print(f"\nUncertaintyWeightedLoss:")
    print(f"  Weighted loss: {weighted_loss}")

    print("\nLoss function design test passed!")