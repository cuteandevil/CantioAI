"""
Loss functions for CantioAI training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    L1 loss (Mean Absolute Error).

    Args:
        pred: Predicted tensor
        target: Target tensor
        mask: Optional binary mask (same shape as pred/target) to ignore certain elements

    Returns:
        L1 loss value
    """
    loss = F.l1_loss(pred, target, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def l2_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    L2 loss (Mean Squared Error).

    Args:
        pred: Predicted tensor
        target: Target tensor
        mask: Optional binary mask (same shape as pred/target) to ignore certain elements

    Returns:
        L2 loss value
    """
    loss = F.mse_loss(pred, target, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Huber loss.

    Args:
        pred: Predicted tensor
        target: Target tensor
        delta: Delta parameter for Huber loss
        mask: Optional binary mask (same shape as pred/target) to ignore certain elements

    Returns:
        Huber loss value
    """
    loss = F.huber_loss(pred, target, delta=delta, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l1",
    mask: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Compute sequence loss over time dimension.

    Args:
        pred: Predicted tensor (B, T, D)
        target: Target tensor (B, T, D)
        loss_type: Type of loss ("l1", "l2", "huber")
        mask: Optional binary mask (B, T) to ignore certain time steps
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Sequence loss value
    """
    # Expand mask to feature dimension if provided
    if mask is not None:
        mask = mask.unsqueeze(-1)  # (B, T, 1)

    if loss_type == "l1":
        return l1_loss(pred, target, mask)
    elif loss_type == "l2":
        return l2_loss(pred, target, mask)
    elif loss_type == "huber":
        delta = kwargs.get("delta", 1.0)
        return huber_loss(pred, target, delta, mask)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def f0_loss(
    f0_pred: torch.Tensor,
    f0_target: torch.Tensor,
    f0_quant: torch.Tensor,
    loss_type: str = "l1",
    mask: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Compute F0-related losses.

    Args:
        f0_pred: Predicted F0 (B, T, 1)
        f0_target: Target F0 (B, T, 1) in Hz
        f0_quant: Quantized F0 (B, T, 1) in Hz
        loss_type: Type of loss ("l1", "l2", "huber")
        mask: Optional binary mask (B, T) for voiced frames
        **kwargs: Additional arguments for loss functions

    Returns:
        F0 loss value
    """
    # Option 1: Loss between predicted and quantized F0 (encourages musical accuracy)
    loss_pred_quant = sequence_loss(
        f0_pred, f0_quant, loss_type=loss_type, mask=mask, **kwargs
    )

    # Option 2: Loss between predicted and original F0 (encourages accuracy to target)
    loss_pred_target = sequence_loss(
        f0_pred, f0_target, loss_type=loss_type, mask=mask, **kwargs
    )

    # Combine losses (can be weighted)
    total_f0_loss = loss_pred_quant + 0.5 * loss_pred_target

    return total_f0_loss


def sp_loss(
    sp_pred: torch.Tensor,
    sp_target: torch.Tensor,
    loss_type: str = "l1",
    mask: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Compute spectral envelope loss.

    Args:
        sp_pred: Predicted spectral envelope (B, T, D_sp)
        sp_target: Target spectral envelope (B, T, D_sp)
        loss_type: Type of loss ("l1", "l2", "huber")
        mask: Optional binary mask (B, T) to ignore certain time steps
        **kwargs: Additional arguments for loss functions

    Returns:
        Spectral envelope loss value
    """
    return sequence_loss(sp_pred, sp_target, loss_type=loss_type, mask=mask, **kwargs)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total loss from model outputs and targets.

    Args:
        outputs: Dictionary of model outputs
            - sp_pred: (B, T, D_sp)
            - f0_pred: (B, T, 1) - optional
            - f0_quant: (B, T, 1) - optional
        targets: Dictionary of target values
            - sp: (B, T, D_sp)
            - f0: (B, T, 1) in Hz - optional
            - f0_quant: (B, T, 1) in Hz - optional
        loss_weights: Weights for different loss components
        **kwargs: Additional arguments for loss functions

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Default loss weights
    default_weights = {
        "sp_loss": 1.0,
        "f0_loss": 0.1,
        "f0_quant_loss": 0.0,  # If using quantizer loss
    }
    if loss_weights is None:
        loss_weights = default_weights
    else:
        # Merge with defaults
        for key, value in default_weights.items():
            if key not in loss_weights:
                loss_weights[key] = value

    loss_dict = {}
    total_loss = 0.0

    # Spectral envelope loss
    if "sp_pred" in outputs and "sp" in targets:
        sp_l = sp_loss(
            outputs["sp_pred"],
            targets["sp"],
            loss_type=kwargs.get("sp_loss_type", "l1"),
            mask=targets.get("sp_mask"),
            delta=kwargs.get("huber_delta", 1.0)
        )
        loss_dict["sp_loss"] = sp_l.item()
        total_loss += loss_weights.get("sp_loss", 1.0) * sp_l

    # F0 loss (if F0 prediction is part of model)
    if "f0_pred" in outputs and "f0" in targets:
        f0_l = f0_loss(
            outputs["f0_pred"],
            targets["f0"],
            outputs.get("f0_quant", torch.zeros_like(outputs["f0_pred"])),
            loss_type=kwargs.get("f0_loss_type", "l1"),
            mask=targets.get("f0_mask"),
            delta=kwargs.get("huber_delta", 1.0)
        )
        loss_dict["f0_loss"] = f0_l.item()
        total_loss += loss_weights.get("f0_loss", 0.1) * f0_l

    # Quantizer loss (encourage predictor to output quantized F0)
    if "f0_pred" in outputs and "f0_quant" in targets and outputs.get("f0_quant") is not None:
        f0_q_l = l1_loss(
            outputs["f0_pred"],
            targets["f0_quant"],
            mask=targets.get("f0_mask")
        )
        loss_dict["f0_quant_loss"] = f0_q_l.item()
        total_loss += loss_weights.get("f0_quant_loss", 0.0) * f0_q_l

    # L2 regularization on weights (optional)
    if kwargs.get("weight_decay", 0) > 0:
        # This is usually handled by optimizer weight_decay
        pass

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict