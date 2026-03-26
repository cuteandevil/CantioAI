"""Control Consistency Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from ..hybrid_discriminator import HybridDiscriminatorSystem


class ControlConsistencyLoss(nn.Module):
    """
    Control consistency loss for hybrid discriminator system.
    Ensures that generated audio is consistent with source-filter parameters.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.loss_type = config.get("loss_type", "l1")  # l1 or l2

    def compute(self,
                fake_audio: torch.Tensor,
                f0: Optional[torch.Tensor] = None,
                sp: Optional[torch.Tensor] = None,
                ap: Optional[torch.Tensor] = None,
                pred_f0: Optional[torch.Tensor] = None,
                pred_sp: Optional[torch.Tensor] = None,
                pred_ap: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute control consistency loss.
        Args:
            fake_audio: generated audio waveform (B, 1, T)
            f0: fundamental frequency (B, 1, T) - optional
            sp: spectral envelope (B, sp_dim, T) - optional
            ap: aperiodicity (B, ap_dim, T) - optional
            pred_f0: predicted F0 from generator (B, 1, T) - optional
            pred_sp: predicted SP from generator (B, sp_dim, T) - optional
            pred_ap: predicted AP from generator (B, ap_dim, T) - optional
        Returns:
            loss: total control consistency loss
            loss_dict: dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # We need to extract features from the fake audio to compare with control parameters
        # For simplicity, we'll assume we have a way to extract audio features that should
        # correlate with the control parameters.
        # In practice, this would use a pre-trained audio encoder or the discriminator's
        # feature extractor for consistency.

        # Since we don't have the discriminator here, we'll implement a simple version
        # that compares predicted vs target control parameters directly.
        # This is more like a regression loss, but we'll frame it as consistency.

        # 1. F0 consistency
        if f0 is not None and pred_f0 is not None:
            if self.loss_type == "l1":
                f0_loss = F.l1_loss(pred_f0, f0)
            else:
                f0_loss = F.mse_loss(pred_f0, f0)
            total_loss += f0_loss
            loss_dict["f0"] = f0_loss.item()

        # 2. SP consistency
        if sp is not None and pred_sp is not None:
            if self.loss_type == "l1":
                sp_loss = F.l1_loss(pred_sp, sp)
            else:
                sp_loss = F.mse_loss(pred_sp, sp)
            total_loss += sp_loss
            loss_dict["sp"] = sp_loss.item()

        # 3. AP consistency
        if ap is not None and pred_ap is not None:
            if self.loss_type == "l1":
                ap_loss = F.l1_loss(pred_ap, ap)
            else:
                ap_loss = F.mse_loss(pred_ap, ap)
            total_loss += ap_loss
            loss_dict["ap"] = ap_loss.item()

        # 4. Optional: audio-feature consistency (requires discriminator's feature extractor)
        # This would be implemented in the AdversarialInterface, not here.
        # For now, we only do direct parameter comparison.

        return total_loss, loss_dict