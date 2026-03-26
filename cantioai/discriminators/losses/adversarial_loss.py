"""Adversarial Loss Functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for hybrid discriminator system.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.leaky_relu_slope = config.get("leaky_relu_slope", 0.1)

    def discriminator_loss(self,
                          real_outputs: dict,
                          fake_outputs: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss.
        Args:
            real_outputs: outputs from discriminator for real data
            fake_outputs: outputs from discriminator for fake data
        Returns:
            loss: total discriminator loss
            loss_dict: dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. MPD loss
        if "mpd" in real_outputs and "mpd" in fake_outputs:
            mpd_loss, mpd_loss_dict = self._mpd_loss(
                real_outputs["mpd"],
                fake_outputs["mpd"]
            )
            total_loss += mpd_loss
            loss_dict.update({f"mpd_{k}": v for k, v in mpd_loss_dict.items()})

        # 2. MSD loss
        if "msd" in real_outputs and "msd" in fake_outputs:
            msd_loss, msd_loss_dict = self._msd_loss(
                real_outputs["msd"],
                fake_outputs["msd"]
            )
            total_loss += msd_loss
            loss_dict.update({f"msd_{k}": v for k, v in msd_loss_dict.items()})

        # 3. Source-filter discriminator loss
        if "source_filter" in real_outputs and "source_filter" in fake_outputs:
            sf_real = real_outputs["source_filter"]
            sf_fake = fake_outputs["source_filter"]
            if sf_real is not None and sf_fake is not None:
                sf_loss, sf_loss_dict = self._source_filter_loss(sf_real, sf_fake)
                total_loss += sf_loss
                loss_dict.update({f"source_filter_{k}": v for k, v in sf_loss_dict.items()})

        # 4. Consistency discriminator loss
        if "consistency" in real_outputs and "consistency" in fake_outputs:
            cc_real = real_outputs["consistency"]
            cc_fake = fake_outputs["consistency"]
            if cc_real is not None and cc_fake is not None:
                cc_loss, cc_loss_dict = self._consistency_loss(cc_real, cc_fake)
                total_loss += cc_loss
                loss_dict.update({f"consistency_{k}": v for k, v in cc_loss_dict.items()})

        return total_loss, loss_dict

    def generator_loss(self,
                      fake_outputs: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute generator adversarial loss.
        Args:
            fake_outputs: outputs from discriminator for fake data
        Returns:
            loss: total generator loss
            loss_dict: dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. MPD loss
        if "mpd" in fake_outputs:
            mpd_loss, mpd_loss_dict = self._mpd_generator_loss(fake_outputs["mpd"])
            total_loss += mpd_loss
            loss_dict.update({f"mpd_{k}": v for k, v in mpd_loss_dict.items()})

        # 2. MSD loss
        if "msd" in fake_outputs:
            msd_loss, msd_loss_dict = self._msd_generator_loss(fake_outputs["msd"])
            total_loss += msd_loss
            loss_dict.update({f"msd_{k}": v for k, v in msd_loss_dict.items()})

        # 3. Source-filter discriminator loss
        if "source_filter" in fake_outputs:
            sf_fake = fake_outputs["source_filter"]
            if sf_fake is not None:
                sf_loss, sf_loss_dict = self._source_filter_generator_loss(sf_fake)
                total_loss += sf_loss
                loss_dict.update({f"source_filter_{k}": v for k, v in sf_loss_dict.items()})

        # 4. Consistency discriminator loss
        if "consistency" in fake_outputs:
            cc_fake = fake_outputs["consistency"]
            if cc_fake is not None:
                cc_loss, cc_loss_dict = self._consistency_generator_loss(cc_fake)
                total_loss += cc_loss
                loss_dict.update({f"consistency_{k}": v for k, v in cc_loss_dict.items()})

        return total_loss, loss_dict

    # Helper methods for MPD
    def _mpd_loss(self,
                  real_outputs: dict,
                  fake_outputs: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MPD discriminator loss."""
        loss_dict = {}
        total_loss = 0.0

        real_outs = real_outputs["outputs"]
        fake_outs = fake_outputs["outputs"]

        for i, (real_out, fake_out) in enumerate(zip(real_outs, fake_outs)):
            # Discriminator wants real_out -> 1, fake_out -> 0
            real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
            fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))
            period_loss = real_loss + fake_loss
            total_loss += period_loss
            loss_dict[f"period_{i}"] = period_loss.item()

        return total_loss, loss_dict

    def _mpd_generator_loss(self,
                           fake_outputs: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MPD generator loss."""
        loss_dict = {}
        total_loss = 0.0

        fake_outs = fake_outputs["outputs"]

        for i, fake_out in enumerate(fake_outs):
            # Generator wants fake_out -> 1
            loss = F.mse_loss(fake_out, torch.ones_like(fake_out))
            total_loss += loss
            loss_dict[f"period_{i}"] = loss.item()

        return total_loss, loss_dict

    # Helper methods for MSD
    def _msd_loss(self,
                  real_outputs: dict,
                  fake_outputs: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MSD discriminator loss."""
        loss_dict = {}
        total_loss = 0.0

        real_outs = real_outputs["outputs"]
        fake_outs = fake_outputs["outputs"]

        for i, (real_out, fake_out) in enumerate(zip(real_outs, fake_outs)):
            real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
            fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))
            scale_loss = real_loss + fake_loss
            total_loss += scale_loss
            loss_dict[f"scale_{i}"] = scale_loss.item()

        return total_loss, loss_dict

    def _msd_generator_loss(self,
                           fake_outputs: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MSD generator loss."""
        loss_dict = {}
        total_loss = 0.0

        fake_outs = fake_outputs["outputs"]

        for i, fake_out in enumerate(fake_outs):
            # Generator wants fake_out -> 1
            loss = F.mse_loss(fake_out, torch.ones_like(fake_out))
            total_loss += loss
            loss_dict[f"scale_{i}"] = loss.item()

        return total_loss, loss_dict

    # Helper methods for source-filter discriminator
    def _source_filter_loss(self,
                           real_out: torch.Tensor,
                           fake_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute source-filter discriminator loss."""
        real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
        fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))
        total_loss = real_loss + fake_loss
        loss_dict = {
            "real": real_loss.item(),
            "fake": fake_loss.item()
        }
        return total_loss, loss_dict

    def _source_filter_generator_loss(self,
                                     fake_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute source-filter generator loss."""
        loss = F.mse_loss(fake_out, torch.ones_like(fake_out))
        loss_dict = {"value": loss.item()}
        return loss, loss_dict

    # Helper methods for consistency discriminator
    def _consistency_loss(self,
                         real_out: torch.Tensor,
                         fake_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute consistency discriminator loss."""
        real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
        fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))
        total_loss = real_loss + fake_loss
        loss_dict = {
            "real": real_loss.item(),
            "fake": fake_loss.item()
        }
        return total_loss, loss_dict

    def _consistency_generator_loss(self,
                                   fake_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute consistency generator loss."""
        loss = F.mse_loss(fake_out, torch.ones_like(fake_out))
        loss_dict = {"value": loss.item()}
        return loss, loss_dict