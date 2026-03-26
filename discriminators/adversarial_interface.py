"""Adversarial Training Interface."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .hybrid_discriminator import HybridDiscriminatorSystem


class AdversarialInterface:
    """
    Interface for adversarial training in CantioAI.
    Handles discriminator and generator losses for the hybrid SVC system.
    """

    def __init__(self,
                 discriminator: HybridDiscriminatorSystem,
                 config: dict):
        """
        Initialize the adversarial interface.
        Args:
            discriminator: the hybrid discriminator system
            config: configuration for adversarial training
        """
        self.discriminator = discriminator
        self.config = config

        # Loss functions
        from .losses.adversarial_loss import AdversarialLoss
        from .losses.feature_matching_loss import FeatureMatchingLoss
        from .losses.control_consistency_loss import ControlConsistencyLoss

        self.adversarial_loss = AdversarialLoss(config.get("adversarial_loss", {}))
        self.feature_matching_loss = FeatureMatchingLoss(config.get("feature_matching_loss", {}))
        self.control_consistency_loss = ControlConsistencyLoss(config.get("control_consistency_loss", {}))

    def discriminator_loss(self,
                          real_audio: torch.Tensor,
                          fake_audio: torch.Tensor,
                          f0: Optional[torch.Tensor] = None,
                          sp: Optional[torch.Tensor] = None,
                          ap: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss.
        Args:
            real_audio: real audio waveform (B, 1, T)
            fake_audio: generated audio waveform (B, 1, T)
            f0: fundamental frequency (B, 1, T) - optional
            sp: spectral envelope (B, sp_dim, T) - optional
            ap: aperiodicity (B, ap_dim, T) - optional
        Returns:
            loss: total discriminator loss
            loss_dict: dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Get discriminator outputs for real and fake audio
        real_outputs = self.discriminator(real_audio, f0, sp, ap, mode="all")
        fake_outputs = self.discriminator(fake_audio, f0, sp, ap, mode="all")

        # 1. Adversarial loss from MPD and MSD
        adv_loss, adv_loss_dict = self.adversarial_loss.discriminator_loss(
            real_outputs, fake_outputs
        )
        total_loss += adv_loss
        loss_dict.update(adv_loss_dict)

        # 2. Source-filter discriminator loss (if enabled)
        if self.config.get("use_source_filter_disc", True) and \
           real_outputs.get("source_filter") is not None and \
           fake_outputs.get("source_filter") is not None:
            sf_loss, sf_loss_dict = self.adversarial_loss.source_filter_loss(
                real_outputs["source_filter"],
                fake_outputs["source_filter"]
            )
            total_loss += sf_loss
            loss_dict.update(sf_loss_dict)

        # 3. Consistency discriminator loss (if enabled)
        if self.config.get("use_consistency_disc", True) and \
           real_outputs.get("consistency") is not None and \
           fake_outputs.get("consistency") is not None:
            cc_loss, cc_loss_dict = self.adversarial_loss.consistency_loss(
                real_outputs["consistency"],
                fake_outputs["consistency"]
            )
            total_loss += cc_loss
            loss_dict.update(cc_loss_dict)

        return total_loss, loss_dict

    def generator_loss(self,
                      fake_audio: torch.Tensor,
                      real_audio: torch.Tensor,
                      f0: Optional[torch.Tensor] = None,
                      sp: Optional[torch.Tensor] = None,
                      ap: Optional[torch.Tensor] = None,
                      pred_f0: Optional[torch.Tensor] = None,
                      pred_sp: Optional[torch.Tensor] = None,
                      pred_ap: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute generator loss.
        Args:
            fake_audio: generated audio waveform (B, 1, T)
            real_audio: real audio waveform (B, 1, T) - for feature matching
            f0: fundamental frequency (B, 1, T) - optional
            sp: spectral envelope (B, sp_dim, T) - optional
            ap: aperiodicity (B, ap_dim, T) - optional
            pred_f0: predicted F0 (B, 1, T) - optional
            pred_sp: predicted SP (B, sp_dim, T) - optional
            pred_ap: predicted AP (B, ap_dim, T) - optional
        Returns:
            loss: total generator loss
            loss_dict: dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Get discriminator outputs for fake audio (and real for feature matching)
        fake_outputs = self.discriminator(fake_audio, f0, sp, ap, mode="all")
        real_outputs = self.discriminator(real_audio, f0, sp, ap, mode="all") if self.config.get("use_feature_matching", True) else None

        # 1. Adversarial loss (generator wants discriminator to think fake is real)
        adv_loss, adv_loss_dict = self.adversarial_loss.generator_loss(fake_outputs)
        total_loss += adv_loss
        loss_dict.update(adv_loss_dict)

        # 2. Feature matching loss
        if self.config.get("use_feature_matching", True) and \
           real_outputs is not None and \
           fake_outputs.get("features") is not None and \
           real_outputs.get("features") is not None:
            fm_loss, fm_loss_dict = self.feature_matching_loss.compute(
                real_outputs["features"],
                fake_outputs["features"]
            )
            total_loss += fm_loss
            loss_dict.update(fm_loss_dict)

        # 3. Consistency loss
        if self.config.get("use_control_consistency", True) and \
           fake_outputs.get("consistency") is not None:
            cc_loss, cc_loss_dict = self.control_consistency_loss.compute(
                fake_audio,
                f0, sp, ap,
                pred_f0, pred_sp, pred_ap
            )
            total_loss += cc_loss
            loss_dict.update(cc_loss_dict)

        return total_loss, loss_dict