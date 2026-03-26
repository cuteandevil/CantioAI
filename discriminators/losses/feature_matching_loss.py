"""Feature Matching Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for hybrid discriminator system.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.loss_type = config.get("loss_type", "l1")  # l1 or l2

    def compute(self,
                real_features: dict,
                fake_features: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute feature matching loss.
        Args:
            real_features: features from real data
            fake_features: features from fake data
        Returns:
            loss: total feature matching loss
            loss_dict: dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. MPD feature matching
        if "mpd" in real_features and "mpd" in fake_features:
            mpd_loss, mpd_loss_dict = self._mpd_feature_matching(
                real_features["mpd"],
                fake_features["mpd"]
            )
            total_loss += mpd_loss
            loss_dict.update({f"mpd_{k}": v for k, v in mpd_loss_dict.items()})

        # 2. MSD feature matching
        if "msd" in real_features and "msd" in fake_features:
            msd_loss, msd_loss_dict = self._msd_feature_matching(
                real_features["msd"],
                fake_features["msd"]
            )
            total_loss += msd_loss
            loss_dict.update({f"msd_{k}": v for k, v in msd_loss_dict.items()})

        return total_loss, loss_dict

    def _mpd_feature_matching(self,
                             real_features: List[List[torch.Tensor]],
                             fake_features: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MPD feature matching loss."""
        loss_dict = {}
        total_loss = 0.0

        for i, (real_fmaps, fake_fmaps) in enumerate(zip(real_features, fake_features)):
            for j, (real_fmap, fake_fmap) in enumerate(zip(real_fmaps, fake_fmaps)):
                if self.loss_type == "l1":
                    diff = F.l1_loss(real_fmap, fake_fmap)
                else:  # l2
                    diff = F.mse_loss(real_fmap, fake_fmap)
                total_loss += diff
                loss_dict[f"period_{i}_layer_{j}"] = diff.item()

        return total_loss, loss_dict

    def _msd_feature_matching(self,
                             real_features: List[List[torch.Tensor]],
                             fake_features: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MSD feature matching loss."""
        loss_dict = {}
        total_loss = 0.0

        for i, (real_fmaps, fake_fmaps) in enumerate(zip(real_features, fake_features)):
            for j, (real_fmap, fake_fmap) in enumerate(zip(real_fmaps, fake_fmaps)):
                if self.loss_type == "l1":
                    diff = F.l1_loss(real_fmap, fake_fmap)
                else:  # l2
                    diff = F.mse_loss(real_fmap, fake_fmap)
                total_loss += diff
                loss_dict[f"scale_{i}_layer_{j}"] = diff.item()

        return total_loss, loss_dict