"""Hybrid Discriminator System."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .control_aware_mpd import ControlAwareMPD
from .control_aware_msd import ControlAwareMSD


class SourceFilterDiscriminator(nn.Module):
    """
    Discriminator for source-filter parameters (F0, SP, AP).
    Judges whether the source-filter parameters are natural.
    """

    def __init__(self,
                 f0_dim: int = 1,
                 sp_dim: int = 60,
                 ap_dim: int = 1,
                 hidden_dim: int = 256):
        super().__init__()
        self.f0_dim = f0_dim
        self.sp_dim = sp_dim
        self.ap_dim = ap_dim
        self.input_dim = f0_dim + sp_dim + ap_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                f0: torch.Tensor,
                sp: torch.Tensor,
                ap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            f0: fundamental frequency (B, f0_dim, T)
            sp: spectral envelope (B, sp_dim, T)
            ap: aperiodicity (B, ap_dim, T)
        Returns:
            output: discriminator score (B, 1, T)
        """
        # Concatenate along channel dimension
        x = torch.cat([f0, sp, ap], dim=1)  # (B, input_dim, T)
        # Transpose to (B, T, input_dim) for linear layers
        x = x.transpose(1, 2)
        x = self.net(x)
        # Transpose back to (B, 1, T)
        x = x.transpose(1, 2)
        return x


class ControlConsistencyDiscriminator(nn.Module):
    """
    Discriminator for consistency between audio features and source-filter parameters.
    """

    def __init__(self,
                 audio_feat_dim: int = 128,
                 control_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        self.audio_feat_dim = audio_feat_dim
        self.control_dim = control_dim
        self.input_dim = audio_feat_dim + control_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                audio_features: torch.Tensor,
                control_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            audio_features: features extracted from audio (B, audio_feat_dim, T)
            control_params: source-filter parameters (B, control_dim, T)
        Returns:
            output: consistency score (B, 1, T)
        """
        # Concatenate along channel dimension
        x = torch.cat([audio_features, control_params], dim=1)  # (B, input_dim, T)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return x


class HybridDiscriminatorSystem(nn.Module):
    """
    Hybrid Discriminator System combining:
    - Multi-Period Discriminator (MPD) for audio quality
    - Multi-Scale Discriminator (MSD) for audio quality at different scales
    - Source-Filter Discriminator for parameter naturalness
    - Control-Consistency Discriminator for audio-parameter alignment
    """

    def __init__(self,
                 config: dict):
        super().__init__()
        self.config = config

        # Audio quality discriminators
        self.mpd = ControlAwareMPD(config.get("mpd", {}))
        self.msd = ControlAwareMSD(config.get("msd", {}))

        # Source-filter parameter discriminator
        sf_config = config.get("source_filter_discriminator", {})
        self.source_filter_disc = SourceFilterDiscriminator(
            f0_dim=sf_config.get("f0_dim", 1),
            sp_dim=sf_config.get("sp_dim", 60),
            ap_dim=sf_config.get("ap_dim", 1),
            hidden_dim=sf_config.get("hidden_dim", 256)
        )

        # Control-consistency discriminator
        cc_config = config.get("control_consistency_discriminator", {})
        self.consistency_disc = ControlConsistencyDiscriminator(
            audio_feat_dim=cc_config.get("audio_feat_dim", 128),
            control_dim=cc_config.get("control_dim", 64),
            hidden_dim=cc_config.get("hidden_dim", 256)
        )

        # Feature extractor for consistency discriminator
        # We'll use a simple CNN to extract features from audio
        self.audio_feat_extractor = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.1)
        )

    def extract_audio_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from audio waveform."""
        return self.audio_feat_extractor(x)

    def forward(self,
                x: torch.Tensor,
                f0: Optional[torch.Tensor] = None,
                sp: Optional[torch.Tensor] = None,
                ap: Optional[torch.Tensor] = None,
                mode: str = "all") -> Dict[str, any]:
        """
        Forward pass.
        Args:
            x: audio waveform (B, 1, T)
            f0: fundamental frequency (B, 1, T) - optional
            sp: spectral envelope (B, sp_dim, T) - optional
            ap: aperiodicity (B, ap_dim, T) - optional
            mode: what to compute
                  "all": all discriminators
                  "audio_only": MPD and MSD
                  "control_only": source_filter and consistency
        Returns:
            dict containing outputs from various discriminators
        """
        results = {}

        if mode in ["all", "audio_only"]:
            # MPD
            mpd_results = self.mpd(x, f0, sp, ap, return_features=True)
            results["mpd"] = mpd_results

            # MSD
            msd_outputs, msd_features = self.msd(x, f0, sp, ap)
            results["msd"] = {
                "outputs": msd_outputs,
                "features": msd_features
            }

        if mode in ["all", "control_only"]:
            if f0 is not None and sp is not None and ap is not None:
                # Source-filter discriminator
                sf_out = self.source_filter_disc(f0, sp, ap)
                results["source_filter"] = sf_out

                # Consistency discriminator
                audio_feats = self.extract_audio_features(x)
                consistency_out = self.consistency_disc(audio_feats, f0, sp, ap)
                results["consistency"] = consistency_out
            else:
                # If control parameters not provided, set to None
                results["source_filter"] = None
                results["consistency"] = None

        return results