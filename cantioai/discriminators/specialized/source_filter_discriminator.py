"""Source Filter Discriminator for specialized parameter judgment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SourceFilterEncoder(nn.Module):
    """Encoder for source-filter parameters."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Conv1d(prev_dim, output_dim, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Encoded features [B, T, output_dim]
        """
        # Transpose for Conv1d: [B, D, T]
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.output_proj(x)

        # Transpose back: [B, T, output_dim]
        return x.transpose(1, 2)


class InteractionNetwork(nn.Module):
    """Network for modeling parameter interactions."""

    def __init__(self, f0_dim: int, sp_dim: int, ap_dim: int, hidden_dim: int):
        super().__init__()

        self.f0_proj = nn.Linear(f0_dim, hidden_dim)
        self.sp_proj = nn.Linear(sp_dim, hidden_dim)
        self.ap_proj = nn.Linear(ap_dim, hidden_dim)

        self.interaction_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, f0_features, sp_features, ap_features):
        """
        Forward pass.

        Args:
            f0_features: F0 encoded features [B, T, f0_dim]
            sp_features: SP encoded features [B, T, sp_dim]
            ap_features: AP encoded features [B, T, ap_dim]

        Returns:
            Interactive features [B, T, hidden_dim]
        """
        # Project to common dimension
        f0_proj = self.f0_proj(f0_features)
        sp_proj = self.sp_proj(sp_features)
        ap_proj = self.ap_proj(ap_features)

        # Concatenate and interact
        combined = torch.cat([f0_proj, sp_proj, ap_proj], dim=-1)
        return self.interaction_layers(combined)


class TemporalModelingNetwork(nn.Module):
    """Network for temporal modeling of parameters."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, bidirectional: bool):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output dimension adjustment for bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(self.output_dim, input_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, input_dim]

        Returns:
            Temporally modeled features [B, T, input_dim]
        """
        lstm_out, _ = self.lstm(x)
        projected = self.output_proj(lstm_out)
        return projected + x  # Residual connection


class SourceFilterDiscriminator(nn.Module):
    """
    Source Filter Discriminator:
    Specialized judge for F0, SP, AP parameter authenticity.

    Design points:
    1. Multi-scale processing: Different scales of source-filter parameters
    2. Parameter interaction modeling: F0, SP, AP interactions
    3. Temporal consistency: Parameter variation patterns over time
    """

    def __init__(self, config: dict):
        super().__init__()

        # Input dimension configuration
        self.f0_dim = config.get("f0_dim", 1)
        self.sp_dim = config.get("sp_dim", 60)  # e.g., MCEP dimension
        self.ap_dim = config.get("ap_dim", 1)

        # Parameter encoders
        self.f0_encoder = SourceFilterEncoder(
            input_dim=self.f0_dim,
            hidden_dims=[32, 64, 128],
            output_dim=128
        )

        self.sp_encoder = SourceFilterEncoder(
            input_dim=self.sp_dim,
            hidden_dims=[128, 256, 512],
            output_dim=512
        )

        self.ap_encoder = SourceFilterEncoder(
            input_dim=self.ap_dim,
            hidden_dims=[16, 32, 64],
            output_dim=64
        )

        # Parameter interaction network
        self.interaction_network = InteractionNetwork(
            f0_dim=128,
            sp_dim=512,
            ap_dim=64,
            hidden_dim=256
        )

        # Temporal modeling network
        self.temporal_network = TemporalModelingNetwork(
            input_dim=256,
            hidden_dim=128,
            num_layers=3,
            bidirectional=True
        )

        # Discriminator head
        self.discriminator_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Auxiliary output: parameter reconstruction quality
        self.reconstruction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3),  # F0, SP, AP reconstruction errors
        )

    def forward(self, f0, sp, ap, return_features=False):
        """
        Forward pass.

        Args:
            f0: Fundamental frequency parameters [B, T, 1]
            sp: Spectral envelope parameters [B, T, D_sp]
            ap: Aperiodicity parameters [B, T, 1]
            return_features: Whether to return intermediate features

        Returns:
            Discriminator results and auxiliary information
        """
        B, T, _ = f0.shape

        # Encode each parameter
        f0_features = self.f0_encoder(f0)  # [B, T, 128]
        sp_features = self.sp_encoder(sp)  # [B, T, 512]
        ap_features = self.ap_encoder(ap)  # [B, T, 64]

        # Parameter interaction
        interactive_features = self.interaction_network(
            f0_features, sp_features, ap_features
        )  # [B, T, 256]

        # Temporal modeling
        temporal_features = self.temporal_network(interactive_features)  # [B, T, 128]

        # Temporal pooling
        pooled_features = temporal_features.mean(dim=1)  # [B, 128]

        # Discriminator output
        discriminator_output = self.discriminator_head(pooled_features)  # [B, 1]

        # Reconstruction quality assessment
        reconstruction_scores = self.reconstruction_head(pooled_features)  # [B, 3]

        result = {
            "discriminator_output": discriminator_output,
            "reconstruction_scores": reconstruction_scores,
            "f0_score": reconstruction_scores[:, 0:1],
            "sp_score": reconstruction_scores[:, 1:2],
            "ap_score": reconstruction_scores[:, 2:3]
        }

        if return_features:
            result.update({
                "f0_features": f0_features,
                "sp_features": sp_features,
                "ap_features": ap_features,
                "interactive_features": interactive_features,
                "temporal_features": temporal_features,
                "pooled_features": pooled_features
            })

        return result

    def compute_parameter_consistency(self, f0, sp, ap, audio_features=None):
        """
        Compute parameter consistency.

        Args:
            f0: Fundamental frequency parameters [B, T, 1]
            sp: Spectral envelope parameters [B, T, D_sp]
            ap: Aperiodicity parameters [B, T, 1]
            audio_features: Audio features (optional), for audio-parameter consistency validation

        Returns:
            Dictionary of consistency metrics
        """
        # 1. Parameter self-consistency (physical consistency between parameters)
        param_consistency = self._compute_parameter_self_consistency(f0, sp, ap)

        # 2. Parameter temporal consistency (smoothness of parameter variation over time)
        temporal_consistency = self._compute_temporal_consistency(f0, sp, ap)

        # 3. If audio features provided, compute audio-parameter consistency
        audio_param_consistency = 0
        if audio_features is not None:
            audio_param_consistency = self._compute_audio_parameter_consistency(
                audio_features, f0, sp, ap
            )

        return {
            "parameter_consistency": param_consistency,
            "temporal_consistency": temporal_consistency,
            "audio_parameter_consistency": audio_param_consistency
        }

    def _compute_parameter_self_consistency(self, f0, sp, ap):
        """Compute parameter self-consistency."""
        # F0-SP consistency: fundamental frequency should align with spectral envelope's main formants
        f0_sp_consistency = self._compute_f0_sp_consistency(f0, sp)

        # SP-AP consistency: relationship between spectral envelope and aperiodicity
        sp_ap_consistency = self._compute_sp_ap_consistency(sp, ap)

        return f0_sp_consistency + 0.5 * sp_ap_consistency

    def _compute_f0_sp_consistency(self, f0, sp):
        """Compute F0-SP consistency."""
        # Simplified implementation: check if F0 is within SP's main energy band
        B, T, D = sp.shape

        # For mel-spectrogram, compute dominant frequency per frame
        # This is a placeholder - actual implementation would be more sophisticated
        return torch.ones(B, 1, device=f0.device) * 0.5

    def _compute_sp_ap_consistency(self, sp, ap):
        """Compute SP-AP consistency."""
        # High aperiodicity typically corresponds to unvoiced or noise portions
        # Unvoiced portions typically have lower spectral envelope energy
        return torch.ones(sp.shape[0], 1, device=sp.device) * 0.5