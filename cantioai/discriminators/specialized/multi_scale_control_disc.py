"""Multi-Scale Control Discriminator for enhanced control-aware discrimination."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiScaleControlDiscriminator(nn.Module):
    """
    Multi-Scale Control Discriminator:
    Discriminator that operates at multiple audio resolutions with control awareness.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.scales = config.get("scales", [1, 2, 4])
        self.control_dim = config.get("control_dim", 64)
        self.leaky_relu_slope = config.get("leaky_relu_slope", 0.1)

        # Discriminators for each scale
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(
                scale=scale,
                control_dim=self.control_dim,
                leaky_relu_slope=self.leaky_relu_slope
            ) for scale in self.scales
        ])

    def forward(self, audio, f0=None, sp=None, ap=None):
        """
        Forward pass through multi-scale discriminators.

        Args:
            audio: Audio waveform [B, 1, T]
            f0: Fundamental frequency [B, 1, T'] (optional)
            sp: Spectral envelope [B, sp_dim, T'] (optional)
            ap: Aperiodicity [B, ap_dim, T'] (optional)

        Returns:
            Outputs and features from each scale
        """
        outputs = []
        features = []

        for i, scale in enumerate(self.scales):
            if scale > 1:
                # Downsample audio for larger scales
                audio_down = F.avg_pool1d(
                    audio,
                    kernel_size=scale,
                    stride=scale,
                    padding=scale//2
                )
            else:
                audio_down = audio

            # Process through scale-specific discriminator
            scale_output, scale_features = self.discriminators[i](
                audio_down, f0, sp, ap
            )

            outputs.append(scale_output)
            features.append(scale_features)

        return outputs, features


class ScaleDiscriminator(nn.Module):
    """Discriminator for a specific scale."""

    def __init__(self, scale: int, control_dim: int, leaky_relu_slope: float):
        super().__init__()

        self.scale = scale
        self.control_dim = control_dim
        self.leaky_relu_slope = leaky_relu_slope

        # Multi-period discriminator components (similar to MPD but for this scale)
        self.discriminators = nn.ModuleList([
            DiscriminatorSubnet(
                period=period,
                control_dim=control_dim,
                leaky_relu_slope=leaky_relu_slope
            ) for period in [2, 3, 5, 7, 11]  # Standard MPD periods
        ])

    def forward(self, audio, f0=None, sp=None, ap=None):
        """
        Forward pass.

        Args:
            audio: Audio waveform [B, 1, T]
            f0: Fundamental frequency [B, 1, T'] (optional)
            sp: Spectral envelope [B, sp_dim, T'] (optional)
            ap: Aperiodicity [B, ap_dim, T'] (optional)

        Returns:
            Output and features
        """
        outputs = []
        features = []

        for disc in self.discriminators:
            output, feature = disc(audio, f0, sp, ap)
            outputs.append(output)
            features.append(feature)

        return outputs, features


class DiscriminatorSubnet(nn.Module):
    """Subnet for discriminator with control awareness."""

    def __init__(self, period: int, control_dim: int, leaky_relu_slope: float):
        super().__init__()

        self.period = period
        self.control_dim = control_dim
        self.leaky_relu_slope = leaky_relu_slope

        # Convolutional layers
        self.convs = nn.ModuleList()
        # First layer: process raw audio
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
                nn.LeakyReLU(leaky_relu_slope)
            )
        )

        # Subsequent layers with downsampling
        in_channels = 16
        for i in range(1, 5):
            out_channels = min(16 * (2 ** i), 1024)
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=41,
                        stride=4,
                        padding=20,
                        groups=4  # For periodicity
                    ),
                    nn.LeakyReLU(leaky_relu_slope)
                )
            )
            in_channels = out_channels

        # Post-processing layers
        self.post_conv1 = nn.Conv1d(in_channels, 512, kernel_size=5, stride=1, padding=2)
        self.post_conv2 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.conv_out = nn.Conv1d(256, 1, kernel_size=3, stride=1, padding=1)

        # Control processing
        self.control_conv = nn.Conv1d(
            control_dim,
            128,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.control_proj = nn.Linear(128, 256)

    def forward(self, audio, f0=None, sp=None, ap=None):
        """
        Forward pass.

        Args:
            audio: Audio waveform [B, 1, T]
            f0: Fundamental frequency [B, 1, T'] (optional)
            sp: Spectral envelope [B, sp_dim, T'] (optional)
            ap: Aperiodicity [B, ap_dim, T'] (optional)

        Returns:
            Output feature map and intermediate features
        """
        # Process audio through convolutions
        feature_maps = []
        x = audio

        for i, conv in enumerate(self.convs):
            x = conv(x)
            feature_maps.append(x)

        # Process through post-convolutions
        x = F.leaky_relu(self.post_conv1(x), self.leaky_relu_slope)
        feature_maps.append(x)
        x = F.leaky_relu(self.post_conv2(x), self.leaky_relu_slope)
        feature_maps.append(x)

        # Final output
        output = self.conv_out(x)

        # Process control signals if provided
        if f0 is not None and sp is not None and ap is not None:
            # This is a simplified control integration
            # In practice, we would extract control features and inject them
            pass

        return output, feature_maps