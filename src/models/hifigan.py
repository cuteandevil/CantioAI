"""
HiFi-GAN Vocoder with Control Parameter Injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block as used in HiFi-GAN."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               stride=1, dilation=dilation,
                               padding=(kernel_size-1)*dilation//2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               stride=1, dilation=1,
                               padding=(kernel_size-1)//2)
        self.norm1 = nn.InstanceNorm1d(channels, affine=True)
        self.norm2 = nn.InstanceNorm1d(channels, affine=True)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + residual


class HiFiGANGenerator(nn.Module):
    """Simplified HiFi-GAN Generator."""
    def __init__(self,
                 in_channels: int = 80,  # mel bins
                 upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
                 upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
                 resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
                 resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = ((1, 1), (3, 5), (7, 11)),
                 gen_base_channels: int = 512):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # Pre-conv
        self.conv_pre = nn.Conv1d(in_channels, gen_base_channels, 7, 1, padding=3)

        # Upsample layers
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                nn.ConvTranspose1d(gen_base_channels // (2**i),
                                   gen_base_channels // (2**(i+1)),
                                   k, u, padding=(k - u)//2)
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsamples)):
            ch = gen_base_channels // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResidualBlock(ch, k, d[0]))
                self.resblocks.append(ResidualBlock(ch, k, d[1]))

        # Post-conv
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: mel-spectrogram (B, in_channels, T)
        Returns:
            audio waveform (B, 1, T')
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = self.upsamples[i](x)
            x = self.activation(x)
            # Apply all residual blocks at this upsampling level
            for j in range(2 * i, 2 * (i + 1)):
                x = self.resblocks[j](x)

        x = self.activation(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class ControlInjectionBlock(nn.Module):
    """Injects control parameters into the generator flow."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1)
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject control via concatenation and convolution.
        Args:
            x: concatenated features (B, in_channels, T)
        Returns:
            processed features (B, out_channels, T)
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ControlAwareHiFiGAN(nn.Module):
    """HiFi-GAN variant that accepts source-filter control parameters."""
    def __init__(self,
                 mel_channels: int = 80,
                 control_dim: int = 3,  # F0, SP, AP (simplified)
                 upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
                 upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
                 resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
                 resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = ((1, 1), (3, 5), (7, 11)),
                 gen_base_channels: int = 512,
                 num_control_layers: int = 4):
        super().__init__()
        self.control_dim = control_dim
        self.num_control_layers = num_control_layers

        # Base HiFi-GAN generator
        self.generator = HiFiGANGenerator(
            in_channels=mel_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            gen_base_channels=gen_base_channels
        )

        # Control injection layers (placed after each upsampling block)
        self.control_injection = nn.ModuleList()
        # Calculate channels at each upsampling stage
        channels = [gen_base_channels // (2**(i+1)) for i in range(len(upsample_rates))]
        for ch in channels:
            self.control_injection.append(
                ControlInjectionBlock(ch + control_dim, ch)
            )

        # Control encoder: maps F0, SP, AP to control signal
        self.control_encoder = nn.Sequential(
            nn.Linear(control_dim, gen_base_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(gen_base_channels // 2, control_dim)
        )

    def forward(self,
                mel: torch.Tensor,
                f0: Optional[torch.Tensor] = None,
                sp: Optional[torch.Tensor] = None,
                ap: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with control parameter injection.
        Args:
            mel: mel-spectrogram (B, mel_channels, T)
            f0: fundamental frequency (B, 1, T) - optional
            sp: spectral envelope (B, sp_dim, T) - optional
            ap: aperiodicity (B, ap_dim, T) - optional
        Returns:
            audio waveform (B, 1, T')
        """
        # Encode control parameters (if provided)
        if f0 is None or sp is None or ap is None:
            # Use neutral control (zeros)
            control_signal = torch.zeros(
                mel.size(0), self.control_dim, mel.size(2)
            ).to(mel.device)
        else:
            # Concatenate and encode control parameters
            control_input = torch.cat([f0, sp, ap], dim=1)  # (B, control_dim, T)
            control_input = control_input.transpose(1, 2)   # (B, T, control_dim)
            control_signal = self.control_encoder(control_input)
            control_signal = control_signal.transpose(1, 2) # (B, control_dim, T)

        # Pre-conv
        x = self.generator.conv_pre(mel)

        # Process through upsampling and residual blocks with control injection
        for i in range(self.generator.num_upsamples):
            x = self.generator.upsamples[i](x)
            x = self.generator.activation(x)

            # Inject control at this level (if we have injection layers for this upsample)
            if i < self.num_control_layers:
                # Upsample control signal to match current resolution
                control_up = F.interpolate(
                    control_signal,
                    size=x.size(2),
                    mode='linear',
                    align_corners=False
                )
                # Concatenate and inject
                x = torch.cat([x, control_up], dim=1)
                x = self.control_injection[i](x)

            # Apply residual blocks at this upsampling level
            for j in range(2 * i, 2 * (i + 1)):
                x = self.generator.resblocks[j](x)

        x = self.generator.activation(x)
        x = self.generator.conv_post(x)
        x = torch.tanh(x)
        return x