"""Control-Aware Multi-Scale Discriminator (MSD)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class ControlAwareDiscriminatorS(nn.Module):
    """
    Control-Aware Discriminator for a specific scale.
    """

    def __init__(self,
                 scale: int = 1,
                 control_dim: int = 64,
                 kernel_size: int = 5,
                 leaky_relu_slope: float = 0.1):
        super().__init__()
        self.scale = scale
        self.control_dim = control_dim
        self.leaky_relu_slope = leaky_relu_slope

        # Convolutional layers
        self.convs = nn.ModuleList()
        # First conv: input channels = 1 (audio) + control_dim (for control-aware)
        self.convs.append(nn.Conv1d(1 + control_dim, 16, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(16, 64, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(64, 256, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(256, 512, kernel_size, 1, padding=4))
        self.convs.append(nn.Conv1d(512, 1024, kernel_size, 1, padding=8))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=16))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=16))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=16))

        # Post-processing
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)

        # Activation
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self,
                x: torch.Tensor,
                control: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: input audio waveform (B, 1, T) - already downsampled by scale
            control: control parameters (B, control_dim, T_control) - optional
        Returns:
            output: discriminator output (B, 1, T')
            feature_maps: list of intermediate feature maps
        """
        fmap = []

        # 1. Prepare control signal
        if control is not None:
            # Ensure control has the same temporal resolution as x after padding for period
            # We'll interpolate control to match the length of x
            control = F.interpolate(
                control,
                size=x.size(2),
                mode='linear',
                align_corners=False
            )
            # Concatenate audio and control
            x = torch.cat([x, control], dim=1)  # (B, 1+control_dim, T)
        else:
            # If no control, we still need to match the expected input channels
            # We'll use zero control
            control = torch.zeros(
                x.size(0), self.control_dim, x.size(2),
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, control], dim=1)

        # 3. Apply convolutional layers
        for i, l in enumerate(self.convs):
            x = l(x)
            x = self.activation(x)
            fmap.append(x)

        # 4. Post-processing
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class ControlAwareMSD(nn.Module):
    """
    Control-Aware Multi-Scale Discriminator.
    """

    def __init__(self,
                 config: dict):
        super().__init__()
        self.config = config
        self.control_dim = config.get("control_dim", 64)
        self.leaky_relu_slope = config.get("leaky_relu_slope", 0.1)
        self.scales = config.get("scales", [1, 2, 4])

        self.discriminators = nn.ModuleList([
            ControlAwareDiscriminatorS(
                scale=scale,
                control_dim=self.control_dim,
                leaky_relu_slope=self.leaky_relu_slope
            )
            for scale in self.scales
        ])

        # Downsampling layers for each scale (except scale=1)
        self.downsamples = nn.ModuleList([
            nn.AvgPool1d(scale) if scale > 1 else nn.Identity()
            for scale in self.scales
        ])

    def forward(self,
                x: torch.Tensor,
                f0: Optional[torch.Tensor] = None,
                sp: Optional[torch.Tensor] = None,
                ap: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass.
        Args:
            x: audio waveform (B, 1, T)
            f0: fundamental frequency (B, 1, T) - optional
            sp: spectral envelope (B, sp_dim, T) - optional
            ap: aperiodicity (B, ap_dim, T) - optional
        Returns:
            outputs: list of discriminator outputs for each scale
            features: list of feature maps for each scale
        """
        # Combine control parameters
        control = None
        if f0 is not None and sp is not None and ap is not None:
            control = torch.cat([f0, sp, ap], dim=1)
            if control.size(1) != self.control_dim:
                if not hasattr(self, 'control_projector'):
                    self.control_projector = nn.Conv1d(
                        control.size(1),
                        self.control_dim,
                        1
                    ).to(control.device)
                control = self.control_projector(control)
        elif f0 is not None:
            control = f0.expand(-1, self.control_dim, -1)
        elif sp is not None:
            control = sp.expand(-1, self.control_dim, -1)
        elif ap is not None:
            control = ap.expand(-1, self.control_dim, -1)

        outputs = []
        features = []

        for i, disc in enumerate(self.discriminators):
            # Downsample input for this scale
            x_down = self.downsamples[i](x)
            out, fmap = disc(x_down, control)
            outputs.append(out)
            features.append(fmap)

        return outputs, features