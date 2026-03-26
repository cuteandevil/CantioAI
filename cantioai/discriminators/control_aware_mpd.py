"""Control-Aware Multi-Period Discriminator (MPD)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class ControlAwareDiscriminatorP(nn.Module):
    """
    Control-Aware Discriminator for a specific period.
    """

    def __init__(self,
                 period: int,
                 control_dim: int = 64,
                 kernel_size: int = 5,
                 leaky_relu_slope: float = 0.1):
        super().__init__()
        self.period = period
        self.control_dim = control_dim
        self.leaky_relu_slope = leaky_relu_slope

        # Convolutional layers
        self.convs = nn.ModuleList()
        # First conv: input channels = (1 + control_dim) * period (for control-aware)
        self.convs.append(nn.Conv1d((1 + control_dim) * period, 32, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(32, 128, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(128, 512, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(512, 1024, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=2))
        self.convs.append(nn.Conv1d(1024, 1024, kernel_size, 1, padding=2))

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
            x: input audio waveform (B, 1, T)
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

        # 2. Apply periodization
        # Reshape to (B, 1+control_dim, T//period, period) -> then to (B, (1+control_dim)*period, T//period)
        b, c, t = x.shape
        if t % self.period != 0:
            # Pad to make divisible by period
            padding = self.period - (t % self.period)
            x = F.pad(x, (0, padding), "constant")
            b, c, t = x.shape

        x = x.view(b, c, t // self.period, self.period)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, period, T//period)
        x = x.view(b, c * self.period, t // self.period)  # (B, C*period, T//period)

        # 3. Apply convolutional layers
        for i, l in enumerate(self.convs):
            x = l(x)
            x = self.activation(x)
            fmap.append(x)

        # 4. Post-processing
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class ControlAwareMPD(nn.Module):
    """
    Control-Aware Multi-Period Discriminator.
    """

    def __init__(self,
                 config: dict):
        super().__init__()
        self.config = config
        self.control_dim = config.get("control_dim", 64)
        self.leaky_relu_slope = config.get("leaky_relu_slope", 0.1)
        self.periods = config.get("periods", [2, 3, 5, 7, 11])

        self.discriminators = nn.ModuleList([
            ControlAwareDiscriminatorP(
                period=period,
                control_dim=self.control_dim,
                leaky_relu_slope=self.leaky_relu_slope
            )
            for period in self.periods
        ])

        # Control projection layer (将F0/SP/AP映射到判别器条件)
        self.control_projector = None  # Will be initialized on first use if needed

    def forward(self,
                x: torch.Tensor,
                f0: Optional[torch.Tensor] = None,
                sp: Optional[torch.Tensor] = None,
                ap: Optional[torch.Tensor] = None,
                return_features: bool = True) -> dict:
        """
        Forward pass.
        Args:
            x: audio waveform (B, 1, T)
            f0: fundamental frequency (B, 1, T) - optional
            sp: spectral envelope (B, sp_dim, T) - optional
            ap: aperiodicity (B, ap_dim, T) - optional
            return_features: whether to return intermediate features
        Returns:
            dict containing:
                - outputs: list of discriminator outputs for each period
                - features: list of feature maps for each period (if return_features=True)
        """
        # Combine control parameters if provided
        control = None
        if f0 is not None and sp is not None and ap is not None:
            # Concatenate F0, SP, AP along channel dimension
            # We assume they are already in the correct temporal resolution
            control = torch.cat([f0, sp, ap], dim=1)  # (B, 1+sp_dim+ap_dim, T)
            # Project to control_dim if necessary
            if control.size(1) != self.control_dim:
                # Use a 1x1 conv to project to control_dim
                if self.control_projector is None:
                    self.control_projector = nn.Conv1d(
                        control.size(1),
                        self.control_dim,
                        1
                    ).to(control.device)
                control = self.control_projector(control)
        elif f0 is not None:
            # If only F0 is provided, we can use it as a simple control
            # But we need to match control_dim
            control = f0.expand(-1, self.control_dim, -1)
        elif sp is not None:
            control = sp.expand(-1, self.control_dim, -1)
        elif ap is not None:
            control = ap.expand(-1, self.control_dim, -1)

        # If still no control, we'll let the discriminator handle it (it will use zeros)
        # But we need to pass control=None to each discriminator

        outputs = []
        features = []

        for disc in self.discriminators:
            out, fmap = disc(x, control)
            outputs.append(out)
            if return_features:
                features.append(fmap)

        return {
            "outputs": outputs,
            "features": features if return_features else None
        }