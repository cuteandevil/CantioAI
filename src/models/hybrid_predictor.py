"""
Hybrid Spectral Predictor Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HybridSpectralPredictor(nn.Module):
    """
    Hybrid source-filter + neural vocoder spectral envelope predictor.

    Predicts high-resolution spectral envelope (e.g., 60-dim MCEP) from:
        - phoneme features (B, T, D_ph)
        - normalized fundamental frequency f0 (B, T, 1)
        - speaker ID (B,)

    The speaker embedding is made adaptive via InstanceNorm1d injected into the
    generator (here we simply condition the LSTM input with the embedded speaker
    vector; more sophisticated AdaIN can be added similarly).

    Architecture:
        1. Speaker ID -> Embedding -> (B, D_spk)
        2. Expand speaker embedding to time axis and concat with phoneme features
           and f0 -> (B, T, D_ph + 1 + D_spk)
        3. 1D convolutional layers to capture local context.
        4. Bidirectional LSTM to model temporal dependencies.
        5. Fully-connected layers to predict spectral envelope of dimension D_sp.
    """

    def __init__(
        self,
        D_ph: int,          # phoneme feature dimension
        D_sp: int = 60,     # output spectral envelope dimension (e.g., MCEP)
        D_spk: int = 128,   # speaker embedding dimension
        n_speakers: int = 100,  # total number of speakers in the dataset
        conv_channels: int = 256,
        conv_kernel_size: int = 5,
        lstm_hidden: int = 256,
        lstm_num_layers: int = 2,
        fc_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Speaker embedding table
        self.spk_embed = nn.Embedding(n_speakers, D_spk)

        # 1D conv expects input shape (B, C, T) -> we will transpose after concat
        self.conv1 = nn.Conv1d(
            in_channels=D_ph + 1 + D_spk,
            out_channels=conv_channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
        )

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Fully-connected layers
        self.fc1 = nn.Linear(lstm_out_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, D_sp)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        phoneme_features: torch.Tensor,  # (B, T, D_ph)
        f0: torch.Tensor,                # (B, T, 1)  normalized f0
        spk_id: torch.Tensor,            # (B,)      speaker IDs
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            sp_pred: (B, T, D_sp) predicted spectral envelope.
        """
        B, T, _ = phoneme_features.shape

        # 1. Speaker embedding -> (B, D_spk)
        spk_emb = self.spk_embed(spk_id)          # (B, D_spk)

        # 2. Expand to time axis and concatenate
        spk_emb_expanded = spk_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_spk)
        # Concatenate along feature dim
        x = torch.cat([phoneme_features, f0, spk_emb_expanded], dim=-1)  # (B, T, D_ph+1+D_spk)

        # 3. Convert to (B, C, T) for Conv1d
        x = x.transpose(1, 2)  # (B, C, T)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Back to (B, T, C) for LSTM
        x = x.transpose(1, 2)  # (B, T, C)

        # 4. Bi-LSTM
        x, _ = self.lstm(x)  # (B, T, 2*lstm_hidden)

        # 5. Fully-connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        sp_pred = self.fc2(x)  # (B, T, D_sp)

        return sp_pred