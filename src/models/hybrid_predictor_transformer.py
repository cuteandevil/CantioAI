"""
Hybrid Spectral Predictor with Transformer Backbone
Replaces CNN+BiLSTM with Transformer encoder for improved long-term dependency modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .transformer import create_transformer_encoder


class HybridSpectralPredictorTransformer(nn.Module):
    """
    Hybrid source-filter + neural vocoder spectral envelope predictor
    with Transformer backbone replacing CNN+BiLSTM

    Predicts high-resolution spectral envelope (e.g., 60-dim MCEP) from:
        - phoneme features (B, T, D_ph)
        - normalized fundamental frequency f0 (B, T, 1)
        - speaker ID (B,)

    The speaker embedding is made adaptive via InstanceNorm1d injected into the
    generator (here we simply condition the Transformer input with the embedded speaker
    vector; more sophisticated AdaIN can be added similarly).

    Architecture:
        1. Speaker ID -> Embedding -> (B, D_spk)
        2. Expand speaker embedding to time axis and concat with phoneme features
           and f0 -> (B, T, D_ph + 1 + D_spk)
        3. Transformer encoder layers to capture local and global dependencies
        4. Fully-connected layers to predict spectral envelope of dimension D_sp
    """

    def __init__(
        self,
        D_ph: int,          # phoneme feature dimension
        D_sp: int = 60,     # output spectral envelope dimension (e.g., MCEP)
        D_spk: int = 128,   # speaker embedding dimension
        n_speakers: int = 100,  # total number of speakers in the dataset
        # Transformer-specific parameters
        transformer_type: str = "hierarchical",  # standard, hierarchical, streaming
        transformer_hidden_dim: int = 512,
        transformer_num_heads: int = 8,
        transformer_num_layers: int = 6,
        transformer_ff_dim: int = 2048,
        transformer_dropout: float = 0.1,
        transformer_max_seq_len: int = 5000,
        use_relative_pos: bool = True,
        use_conditional_pos_encoding: bool = True,
        # Local processing window sizes (for hierarchical)
        local_window: int = 32,
        medium_window: int = 128,
        global_window: str = "full",
        downsampling_factors: list = [2, 4],
    ):
        super().__init__()
        # Speaker embedding table
        self.spk_embed = nn.Embedding(n_speakers, D_spk)

        # Store dimensions for reference
        self.D_ph = D_ph
        self.D_sp = D_sp
        self.D_spk = D_spk

        # Create Transformer encoder configuration
        transformer_config = {
            'model': {
                'transformer': {
                    'type': transformer_type,
                    'hidden_dim': transformer_hidden_dim,
                    'num_heads': transformer_num_heads,
                    'num_layers': transformer_num_layers,
                    'ff_dim': transformer_ff_dim,
                    'dropout': transformer_dropout,
                    'max_seq_len': transformer_max_seq_len,
                    'positional_encoding': {
                        'type': 'relative_bias' if use_relative_pos else 'standard',
                    },
                    'hierarchical': {
                        'local_window': local_window,
                        'medium_window': medium_window,
                        'global_window': global_window,
                        'downsampling_factors': downsampling_factors,
                    },
                    'streaming': {
                        'causal': True,  # For streaming transformer
                    }
                }
            }
        }

        # Create Transformer encoder
        self.transformer_encoder = create_transformer_encoder(transformer_config)

        # Calculate the concatenated input dimension
        concat_dim = D_ph + 1 + D_spk  # phoneme features + f0 + speaker embedding

        # Input projection to match transformer dimension (if needed)
        self.input_proj = None
        if concat_dim != transformer_hidden_dim:
            self.input_proj = nn.Linear(concat_dim, transformer_hidden_dim)

        # Fully-connected layers for spectral envelope prediction
        self.fc1 = nn.Linear(transformer_hidden_dim, transformer_hidden_dim)
        self.fc2 = nn.Linear(transformer_hidden_dim, D_sp)

        self.dropout = nn.Dropout(transformer_dropout)

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

        # Concatenate along feature dimension: (B, T, D_ph + 1 + D_spk)
        x = torch.cat([phoneme_features, f0, spk_emb_expanded], dim=-1)

        # Project to transformer dimension if needed
        if self.input_proj is not None:
            x = self.input_proj(x)  # (B, T, transformer_hidden_dim)

        # Apply Transformer encoder
        # Note: We don't pass f0, sp, ap separately here since they're already concatenated
        # But we can still pass speaker embedding for conditional encoding if needed
        x = self.transformer_encoder(
            x,
            spk_emb=spk_emb  # Pass speaker embedding for conditional positional encoding
        )  # (B, T, transformer_hidden_dim)

        # Apply fully-connected layers with activation and dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        sp_pred = self.fc2(x)  # (B, T, D_sp)

        return sp_pred


def create_hybrid_predictor(
    predictor_type: str = "transformer",
    **kwargs
) -> nn.Module:
    """
    Factory function to create hybrid spectral predictor

    Args:
        predictor_type: Type of predictor ('cnn_lstm' or 'transformer')
        **kwargs: Additional arguments passed to predictor constructor

    Returns:
        Hybrid spectral predictor module
    """
    if predictor_type.lower() == "transformer":
        return HybridSpectralPredictorTransformer(**kwargs)
    elif predictor_type.lower() == "cnn_lstm":
        # Import and return the original CNN+BiLSTM version
        from .hybrid_predictor import HybridSpectralPredictor
        return HybridSpectralPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")