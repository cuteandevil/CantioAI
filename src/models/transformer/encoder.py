"""
Transformer Encoder for CantioAI
Implements various Transformer architectures to replace CNN+BiLSTM backbone
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Memory-efficient relative positional encoding with bias for Transformer"""

    def __init__(self, d_model: int, max_seq_len: int = 5000, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Relative position bias table (same size as before)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_seq_len - 1), num_heads)
        )

        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Memory-efficient forward pass that computes relative position biases on-the-fly

        Args:
            seq_len: Length of the sequence
        Returns:
            Relative position bias tensor of shape (seq_len, seq_len, num_heads)
        """
        # Create coordinates efficiently
        coords = torch.arange(seq_len, dtype=torch.long, device=self.relative_position_bias_table.device)

        # Compute relative coordinates: (i, j) -> i - j
        # Using broadcasting to avoid storing full matrix
        relative_coords = coords[:, None] - coords[None, :]  # Shape: (seq_len, seq_len)

        # Shift to start from 0 and clamp to valid range [0, 2*max_seq_len-2]
        relative_coords += self.max_seq_len - 1
        relative_coords = relative_coords.clamp(0, 2 * self.max_seq_len - 2)

        # Look up biases for each head
        # Shape: (seq_len, seq_len, num_heads)
        relative_position_bias = self.relative_position_bias_table[relative_coords]

        return relative_position_bias


class ConditionalPositionalEncoding(nn.Module):
    """Conditional positional encoding that incorporates source-filter parameters"""

    def __init__(self, d_model: int, max_seq_len: int = 5000,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Standard positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        # Relative positional encoding
        self.rel_pos_encoding = RelativePositionalEncoding(d_model, max_seq_len, num_heads)
        # Source-filter conditioning projections
        self.f0_proj = nn.Linear(1, d_model)  # F0 conditioning
        self.sp_proj = nn.Linear(1, d_model)  # SP conditioning
        self.ap_proj = nn.Linear(1, d_model)  # AP conditioning
        # Speaker conditioning
        self.spk_proj = nn.Linear(d_model, d_model)  # Speaker embedding conditioning

    def forward(self, x: torch.Tensor, f0: torch.Tensor = None,
                sp: torch.Tensor = None, ap: torch.Tensor = None,
                spk_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            f0: Fundamental frequency of shape (batch_size, seq_len, 1)
            sp: Spectral parameters of shape (batch_size, seq_len, 1)
            ap: Aperiodicity parameters of shape (batch_size, seq_len, 1)
            spk_emb: Speaker embedding of shape (batch_size, d_model) or (batch_size, seq_len, d_model)
        Returns:
            Tensor with conditional positional encoding
        """
        # Add standard positional encoding
        x = self.pos_encoding(x)

        batch_size, seq_len, _ = x.shape

        if f0 is not None:
            f0_cond = self.f0_proj(f0)  # (batch_size, seq_len, d_model)
            x = x + f0_cond

        if sp is not None:
            sp_cond = self.sp_proj(sp)  # (batch_size, seq_len, d_model)
            x = x + sp_cond

        if ap is not None:
            ap_cond = self.ap_proj(ap)  # (batch_size, seq_len, d_model)
            x = x + ap_cond

        if spk_emb is not None:
            if spk_emb.dim() == 2:  # (batch_size, d_model)
                spk_emb = spk_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
            spk_cond = self.spk_proj(spk_emb)  # (batch_size, seq_len, d_model)
            x = x + spk_cond

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                relative_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            relative_pos_bias: Optional relative position bias of shape (seq_len, seq_len, num_heads)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape

        # Linear projections
        Q = self.q_proj(query)  # (batch_size, seq_len, d_model)
        K = self.k_proj(key)    # (batch_size, seq_len, d_model)
        V = self.v_proj(value)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)

        # Add relative position bias if provided
        if relative_pos_bias is not None:
            scores = scores + relative_pos_bias.unsqueeze(0).transpose(1, 3)  # (batch_size, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )  # (batch_size, seq_len, d_model)

        # Final linear projection
        output = self.out_proj(attn_output)

        return output


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer"""

    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: int = 2048,
                 dropout: float = 0.1, use_relative_pos: bool = True):
        super().__init__()
        self.use_relative_pos = use_relative_pos
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if use_relative_pos:
            self.rel_pos_encoding = RelativePositionalEncoding(d_model, num_heads=num_heads)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention block
        residual = x
        x = self.norm1(x)

        # Get relative position bias if needed
        rel_pos_bias = None
        if self.use_relative_pos:
            rel_pos_bias = self.rel_pos_encoding(seq_len)

        x = self.self_attn(x, x, x, mask, rel_pos_bias)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder for replacing CNN+BiLSTM backbone"""

    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, ff_dim: int = 2048,
                 dropout: float = 0.1, max_seq_len: int = 5000,
                 use_relative_pos: bool = True,
                 conditional_pos_encoding: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Input projection (to match expected input dimension)
        self.input_proj = None  # Will be set dynamically based on input

        # Positional encoding
        self.pos_encoding = None
        self.conditional_pos_encoding = None
        if conditional_pos_encoding:
            self.conditional_pos_encoding = ConditionalPositionalEncoding(
                d_model, max_seq_len, num_heads, dropout
            )
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, ff_dim, dropout, use_relative_pos
            ) for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, f0: torch.Tensor = None,
                sp: torch.Tensor = None, ap: torch.Tensor = None,
                spk_emb: torch.Tensor = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            f0: Fundamental frequency of shape (batch_size, seq_len, 1)
            sp: Spectral parameters of shape (batch_size, seq_len, 1)
            ap: Aperiodicity parameters of shape (batch_size, seq_len, 1)
            spk_emb: Speaker embedding of shape (batch_size, speaker_embed_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Initialize input projection if not done yet
        if self.input_proj is None:
            input_dim = x.shape[-1]
            self.input_proj = nn.Linear(input_dim, self.d_model).to(x.device)

        # Project input to model dimension
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        if self.conditional_pos_encoding is not None:
            x = self.conditional_pos_encoding(x, f0, sp, ap, spk_emb)
        elif self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Apply Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        return x


class HierarchicalTransformerEncoder(nn.Module):
    """Hierarchical Transformer Encoder with multi-scale processing"""

    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, ff_dim: int = 2048,
                 dropout: float = 0.1, max_seq_len: int = 5000,
                 local_window: int = 32, medium_window: int = 128,
                 global_window: str = "full",
                 downsampling_factors: list = [2, 4],
                 use_relative_pos: bool = True,
                 conditional_pos_encoding: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.local_window = local_window
        self.medium_window = medium_window
        self.global_window = global_window
        self.downsampling_factors = downsampling_factors

        # Input projection
        self.input_proj = None

        # Positional encoding
        if conditional_pos_encoding:
            self.pos_encoding = ConditionalPositionalEncoding(
                d_model, max_seq_len, num_heads, dropout
            )
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Multi-scale processing layers
        self.local_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, ff_dim, dropout, use_relative_pos
            ) for _ in range(num_layers // 3)
        ])

        self.medium_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, ff_dim, dropout, use_relative_pos
            ) for _ in range(num_layers // 3)
        ])

        self.global_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, ff_dim, dropout, use_relative_pos
            ) for _ in range(num_layers // 3)
        ])

        # Downsampling and upsampling layers
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        prev_dim = d_model
        for factor in downsampling_factors:
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv1d(prev_dim, prev_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm(prev_dim // 2),
                    nn.GELU()
                )
            )
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(prev_dim // 2, prev_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.LayerNorm(prev_dim),
                    nn.GELU()
                )
            )

            prev_dim = prev_dim // 2

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, f0: torch.Tensor = None,
                sp: torch.Tensor = None, ap: torch.Tensor = None,
                spk_emb: torch.Tensor = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            f0: Fundamental frequency of shape (batch_size, seq_len, 1)
            sp: Spectral parameters of shape (batch_size, seq_len, 1)
            ap: Aperiodicity parameters of shape (batch_size, seq_len, 1)
            spk_emb: Speaker embedding of shape (batch_size, speaker_embed_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Initialize input projection if not done yet
        if self.input_proj is None:
            input_dim = x.shape[-1]
            self.input_proj = nn.Linear(input_dim, self.d_model).to(x.device)

        # Project input to model dimension
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, f0, sp, ap, spk_emb)

        # Store original shape for skip connections
        original_x = x

        # Local processing (frame-level features)
        local_x = x
        for layer in self.local_layers:
            local_x = layer(local_x, mask)

        # Downsample for medium-level processing (phoneme-level)
        medium_x = local_x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        medium_mask = None
        if mask is not None:
            # Downsample mask using max pooling
            medium_mask = F.max_pool1d(mask.float().unsqueeze(1),
                                   kernel_size=2, stride=2).squeeze(1) > 0

        for i, layer in enumerate(self.medium_layers):
            medium_x = medium_x.transpose(1, 2)  # (batch_size, d_model, seq_len)
            medium_x = layer(medium_x, medium_mask)

        medium_x = medium_x.transpose(1, 2)  # (batch_size, d_model, seq_len)

        # Apply downsampling (except for last layer)
        if i < len(self.medium_layers) - 1:
            medium_x = self.downsample_layers[i](medium_x)

        # Downsample for global-level processing (sentence-level)
        global_x = medium_x
        for i, layer in enumerate(self.downsample_layers[len(self.medium_layers):]):
            global_x = layer(global_x)

        # Global processing
        for layer in self.global_layers:
            global_x = global_x.transpose(1, 2)  # (batch_size, d_model, seq_len)

        # Global processing
        global_x = layer(global_x, None)  # No mask for global (full sequence)

        global_x = global_x.transpose(1, 2)  # (batch_size, d_model, seq_len)

        # Upsample back to medium resolution
        for i in range(len(self.global_layers)):
            global_x = self.upsample_layers[-(i+1)](global_x)

        # Upsample back to local resolution
        for i in range(len(self.medium_layers)):
            global_x = self.upsample_layers[-(i+1)](global_x)

        # Residual connection
        x = original_x + global_x

        # Final normalization
        x = self.norm(x)

        return x


def create_transformer_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create Transformer encoder based on configuration

    Args:
        config: Configuration dictionary containing transformer settings

    Returns:
        Transformer encoder module
    """
    transformer_config = config.get('model', {}).get('transformer', {})
    encoder_type = transformer_config.get('type', 'standard')

    common_params = {
        'd_model': transformer_config.get('hidden_dim', 512),
        'num_heads': transformer_config.get('num_heads', 8),
        'num_layers': transformer_config.get('num_layers', 6),
        'ff_dim': transformer_config.get('ff_dim', 2048),
        'dropout': transformer_config.get('dropout', 0.1),
        'max_seq_len': transformer_config.get('positional_encoding', {}).get('max_seq_len', 5000),
        'use_relative_pos': transformer_config.get('positional_encoding', {}).get('type', 'relative_bias') == 'relative_bias',
        'conditional_pos_encoding': True
    }

    if encoder_type == 'standard':
        return TransformerEncoder(**common_params)
    elif encoder_type == 'hierarchical':
        hierarchical_params = {
            'local_window': transformer_config.get('hierarchical', {}).get('local_window', 32),
            'medium_window': transformer_config.get('hierarchical', {}).get('medium_window', 128),
            'global_window': transformer_config.get('hierarchical', {}).get('global_window', 'full'),
            'downsampling_factors': transformer_config.get('hierarchical', {}).get('downsampling_factors', [2, 4])
        }
        common_params.update(hierarchical_params)
        return HierarchicalTransformerEncoder(**common_params)
    elif encoder_type == 'streaming':
        # For streaming, we'd need causal attention - simplified version for now
        streaming_params = {
            'causal': transformer_config.get('streaming', {}).get('causal', True)
        }
        common_params.update(streaming_params)
        return TransformerEncoder(**common_params)
    else:
        raise ValueError(f"Unknown transformer type: {encoder_type}")