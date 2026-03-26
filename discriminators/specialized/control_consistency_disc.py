"""Control Consistency Discriminator for audio-parameter consistency judgment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AudioFeatureEncoder(nn.Module):
    """Encoder for audio features."""

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


class ControlFeatureEncoder(nn.Module):
    """Encoder for control parameters (F0, SP, AP)."""

    def __init__(self, f0_dim: int, sp_dim: int, ap_dim: int, output_dim: int):
        super().__init__()

        self.f0_encoder = nn.Sequential(
            nn.Conv1d(f0_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(64),
            nn.Conv1d(64, output_dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.sp_encoder = nn.Sequential(
            nn.Conv1d(sp_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(128),
            nn.Conv1d(128, output_dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.ap_encoder = nn.Sequential(
            nn.Conv1d(ap_dim, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(32),
            nn.Conv1d(32, output_dim//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, f0, sp, ap):
        """
        Forward pass.

        Args:
            f0: Fundamental frequency [B, T, 1]
            sp: Spectral envelope [B, T, D_sp]
            ap: Aperiodicity [B, T, 1]

        Returns:
            Encoded control features [B, T, output_dim]
        """
        # Transpose for Conv1d: [B, D, T]
        f0 = f0.transpose(1, 2)
        sp = sp.transpose(1, 2)
        ap = ap.transpose(1, 2)

        f0_encoded = self.f0_encoder(f0)  # [B, output_dim//2, T]
        sp_encoded = self.sp_encoder(sp)  # [B, output_dim//2, T]
        ap_encoded = self.ap_encoder(ap)  # [B, output_dim//4, T]

        # Concatenate features
        combined = torch.cat([f0_encoded, sp_encoded, ap_encoded], dim=1)  # [B, output_dim, T]

        # Transpose back and project
        combined = combined.transpose(1, 2)  # [B, T, output_dim]
        return self.output_proj(combined)


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""

    def __init__(self, query_dim: int, key_dim: int, value_dim: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        """
        Forward pass.

        Args:
            query: Query tensor [B, T_q, D_q]
            key: Key tensor [B, T_k, D_k]
            value: Value tensor [B, T_v, D_v]

        Returns:
            Attended output, attended value, attention weights
        """
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape

        # Linear projections
        Q = self.q_proj(query)  # [B, T_q, D_q]
        K = self.k_proj(key)    # [B, T_k, D_q]
        V = self.v_proj(value)  # [B, T_v, D_q]

        # Reshape for multi-head attention
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T_q, head_dim]
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T_k, head_dim]
        V = V.view(B, T_v, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T_v, head_dim]

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, T_q, T_k]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, T_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, -1)  # [B, T_q, D_q]

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, V.transpose(1, 2).contiguous().view(B, T_v, -1), attn_weights


class ConsistencyMeasurementNetwork(nn.Module):
    """Network for measuring consistency."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, *, input_dim]

        Returns:
            Output tensor [B, *, output_dim]
        """
        original_shape = x.shape
        # Flatten all but last dimension
        x = x.view(-1, original_shape[-1])
        x = self.network(x)
        # Restore original shape except last dimension
        return x.view(*original_shape[:-1], -1)


class MultiScaleConsistencyChecker(nn.Module):
    """Multi-scale consistency checker."""

    def __init__(self, scales: list, feature_dim: int):
        super().__init__()

        self.scales = scales
        self.feature_dim = feature_dim

        # Consistency checkers for each scale
        self.checkers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(feature_dim, 1),
                nn.Sigmoid()
            ) for _ in scales
        ])

    def forward(self, audio_features, control_features):
        """
        Forward pass.

        Args:
            audio_features: Audio features [B, T, D]
            control_features: Control features [B, T, D]

        Returns:
            Multi-scale consistency results
        """
        results = {}

        for i, scale in enumerate(self.scales):
            if scale > 1:
                # Downsample for larger scales
                audio_down = F.avg_pool1d(
                    audio_features.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)

                control_down = F.avg_pool1d(
                    control_features.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
            else:
                audio_down = audio_features
                control_down = control_features

            # Check consistency at this scale
            combined = torch.cat([audio_down, control_down], dim=-1)  # [B, T_down, 2*D]
            consistency = self.checkers[i](combined)  # [B, T_down, 1]

            results[f"scale_{scale}"] = consistency

        return results


class ControlConsistencyDiscriminator(nn.Module):
    """
    Control Consistency Discriminator:
    Ensures generated audio features are consistent with source-filter parameters.

    Design points:
    1. Multi-modal alignment: Alignment between audio features and source-filter parameters
    2. Cross-modal attention: Attention mechanism between audio and parameters
    3. Consistency scoring: Quantitative consistency scoring
    """

    def __init__(self, config: dict):
        super().__init__()

        # Audio feature encoder
        self.audio_encoder = AudioFeatureEncoder(
            input_dim=config.get("audio_feat_dim", 80),  # Mel-spectrogram dimension
            hidden_dims=[128, 256, 512],
            output_dim=256
        )

        # Control feature encoder
        self.control_encoder = ControlFeatureEncoder(
            f0_dim=config.get("f0_dim", 1),
            sp_dim=config.get("sp_dim", 60),
            ap_dim=config.get("ap_dim", 1),
            output_dim=256
        )

        # Cross-modal attention mechanism
        self.cross_modal_attention = CrossModalAttention(
            query_dim=256,
            key_dim=256,
            value_dim=256,
            num_heads=8
        )

        # Consistency measurement network
        self.consistency_network = ConsistencyMeasurementNetwork(
            input_dim=512,  # Audio features + control features concatenated
            hidden_dims=[256, 128, 64],
            output_dim=1
        )

        # Multi-scale consistency checker
        self.multi_scale_consistency = MultiScaleConsistencyChecker(
            scales=[1, 2, 4],
            feature_dim=256
        )

    def forward(self, audio_features, f0, sp, ap, return_attentions=False):
        """
        Evaluate consistency between audio features and source-filter parameters.

        Args:
            audio_features: Audio features [B, T, D_audio]
            f0, sp, ap: Source-filter parameters
            return_attentions: Whether to return attention weights

        Returns:
            Consistency scores and intermediate information
        """
        B, T_a, _ = audio_features.shape
        _, T_c, _ = f0.shape

        # Encode audio features
        audio_encoded = self.audio_encoder(audio_features)  # [B, T_a, 256]

        # Encode control features
        control_encoded = self.control_encoder(f0, sp, ap)  # [B, T_c, 256]

        # Temporal alignment (if lengths differ)
        if T_a != T_c:
            # Linear interpolation alignment
            control_encoded = F.interpolate(
                control_encoded.transpose(1, 2),
                size=T_a,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Cross-modal attention
        attended_audio, attended_control, attention_weights = self.cross_modal_attention(
            query=audio_encoded,
            key=control_encoded,
            value=control_encoded
        )

        # Feature fusion
        fused_features = torch.cat([attended_audio, attended_control], dim=-1)  # [B, T, 512]

        # Consistency scoring
        consistency_scores = self.consistency_network(fused_features)  # [B, T, 1]
        global_consistency = consistency_scores.mean(dim=1)  # [B, 1]

        # Multi-scale consistency check
        multi_scale_results = self.multi_scale_consistency(
            audio_encoded, control_encoded
        )

        result = {
            "consistency_score": global_consistency,
            "frame_scores": consistency_scores,
            "audio_encoded": audio_encoded,
            "control_encoded": control_encoded,
            "multi_scale_consistency": multi_scale_results
        }

        if return_attentions:
            result["attention_weights"] = attention_weights

        return result

    def compute_detailed_consistency(self, audio_features, f0, sp, ap):
        """
        Compute detailed consistency metrics.

        Args:
            audio_features: Audio features [B, T, D_audio]
            f0: Fundamental frequency [B, T, 1]
            sp: Spectral envelope [B, T, D_sp]
            ap: Aperiodicity [B, T, 1]

        Returns:
            Dictionary of detailed consistency metrics
        """
        # 1. Spectral consistency
        spectral_consistency = self._compute_spectral_consistency(audio_features, sp)

        # 2. Fundamental frequency consistency
        pitch_consistency = self._compute_pitch_consistency(audio_features, f0)

        # 3. Voiced/unvoiced consistency
        voiced_unvoiced_consistency = self._compute_voiced_unvoiced_consistency(audio_features, ap)

        # 4. Temporal dynamics consistency
        temporal_dynamics_consistency = self._compute_temporal_dynamics_consistency(
            audio_features, f0, sp, ap
        )

        # Weighted overall consistency
        overall_consistency = (
            spectral_consistency * 0.3 +
            pitch_consistency * 0.4 +
            voiced_unvoiced_consistency * 0.2 +
            temporal_dynamics_consistency * 0.1
        )

        return {
            "spectral_consistency": spectral_consistency,
            "pitch_consistency": pitch_consitency,
            "voiced_unvoiced_consistency": voiced_unvoiced_consistency,
            "temporal_dynamics_consistency": temporal_dynamics_consitency,
            "overall_consistency": overall_consistency
        }

    def _compute_spectral_consistency(self, audio_features, sp):
        """Compute spectral consistency."""
        # Simplified: correlation between audio spectral shape and SP
        # Actual implementation would be more sophisticated
        return torch.ones(audio_features.shape[0], 1, device=audio_features.device) * 0.5

    def _compute_pitch_consistency(self, audio_features, f0):
        """Compute pitch consistency."""
        # Simplified: alignment between audio pitch contours and F0
        return torch.ones(audio_features.shape[0], 1, device=audio_features.device) * 0.5

    def _compute_voiced_unvoiced_consistency(self, audio_features, ap):
        """Compute voiced/unvoiced consistency."""
        # Simplified: consistency between audio voicing and AP
        return torch.ones(audio_features.shape[0], 1, device=audio_features.device) * 0.5

    def _compute_temporal_dynamics_consistency(self, audio_features, f0, sp, ap):
        """Compute temporal dynamics consistency."""
        # Simplified: consistency of temporal dynamics
        return torch.ones(audio_features.shape[0], 1, device=audio_features.device) * 0.5