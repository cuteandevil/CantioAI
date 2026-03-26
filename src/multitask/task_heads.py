"""
Task-specific heads for multi-task learning framework.
Implements prediction heads for singing-to-singing conversion,
speech-to-speech conversion, and noise robustness learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SingingConversionHead(nn.Module):
    """
    Task-specific head for singing-to-singing conversion.
    Predicts acoustic parameters (F0, SP, AP) for singing voice conversion.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        predict_f0: bool = True,
        predict_sp: bool = True,
        predict_ap: bool = True,
        sp_dim: int = 60,
        use_periodicity: bool = True
    ):
        """
        Initialize singing conversion head.

        Args:
            shared_dim: Dimension of shared features from encoder
            hidden_dim: Hidden dimension for prediction layers
            num_layers: Number of prediction layers
            dropout: Dropout rate
            predict_f0: Whether to predict F0 (fundamental frequency)
            predict_sp: Whether to predict SP (spectral envelope)
            predict_ap: Whether to predict AP (aperiodicity)
            sp_dim: Dimension of spectral envelope
            use_periodicity: Whether to use periodicity-aware modeling
        """
        super().__init__()

        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.predict_f0 = predict_f0
        self.predict_sp = predict_sp
        self.predict_ap = predict_ap
        self.sp_dim = sp_dim
        self.use_periodicity = use_periodicity

        # Build prediction layers
        self.f0_head = None
        self.sp_head = None
        self.ap_head = None

        if predict_f0:
            self.f0_head = self._build_prediction_head(shared_dim, hidden_dim, 1, num_layers, dropout)
        if predict_sp:
            self.sp_head = self._build_prediction_head(shared_dim, hidden_dim, sp_dim, num_layers, dropout)
        if predict_ap:
            self.ap_head = self._build_prediction_head(shared_dim, hidden_dim, 1, num_layers, dropout)  # AP is typically 1-dim per frequency bin, but we'll predict frame-level

        # Periodicity-aware components (optional)
        if use_periodicity and predict_f0:
            self.periodicity_predictor = nn.Sequential(
                nn.Linear(shared_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.periodicity_predictor = None

        logger.info(
            f"Initialized SingingConversionHead:\n"
            f"  Shared dim: {shared_dim}\n"
            f"  Predict F0: {predict_f0}\n"
            f"  Predict SP: {predict_sp} (dim: {sp_dim})\n"
            f"  Predict AP: {predict_ap}\n"
            f"  Use periodicity: {use_periodicity}"
        )

    def _build_prediction_head(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float
    ) -> nn.Module:
        """Build a prediction head with multiple layers."""
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:  # No activation/dropout on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for singing conversion head.

        Args:
            shared_features: Shared representation tensor (B, T, shared_dim) or (B, shared_dim)

        Returns:
            Dictionary containing predictions:
                - f0: Fundamental frequency predictions (B, T, 1) or (B, 1)
                - sp: Spectral envelope predictions (B, T, sp_dim) or (B, sp_dim)
                - ap: Aperiodicity predictions (B, T, 1) or (B, 1)
                - periodicity: Periodicity score (B, T, 1) or (B, 1) if enabled
        """
        # Handle both sequence and frame-level inputs
        if shared_features.dim() == 3:
            # Sequence input: (B, T, shared_dim)
            B, T, _ = shared_features.shape
            output_shape = (B, T)
        elif shared_features.dim() == 2:
            # Frame-level input: (B, shared_dim)
            B, _ = shared_features.shape
            T = 1
            output_shape = (B,)
        else:
            raise ValueError(f"Unsupported input shape: {shared_features.shape}")

        outputs = {}

        # Predict F0
        if self.f0_head is not None:
            f0_output = self.f0_head(shared_features)
            if self.f0_head[-1].out_features == 1:  # If final layer outputs 1 dim
                f0_output = f0_output.unsqueeze(-1)  # (B, T, 1) or (B, 1, 1)
            outputs["f0"] = f0_output

        # Predict SP
        if self.sp_head is not None:
            sp_output = self.sp_head(shared_features)
            if self.sp_head[-1].out_features == self.sp_dim:
                sp_output = sp_output.view(*output_shape, self.sp_dim)
            outputs["sp"] = sp_output

        # Predict AP
        if self.ap_head is not None:
            ap_output = self.ap_head(shared_features)
            if self.ap_head[-1].out_features == 1:
                ap_output = ap_output.unsqueeze(-1)  # (B, T, 1) or (B, 1, 1)
            outputs["ap"] = ap_output

        # Predict periodicity (if enabled)
        if self.periodicity_predictor is not None:
            periodicity = self.periodicity_predictor(shared_features)
            if periodicity.shape[-1] == 1:
                periodicity = periodicity.unsqueeze(-1)  # (B, T, 1) or (B, 1, 1)
            outputs["periodicity"] = periodicity

        return outputs


class SpeechConversionHead(nn.Module):
    """
    Task-specific head for speech-to-speech conversion.
    Predicts acoustic parameters for speech voice conversion.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        predict_f0: bool = True,
        predict_sp: bool = True,
        predict_ap: bool = False,  # AP less critical for speech
        sp_dim: int = 60,
        use_speaker_embedding: bool = True,
        speaker_embed_dim: int = 128
    ):
        """
        Initialize speech conversion head.

        Args:
            shared_dim: Dimension of shared features from encoder
            hidden_dim: Hidden dimension for prediction layers
            num_layers: Number of prediction layers
            dropout: Dropout rate
            predict_f0: Whether to predict F0 (fundamental frequency)
            predict_sp: Whether to predict SP (spectral envelope)
            predict_ap: Whether to predict AP (aperiodicity)
            sp_dim: Dimension of spectral envelope
            use_speaker_embedding: Whether to use speaker embeddings
            speaker_embed_dim: Dimension of speaker embeddings
        """
        super().__init__()

        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.predict_f0 = predict_f0
        self.predict_sp = predict_sp
        self.predict_ap = predict_ap
        self.sp_dim = sp_dim
        self.use_speaker_embedding = use_speaker_embedding
        self.speaker_embed_dim = speaker_embed_dim

        # Build prediction layers
        self.f0_head = None
        self.sp_head = None
        self.ap_head = None

        if predict_f0:
            self.f0_head = self._build_prediction_head(shared_dim, hidden_dim, 1, num_layers, dropout)
        if predict_sp:
            self.sp_head = self._build_prediction_head(shared_dim, hidden_dim, sp_dim, num_layers, dropout)
        if predict_ap:
            self.ap_head = self._build_prediction_head(shared_dim, hidden_dim, 1, num_layers, dropout)

        # Speaker embedding projection (optional)
        if use_speaker_embedding:
            self.speaker_proj = nn.Linear(speaker_embed_dim, shared_dim)
        else:
            self.speaker_proj = None

        logger.info(
            f"Initialized SpeechConversionHead:\n"
            f"  Shared dim: {shared_dim}\n"
            f"  Predict F0: {predict_f0}\n"
            f"  Predict SP: {predict_sp} (dim: {sp_dim})\n"
            f"  Predict AP: {predict_ap}\n"
            f"  Use speaker embedding: {use_speaker_embedding}"
        )

    def _build_prediction_head(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float
    ) -> nn.Module:
        """Build a prediction head with multiple layers."""
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:  # No activation/dropout on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, shared_features: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for speech conversion head.

        Args:
            shared_features: Shared representation tensor (B, T, shared_dim) or (B, shared_dim)
            speaker_embedding: Optional speaker embedding (B, speaker_embed_dim)

        Returns:
            Dictionary containing predictions:
                - f0: Fundamental frequency predictions (B, T, 1) or (B, 1)
                - sp: Spectral envelope predictions (B, T, sp_dim) or (B, sp_dim)
                - ap: Aperiodicity predictions (B, T, 1) or (B, 1)
        """
        # Handle both sequence and frame-level inputs
        if shared_features.dim() == 3:
            # Sequence input: (B, T, shared_dim)
            B, T, _ = shared_features.shape
            output_shape = (B, T)
        elif shared_features.dim() == 2:
            # Frame-level input: (B, shared_dim)
            B, _ = shared_features.shape
            T = 1
            output_shape = (B,)
        else:
            raise ValueError(f"Unsupported input shape: {shared_features.shape}")

        # Incorporate speaker embedding if provided
        if self.speaker_proj is not None and speaker_embedding is not None:
            # Project speaker embedding to shared dimension and add
            speaker_proj = self.speaker_proj(speaker_embedding)  # (B, shared_dim)
            if shared_features.dim() == 3:
                # Expand to sequence length
                speaker_proj = speaker_proj.unsqueeze(1)  # (B, 1, shared_dim)
                shared_features = shared_features + speaker_proj
            else:
                shared_features = shared_features + speaker_proj

        outputs = {}

        # Predict F0
        if self.f0_head is not None:
            f0_output = self.f0_head(shared_features)
            if self.f0_head[-1].out_features == 1:
                f0_output = f0_output.unsqueeze(-1)
            outputs["f0"] = f0_output

        # Predict SP
        if self.sp_head is not None:
            sp_output = self.sp_head(shared_features)
            if self.sp_head[-1].out_features == self.sp_dim:
                sp_output = sp_output.view(*output_shape, self.sp_dim)
            outputs["sp"] = sp_output

        # Predict AP
        if self.ap_head is not None:
            ap_output = self.ap_head(shared_features)
            if self.ap_head[-1].out_features == 1:
                ap_output = ap_output.unsqueeze(-1)
            outputs["ap"] = ap_output

        return outputs


class NoiseRobustnessHead(nn.Module):
    """
    Task-specific head for noise robustness learning.
    Learns to enhance robustness to noise or predict clean features from noisy inputs.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        predict_mask: bool = True,      # Predict noise mask
        predict_clean_features: bool = True,  # Predict clean acoustic features
        predict_snr: bool = True,       # Predict signal-to-noise ratio
        feature_dim: int = 60,          # Dimension of acoustic features (e.g., SP)
        use_temporal_context: bool = True,
        context_window: int = 5
    ):
        """
        Initialize noise robustness head.

        Args:
            shared_dim: Dimension of shared features from encoder
            hidden_dim: Hidden dimension for prediction layers
            num_layers: Number of prediction layers
            dropout: Dropout rate
            predict_mask: Whether to predict noise mask (0-1 values)
            predict_clean_features: Whether to predict clean acoustic features
            predict_snr: Whether to predict SNR estimate
            feature_dim: Dimension of acoustic features to predict
            use_temporal_context: Whether to use temporal context modeling
            context_window: Window size for temporal context
        """
        super().__init__()

        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.predict_mask = predict_mask
        self.predict_clean_features = predict_clean_features
        self.predict_snr = predict_snr
        self.feature_dim = feature_dim
        self.use_temporal_context = use_temporal_context
        self.context_window = context_window

        # Build prediction heads
        self.mask_head = None
        self.clean_features_head = None
        self.snr_head = None

        if predict_mask:
            self.mask_head = self._build_prediction_head(shared_dim, hidden_dim, 1, num_layers, dropout)
        if predict_clean_features:
            self.clean_features_head = self._build_prediction_head(shared_dim, hidden_dim, feature_dim, num_layers, dropout)
        if predict_snr:
            self.snr_head = self._build_prediction_head(shared_dim, hidden_dim, 1, num_layers, dropout)

        # Temporal context modeling (optional)
        # Note: Actual application happens in forward method based on input dimensions
        if use_temporal_context:
            self.temporal_context = nn.Sequential(
                nn.Conv1d(shared_dim, hidden_dim, context_window, padding=context_window//2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, shared_dim, context_window, padding=context_window//2)
            )
        else:
            self.temporal_context = None

        logger.info(
            f"Initialized NoiseRobustnessHead:\n"
            f"  Shared dim: {shared_dim}\n"
            f"  Predict mask: {predict_mask}\n"
            f"  Predict clean features: {predict_clean_features} (dim: {feature_dim})\n"
            f"  Predict SNR: {predict_snr}\n"
            f"  Use temporal context: {use_temporal_context}"
        )

    def _build_prediction_head(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float
    ) -> nn.Module:
        """Build a prediction head with multiple layers."""
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:  # No activation/dropout on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for noise robustness head.

        Args:
            shared_features: Shared representation tensor (B, T, shared_dim) or (B, shared_dim)

        Returns:
            Dictionary containing predictions:
                - mask: Noise mask (0-1, where 1 = clean, 0 = noisy) (B, T, 1) or (B, 1)
                - clean_features: Predicted clean acoustic features (B, T, feature_dim) or (B, feature_dim)
                - snr: Estimated signal-to-noise ratio (B, T, 1) or (B, 1)
        """
        # Handle both sequence and frame-level inputs
        if shared_features.dim() == 3:
            # Sequence input: (B, T, shared_dim)
            B, T, _ = shared_features.shape
            output_shape = (B, T)
        elif shared_features.dim() == 2:
            # Frame-level input: (B, shared_dim)
            B, _ = shared_features.shape
            T = 1
            output_shape = (B,)
        else:
            raise ValueError(f"Unsupported input shape: {shared_features.shape}")

        # Apply temporal context modeling if enabled
        if self.temporal_context is not None and shared_features.dim() == 3:
            # Convert to (B, shared_dim, T) for Conv1d
            features_for_temporal = shared_features.transpose(1, 2)  # (B, shared_dim, T)
            temporal_features = self.temporal_context(features_for_temporal)  # (B, shared_dim, T)
            # Add residual connection
            shared_features = shared_features + temporal_features.transpose(1, 2)  # Back to (B, T, shared_dim)

        outputs = {}

        # Predict noise mask
        if self.mask_head is not None:
            mask_output = self.mask_head(shared_features)
            mask_output = torch.sigmoid(mask_output)  # Ensure output is in [0, 1]
            if self.mask_head[-1].out_features == 1:
                mask_output = mask_output.unsqueeze(-1)
            outputs["mask"] = mask_output

        # Predict clean features
        if self.clean_features_head is not None:
            clean_output = self.clean_features_head(shared_features)
            if self.clean_features_head[-1].out_features == self.feature_dim:
                clean_output = clean_output.view(*output_shape, self.feature_dim)
            outputs["clean_features"] = clean_output

        # Predict SNR
        if self.snr_head is not None:
            snr_output = self.snr_head(shared_features)
            # SNR can be any real value, no activation needed
            if self.snr_head[-1].out_features == 1:
                snr_output = snr_output.unsqueeze(-1)
            outputs["snr"] = snr_output

        return outputs


def create_task_head(
    task_type: str,
    shared_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create task-specific heads.

    Args:
        task_type: Type of task head ('singing', 'speech', 'noise_robustness')
        shared_dim: Dimension of shared features from encoder
        **kwargs: Additional arguments for specific head types

    Returns:
        Task-specific head module
    """
    if task_type == "singing":
        return SingingConversionHead(shared_dim=shared_dim, **kwargs)
    elif task_type == "speech":
        return SpeechConversionHead(shared_dim=shared_dim, **kwargs)
    elif task_type == "noise_robustness":
        return NoiseRobustnessHead(shared_dim=shared_dim, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}. Supported types: 'singing', 'speech', 'noise_robustness'")


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 10
    shared_dim = 256

    # Create shared features (simulating encoder output)
    shared_features = torch.randn(batch_size, seq_len, shared_dim)

    # Test singing conversion head
    singing_head = SingingConversionHead(
        shared_dim=shared_dim,
        predict_f0=True,
        predict_sp=True,
        predict_ap=True,
        sp_dim=60
    )
    singing_outputs = singing_head(shared_features)
    print("SingingConversionHead outputs:")
    for key, value in singing_outputs.items():
        print(f"  {key}: {value.shape}")

    # Test speech conversion head
    speech_head = SpeechConversionHead(
        shared_dim=shared_dim,
        predict_f0=True,
        predict_sp=True,
        predict_ap=False,
        sp_dim=60
    )
    speech_outputs = speech_head(shared_features)
    print("\nSpeechConversionHead outputs:")
    for key, value in speech_outputs.items():
        print(f"  {key}: {value.shape}")

    # Test noise robustness head
    noise_head = NoiseRobustnessHead(
        shared_dim=shared_dim,
        predict_mask=True,
        predict_clean_features=True,
        predict_snr=True,
        feature_dim=60
    )
    noise_outputs = noise_head(shared_features)
    print("\nNoiseRobustnessHead outputs:")
    for key, value in noise_outputs.items():
        print(f"  {key}: {value.shape}")

    print("\nAll task heads test passed!")