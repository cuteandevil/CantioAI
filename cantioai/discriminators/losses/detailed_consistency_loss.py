"""Enhanced Consistency Loss with detailed consistency measurements."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class DetailedConsistencyLoss(nn.Module):
    """
    Enhanced Consistency Loss:
    Computes detailed consistency metrics between audio and control parameters
    using multiple consistency dimensions.

    Design points:
    1. Multi-dimensional consistency: Spectral, pitch, voiced/unvoiced, temporal dynamics
    2. Weighted combination: Different consistency aspects have different importance
    3. Feature-based consistency: Uses audio features for deeper consistency checking
    4. Temporal consistency: Consistency over time sequences
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.loss_type = config.get("loss_type", "l1")

        # Consistency component weights
        self.consistency_weights = {
            "spectral": config.get("spectral_weight", 0.3),
            "pitch": config.get("pitch_weight", 0.4),
            "voiced_unvoiced": config.get("voiced_unvoiced_weight", 0.2),
            "temporal_dynamics": config.get("temporal_dynamics_weight", 0.1)
        }

    def compute(self,
                fake_audio: torch.Tensor,
                f0: Optional[torch.Tensor] = None,
                sp: Optional[torch.Tensor] = None,
                ap: Optional[torch.Tensor] = None,
                pred_f0: Optional[torch.Tensor] = None,
                pred_sp: Optional[torch.Tensor] = None,
                pred_ap: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute enhanced consistency loss.

        Args:
            fake_audio: Generated audio waveform [B, 1, T]
            f0: Fundamental frequency [B, 1, T] - optional
            sp: Spectral envelope [B, sp_dim, T] - optional
            ap: Aperiodicity [B, ap_dim, T] - optional
            pred_f0: Predicted F0 from generator [B, 1, T] - optional
            pred_sp: Predicted SP from generator [B, sp_dim, T] - optional
            pred_ap: Predicted AP from generator [B, ap_dim, T] - optional

        Returns:
            Total consistency loss and dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Extract audio features for consistency computation
        # In practice, this would use a pre-trained encoder or discriminator features
        audio_features = self._extract_audio_features(fake_audio)

        # 1. Spectral consistency
        if sp is not None and pred_sp is not None:
            spectral_loss = self._compute_spectral_consistency_loss(
                audio_features, sp, pred_sp
            )
            weighted_loss = self.consistency_weights["spectral"] * spectral_loss
            total_loss += weighted_loss
            loss_dict["spectral_consistency"] = spectral_loss.item()
            loss_dict["weighted_spectral"] = weighted_loss.item()

        # 2. Pitch consistency
        if f0 is not None and pred_f0 is not None:
            pitch_loss = self._compute_pitch_consistency_loss(
                audio_features, f0, pred_f0
            )
            weighted_loss = self.consistency_weights["pitch"] * pitch_loss
            total_loss += weighted_loss
            loss_dict["pitch_consistency"] = pitch_loss.item()
            loss_dict["weighted_pitch"] = weighted_loss.item()

        # 3. Voiced/unvoiced consistency
        if ap is not None and pred_ap is not None:
            vuv_loss = self._compute_voiced_unvoiced_consistency_loss(
                audio_features, ap, pred_ap
            )
            weighted_loss = self.consistency_weights["voiced_unvoiced"] * vuv_loss
            total_loss += weighted_loss
            loss_dict["voiced_unvoiced_consistency"] = vuv_loss.item()
            loss_dict["weighted_voiced_unvoiced"] = weighted_loss.item()

        # 4. Temporal dynamics consistency
        if f0 is not None and sp is not None and ap is not None:
            temp_loss = self._compute_temporal_dynamics_consistency_loss(
                audio_features, f0, sp, ap, pred_f0, pred_sp, pred_ap
            )
            weighted_loss = self.consistency_weights["temporal_dynamics"] * temp_loss
            total_loss += weighted_loss
            loss_dict["temporal_dynamics_consistency"] = temp_loss.item()
            loss_dict["weighted_temporal_dynamics"] = weighted_loss.item()

        # 5. Direct parameter consistency (fallback)
        param_loss, param_loss_dict = self._compute_direct_parameter_consistency(
            f0, sp, ap, pred_f0, pred_sp, pred_ap
        )
        total_loss += param_loss
        loss_dict.update(param_loss_dict)

        return total_loss, loss_dict

    def _extract_audio_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio for consistency computation.
        In practice, this would use a pre-trained encoder.
        """
        # Simple feature extraction using spectrogram-like operations
        # This is a placeholder - actual implementation would use mel-spectrogram etc.
        B, _, T = audio.shape

        # Compute power spectrum (simplified)
        # In reality, we'd use STFT or mel-spectrogram
        # For now, just return a feature tensor of appropriate shape
        feature_dim = 80  # Typical mel-spectrogram dimension
        time_steps = T // 16  # Typical downsampling factor

        # Create dummy features (in practice, this would be learned)
        features = torch.randn(B, feature_dim, time_steps, device=audio.device)
        return features

    def _compute_spectral_consistency_loss(self, audio_features: torch.Tensor,
                                         sp: torch.Tensor,
                                         pred_sp: torch.Tensor) -> torch.Tensor:
        """Compute spectral consistency loss."""
        # Align temporal dimensions
        min_time = min(audio_features.shape[-1], sp.shape[-1], pred_sp.shape[-1])
        audio_aligned = audio_features[..., :min_time]
        sp_aligned = sp[..., :min_time]
        pred_sp_aligned = pred_sp[..., :min_time]

        # Compute consistency between audio features and predicted SP
        # vs audio features and target SP
        if self.loss_type == "l1":
            pred_consistency = F.l1_loss(audio_aligned, pred_sp_aligned)
            target_consistency = F.l1_loss(audio_aligned, sp_aligned)
        else:  # l2
            pred_consistency = F.mse_loss(audio_aligned, pred_sp_aligned)
            target_consistency = F.mse_loss(audio_aligned, sp_aligned)

        # Loss encourages predicted consistency to match target consistency
        consistency_loss = F.l1_loss(pred_consistency, target_consistency)
        return consistency_loss

    def _compute_pitch_consistency_loss(self, audio_features: torch.Tensor,
                                      f0: torch.Tensor,
                                      pred_f0: torch.Tensor) -> torch.Tensor:
        """Compute pitch consistency loss."""
        # Align temporal dimensions
        min_time = min(audio_features.shape[-1], f0.shape[-1], pred_f0.shape[-1])
        audio_aligned = audio_features[..., :min_time]
        f0_aligned = f0[..., :min_time]
        pred_f0_aligned = pred_f0[..., :min_time]

        # Compute pitch-related features from audio (simplified)
        # In practice, we'd extract pitch contour from audio
        audio_pitch_features = torch.sin(audio_features.mean(dim=1, keepdim=True))  # Placeholder

        # Compare pitch consistency
        if self.loss_type == "l1":
            pred_consistency = F.l1_loss(audio_pitch_features, pred_f0_aligned)
            target_consistency = F.l1_loss(audio_pitch_features, f0_aligned)
        else:  # l2
            pred_consistency = F.mse_loss(audio_pitch_features, pred_f0_aligned)
            target_consistency = F.mse_loss(audio_pitch_features, f0_aligned)

        consistency_loss = F.l1_loss(pred_consistency, target_consistency)
        return consistency_loss

    def _compute_voiced_unvoiced_consistency_loss(self, audio_features: torch.Tensor,
                                                ap: torch.Tensor,
                                                pred_ap: torch.Tensor) -> torch.Tensor:
        """Compute voiced/unvoiced consistency loss."""
        # Align temporal dimensions
        min_time = min(audio_features.shape[-1], ap.shape[-1], pred_ap.shape[-1])
        audio_aligned = audio_features[..., :min_time]
        ap_aligned = ap[..., :min_time]
        pred_ap_aligned = pred_ap[..., :min_time]

        # Compute voicing features from audio (simplified)
        # In practice, we'd compute zero-crossing rate, energy, etc.
        audio_voicing = torch.sigmoid(audio_features.mean(dim=1, keepdim=True))  # Placeholder

        # Compare voicing consistency
        if self.loss_type == "l1":
            pred_consistency = F.l1_loss(audio_voicing, pred_ap_aligned)
            target_consistency = F.l1_loss(audio_voicing, ap_aligned)
        else:  # l2
            pred_consistency = F.mse_loss(audio_voicing, pred_ap_aligned)
            target_consistency = F.mse_loss(audio_voicing, ap_aligned)

        consistency_loss = F.l1_loss(pred_consistency, target_consistency)
        return consistency_loss

    def _compute_temporal_dynamics_consistency_loss(self, audio_features: torch.Tensor,
                                                  f0: torch.Tensor,
                                                  sp: torch.Tensor,
                                                  ap: torch.Tensor,
                                                  pred_f0: torch.Tensor,
                                                  pred_sp: torch.Tensor,
                                                  pred_ap: torch.Tensor) -> torch.Tensor:
        """Compute temporal dynamics consistency loss."""
        # Compute temporal derivatives (dynamics)
        def compute_derivative(x):
            if x.shape[-1] < 2:
                return torch.zeros_like(x)
            return x[..., 1:] - x[..., :-1]

        min_time = min(
            audio_features.shape[-1],
            f0.shape[-1],
            sp.shape[-1],
            ap.shape[-1],
            pred_f0.shape[-1],
            pred_sp.shape[-1],
            pred_ap.shape[-1]
        ) - 1  # For derivatives

        if min_time <= 0:
            return torch.tensor(0.0, device=audio_features.device)

        # Align all features
        audio_aligned = audio_features[..., :min_time+1]
        f0_aligned = f0[..., :min_time+1]
        sp_aligned = sp[..., :min_time+1]
        ap_aligned = ap[..., :min_time+1]
        pred_f0_aligned = pred_f0[..., :min_time+1]
        pred_sp_aligned = pred_sp[..., :min_time+1]
        pred_ap_aligned = pred_ap[..., :min_time+1]

        # Compute derivatives
        audio_deriv = compute_derivative(audio_aligned)
        f0_deriv = compute_derivative(f0_aligned)
        sp_deriv = compute_derivative(sp_aligned)
        ap_deriv = compute_derivative(ap_aligned)
        pred_f0_deriv = compute_derivative(pred_f0_aligned)
        pred_sp_deriv = compute_derivative(pred_sp_aligned)
        pred_ap_deriv = compute_derivative(pred_ap_aligned)

        # Compute consistency of derivatives
        if self.loss_type == "l1":
            f0_deriv_loss = F.l1_loss(pred_f0_deriv, f0_deriv)
            sp_deriv_loss = F.l1_loss(sp_deriv, sp_deriv)
            ap_deriv_loss = F.l1_loss(ap_deriv, ap_deriv)
            audio_deriv_loss = F.l1_loss(
                torch.cat([pred_f0_deriv, pred_sp_deriv, pred_ap_deriv], dim=1),
                torch.cat([f0_deriv, sp_deriv, ap_deriv], dim=1)
            )
        else:  # l2
            f0_deriv_loss = F.mse_loss(pred_f0_deriv, f0_deriv)
            sp_deriv_loss = F.mse_loss(sp_deriv, sp_deriv)
            ap_deriv_loss = F.mse_loss(ap_deriv, ap_deriv)
            audio_deriv_loss = F.mse_loss(
                torch.cat([pred_f0_deriv, pred_sp_deriv, pred_ap_deriv], dim=1),
                torch.cat([f0_deriv, sp_deriv, ap_deriv], dim=1)
            )

        # Combine losses
        consistency_loss = (f0_deriv_loss + sp_deriv_loss + ap_deriv_loss + audio_deriv_loss) / 4.0
        return consistency_loss

    def _compute_direct_parameter_consistency(self, f0: Optional[torch.Tensor],
                                            sp: Optional[torch.Tensor],
                                            ap: Optional[torch.Tensor],
                                            pred_f0: Optional[torch.Tensor],
                                            pred_sp: Optional[torch.Tensor],
                                            pred_ap: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute direct parameter consistency as fallback."""
        loss_dict = {}
        total_loss = 0.0

        # F0 consistency
        if f0 is not None and pred_f0 is not None:
            if self.loss_type == "l1":
                f0_loss = F.l1_loss(pred_f0, f0)
            else:
                f0_loss = F.mse_loss(pred_f0, f0)
            total_loss += f0_loss
            loss_dict["direct_f0"] = f0_loss.item()

        # SP consistency
        if sp is not None and pred_sp is not None:
            if self.loss_type == "l1":
                sp_loss = F.l1_loss(pred_sp, sp)
            else:
                sp_loss = F.mse_loss(pred_sp, sp)
            total_loss += sp_loss
            loss_dict["direct_sp"] = sp_loss.item()

        # AP consistency
        if ap is not None and pred_ap is not None:
            if self.loss_type == "l1":
                ap_loss = F.l1_loss(pred_ap, ap)
            else:
                ap_loss = F.mse_loss(pred_ap, ap)
            total_loss += ap_loss
            loss_dict["direct_ap"] = ap_loss.item()

        return total_loss, loss_dict