"""
Diffusion model integration for CantioAI.
Provides interfaces for using diffusion models as post-processors or in joint training
with the existing HybridSVC system while preserving source-filter parameter control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union
import logging
import os
import sys

# Add the src directory to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .diffusion import ConditionalDiffusionModel, DiffusionPostProcessor
from .hybrid_svc import MultiTaskHybridSVC
from ..training.diffusion_trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


class DiffusionEnhancedVocoder(nn.Module):
    """
    Vocoder system that combines the existing HybridSVC with a diffusion model
    for enhanced audio generation while preserving source-filter control.

    Supports three modes:
    1. Post-processing: Generate audio with HybridSVC, then enhance with diffusion
    2. Joint training: Train HybridSVC and diffusion model together
    3. Direct generation: Use diffusion model directly with conditioning from HybridSVC
    """

    def __init__(
        self,
        base_vocoder: MultiTaskHybridSVC,
        diffusion_model: ConditionalDiffusionModel,
        mode: str = "postprocess",  # postprocess, joint, direct
        condition_projector: Optional[nn.Module] = None,
        feature_extractor: Optional[nn.Module] = None
    ):
        """
        Initialize diffusion-enhanced vocoder.

        Args:
            base_vocoder: Existing HybridSVC model
            diffusion_model: Conditional diffusion model for enhancement
            mode: Integration mode - "postprocess", "joint", or "direct"
            condition_projector: Optional network to project features to diffusion conditioning space
            feature_extractor: Optional network to extract features from audio for conditioning
        """
        super().__init__()
        self.base_vocoder = base_vocoder
        self.diffusion_model = diffusion_model
        self.mode = mode
        self.condition_projector = condition_projector
        self.feature_extractor = feature_extractor

        # Validate mode
        valid_modes = ["postprocess", "joint", "direct"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        logger.info(f"Initialized DiffusionEnhancedVocoder in {mode} mode")

    def forward(
        self,
        phoneme_features: torch.Tensor,
        f0: torch.Tensor,
        spk_id: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        f0_is_hz: bool = True,
        return_quantized_f0: bool = False,
        return_task_outputs: bool = False,
        diffusion_steps: Optional[int] = None,
        enhance_audio: bool = True
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]],  # Standard output
        Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]  # With enhanced audio
    ]:
        """
        Forward pass through the diffusion-enhanced vocoder.

        Args:
            phoneme_features: Linguistic/phonetic features (B, T, D_ph)
            f0: Fundamental frequency (B, T, 1) - in Hz if f0_is_hz=True
            spk_id: Speaker IDs (B,)
            task_id: Task IDs (B,) for multi-task learning (optional)
            f0_is_hz: Whether input F0 is in Hz (True) or normalized (False)
            return_quantized_f0: Whether to return quantized F0
            return_task_outputs: Whether to return outputs from all tasks (multi-task)
            diffusion_steps: Number of diffusion steps for enhancement (if None, use model default)
            enhance_audio: Whether to apply diffusion enhancement (False to disable)

        Returns:
            If enhance_audio=False or mode="joint" during training:
                Standard HybridSVC output tuple
            If enhance_audio=True:
                Tuple of (sp_pred, f0_quant, extras, enhanced_audio)
        """
        # Get base vocoder output
        base_outputs = self.base_vocoder(
            phoneme_features=phoneme_features,
            f0=f0,
            spk_id=spk_id,
            task_id=task_id,
            f0_is_hz=f0_is_hz,
            return_quantized_f0=return_quantized_f0,
            return_task_outputs=return_task_outputs
        )

        sp_pred, f0_quant, extras = base_outputs

        # Apply diffusion enhancement if requested and not in joint training mode
        enhanced_audio = None
        if enhance_audio and self.mode != "joint":
            # Extract conditioning information for diffusion model
            conditioning = self._extract_diffusion_conditioning(
                phoneme_features=phoneme_features,
                f0=f0,
                spk_id=spk_id,
                base_outputs=base_outputs,
                f0_is_hz=f0_is_hz
            )

            # Generate initial audio from base vocoder (WORLD synthesis)
            initial_audio = self._synthesize_initial_audio(
                sp_pred=sp_pred,
                f0=f0 if not return_quantized_f0 else f0_quant,
                spk_id=spk_id
            )

            # Apply diffusion enhancement
            enhanced_audio = self._enhance_with_diffusion(
                initial_audio=initial_audio,
                conditioning=conditioning,
                diffusion_steps=diffusion_steps
            )

        # Return appropriate format
        if enhance_audio and enhanced_audio is not None:
            return sp_pred, f0_quant, extras, enhanced_audio
        else:
            return sp_pred, f0_quant, extras

    def _extract_diffusion_conditioning(
        self,
        phoneme_features: torch.Tensor,
        f0: torch.Tensor,
        spk_id: torch.Tensor,
        base_outputs: Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]],
        f0_is_hz: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Extract conditioning information for the diffusion model.

        Args:
            phoneme_features: Input phoneme features
            f0: Input F0
            spk_id: Speaker IDs
            base_outputs: Outputs from base vocoder
            f0_is_hz: Whether F0 is in Hz

        Returns:
            Dictionary containing conditioning tensors for diffusion model
        """
        sp_pred, f0_quant, extras = base_outputs

        # Use quantized F0 if available and requested, otherwise use original
        f0_for_cond = f0_quant if (f0_quant is not None and return_quantized_f0) else f0

        # Ensure F0 is in the right shape and format
        if f0_for_cond.dim() == 2:
            f0_for_cond = f0_for_cond.unsqueeze(-1)  # (B, T, 1)

        # Extract spectral envelope prediction
        sp_pred_for_cond = sp_pred  # (B, T, D_sp)

        # Extract aperiodicity if available from task outputs
        ap_for_cond = None
        if return_task_outputs and 'task_outputs' in extras:
            task_outputs = extras['task_outputs']
            # Try to get AP from noise robustness task or create dummy
            if 'noise_robustness' in task_outputs and 'ap' in task_outputs['noise_robustness']:
                ap_for_cond = task_outputs['noise_robustness']['ap']
            elif 'singing' in task_outputs and 'ap' in task_outputs['singing']:
                ap_for_cond = task_outputs['singing']['ap']
            else:
                # Create dummy AP conditioning
                ap_for_cond = torch.zeros_like(f0_for_cond)  # (B, T, 1)
        else:
            # Create dummy AP conditioning
            ap_for_cond = torch.zeros_like(f0_for_cond)  # (B, T, 1)

        # Extract or create HuBERT-like conditioning
        hubert_for_cond = phoneme_features  # Use phoneme features as HuBERT proxy
        if hubert_for_cond.dim() != 3:
            # Expand to match audio length if needed
            hubert_for_cond = hubert_for_cond.unsqueeze(1).expand(-1, f0_for_cond.shape[1], -1)

        # Apply condition projector if provided
        if self.condition_projector is not None:
            # Project each conditioning component
            if f0_for_cond is not None:
                f0_for_cond = self.condition_projector(f0_for_cond)
            if sp_pred_for_cond is not None:
                sp_pred_for_cond = self.condition_projector(sp_pred_for_cond)
            if ap_for_cond is not None:
                ap_for_cond = self.condition_projector(ap_for_cond)
            if hubert_for_cond is not None:
                hubert_for_cond = self.condition_projector(hubert_for_cond)

        return {
            "f0": f0_for_cond,
            "sp": sp_pred_for_cond,
            "ap": ap_for_cond,
            "hubert": hubert_for_cond
        }

    def _synthesize_initial_audio(
        self,
        sp_pred: torch.Tensor,
        f0: torch.Tensor,
        spk_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Synthesize initial audio using WORLD vocoder from base vocoder predictions.

        Args:
            sp_pred: Predicted spectral envelope (B, T, D_sp)
            f0: Fundamental frequency (B, T, 1)
            spk_id: Speaker IDs (B,)

        Returns:
            Initial audio tensor (B, 1, T_samples)
        """
        # Use the base vocoder's synthesis capabilities if available
        if hasattr(self.base_vocoder, 'synthesize'):
            return self.base_vocoder.synthesize(sp_pred, f0, spk_id)
        else:
            # Fallback: simple audio synthesis (placeholder)
            # In practice, this would use WORLD synthesis
            batch_size, seq_len, _ = sp_pred.shape
            # Generate dummy audio signal
            audio_length = seq_len * 200  # Approximate samples based on 5ms frame period
            initial_audio = torch.randn(batch_size, 1, audio_length, device=sp_pred.device)
            return initial_audio

    def _enhance_with_diffusion(
        self,
        initial_audio: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        diffusion_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Enhance audio using the diffusion model.

        Args:
            initial_audio: Initial audio tensor (B, 1, T_samples)
            conditioning: Dictionary of conditioning tensors
            diffusion_steps: Number of diffusion steps (if None, use model default)

        Returns:
            Enhanced audio tensor (B, 1, T_samples)
        """
        # Wrap diffusion model in post-processor if not already
        if not isinstance(self.diffusion_model, DiffusionPostProcessor):
            diffusion_processor = DiffusionPostProcessor(self.diffusion_model)
        else:
            diffusion_processor = self.diffusion_model

        # Apply enhancement
        enhanced_audio = diffusion_processor.enhance(
            audio=initial_audio,
            f0=conditioning["f0"],
            sp=conditioning["sp"],
            ap=conditioning["ap"],
            hubert_features=conditioning["hubert"],
            num_steps=diffusion_steps
        )

        return enhanced_audio

    def enhance_audio(
        self,
        audio: torch.Tensor,
        f0: torch.Tensor,
        sp: torch.Tensor,
        ap: torch.Tensor,
        hubert_features: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Enhance existing audio using the diffusion model (standalone enhancement).

        Args:
            audio: Input audio tensor (B, 1, T_samples)
            f0: F0 contour (B, T, 1)
            sp: Spectral parameters (B, T, D_sp)
            ap: Aperiodicity parameters (B, T, 1)
            hubert_features: HuBERT features (B, T, D_hubert)
            num_steps: Number of diffusion steps (if None, use model default)

        Returns:
            Enhanced audio tensor (B, 1, T_samples)
        """
        # Wrap diffusion model in post-processor if not already
        if not isinstance(self.diffusion_model, DiffusionPostProcessor):
            diffusion_processor = DiffusionPostProcessor(self.diffusion_model)
        else:
            diffusion_processor = self.diffusion_model

        # Apply enhancement
        enhanced_audio = diffusion_processor.enhance(
            audio=audio,
            f0=f0,
            sp=sp,
            ap=ap,
            hubert_features=hubert_features,
            num_steps=num_steps
        )

        return enhanced_audio

    def get_training_interface(self) -> DiffusionTrainer:
        """
        Get a diffusion trainer for training the diffusion model.

        Returns:
            DiffusionTrainer instance configured for this setup
        """
        return DiffusionTrainer(
            diffusion_model=self.diffusion_model,
            base_model=self.base_vocoder if self.mode == "joint" else None,
            config=getattr(self, 'config', {})
        )


def create_diffusion_enhanced_vocoder(
    base_vocoder: MultiTaskHybridSVC,
    diffusion_config: Optional[Dict[str, Any]] = None,
    mode: str = "postprocess"
) -> DiffusionEnhancedVocoder:
    """
    Factory function to create a diffusion-enhanced vocoder.

    Args:
        base_vocoder: Existing HybridSVC model
        diffusion_config: Configuration for diffusion model (if None, use defaults)
        mode: Integration mode - "postprocess", "joint", or "direct"

    Returns:
        Configured DiffusionEnhancedVocoder instance
    """
    # Default diffusion configuration
    default_diffusion_config = {
        "input_dim": 1,  # Audio waveform
        "output_dim": 1,  # Audio waveform
        "condition_dim": 256,  # Conditioning dimension
        "hidden_dim": 128,
        "num_layers": 30,
        "num_cycles": 10,
        "diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "dropout": 0.1
    }

    # Use provided config or defaults
    if diffusion_config is None:
        diffusion_config = default_diffusion_config
    else:
        # Merge with defaults
        for key, value in default_diffusion_config.items():
            if key not in diffusion_config:
                diffusion_config[key] = value

    # Create diffusion model
    diffusion_model = ConditionalDiffusionModel(**diffusion_config)

    # Create condition projector if needed (simple linear projection for now)
    condition_projector = None
    if diffusion_config.get("condition_projection", False):
        # Project from base model feature space to diffusion conditioning space
        base_feature_dim = (
            base_vocoder.phoneme_feature_dim +  # phoneme features
            1 +  # F0
            base_vocoder.spectral_envelope_dim  # spectral envelope
        )
        condition_projector = nn.Linear(base_feature_dim, diffusion_config["condition_dim"])

    # Create enhanced vocoder
    enhanced_vocoder = DiffusionEnhancedVocoder(
        base_vocoder=base_vocoder,
        diffusion_model=diffusion_model,
        mode=mode,
        condition_projector=condition_projector
    )

    logger.info(f"Created diffusion-enhanced vocoder in {mode} mode")
    return enhanced_vocoder