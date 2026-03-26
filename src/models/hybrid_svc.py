"""
Multi-task Hybrid Source-Filter + Neural Vocoder Voice Conversion System.

Extends the original HybridSVC to support multi-task learning while maintaining
backward compatibility with single-task voice conversion.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
import logging
import sys
import os

# Add the src directory to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import with fallback strategy
try:
    # Try relative imports first (when used as part of package)
    from .hybrid_predictor import HybridSpectralPredictor
    from .pitch_quantizer import DifferentiablePitchQuantizer
    from ..multitask.shared_encoder import MultiTaskSharedEncoder
    from ..multitask.task_heads import (
        SingingConversionHead,
        SpeechConversionHead,
        NoiseRobustnessHead,
        create_task_head
    )
    from ..multitask.adaptive_norm import TaskConditionedAdaIN
except ImportError:
    # Fallback: direct module imports
    # We need to construct the paths carefully to avoid importing the full models package
    models_path = os.path.join(src_path, 'models')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)

    # Import just the specific modules we need
    from hybrid_predictor import HybridSpectralPredictor
    from pitch_quantizer import DifferentiablePitchQuantizer

    # multitask is at src level
    from multitask.shared_encoder import MultiTaskSharedEncoder
    from multitask.task_heads import (
        SingingConversionHead,
        SpeechConversionHead,
        NoiseRobustnessHead,
        create_task_head
    )
    from multitask.adaptive_norm import TaskConditionedAdaIN

logger = logging.getLogger(__name__)


class MultiTaskHybridSVC(nn.Module):
    """
    Multi-task Hybrid Source-Filter + Neural Vocoder Voice Conversion System.

    Combines:
    - Multi-task shared encoder for feature extraction
    - Task-specific heads for singing conversion, speech conversion, and noise robustness
    - WORLD excitation source (F0, AP)
    - Optional differentiable pitch quantization
    - Task-conditioned adaptive instance normalization

    The model can operate in single-task mode (backward compatible) or multi-task mode.
    """

    def __init__(
        self,
        phoneme_feature_dim: int,
        spectral_envelope_dim: int = 60,
        speaker_embed_dim: int = 128,
        n_speakers: int = 100,
        use_pitch_quantizer: bool = True,
        pitch_quantizer_config: Optional[Dict[str, Any]] = None,
        # Multi-task specific parameters
        enable_multitask: bool = False,
        tasks: Optional[List[str]] = None,
        shared_dim: int = 256,
        num_shared_layers: int = 2,
        num_task_shared_layers: int = 2,
        task_specific_dim: int = 128,
        dropout: float = 0.1,
        use_task_conditioned_adain: bool = True,
        **predictor_kwargs
    ):
        """
        Initialize Multi-task HybridSVC model.

        Args:
            phoneme_feature_dim: Dimension of phoneme/linguistic features
            spectral_envelope_dim: Dimension of spectral envelope output (e.g., 60 for MCEP)
            speaker_embed_dim: Dimension of speaker embeddings
            n_speakers: Number of speakers in training data
            use_pitch_quantizer: Whether to use differentiable pitch quantization
            pitch_quantizer_config: Configuration for pitch quantizer
            enable_multitask: Whether to enable multi-task learning mode
            tasks: List of task names to enable (default: ['singing', 'speech', 'noise_robustness'])
            shared_dim: Dimension of shared features in encoder
            num_shared_layers: Number of layers in shared encoder
            num_task_shared_layers: Number of layers in task-shared encoder
            task_specific_dim: Dimension of task-specific features
            dropout: Dropout rate
            use_task_conditioned_adain: Whether to use task-conditioned AdaIN
            **predictor_kwargs: Additional arguments for HybridSpectralPredictor
        """
        super().__init__()

        # Store configuration
        self.phoneme_feature_dim = phoneme_feature_dim
        self.spectral_envelope_dim = spectral_envelope_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.n_speakers = n_speakers
        self.use_pitch_quantizer = use_pitch_quantizer
        self.enable_multitask = enable_multitask
        self.tasks = tasks or ['singing', 'speech', 'noise_robustness']
        self.num_tasks = len(self.tasks)
        self.use_task_conditioned_adain = use_task_conditioned_adain

        # Initialize Hybrid Spectral Predictor as base encoder
        self.base_predictor = HybridSpectralPredictor(
            D_ph=phoneme_feature_dim,
            D_sp=spectral_envelope_dim,
            D_spk=speaker_embed_dim,
            n_speakers=n_speakers,
            **predictor_kwargs
        )

        # Initialize Pitch Quantizer (optional)
        if self.use_pitch_quantizer:
            quantizer_config = pitch_quantizer_config or {}
            self.pitch_quantizer = DifferentiablePitchQuantizer(**quantizer_config)
        else:
            self.pitch_quantizer = None

        # Initialize multi-task components if enabled
        if self.enable_multitask:
            self._init_multitask_components(
                shared_dim=shared_dim,
                num_shared_layers=num_shared_layers,
                num_task_shared_layers=num_task_shared_layers,
                task_specific_dim=task_specific_dim,
                dropout=dropout
            )
        else:
            # In single-task mode, keep references for compatibility
            self.shared_encoder = None
            self.task_heads = None
            self.task_adain = None

        logger.info(
            f"Initialized MultiTaskHybridSVC model:\n"
            f"  Phoneme feature dim: {phoneme_feature_dim}\n"
            f"  Spectral envelope dim: {spectral_envelope_dim}\n"
            f"  Speaker embed dim: {speaker_embed_dim}\n"
            f"  Number of speakers: {n_speakers}\n"
            f"  Use pitch quantizer: {use_pitch_quantizer}\n"
            f"  Enable multi-task: {enable_multitask}\n"
            f"  Tasks: {self.tasks}\n"
            f"  Use task-conditioned AdaIN: {use_task_conditioned_adain}"
        )

    def _init_multitask_components(
        self,
        shared_dim: int,
        num_shared_layers: int,
        num_task_shared_layers: int,
        task_specific_dim: int,
        dropout: float
    ):
        """Initialize multi-task learning components."""

        # Create shared encoder using the base predictor
        self.shared_encoder = MultiTaskSharedEncoder(
            base_encoder=self.base_predictor,
            shared_dim=shared_dim,
            num_layers=num_shared_layers,
            dropout=dropout
        )

        # Initialize task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name in self.tasks:
            if task_name == 'singing':
                self.task_heads[task_name] = SingingConversionHead(
                    shared_dim=shared_dim,
                    predict_f0=True,
                    predict_sp=True,
                    predict_ap=True,
                    sp_dim=self.spectral_envelope_dim,
                    use_periodicity=True
                )
            elif task_name == 'speech':
                self.task_heads[task_name] = SpeechConversionHead(
                    shared_dim=shared_dim,
                    predict_f0=True,
                    predict_sp=True,
                    predict_ap=False,  # AP less critical for speech
                    sp_dim=self.spectral_envelope_dim,
                    use_speaker_embedding=True,
                    speaker_embed_dim=self.speaker_embed_dim
                )
            elif task_name == 'noise_robustness':
                self.task_heads[task_name] = NoiseRobustnessHead(
                    shared_dim=shared_dim,
                    predict_mask=True,
                    predict_clean_features=True,
                    predict_snr=True,
                    feature_dim=self.spectral_envelope_dim,
                    use_temporal_context=True
                )
            else:
                raise ValueError(f"Unknown task: {task_name}")

        # Initialize task-conditioned AdaIN if enabled
        if self.use_task_conditioned_adain:
            self.task_adain = TaskConditionedAdaIN(
                num_features=shared_dim,
                speaker_embed_dim=self.speaker_embed_dim,
                task_embed_dim=64,
                num_speakers=self.n_speakers,
                num_tasks=self.num_tasks,
                use_task_embedding=True
            )
        else:
            self.task_adain = None

    def forward(
        self,
        phoneme_features: torch.Tensor,  # (B, T, D_ph)
        f0: torch.Tensor,                # (B, T, 1)  - can be normalized or Hz
        spk_id: torch.Tensor,            # (B,)       - speaker IDs
        task_id: Optional[torch.Tensor] = None,  # (B,)     - task IDs for multi-task
        f0_is_hz: bool = True,           # Whether input F0 is in Hz (True) or normalized (False)
        return_quantized_f0: bool = False,
        return_task_outputs: bool = False  # Whether to return all task outputs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass of the voice conversion model.

        Args:
            phoneme_features: Linguistic/phonetic features (B, T, D_ph)
            f0: Fundamental frequency (B, T, 1) - in Hz if f0_is_hz=True
            spk_id: Speaker IDs (B,)
            task_id: Task IDs (B,) for multi-task learning (optional in single-task mode)
            f0_is_hz: Whether input F0 is in Hz (requires quantization if enabled)
            return_quantized_f0: Whether to return quantized F0
            return_task_outputs: Whether to return outputs from all tasks (for multi-task training)

        Returns:
            If return_task_outputs=False (single-task mode or inference):
                Tuple of:
                - sp_pred: Predicted spectral envelope (B, T, D_sp)
                - f0_quant: Quantized F0 (B, T, 1) if return_quantized_f0=True and use_pitch_quantizer, else None
                - extras: Dictionary with additional outputs
            If return_task_outputs=True (multi-task training):
                Tuple of:
                - sp_pred: Predicted spectral envelope for primary task (B, T, D_sp)
                - f0_quant: Quantized F0 (B, T, 1) if applicable
                - extras: Dictionary containing:
                    - 'task_outputs': Dict mapping task names to their outputs
                    - 'shared_features': Shared representation tensor
                    - Plus original extras
        """
        B, T, _ = phoneme_features.shape

        # Process F0: quantize if needed and requested
        f0_quant = None
        f0_for_encoder = f0  # Default: use input F0 as-is

        if self.use_pitch_quantizer and f0_is_hz:
            # Quantize F0 to musical scale
            f0_quant = self.pitch_quantizer(f0)  # (B, T, 1)
            if return_quantized_f0:
                f0_for_encoder = f0_quant  # Use quantized F0 for prediction
            # Note: For training, we might use original F0 for encoder but quantized for loss
        elif not f0_is_hz:
            # F0 is already normalized, use as-is
            f0_for_encoder = f0

        if self.enable_multitask and task_id is not None:
            # Multi-task mode
            return self._forward_multitask(
                phoneme_features=phoneme_features,
                f0=f0_for_encoder,
                spk_id=spk_id,
                task_id=task_id,
                f0_quant=f0_quant,
                return_quantized_f0=return_quantized_f0,
                return_task_outputs=return_task_outputs
            )
        else:
            # Single-task mode (backward compatible)
            return self._forward_singletask(
                phoneme_features=phoneme_features,
                f0=f0_for_encoder,
                spk_id=spk_id,
                f0_quant=f0_quant,
                return_quantized_f0=return_quantized_f0
            )

    def _forward_singletask(
        self,
        phoneme_features: torch.Tensor,
        f0: torch.Tensor,
        spk_id: torch.Tensor,
        f0_quant: Optional[torch.Tensor],
        return_quantized_f0: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Single-task forward pass (backward compatible with original HybridSVC)."""
        # Predict spectral envelope using base predictor
        sp_pred = self.base_predictor(
            phoneme_features=phoneme_features,
            f0=f0,
            spk_id=spk_id
        )  # (B, T, D_sp)

        # Prepare extras dictionary
        extras = {}
        if self.use_pitch_quantizer and f0_quant is not None:
            extras["f0_quant"] = f0_quant
            extras["f0_original"] = f0

        return sp_pred, f0_quant, extras

    def _forward_multitask(
        self,
        phoneme_features: torch.Tensor,
        f0: torch.Tensor,
        spk_id: torch.Tensor,
        task_id: torch.Tensor,
        f0_quant: Optional[torch.Tensor],
        return_quantized_f0: bool,
        return_task_outputs: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Multi-task forward pass."""

        # The shared encoder expects the RAW inputs to the base encoder
        # It will internally call the base encoder and then process the outputs
        encoder_outputs_dict = self.shared_encoder(
            phoneme_features=phoneme_features,
            f0=f0,
            spk_id=spk_id
        )

        # Extract shared features and encoder outputs
        shared_features = encoder_outputs_dict["shared_features"]
        encoder_outputs = encoder_outputs_dict["encoder_outputs"]
        task_logits = encoder_outputs_dict["task_logits"]

        # Apply task-conditioned AdaIN if enabled
        if self.task_adain is not None and task_id is not None:
            # AdaIN expects (B, C, *) format, so we need to reshape
            # shared_features is (B, T, shared_dim) -> (B, shared_dim, T)
            shared_features_for_adain = shared_features.transpose(1, 2)  # (B, shared_dim, T)
            normalized_features = self.task_adain(shared_features_for_adain, spk_id, task_id)
            # Back to (B, T, shared_dim)
            shared_features = normalized_features.transpose(1, 2)

        # Compute task-specific predictions using the shared features
        task_outputs = {}
        for task_name, task_head in self.task_heads.items():
            task_outputs[task_name] = task_head(shared_features)

        # For backward compatibility, we need to extract the spectral envelope prediction
        # from the encoder_outputs (which is the direct output of HybridSpectralPredictor)
        # The encoder_outputs should be the spectral envelope prediction
        sp_pred = encoder_outputs

        # For F0 prediction, we need to get it from the appropriate task head
        # In our case, we'll use the singing task head's F0 prediction for compatibility
        f0_pred = None
        if 'singing' in task_outputs and 'f0' in task_outputs['singing']:
            f0_pred = task_outputs['singing']['f0']

        # Prepare extras dictionary
        extras = {
            "shared_features": shared_features,
            "encoder_outputs": encoder_outputs,
            "task_outputs": task_outputs
        }

        # Add pitch quantization info if applicable
        if self.use_pitch_quantizer and f0_quant is not None:
            extras["f0_quant"] = f0_quant
            extras["f0_original"] = f0

        # Return format depends on whether we want all task outputs
        if return_task_outputs:
            return sp_pred, f0_quant, extras
        else:
            # For compatibility, return same format as single-task
            return sp_pred, f0_quant, {
                "f0_quant": f0_quant if self.use_pitch_quantizer else None,
                "f0_original": f0 if self.use_pitch_quantizer else None
            }

    def predict_sp(
        self,
        phoneme_features: torch.Tensor,
        f0: torch.Tensor,
        spk_id: torch.Tensor,
        f0_is_hz: bool = True
    ) -> torch.Tensor:
        """
        Predict spectral envelope (convenience method).

        Args:
            phoneme_features: (B, T, D_ph)
            f0: (B, T, 1) in Hz
            spk_id: (B,)
            f0_is_hz: Whether F0 is in Hz

        Returns:
            sp_pred: (B, T, D_sp)
        """
        sp_pred, _, _ = self.forward(
            phoneme_features, f0, spk_id,
            f0_is_hz=f0_is_hz,
            return_quantized_f0=False
        )
        return sp_pred

    def quantize_f0(self, f0_hz: torch.Tensor) -> torch.Tensor:
        """
        Quantize F0 to musical scale (if quantizer enabled).

        Args:
            f0_hz: F0 in Hz (B, T, 1)

        Returns:
            f0_quant: Quantized F0 in Hz (B, T, 1)
        """
        if not self.use_pitch_quantizer:
            logger.warning("Pitch quantizer not enabled, returning original F0")
            return f0_hz
        return self.pitch_quantizer(f0_hz)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary containing model configuration
        """
        config = {
            "phoneme_feature_dim": self.phoneme_feature_dim,
            "spectral_envelope_dim": self.spectral_envelope_dim,
            "speaker_embed_dim": self.speaker_embed_dim,
            "n_speakers": self.n_speakers,
            "use_pitch_quantizer": self.use_pitch_quantizer,
            "enable_multitask": self.enable_multitask,
            "tasks": self.tasks,
            "use_task_conditioned_adain": self.use_task_conditioned_adain
        }
        if self.use_pitch_quantizer:
            config["pitch_quantizer_config"] = {
                "ref_freq": self.pitch_quantizer.ref_freq,
                "ref_midi": self.pitch_quantizer.ref_midi,
                "octaves": self.pitch_quantizer.octaves,
                "use_ste": self.pitch_quantizer.use_ste,
            }
        return config

    def set_task_weights(self, weights: Dict[str, float]):
        """
        Set task-specific weights for loss balancing.

        Args:
            weights: Dictionary mapping task names to weights
        """
        # This would be used by the training strategy
        self.task_weights = weights

    def get_task_head(self, task_name: str):
        """
        Get a specific task head.

        Args:
            task_name: Name of the task

        Returns:
            Task-specific head module or None
        """
        if self.enable_multitask and self.task_heads:
            return self.task_heads[task_name] if task_name in self.task_heads else None
        return None

    def enable_task(self, task_name: str):
        """
        Enable a specific task (for dynamic task activation).

        Args:
            task_name: Name of the task to enable
        """
        if not self.enable_multitask:
            logger.warning("Multi-task not enabled, cannot enable tasks")
            return

        if task_name not in self.tasks:
            self.tasks.append(task_name)
            self.num_tasks = len(self.tasks)
            # Would need to reinitialize task heads and potentially AdaIN
            logger.info(f"Task {task_name} added to multi-task framework")

    def disable_task(self, task_name: str):
        """
        Disable a specific task (for dynamic task deactivation).

        Args:
            task_name: Name of the task to disable
        """
        if not self.enable_multitask:
            logger.warning("Multi-task not enabled, cannot disable tasks")
            return

        if task_name in self.tasks and len(self.tasks) > 1:
            self.tasks.remove(task_name)
            self.num_tasks = len(self.tasks)
            # Would need to reinitialize task heads and potentially AdaIN
            logger.info(f"Task {task_name} removed from multi-task framework")


# Backward compatibility alias
HybridSVC = MultiTaskHybridSVC


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 50
    phoneme_dim = 32
    sp_dim = 60
    n_speakers = 10

    # Test single-task mode (backward compatibility)
    print("Testing single-task mode...")
    model_st = MultiTaskHybridSVC(
        phoneme_feature_dim=phoneme_dim,
        spectral_envelope_dim=sp_dim,
        speaker_embed_dim=128,
        n_speakers=n_speakers,
        use_pitch_quantizer=True,
        enable_multitask=False  # Single-task mode
    )

    # Create dummy inputs
    phoneme_features = torch.randn(batch_size, seq_len, phoneme_dim)
    f0_hz = torch.rand(batch_size, seq_len, 1) * 400 + 80  # 80-480 Hz
    spk_id = torch.randint(0, n_speakers, (batch_size,))

    # Forward pass
    sp_pred, f0_quant, extras = model_st(
        phoneme_features, f0_hz, spk_id,
        f0_is_hz=True, return_quantized_f0=True
    )

    print(f"Single-task - Input shapes:")
    print(f"  phoneme_features: {phoneme_features.shape}")
    print(f"  f0_hz: {f0_hz.shape}")
    print(f"  spk_id: {spk_id.shape}")
    print(f"Single-task - Output shapes:")
    print(f"  sp_pred: {sp_pred.shape}")
    print(f"  f0_quant: {f0_quant.shape if f0_quant is not None else None}")
    print(f"Single-task - Extras keys: {list(extras.keys())}")

    # Check output shapes
    assert sp_pred.shape == (batch_size, seq_len, sp_dim)
    assert f0_quant is not None and f0_quant.shape == (batch_size, seq_len, 1)
    assert "f0_quant" in extras
    assert "f0_original" in extras

    # Test multi-task mode
    print("\nTesting multi-task mode...")
    model_mt = MultiTaskHybridSVC(
        phoneme_feature_dim=phoneme_dim,
        spectral_envelope_dim=sp_dim,
        speaker_embed_dim=128,
        n_speakers=n_speakers,
        use_pitch_quantizer=True,
        enable_multitask=True,  # Multi-task mode
        tasks=['singing', 'speech', 'noise_robustness']
    )

    # Create task IDs (0: singing, 1: speech, 2: noise_robustness)
    task_id = torch.randint(0, 3, (batch_size,))  # Random task IDs

    # Forward pass with task IDs
    sp_pred_mt, f0_quant_mt, extras_mt = model_mt(
        phoneme_features, f0_hz, spk_id, task_id=task_id,
        f0_is_hz=True, return_quantized_f0=True, return_task_outputs=True
    )

    print(f"Multi-task - Input shapes:")
    print(f"  phoneme_features: {phoneme_features.shape}")
    print(f"  f0_hz: {f0_hz.shape}")
    print(f"  spk_id: {spk_id.shape}")
    print(f"  task_id: {task_id.shape}")
    print(f"Multi-task - Output shapes:")
    print(f"  sp_pred: {sp_pred_mt.shape}")
    print(f"  f0_quant: {f0_quant_mt.shape if f0_quant_mt is not None else None}")
    print(f"Multi-task - Extras keys: {list(extras_mt.keys())}")
    if 'task_outputs' in extras_mt:
        print(f"Multi-task - Task outputs: {list(extras_mt['task_outputs'].keys())}")
        for task_name, task_output in extras_mt['task_outputs'].items():
            print(f"  {task_name}: {list(task_output.keys())}")

    # Check that we can get individual task heads
    singing_head = model_mt.get_task_head('singing')
    assert singing_head is not None

    print("\nAll tests passed!")