"""
Audio synthesis module for CantioAI.
"""

import torch
import numpy as np
import pyworld as world
from typing import Tuple, Optional, Union
import logging
from pathlib import Path

from ..models.hybrid_svc import HybridSVC
from ..data.utils import (
    inv_preemphasis, preemphasis, normalize_f0, denormalize_f0,
    trim_silence, load_wav, save_wav
)

logger = logging.getLogger(__name__)


def synthesize_audio(
    model: HybridSVC,
    phoneme_features: torch.Tensor,
    f0: torch.Tensor,
    spk_id: torch.Tensor,
    f0_is_hz: bool = True,
    apply_preemphasis: bool = True,
    preemphasis_coeff: float = 0.97,
    normalize_output: bool = True,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Synthesize audio from linguistic features using the trained model.

    Args:
        model: Trained HybridSVC model
        phoneme_features: Linguistic/phonetic features (T, D_ph)
        f0: Fundamental frequency (T, 1) in Hz
        spk_id: Speaker IDs (T,)
        f0_is_hz: Whether input F0 is in Hz
        apply_preemphasis: Whether to apply preemphasis to output
        preemphasis_coeff: Preemphasis coefficient
        normalize_output: Whether to normalize output audio
        device: Device to run inference on

    Returns:
        waveform: Synthesized audio signal (numpy array, float32)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Set model to eval mode
    model.eval()
    model.to(device)

    # Move inputs to device
    phoneme_features = phoneme_features.to(device)
    f0 = f0.to(device)
    spk_id = spk_id.to(device)

    # Forward pass
    with torch.no_grad():
        sp_pred, f0_quant, _ = model(
            phoneme_features=phoneme_features,
            f0=f0,
            spk_id=spk_id,
            f0_is_hz=f0_is_hz,
            return_quantized_f0=False  # Don't need quantized F0 for synthesis
        )

    # Convert to numpy
    sp_pred = sp_pred.cpu().numpy()  # (T, D_sp)
    f0_hz = f0.cpu().numpy()       # (T, 1)

    # WORLD synthesis: F0 + SP_pred -> spectral envelope
    # Note: WORLD expects F0 in Hz, SP as spectral envelope (world coded)
    sp = world.decode_spectral_envelope(sp_pred, model.spectral_envelope_dim)
    # The above line is incorrect - we need to go from mel-cepstral to world coded
    # Let's fix this:
    # sp_pred is mel-cepstral coefficients (output of model)
    # Convert to world coded spectral envelope
    sp = world.decode_spectral_envelope(sp_pred, fs=24000, fft_size=1024)

    # Synthesize waveform
    waveform = world.synthesize(f0_hz, sp, None, fs=24000, frame_period=5.0)
    # Note: ap=None for voiced speech (can be estimated or set to zero for unvoiced)

    waveform = waveform.astype(np.float32)

    # Post-processing
    if normalize_output:
        # Peak normalize
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

    if apply_preemphasis:
        waveform = inv_preemphasis(waveform, preemphasis_coeff)

    # Optional: trim silence
    waveform = trim_silence(waveform)

    return waveform


class VocoderInference:
    """
    Vocoder-based inference wrapper.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Union[str, Path],
        device: Optional[str] = None
    ):
        """
        Initialize vocoder inference.

        Args:
            model_path: Path to model checkpoint (.pt file)
            config_path: Path to configuration file (.yaml)
            device: Device to run inference on
        """
        import yaml

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize model
        model_config = self.config.get("model", {})
        self.model = HybridSVC(
            phoneme_feature_dim=model_config.get("phoneme_feature_dim", 32),
            spectral_envelope_dim=model_config.get("spectral_envelope_dim", 60),
            speaker_embed_dim=model_config.get("speaker_embed_dim", 128),
            n_speakers=model_config.get("n_speakers", 100),
            use_pitch_quantizer=model_config.get("use_pitch_quantizer", True),
        )

        # Load checkpoint
        checkpoint_epoch = self.model.load_checkpoint(model_path, load_optimizer=False)
        logger.info(f"Loaded model from epoch {checkpoint_epoch}")

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Vocoder settings
        self.vocoder_sample_rate = self.config.get("feature", {}).get("sample_rate", 24000)
        self.vocoder_frame_period = self.config.get("feature", {}).get("frame_period", 5.0)
        self.vocoder_fft_size = self.config.get("feature", {}).get("fft_size", 1024)

        logger.info("VocoderInference initialized")

    def synthesize(
        self,
        phoneme_features: Union[np.ndarray, torch.Tensor],
        f0: Union[np.ndarray, torch.Tensor],
        spk_id: Union[np.ndarray, torch.Tensor],
        f0_is_hz: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize audio from features.

        Args:
            phoneme_features: Linguistic features (T, D_ph)
            f0: Fundamental frequency (T, 1) in Hz
            spk_id: Speaker IDs (T,)
            f0_is_hz: Whether input F0 is in Hz
            **kwargs: Additional arguments for synthesize_audio

        Returns:
            waveform: Synthesized audio signal
        """
        # Convert to torch tensors if needed
        if isinstance(phoneme_features, np.ndarray):
            phoneme_features = torch.from_numpy(phoneme_features)
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0)
        if isinstance(spk_id, np.ndarray):
            spk_id = torch.from_numpy(spk_id).long()

        # Synthesize
        waveform = synthesize_audio(
            self.model,
            phoneme_features,
            f0,
            spk_id,
            f0_is_hz=f0_is_hz,
            device=self.device,
            **kwargs
        )

        return waveform

    def synthesize_from_file(
        self,
        features_path: Union[str, Path],
        f0_path: Optional[Union[str, Path]] = None,
        spk_id_path: Optional[Union[str, Path]] = None,
        output_path: Union[str, Path] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize audio from saved features.

        Args:
            features_path: Path to phoneme features (.npy or .pt)
            f0_path: Path to F0 values (.npy or .pt)
            spk_id_path: Path to speaker IDs (.npy or .pt)
            output_path: Path to save output audio (.wav)
            **kwargs: Additional arguments for synthesize

        Returns:
            waveform: Synthesized audio signal (if output_path is None)
        """
        # Load features
        if features_path.suffix() == ".npy":
            phoneme_features = np.load(features_path)
        else:  # Assume .pt
            phoneme_features = torch.load(features_path)

        if f0_path is not None:
            if f0_path.suffix() == ".npy":
                f0 = np.load(f0_path)
            else:
                f0 = torch.load(f0_path)
        else:
            # Generate constant F0 if not provided (e.g., for zero-shot)
            f0_hz = self.config.get("inference", {}).get("default_f0_hz", 220.0)
            f0 = np.full_like(phoneme_features, f0_hz) if isinstance(phoneme_features, np.ndarray) else \
                   torch.full_like(phoneme_features, f0_hz)

        if spk_id_path is not None:
            if spk_id_path.suffix() == ".npy":
                spk_id = np.load(spk_id_path)
            else:
                spk_id = torch.load(spk_id_path).long()
        else:
            # Use first speaker ID if not provided
            spk_id = np.zeros_like(phoneme_features.shape[:-1] + (1,)) if isinstance(phoneme_features, np.ndarray) else \
                   torch.zeros_like(phoneme_features.shape[:-1] + (1,), dtype=torch.long)

        # Synthesize
        waveform = self.synthesize(
            phoneme_features,
            f0,
            spk_id,
            f0_is_hz=f0_is_hz,
            **kwargs
        )

        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            sf.write(str(output_path), waveform, self.vocoder_sample_rate)
            logger.info(f"Saved synthesized audio to {output_path}")

        return waveform

    def batch_synthesize(
        self,
        batch: Dict[str, Union[np.ndarray, torch.Tensor]],
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize audio from batch dictionary.

        Args:
            batch: Dictionary containing:
                - phoneme_features
                - f0
                - spk_id
            **kwargs: Additional arguments for synthesize

        Returns:
            waveform: Synthesized audio signal
        """
        return self.synthesize(
            batch["phoneme_features"],
            batch["f0"],
            batch["spk_id"],
            f0_is_hz=kwargs.get("f0_is_hz", True),
            **kwargs
        )


if __name__ == "__main__":
    # Simple test
    import yaml

    # Create dummy config
    config = {
        "feature": {"sample_rate": 24000, "frame_period": 5.0, "fft_size": 1024},
        "model": {
            "phoneme_feature_dim": 32,
            "spectral_envelope_dim": 60,
            "speaker_embed_dim": 128,
            "n_speakers": 10,
            "use_pitch_quantizer": True
        },
        "inference": {"default_f0_hz": 220.0}
    }
    config_path = "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Create and save model
    model = HybridSVC(
        phoneme_feature_dim=32,
        spectral_envelope_dim=60,
        speaker_embed_dim=128,
        n_speakers=10,
        use_pitch_quantizer=True
    )

    # Save model checkpoint
    model_path = "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 0,
    }, model_path)

    # Initialize inference
    inferencer = VocoderInference(model_path, config_path)

    # Create dummy inputs
    seq_len = 20
    phoneme_features = np.random.randn(seq_len, 32).astype(np.float32)
    f0_hz = np.full((seq_len, 1), 220.0, dtype=np.float32)  # Constant F0
    spk_id = np.zeros((seq_len,), dtype=np.int64)  # Speaker ID 0

    # Synthesize audio
    waveform = inferencer.synthesize(
        phoneme_features, f0_hz, spk_id, f0_is_hz=True
    )

    print(f"Input shapes:")
    print(f"  phoneme_features: {phoneme_features.shape}")
    print(f"  f0: {f0.shape}")
    print(f"  spk_id: {spk_id.shape}")
    print(f"Output shape: {waveform.shape}")

    # Basic checks
    assert waveform.shape == (seq_len,)
    assert waveform.dtype == np.float32
    assert np.isfinite(waveform).all(), "Output contains NaN or Inf"

    print("VocoderInference test passed.")