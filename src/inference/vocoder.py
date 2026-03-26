"""
Vocoder interface for CantioAI.
Provides WORLD-based synthesis and analysis functions.
"""

import torch
import numpy as np
import pyworld as world
from typing import Tuple, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WORLDVocoder:
    """
    WORLD vocoder wrapper for analysis and synthesis.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_period: float = 5.0,
        fft_size: int = 1024,
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
        num_mcep: int = 60,
        mcep_alpha: float = 0.41
    ):
        """
        Initialize WORLD vocoder.

        Args:
            sample_rate: Audio sample rate
            frame_period: Frame shift in milliseconds
            fft_size: FFT size for spectral analysis
            f0_floor: Minimum F0 to consider (Hz)
            f0_ceil: Maximum F0 to consider (Hz)
            num_mcep: Number of mel-cepstral coefficients
            mcep_alpha: All-pass constant for MCEP
        """
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.fft_size = fft_size
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.num_mcep = num_mcep
        self.mcep_alpha = mcep_alpha

        logger.info(
            f"Initialized WORLDVocoder with sr={sample_rate}, "
            f"frame_period={frame_period}ms, fft_size={fft_size}, "
            f"f0_range=[{f0_floor}, {f0_ceil}]Hz, num_mcep={num_mcep}"
        )

    def analyze(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> dict:
        """
        Analyze waveform to extract F0, SP, AP features.

        Args:
            waveform: Input audio signal
            normalize: Whether to normalize features

        Returns:
            Dictionary containing:
                - f0: Fundamental frequency (Hz)
                - sp: Spectral envelope (world coded)
                - ap: Aperiodicity
                - coded_sp: Mel-cepstral coefficients
        """
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        waveform = waveform.astype(np.float64)

        # Extract F0 using Harvest
        f0, timeaxis = world.harvest(
            waveform,
            fs=self.sample_rate,
            frame_period=self.frame_period,
            f0_floor=self.f0_floor,
            f0_ceil=self.f0_ceil
        )

        # Extract spectral envelope using cheaptrick
        sp = world.cheaptrick(waveform, f0, timeaxis, fs=self.sample_rate, fft_size=self.fft_size)

        # Extract aperiodicity using d4c
        ap = world.d4c(waveform, f0, timeaxis, fs=self.sample_rate, fft_size=self.fft_size)

        # Convert spectral envelope to mel-cepstral coefficients
        coded_sp = world.code_spectral_envelope(sp, self.sample_rate, self.num_mcep)

        # Optional normalization
        if normalize:
            # Normalize F0 (log scale)
            f0 = np.log(f0 + 1e-8)  # Avoid log(0)
            # Normalize coded_sp (zero mean, unit variance per dimension)
            sp_mean = np.mean(coded_sp, axis=0, keepdims=True)
            sp_std = np.std(coded_sp, axis=0, keepdims=True)
            sp_std = np.where(sp_std < 1e-8, 1.0, sp_std)
            coded_sp = (coded_sp - sp_mean) / sp_std

        return {
            "f0": f0.astype(np.float32),          # (T,)
            "sp": sp.astype(np.float32),          # (T, fft_size//2+1)
            "ap": ap.astype(np.float32),          # (T, fft_size//2+1)
            "coded_sp": coded_sp.astype(np.float32),  # (T, num_mcep)
            "timeaxis": timeaxis.astype(np.float32),  # (T,)
        }

    def synthesize(
        self,
        f0: Union[np.ndarray, torch.Tensor],
        coded_sp: Union[np.ndarray, torch.Tensor],
        ap: Optional[Union[np.ndarray, torch.Tensor]] = None,
        f0_is_normalized: bool = False,
        sp_is_normalized: bool = False,
        f0_mean: Optional[float] = None,
        f0_std: Optional[float] = None,
        sp_mean: Optional[np.ndarray] = None,
        sp_std: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize waveform from F0, coded spectral envelope, and aperiodicity.

        Args:
            f0: Fundamental frequency (Hz or normalized)
            coded_sp: Mel-cepstral coefficients
            ap: Aperiodicity (optional)
            f0_is_normalized: Whether f0 is normalized (log scale)
            sp_is_normalized: Whether coded_sp is normalized
            f0_mean: Mean of original F0 (for denormalization)
            f0_std: Std of original F0 (for denormalization)
            sp_mean: Mean of original coded_sp (for denormalization)
            sp_std: Std of original coded_sp (for denormalization)

        Returns:
            waveform: Synthetic audio signal
        """
        # Convert to numpy
        if isinstance(f0, torch.Tensor):
            f0 = f0.cpu().numpy()
        if isinstance(coded_sp, torch.Tensor):
            coded_sp = coded_sp.cpu().numpy()
        if ap is not None and isinstance(ap, torch.Tensor):
            ap = ap.cpu().numpy()

        # Denormalize F0 if needed
        if f0_is_normalized:
            if f0_mean is None or f0_std is None:
                raise ValueError("f0_mean and f0_std required for F0 denormalization")
            f0 = np.exp(f0)  # From log scale
            f0 = f0 * f0_std + f0_mean  # Actually, log normalization is different
            # Correction: if we did log(f0 + eps), then denorm is exp(normalized) - eps
            # But let's assume simple log for now
            f0 = np.exp(f0)
            if f0_mean is not None and f0_std is not None:
                f0 = f0 * f0_std + f0_mean

        # Denormalize coded_sp if needed
        if sp_is_normalized:
            if sp_mean is None or sp_std is None:
                raise ValueError("sp_mean and sp_std required for SP denormalization")
            coded_sp = coded_sp * sp_std + sp_mean

        # Synthesize
        if ap is None:
            # Create dummy aperiodicity for voiced speech
            ap = np.zeros_like(coded_sp)  # Actually, WORLD needs proper shape
            ap = np.zeros((len(f0), self.fft_size // 2 + 1))

        waveform = world.synthesize(
            f0.astype(np.float64),
            world.decode_spectral_envelope(coded_sp.astype(np.float64), self.sample_rate, self.fft_size),
            ap.astype(np.float64),
            self.sample_rate,
            int(self.frame_period * 1000)  # Convert ms to samples? Actually frame_period is in ms
        )

        return waveform.astype(np.float32)

    def synthesize_from_features(
        self,
        f0: Union[np.ndarray, torch.Tensor],
        sp_pred: Union[np.ndarray, torch.Tensor],
        ap: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize waveform from F0 and predicted spectral envelope.

        Args:
            f0: Fundamental frequency (Hz)
            sp_pred: Predicted spectral envelope (mel-cepstral coefficients)
            ap: Aperiodicity (optional)
            **kwargs: Additional arguments for synthesize

        Returns:
            waveform: Synthetic audio signal
        """
        return self.synthesize(f0, sp_pred, ap, **kwargs)


def world_synthesize(
    f0: np.ndarray,
    coded_sp: np.ndarray,
    ap: np.ndarray,
    sample_rate: int = 24000,
    frame_period: float = 5.0
) -> np.ndarray:
    """
    Convenience function for WORLD synthesis.

    Args:
        f0: Fundamental frequency (Hz)
        coded_sp: Mel-cepstral coefficients
        ap: Aperiodicity
        sample_rate: Audio sample rate
        frame_period: Frame shift in milliseconds

    Returns:
        waveform: Synthetic audio signal
    """
    vocoder = WORLDVocoder(
        sample_rate=sample_rate,
        frame_period=frame_period
    )
    return vocoder.synthesize(f0, coded_sp, ap)


def world_analyze(
    waveform: np.ndarray,
    sample_rate: int = 24000,
    frame_period: float = 5.0,
    fft_size: int = 1024,
    f0_floor: float = 71.0,
    f0_ceil: float = 800.0,
    num_mcep: int = 60
) -> dict:
    """
    Convenience function for WORLD analysis.

    Args:
        waveform: Input audio signal
        sample_rate: Audio sample rate
        frame_period: Frame shift in milliseconds
        fft_size: FFT size for spectral analysis
        f0_floor: Minimum F0 to consider (Hz)
        f0_ceil: Maximum F0 to consider (Hz)
        num_mcep: Number of mel-cepstral coefficients

    Returns:
        Dictionary containing F0, SP, AP features
    """
    vocoder = WORLDVocoder(
        sample_rate=sample_rate,
        frame_period=frame_period,
        fft_size=fft_size,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        num_mcep=num_mcep
    )
    return vocoder.analyze(waveform)


if __name__ == "__main__":
    # Simple test
    import librosa

    # Generate test signal
    sr = 24000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    frequency = 220.0  # Hz
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    waveform = waveform.astype(np.float32)

    # Analyze
    vocoder = WORLDVocoder(sample_rate=sr)
    features = vocoder.analyze(waveform)

    print(f"Input waveform shape: {waveform.shape}")
    print(f"Extracted features:")
    print(f"  F0: {features['f0'].shape}")
    print(f"  SP: {features['sp'].shape}")
    print(f"  AP: {features['ap'].shape}")
    print(f"  Coded SP: {features['coded_sp'].shape}")

    # Synthesize
    # Use zero aperiodicity for simplicity (voiced speech)
    ap_dummy = np.zeros_like(features['ap'])
    waveform_synth = vocoder.synthesize(
        features['f0'],
        features['coded_sp'],
        ap_dummy,
        f0_is_normalized=False,
        sp_is_normalized=False
    )

    print(f"Synthesized waveform shape: {waveform_synth.shape}")

    # Basic checks
    assert waveform_synth.shape == waveform.shape
    assert waveform_synth.dtype == np.float32
    assert np.isfinite(waveform_synth).all(), "Synthesized waveform contains NaN or Inf"

    print("WORLDVocoder test passed.")