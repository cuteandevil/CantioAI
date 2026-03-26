"""
Feature extraction module using WORLD vocoder.
"""

import numpy as np
import torch
import pyworld as world
import librosa
from typing import Tuple, Optional, Union, Dict, Any
import logging
from pathlib import Path

from .utils import (
    load_wav, save_wav, normalize_f0, denormalize_f0,
    interpolate_f0, trim_silence, preemphasis, inv_preemphasis
)

logger = logging.getLogger(__name__)


class WorldFeatureExtractor:
    """
    Extract features using WORLD vocoder: F0, spectral envelope (SP), aperiodicity (AP).
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_period: float = 5.0,
        fft_size: int = 1024,
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
        num_mcep: int = 60,
        mcep_alpha: float = 0.41,
        normalize_f0_method: str = "log",
        normalize_sp_method: str = "standard",
        silence_threshold: float = 0.03,
        f0_interpolation: str = "linear"
    ):
        """
        Initialize WORLD feature extractor.

        Args:
            sample_rate: Audio sample rate
            frame_period: Frame shift in milliseconds
            fft_size: FFT size for spectral analysis
            f0_floor: Minimum F0 to consider (Hz)
            f0_ceil: Maximum F0 to consider (Hz)
            num_mcep: Number of mel-cepstral coefficients
            mcep_alpha: All-pass constant for MCEP
            normalize_f0_method: F0 normalization method
            normalize_sp_method: Spectral envelope normalization method
            silence_threshold: Amplitude threshold for silence detection
            f0_interpolation: Method for F0 interpolation ("linear" or "none")
        """
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.fft_size = fft_size
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.num_mcep = num_mcep
        self.mcep_alpha = mcep_alpha
        self.normalize_f0_method = normalize_f0_method
        self.normalize_sp_method = normalize_sp_method
        self.silence_threshold = silence_threshold
        self.f0_interpolation = f0_interpolation

        # Precompute alpha for MCEP
        self._alpha = mcep_alpha

        logger.info(
            f"Initialized WorldFeatureExtractor with sr={sample_rate}, "
            f"frame_period={frame_period}ms, fft_size={fft_size}, "
            f"f0_range=[{f0_floor}, {f0_ceil}]Hz, num_mcep={num_mcep}"
        )

    def extract_features(
        self,
        waveform: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract F0, SP, AP features from waveform using WORLD.

        Args:
            waveform: Input audio signal (numpy array, float32, [-1, 1])
            normalize: Whether to normalize features

        Returns:
            Dictionary containing:
                - f0: Fundamental frequency (Hz)
                - sp: Spectral envelope (world coded spectral envelope)
                - ap: Aperiodicity
                - coded_sp: Mel-cepstral coefficients (if needed)
        """
        # Ensure waveform is float64 for WORLD
        waveform = waveform.astype(np.float64)

        # Extract F0 using Harvest (better for singing voice)
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

        # Post-processing
        if normalize:
            # Normalize F0
            f0 = normalize_f0(f0, method=self.normalize_f0_method)

            # Interpolate F0 to remove zeros if requested
            if self.f0_interpolation != "none":
                f0 = interpolate_f0(f0, kind=self.f0_interpolation)

            # Normalize spectral envelope (codec)
            if self.normalize_sp_method == "standard":
                # Compute mean and std per dimension
                sp_mean = np.mean(coded_sp, axis=0, keepdims=True)
                sp_std = np.std(coded_sp, axis=0, keepdims=True)
                sp_std = np.where(sp_std < 1e-8, 1.0, sp_std)  # Avoid division by zero
                coded_sp = (coded_sp - sp_mean) / sp_std
                # Store statistics for denormalization
                norm_stats = {"sp_mean": sp_mean, "sp_std": sp_std}
            elif self.normalize_sp_method == "minmax":
                sp_min = np.min(coded_sp, axis=0, keepdims=True)
                sp_max = np.max(coded_sp, axis=0, keepdims=True)
                sp_range = sp_max - sp_min
                sp_range = np.where(sp_range < 1e-8, 1.0, sp_range)
                coded_sp = (coded_sp - sp_min) / sp_range
                norm_stats = {"sp_min": sp_min, "sp_max": sp_max}
            else:
                norm_stats = None
        else:
            norm_stats = None

        # Convert back to spectral envelope if needed for synthesis
        if normalize and self.normalize_sp_method != "none":
            # We'll keep coded_sp normalized and decode when needed
            sp_decoded = world.decode_spectral_envelope(coded_sp, self.sample_rate, self.fft_size)
        else:
            sp_decoded = sp

        return {
            "f0": f0.astype(np.float32),          # (T,)
            "sp": sp_decoded.astype(np.float32),  # (T, fft_size//2+1)
            "ap": ap.astype(np.float32),          # (T, fft_size//2+1)
            "coded_sp": coded_sp.astype(np.float32),  # (T, num_mcep)
            "timeaxis": timeaxis.astype(np.float32),  # (T,)
            "norm_stats": norm_stats
        }

    def synthesize(
        self,
        f0: np.ndarray,
        coded_sp: np.ndarray,
        ap: np.ndarray,
        normalize: bool = True,
        norm_stats: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Synthesize waveform from F0, coded spectral envelope, and aperiodicity.

        Args:
            f0: Fundamental frequency (Hz)
            coded_sp: Mel-cepstral coefficients (normalized if normalize=True)
            ap: Aperiodicity
            normalize: Whether features are normalized
            norm_stats: Normalization statistics if normalize=True

        Returns:
            Synthetic waveform
        """
        # Denormalize if needed
        if normalize and norm_stats is not None:
            if self.normalize_sp_method == "standard":
                coded_sp = coded_sp * norm_stats["sp_std"] + norm_stats["sp_mean"]
            elif self.normalize_sp_method == "minmax":
                coded_sp = coded_sp * (norm_stats["sp_max"] - norm_stats["sp_min"]) + norm_stats["sp_min"]
            # For log F0, denormalization happens in vocoder.py

        # Decode spectral envelope
        sp = world.decode_spectral_envelope(coded_sp, self.sample_rate, self.fft_size)

        # Synthesize waveform
        waveform = world.synthesize(f0, sp, ap, self.sample_rate, self.frame_period)
        waveform = waveform.astype(np.float32)

        return waveform

    def process_file(
        self,
        wav_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process a single audio file: load, extract features, optionally save.

        Args:
            wav_path: Path to input audio file
            output_dir: Directory to save extracted features (if None, don't save)

        Returns:
            Dictionary of extracted features
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # Load audio
        waveform, sr = load_wav(str(wav_path), sr=self.sample_rate)
        if sr != self.sample_rate:
            logger.warning(
                f"Sample rate mismatch: expected {self.sample_rate}, got {sr}. "
                f"Resampling..."
            )
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # Optional: trim silence
        waveform = trim_silence(waveform)

        # Extract features
        features = self.extract_features(waveform)

        # Save if output directory specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = wav_path.stem
            np.savez(
                output_dir / f"{base_name}_features.npz",
                **{k: v for k, v in features.items() if k != "norm_stats"},
                norm_stats=features["norm_stats"]
            )
            logger.info(f"Saved features to {output_dir / f'{base_name}_features.npz'}")

        return features

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3")
    ) -> None:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save extracted features
            file_extensions: Tuple of file extensions to process
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))

        if not audio_files:
            logger.warning(f"No audio files found in {input_dir} with extensions {file_extensions}")
            return

        logger.info(f"Processing {len(audio_files)} audio files from {input_dir}")

        for wav_path in audio_files:
            try:
                self.process_file(wav_path, output_dir)
            except Exception as e:
                logger.error(f"Failed to process {wav_path}: {e}")
                continue

        logger.info(f"Finished processing. Features saved to {output_dir}")


def extract_features_from_wav(
    wav_path: Union[str, Path],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Convenience function to extract features from a single WAV file.

    Args:
        wav_path: Path to input audio file
        **kwargs: Arguments passed to WorldFeatureExtractor

    Returns:
        Dictionary of extracted features
    """
    extractor = WorldFeatureExtractor(**kwargs)
    return extractor.process_file(wav_path)


def extract_features_from_dir(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    **kwargs
) -> None:
    """
    Convenience function to extract features from all WAV files in a directory.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save extracted features
        **kwargs: Arguments passed to WorldFeatureExtractor
    """
    extractor = WorldFeatureExtractor(**kwargs)
    extractor.process_directory(input_dir, output_dir)