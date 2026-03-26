"""
Audio utility functions for CantioAI.
"""

import numpy as np
import torch
import librosa
import pyworld as world
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_wav(wav_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return waveform and sample rate.

    Args:
        wav_path: Path to audio file
        sr: Target sample rate (None to keep original)

    Returns:
        waveform: Audio signal as numpy array
        sample_rate: Sample rate of the audio

    Raises:
        FileNotFoundError: If wav_path does not exist
        ValueError: If audio file cannot be loaded
    """
    try:
        waveform, sample_rate = librosa.load(wav_path, sr=sr, mono=True)
        return waveform, sample_rate
    except Exception as e:
        logger.error(f"Failed to load audio file {wav_path}: {e}")
        raise


def save_wav(wav_path: str, waveform: np.ndarray, sample_rate: int) -> None:
    """
    Save waveform to audio file.

    Args:
        wav_path: Output path for audio file
        waveform: Audio signal as numpy array
        sample_rate: Sample rate of the audio
    """
    try:
        librosa.output.write_wav(wav_path, waveform, sample_rate)
    except AttributeError:
        # Newer versions of librosa use soundfile
        import soundfile as sf
        sf.write(wav_path, waveform, sample_rate)
    except Exception as e:
        logger.error(f"Failed to save audio file {wav_path}: {e}")
        raise


def normalize_f0(f0: np.ndarray, method: str = "log") -> np.ndarray:
    """
    Normalize fundamental frequency.

    Args:
        f0: Fundamental frequency array (can contain zeros for unvoiced)
        method: Normalization method ("log", "standard", "minmax")

    Returns:
        Normalized F0 array with same shape as input
    """
    f0_norm = f0.copy()
    voiced = f0 > 0

    if not np.any(voiced):
        logger.warning("No voiced frames found in F0 array")
        return f0_norm

    if method == "log":
        f0_norm[voiced] = np.log(f0[voiced])
    elif method == "standard":
        mean = np.mean(f0[voiced])
        std = np.std(f0[voiced])
        if std > 0:
            f0_norm[voiced] = (f0[voiced] - mean) / std
        else:
            f0_norm[voiced] = 0.0
    elif method == "minmax":
        min_val = np.min(f0[voiced])
        max_val = np.max(f0[voiced])
        if max_val > min_val:
            f0_norm[voiced] = (f0[voiced] - min_val) / (max_val - min_val)
        else:
            f0_norm[voiced] = 0.0
    else:
        raise ValueError(f"Unknown F0 normalization method: {method}")

    return f0_norm


def denormalize_f0(f0_norm: np.ndarray, method: str = "log",
                   f0_mean: Optional[float] = None, f0_std: Optional[float] = None,
                   f0_min: Optional[float] = None, f0_max: Optional[float] = None) -> np.ndarray:
    """
    Denormalize fundamental frequency.

    Args:
        f0_norm: Normalized F0 array
        method: Normalization method used ("log", "standard", "minmax")
        f0_mean: Mean of original F0 (for standard normalization)
        f0_std: Std of original F0 (for standard normalization)
        f0_min: Min of original F0 (for minmax normalization)
        f0_max: Max of original F0 (for minmax normalization)

    Returns:
        Denormalized F0 array with same shape as input
    """
    f0 = f0_norm.copy()
    voiced = f0_norm != 0  # Assuming zero represents unvoiced

    if not np.any(voiced):
        logger.warning("No voiced frames found in normalized F0 array")
        return f0

    if method == "log":
        f0[voiced] = np.exp(f0_norm[voiced])
    elif method == "standard":
        if f0_mean is None or f0_std is None:
            raise ValueError("f0_mean and f0_std required for standard denormalization")
        f0[voiced] = f0_norm[voiced] * f0_std + f0_mean
    elif method == "minmax":
        if f0_min is None or f0_max is None:
            raise ValueError("f0_min and f0_max required for minmax denormalization")
        f0[voiced] = f0_norm[voiced] * (f0_max - f0_min) + f0_min
    else:
        raise ValueError(f"Unknown F0 normalization method: {method}")

    return f0


def interpolate_f0(f0: np.ndarray, kind: str = "linear") -> np.ndarray:
    """
    Interpolate F0 to remove zeros (unvoiced frames).

    Args:
        f0: Fundamental frequency array (can contain zeros)
        kind: Interpolation method ("linear", "none")

    Returns:
        F0 array with zeros replaced by interpolated values
    """
    if kind == "none":
        return f0

    f0_interp = f0.copy()
    voiced = f0 > 0

    if not np.any(voiced):
        logger.warning("No voiced frames found, returning original F0")
        return f0_interp

    if kind == "linear":
        # Find indices of voiced frames
        voiced_indices = np.where(voiced)[0]
        # Interpolate using librosa (handles edge cases)
        f0_interp = np.interp(
            np.arange(len(f0)),
            voiced_indices,
            f0[voiced],
            left=f0[voiced_indices[0]],
            right=f0[voiced_indices[-1]]
        )
    else:
        raise ValueError(f"Unknown interpolation method: {kind}")

    return f0_interp


def trim_silence(waveform: np.ndarray, top_db: int = 60,
                 frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Remove leading and trailing silence from audio waveform.

    Args:
        waveform: Input audio signal
        top_db: Threshold (in decibels) below reference to consider as silence
        frame_length: Frame length for STFT
        hop_length: Hop length for STFT

    Returns:
        Trimmed audio waveform
    """
    try:
        trimmed, _ = librosa.effects.trim(
            waveform,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return trimmed
    except Exception as e:
        logger.warning(f"Failed to trim silence: {e}. Returning original waveform.")
        return waveform


def preemphasis(waveform: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply preemphasis filter to waveform.

    Args:
        waveform: Input audio signal
        coeff: Preemphasis coefficient

    Returns:
        Preemphasized waveform
    """
    return np.append(waveform[0], waveform[1:] - coeff * waveform[:-1])


def inv_preemphasis(waveform: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply inverse preemphasis filter to waveform.

    Args:
        waveform: Input audio signal
        coeff: Preemphasis coefficient

    Returns:
        Original waveform
    """
    y = np.zeros_like(waveform)
    y[0] = waveform[0]
    for i in range(1, len(waveform)):
        y[i] = waveform[i] + coeff * y[i-1]
    return y