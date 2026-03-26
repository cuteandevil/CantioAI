"""
Hybrid Vocoder System: Combines WORLD source-filter control with neural enhancement
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Tuple, Optional, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.utils import load_wav
from .hifigan import ControlAwareHiFiGAN


class WorldSynthesizer:
    """WORLD vocoder synthesizer for precise source-filter control."""
    def __init__(self,
                 sample_rate: int = 24000,
                 frame_period: float = 5.0,
                 fft_size: int = 1024,
                 f0_floor: float = 71.0,
                 f0_ceil: float = 800.0,
                 num_mcep: int = 60,
                 mcep_alpha: float = 0.41):
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.fft_size = fft_size
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.num_mcep = num_mcep
        self.mcep_alpha = mcep_alpha

    def synthesize(self,
                   f0: np.ndarray,
                   sp: np.ndarray,
                   ap: np.ndarray) -> np.ndarray:
        """Synthesize waveform from F0, SP, AP.
        Args:
            f0: fundamental frequency (Hz) - shape (T,)
            sp: spectral envelope (world coded) - shape (T, fft_size//2+1)
            ap: aperiodicity - shape (T, fft_size//2+1)
        Returns:
            waveform: synthetic audio - shape (T * hop_length,)
        """
        # Ensure correct dtype for WORLD
        f0 = f0.astype(np.float64)
        sp = sp.astype(np.float64)
        ap = ap.astype(np.float64)

        # Convert frame period to samples for WORLD synthesis
        frame_period_samples = int(self.frame_period * self.sample_rate / 1000)

        # Synthesize using WORLD
        audio = librosa.util.fix_length(
            librosa.core.synthesize(
                f0,
                sp,
                ap,
                fs=self.sample_rate,
                frame_period=frame_period_samples
            ),
            size=len(f0) * int(self.frame_period * self.sample_rate / 1000)
        )
        return audio.astype(np.float32)


class FusionController(nn.Module):
    """Controls fusion of WORLD and neural audio outputs."""
    def __init__(self,
                 fusion_type: str = "adaptive",
                 sample_rate: int = 24000):
        super().__init__()
        self.fusion_type = fusion_type
        self.sample_rate = sample_rate

        # Adaptive fusion weights network
        if fusion_type == "adaptive":
            self.fusion_net = nn.Sequential(
                nn.Conv1d(2, 16, 3, 1, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(16, 2, 3, 1, padding=1),
                nn.Softmax(dim=1)
            )
        else:
            self.fusion_net = None

    def forward(self,
                world_audio: torch.Tensor,
                neural_audio: torch.Tensor,
                control_params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Fuse WORLD and neural audio.
        Args:
            world_audio: waveform from WORLD vocoder (B, 1, T_w)
            neural_audio: waveform from neural vocoder (B, 1, T_n)
            control_params: dict containing control parameters (F0, SP, AP, etc.)
        Returns:
            fused audio (B, 1, T_fused)
        """
        # Ensure same length (use neural audio length as reference)
        T_target = neural_audio.size(2)
        if world_audio.size(2) != T_target:
            world_audio = F.interpolate(
                world_audio,
                size=T_target,
                mode='linear',
                align_corners=False
            )

        if self.fusion_type == "adaptive" and self.fusion_net is not None:
            # Stack audio signals for fusion network input
            audio_stack = torch.cat([world_audio, neural_audio], dim=1)  # (B, 2, T)
            weights = self.fusion_net(audio_stack)  # (B, 2, T)
            fused = weights[:, 0:1, :] * world_audio + weights[:, 1:2, :] * neural_audio
        elif self.fusion_type == "fixed":
            # Fixed 50/50 fusion
            fused = 0.5 * world_audio + 0.5 * neural_audio
        else:
            # Default to neural audio (enhancement focus)
            fused = neural_audio

        return fused


class HybridVocoderSystem(nn.Module):
    """Hybrid vocoder system preserving source-filter control with neural enhancement."""
    def __init__(self,
                 config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.vocoder_mode = config.get("vocoder_mode", "hybrid")  # hybrid, neural_only, world_only
        self.sample_rate = config.get("sample_rate", 24000)
        self.frame_period = config.get("frame_period", 5.0)
        self.fft_size = config.get("fft_size", 1024)

        # Initialize WORLD feature extractor (for analysis if needed)
        from ..data.preprocess import WorldFeatureExtractor
        self.world_extractor = WorldFeatureExtractor(
            sample_rate=self.sample_rate,
            frame_period=self.frame_period,
            fft_size=self.fft_size,
            f0_floor=config.get("f0_floor", 71.0),
            f0_ceil=config.get("f0_ceil", 800.0),
            num_mcep=config.get("num_mcep", 60),
            mcep_alpha=config.get("mcep_alpha", 0.41)
        )

        # Initialize WORLD synthesizer (precise control path)
        if self.vocoder_mode in ["world_only", "hybrid"]:
            self.world_synthesizer = WorldSynthesizer(
                sample_rate=self.sample_rate,
                frame_period=self.frame_period,
                fft_size=self.fft_size,
                f0_floor=config.get("f0_floor", 71.0),
                f0_ceil=config.get("f0_ceil", 800.0),
                num_mcep=config.get("num_mcep", 60),
                mcep_alpha=config.get("mcep_alpha", 0.41)
            )

        # Initialize neural vocoder (enhancement path)
        if self.vocoder_mode in ["neural_only", "hybrid"]:
            self.neural_vocoder = ControlAwareHiFiGAN(
                mel_channels=config.get("mel_channels", 80),
                control_dim=config.get("control_dim", 3),  # F0, SP, AP
                upsample_rates=tuple(config.get("upsample_rates", [8, 8, 2, 2])),
                upsample_kernel_sizes=tuple(config.get("upsample_kernel_sizes", [16, 16, 4, 4])),
                resblock_kernel_sizes=tuple(config.get("resblock_kernel_sizes", [3, 7, 11])),
                resblock_dilation_sizes=tuple(tuple(x) for x in config.get(
                    "resblock_dilation_sizes", [[1, 1], [3, 5], [7, 11]])),
                gen_base_channels=config.get("gen_base_channels", 512),
                num_control_layers=config.get("num_control_layers", 4)
            )

        # Initialize fusion controller (for hybrid mode)
        if self.vocoder_mode == "hybrid":
            self.fusion_controller = FusionController(
                fusion_type=config.get("fusion_type", "adaptive"),
                sample_rate=self.sample_rate
            )

        # Precompute mel basis for world_to_mel conversion
        self._precompute_mel_basis()

    def _precompute_mel_basis(self):
        """Precompute mel basis matrix for efficient conversion."""
        n_mels = self.config.get("mel_channels", 80)
        self.mel_basis = np.array(
            librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.fft_size,
                n_mels=n_mels,
                fmin=0,
                fmax=None,
            )
        ).T  # Shape: [n_fft//2+1, n_mels]

    def world_to_mel(self,
                     f0: torch.Tensor,
                     sp: torch.Tensor,
                     ap: torch.Tensor) -> torch.Tensor:
        """Convert WORLD parameters to mel-spectrogram.
        Args:
            f0: fundamental frequency (Hz) - (B, 1, T)
            sp: spectral envelope (world coded) - (B, sp_dim, T)
            ap: aperiodicity - (B, ap_dim, T)
        Returns:
            log-mel spectrogram - (B, mel_channels, T)
        """
        # Note: sp from WORLD is linear spectrogram magnitude
        # Convert to mel scale using precomputed basis
        sp_np = sp.detach().cpu().numpy() if sp.is_cuda else sp.numpy()
        mel_spectrogram = np.dot(sp_np, self.mel_basis)  # [B, T, n_mels]
        mel_spectrogram = torch.from_numpy(mel_spectrogram).to(sp.device)
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)  # [B, n_mels, T]

        # Apply log compression
        log_mel = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        return log_mel

    def synthesize_with_control(self,
                              f0: torch.Tensor,
                              sp: torch.Tensor,
                              ap: torch.Tensor,
                              control_params: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Synthesize audio with precise source-filter control.
        Args:
            f0: fundamental frequency (Hz) - (B, 1, T)
            sp: spectral envelope (world coded) - (B, sp_dim, T)
            ap: aperiodicity - (B, ap_dim, T)
            control_params: dict containing control parameters (e.g., breathiness, brightness)
        Returns:
            waveform: synthetic audio - (B, 1, T_wav)
        """
        if control_params is None:
            control_params = {}

        # Initialize outputs
        world_audio = None
        neural_audio = None

        # Path 1: WORLD synthesizer (precise control)
        if self.vocoder_mode in ["world_only", "hybrid"]:
            # Convert to numpy for WORLD synthesis
            f0_np = f0.detach().cpu().numpy()
            sp_np = sp.detach().cpu().numpy()
            ap_np = ap.detach().cpu().numpy()

            # WORLD expects 1D arrays per batch item
            world_audio_list = []
            for i in range(f0_np.shape[0]):
                audio_i = self.world_synthesizer.synthesize(
                    f0_np[i, 0, :],  # (T,)
                    sp_np[i, :, :],  # (T, sp_dim) - note: WORLD sp is [T, fft_size//2+1]
                    ap_np[i, :, :]   # (T, ap_dim)
                )
                world_audio_list.append(audio_i)

            # Pad to same length and convert to tensor
            max_len = max(len(a) for a in world_audio_list)
            world_audio_padded = []
            for audio in world_audio_list:
                padded = np.pad(audio, (0, max_len - len(a)), 'constant')
                world_audio_padded.append(padded)
            world_audio = torch.from_numpy(np.stack(world_audio_padded)).float()
            world_audio = world_audio.unsqueeze(1).to(f0.device)  # (B, 1, T_wav)

        # Path 2: Neural vocoder (enhanced quality)
        if self.vocoder_mode in ["neural_only", "hybrid"]:
            # Convert WORLD parameters to mel-spectrogram
            mel = self.world_to_mel(f0, sp, ap)  # (B, mel_channels, T)

            # Generate audio with control parameters
            # Extract control signals for neural vocoder
            f0_ctrl = f0 if "f0" in control_params else None
            sp_ctrl = sp if "sp" in control_params else None
            ap_ctrl = ap if "ap" in control_params else None

            neural_audio = self.neural_vocoder(
                mel=mel,
                f0=f0_ctrl,
                sp=sp_ctrl,
                ap=ap_ctrl
            )  # (B, 1, T_n)

        # Select or fuse outputs
        if self.vocoder_mode == "world_only":
            return world_audio
        elif self.vocoder_mode == "neural_only":
            return neural_audio
        else:  # hybrid mode
            # Ensure same length for fusion
            T_target = world_audio.size(2) if world_audio is not None else neural_audio.size(2)
            if world_audio is not None and world_audio.size(2) != T_target:
                world_audio = F.interpolate(
                    world_audio,
                    size=T_target,
                    mode='linear',
                    align_corners=False
                )
            if neural_audio is not None and neural_audio.size(2) != T_target:
                neural_audio = F.interpolate(
                    neural_audio,
                    size=T_target,
                    mode='linear',
                    align_corners=False
                )

            # Fuse using controller
            fused_audio = self.fusion_controller(
                world_audio=world_audio,
                neural_audio=neural_audio,
                control_params=control_params
            )
            return fused_audio