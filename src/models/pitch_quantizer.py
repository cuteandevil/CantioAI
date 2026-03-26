"""
Differentiable pitch quantization module.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union


class DifferentiablePitchQuantizer(nn.Module):
    """
    Differentiable pitch quantization module.

    Maps a continuous fundamental frequency (f0) in Hz to the nearest
    semitone of a target scale (default: 12-TET with A4 = 440 Hz) while
    preserving gradients via a straight-through estimator (STE).

    The forward pass returns the quantized f0 value; during backward,
    gradients are passed through as if the quantization were identity
    (i.e., d/dx round(x) ≈ 1). This enables end-to-end training of models
    that should produce musically accurate pitches.

    Optional helper `get_scale_notes` can generate target frequencies for
    arbitrary scales (e.g., just intonation, pentatonic).

    Shapes:
        Input:  f0_cont (B, T, 1) in Hz
        Output: f0_quantized (B, T, 1) in Hz
    """

    def __init__(
        self,
        ref_freq: float = 440.0,   # Reference frequency for A4
        ref_midi: int = 69,        # MIDI number of reference note (A4)
        octaves: int = 10,         # How many octaves to cover (both directions)
        use_ste: bool = True,      # Straight-through estimator flag
    ):
        super().__init__()
        self.ref_freq = ref_freq
        self.ref_midi = ref_midi
        self.octaves = octaves
        self.use_ste = use_ste

        # Precompute target frequencies for all MIDI notes in range
        min_midi = ref_midi - 12 * octaves
        max_midi = ref_midi + 12 * octaves
        midi_numbers = torch.arange(min_midi, max_midi + 1, dtype=torch.float32)
        self.register_buffer("midi_numbers", midi_numbers)  # (N,)
        self.target_freqs = self._midi_to_hz(midi_numbers)  # (N,)

    @staticmethod
    def _hz_to_midi(freq_hz: torch.Tensor, ref_freq: float = 440.0, ref_midi: int = 69) -> torch.Tensor:
        """
        Convert frequency in Hz to MIDI note number (float).
        """
        # Avoid log(0) by clamping
        freq_hz = torch.clamp(freq_hz, min=1e-8)
        return ref_midi + 12 * torch.log2(freq_hz / ref_freq)

    @staticmethod
    def _midi_to_hz(midi: torch.Tensor, ref_freq: float = 440.0, ref_midi: int = 69) -> torch.Tensor:
        """
        Convert MIDI note number (float or int) to frequency in Hz.
        """
        return ref_freq * 2.0 ** ((midi - ref_midi) / 12.0)

    def forward(self, f0_cont: torch.Tensor) -> torch.Tensor:
        """
        Quantize f0_cont to nearest semitone using STE.
        """
        # f0_cont: (B, T, 1) or (B, T)
        if f0_cont.dim() == 3 and f0_cont.shape[2] == 1:
            f0_squeeze = f0_cont.squeeze(-1)  # (B, T)
        else:
            f0_squeeze = f0_cont

        # Convert to MIDI (float)
        midi_float = self._hz_to_midi(f0_squeeze, self.ref_freq, self.ref_midi)  # (B, T)

        # Find nearest MIDI index by rounding
        midi_rounded = torch.round(midi_float)  # (B, T)

        # Straight-through estimator: treat rounding as identity for backward
        if self.use_ste:
            # midi_out = midi_float + (midi_rounded - midi_float).detach()
            midi_out = midi_float + (midi_rounded - midi_float).detach()
        else:
            midi_out = midi_rounded

        # Convert back to Hz
        f0_quantized = self._midi_to_hz(midi_out, self.ref_freq, self.ref_midi)  # (B, T)

        # Restore original shape
        if f0_cont.dim() == 3 and f0_cont.shape[2] == 1:
            f0_quantized = f0_quantized.unsqueeze(-1)

        return f0_quantized

    def quantize(self, f0_cont: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward, kept for explicitness.
        """
        return self.forward(f0_cont)

    def get_scale_notes(
        self,
        root_midi: int = 69,
        scale_intervals: Optional[List[int]] = None,
        octaves: int = 2,
    ) -> torch.Tensor:
        """
        Generate target frequencies for a musical scale.

        Args:
            root_midi: MIDI number of the scale root (default A4=69).
            scale_intervals: List of semitone steps from root (e.g., [0,2,4,5,7,9,11] for major).
                            If None, returns chromatic scale.
            octaves: Number of octaves above and below root to include.

        Returns:
            Tensor of shape (N,) containing frequencies in Hz.
        """
        if scale_intervals is None:
            # Chromatic scale: all semitones
            intervals = list(range(-12 * octaves, 12 * octaves + 1))
        else:
            # Generate repeating pattern across octaves
            intervals = []
            for o in range(-octaves, octaves + 1):
                for interval in scale_intervals:
                    intervals.append(o * 12 + interval)
            intervals = sorted(set(intervals))

        midi_numbers = torch.tensor(intervals, dtype=torch.float32) + root_midi
        return self._midi_to_hz(midi_numbers, self.ref_freq, self.ref_midi)


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 30
    f0_cont = torch.rand(batch_size, seq_len, 1) * 400 + 80  # range ~80-480 Hz

    quantizer = DifferentiablePitchQuantizer(ref_freq=440.0, use_ste=True)
    f0_quant = quantizer(f0_cont)

    print(f"Input shape: {f0_cont.shape}")
    print(f"Output shape: {f0_quant.shape}")
    # Check that output frequencies are close to nearest semitone
    midi_out = quantizer._hz_to_midi(f0_quant.squeeze(-1))
    midi_rounded = torch.round(midi_out)
    max_dev = torch.max(torch.abs(midi_out - midi_rounded)).item()
    print(f"Max deviation from nearest semitone (in MIDI units): {max_dev:.6f}")
    assert max_dev < 1e-3, "Quantization not accurate"
    print("DifferentiablePitchQuantizer test passed.")