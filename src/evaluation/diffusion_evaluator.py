"""
Evaluation framework for diffusion models in CantioAI.
Provides comprehensive evaluation metrics for audio quality, control preservation,
and multi-task compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import os
import sys
from pathlib import Path

# Add the src directory to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

logger = logging.getLogger(__name__)


class DiffusionEvaluator:
    """
    Comprehensive evaluator for diffusion models in CantioAI.
    Evaluates audio quality, control parameter preservation, and multi-task compatibility.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize diffusion evaluator.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Evaluation settings
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
        self.hop_length = self.config.get("audio", {}).get("hop_length", 240)  # 10ms at 24kHz
        self.win_length = self.config.get("audio", {}).get("win_length", 960)  # 40ms at 24kHz

        # Metric weights
        self.metric_weights = self.config.get("evaluation", {}).get("metric_weights", {
            "pesq": 0.25,
            "stoi": 0.25,
            "f0_rmse": 0.2,
            "sp_distortion": 0.2,
            "ap_correlation": 0.1
        })

        # Initialize objective metrics if available
        self._init_objective_metrics()

    def _init_objective_metrics(self):
        """Initialize objective metric calculators if libraries are available."""
        try:
            # Try to import PESQ
            import pesq
            self.pesq_available = True
            self.logger.info("PESQ metric available")
        except ImportError:
            self.pesq_available = False
            self.logger.warning("PESQ not available - install with: pip install pesq")

        try:
            # Try to import STOI
            from stoi import stoi
            self.stoi_available = True
            self.logger.info("STOI metric available")
        except ImportError:
            self.stoi_available = False
            self.logger.warning("STOI not available - install with: pip install pystoi")

    def evaluate_audio_quality(self,
                              reference_audio: torch.Tensor,
                              enhanced_audio: torch.Tensor,
                              sample_rate: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate audio quality using objective metrics.

        Args:
            reference_audio: Reference audio tensor (B, 1, T) or (B, T)
            enhanced_audio: Enhanced audio tensor (B, 1, T) or (B, T)
            sample_rate: Audio sample rate (if None, use config default)

        Returns:
            Dictionary of audio quality metrics
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Ensure proper shape
        if reference_audio.dim() == 3:
            reference_audio = reference_audio.squeeze(1)  # (B, T)
        if enhanced_audio.dim() == 3:
            enhanced_audio = enhanced_audio.squeeze(1)  # (B, T)

        # Convert to numpy for metric computation
        ref_np = reference_audio.detach().cpu().numpy()
        enh_np = enhanced_audio.detach().cpu().numpy()

        metrics = {}

        # PESQ (Perceptual Evaluation of Speech Quality)
        if self.pesq_available:
            try:
                pesq_scores = []
                for i in range(ref_np.shape[0]):
                    # PESQ expects 1D arrays
                    score = pesq.pesq(sample_rate, ref_np[i], enh_np[i], 'wb')
                    pesq_scores.append(score)
                metrics["pesq"] = np.mean(pesq_scores)
            except Exception as e:
                self.logger.warning(f"PESQ computation failed: {e}")
                metrics["pesq"] = -1.0  # Invalid value
        else:
            metrics["pesq"] = -1.0  # Placeholder

        # STOI (Short-Time Objective Intelligibility)
        if self.stoi_available:
            try:
                stoi_scores = []
                for i in range(ref_np.shape[0]):
                    score = stoi(ref_np[i], enh_np[i], sample_rate, extended=False)
                    stoi_scores.append(score)
                metrics["stoi"] = np.mean(stoi_scores)
            except Exception as e:
                self.logger.warning(f"STOI computation failed: {e}")
                metrics["stoi"] = -1.0  # Invalid value
        else:
            metrics["stoi"] = -1.0  # Placeholder

        # SNR improvement
        try:
            snr_ref = self._compute_snr(ref_np)
            snr_enh = self._compute_snr(enh_np)
            metrics["snr_improvement"] = snr_enh - snr_ref
        except Exception as e:
            self.logger.warning(f"SNR computation failed: {e}")
            metrics["snr_improvement"] = 0.0

        # Log-spectral distance
        try:
            lsd = self._compute_log_spectral_distance(ref_np, enh_np)
            metrics["log_spectral_distance"] = lsd
        except Exception as e:
            self.logger.warning(f"Log-spectral distance computation failed: {e}")
            metrics["log_spectral_distance"] = float('inf')

        return metrics

    def evaluate_control_preservation(self,
                                    reference_f0: torch.Tensor,
                                    reference_sp: torch.Tensor,
                                    reference_ap: torch.Tensor,
                                    enhanced_f0: torch.Tensor,
                                    enhanced_sp: torch.Tensor,
                                    enhanced_ap: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate how well control parameters are preserved.

        Args:
            reference_f0: Reference F0 (B, T, 1) or (B, T)
            reference_sp: Reference spectral parameters (B, T, D_sp) or (B, T, D_sp)
            reference_ap: Reference aperiodicity parameters (B, T, 1) or (B, T)
            enhanced_f0: Enhanced F0 (B, T, 1) or (B, T)
            enhanced_sp: Enhanced spectral parameters (B, T, D_sp) or (B, T, D_sp)
            enhanced_ap: Enhanced aperiodicity parameters (B, T, 1) or (B, T)

        Returns:
            Dictionary of control preservation metrics
        """
        # Ensure proper shapes
        def _ensure_shape(tensor):
            if tensor.dim() == 2:
                return tensor.unsqueeze(-1)  # (B, T) -> (B, T, 1)
            elif tensor.dim() == 3:
                return tensor  # (B, T, D)
            else:
                raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        ref_f0 = _ensure_shape(reference_f0)
        ref_sp = _ensure_shape(reference_sp)
        ref_ap = _ensure_shape(reference_ap)
        enh_f0 = _ensure_shape(enhanced_f0)
        enh_sp = _ensure_shape(enhanced_sp)
        enh_ap = _ensure_shape(enhanced_ap)

        metrics = {}

        # F0 RMSE (Root Mean Square Error)
        try:
            f0_rmse = torch.sqrt(F.mse_loss(enh_f0, ref_f0)).item()
            metrics["f0_rmse"] = f0_rmse
            metrics["f0_rmse_cents"] = f0_rmse * 1200 / np.log(2)  # Convert to cents
        except Exception as e:
            self.logger.warning(f"F0 RMSE computation failed: {e}")
            metrics["f0_rmse"] = float('inf')
            metrics["f0_rmse_cents"] = float('inf')

        # Spectral distortion
        try:
            sp_dist = F.mse_loss(enh_sp, ref_sp).item()
            metrics["sp_distortion"] = sp_dist
            metrics["sp_distortion_db"] = 10 * np.log10(sp_dist + 1e-10)
        except Exception as e:
            self.logger.warning(f"Spectral distortion computation failed: {e}")
            metrics["sp_distortion"] = float('inf')
            metrics["sp_distortion_db"] = float('inf')

        # Aperiodicity correlation
        try:
            # Flatten for correlation computation
            ref_ap_flat = ref_ap.view(ref_ap.shape[0], -1)
            enh_ap_flat = enh_ap.view(enh_ap.shape[0], -1)
            corr = F.cosine_similarity(ref_ap_flat, enh_ap_flat, dim=1).mean().item()
            metrics["ap_correlation"] = corr
        except Exception as e:
            self.logger.warning(f"Aperiodicity correlation computation failed: {e}")
            metrics["ap_correlation"] = 0.0

        # F0 correlation
        try:
            ref_f0_flat = ref_f0.view(ref_f0.shape[0], -1)
            enh_f0_flat = enh_f0.view(enh_f0.shape[0], -1)
            f0_corr = F.cosine_similarity(ref_f0_flat, enh_f0_flat, dim=1).mean().item()
            metrics["f0_correlation"] = f0_corr
        except Exception as e:
            self.logger.warning(f"F0 correlation computation failed: {e}")
            metrics["f0_correlation"] = 0.0

        # Spectral correlation
        try:
            ref_sp_flat = ref_sp.view(ref_sp.shape[0], -1)
            enh_sp_flat = enh_sp.view(enh_sp.shape[0], -1)
            sp_corr = F.cosine_similarity(ref_sp_flat, enh_sp_flat, dim=1).mean().item()
            metrics["sp_correlation"] = sp_corr
        except Exception as e:
            self.logger.warning(f"Spectral correlation computation failed: {e}")
            metrics["sp_correlation"] = 0.0

        return metrics

    def evaluate_multi_task_compatibility(self,
                                        model_outputs: Dict[str, torch.Tensor],
                                        task_targets: Dict[str, torch.Tensor],
                                        task_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Evaluate multi-task compatibility of the diffusion-enhanced system.

        Args:
            model_outputs: Dictionary of model outputs by task
            task_targets: Dictionary of target values by task
            task_weights: Optional weights for each task

        Returns:
            Dictionary of multi-task compatibility metrics
        """
        if task_weights is None:
            # Equal weights if not provided
            task_weights = {task: 1.0 for task in model_outputs.keys()}

        metrics = {}
        total_weighted_loss = 0.0
        total_weight = 0.0

        # Evaluate each task
        for task_name in model_outputs.keys():
            if task_name not in task_targets:
                self.logger.warning(f"No target provided for task: {task_name}")
                continue

            outputs = model_outputs[task_name]
            targets = task_targets[task_name]
            weight = task_weights.get(task_name, 1.0)

            # Compute task-specific loss
            try:
                if task_name in ['singing', 'speech']:
                    # For conversion tasks, compute L1 loss on spectral parameters
                    if 'sp_pred' in outputs and 'sp' in targets:
                        task_loss = F.l1_loss(outputs['sp_pred'], targets['sp']).item()
                    elif 'sp' in outputs and 'sp' in targets:
                        task_loss = F.l1_loss(outputs['sp'], targets['sp']).item()
                    else:
                        task_loss = 0.0
                elif task_name == 'noise_robustness':
                    # For noise robustness, compute mask-related losses
                    if 'mask_pred' in outputs and 'mask' in targets:
                        task_loss = F.binary_cross_entropy_with_logits(
                            outputs['mask_pred'], targets['mask']
                        ).item()
                    else:
                        task_loss = 0.0
                else:
                    # Generic L1 loss
                    task_loss = F.l1_loss(outputs, targets).item()

                metrics[f"{task_name}_loss"] = task_loss
                total_weighted_loss += task_loss * weight
                total_weight += weight

            except Exception as e:
                self.logger.warning(f"Loss computation failed for task {task_name}: {e}")
                metrics[f"{task_name}_loss"] = float('inf')

        # Overall multi-task loss
        if total_weight > 0:
            metrics["multitask_weighted_loss"] = total_weighted_loss / total_weight
        else:
            metrics["multitask_weighted_loss"] = float('inf')

        # Task balance metric (how evenly losses are distributed)
        task_losses = [metrics.get(f"{task}_loss", 0.0) for task in model_outputs.keys()
                      if f"{task}_loss" in metrics and metrics[f"{task}_loss"] != float('inf')]
        if len(task_losses) > 1:
            loss_std = np.std(task_losses)
            loss_mean = np.mean(task_losses)
            if loss_mean > 0:
                metrics["task_balance"] = loss_std / loss_mean  # Coefficient of variation
            else:
                metrics["task_balance"] = 0.0
        else:
            metrics["task_balance"] = 0.0

        return metrics

    def evaluate_efficiency(self,
                           model: nn.Module,
                           input_shape: Tuple[int, ...],
                           num_runs: int = 100,
                           warmup_runs: int = 10) -> Dict[str, float]:
        """
        Evaluate computational efficiency of the model.

        Args:
            model: Model to evaluate
            input_shape: Shape of input tensor (B, C, T) or similar
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs (not timed)

        Returns:
            Dictionary of efficiency metrics
        """
        model.eval()
        device = next(model.parameters()).device

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        # Timed runs
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        import time
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_run = total_time / num_runs

        # Compute real-time factor (assuming audio duration)
        # For audio, we need to know the actual duration represented by input_shape
        # Assuming input_shape[2] is time dimension in samples
        if len(input_shape) >= 3:
            audio_duration_seconds = input_shape[2] / self.sample_rate
            real_time_factor = avg_time_per_run / audio_duration_seconds if audio_duration_seconds > 0 else float('inf')
        else:
            real_time_factor = float('inf')

        metrics = {
            "avg_inference_time_ms": avg_time_per_run * 1000,
            "fps": 1.0 / avg_time_per_run if avg_time_per_run > 0 else 0.0,
            "real_time_factor": real_time_factor,
            "throughput_samples_per_second": input_shape[2] / avg_time_per_run if avg_time_per_run > 0 and len(input_shape) >= 3 else 0.0
        }

        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(dummy_input)
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            metrics["peak_memory_mb"] = peak_memory_mb

        return metrics

    def _compute_snr(self, audio: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio."""
        # Simple SNR estimation: signal power / noise power
        # Assume first 10ms is noise for simplicity
        noise_len = min(int(0.01 * self.sample_rate), audio.shape[-1] // 10)
        if noise_len > 0:
            noise_power = np.mean(audio[..., :noise_len] ** 2)
            signal_power = np.mean(audio[..., noise_len:] ** 2)
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
        return 0.0

    def _compute_log_spectral_distance(self, ref_audio: np.ndarray, enh_audio: np.ndarray) -> float:
        """Compute log-spectral distance between two audio signals."""
        try:
            import librosa
            # Compute spectrograms
            S_ref = np.abs(librosa.stft(ref_audio, n_fft=1024, hop_length=self.hop_length))
            S_enh = np.abs(librosa.stft(enh_audio, n_fft=1024, hop_length=self.hop_length))

            # Avoid log(0)
            S_ref = np.maximum(S_ref, 1e-10)
            S_enh = np.maximum(S_enh, 1e-10)

            # Log-spectral distance
            log_S_ref = np.log(S_ref)
            log_S_enh = np.log(S_enh)
            lsd = np.mean((log_S_ref - log_S_enh) ** 2)
            return np.sqrt(lsd)
        except ImportError:
            self.logger.warning("Librosa not available for log-spectral distance")
            return 0.0
        except Exception as e:
            self.logger.warning(f"Log-spectral distance computation failed: {e}")
            return float('inf')

    def compute_overall_score(self,
                            audio_quality_metrics: Dict[str, float],
                            control_preservation_metrics: Dict[str, float],
                            multitask_metrics: Dict[str, float],
                            efficiency_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compute overall score combining all evaluation aspects.

        Args:
            audio_quality_metrics: Audio quality metrics
            control_preservation_metrics: Control preservation metrics
            multitask_metrics: Multi-task compatibility metrics
            efficiency_metrics: Efficiency metrics

        Returns:
            Dictionary containing overall scores and breakdown
        """
        scores = {}

        # Audio quality score (normalize PESQ and STOI to 0-1 range)
        pesq_score = audio_quality_metrics.get("pesq", -1.0)
        stoi_score = audio_quality_metrics.get("stoi", -1.0)

        # Normalize PESQ (-0.5 to 4.5) to (0, 1)
        if pesq_score >= 0:
            pesq_norm = min(max((pesq_score + 0.5) / 5.0, 0.0), 1.0)
        else:
            pesq_norm = 0.0

        # Normalize STOI (0 to 1) to (0, 1) - already in range
        if stoi_score >= 0:
            stoi_norm = min(max(stoi_score, 0.0), 1.0)
        else:
            stoi_norm = 0.0

        audio_quality_score = (pesq_norm * 0.5 + stoi_norm * 0.5)  # Equal weight
        scores["audio_quality_score"] = audio_quality_score

        # Control preservation score (lower is better for errors, higher is better for correlations)
        f0_rmse = control_preservation_metrics.get("f0_rmse", float('inf'))
        sp_dist = control_preservation_metrics.get("sp_distortion", float('inf'))
        ap_corr = control_preservation_metrics.get("ap_correlation", 0.0)
        f0_corr = control_preservation_metrics.get("f0_correlation", 0.0)

        # Normalize errors (invert and cap)
        f0_score = max(0.0, 1.0 - min(f0_rmse / 0.1, 1.0))  # Assume 0.1 is bad RMSE
        sp_score = max(0.0, 1.0 - min(sp_dist / 0.01, 1.0))   # Assume 0.01 is bad distortion

        # Correlations are already 0-1
        ap_score = max(0.0, min(ap_corr, 1.0))
        f0_score_corr = max(0.0, min(f0_corr, 1.0))

        control_score = (f0_score * 0.3 + sp_score * 0.3 + ap_score * 0.2 + f0_score_corr * 0.2)
        scores["control_preservation_score"] = control_score

        # Multi-task score (lower loss is better)
        multi_task_loss = multitask_metrics.get("multitask_weighted_loss", float('inf'))
        multi_task_score = max(0.0, 1.0 - min(multi_task_loss / 1.0, 1.0))  # Assume 1.0 is bad loss
        # Task balance bonus (lower is better)
        task_balance = multitask_metrics.get("task_balance", float('inf'))
        balance_bonus = max(0.0, 1.0 - min(task_balance / 2.0, 1.0))  # Assume 2.0 is poor balance
        multitask_score = multi_task_score * 0.7 + balance_bonus * 0.3
        scores["multitask_score"] = multitask_score

        # Efficiency score (lower real-time factor is better)
        rt_factor = efficiency_metrics.get("real_time_factor", float('inf'))
        # Score based on real-time factor: 1.0 for real-time or better, decreasing for slower
        if rt_factor <= 1.0:
            efficiency_score = 1.0
        elif rt_factor <= 2.0:
            efficiency_score = 0.8
        elif rt_factor <= 3.0:
            efficiency_score = 0.6
        elif rt_factor <= 5.0:
            efficiency_score = 0.4
        else:
            efficiency_score = 0.2
        scores["efficiency_score"] = efficiency_score

        # Weighted overall score
        weights = {
            "audio_quality_score": 0.3,
            "control_preservation_score": 0.3,
            "multitask_score": 0.25,
            "efficiency_score": 0.15
        }

        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        scores["overall_score"] = overall_score

        return scores


def create_diffusion_evaluator(config: Optional[Dict[str, Any]] = None) -> DiffusionEvaluator:
    """
    Factory function to create a diffusion evaluator.

    Args:
        config: Configuration dictionary (optional)

    Returns:
        Configured DiffusionEvaluator instance
    """
    return DiffusionEvaluator(config)