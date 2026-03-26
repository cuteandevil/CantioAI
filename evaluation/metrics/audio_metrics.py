"""
音频质量评估指标
"""

import torch
import numpy as np
import torchaudio
from typing import Dict, List, Optional, Tuple
import logging
import warnings

# 尝试导入可选依赖
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("PESQ not installed. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    warnings.warn("STOI not installed. Install with: pip install pystoi")

try:
    import mir_eval
    MIREVAL_AVAILABLE = True
except ImportError:
    MIREVAL_AVAILABLE = False
    warnings.warn("mir_eval not installed. Install with: pip install mir_eval")

logger = logging.getLogger(__name__)


class AudioQualityMetrics:
    """
    音频质量评估指标集合
    """

    def __init__(self, sample_rate: int = 24000, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device

        # 梅尔频谱提取器
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80
        ).to(device)

        # STFT提取器
        self.stft_func = lambda x: torch.stft(
            x, n_fft=1024, hop_length=256, win_length=1024,
            window=torch.hann_window(1024).to(x.device),
            return_complex=True
        )

        logger.info(f"AudioQualityMetrics 初始化完成，采样率: {sample_rate}, 设备: {device}")

    def compute_all_metrics(self, real_audio: torch.Tensor, gen_audio: torch.Tensor,
                           compute_pesq_stoi: bool = True) -> Dict[str, float]:
        """
        计算所有音频质量指标

        参数:
            real_audio: 真实音频 [B, T] 或 [T]
            gen_audio: 生成音频 [B, T] 或 [T]
            compute_pesq_stoi: 是否计算PESQ和STOI（较慢）

        返回:
            指标字典
        """
        # 确保是批次格式
        if real_audio.dim() == 1:
            real_audio = real_audio.unsqueeze(0)
        if gen_audio.dim() == 1:
            gen_audio = gen_audio.unsqueeze(0)

        # 确保音频长度相同
        min_len = min(real_audio.shape[-1], gen_audio.shape[-1])
        real_audio = real_audio[..., :min_len]
        gen_audio = gen_audio[..., :min_len]

        metrics = {}

        # 计算梅尔频谱距离
        metrics.update(self.compute_spectral_distance(real_audio, gen_audio))

        # 计算波形距离
        metrics.update(self.compute_waveform_distance(real_audio, gen_audio))

        # 计算频谱特征距离
        metrics.update(self.compute_spectral_feature_distance(real_audio, gen_audio))

        # 计算PESQ和STOI（较慢）
        if compute_pesq_stoi:
            try:
                metrics.update(self.compute_pesq_stoi(real_audio, gen_audio))
            except Exception as e:
                logger.warning(f"计算PESQ/STOI失败: {e}")

        return metrics

    def compute_spectral_distance(self, real_audio: torch.Tensor, gen_audio: torch.Tensor) -> Dict[str, float]:
        """计算频谱距离指标"""
        # 梅尔频谱
        mel_real = self.mel_extractor(real_audio)
        mel_gen = self.mel_extractor(gen_audio)

        # 对数梅尔频谱
        log_mel_real = torch.log(torch.clamp(mel_real, min=1e-5))
        log_mel_gen = torch.log(torch.clamp(mel_gen, min=1e-5))

        # 梅尔倒谱系数
        mfcc_real = self.compute_mfcc(mel_real)
        mfcc_gen = self.compute_mfcc(mel_gen)

        # 计算各种距离
        mel_l1 = torch.mean(torch.abs(mel_real - mel_gen)).item()
        mel_l2 = torch.nn.functional.mse_loss(mel_real, mel_gen).item()
        log_mel_l1 = torch.mean(torch.abs(log_mel_real - log_mel_gen)).item()
        mfcc_l1 = torch.mean(torch.abs(mfcc_real - mfcc_gen)).item()

        # 频谱收敛性
        spectral_convergence = self.compute_spectral_convergence(real_audio, gen_audio)

        return {
            "mel_l1": mel_l1,
            "mel_l2": mel_l2,
            "log_mel_l1": log_mel_l1,
            "mfcc_l1": mfcc_l1,
            "spectral_convergence": spectral_convergence
        }

    def compute_waveform_distance(self, real_audio: torch.Tensor, gen_audio: torch.Tensor) -> Dict[str, float]:
        """计算波形距离指标"""
        # L1 距离（平均绝对误差）
        waveform_l1 = torch.mean(torch.abs(real_audio - gen_audio)).item()

        # L2 距离（均方误差）
        waveform_l2 = torch.mean((real_audio - gen_audio) ** 2).item()

        # 信噪比 (SNR)
        signal_power = torch.mean(real_audio ** 2)
        noise_power = torch.mean((real_audio - gen_audio) ** 2)
        if noise_power > 0:
            snr = 10 * torch.log10(signal_power / noise_power).item()
        else:
            snr = float('inf')

        return {
            "waveform_l1": waveform_l1,
            "waveform_l2": waveform_l2,
            "snr": snr
        }

    def compute_spectral_feature_distance(self, real_audio: torch.Tensor, gen_audio: torch.Tensor) -> Dict[str, float]:
        """计算频谱特征距离"""
        # 频谱质心
        spectral_centroid_real = self.compute_spectral_centroid(real_audio)
        spectral_centroid_gen = self.compute_spectral_centroid(gen_audio)
        spectral_centroid_l1 = torch.mean(torch.abs(spectral_centroid_real - spectral_centroid_gen)).item()

        # 频谱带宽
        spectral_bandwidth_real = self.compute_spectral_bandwidth(real_audio)
        spectral_bandwidth_gen = self.compute_spectral_bandwidth(gen_audio)
        spectral_bandwidth_l1 = torch.mean(torch.abs(spectral_bandwidth_real - spectral_bandwidth_gen)).item()

        # 频谱滚降
        spectral_rolloff_real = self.compute_spectral_rolloff(real_audio)
        spectral_rolloff_gen = self.compute_spectral_rolloff(gen_audio)
        spectral_rolloff_l1 = torch.mean(torch.abs(spectral_rolloff_real - spectral_rolloff_gen)).item()

        # 零交叉率
        zcr_real = self.compute_zero_crossing_rate(real_audio)
        zcr_gen = self.compute_zero_crossing_rate(gen_audio)
        zcr_l1 = torch.mean(torch.abs(zcr_real - zcr_gen)).item()

        return {
            "spectral_centroid_l1": spectral_centroid_l1,
            "spectral_bandwidth_l1": spectral_bandwidth_l1,
            "spectral_rolloff_l1": spectral_rolloff_l1,
            "zcr_l1": zcr_l1
        }

    def compute_pesq_stoi(self, real_audio: torch.Tensor, gen_audio: torch.Tensor) -> Dict[str, float]:
        """计算PESQ和STOI指标"""
        metrics = {}

        # 转换为numpy数组并确保是1D
        real_np = real_audio.squeeze().cpu().numpy()
        gen_np = gen_audio.squeeze().cpu().numpy()

        # 确保长度相同
        min_len = min(len(real_np), len(gen_np))
        real_np = real_np[:min_len]
        gen_np = gen_np[:min_len]

        # 计算PESQ
        if PESQ_AVAILABLE and len(real_np) > 0:
            try:
                # PESQ需要特定的采样率
                pesq_score = pesq(self.sample_rate, real_np, gen_np, 'wb')
                metrics["pesq"] = pesq_score
            except Exception as e:
                logger.warning(f"PESQ计算失败: {e}")

        # 计算STOI
        if STOI_AVAILABLE and len(real_np) > 0:
            try:
                stoi_score = stoi(real_np, gen_np, self.sample_rate, extended=False)
                metrics["stoi"] = stoi_score
            except Exception as e:
                logger.warning(f"STOI计算失败: {e}")

        return metrics

    # 辅助方法
    def compute_mfcc(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """计算MFCC"""
        # 对数梅尔频谱
        log_mel = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        # 离散余弦变换 (DCT) 近似
        # 这里使用简化版本，实际应用中可以使用torch.fft或scipy
        return torch.nn.functional.linear(
            log_mel,
            torch.randn(20, log_mel.size(-1), device=log_mel.device)  # 简化的DCT矩阵
        )

    def compute_spectral_convergence(self, real_audio: torch.Tensor, gen_audio: torch.Tensor) -> float:
        """计算频谱收敛性"""
        # 计算幅度谱
        real_stft = self.stft_func(real_audio)
        gen_stft = self.stft_func(gen_audio)

        real_mag = torch.abs(real_stft)
        gen_mag = torch.abs(gen_stft)

        # 频谱收敛性 = || |X| - |Y| ||_F / || |X| ||_F
        numerator = torch.norm(real_mag - gen_mag, p='fro')
        denominator = torch.norm(real_mag, p='fro')

        if denominator > 0:
            return (numerator / denominator).item()
        else:
            return 0.0

    def compute_spectral_centroid(self, audio: torch.Tensor) -> torch.Tensor:
        """计算频谱质心"""
        # 计算幅度谱
        stft = self.stft_func(audio)
        mag = torch.abs(stft)

        # 频率 bins
        freqs = torch.linspace(0, self.sample_rate // 2, mag.shape[-2], device=audio.device)
        freqs = freqs.unsqueeze(0).unsqueeze(-1)  # [1, freq_bins, 1]

        # 计算质心
        centroid = torch.sum(mag * freqs, dim=-2) / (torch.sum(mag, dim=-2) + 1e-8)
        return centroid

    def compute_spectral_bandwidth(self, audio: torch.Tensor) -> torch.Tensor:
        """计算频谱带宽"""
        centroid = self.compute_spectral_centroid(audio)
        stft = self.stft_func(audio)
        mag = torch.abs(stft)

        freqs = torch.linspace(0, self.sample_rate // 2, mag.shape[-2], device=audio.device)
        freqs = freqs.unsqueeze(0).unsqueeze(-1)

        # 计算带宽
        bandwidth = torch.sqrt(torch.sum(mag * (freqs - centroid.unsqueeze(-2)) ** 2, dim=-2) /
                              (torch.sum(mag, dim=-2) + 1e-8))
        return bandwidth

    def compute_spectral_rolloff(self, audio: torch.Tensor, rolloff: float = 0.85) -> torch.Tensor:
        """计算频谱滚降"""
        stft = self.stft_func(audio)
        mag = torch.abs(stft)

        # 计算累积能量
        cumsum_mag = torch.cumsum(mag, dim=-2)
        total_energy = torch.sum(mag, dim=-2, keepdim=True)

        # 找到滚降点
        threshold = rolloff * total_energy
        rolled_off = cumsum_mag >= threshold

        # 找到第一个满足条件的频率 bin
        max_freq_bins = mag.shape[-2]
        freq_indices = torch.arange(max_freq_bins, device=audio.device).float()
        freq_indices = freq_indices.unsqueeze(0).unsqueeze(-1)

        # 应用掩码并取最小值
        masked_indices = freq_indices * rolled_off.float()
        rolloff_bin = torch.min(masked_indices.masked_fill(masked_indices == 0, float('inf')), dim=-2)[0]

        # 转换为Hz
        freqs = torch.linspace(0, self.sample_rate // 2, mag.shape[-2], device=audio.device)
        rolloff_hz = freqs[rolloff_bin.long()]

        return rolloff_hz

    def compute_zero_crossing_rate(self, audio: torch.Tensor) -> torch.Tensor:
        """计算零交叉率"""
        # 计算符号变化
        signs = torch.sign(audio)
        zero_crossings = torch.abs(torch.diff(signs, dim=-1))
        # 计算零交叉率
        zcr = torch.sum(zero_crossings, dim=-1) / (2 * (audio.shape[-1] - 1))
        return zcr