"""
Real-time inference engine for CantioAI with integrated optimizations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time
import logging
from pathlib import Path

from .optimization import OptimizationManager, StreamingInferenceEngine, benchmark_inference

logger = logging.getLogger(__name__)


class RealTimeInferenceEngine:
    """
    Real-time inference engine that integrates all optimization techniques
    for CantioAI voice conversion system
    """

    def __init__(self,
                 model: nn.Module,
                 config_path: str = "configs/optimization/realtime.yaml",
                 device: str = "auto"):
        """
        Initialize the real-time inference engine

        Args:
            model: The base model to optimize
            config_path: Path to optimization configuration file
            device: Device to run inference on ("auto", "cpu", "cuda", etc.)
        """
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.config_path = config_path

        # Initialize optimization manager
        self.opt_manager = OptimizationManager(config_path)

        # Apply optimizations
        self.optimized_model, self.engine_config = create_optimized_model(
            self.model, config_path
        )

        # Initialize streaming engine if enabled
        if self.engine_config.get('streaming_enabled', False):
            self.streaming_engine = StreamingInferenceEngine(
                self.optimized_model,
                {
                    'chunk_size': self.engine_config.get('chunk_size', 512),
                    'overlap': self.engine_config.get('overlap', 128),
                    'causal': self.engine_config.get('causal', True)
                }
            )
        else:
            self.streaming_engine = None

        # Performance tracking
        self.latency_history = []
        self.throughput_history = []
        self.is_warmed_up = False

        logger.info(f"RealTimeInferenceEngine initialized on {self.device}")
        logger.info(f"Optimization level: {self.engine_config['optimization_level']}")
        logger.info(f"Target latency: {self.engine_config['target_latency']} ms")

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def warm_up(self, input_shape: Tuple[int, ...], num_warmup: int = 50):
        """
        Warm up the model for consistent performance measurements

        Args:
            input_shape: Shape of input tensor (batch_size, seq_len, features)
            num_warmup: Number of warmup iterations
        """
        logger.info(f"Warming up model with {num_warmup} iterations...")
        dummy_input = torch.randn(input_shape, device=self.device)

        self.optimized_model.eval()
        with torch.no_grad():
            for i in range(num_warmup):
                _ = self.optimized_model(dummy_input)
                if i % 10 == 0:
                    logger.debug(f"Warmup iteration {i}/{num_warmup}")

        self.is_warmed_up = True
        logger.info("Warmup completed")

    def infer(self,
              input_features: torch.Tensor,
              return_latency: bool = False) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Perform inference with latency measurement

        Args:
            input_features: Input tensor for inference
            return_latency: Whether to return latency measurement

        Returns:
            Tuple of (output, latency_ms) if return_latency=True, else just output
        """
        if not self.is_warmed_up:
            logger.warning("Model not warmed up, calling warm_up() with default shape")
            # Infer shape from input
            self.warm_up(input_features.shape)

        # Move input to device
        input_features = input_features.to(self.device)

        # Time the inference
        start_time = time.perf_counter()

        with torch.no_grad():
            if self.streaming_engine is not None:
                # Streaming inference
                output = self.streaming_engine.process_batch(input_features)
            else:
                # Regular batch inference
                output = self.optimized_model(input_features)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Track performance
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 1000:  # Keep history manageable
            self.latency_history = self.latency_history[-1000:]

        # Calculate throughput (assuming real-time factor)
        audio_duration_ms = input_features.shape[1] * 5.0  # Assuming 5ms frame period
        if audio_duration_ms > 0:
            rtf = latency_ms / audio_duration_ms  # Real-time factor
            throughput = 1.0 / rtf if rtf > 0 else 0.0
            self.throughput_history.append(throughput)
            if len(self.throughput_history) > 1000:
                self.throughput_history = self.throughput_history[-1000:]

        if return_latency:
            return output, latency_ms
        else:
            return output

    def stream_inference(self, audio_stream: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process a stream of audio chunks for real-time applications

        Args:
            audio_stream: List of audio tensor chunks

        Returns:
            List of processed audio chunks
        """
        if self.streaming_engine is None:
            logger.warning("Streaming not enabled, falling back to batch processing")
            return [self.infer(chunk)[0] for chunk in audio_stream]

        outputs = []
        self.streaming_engine.reset_state()

        for chunk in audio_stream:
            output = self.streaming_engine.process_stream(chunk)
            outputs.append(output)

        return outputs

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics

        Returns:
            Dictionary containing performance metrics
        """
        if not self.latency_history:
            return {"status": "No performance data available"}

        latencies = np.array(self.latency_history)
        throughputs = np.array(self.throughput_history) if self.throughput_history else np.array([])

        stats = {
            'latency_ms': {
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
                'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99))
            },
            'target_latency_met': float(np.mean(latencies)) <= self.engine_config['target_latency'],
            'optimization_level': self.engine_config['optimization_level'],
            'device': str(self.device),
            'is_warmed_up': self.is_warmed_up
        }

        if len(throughputs) > 0:
            stats['throughput'] = {
                'mean_rtf': float(np.mean(throughputs)),
                'std_rtf': float(np.std(throughputs)),
                'min_rtf': float(np.min(throughputs)),
                'max_rtf': float(np.max(throughputs))
            }

        return stats

    def benchmark(self,
                  input_shape: Tuple[int, ...],
                  num_runs: int = 100) -> Dict[str, float]:
        """
        Run benchmark on the optimized model

        Args:
            input_shape: Shape of input tensor for benchmarking
            num_runs: Number of inference runs for benchmark

        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Running benchmark with {num_runs} runs...")
        return benchmark_inference(self.optimized_model, input_shape, num_runs)

    def export_optimized_model(self, export_path: str, format: str = "torchscript"):
        """
        Export the optimized model for deployment

        Args:
            export_path: Path to save the exported model
            format: Export format ("torchscript", "onnx")
        """
        logger.info(f"Exporting optimized model to {export_path} in {format} format")

        self.optimized_model.eval()

        if format == "torchscript":
            # Example input for tracing
            dummy_input = torch.randn(1, 100, 512, device=self.device)
            traced_model = torch.jit.trace(self.optimized_model, dummy_input)
            traced_model.save(export_path)
        elif format == "onnx":
            try:
                import torch.onnx
                dummy_input = torch.randn(1, 100, 512, device=self.device)
                torch.onnx.export(
                    self.optimized_model,
                    dummy_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size', 1: 'sequence_length'},
                        'output': {0: 'batch_size', 1: 'sequence_length'}
                    }
                )
            except ImportError:
                logger.error("ONNX export requires torch and onnx packages")
                raise
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Model exported successfully to {export_path}")


def create_realtime_engine(model: nn.Module,
                          config_path: str = "configs/optimization/realtime.yaml",
                          device: str = "auto") -> RealTimeInferenceEngine:
    """
    Factory function to create a real-time inference engine

    Args:
        model: The base model to optimize
        config_path: Path to optimization configuration file
        device: Device to run inference on

    Returns:
        Configured RealTimeInferenceEngine instance
    """
    return RealTimeInferenceEngine(model, config_path, device)


# Export main classes and functions
__all__ = [
    'RealTimeInferenceEngine',
    'create_realtime_engine'
]