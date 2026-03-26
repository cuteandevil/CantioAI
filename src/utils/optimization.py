"""
Optimization utilities for CantioAI real-time inference optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import yaml
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizationManager:
    """Manages various optimization techniques for real-time inference"""

    def __init__(self, config_path: str = "configs/optimization/realtime.yaml"):
        self.config = self._load_config(config_path)
        self.optimization_level = self.config['realtime']['optimization_level']
        self.target_latency = self.config['realtime']['target_latency']
        self.enabled = self.config['realtime']['enabled']

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load optimization configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Optimization config not found at {config_path}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default optimization configuration"""
        return {
            'realtime': {
                'enabled': True,
                'target_latency': 30.0,
                'target_device': 'auto',
                'optimization_level': 'balanced',
                'model_optimization': {
                    'quantization': {'enabled': True, 'dtype': 'int8', 'per_channel': True, 'calibration_samples': 100},
                    'pruning': {'enabled': False, 'method': 'structured', 'sparsity': 0.3, 'iterative': True},
                    'distillation': {'enabled': False, 'teacher_checkpoint': None, 'temperature': 3.0},
                    'operator_fusion': {'enabled': True, 'fuse_conv_bn': True, 'fuse_attention': True}
                },
                'runtime_optimization': {
                    'jit_compilation': True, 'graph_optimization': True, 'memory_pool': True,
                    'dynamic_batching': True, 'cache_system': True
                },
                'algorithmic_optimization': {
                    'approximation': {'low_rank': False, 'sparse_attention': True, 'flash_attention': True},
                    'adaptive_computation': {'early_exit': False, 'dynamic_depth': False, 'conditional_computation': True},
                    'streaming': {'enabled': True, 'chunk_size': 512, 'overlap': 128, 'causal': True}
                },
                'hardware_optimization': {
                    'cpu': {'simd': True, 'threads': 0, 'memory_alignment': 64},
                    'gpu': {'tensorrt': True, 'fp16': True, 'stream_concurrency': 2},
                    'edge': {'tflite': False, 'coreml': False, 'onnx_runtime': True}
                },
                'monitoring': {
                    'latency_tracking': True, 'memory_tracking': True, 'throughput_tracking': True, 'auto_tuning': False
                }
            }
        }

    def apply_model_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply model-level optimizations"""
        if not self.enabled:
            return model

        optimized_model = model

        # Quantization
        if self.config['realtime']['model_optimization']['quantization']['enabled']:
            optimized_model = self._apply_quantization(optimized_model)

        # Pruning
        if self.config['realtime']['model_optimization']['pruning']['enabled']:
            optimized_model = self._apply_pruning(optimized_model)

        # Operator fusion
        if self.config['realtime']['model_optimization']['operator_fusion']['enabled']:
            optimized_model = self._apply_operator_fusion(optimized_model)

        # Knowledge distillation (if teacher provided)
        if self.config['realtime']['model_optimization']['distillation']['enabled']:
            optimized_model = self._apply_distillation(optimized_model)

        return optimized_model

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model"""
        dtype_str = self.config['realtime']['model_optimization']['quantization']['dtype']
        per_channel = self.config['realtime']['model_optimization']['quantization']['per_channel']

        logger.info(f"Applying {dtype_str} quantization (per_channel={per_channel})")

        # For now, we'll prepare the model for quantization
        # Actual quantization would happen during export/conversion
        if dtype_str == 'int8':
            # Prepare for INT8 quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
        elif dtype_str in ['float16', 'bfloat16']:
            # Convert to half precision
            if dtype_str == 'float16':
                model.half()
            elif dtype_str == 'bfloat16':
                # Note: bfloat16 support depends on PyTorch version and hardware
                try:
                    model = model.to(torch.bfloat16)
                except Exception as e:
                    logger.warning(f"bfloat16 not supported, falling back to float32: {e}")

        return model

    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to model"""
        method = self.config['realtime']['model_optimization']['pruning']['method']
        sparsity = self.config['realtime']['model_optimization']['pruning']['sparsity']
        iterative = self.config['realtime']['model_optimization']['pruning']['iterative']

        logger.info(f"Applying {method} pruning with sparsity {sparsity} (iterative={iterative})")

        # Implement magnitude-based pruning
        if method == 'structured':
            self._structured_pruning(model, sparsity)
        else:
            self._unstructured_pruning(model, sparsity)

        return model

    def _structured_pruning(self, model: nn.Module, sparsity: float):
        """Apply structured pruning (remove entire channels/filters)"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                # Calculate L2 norm of filters/columns
                weight = module.weight.data
                if len(weight.shape) >= 2:
                    # For conv layers, compute norm over input/output dimensions
                    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                        # Structured pruning along output channels
                        norms = torch.norm(weight, p=2, dim=list(range(1, len(weight.shape))))
                    else:  # Linear
                        # Structured pruning along output features
                        norms = torch.norm(weight, p=2, dim=1)

                    # Calculate threshold for pruning
                    threshold = torch.quantile(norms, sparsity)
                    mask = norms > threshold

                    # Apply mask
                    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                        # Prune output channels
                        module.out_channels = mask.sum().item()
                        module.weight.data = weight[mask]
                        if module.bias is not None:
                            module.bias.data = module.bias.data[mask]
                    else:  # Linear
                        # Prune output features
                        module.out_features = mask.sum().item()
                        module.weight.data = weight[mask]
                        if module.bias is not None:
                            module.bias.data = module.bias.data[mask]

    def _unstructured_pruning(self, model: nn.Module, sparsity: float):
        """Apply unstructured pruning (remove individual weights)"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                # Flatten and find threshold
                flat_weight = weight.view(-1)
                threshold = torch.quantile(torch.abs(flat_weight), sparsity)
                # Create mask
                mask = torch.abs(weight) > threshold
                # Apply mask
                module.weight.data = weight * mask.float()

    def _apply_operator_fusion(self, model: nn.Module) -> nn.Module:
        """Apply operator fusion optimizations"""
        fuse_conv_bn = self.config['realtime']['model_optimization']['operator_fusion']['fuse_conv_bn']
        fuse_attention = self.config['realtime']['model_optimization']['operator_fusion']['fuse_attention']

        logger.info(f"Applying operator fusion (conv_bn={fuse_conv_bn}, attention={fuse_attention})")

        # In practice, this would be done during model export/conversion
        # For now, we'll mark the model as ready for fusion
        model._ready_for_fusion = True
        return model

    def _apply_distillation(self, model: nn.Module) -> nn.Module:
        """Apply knowledge distillation (placeholder)"""
        logger.info("Knowledge distillation would be applied during training")
        return model

    def apply_runtime_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply runtime optimizations"""
        if not self.enabled:
            return model

        optimized_model = model

        # JIT compilation
        if self.config['realtime']['runtime_optimization']['jit_compilation']:
            optimized_model = self._apply_jit_compilation(optimized_model)

        # Graph optimization
        if self.config['realtime']['runtime_optimization']['graph_optimization']:
            optimized_model = self._apply_graph_optimization(optimized_model)

        return optimized_model

    def _apply_jit_compilation(self, model: nn.Module) -> nn.Module:
        """Apply JIT compilation (TorchScript)"""
        logger.info("Applying TorchScript JIT compilation")
        try:
            model.eval()
            # Trace or script the model
            traced_model = torch.jit.trace(model, torch.randn(1, 100, 512))  # Dummy input
            return traced_model
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}, using original model")
            return model

    def _apply_graph_optimization(self, model: nn.Module) -> nn.Module:
        """Apply graph optimization"""
        logger.info("Applying graph optimization")
        # In practice, this would involve constant folding, dead code elimination, etc.
        # For now, we'll just ensure the model is in evaluation mode
        model.eval()
        return model

    def get_inference_engine_config(self) -> Dict[str, Any]:
        """Get configuration for the inference engine"""
        return {
            'target_latency': self.target_latency,
            'optimization_level': self.optimization_level,
            'streaming_enabled': self.config['realtime']['algorithmic_optimization']['streaming']['enabled'],
            'chunk_size': self.config['realtime']['algorithmic_optimization']['streaming']['chunk_size'],
            'overlap': self.config['realtime']['algorithmic_optimization']['streaming']['overlap'],
            'causal': self.config['realtime']['algorithmic_optimization']['streaming']['causal'],
            'dynamic_batching': self.config['realtime']['runtime_optimization']['dynamic_batching'],
            'cache_system': self.config['realtime']['runtime_optimization']['cache_system'],
            'memory_pool': self.config['realtime']['runtime_optimization']['memory_pool'],
            'hardware': {
                'cpu': self.config['realtime']['hardware_optimization']['cpu'],
                'gpu': self.config['realtime']['hardware_optimization']['gpu'],
                'edge': self.config['realtime']['hardware_optimization']['edge']
            },
            'monitoring': self.config['realtime']['monitoring']
        }


class StreamingInferenceEngine:
    """Handles streaming inference for real-time processing"""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.overlap = config.get('overlap', 128)
        self.causal = config.get('causal', True)
        self.reset_state()

    def reset_state(self):
        """Reset internal state for streaming"""
        self.buffer = []
        self.hidden_states = {}

    def process_stream(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """Process a streaming audio chunk"""
        # Add chunk to buffer
        self.buffer.append(audio_chunk)

        # Process when we have enough data
        total_length = sum(chunk.shape[-1] for chunk in self.buffer)
        if total_length >= self.chunk_size:
            # Concatenate buffer along the sequence dimension
            combined = torch.cat(self.buffer, dim=-1)

            # Process with model
            with torch.no_grad():
                output = self.model(combined)

            # Keep overlap for next chunk
            overlap_samples = min(self.overlap, output.shape[-1])
            if overlap_samples > 0:
                self.buffer = [output[..., -overlap_samples:]]
            else:
                self.buffer = []

            # Return non-overlapping part
            if overlap_samples > 0:
                return output[..., :-overlap_samples]
            else:
                return output
        else:
            # Not enough data yet, return silence or zeros
            return torch.zeros_like(audio_chunk)

    def process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of audio (non-streaming)"""
        with torch.no_grad():
            return self.model(batch)


def create_optimized_model(model: nn.Module, config_path: str = "configs/optimization/realtime.yaml") -> Tuple[nn.Module, Dict[str, Any]]:
    """Create an optimized version of the model"""
    optimizer = OptimizationManager(config_path)

    # Apply model optimizations
    optimized_model = optimizer.apply_model_optimizations(model)

    # Apply runtime optimizations
    optimized_model = optimizer.apply_runtime_optimizations(optimized_model)

    # Get inference engine configuration
    engine_config = optimizer.get_inference_engine_config()

    return optimized_model, engine_config


def benchmark_inference(model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
    """Benchmark inference latency"""
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure latency
    import time
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    latencies = np.array(latencies)

    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99))
    }


# Export main classes and functions
__all__ = [
    'OptimizationManager',
    'StreamingInferenceEngine',
    'create_optimized_model',
    'benchmark_inference'
]