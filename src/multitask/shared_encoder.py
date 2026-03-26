"""
Multi-task Learning Shared Encoder Architecture
Updated to support Transformer-based Hybrid Spectral Predictor
Implements shared encoder for multi-task learning with task-specific heads
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MultiTaskSharedEncoder(nn.Module):
    """
    Shared encoder for multi-task learning
    Extracts shared features across different tasks
    """
    def __init__(
        self,
        base_encoder: nn.Module,
        shared_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize multi-task shared encoder
        Args:
            base_encoder: Base encoder model (e.g., Transformer, CNN)
            shared_dim: Dimension of shared features
            num_layers: Number of shared transformation layers
            dropout: Dropout rate
        """
        super().__init__()
        self.base_encoder = base_encoder
        self.shared_dim = shared_dim
        # Shared projection layers
        # First layer: encoder_dim -> shared_dim
        # Subsequent layers: shared_dim -> shared_dim
        encoder_dim = self._get_encoder_output_dim(base_encoder)
        self.shared_proj = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: map from encoder output dimension to shared dimension
                in_dim = encoder_dim
            else:
                # Subsequent layers: maintain shared dimension
                in_dim = shared_dim
            self.shared_proj.append(
                nn.Sequential(
                    nn.Linear(in_dim, shared_dim),
                    nn.LayerNorm(shared_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        # Task-specific heads (to be implemented by subclasses)
        self.task_heads = nn.ModuleDict()

    def _get_encoder_output_dim(self, encoder: nn.Module) -> int:
        """Get output dimension of encoder"""
        # Try to get from common attributes
        if hasattr(encoder, 'config') and hasattr(encoder.config, 'hidden_size'):
            return encoder.config.hidden_size
        elif hasattr(encoder, 'output_dim'):
            return encoder.output_dim
        elif hasattr(encoder, 'num_features'):
            return encoder.num_features
        # Handle HybridSpectralPredictor specifically (original CNN+BiLSTM version)
        elif hasattr(encoder, 'fc2'):
            # For HybridSpectralPredictor, output dimension is the fc2 output features
            return encoder.fc2.out_features
        # Handle Transformer-based HybridSpectralPredictor
        elif hasattr(encoder, 'transformer_encoder'):
            # For Transformer-based predictor, output dimension is the transformer hidden dim
            # We need to get it from the config or the transformer encoder directly
            if hasattr(encoder, 'transformer_encoder'):
                if hasattr(encoder.transformer_encoder, 'd_model'):
                    return encoder.transformer_encoder.d_model
                # Try to get from config
                elif hasattr(encoder, 'config'):
                    transformer_config = getattr(encoder, 'config', {})
                    if isinstance(transformer_config, dict):
                        model_config = transformer_config.get('model', {})
                        if isinstance(model_config, dict):
                            transformer_specific = model_config.get('transformer', {})
                            if isinstance(transformer_specific, dict):
                                return transformer_specific.get('hidden_dim', 512)
            # Fallback to checking the transformer encoder's norm layer
            elif hasattr(encoder.transformer_encoder, 'norm'):
                if hasattr(encoder.transformer_encoder.norm, 'normalized_shape'):
                    norm_shape = encoder.transformer_encoder.norm.normalized_shape
                    if isinstance(norm_shape, tuple) and len(norm_shape) > 0:
                        return norm_shape[-1]
        # Handle our new Transformer encoder directly
        elif hasattr(encoder, 'd_model'):
            return encoder.d_model
        elif hasattr(encoder, 'hidden_dim'):
            return encoder.hidden_dim
        else:
            # Default fallback
            return 768  # Common default for many encoders

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared encoder
        Args:
            **kwargs: Input to base encoder
        Returns:
            Dictionary containing:
                - shared_features: Shared representation tensor
                - encoder_outputs: Raw encoder outputs (if needed)
                - task_logits: Dictionary of task-specific predictions
        """
        # Get base encoder outputs
        encoder_outputs = self.base_encoder(**kwargs)
        # Extract shared features
        shared_features = encoder_outputs
        for proj_layer in self.shared_proj:
            shared_features = proj_layer(shared_features)
        # Compute task-specific predictions
        task_logits = {}
        for task_name, task_head in self.task_heads.items():
            task_logits[task_name] = task_head(shared_features)
        return {
            "shared_features": shared_features,
            "encoder_outputs": encoder_outputs,
            "task_logits": task_logits
        }

    def add_task_head(self, task_name: str, task_head: nn.Module):
        """
        Add a task-specific head
        Args:
            task_name: Name of the task
            task_head: Task-specific prediction head
        """
        self.task_heads[task_name] = task_head

    def remove_task_head(self, task_name: str):
        """
        Remove a task-specific head
        Args:
            task_name: Name of the task to remove
        """
        if task_name in self.task_heads:
            del self.task_heads[task_name]

    def get_task_head(self, task_name: str) -> Optional[nn.Module]:
        """
        Get a task-specific head
        Args:
            task_name: Name of the task
        Returns:
            Task-specific head or None if not found
        """
        return self.task_heads.get(task_name)

    def freeze_base_encoder(self):
        """Freeze base encoder parameters"""
        for param in self.base_encoder.parameters():
            param.requires_grad = False

    def unfreeze_base_encoder(self):
        """Unfreeze base encoder parameters"""
        for param in self.base_encoder.parameters():
            param.requires_grad = True

    def freeze_shared_proj(self):
        """Freeze shared projection parameters"""
        for param in self.shared_proj.parameters():
            param.requires_grad = False

    def unfreeze_shared_proj(self):
        """Unfreeze shared projection parameters"""
        for param in self.shared_proj.parameters():
            param.requires_grad = True

    def freeze_task_heads(self):
        """Freeze all task-specific head parameters"""
        for task_head in self.task_heads.values():
            for param in task_head.parameters():
                param.requires_grad = False

    def unfreeze_task_heads(self):
        """Unfreeze all task-specific head parameters"""
        for task_head in self.task_heads.values():
            for param in task_head.parameters():
                param.requires_grad = True