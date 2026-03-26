"""HuBERT Model Manager for CantioAI
Manages different HuBERT variants and provides unified interface for feature extraction
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import os
import logging

# Import HubertFeatureExtractor directly to avoid package import issues
from .hubert import HubertFeatureExtractor

logger = logging.getLogger(__name__)


class HuBERTManager:
    """Manages HuBERT models for CantioAI with support for different variants and configurations"""

    # Supported HuBERT models and their configurations
    SUPPORTED_MODELS = {
        "hubert-base": {
            "model_name": "facebook/hubert-base-ls960",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
        },
        "hubert-large": {
            "model_name": "facebook/hubert-large-ls960-ft",
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
        },
        "hubert-large-ll60k": {
            "model_name": "facebook/hubert-large-ls960-ft-lt60k",
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
        },
        "xlsr-53": {
            "model_name": "facebook/wav2vec2-large-xlsr-53",
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
        },
        "wavlm-base": {
            "model_name": "microsoft/wavlm-base",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
        },
        "wavlm-large": {
            "model_name": "microsoft/wavlm-large",
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HuBERT manager with configuration

        Args:
            config: Configuration dictionary for HuBERT settings
        """
        self.config = config or self._get_default_config()
        self._models: Dict[str, nn.Module] = {}
        self._extractors: Dict[str, HuBERTFeatureExtractor] = {}
        self._current_model: Optional[str] = None

        logger.info("HuBERT Manager initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default HuBERT configuration"""
        return {
            "enabled": False,
            "mode": "hybrid",  # hubert_only, phoneme_only, hybrid
            "model": {
                "type": "hubert-base",
                "checkpoint": None,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            "extraction": {
                "mode": "offline",  # offline, online, streaming
                "sample_rate": 16000,
                "layer_index": 12,
                "use_all_layers": False,
                "normalize": True,
                "cache_dir": "./data/hubert_features",
            },
            "processing": {
                "target_dim": 256,
                "alignment_method": "dtw",  # linear, dtw, attention
                "normalize_features": True,
                "apply_pca": False,
                "pca_components": 128,
            },
            "fusion": {
                "enabled": True,
                "strategy": "gated",  # concat, gated, cross_attention, hierarchical
                "gated_temperature": 0.1,
                "cross_attention_heads": 8,
                "fusion_layers": [3, 6, 9],
            },
            "finetune": {
                "enabled": False,
                "strategy": "task_specific",
                "freeze_layers": [0, 1, 2, 3, 4, 5],
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
            },
            "streaming": {
                "enabled": False,
                "chunk_size": 16000,
                "overlap": 0.1,
                "causal": true,
                "use_cache": true,
                "cache_size": 10,
            }
        }

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific HuBERT model type"""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported HuBERT model type: {model_type}. "
                           f"Supported types: {list(self.SUPPORTED_MODELS.keys())}")

        return self.SUPPORTED_MODELS[model_type].copy()

    def load_model(self, model_type: str = "hubert-base", **kwargs) -> HuBERTFeatureExtractor:
        """Load a HuBERT model and return feature extractor

        Args:
            model_type: Type of HuBERT model to load
            **kwargs: Additional arguments passed to HuBERTFeatureExtractor

        Returns:
            HuBERTFeatureExtractor instance
        """
        if model_type in self._extractors:
            logger.info(f"Returning cached HuBERT model: {model_type}")
            return self._extractors[model_type]

        # Get model configuration
        model_config = self.get_model_config(model_type)

        # Override with any provided kwargs
        model_config.update(kwargs)

        # Create feature extractor
        extractor = HuBERTFeatureExtractor(
            model_name=model_config["model_name"],
            **model_config
        )

        # Cache the extractor
        self._extractors[model_type] = extractor
        self._current_model = model_type

        logger.info(f"Loaded HuBERT model: {model_type} ({model_config['model_name']})")
        return extractor

    def get_extractor(self, model_type: Optional[str] = None) -> HuBERTFeatureExtractor:
        """Get HuBERT feature extractor

        Args:
            model_type: Specific model type to get (uses current if None)

        Returns:
            HuBERTFeatureExtractor instance
        """
        model_type = model_type or self._current_model
        if model_type is None:
            model_type = "hubert-base"  # Default

        if model_type not in self._extractors:
            return self.load_model(model_type)

        return self._extractors[model_type]

    def is_available(self) -> bool:
        """Check if HuBERT models are available (transformers installed)"""
        try:
            import transformers
            return True
        except ImportError:
            logger.warning("transformers library not available")
            return False

    def list_available_models(self) -> List[str]:
        """List all available HuBERT models"""
        return list(self.SUPPORTED_MODELS.keys())

    def get_feature_dimension(self, model_type: Optional[str] = None) -> int:
        """Get feature dimension for specified model type"""
        extractor = self.get_extractor(model_type)
        return extractor.get_output_size()

    def clear_cache(self):
        """Clear cached models to free memory"""
        self._extractors.clear()
        self._models.clear()
        self._current_model = None
        logger.info("Cleared HuBERT model cache")


class HubertFeatureProjection(nn.Module):
    """Projects HuBERT features to target dimension for fusion with CantioAI features"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_type: str = "linear",
        dropout: float = 0.1,
        **kwargs
    ):
        """Initialize HuBERT feature projection

        Args:
            input_dim: Input feature dimension (HuBERT output)
            output_dim: Target output dimension
            projection_type: Type of projection (linear, mlp, attention)
            dropout: Dropout probability
            **kwargs: Additional arguments
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_type = projection_type

        if projection_type == "linear":
            self.projection = nn.Linear(input_dim, output_dim)
        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.Dropout(dropout)
            )
        elif projection_type == "attention":
            self.projection = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=min(8, input_dim // 64),
                dropout=dropout,
                batch_first=True
            )
            self.output_proj = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported projection type: {projection_type}")

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project HuBERT features to target dimension

        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)

        Returns:
            Projected features of shape (batch_size, seq_len, output_dim)
        """
        if self.projection_type == "linear":
            x = self.projection(x)
        elif self.projection_type == "mlp":
            x = self.projection(x)
        elif self.projection_type == "attention":
            # Self-attention then projection
            attn_output, _ = self.projection(x, x, x)
            x = self.output_proj(attn_output)

        x = self.dropout(x)
        x = self.norm(x)

        return x
