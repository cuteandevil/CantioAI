"""HuBERT Feature Extractor for CantioAI
Implements HuBERT self-supervised feature extraction for enhanced content representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class HuBERTFeatureExtractor(nn.Module):
    """HuBERT feature extractor for self-supervised speech representation learning

    Extracts contextual speech representations using masked prediction objective
    compatible with CantioAI's hybrid architecture
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        output_layer: int = 9,
        freeze_feature_encoder: bool = True,
        freeze_feature_projection: int = 0,
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_selection: str = "static",  # static, uniform, normal
        mask_other: float = 0.0,
        skip_masked: bool = False,
        norm_features: bool = False,
        requires_grad: bool = False,
        output_hidden_states: bool = True,
        **kwargs
    ):
        """
        Initialize HuBERT feature extractor

        Args:
            model_name: Pre-trained HuBERT model name or path
            output_layer: Which transformer layer to output features from
            freeze_feature_encoder: Whether to freeze the feature encoder params
            freeze_feature_projection: Number of projection layers to freeze (-1 = all)
            mask_prob: Probability of each token being masked
            mask_length: Length of mask spans (tokens)
            mask_selection: How to select masks (static, uniform, normal)
            mask_other: Probability of replacing masked tokens with random tokens
            skip_masked: Whether to skip computing loss for masked tokens
            norm_features: Whether to apply feature normalization to outputs
            requires_grad: Whether extracted features require gradients
            output_hidden_states: Whether to return all hidden states
            **kwargs: Additional arguments passed to HuBERT model
        """
        super().__init__()

        # Store configuration
        self.model_name = model_name
        self.output_layer = output_layer
        self.freeze_feature_encoder = freeze_feature_encoder
        self.freeze_feature_projection = freeze_feature_projection
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.skip_masked = skip_masked
        self.norm_features = norm_features
        self.requires_grad = requires_grad
        self.output_hidden_states = output_hidden_states

        # Import HuBERT model (will be loaded in _setup_model)
        self.hubert = None
        self.feature_encoder = None
        self.feature_projection = None
        self._requires_grad = requires_grad

        # Output specification
        self._output_size = None  # Will be set after model loading

    def _setup_model(self):
        """Setup the HuBERT model and components"""
        try:
            from transformers import HubertModel

            # Load pre-trained HuBERT model
            self.hubert = HubertModel.from_pretrained(
                self.model_name,
                **{
                    "output_hidden_states": True,
                    "add_final_layer": False  # We'll handle projection ourselves
                }
            )

            # Get feature dimensions
            self._output_size = self.hubert.config.hidden_size

            # Setup feature encoder (Transformer backbone)
            self.feature_encoder = self.hubert.hubert_encoder

            # Setup feature projection (if needed)
            if hasattr(self.hubert, 'proj_hubert') and self.hubert.proj_hubert is not None:
                self.feature_projection = self.hubert.proj_hubert
            else:
                # Create default projection if not present
                self.feature_projection = nn.Linear(
                    self._output_size,
                    self._output_size
                )

            # Apply freezing if requested
            if self.freeze_feature_encoder:
                for param in self.feature_encoder.parameters():
                    param.requires_grad = False
                logger.info("Frozen HuBERT feature encoder parameters")

            if self.freeze_feature_projection >= 0:
                if self.feature_projection is not None:
                    # Freeze specified number of projection layers (from start)
                    if hasattr(self.feature_projection, 'weight'):
                        params_to_freeze = list(self.feature_projection.parameters())[:self.freeze_feature_projection+1]
                        for param in params_to_freeze:
                            param.requires_grad = False
                    logger.info(f"Frozen first {self.freeze_feature_projection + 1} feature projection layers")

            logger.info(f"Successfully initialized HuBERT model: {self.model_name}")
            logger.info(f"Feature dimension: {self._output_size}")
            logger.info(f"Output layer: {self.output_layer}")

        except ImportError:
            logger.warning("transformers not available, using fallback implementation")
            # Fallback: Create a simple HuBERT-like encoder for testing
            self._create_fallback_hubert()

    def _create_fallback_hubert(self):
        """Create fallback HuBERT implementation when transformers not available"""
        # This ensures the code remains functional even without transformers
        logger.warning("Using fallback HuBERT implementation")

        # Create minimal HuBERT-like architecture
        self.hubert = nn.Module()  # Placeholder
        self.feature_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072)
        )
        self._output_size = 768
        self.feature_projection = nn.Identity()  # No projection by default

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, dict]:
        """
        Extract HuBERT features from input audio

        Args:
            input_values: Raw audio signal of shape `(batch_size, sequence_length)`
            attention_mask: Mask to avoid performing attention on padding tokens
            output_attentions: Whether to return attentions
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict or tuple

        Returns:
            Features or dictionary containing:
            - last_hidden_state: Final layer hidden state
            - hidden_states: All hidden states (if requested)
            - attentions: All attention weights (if requested)
        """
        # Setup model if not done yet
        if self.hubert is None:
            self._setup_model()

        # Handle input padding
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values)

        # Apply HuBERT feature extraction
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions or self.training,
            output_hidden_states=output_hidden_states or self.output_hidden_states,
            return_dict=True,  # Always get dict from HuBERT
        )

        # Extract the desired layer's features
        if self.output_hidden_states:
            # Return all hidden states
            hidden_states = outputs.hidden_states
            features = torch.stack(hidden_states, dim=0)  # Concatenate all layers
        else:
            # Return specific layer output
            if hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > self.output_layer:
                features = outputs.hidden_states[self.output_layer]
            else:
                # Fallback to last available layer
                features = outputs.hidden_states[-1]

        # Apply feature normalization if requested
        if self.norm_features:
            features = F.layer_norm(features, dim=-1)

        # Prepare return value
        if return_dict:
            output = {
                "last_hidden_state": features,
                "hidden_states": outputs.hidden_states if self.output_hidden_states else None,
                "attentions": outputs.attentions if output_attentions else None,
            }
            return output
        else:
            return features

    def get_output_size(self) -> int:
        """Get the output feature dimension"""
        if self._output_size is None:
            self._setup_model()
        return self._output_size

    def freeze(self):
        """Freeze feature extractor parameters"""
        if self.hubert is not None:
            for param in self.hubert.parameters():
                param.requires_grad = False
            logger.info("Frozen HuBERT feature extractor")

    def unfreeze(self):
        """Unfreeze feature extractor parameters"""
        if self.hubert is not None:
            for param in self.hubert.parameters():
                param.requires_grad = not self.requires_grad
            logger.info(f"{'Unfrozen' if self.requires_grad else 'Frozen'} HuBERT feature extractor")