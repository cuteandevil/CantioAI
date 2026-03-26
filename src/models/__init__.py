"""
CantioAI Models Package
Updated to include Transformer-based Hybrid Spectral Predictor and HuBERT features
"""
from .hybrid_predictor import HybridSpectralPredictor
from .hybrid_predictor_transformer import (
    HybridSpectralPredictorTransformer,
    create_hybrid_predictor
)
from .hifigan import ControlAwareHiFiGAN
from .hybrid_svc import MultiTaskHybridSVC, HybridSVC
from .hybrid_vocoder import HybridVocoder
from .pitch_quantizer import PitchQuantizer
from .differentiable_pitch_quantizer import DifferentiablePitchQuantizer
from .transformer import (
    PositionalEncoding,
    RelativePositionalEncoding,
    ConditionalPositionalEncoding,
    MultiHeadAttention,
    FeedForwardNetwork,
    TransformerEncoderLayer,
    TransformerEncoder,
    HierarchicalTransformerEncoder,
    create_transformer_encoder
)
from .hubert import HubertFeatureExtractor, HuBERTManager, HubertFeatureProjection

__all__ = [
    "HybridSpectralPredictor",
    "HybridSpectralPredictorTransformer",
    "create_hybrid_predictor",
    "ControlAwareHiFiGAN",
    "MultiTaskHybridSVC",
    "HybridSVC",
    "HybridVocoder",
    "PitchQuantizer",
    "DifferentiablePitchQuantizer",
    "PositionalEncoding",
    "RelativePositionalEncoding",
    "ConditionalPositionalEncoding",
    "MultiHeadAttention",
    "FeedForwardNetwork",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "HierarchicalTransformerEncoder",
    "create_transformer_encoder"
]