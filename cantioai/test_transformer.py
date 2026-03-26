#!/usr/bin/env python3
"""
Test script to verify Transformer implementation works correctly
"""
import torch
import numpy as np
import sys
import os

# Add the parent directory to path so we can import cantioai modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hybrid_predictor_transformer import HybridSpectralPredictorTransformer
from src.models.transformer import TransformerEncoder
from src.config import load_config

def test_transformer_encoder():
    """Test the Transformer encoder directly"""
    print("Testing Transformer Encoder...")

    # Create a simple Transformer encoder
    encoder = TransformerEncoder(
        d_model=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1
    )

    # Create dummy input
    batch_size, seq_len, input_dim = 2, 100, 32+1+128  # phoneme + f0 + speaker
    x = torch.randn(batch_size, seq_len, input_dim)
    f0 = torch.randn(batch_size, seq_len, 1)
    spk_emb = torch.randn(batch_size, 128)

    # Forward pass
    output = encoder(x, f0=f0, spk_emb=spk_emb)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 512), f"Expected (2, 100, 512), got {output.shape}"
    print("✓ Transformer Encoder test passed!")

def test_hybrid_predictor_transformer():
    """Test the Transformer-based hybrid predictor"""
    print("\nTesting Hybrid Spectral Predictor with Transformer...")

    # Create Transformer-based predictor
    predictor = HybridSpectralPredictorTransformer(
        D_ph=32,
        D_sp=60,
        D_spk=128,
        n_speakers=100,
        transformer_type="hierarchical",
        transformer_hidden_dim=512,
        transformer_num_heads=8,
        transformer_num_layers=6,
        transformer_ff_dim=2048,
        transformer_dropout=0.1
    )

    # Create dummy input
    batch_size, seq_len = 2, 100
    phoneme_features = torch.randn(batch_size, seq_len, 32)
    f0 = torch.randn(batch_size, seq_len, 1)
    spk_id = torch.randint(0, 100, (batch_size,))

    # Forward pass
    output = predictor(phoneme_features, f0, spk_id)

    print(f"Phoneme features shape: {phoneme_features.shape}")
    print(f"F0 shape: {f0.shape}")
    print(f"Speaker ID shape: {spk_id.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, 60), f"Expected (2, 100, 60), got {output.shape}"
    print("✓ Hybrid Spectral Predictor Transformer test passed!")

def test_create_hybrid_predictor():
    """Test the factory function"""
    print("\nTesting create_hybrid_predictor factory...")

    # Test CNN+LSTM version
    predictor_cnn = create_hybrid_predictor(
        predictor_type="cnn_lstm",
        D_ph=32,
        D_sp=60,
        D_spk=128,
        n_speakers=100
    )

    # Test Transformer version
    predictor_transformer = create_hybrid_predictor(
        predictor_type="transformer",
        D_ph=32,
        D_sp=60,
        D_spk=128,
        n_speakers=100,
        transformer_type="hierarchical"
    )

    print(f"CNN+LSTM predictor type: {type(predictor_cnn).__name__}")
    print(f"Transformer predictor type: {type(predictor_transformer).__name__}")

    assert isinstance(predictor_cnn, HybridSpectralPredictor), "Should be CNN+LSTM version"
    assert isinstance(predictor_transformer, HybridSpectralPredictorTransformer), "Should be Transformer version"
    print("✓ Factory function test passed!")

def test_config_loading():
    """Test loading configuration"""
    print("\nTesting configuration loading...")

    try:
        config = load_config("../config.yaml")
        print("✓ Configuration loaded successfully")

        # Check if transformer config exists
        transformer_config = config.get('transformer', {})
        print(f"Transformer type: {transformer_config.get('type', 'Not found')}")
        print(f"Transformer hidden dim: {transformer_config.get('hidden_dim', 'Not found')}")

        assert transformer_config.get('type') == 'hierarchical', "Should be hierarchical transformer"
        assert transformer_config.get('hidden_dim') == 512, "Should have hidden dim 512"
        print("✓ Configuration test passed!")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        raise

if __name__ == "__main__":
    print("Running Transformer implementation tests...\n")

    try:
        test_transformer_encoder()
        test_hybrid_predictor_transformer()
        test_create_hybrid_predictor()
        test_config_loading()

        print("\n🎉 All tests passed! Transformer implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise