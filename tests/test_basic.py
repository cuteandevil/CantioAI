"""
Basic tests for CantioAI components.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from cantioai.src.models.hybrid_svc import HybridSVC
from cantioai.src.data.preprocess import WorldFeatureExtractor
from cantioai.src.data.dataset import CantioAIDataset
from cantioai.src.utils.logging import setup_logging
from cantioai.src.utils.config import load_config

logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    # This is just to verify imports work
    assert HybridSVC is not None
    assert WorldFeatureExtractor is not None
    assert CantioAIDataset is not None
    assert setup_logging is not None
    assert load_config is not None
    logger.info("All imports successful")


def test_world_feature_extractor():
    """Test WORLD feature extractor basic functionality."""
    # Initialize extractor
    extractor = WorldFeatureExtractor(
        sample_rate=16000,
        frame_period=5.0,
        fft_size=512
    )

    # Create dummy waveform
    sr = 16000
    duration = 0.1  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    frequency = 220.0  # Hz
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    waveform = waveform.astype(np.float32)

    # Extract features
    features = extractor.extract_features(waveform)

    # Check output structure
    assert "f0" in features
    assert "sp" in features
    assert "ap" in features
    assert "coded_sp" in features

    # Check shapes
    T = waveform.shape[0]
    assert features["f0"].shape == (T, 1)
    assert features["sp"].shape == (T, 256)  # fft_size//2+1
    assert features["ap"].shape == (T, 256)  # fft_size//2+1
    assert features["coded_sp"].shape == (T, 60)  # num_mcep

    # Check value types
    assert features["f0"].dtype == np.float32
    assert features["sp"].dtype == np.float32
    assert features["ap"].dtype == np.float32
    assert features["coded_sp"].dtype == np.float32

    # Check that F0 is in reasonable range
    assert np.all(features["f0"] >= 0)
    assert np.any(features["f0"] > 0)  # At least some voiced frames

    logger.info("WorldFeatureExtractor test passed")


def test_dataset_loading():
    """Test dataset loading and basic functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy feature files
        feature_dir = tmpdir / "features"
        feature_dir.mkdir()

        # Create a few dummy .npz files
        for i in range(3):
            feature_file = feature_dir / f"sample_{i:02d}_features.npz"
            dummy_data = {
                "phoneme_features": np.random.randn(10, 32).astype(np.float32),
                "f0": np.random.uniform(80, 300, (10, 1)).astype(np.float32),  # Hz
                "spk_id": np.full((10, 1), 42, dtype=np.int64),
                "target_sp": np.random.randn(10, 60).astype(np.float32),
            }
            np.savez(feature_file, **dummy_data)

        # Test dataset creation
        dataset = CantioAIDataset(
            data_dir=tmpdir,
            split="train"
        )

        assert len(dataset) == 3

        # Test data loading
        sample = dataset[0]
        assert "phoneme_features" in sample
        assert "f0" in sample
        assert "spk_id" in sample
        assert "target_sp" in sample

        # Check shapes and types
        assert sample["phoneme_features"].shape == (10, 32)
        assert sample["phoneme_features"].dtype == torch.float32
        assert sample["f0"].shape == (1, 1)  # unsqueezed
        assert sample["f0"].dtype == torch.float32
        assert sample["spk_id"].shape == ()
        assert sample["spk_id"].dtype == torch.int64
        assert sample["target_sp"].shape == (10, 60)
        assert sample["target_sp"].dtype == torch.float32

        # Test iteration
        for i, sample in enumerate(dataset):
            # Just verify we can iterate
            pass

        # Test speaker IDs
        speaker_ids = dataset.get_speaker_ids()
        assert len(speaker_ids) == 1
        assert speaker_ids[0] == 42

        # Test statistics
        stats = dataset.get_statistics()
        assert stats["num_samples"] == 3
        assert stats["num_speakers"] == 1

        logger.info("CantioAIDataset test passed")


def test_hybrid_model_forward():
    """Test HybridSVC model forward pass."""
    # Initialize model
    model = HybridSVC(
        phoneme_feature_dim=32,
        spectral_envelope_dim=60,
        speaker_embed_dim=128,
        n_speakers=10,
        use_pitch_quantizer=False  # Disable for simpler test
    )

    batch_size = 2
    seq_len = 20

    # Create dummy inputs
    phoneme_features = torch.randn(batch_size, seq_len, 32)
    f0_hz = torch.rand(batch_size, seq_len, 1) * 200 + 100  # 100-300 Hz
    spk_id = torch.randint(0, 10, (batch_size,))

    # Forward pass
    sp_pred, _, _ = model(
        phoneme_features=phoneme_features,
        f0=f0_hz,
        spk_id=spk_id,
        f0_is_hz=True,
        return_quantized_f0=False
    )

    # Check output
    assert sp_pred is not None
    assert sp_pred.shape == (batch_size, seq_len, 60)
    assert sp_pred.dtype == torch.float32

    logger.info("HybridSVC forward pass test passed")


def test_config_loading():
    """Test configuration loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_path = f.name

        # Write minimal config
        config_data = {
            "feature": {"sample_rate": 16000},
            "model": {"phoneme_feature_dim": 32},
            "training": {"batch_size": 4}
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_config(config_path)

        # Check values
        assert config["feature"]["sample_rate"] == 16000
        assert config["model"]["phoneme_feature_dim"] == 32
        assert config["training"]["batch_size"] == 4

        logger.info("Configuration loading test passed")


def test_logger_setup():
    """Test logging setup."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name

        # Setup logging
        logger = setup_logging(logging.DEBUG, log_file=log_file)

        # Test logging
        logger.info("Test INFO message")
        logger.debug("Test DEBUG message")
        logger.warning("Test WARNING message")
        logger.error("Test ERROR message")

        # Check log file
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test INFO message" in content
            assert "Test DEBUG message" in content
            assert "Test WARNING message" in content
            assert "Test ERROR message" in content

        logger.info("Logger setup test passed")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_imports()
    test_world_feature_extractor()
    test_dataset_loading()
    test_hybrid_model_forward()
    test_config_loading()
    test_logger_setup()
    print("All basic tests passed!")