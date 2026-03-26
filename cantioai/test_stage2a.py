"""
Test script for Stage 2A: Adversarial Training Components Test
"""
import torch
import importlib.util

def load_module(file_path):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_control_aware_mpd():
    """Test Control-Aware MPD."""
    print("Testing Control-Aware MPD...")
    mod = load_module("cantioai/discriminators/control_aware_mpd.py")
    config = {
        'control_dim': 64,
        'leaky_relu_slope': 0.1,
        'periods': [2, 3, 5, 7, 11]
    }
    disc = mod.ControlAwareMPD(config)
    print(f"  Instantiated ControlAwareMPD with periods {config['periods']}")

    # Test forward pass
    x = torch.randn(2, 1, 1000)  # B, 1, T
    f0 = torch.rand(2, 1, 100) * 200 + 100  # B, 1, T
    sp = torch.rand(2, 16, 100)   # B, sp_dim, T
    ap = torch.rand(2, 1, 100)    # B, ap_dim, T
    output = disc(x, f0, sp, ap)
    print(f"  Forward pass successful")
    print(f"  Number of outputs: {len(output['outputs'])}")
    print(f"  Shape of first output: {output['outputs'][0].shape}")
    assert len(output['outputs']) == len(config['periods'])
    print("  PASS Control-Aware MPD test\n")

def test_control_aware_msd():
    """Test Control-Aware MSD."""
    print("Testing Control-Aware MSD...")
    mod = load_module("cantioai/discriminators/control_aware_msd.py")
    config = {
        'control_dim': 64,
        'leaky_relu_slope': 0.1,
        'scales': [1, 2, 4]
    }
    disc = mod.ControlAwareMSD(config)
    print(f"  Instantiated ControlAwareMSD with scales {config['scales']}")

    # Test forward pass
    x = torch.randn(2, 1, 1000)  # B, 1, T
    f0 = torch.rand(2, 1, 100) * 200 + 100  # B, 1, T
    sp = torch.rand(2, 16, 100)   # B, sp_dim, T
    ap = torch.rand(2, 1, 100)    # B, ap_dim, T
    outputs, features = disc(x, f0, sp, ap)
    print(f"  Forward pass successful")
    print(f"  Number of outputs: {len(outputs)}")
    print(f"  Number of feature lists: {len(features)}")
    assert len(outputs) == len(config['scales'])
    print("  PASS Control-Aware MSD test\n")

def test_hybrid_discriminator():
    """Test Hybrid Discriminator System."""
    print("Testing Hybrid Discriminator System...")
    mod = load_module("cantioai/discriminators/hybrid_discriminator.py")
    config = {
        "mpd": {
            "control_dim": 64,
            "leaky_relu_slope": 0.1,
            "periods": [2, 3, 5, 7, 11]
        },
        "msd": {
            "control_dim": 64,
            "leaky_relu_slope": 0.1,
            "scales": [1, 2, 4]
        },
        "source_filter_discriminator": {
            "f0_dim": 1,
            "sp_dim": 60,
            "ap_dim": 1,
            "hidden_dim": 256
        },
        "consistency_discriminator": {
            "audio_feat_dim": 128,
            "control_dim": 64,
            "hidden_dim": 256
        }
    }
    disc = mod.HybridDiscriminatorSystem(config)
    print(f"  Instantiated HybridDiscriminatorSystem")

    # Test forward pass
    x = torch.randn(2, 1, 1000)  # B, 1, T
    f0 = torch.rand(2, 1, 100) * 200 + 100  # B, 1, T
    sp = torch.rand(2, 16, 100)   # B, sp_dim, T
    ap = torch.rand(2, 1, 100)    # B, ap_dim, T
    results = disc(x, f0, sp, ap, mode="all")
    print(f"  Forward pass successful")
    print(f"  Results keys: {list(results.keys())}")
    assert "mpd" in results
    assert "msd" in results
    assert "source_filter" in results
    assert "consistency" in results
    assert results["source_filter"] is not None
    assert results["consistency"] is not None
    print("  PASS Hybrid Discriminator System test\n")

def test_losses():
    """Test loss functions."""
    print("Testing Loss Functions...")
    # Adversarial loss
    mod = load_module("cantioai/discriminators/losses/adversarial_loss.py")
    loss = mod.AdversarialLoss({})
    print(f"  Instantiated AdversarialLoss")

    # Feature matching loss
    mod = load_module("cantioai/discriminators/losses/feature_matching_loss.py")
    loss = mod.FeatureMatchingLoss({})
    print(f"  Instantiated FeatureMatchingLoss")

    # Control consistency loss
    mod = load_module("cantioai/discriminators/losses/control_consistency_loss.py")
    loss = mod.ControlConsistencyLoss({})
    print(f"  Instantiated ControlConsistencyLoss")
    print("  PASS Loss Functions test\n")

def test_adversarial_interface():
    """Test Adversarial Interface."""
    print("Testing Adversarial Interface...")
    # Load hybrid discriminator
    disc_mod = load_module("cantioai/discriminators/hybrid_discriminator.py")
    disc_config = {
        "mpd": {
            "control_dim": 64,
            "leaky_relu_slope": 0.1,
            "periods": [2, 3, 5, 7, 11]
        },
        "msd": {
            "control_dim": 64,
            "leaky_relu_slope": 0.1,
            "scales": [1, 2, 4]
        },
        "source_filter_discriminator": {
            "f0_dim": 1,
            "sp_dim": 60,
            "ap_dim": 1,
            "hidden_dim": 256
        },
        "consistency_discriminator": {
            "audio_feat_dim": 128,
            "control_dim": 64,
            "hidden_dim": 256
        }
    }
    disc = disc_mod.HybridDiscriminatorSystem(disc_config)

    # Load adversarial interface
    interface_mod = load_module("cantioai/discriminators/adversarial_interface.py")
    interface = interface_mod.AdversarialInterface(disc, {
        "use_source_filter_disc": True,
        "use_consistency_disc": True,
        "use_feature_matching": True,
        "use_control_consistency": True
    })
    print(f"  Instantiated AdversarialInterface")

    # Test losses
    x = torch.randn(2, 1, 1000)  # B, 1, T
    f0 = torch.rand(2, 1, 100) * 200 + 100  # B, 1, T
    sp = torch.rand(2, 16, 100)   # B, sp_dim, T
    ap = torch.rand(2, 1, 100)    # B, ap_dim, T
    pred_f0 = f0.clone()
    pred_sp = sp.clone()
    pred_ap = ap.clone()

    # Discriminator loss
    d_loss, d_loss_dict = interface.discriminator_loss(x, x.clone(), f0, sp, ap)
    print(f"  Discriminator loss computed: {d_loss.item():.4f}")

    # Generator loss
    g_loss, g_loss_dict = interface.generator_loss(x, x, f0, sp, ap, pred_f0, pred_sp, pred_ap)
    print(f"  Generator loss computed: {g_loss.item():.4f}")

    print("  PASS Adversarial Interface test\n")

if __name__ == "__main__":
    print("=" * 50)
    print("Stage 2A: Adversarial Training Components Test")
    print("=" * 50)

    try:
        test_control_aware_mpd()
        test_control_aware_msd()
        test_hybrid_discriminator()
        test_losses()
        test_adversarial_interface()

        print("=" * 50)
        print("ALL STAGE 2A TESTS PASSED")
        print("=" * 50)
    except Exception as e:
        print(f"FAILED: {e}")
        raise