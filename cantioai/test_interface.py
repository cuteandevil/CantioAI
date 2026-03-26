"""
Test script for Adversarial Interface using a mock discriminator
"""
import torch
import importlib.util

def load_module(file_path):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_adversarial_interface_with_mock():
    """Test Adversarial Interface with a mock discriminator."""
    print("Testing Adversarial Interface with mock discriminator...")

    # Load adversarial interface
    interface_mod = load_module("cantioai/discriminators/adversarial_interface.py")

    # Create a mock discriminator
    class MockDiscriminator:
        def __init__(self):
            pass

        def __call__(self, x, f0=None, sp=None, ap=None, mode="all"):
            # Return mock outputs
            if mode == "all":
                return {
                    "mpd": {
                        "outputs": [torch.randn(2, 1, 100) for _ in range(5)],
                        "features": [[torch.randn(2, 32, 100) for _ in range(12)] for _ in range(5)]
                    },
                    "msd": {
                        "outputs": [torch.randn(2, 1, 50) for _ in range(3)],
                        "features": [[torch.randn(2, 16, 50) for _ in range(8)] for _ in range(3)]
                    },
                    "source_filter": torch.randn(2, 1, 100),
                    "consistency": torch.randn(2, 1, 100)
                }
            elif mode == "audio_only":
                return {
                    "mpd": {
                        "outputs": [torch.randn(2, 1, 100) for _ in range(5)],
                        "features": [[torch.randn(2, 32, 100) for _ in range(12)] for _ in range(5)]
                    },
                    "msd": {
                        "outputs": [torch.randn(2, 1, 50) for _ in range(3)],
                        "features": [[torch.randn(2, 16, 50) for _ in range(8)] for _ in range(3)]
                    }
                }
            elif mode == "control_only":
                return {
                    "source_filter": torch.randn(2, 1, 100),
                    "consistency": torch.randn(2, 1, 100)
                }
            else:
                raise ValueError(f"Unknown mode: {mode}")

        def extract_audio_features(self, x):
            return torch.randn(2, 128, x.size(2))

    mock_disc = MockDiscriminator()

    # Create interface
    interface = interface_mod.AdversarialInterface(mock_disc, {
        "use_source_filter_disc": True,
        "use_consistency_disc": True,
        "use_feature_matching": True,
        "use_control_consistency": True
    })
    print(f"  Instantiated AdversarialInterface with mock discriminator")

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
    print(f"  Loss components: {list(d_loss_dict.keys())}")

    # Generator loss
    g_loss, g_loss_dict = interface.generator_loss(x, x, f0, sp, ap, pred_f0, pred_sp, pred_ap)
    print(f"  Generator loss computed: {g_loss.item():.4f}")
    print(f"  Loss components: {list(g_loss_dict.keys())}")

    print("  PASS Adversarial Interface test with mock discriminator\n")

if __name__ == "__main__":
    print("=" * 50)
    print("Adversarial Interface Test")
    print("=" * 50)

    try:
        test_adversarial_interface_with_mock()

        print("=" * 50)
        print("INTERFACE TEST PASSED")
        print("=" * 50)
    except Exception as e:
        print(f"FAILED: {e}")
        raise