"""CantioAI Discriminators Package."""
from .control_aware_mpd import ControlAwareMPD
from .control_aware_msd import ControlAwareMSD
from .hybrid_discriminator import HybridDiscriminatorSystem
from .adversarial_interface import AdversarialInterface
from .specialized import SourceFilterDiscriminator, ControlConsistencyDiscriminator, MultiScaleControlDiscriminator

__all__ = [
    "ControlAwareMPD",
    "ControlAwareMSD",
    "HybridDiscriminatorSystem",
    "AdversarialInterface",
    "SourceFilterDiscriminator",
    "ControlConsistencyDiscriminator",
    "MultiScaleControlDiscriminator"
]