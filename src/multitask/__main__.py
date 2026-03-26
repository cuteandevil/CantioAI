"""
Main entry point for multi-task learning framework.
Demonstrates usage of all multi-task components.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from .shared_encoder import MultiTaskSharedEncoder
from .task_heads import (
    SingingConversionHead,
    SpeechConversionHead,
    NoiseRobustnessHead,
    create_task_head
)
from .adaptive_norm import TaskConditionedAdaIN
from .dynamic_routing import DynamicTaskRouter, HierarchicalTaskRouter
from .loss_design import (
    UncertaintyWeighting,
    GradientNormalization,
    TaskPrioritization,
    UncertaintyWeightedLoss
)
from .dataloader import create_multitask_dataloader
from .training_strategies import (
    MultiTaskTrainer,
    create_progressive_multi_task_trainer,
    create_curriculum_learning_trainer
)
from .evaluation_framework import (
    MultiTaskEvaluator,
    create_multitask_evaluator
)
from .config_extension import add_multitask_config, get_multitask_defaults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    """
    Example multi-task model combining shared encoder with task-specific heads.
    """

    def __init__(
        self,
        shared_encoder: nn.Module,
        task_heads: Dict[str, nn.Module],
        use_dynamic_routing: bool = True,
        routing_type: str = "attention"
    ):
        """
        Initialize multi-task model.

        Args:
            shared_encoder: Shared encoder module
            task_heads: Dictionary mapping task names to task-specific heads
            use_dynamic_routing: Whether to use dynamic task routing
            routing_type: Type of routing mechanism
        """
        super().__init__()

        self.shared_encoder = shared_encoder
        self.task_heads = nn.ModuleDict(task_heads)
        self.use_dynamic_routing = use_dynamic_routing
        self.routing_type = routing_type

        # Task names
        self.task_names = list(task_heads.keys())
        self.num_tasks = len(self.task_names)

        # Dynamic routing (optional)
        if use_dynamic_routing:
            # Get shared encoder output dimension
            # This is simplified - in practice would need to know the actual dimension
            shared_dim = 256  # Placeholder
            self.router = DynamicTaskRouter(
                shared_dim=shared_dim,
                hidden_dim=512,
                num_tasks=self.num_tasks,
                routing_type=routing_type,
                task_names=self.task_names
            )
        else:
            self.router = None

        logger.info(
            f"Initialized MultiTaskModel:\n"
            f"  Tasks: {self.task_names}\n"
            f"  Num tasks: {self.num_tasks}\n"
            f"  Use dynamic routing: {self.use_dynamic_routing}\n"
            f"  Routing type: {self.routing_type}"
        )

    def forward(
        self,
        x: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        return_routing_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass of multi-task model.

        Args:
            x: Input tensor (B, T, input_dim) or (B, input_dim)
            task_ids: Optional task IDs for conditioning (B,)
            return_routing_weights: Whether to return routing weights

        Returns:
            Dictionary containing:
                - task_outputs: Dictionary mapping task names to outputs
                - shared_features: Shared representation (if needed)
                - routing_weights: Routing weights (if return_routing_weights=True)
        """
        # Get shared features
        shared_features = self.shared_encoder(x)

        # Apply dynamic routing if enabled
        if self.router is not None:
            if return_routing_weights:
                routed_features, routing_weights = self.router(
                    shared_features, return_routing_weights=True
                )
            else:
                routed_features = self.router(shared_features, return_routing_weights=False)
                routing_weights = None
        else:
            routed_features = shared_features
            routing_weights = None

        # Compute task-specific outputs
        task_outputs = {}
        for task_name, task_head in self.task_heads.items():
            if isinstance(task_head, (SingingConversionHead, SpeechConversionHead)):
                # These heads may need additional inputs
                task_output = task_head(routed_features)
            else:
                # Standard task heads
                task_output = task_head(routed_features)
            task_outputs[task_name] = task_output

        # Prepare return dictionary
        result = {
            "task_outputs": task_outputs,
            "shared_features": shared_features
        }

        if return_routing_weights and routing_weights is not None:
            result["routing_weights"] = routing_weights

        return result


def create_demo_model() -> MultiTaskModel:
    """
    Create a demonstration multi-task model.

    Returns:
        MultiTaskModel instance for testing
    """
    # Create a simple shared encoder (placeholder)
    class SimpleSharedEncoder(nn.Module):
        def __init__(self, input_dim: int = 10, shared_dim: int = 256):
            super().__init__()
            self.input_dim = input_dim
            self.shared_dim = shared_dim
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, shared_dim),
                nn.ReLU()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    # Create shared encoder
    shared_encoder = SimpleSharedEncoder(input_dim=10, shared_dim=256)

    # Create task heads
    task_heads = {
        "singing": SingingConversionHead(
            shared_dim=256,
            predict_f0=True,
            predict_sp=True,
            predict_ap=True,
            sp_dim=60
        ),
        "speech": SpeechConversionHead(
            shared_dim=256,
            predict_f0=True,
            predict_sp=True,
            predict_ap=False,
            sp_dim=60
        ),
        "noise_robustness": NoiseRobustnessHead(
            shared_dim=256,
            predict_mask=True,
            predict_clean_features=True,
            predict_snr=True,
            feature_dim=60
        )
    }

    # Create multi-task model
    model = MultiTaskModel(
        shared_encoder=shared_encoder,
        task_heads=task_heads,
        use_dynamic_routing=True,
        routing_type="attention"
    )

    return model


def create_demo_data() -> Dict[str, TensorDataset]:
    """
    Create demonstration datasets for testing.

    Returns:
        Dictionary mapping task names to TensorDatasets
    """
    # Create dummy datasets for three tasks
    task1_data = torch.randn(100, 10)  # 100 samples, 10 features
    task1_labels = torch.randint(0, 2, (100,))  # Binary labels
    task1_dataset = TensorDataset(task1_data, task1_labels)

    task2_data = torch.randn(150, 10)  # 150 samples
    task2_labels = torch.randint(0, 3, (150,))  # 3-class labels
    task2_dataset = TensorDataset(task2_data, task2_labels)

    task3_data = torch.randn(80, 10)   # 80 samples
    task3_labels = torch.randint(0, 2, (80,))   # Binary labels
    task3_dataset = TensorDataset(task3_data, task3_labels)

    datasets = {
        "singing": task1_dataset,
        "speech": task2_dataset,
        "noise_robustness": task3_dataset
    }

    return datasets


def main():
    """Main function demonstrating multi-task learning framework."""
    print("=" * 60)
    print("CantioAI Multi-Task Learning Framework Demo")
    print("=" * 60)

    # 1. Create configuration
    print("\n1. Configuration System Extension")
    print("-" * 40)
    cfg = get_multitask_defaults()
    print("✓ Multi-task configuration extended successfully")

    # 2. Create shared encoder
    print("\n2. Shared Encoder")
    print("-" * 40)
    shared_encoder = MultiTaskSharedEncoder(
        base_encoder=nn.Linear(10, 256),  # Simple placeholder
        shared_dim=256,
        num_layers=2,
        dropout=0.1
    )
    print("✓ Multi-task shared encoder created successfully")

    # 3. Create task heads
    print("\n3. Task-Specific Heads")
    print("-" * 40)
    task_heads = {
        "singing": SingingConversionHead(
            shared_dim=256,
            predict_f0=True,
            predict_sp=True,
            predict_ap=True,
            sp_dim=60
        ),
        "speech": SpeechConversionHead(
            shared_dim=256,
            predict_f0=True,
            predict_sp=True,
            predict_ap=False,
            sp_dim=60
        ),
        "noise_robustness": NoiseRobustnessHead(
            shared_dim=256,
            predict_mask=True,
            predict_clean_features=True,
            predict_snr=True,
            feature_dim=60
        )
    }
    print("✓ Task-specific heads created successfully")

    # 4. Create adaptive normalization
    print("\n4. Adaptive Normalization")
    print("-" * 40)
    adain = TaskConditionedAdaIN(
        num_features=256,
        speaker_embed_dim=128,
        task_embed_dim=64,
        num_speakers=100,
        num_tasks=3
    )
    print("✓ Task-conditioned AdaIN created successfully")

    # 5. Create dynamic routing
    print("\n5. Dynamic Task Routing")
    print("-" * 40)
    router = DynamicTaskRouter(
        shared_dim=256,
        hidden_dim=512,
        num_tasks=3,
        routing_type="attention",
        task_names=["singing", "speech", "noise_robustness"]
    )
    print("✓ Dynamic task router created successfully")

    # 6. Create loss design components
    print("\n6. Loss Function Design")
    print("-" * 40)
    uncertainty_weighting = UncertaintyWeighting(num_tasks=3)
    gradient_normalization = GradientNormalization(num_tasks=3)
    task_prioritization = TaskPrioritization(num_tasks=3)
    uncertainty_weighted_loss = UncertaintyWeightedLoss(num_tasks=3)
    print("✓ Loss function design components created successfully")

    # 7. Create data loader
    print("\n7. Multi-Task Data Loader")
    print("-" * 40)
    datasets = create_demo_data()
    dataloader = create_multitask_dataloader(
        datasets=datasets,
        batch_size=16,
        sampling_strategy="uniform"
    )
    print("✓ Multi-task data loader created successfully")

    # 8. Create training strategies
    print("\n8. Training Strategies")
    print("-" * 40)
    model = create_demo_model()
    criterion = nn.MSELoss()  # Simple placeholder
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    progressive_trainer = create_progressive_multi_task_trainer(
        model=model,
        tasks=["singing", "speech", "noise_robustness"],
        datasets=datasets,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    curriculum_trainer = create_curriculum_learning_trainer(
        model=model,
        tasks=["singing", "speech", "noise_robustness"],
        datasets=datasets,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    print("✓ Training strategies created successfully")

    # 9. Create evaluation framework
    print("\n9. Evaluation Framework")
    print("-" * 40)
    evaluator = create_multitask_evaluator(
        tasks=["singing", "speech", "noise_robustness"]
    )
    print("✓ Evaluation framework created successfully")

    # 10. Demonstrate forward pass
    print("\n10. Model Forward Pass Demo")
    print("-" * 40)
    model.eval()
    with torch.no_grad():
        # Create dummy input
        dummy_input = torch.randn(4, 10)  # Batch size 4, input dim 10
        dummy_task_ids = torch.randint(0, 3, (4,))  # Random task IDs

        # Forward pass
        outputs = model(dummy_input, task_ids=dummy_task_ids, return_routing_weights=True)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Task IDs shape: {dummy_task_ids.shape}")
        print(f"Shared features shape: {outputs['shared_features'].shape}")
        print(f"Task outputs keys: {list(outputs['task_outputs'].keys())}")
        for task_name, task_output in outputs['task_outputs'].items():
            print(f"  {task_name} output shape: {task_output.shape}")
        if 'routing_weights' in outputs:
            print(f"Routing weights shape: {outputs['routing_weights'].shape}")

    print("\n" + "=" * 60)
    print("Multi-Task Learning Framework Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()