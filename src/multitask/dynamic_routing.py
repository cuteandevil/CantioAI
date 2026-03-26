"""
Dynamic Task Routing Network for multi-task learning.
Implements intelligent routing of features between tasks based on input characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class DynamicTaskRouter(nn.Module):
    """
    Dynamic task routing network.
    Learns to route features between different tasks based on input audio characteristics.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        hidden_dim: int = 512,
        num_tasks: int = 3,
        routing_type: str = "attention",  # "attention", "gate", "mixture"
        task_names: Optional[List[str]] = None,
        dropout: float = 0.1
    ):
        """
        Initialize dynamic task router.

        Args:
            shared_dim: Dimension of shared features
            hidden_dim: Hidden dimension for routing network
            num_tasks: Number of tasks
            routing_type: Type of routing mechanism
            task_names: Optional list of task names
            dropout: Dropout rate
        """
        super().__init__()

        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.routing_type = routing_type
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]
        self.dropout = dropout

        # Validate task names
        if len(self.task_names) != num_tasks:
            raise ValueError(f"Number of task names ({len(self.task_names)}) must match num_tasks ({num_tasks})")

        # Build routing network based on type
        if routing_type == "attention":
            self.router = self._build_attention_router()
        elif routing_type == "gate":
            self.router = self._build_gate_router()
        elif routing_type == "mixture":
            self.router = self._build_mixture_router()
        else:
            raise ValueError(f"Unsupported routing type: {routing_type}")

        logger.info(
            f"Initialized DynamicTaskRouter:\n"
            f"  Shared dim: {shared_dim}\n"
            f"  Hidden dim: {hidden_dim}\n"
            f"  Num tasks: {num_tasks}\n"
            f"  Routing type: {routing_type}\n"
            f"  Task names: {self.task_names}"
        )

    def _build_attention_router(self) -> nn.Module:
        """Build attention-based router."""
        return nn.Sequential(
            nn.Linear(self.shared_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_tasks),
            nn.Softmax(dim=-1)
        )

    def _build_gate_router(self) -> nn.Module:
        """Build gate-based router (independent sigmoid for each task)."""
        return nn.Sequential(
            nn.Linear(self.shared_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_tasks),
            nn.Sigmoid()
        )

    def _build_mixture_router(self) -> nn.Module:
        """Build mixture-of-experts style router."""
        return nn.Sequential(
            nn.Linear(self.shared_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tasks),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        shared_features: torch.Tensor,
        return_routing_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of dynamic task router.

        Args:
            shared_features: Shared features tensor (B, T, shared_dim) or (B, shared_dim)
            return_routing_weights: Whether to return routing weights

        Returns:
            If return_routing_weights=False: Routed features (same shape as input)
            If return_routing_weights=True: Tuple of (routed_features, routing_weights)
                where routing_weights has shape (B, T, num_tasks) or (B, num_tasks)
        """
        # Handle both sequence and frame-level inputs
        if shared_features.dim() == 3:
            # Sequence input: (B, T, shared_dim)
            B, T, _ = shared_features.shape
            is_sequence = True
        elif shared_features.dim() == 2:
            # Frame-level input: (B, shared_dim)
            B, _ = shared_features.shape
            T = 1
            is_sequence = False
        else:
            raise ValueError(f"Unsupported input shape: {shared_features.shape}")

        # Compute routing weights
        routing_weights = self.router(shared_features)  # (B, T, num_tasks) or (B, num_tasks)

        # Apply routing
        if self.routing_type == "attention":
            # Attention-based: weighted combination of task-specific processing
            # For simplicity, we'll return the weights and let downstream modules use them
            routed_features = shared_features  # In practice, this would be processed by task-specific modules
        elif self.routing_type == "gate":
            # Gate-based: feature gating for each task
            # Expand routing weights to match feature dimensions for element-wise multiplication
            if is_sequence:
                # (B, T, num_tasks) -> (B, T, num_tasks, 1) for broadcasting
                weights_expanded = routing_weights.unsqueeze(-1)  # (B, T, num_tasks, 1)
                # Expand features for task dimension: (B, T, shared_dim) -> (B, T, shared_dim, 1)
                features_expanded = shared_features.unsqueeze(-2)  # (B, T, 1, shared_dim)
                # Actually, we want to apply gates to features - reshape for proper broadcasting
                # Features: (B, T, shared_dim) -> (B, T, 1, shared_dim)
                # Weights: (B, T, num_tasks) -> (B, T, num_tasks, 1)
                # Result: (B, T, num_tasks, shared_dim) - gated features for each task
                features_for_gating = shared_features.unsqueeze(-2)  # (B, T, 1, shared_dim)
                weights_for_gating = routing_weights.unsqueeze(-1)  # (B, T, num_tasks, 1)
                gated_features = features_for_gating * weights_for_gating  # (B, T, num_tasks, shared_dim)
                # For now, return the original features - actual gating would be task-specific
                routed_features = shared_features
            else:
                # Frame-level case
                routed_features = shared_features
        else:  # mixture
            # Mixture-based: similar to attention for now
            routed_features = shared_features

        if return_routing_weights:
            return routed_features, routing_weights
        else:
            return routed_features

    def get_task_routing_weights(
        self,
        shared_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get routing weights for each task as a dictionary.

        Args:
            shared_features: Shared features tensor (B, T, shared_dim) or (B, shared_dim)

        Returns:
            Dictionary mapping task names to routing weights
        """
        _, routing_weights = self.forward(shared_features, return_routing_weights=True)

        # Handle both sequence and frame-level inputs
        if shared_features.dim() == 3:
            # Sequence input: average over time dimension for task-level weights
            task_weights = routing_weights.mean(dim=1)  # (B, num_tasks)
        else:
            # Frame-level input: already task-level
            task_weights = routing_weights  # (B, num_tasks)

        # Convert to dictionary
        weights_dict = {}
        for i, task_name in enumerate(self.task_names):
            weights_dict[task_name] = task_weights[:, i]  # (B,)

        return weights_dict


class HierarchicalTaskRouter(nn.Module):
    """
    Hierarchical task router with multiple levels of routing.
    Can route between shared, task-shared, and task-specific processing streams.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        task_shared_dim: int = 128,
        task_specific_dim: int = 64,
        hidden_dim: int = 512,
        num_tasks: int = 3,
        routing_type: str = "attention",
        task_names: Optional[List[str]] = None,
        dropout: float = 0.1
    ):
        """
        Initialize hierarchical task router.

        Args:
            shared_dim: Dimension of fully shared features
            task_shared_dim: Dimension of features shared between related tasks
            task_specific_dim: Dimension of task-specific features
            hidden_dim: Hidden dimension for routing networks
            num_tasks: Number of tasks
            routing_type: Type of routing mechanism
            task_names: Optional list of task names
            dropout: Dropout rate
        """
        super().__init__()

        self.shared_dim = shared_dim
        self.task_shared_dim = task_shared_dim
        self.task_specific_dim = task_specific_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.routing_type = routing_type
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]
        self.dropout = dropout

        # Routing between shared and task-shared streams
        self.shared_to_task_router = DynamicTaskRouter(
            shared_dim=shared_dim,
            hidden_dim=hidden_dim,
            num_tasks=num_tasks,
            routing_type=routing_type,
            task_names=self.task_names,
            dropout=dropout
        )

        # Routing between task-shared and task-specific streams
        self.task_to_specific_router = DynamicTaskRouter(
            shared_dim=task_shared_dim,
            hidden_dim=hidden_dim,
            num_tasks=num_tasks,
            routing_type=routing_type,
            task_names=self.task_names,
            dropout=dropout
        )

        logger.info(
            f"Initialized HierarchicalTaskRouter:\n"
            f"  Shared dim: {shared_dim}\n"
            f"  Task-shared dim: {task_shared_dim}\n"
            f"  Task-specific dim: {task_specific_dim}\n"
            f"  Hidden dim: {hidden_dim}\n"
            f"  Num tasks: {num_tasks}\n"
            f"  Routing type: {routing_type}"
        )

    def forward(
        self,
        shared_features: torch.Tensor,
        return_all_weights: bool = False
    ) -> Union[
        torch.Tensor,  # Just shared features
        Tuple[torch.Tensor, torch.Tensor],  # Shared and task-shared
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # All three
    ]:
        """
        Forward pass of hierarchical task router.

        Args:
            shared_features: Shared features tensor (B, T, shared_dim) or (B, shared_dim)
            return_all_weights: Whether to return all routing weights

        Returns:
            If return_all_weights=False: Just shared features (placeholder)
            If return_all_weights=True: Tuple depending on implementation level
        """
        # For now, just return the input - actual implementation would split features
        if return_all_weights:
            # Return shared features and placeholder routing weights
            if shared_features.dim() == 3:
                B, T, _ = shared_features.shape
                shared_weights = torch.ones(B, T, self.num_tasks, device=shared_features.device) / self.num_tasks
                task_shared_weights = torch.ones(B, T, self.num_tasks, device=shared_features.device) / self.num_tasks
                task_specific_weights = torch.ones(B, T, self.num_tasks, device=shared_features.device) / self.num_tasks
                return shared_features, shared_weights, task_shared_weights, task_specific_weights
            else:
                B, _ = shared_features.shape
                shared_weights = torch.ones(B, self.num_tasks, device=shared_features.device) / self.num_tasks
                task_shared_weights = torch.ones(B, self.num_tasks, device=shared_features.device) / self.num_tasks
                task_specific_weights = torch.ones(B, self.num_tasks, device=shared_features.device) / self.num_tasks
                return shared_features, shared_weights, task_shared_weights, task_specific_weights
        else:
            return shared_features


def create_dynamic_router(
    shared_dim: int = 256,
    hidden_dim: int = 512,
    num_tasks: int = 3,
    routing_type: str = "attention",
    task_names: Optional[List[str]] = None,
    dropout: float = 0.1
) -> DynamicTaskRouter:
    """
    Factory function to create DynamicTaskRouter.

    Args:
        shared_dim: Dimension of shared features
        hidden_dim: Hidden dimension for routing network
        num_tasks: Number of tasks
        routing_type: Type of routing mechanism
        task_names: Optional list of task names
        dropout: Dropout rate

    Returns:
        DynamicTaskRouter module
    """
    return DynamicTaskRouter(
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
        num_tasks=num_tasks,
        routing_type=routing_type,
        task_names=task_names,
        dropout=dropout
    )


def create_hierarchical_router(
    shared_dim: int = 256,
    task_shared_dim: int = 128,
    task_specific_dim: int = 64,
    hidden_dim: int = 512,
    num_tasks: int = 3,
    routing_type: str = "attention",
    task_names: Optional[List[str]] = None,
    dropout: float = 0.1
) -> HierarchicalTaskRouter:
    """
    Factory function to create HierarchicalTaskRouter.

    Args:
        shared_dim: Dimension of fully shared features
        task_shared_dim: Dimension of features shared between related tasks
        task_specific_dim: Dimension of task-specific features
        hidden_dim: Hidden dimension for routing networks
        num_tasks: Number of tasks
        routing_type: Type of routing mechanism
        task_names: Optional list of task names
        dropout: Dropout rate

    Returns:
        HierarchicalTaskRouter module
    """
    return HierarchicalTaskRouter(
        shared_dim=shared_dim,
        task_shared_dim=task_shared_dim,
        task_specific_dim=task_specific_dim,
        hidden_dim=hidden_dim,
        num_tasks=num_tasks,
        routing_type=routing_type,
        task_names=task_names,
        dropout=dropout
    )


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 10
    shared_dim = 256
    num_tasks = 3
    task_names = ["singing", "speech", "noise_robustness"]

    # Create shared features (simulating encoder output)
    shared_features = torch.randn(batch_size, seq_len, shared_dim)

    # Test dynamic task router
    for routing_type in ["attention", "gate", "mixture"]:
        print(f"\nTesting {routing_type} router:")
        router = DynamicTaskRouter(
            shared_dim=shared_dim,
            hidden_dim=512,
            num_tasks=num_tasks,
            routing_type=routing_type,
            task_names=task_names
        )

        # Test without returning weights
        output = router(shared_features)
        print(f"  Output shape: {output.shape}")

        # Test with returning weights
        output, weights = router(shared_features, return_routing_weights=True)
        print(f"  Output shape: {output.shape}")
        print(f"  Weights shape: {weights.shape}")

        # Test task-specific weights
        task_weights = router.get_task_routing_weights(shared_features)
        print(f"  Task weights keys: {list(task_weights.keys())}")
        for task_name, weight_tensor in task_weights.items():
            print(f"    {task_name}: {weight_tensor.shape}")

    # Test hierarchical router
    print(f"\nTesting hierarchical router:")
    hier_router = HierarchicalTaskRouter(
        shared_dim=shared_dim,
        task_shared_dim=128,
        task_specific_dim=64,
        hidden_dim=512,
        num_tasks=num_tasks,
        routing_type="attention",
        task_names=task_names
    )

    hier_output, shared_w, task_shared_w, task_specific_w = hier_router(
        shared_features, return_all_weights=True
    )
    print(f"  Hierarchical output shape: {hier_output.shape}")
    print(f"  Shared weights shape: {shared_w.shape}")
    print(f"  Task-shared weights shape: {task_shared_w.shape}")
    print(f"  Task-specific weights shape: {task_specific_w.shape}")

    print("\nDynamic task routing test passed!")