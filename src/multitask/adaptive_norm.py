"""
Multi-task Adaptive Instance Normalization (AdaIN) with task conditioning.
Extends standard AdaIN to include task-specific adaptive parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TaskConditionedAdaIN(nn.Module):
    """
    Task-conditioned Adaptive Instance Normalization.
    Computes adaptive parameters based on both speaker ID and task ID.
    """

    def __init__(
        self,
        num_features: int,
        speaker_embed_dim: int = 128,
        task_embed_dim: int = 64,
        num_speakers: int = 100,
        num_tasks: int = 3,
        eps: float = 1e-5,
        momentum: float = 0.1,
        use_task_embedding: bool = True
    ):
        """
        Initialize task-conditioned AdaIN.

        Args:
            num_features: Number of features/channels to normalize
            speaker_embed_dim: Dimension of speaker embeddings
            task_embed_dim: Dimension of task embeddings
            num_speakers: Number of speakers in training data
            num_tasks: Number of tasks in multi-task learning
            eps: Small constant for numerical stability
            momentum: Momentum for running mean/var
            use_task_embedding: Whether to use task embeddings
        """
        super().__init__()

        self.num_features = num_features
        self.speaker_embed_dim = speaker_embed_dim
        self.task_embed_dim = task_embed_dim
        self.num_speakers = num_speakers
        self.num_tasks = num_tasks
        self.eps = eps
        self.momentum = momentum
        self.use_task_embedding = use_task_embedding

        # Speaker embedding
        self.speaker_emb = nn.Embedding(num_speakers, speaker_embed_dim)

        # Task embedding (optional)
        if use_task_embedding:
            self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
            embed_dim = speaker_embed_dim + task_embed_dim
        else:
            self.task_emb = None
            embed_dim = speaker_embed_dim

        # Parameter projection networks
        self.weight_generater = nn.Sequential(
            nn.Linear(embed_dim, num_features),
            nn.ReLU()
        )
        self.bias_generater = nn.Sequential(
            nn.Linear(embed_dim, num_features),
            nn.ReLU()
        )

        # Running statistics (for inference mode)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        logger.info(
            f"Initialized TaskConditionedAdaIN:\n"
            f"  Num features: {num_features}\n"
            f"  Speaker embed dim: {speaker_embed_dim}\n"
            f"  Task embed dim: {task_embed_dim if use_task_embedding else 0}\n"
            f"  Num speakers: {num_speakers}\n"
            f"  Num tasks: {num_tasks}\n"
            f"  Use task embedding: {use_task_embedding}"
        )

    def forward(
        self,
        x: torch.Tensor,
        speaker_ids: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of task-conditioned AdaIN.

        Args:
            x: Input tensor of shape (B, C, *) where C = num_features
            speaker_ids: Speaker IDs of shape (B,)
            task_ids: Optional task IDs of shape (B,) - required if use_task_embedding=True

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute instance mean and variance
        # x shape: (B, C, *, ...) -> reduce over spatial dimensions
        dims = list(range(2, x.dim()))  # Dimensions to reduce (spatial)
        if len(dims) == 0:
            # If no spatial dimensions (e.g., (B, C)), reduce nothing
            mean = x.mean(dim=[1], keepdim=True)  # (B, 1, *)
            var = x.var(dim=[1], keepdim=True, unbiased=False)  # (B, 1, *)
        else:
            # Reduce over spatial dimensions
            mean = x.mean(dim=dims, keepdim=True)  # (B, C, 1, ...)
            var = x.var(dim=dims, keepdim=True, unbiased=False)  # (B, C, 1, ...)

        # Get speaker embeddings
        speaker_embedded = self.speaker_emb(speaker_ids)  # (B, speaker_embed_dim)

        # Get task embeddings if enabled
        if self.use_task_embedding:
            if task_ids is None:
                raise ValueError("task_ids must be provided when use_task_embedding=True")
            task_embedded = self.task_emb(task_ids)  # (B, task_embed_dim)
            # Concatenate speaker and task embeddings
            embedded = torch.cat([speaker_embedded, task_embedded], dim=-1)  # (B, speaker_embed_dim + task_embed_dim)
        else:
            embedded = speaker_embedded  # (B, speaker_embed_dim)

        # Generate adaptive parameters
        weight = self.weight_generater(embedded)  # (B, num_features)
        bias = self.bias_generater(embedded)  # (B, num_features)

        # Reshape weight and bias for broadcasting
        # weight/bias shape: (B, num_features) -> (B, num_features, 1, ...)
        weight_shape = [weight.shape[0], weight.shape[1]] + [1] * len(dims)
        bias_shape = [bias.shape[0], bias.shape[1]] + [1] * len(dims)
        weight = weight.view(*weight_shape)
        bias = bias.view(*bias_shape)

        # Normalize and scale
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        output = weight * x_normalized + bias

        # Update running statistics (for inference)
        if not self.training:
            with torch.no_grad():
                # Compute mean and var over batch and spatial dimensions
                batch_mean = x.mean(dim=[0] + dims, keepdim=False)  # (num_features,)
                batch_var = x.var(dim=[0] + dims, keepdim=False, unbiased=False)  # (num_features,)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        return output

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, speaker_embed_dim={self.speaker_embed_dim}, task_embed_dim={self.task_embed_dim if self.use_task_embedding else 0}, num_speakers={self.num_speakers}, num_tasks={self.num_tasks}, eps={self.eps}, momentum={self.momentum}'


def create_adain(
    num_features: int,
    speaker_embed_dim: int = 128,
    task_embed_dim: int = 64,
    num_speakers: int = 100,
    num_tasks: int = 3,
    eps: float = 1e-5,
    momentum: float = 0.1,
    use_task_embedding: bool = True
) -> TaskConditionedAdaIN:
    """
    Factory function to create TaskConditionedAdaIN.

    Args:
        num_features: Number of features/channels to normalize
        speaker_embed_dim: Dimension of speaker embeddings
        task_embed_dim: Dimension of task embeddings
        num_speakers: Number of speakers in training data
        num_tasks: Number of tasks in multi-task learning
        eps: Small constant for numerical stability
        momentum: Momentum for running mean/var
        use_task_embedding: Whether to use task embeddings

    Returns:
        TaskConditionedAdaIN module
    """
    return TaskConditionedAdaIN(
        num_features=num_features,
        speaker_embed_dim=speaker_embed_dim,
        task_embed_dim=task_embed_dim,
        num_speakers=num_speakers,
        num_tasks=num_tasks,
        eps=eps,
        momentum=momentum,
        use_task_embedding=use_task_embedding
    )


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    num_features = 64
    seq_len = 10
    num_speakers = 10
    num_tasks = 3

    # Create input tensor (B, C, T)
    x = torch.randn(batch_size, num_features, seq_len)

    # Create speaker and task IDs
    speaker_ids = torch.randint(0, num_speakers, (batch_size,))
    task_ids = torch.randint(0, num_tasks, (batch_size,))

    # Test with task embedding
    adain_with_task = TaskConditionedAdaIN(
        num_features=num_features,
        speaker_embed_dim=128,
        task_embed_dim=64,
        num_speakers=num_speakers,
        num_tasks=num_tasks,
        use_task_embedding=True
    )
    output_with_task = adain_with_task(x, speaker_ids, task_ids)
    print(f"TaskConditionedAdaIN with task embedding:")
    print(f"  Input shape: {x.shape}")
    print(f"  Speaker IDs shape: {speaker_ids.shape}")
    print(f"  Task IDs shape: {task_ids.shape}")
    print(f"  Output shape: {output_with_task.shape}")

    # Test without task embedding
    adain_without_task = TaskConditionedAdaIN(
        num_features=num_features,
        speaker_embed_dim=128,
        task_embed_dim=64,
        num_speakers=num_speakers,
        num_tasks=num_tasks,
        use_task_embedding=False
    )
    output_without_task = adain_without_task(x, speaker_ids)
    print(f"\nTaskConditionedAdaIN without task embedding:")
    print(f"  Input shape: {x.shape}")
    print(f"  Speaker IDs shape: {speaker_ids.shape}")
    print(f"  Output shape: {output_without_task.shape}")

    print("\nTask-conditioned AdaIN test passed!")