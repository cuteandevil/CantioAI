"""
Multitask DataLoader for CantioAI
Creates train, validation, and test data loaders for multi-task learning
"""
from torch.utils.data import DataLoader

from .dataset import CantioAIDataset
from typing import Tuple, Dict, Any, Optional, List, Union

import logging

logger = logging.getLogger(__name__)


def create_multitask_dataloader(
    datasets: Dict[str, CantioAIDataset],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    Args:
        datasets: Dictionary mapping task names to CantioAIDataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker threads for data loading
        pin_memory: Whether to pin memory for GPU transfer
        **dataset_kwargs: Additional arguments passed to CantioAIDataset
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = CantioAIDataset(
        split="train",
        **dataset_kwargs
    )

    val_dataset = CantioAIDataset(
        split="val",
        **dataset_kwargs
    )

    test_dataset = CantioAIDataset(
        split="test",
        **dataset_kwargs
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader