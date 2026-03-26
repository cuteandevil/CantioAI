"""
Multi-task data loader for CantioAI.
Implements unified data loading for multiple tasks with dynamic sampling strategies.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset, Subset
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import random
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    """
    Wrapper dataset that combines multiple task-specific datasets.
    Provides unified interface for multi-task learning.
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        task_ids: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize multi-task dataset.

        Args:
            datasets: Dictionary mapping task names to task-specific datasets
            task_ids: Optional mapping from task names to integer IDs
            transform: Optional transform to apply to samples
        """
        self.datasets = datasets
        self.task_names = list(datasets.keys())
        self.num_tasks = len(self.task_names)

        # Assign task IDs if not provided
        if task_ids is None:
            self.task_ids = {name: idx for idx, name in enumerate(self.task_names)}
        else:
            self.task_ids = task_ids

        # Validate task IDs
        if set(self.task_ids.keys()) != set(self.task_names):
            raise ValueError("Task IDs must be provided for all tasks")

        self.transform = transform

        # Create cumulative sizes for indexing
        self.dataset_sizes = [len(dataset) for dataset in self.datasets.values()]
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes).tolist()
        self.total_size = sum(self.dataset_sizes)

        logger.info(
            f"Initialized MultiTaskDataset:\n"
            f"  Tasks: {self.task_names}\n"
            f"  Task IDs: {self.task_ids}\n"
            f"  Dataset sizes: {dict(zip(self.task_names, self.dataset_sizes))}\n"
            f"  Total size: {self.total_size}"
        )

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item from multi-task dataset.

        Args:
            idx: Global index in the concatenated dataset

        Returns:
            Dictionary containing:
                - data: The actual data sample
                - task_name: Name of the task
                - task_id: Integer ID of the task
                - dataset_idx: Index within the specific task dataset
        """
        # Find which dataset and local index
        task_idx = 0
        while task_idx < self.num_tasks and idx >= self.cumulative_sizes[task_idx + 1]:
            task_idx += 1

        if task_idx >= self.num_tasks:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")

        # Get local index within the task dataset
        local_idx = idx - self.cumulative_sizes[task_idx]
        task_name = self.task_names[task_idx]
        task_id = self.task_ids[task_name]

        # Get item from task-specific dataset
        dataset = self.datasets[task_name]
        sample = dataset[local_idx]

        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)

        # Ensure sample is a dictionary and add task information
        if not isinstance(sample, dict):
            sample = {"data": sample}

        sample.update({
            "task_name": task_name,
            "task_id": task_id,
            "dataset_idx": local_idx
        })

        return sample


class DynamicTaskSampler(Sampler):
    """
    Dynamic sampler for multi-task datasets.
    Implements various sampling strategies for multi-task learning.
    """

    def __init__(
        self,
        data_source: MultiTaskDataset,
        sampling_strategy: str = "uniform",  # "uniform", "proportional", "inverse_proportional", "dynamic"
        task_weights: Optional[Dict[str, float]] = None,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None
    ):
        """
        Initialize dynamic task sampler.

        Args:
            data_source: MultiTaskDataset instance
            sampling_strategy: Strategy for sampling tasks
            task_weights: Optional manual weights for each task
            replacement: Whether to sample with replacement
            num_samples: Number of samples to draw (if None, equals len(data_source))
            generator: Random number generator
        """
        self.data_source = data_source
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.num_samples = num_samples if num_samples is not None else len(data_source)
        self.generator = generator

        # Task information
        self.task_names = data_source.task_names
        self.task_ids = data_source.task_ids
        self.dataset_sizes = data_source.dataset_sizes
        self.num_tasks = data_source.num_tasks

        # Initialize task weights
        if task_weights is None:
            if sampling_strategy == "uniform":
                self.task_weights = {name: 1.0 for name in self.task_names}
            elif sampling_strategy == "proportional":
                # Weight by dataset size
                total_size = sum(self.dataset_sizes)
                self.task_weights = {
                    name: size / total_size
                    for name, size in zip(self.task_names, self.dataset_sizes)
                }
            elif sampling_strategy == "inverse_proportional":
                # Weight inversely by dataset size (smaller datasets get higher weight)
                total_inv_size = sum(1.0 / size for size in self.dataset_sizes)
                self.task_weights = {
                    name: (1.0 / size) / total_inv_size
                    for name, size in zip(self.task_names, self.dataset_sizes)
                }
            else:
                # Uniform by default
                self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

        # Normalize weights
        weight_sum = sum(self.task_weights.values())
        self.task_weights = {name: weight / weight_sum for name, weight in self.task_weights.items()}

        # Create task-to-indices mapping for efficient sampling
        self.task_indices = {}
        for task_name in self.task_names:
            task_id = self.task_ids[task_name]
            start_idx = data_source.cumulative_sizes[task_id]
            end_idx = data_source.cumulative_sizes[task_id + 1]
            self.task_indices[task_name] = list(range(start_idx, end_idx))

        logger.info(
            f"Initialized DynamicTaskSampler:\n"
            f"  Sampling strategy: {sampling_strategy}\n"
            f"  Task weights: {self.task_weights}\n"
            f"  Replacement: {replacement}\n"
            f"  Num samples: {self.num_samples}"
        )

    def __iter__(self):
        """Generate indices for sampling."""
        if self.replacement:
            # Sampling with replacement
            indices = []
            task_names = list(self.task_weights.keys())
            task_weights = list(self.task_weights.values())

            for _ in range(self.num_samples):
                # Sample task based on weights
                task_name = random.choices(task_names, weights=task_weights, generator=self.generator)[0]
                # Sample index from selected task
                task_idx_list = self.task_indices[task_name]
                local_idx = random.choice(task_idx_list)
                indices.append(local_idx)
            return iter(indices)
        else:
            # Sampling without replacement - more complex
            # For simplicity, we'll create balanced batches
            indices = []
            samples_per_task = self.num_samples // self.num_tasks
            remainder = self.num_samples % self.num_tasks

            task_names = list(self.task_names)
            for i, task_name in enumerate(task_names):
                # Calculate samples for this task
                n_samples = samples_per_task + (1 if i < remainder else 0)
                # Sample without replacement from this task's indices
                task_idx_list = self.task_indices[task_name]
                if n_samples > len(task_idx_list):
                    # If we need more samples than available, sample with replacement
                    sampled_indices = random.choices(task_idx_list, k=n_samples)
                else:
                    sampled_indices = random.sample(task_idx_list, n_samples)
                indices.extend(sampled_indices)

            # Shuffle the final indices
            random.shuffle(indices)
            return iter(indices[:self.num_samples])

    def __len__(self) -> int:
        return self.num_samples


class CurriculumTaskSampler(Sampler):
    """
    Curriculum learning sampler for multi-task datasets.
    Starts with easier tasks and gradually introduces harder ones.
    """

    def __init__(
        self,
        data_source: MultiTaskDataset,
        task_difficulty: Dict[str, float],
        num_epochs: int = 100,
        warmup_epochs: int = 20,
        samples_per_epoch: Optional[int] = None,
        generator=None
    ):
        """
        Initialize curriculum task sampler.

        Args:
            data_source: MultiTaskDataset instance
            task_difficulty: Dictionary mapping task names to difficulty scores (0-1, higher = harder)
            num_epochs: Total number of training epochs
            warmup_epochs: Number of epochs to focus on easiest tasks
            samples_per_epoch: Number of samples per epoch (if None, equals len(data_source))
            generator: Random number generator
        """
        self.data_source = data_source
        self.task_difficulty = task_difficulty
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch is not None else len(data_source)
        self.generator = generator

        # Validate task difficulty
        for task_name in self.data_source.task_names:
            if task_name not in task_difficulty:
                raise ValueError(f"Difficulty score missing for task: {task_name}")
            if not 0 <= task_difficulty[task_name] <= 1:
                raise ValueError(f"Difficulty score for {task_name} must be in [0, 1]")

        self.epoch = 0
        self.task_names = data_source.task_names
        self.dataset_sizes = data_source.dataset_sizes
        self.num_tasks = data_source.num_tasks

        # Create task-to-indices mapping
        self.task_indices = {}
        for task_name in self.task_names:
            task_id = data_source.task_ids[task_name]
            start_idx = data_source.cumulative_sizes[task_id]
            end_idx = data_source.cumulative_sizes[task_id + 1]
            self.task_indices[task_name] = list(range(start_idx, end_idx))

        logger.info(
            f"Initialized CurriculumTaskSampler:\n"
            f"  Task difficulty: {task_difficulty}\n"
            f"  Num epochs: {num_epochs}\n"
            f"  Warmup epochs: {warmup_epochs}\n"
            f"  Samples per epoch: {self.samples_per_epoch}"
        )

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum calculation."""
        self.epoch = min(epoch, self.num_epochs - 1)

    def __iter__(self):
        """Generate indices for sampling based on curriculum."""
        # Calculate difficulty threshold based on epoch progress
        progress = self.epoch / max(self.num_epochs - 1, 1)
        if self.epoch < self.warmup_epochs:
            # During warmup, only use easiest tasks
            sorted_tasks = sorted(self.task_difficulty.items(), key=lambda x: x[1])
            easiest_tasks = [task for task, diff in sorted_tasks[:max(1, self.num_tasks // 2)]]
            threshold_difficulty = max(self.task_difficulty[task] for task in easiest_tasks)
        else:
            # Gradually increase difficulty threshold
            min_diff = min(self.task_difficulty.values())
            max_diff = max(self.task_difficulty.values())
            threshold_difficulty = min_diff + progress * (max_diff - min_diff)

        # Select tasks below difficulty threshold
        available_tasks = [
            task for task in self.task_names
            if self.task_difficulty[task] <= threshold_difficulty
        ]

        if not available_tasks:
            # Fallback to easiest task if none available
            easiest_task = min(self.task_difficulty.items(), key=lambda x: x[1])[0]
            available_tasks = [easiest_task]

        # Calculate sampling weights based on inverse difficulty (easier tasks sampled more)
        task_weights = {}
        for task in available_tasks:
            # Inverse difficulty: easier tasks get higher weight
            diff = self.task_difficulty[task]
            # Avoid division by zero
            weight = 1.0 / (diff + 0.1)
            task_weights[task] = weight

        # Normalize weights
        weight_sum = sum(task_weights.values())
        if weight_sum > 0:
            task_weights = {task: weight / weight_sum for task, weight in task_weights.items()}
        else:
            # Uniform weights if all difficulties are 0
            task_weights = {task: 1.0 / len(available_tasks) for task in available_tasks}

        # Generate indices
        indices = []
        for _ in range(self.samples_per_epoch):
            # Sample task based on weights
            task_name = random.choices(
                list(task_weights.keys()),
                weights=list(task_weights.values()),
                generator=self.generator
            )[0]
            # Sample index from selected task
            task_idx_list = self.task_indices[task_name]
            local_idx = random.choice(task_idx_list)
            indices.append(local_idx)

        return iter(indices)

    def __len__(self) -> int:
        return self.samples_per_epoch


def create_multitask_dataloader(
    datasets: Dict[str, Dataset],
    batch_size: int = 32,
    sampling_strategy: str = "uniform",
    task_weights: Optional[Dict[str, float]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
    timeout: int = 0,
    worker_init_fn: Optional[Callable] = None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    curriculum_info: Optional[Dict[str, Any]] = None
) -> DataLoader:
    """
    Create a multi-task data loader with various sampling strategies.

    Args:
        datasets: Dictionary mapping task names to task-specific datasets
        batch_size: Batch size for data loading
        sampling_strategy: Strategy for sampling tasks ("uniform", "proportional",
                         "inverse_proportional", "dynamic", "curriculum")
        task_weights: Optional manual weights for each task
        num_workers: Number of subprocesses for data loading
        pin_memory: Whether to copy tensors to CUDA pinned memory
        shuffle: Whether to shuffle data (ignored for custom samplers)
        drop_last: Whether to drop last incomplete batch
        timeout: Timeout for collecting batch from workers
        worker_init_fn: Function to initialize workers
        multiprocessing_context: Context for multiprocessing
        generator: Random number generator for sampling
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Whether to keep workers alive
        curriculum_info: Optional dict with curriculum learning info

    Returns:
        Configured DataLoader for multi-task learning
    """
    # Create multi-task dataset
    multitask_dataset = MultiTaskDataset(datasets=datasets)

    # Create sampler based on strategy
    if sampling_strategy == "curriculum":
        if curriculum_info is None:
            raise ValueError("curriculum_info required for curriculum sampling")
        sampler = CurriculumTaskSampler(
            data_source=multitask_dataset,
            task_difficulty=curriculum_info.get("task_difficulty", {}),
            num_epochs=curriculum_info.get("num_epochs", 100),
            warmup_epochs=curriculum_info.get("warmup_epochs", 20),
            samples_per_epoch=curriculum_info.get("samples_per_epoch"),
            generator=generator
        )
        # For curriculum learning, we typically don't shuffle in the DataLoader
        shuffle = False
    elif sampling_strategy in ["uniform", "proportional", "inverse_proportional", "dynamic"]:
        sampler = DynamicTaskSampler(
            data_source=multitask_dataset,
            sampling_strategy=sampling_strategy,
            task_weights=task_weights,
            generator=generator
        )
        # For custom samplers, we typically don't shuffle in the DataLoader
        shuffle = False
    else:
        raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

    # Create data loader
    dataloader = DataLoader(
        dataset=multitask_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    logger.info(
        f"Created MultiTaskDataLoader:\n"
        f"  Batch size: {batch_size}\n"
        f"  Sampling strategy: {sampling_strategy}\n"
        f"  Num workers: {num_workers}\n"
        f"  Pin memory: {pin_memory}\n"
        f"  Drop last: {drop_last}"
    )

    return dataloader


if __name__ == "__main__":
    # Simple test with dummy datasets
    from torch.utils.data import TensorDataset

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
        "task1": task1_dataset,
        "task2": task2_dataset,
        "task3": task3_dataset
    }

    # Test multi-task dataset
    multitask_dataset = MultiTaskDataset(datasets=datasets)
    print(f"MultiTaskDataset size: {len(multitask_dataset)}")

    # Test getting an item
    sample = multitask_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Task name: {sample['task_name']}")
    print(f"Task ID: {sample['task_id']}")

    # Test data loaders with different strategies
    for strategy in ["uniform", "proportional", "inverse_proportional"]:
        print(f"\nTesting {strategy} sampler:")
        dataloader = create_multitask_dataloader(
            datasets=datasets,
            batch_size=16,
            sampling_strategy=strategy
        )

        # Get one batch
        for batch in dataloader:
            print(f"  Batch task names: {batch['task_name']}")
            print(f"  Batch task IDs: {batch['task_id']}")
            print(f"  Data shape: {batch['data'].shape}")
            break

    # Test curriculum learning
    print(f"\nTesting curriculum sampler:")
    curriculum_info = {
        "task_difficulty": {
            "task1": 0.2,  # Easy
            "task2": 0.8,  # Hard
            "task3": 0.5   # Medium
        },
        "num_epochs": 10,
        "warmup_epochs": 3
    }

    dataloader = create_multitask_dataloader(
        datasets=datasets,
        batch_size=16,
        sampling_strategy="curriculum",
        curriculum_info=curriculum_info
    )

    # Set epoch and test
    dataloader.batch_sampler.sampler.set_epoch(0)  # Epoch 0 (warmup)
    for batch in dataloader:
        print(f"  Epoch 0 batch task names: {batch['task_name'].tolist()}")
        break

    dataloader.batch_sampler.sampler.set_epoch(5)  # Epoch 5 (mid training)
    for batch in dataloader:
        print(f"  Epoch 5 batch task names: {batch['task_name'].tolist()}")
        break

    print("\nMulti-task data loader test passed!")