"""
Multi-task training strategies.
Implements progressive multi-task training and curriculum learning strategies.
"""
import torch
import torch.nn as nn
import torch.optim as Optimizer
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import math
from ..dataloader import create_multitask_dataloader
from ..loss_design import (
    UncertaintyWeighting,
    GradientNormalization,
    TaskPrioritization,
    UncertaintyWeightedLoss
)
logger = logging.getLogger(__name__)

class MultiTaskTrainer:
    """
    Multi-task trainer that orchestrates training across multiple tasks.
    Implements various training strategies including progressive multi-task and curriculum learning.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: List[str],
        datasets: Dict[str, Any],
        strategy: str = "progressive_multi_task",  # Options: progressive_multi_task, curriculum_learning
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        weighting: str = "uncertainty",  # Options: uncertainty, gradient, priority
        grad_clip: float = 1.0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_checkpoint_interval: int = 10000,
        num_epochs: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize multi-task trainer.
        Args:
            model: Multi-task model to train
            tasks: List of task names
            datasets: Dictionary mapping task names to datasets
            strategy: Training strategy to use
            criterion: Loss function(s) for training
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            weighting: Loss weighting strategy to use
            grad_clip: Gradient clipping max norm
            log_interval: Interval for logging training info
            eval_interval: Interval for running evaluation
            save_checkpoint_interval: Interval for saving checkpoints
            num_epochs: Number of training epochs
            device: Device to run training on
        """
        self.model = model
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.strategy = strategy
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weighting = weighting
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_checkpoint_interval = save_checkpoint_interval
        self.num_epochs = num_epochs
        self.device = device

        # Validate strategy
        valid_strategies = ["progressive_multi_task", "curriculum_learning"]
        if strategy not in valid_strategies:
            raise ValueError(f"Unsupported strategy: {strategy}")

        # Validate weighting
        valid_weightings = ["uncertainty", "gradient", "priority"]
        if weighting not in valid_weightings:
            raise ValueError(f"Unsupported weighting: {weighting}")

        # Initialize components based on strategy
        if strategy == "progressive_multi_task":
            self.dataloaders = self._init_progressive_dataloaders(datasets)
            self.epoch_lengths = {task: 1 for task in tasks}  # Will be updated dynamically
        elif strategy == "curriculum_learning":
            self.dataloaders = self._init_curriculum_dataloaders(datasets)
            self.task_difficulty = {task: 0.5 for task in tasks}  # Placeholder - should come from datasets
            self.epoch_lengths = {task: num_epochs for task in tasks}  # Fixed for curriculum

        # Initialize loss weighting
        if weighting == "uncertainty":
            self.loss_weighting = UncertaintyWeightedLoss(num_tasks=self.num_tasks)
        elif weighting == "gradient":
            self.loss_weighting = GradientNormalization(num_tasks=self.num_tasks)
        elif weighting == "priority":
            self.loss_weighting = TaskPrioritization(num_tasks=self.num_tasks)

        # Initialize model to device
        if self.device is not None:
            self.model = self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device

        logger.info(
            f"Initialized MultiTaskTrainer:\n"
            f"  Model: {type(model).__name__}\n"
            f"  Tasks: {tasks}\n"
            f"  Strategy: {strategy}\n"
            f"  Weighting: {weighting}\n"
            f"  Num epochs: {num_epochs}"
        )

    def _init_progressive_dataloaders(
        self,
        datasets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Initialize dataloaders for progressive multi-task strategy.
        Args:
            datasets: Dictionary mapping task names to datasets
        Returns:
            Dictionary mapping task names to dataloaders
        """
        dataloaders = {}
        for task_name, dataset in datasets.items():
            dataloaders[task_name] = create_multitask_dataloader(
                datasets={task_name: dataset},
                batch_size=32,
                sampling_strategy="uniform"
            )
        return dataloaders

    def _init_curriculum_dataloaders(
        self,
        datasets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Initialize dataloaders for curriculum learning strategy.
        Args:
            datasets: Dictionary mapping task names to datasets
        Returns:
            Dictionary mapping task names to dataloaders
        """
        dataloaders = {}
        for task_name, dataset in datasets.items():
            curriculum_info = {
                "task_difficulty": {
                    task_name: 0.5
                },
                "num_epochs": 10,
                "warmup_epochs": 2
            }
            dataloaders[task_name] = create_multitask_dataloader(
                datasets={task_name: dataset},
                batch_size=16,
                sampling_strategy="curriculum",
                curriculum_info=curriculum_info
            )
        return dataloaders

    def train(self):
        """Run the training loop."""
        self.model.train()
        for epoch in range(self.num_epochs):
            # Training phase
            train_losses = self._train_epoch(epoch)

            # Evaluation phase
            if (epoch + 1) % self.eval_interval == 0:
                eval_results = self._eval_epoch(epoch)

            # Checkpoint saving
            if (epoch + 1) % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            # Logging
            if epoch % self.log_interval == 0:
                self._log_training_info(epoch)

        # Final evaluation
        final_results = self._eval_epoch(self.num_epochs - 1)
        return final_results

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch.
        Args:
            epoch: Current epoch index
        Returns:
            Dictionary of task losses
        """
        epoch_losses = {}
        for task_name, dataset in self.datasets.items():
            dataloader = self.dataloaders[task_name]
            # Get one batch
            try:
                batch = next(iter(dataloader))
                inputs = batch["data"]
                targets = batch["targets"] if "targets" in batch else None

                # Zero gradients
                if self.optimizer is not None:
                    self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                if self.criterion is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    raise ValueError("Criterion must be provided")

                # Apply loss weighting
                if self.loss_weighting is not None:
                    weighted_loss = self.loss_weighting(loss)
                else:
                    raise ValueError("Loss weighting must be provided")

                # Backward pass
                if self.optimizer is not None:
                    self.optimizer.step()

                # Collect loss
                epoch_losses[task_name] = weighted_loss.item()
            except StopIteration:
                # End of dataset
                pass
        return epoch_losses

    def _eval_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one evaluation epoch.
        Args:
            epoch: Current epoch index
        Returns:
            Dictionary of task metrics
        """
        epoch_results = {}
        for task_name, dataset in self.datasets.items():
            dataloader = self.dataloaders[task_name]
            # Get one batch
            try:
                batch = next(iter(dataloader))
                inputs = batch["data"]
                targets = batch["targets"] if "targets" in batch else None

                # Forward pass
                outputs = self.model(inputs)

                # Compute metrics
                if task_name == "singing":
                    metrics = self._compute_singing_metrics(outputs)
                elif task_name == "speech":
                    metrics = self._compute_speech_metrics(outputs)
                elif task_name == "noise_robustness":
                    metrics = self._compute_noise_robustness_metrics(outputs)
                else:
                    raise ValueError(f"No metrics function for task: {task_name}")

                epoch_results[task_name] = metrics
            except StopIteration:
                # End of dataset
                pass
        return epoch_results

    def _save_checkpoint(self, epoch: int):
        """
        Save model checkpoint.
        Args:
            epoch: Current epoch index
        Returns:
            None
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None
        }
        # In practice, would save to filesystem
        pass

    def _log_training_info(self, epoch: int):
        """
        Log training information.
        Args:
            epoch: Current epoch index
        Returns:
            None
        """
        # In practice, would log to tracking system
        pass

    def _compute_singing_metrics(self, outputs: torch.Tensor) -> Dict[str, float]:
        """Compute singing-specific metrics."""
        # Placeholder - would compute F0, SP, AP related metrics
        return {
            "f0_l1_error": 0.1,
            "sp_l1_error": 0.2,
            "ap_l1_error": 0.3
        }

    def _compute_speech_metrics(self, outputs: torch.Tensor) -> Dict[str, float]:
        """Compute speech-specific metrics."""
        # Placeholder - would compute F0, related metrics
        return {
            "f0_l1_error": 0.1,
            "f0_l1_accuracy": 0.9
        }

    def _compute_noise_robustness_metrics(self, outputs: torch.Tensor) -> Dict[str, float]:
        """Compute noise robustness specific metrics."""
        # Placeholder - would compute SNR, related metrics
        return {
            "snr": 0.5,
            "signal_power": 0.6
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.
        Args:
            batch: Dictionary containing:
                - data: Input tensor
                - targets: Optional target tensor
        Returns:
            Dictionary of task losses
        """
        # Zero gradients
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(batch["data"])

        # Compute loss
        if self.criterion is not None:
            loss = self.criterion(outputs, batch["targets"] if "targets" in batch else None)
        else:
            raise ValueError("Criterion must be provided")

        # Apply loss weighting
        if self.loss_weighting is not None:
            weighted_loss = self.loss_weighting(loss)
        else:
            raise ValueError("Loss weighting must be provided")

        # Backward pass
        if self.optimizer is not None:
            self.optimizer.step()

        # Collect loss
        return {task_name: weighted_loss.item() for task_name, batch in batch.items()}

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single evaluation step.
        Args:
            batch: Dictionary containing:
                - data: Input tensor
                - targets: Optional target tensor
        Returns:
            Dictionary of task metrics
        """
        # Forward pass
        outputs = self.model(batch["data"])

        # Compute metrics
        if self.scheduler is not None:
            self.scheduler.step()

        # Compute metrics
        if batch["task_name"] == "singing":
            metrics = self._compute_singing_metrics(outputs)
        elif batch["task_name"] == "speech":
            metrics = self._compute_speech_metrics(outputs)
        elif batch["task_name"] == "noise_robustness":
            metrics = self._compute_noise_robustness_metrics(outputs)
        else:
            raise ValueError(f"No metrics function for task: {batch['task_name']}")

        return {task_name: metrics for task_name, batch in batch.items()}

    def extra_repr(self) -> str:
        return f'model={type(self.model).__name__}, tasks={self.tasks}, strategy={self.strategy}, weighting={self.weighting}'


def create_progressive_multi_task_trainer(
    model: nn.Module,
    tasks: List[str],
    datasets: Dict[str, Any],
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    weighting: str = "uncertainty",
    grad_clip: float = 1.0,
    log_interval: int = 100,
    eval_interval: int = 1000,
    save_checkpoint_interval: int = 10000,
    num_epochs: int = 100,
    device: Optional[torch.device] = None
) -> MultiTaskTrainer:
    """
    Create a progressive multi-task trainer.
    Args:
        model: Multi-task model to train
        tasks: List of task names
        datasets: Dictionary mapping task names to datasets
        criterion: Loss function(s) for training
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        weighting: Loss weighting strategy to use
        grad_clip: Gradient clipping max norm
        log_interval: Interval for logging training info
        eval_interval: Interval for running evaluation
        save_checkpoint_interval: Interval for saving checkpoints
        num_epochs: Number of training epochs
        device: Device to run training on
    Returns:
        MultiTaskTrainer configured for progressive multi-task
    """
    return MultiTaskTrainer(
        model=model,
        tasks=tasks,
        datasets=datasets,
        strategy="progressive_multi_task",
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        weighting=weighting,
        grad_clip=grad_clip,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_checkpoint_interval=save_checkpoint_interval,
        num_epochs=num_epochs,
        device=device
    )


def create_curriculum_learning_trainer(
    model: nn.Module,
    tasks: List[str],
    datasets: Dict[str, Any],
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    weighting: str = "uncertainty",
    grad_clip: float = 1.0,
    log_interval: int = 100,
    eval_interval: int = 1000,
    save_checkpoint_interval: int = 10000,
    num_epochs: int = 100,
    device: Optional[torch.device] = None
) -> MultiTaskTrainer:
    """
    Create a curriculum learning trainer.
    Args:
        model: Multi-task model to train
        tasks: List of task names
        datasets: Dictionary mapping task names to datasets
        criterion: Loss function(s) for training
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        weighting: Loss weighting strategy to use
        grad_clip: Gradient clipping max norm
        log_interval: Interval for logging training info
        eval_interval: Interval for running evaluation
        save_checkpoint_interval: Interval for saving checkpoints
        num_epochs: Number of training epochs
        device: Device to run training on
    Returns:
        MultiTaskTrainer configured for curriculum learning
    """
    return MultiTaskTrainer(
        model=model,
        tasks=tasks,
        datasets=datasets,
        strategy="curriculum_learning",
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        weighting=weighting,
        grad_clip=grad_clip,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_checkpoint_interval=save_checkpoint_interval,
        num_epochs=num_epochs,
        device=device
    )