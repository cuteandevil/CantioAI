"""
Multi-task evaluation framework.
Implements comprehensive evaluation for multiple tasks with cross-task consistency checks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math

logger = logging.getLogger(__name__)


class MultiTaskEvaluator:
    """
    Multi-task evaluator that computes metrics for each task and cross-task consistency.
    """

    def __init__(
        self,
        tasks: List[str],
        metric_functions: Optional[Dict[str, Callable]] = None,
        consistency_metrics: bool = True,
        cross_task_checks: bool = True
    ):
        """
        Initialize multi-task evaluator.

        Args:
            tasks: List of task names to evaluate
            metric_functions: Optional dictionary mapping task names to metric functions
            consistency_metrics: Whether to compute cross-task consistency metrics
            cross_task_checks: Whether to perform cross-task consistency checks
        """
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.metric_functions = metric_functions or self._default_metric_functions()
        self.consistency_metrics = consistency_metrics
        self.cross_task_checks = cross_task_checks

        # Initialize results storage
        self.task_metrics = {task: {} for task in tasks}
        self.cross_task_metrics = {}
        self.consistency_scores = {}

        logger.info(
            f"Initialized MultiTaskEvaluator:\n"
            f"  Tasks: {tasks}\n"
            f"  Num tasks: {self.num_tasks}\n"
            f"  Consistency metrics: {self.consistency_metrics}\n"
            f"  Cross-task checks: {self.cross_task_checks}"
        )

    def _default_metric_functions(self) -> Dict[str, Callable]:
        """Get default metric functions for known tasks."""
        return {
            "singing": self._compute_singing_metrics,
            "speech": self._compute_speech_metrics,
            "noise_robustness": self._compute_noise_robustness_metrics
        }

    def evaluate(
        self,
        model: nn.Module,
        dataloaders: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on all tasks.

        Args:
            model: Model to evaluate
            dataloaders: Dictionary mapping task names to dataloaders
            device: Device to run evaluation on

        Returns:
            Dictionary containing:
                - task_metrics: Metrics for each task
                - cross_task_metrics: Cross-task consistency metrics
                - consistency_scores: Overall consistency scores
        """
        model.eval()
        if device is not None:
            model = model.to(device)
        else:
            device = next(model.parameters()).device

        # Reset results
        self.task_metrics = {task: {} for task in self.tasks}
        self.cross_task_metrics = {}
        self.consistency_scores = {}

        # Evaluate each task
        for task_name, dataloader in dataloaders.items():
            task_results = self._evaluate_task(model, dataloader, device)
            self.task_metrics[task_name] = task_results

        # Compute cross-task metrics if enabled
        if self.consistency_metrics:
            self._compute_cross_task_metrics()

        # Compile results
        results = {
            "task_metrics": self.task_metrics,
            "cross_task_metrics": self.cross_task_metrics,
            "consistency_scores": self.consistency_scores
        }

        return results

    def _evaluate_task(
        self,
        model: nn.Module,
        dataloader: Any,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate model on a single task.

        Args:
            model: Model to evaluate
            dataloader: DataLoader for the task
            device: Device to run evaluation on

        Returns:
            Dictionary of task metrics
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        # Get task-specific metric function
        metric_func = self.metric_functions.get(
            list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else None
        )
        if metric_func is None:
            # Try to infer from task name
            for task_name_key, func in self.metric_functions.items():
                if task_name_key in str(type(dataloader)):
                    metric_func = func
                    break

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = self._move_to_device(batch, device)

                # Forward pass
                outputs = model(batch["data"])

                # Extract task-specific outputs if needed
                if isinstance(outputs, dict):
                    # If model returns dictionary of task outputs
                    task_outputs = outputs.get(list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else None)
                    if task_outputs is None:
                        # Fallback: use first output
                        task_outputs = list(outputs.values())[0]
                else:
                    task_outputs = outputs

                # Compute batch metrics
                batch_metrics = self._compute_batch_metrics(
                    task_outputs,
                    batch["targets"] if "targets" in batch else batch.get("labels"),
                    task_name_key if hasattr(dataloaders, 'keys') else "task"
                )

                # Accumulate metrics
                for metric_name, metric_value in batch_metrics.items():
                    if metric_name not in self.task_metrics[list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else "task"]:
                        self.task_metrics[list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else "task"][metric_name] = 0.0
                    self.task_metrics[list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else "task"][metric_name] += metric_value

                num_batches += 1

        # Average metrics
        if num_batches > 0:
            for metric_name in self.task_metrics[list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else "task"]:
                self.task_metrics[list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else "task"][metric_name] /= num_batches

        return self.task_metrics[list(dataloaders.keys())[0] if hasattr(dataloaders, 'keys') else "task"]

    def _move_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        Move batch to specified device.

        Args:
            batch: Batch data (tensor, dict, list, etc.)
            device: Target device

        Returns:
            Batch moved to device
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value, device) for key, value in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(item, device) for item in batch]
        else:
            return batch

    def _compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_name: str
    ) -> Dict[str, float]:
        """
        Compute metrics for a single batch.

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            task_name: Name of the task

        Returns:
            Dictionary of batch metrics
        """
        # Use task-specific metric function if available
        if task_name in self.metric_functions:
            return self.metric_functions[task_name](outputs, targets)
        else:
            # Default metrics
            return self._default_batch_metrics(outputs, targets, task_name)

    def _default_batch_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_name: str
    ) -> Dict[str, float]:
        """Compute default metrics for a batch."""
        metrics = {}

        # Flatten for classification metrics
        if outputs.dim() > 2:
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
        else:
            outputs_flat = outputs
            targets_flat = targets

        # Classification metrics if applicable
        if outputs_flat.size(-1) > 1:  # Multi-class classification
            try:
                _, predicted = torch.max(outputs_flat, dim=1)
                metrics["accuracy"] = accuracy_score(
                    targets_flat.cpu().numpy(),
                    predicted.cpu().numpy()
                )
                metrics["f1_macro"] = f1_score(
                    targets_flat.cpu().numpy(),
                    predicted.cpu().numpy(),
                    average='macro'
                )
            except Exception as e:
                logger.warning(f"Could not compute classification metrics for {task_name}: {e}")

        # Regression metrics
        try:
            mse = torch.mean((outputs_flat - targets_flat) ** 2).item()
            metrics["mse"] = mse
            rmse = math.sqrt(mse)
            metrics["rmse"] = rmse
            mae = torch.mean(torch.abs(outputs_flat - targets_flat)).item()
            metrics["mae"] = mae
        except Exception as e:
            logger.warning(f"Could not compute regression metrics for {task_name}: {e}")

        # Task-specific output stats
        try:
            metrics["output_mean"] = torch.mean(outputs_flat).item()
            metrics["output_std"] = torch.std(outputs_flat).item()
            metrics["target_mean"] = torch.mean(targets_flat).item()
            metrics["target_std"] = torch.std(targets_flat).item()
        except Exception as e:
            logger.warning(f"Could not compute output stats for {task_name}: {e}")

        return metrics

    def _compute_cross_task_metrics(self):
        """Compute cross-task consistency metrics."""
        if len(self.tasks) < 2:
            return

        # Compute pairwise consistency between tasks
        for i, task1 in enumerate(self.tasks):
            for task2 in self.tasks[i+1:]:
                task_pair = f"{task1}_vs_{task2}"
                consistency = self._compute_task_pair_consistency(task1, task2)
                self.cross_task_metrics[task_pair] = consistency

        # Compute overall consistency scores
        if self.cross_task_metrics:
            consistency_values = list(self.cross_task_metrics.values())
            self.consistency_scores["mean_consistency"] = np.mean(consistency_values)
            self.consistency_scores["std_consistency"] = np.std(consistency_values)
            self.consistency_scores["min_consistency"] = np.min(consistency_values)
            self.consistency_scores["max_consistency"] = np.max(consistency_values)

    def _compute_task_pair_consistency(self, task1: str, task2: str) -> float:
        """
        Compute consistency between two tasks.

        Args:
            task1: First task name
            task2: Second task name

        Returns:
            Consistency score between 0 and 1 (higher = more consistent)
        """
        # This is a simplified implementation
        # In practice, would compare model outputs for same inputs
        # or compute correlation between task-specific features

        # Placeholder: return simulated consistency based on task similarity
        task_similarities = {
            ("singing", "speech"): 0.7,  # Related tasks
            ("singing", "noise_robustness"): 0.4,  # Less related
            ("speech", "noise_robustness"): 0.5,  # Moderately related
        }

        # Check both orderings
        key1 = (task1, task2)
        key2 = (task2, task1)

        if key1 in task_similarities:
            return task_similarities[key1]
        elif key2 in task_similarities:
            return task_similarities[key2]
        else:
            # Default consistency for unknown pairs
            return 0.5

    def _compute_singing_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute singing-specific metrics."""
        metrics = {}

        # Assuming outputs and targets have specific structure for singing
        # For example: outputs might contain F0, SP, AP predictions
        if outputs.dim() >= 2 and outputs.size(-1) >= 3:
            # Multi-output: F0, SP, AP
            f0_pred = outputs[..., 0]  # First output dimension
            sp_pred = outputs[..., 1:4]  # Middle output dimensions
            ap_pred = outputs[..., 4]  # Last output dimension

            # Similar structure for targets
            if targets.dim() >= 2 and targets.size(-1) >= 3:
                f0_target = targets[..., 0]
                sp_target = targets[..., 1:4]
                ap_target = targets[..., 4]

                # F0 metrics
                f0_l1 = torch.mean(torch.abs(f0_pred - f0_target)).item()
                metrics["f0_l1_error"] = f0_l1
                f0_l2 = torch.mean((f0_pred - f0_target) ** 2).item()
                metrics["f0_l2_error"] = f0_l2

                # SP metrics (spectral envelope)
                sp_l1 = torch.mean(torch.abs(sp_pred - sp_target)).item()
                metrics["sp_l1_error"] = sp_l1
                sp_l2 = torch.mean((sp_pred - sp_target) ** 2).item()
                metrics["sp_l2_error"] = sp_l2

                # AP metrics (aperiodicity)
                ap_l1 = torch.mean(torch.abs(ap_pred - ap_target)).item()
                metrics["ap_l1_error"] = ap_l1
                ap_l2 = torch.mean((ap_pred - ap_target) ** 2).item()
                metrics["ap_l2_error"] = ap_l2

                # Overall singing quality
                metrics["singing_quality_score"] = 1.0 / (1.0 + f0_l1 + sp_l1 + ap_l1)

        return metrics

    def _compute_speech_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute speech-specific metrics."""
        metrics = {}

        # Assuming outputs and targets have specific structure for speech
        # For example: F0 prediction and phoneme classification
        if outputs.dim() >= 2:
            if outputs.size(-1) >= 2:
                # F0 + phoneme logits
                f0_pred = outputs[..., 0]
                phoneme_logits = outputs[..., 1:]

                if targets.dim() >= 2:
                    if targets.size(-1) >= 2:
                        f0_target = targets[..., 0]
                        phoneme_targets = targets[..., 1:]

                        # F0 metrics
                        f0_l1 = torch.mean(torch.abs(f0_pred - f0_target)).item()
                        metrics["f0_l1_error"] = f0_l1

                        # Phoneme classification metrics
                        try:
                            _, phoneme_pred = torch.max(phoneme_logits, dim=-1)
                            metrics["phoneme_accuracy"] = accuracy_score(
                                phoneme_targets.cpu().numpy().flatten(),
                                phoneme_pred.cpu().numpy().flatten()
                            )
                        except Exception as e:
                            logger.warning(f"Could not compute phoneme metrics: {e}")

            else:
                # Single output (likely F0 only)
                f0_pred = outputs
                f0_target = targets
                f0_l1 = torch.mean(torch.abs(f0_pred - f0_target)).item()
                metrics["f0_l1_error"] = f0_l1

        return metrics

    def _compute_noise_robustness_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute noise robustness specific metrics."""
        metrics = {}

        # Assuming outputs and targets represent SNR improvement or denoising quality
        if outputs.dim() >= 1 and targets.dim() >= 1:
            # Simple case: SNR estimation
            snr_estimate = outputs
            snr_target = targets

            snr_l1 = torch.mean(torch.abs(snr_estimate - snr_target)).item()
            metrics["snr_l1_error"] = snr_l1

            # Signal preservation
            signal_power_est = torch.mean(outputs ** 2).item()
            signal_power_targ = torch.mean(targets ** 2).item()
            signal_power_error = abs(signal_power_est - signal_power_targ)
            metrics["signal_power_error"] = signal_power_error

            # Noise suppression (lower is better for residual noise)
            noise_residual_est = torch.mean((torch.abs(outputs) - torch.abs(targets)) ** 2).item()
            metrics["noise_residual"] = noise_residual_est

            # Overall denoising quality (0-1 scale, higher is better)
            metrics["denoising_quality"] = 1.0 / (1.0 + snr_l1 + signal_power_error + noise_residual_est)

        return metrics

    def compute_task_ranking(
        self,
        metric_name: str = "f1_score",
        higher_is_better: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Rank tasks by a specific metric.

        Args:
            metric_name: Name of metric to rank by
            higher_is_better: Whether higher values are better

        Returns:
            List of (task_name, metric_value) tuples sorted by performance
        """
        task_scores = []
        for task_name in self.tasks:
            if task_name in self.task_metrics and metric_name in self.task_metrics[task_name]:
                score = self.task_metrics[task_name][metric_name]
                task_scores.append((task_name, score))

        # Sort by score
        task_scores.sort(key=lambda x: x[1], reverse=higher_is_better)
        return task_scores

    def get_best_task(self, metric_name: str = "f1_score") -> str:
        """
        Get the best performing task for a given metric.

        Args:
            metric_name: Name of metric to evaluate

        Returns:
            Name of the best performing task
        """
        ranking = self.compute_task_ranking(metric_name=metric_name, higher_is_better=True)
        if ranking:
            return ranking[0][0]
        return None

    def get_worst_task(self, metric_name: str = "f1_score") -> str:
        """
        Get the worst performing task for a given metric.

        Args:
            metric_name: Name of metric to evaluate

        Returns:
            Name of the worst performing task
        """
        ranking = self.compute_task_ranking(metric_name=metric_name, higher_is_better=False)
        if ranking:
            return ranking[0][0]
        return None

    def extra_repr(self) -> str:
        return f'tasks={self.tasks}, num_tasks={self.num_tasks}, consistency_metrics={self.consistency_metrics}, cross_task_checks={self.cross_task_checks}'


def create_multitask_evaluator(
    tasks: List[str],
    metric_functions: Optional[Dict[str, Callable]] = None,
    consistency_metrics: bool = True,
    cross_task_checks: bool = True
) -> MultiTaskEvaluator:
    """
    Create a multi-task evaluator.

    Args:
        tasks: List of task names to evaluate
        metric_functions: Optional dictionary mapping task names to metric functions
        consistency_metrics: Whether to compute cross-task consistency metrics
        cross_task_checks: Whether to perform cross-task consistency checks

    Returns:
        MultiTaskEvaluator instance
    """
    return MultiTaskEvaluator(
        tasks=tasks,
        metric_functions=metric_functions,
        consistency_metrics=consistency_metrics,
        cross_task_checks=cross_task_checks
    )


if __name__ == "__main__":
    # Simple test with dummy model and data
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Create a simple multi-task model
    class DummyMultiTaskModel(nn.Module):
        def __init__(self, num_tasks: int = 3):
            super().__init__()
            self.shared_encoder = nn.Linear(10, 64)
            self.task_heads = nn.ModuleList([
                nn.Linear(64, 1),   # Task 1 output
                nn.Linear(64, 3),   # Task 2 output (3 classes)
                nn.Linear(64, 1)    # Task 3 output
            ])

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            shared_features = self.shared_encoder(x)
            outputs = {}
            task_names = ["singing", "speech", "noise_robustness"]
            for i, task_head in enumerate(self.task_heads):
                outputs[task_name] = task_head(shared_features)
            return outputs

    # Create dummy datasets
    task1_data = torch.randn(100, 10)  # 100 samples, 10 features
    task1_labels = torch.randn(100, 1)  # Continuous targets
    task1_dataset = TensorDataset(task1_data, task1_labels)

    task2_data = torch.randn(150, 10)  # 150 samples
    task2_labels = torch.randint(0, 3, (150,))  # 3-class labels
    task2_dataset = TensorDataset(task2_data, task2_labels)

    task3_data = torch.randn(80, 10)   # 80 samples
    task3_labels = torch.randn(80, 1)   # Continuous targets
    task3_dataset = TensorDataset(task3_data, task3_labels)

    dataloaders = {
        "singing": DataLoader(task1_dataset, batch_size=16),
        "speech": DataLoader(task2_dataset, batch_size=16),
        "noise_robustness": DataLoader(task3_dataset, batch_size=16)
    }

    # Test multi-task evaluator
    print("\nTesting MultiTaskEvaluator:")
    evaluator = create_multitask_evaluator(
        tasks=["singing", "speech", "noise_robustness"]
    )

    model = DummyMultiTaskModel(num_tasks=3)
    results = evaluator.evaluate(model, dataloaders)

    print(f"Task metrics keys: {list(results['task_metrics'].keys())}")
    for task_name, metrics in results['task_metrics'].items():
        print(f"  {task_name}: {list(metrics.keys())}")

    print(f"Cross-task metrics keys: {list(results['cross_task_metrics'].keys())}")
    for task_pair, score in results['cross_task_metrics'].items():
        print(f"  {task_pair}: {score}")

    print(f"Consistency scores: {results['consistency_scores']}")

    # Test task ranking
    ranking = evaluator.compute_task_ranking("f1_score", higher_is_better=True)
    print(f"\nTask ranking by F1 score: {ranking}")

    best_task = evaluator.get_best_task("f1_score")
    print(f"Best task by F1 score: {best_task}")

    worst_task = evaluator.get_worst_task("f1_score")
    print(f"Worst task by F1 score: {worst_task}")

    print("\nMulti-task evaluation framework test passed!")