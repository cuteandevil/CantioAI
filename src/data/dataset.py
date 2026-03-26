"""
PyTorch Dataset for CantioAI features.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class CantioAIDataset(Dataset):
    """
    Dataset for loading preprocessed features (.npz files) for voice conversion.

    Each sample contains:
        - phoneme_features: (T, D_ph) - phonetic/linguistic features
        - f0: (T, 1) - normalized fundamental frequency
        - spk_id: scalar - speaker identifier
        - target_sp: (T, D_sp) - target spectral envelope (mel-cepstral)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_in_memory: bool = False,
        phoneme_feature_dim: int = 32,
        spectral_envelope_dim: int = 60,
        augment: bool = False
    ):
        """
        Initialize CantioAIDataset.

        Args:
            data_dir: Directory containing .npz feature files
            split: Dataset split ("train", "val", "test")
            transform: Transform to apply to input features
            target_transform: Transform to apply to target features
            cache_in_memory: Whether to load all data into memory
            phoneme_feature_dim: Dimension of phoneme features
            spectral_envelope_dim: Dimension of spectral envelope (target)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cache_in_memory = cache_in_memory
        self.phoneme_feature_dim = phoneme_feature_dim
        self.spectral_envelope_dim = spectral_envelope_dim
        self.augment = augment

        # Find all feature files
        self.feature_files = sorted(list(self.data_dir.glob(f"*_features.npz")))
        if not self.feature_files:
            raise ValueError(
                f"No feature files found in {self.data_dir} "
                f"matching pattern '*_features.npz'"
            )

        logger.info(f"Found {len(self.feature_files)} feature files for {split} split")

        # Optional: create train/val/test splits
        if split in ["train", "val", "test"]:
            self._create_splits()
        # Otherwise, use all files (for full dataset)

        # Cache data if requested
        if self.cache_in_memory:
            self._cache_data()
        else:
            self.cached_data = None

    def _create_splits(self) -> None:
        """Create train/val/test splits from feature files."""
        n_total = len(self.feature_files)
        indices = list(range(n_total))
        random.shuffle(indices)

        # Standard splits: 80% train, 10% val, 10% test
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)

        if self.split == "train":
            self.feature_files = [self.feature_files[i] for i in indices[:n_train]]
        elif self.split == "val":
            self.feature_files = [self.feature_files[i] for i in indices[n_train:n_train + n_val]]
        else:  # test
            self.feature_files = [self.feature_files[i] for i in indices[n_train + n_val:]]

        logger.info(
            f"Split {self.split}: {len(self.feature_files)} files "
            f"({n_total} total)"
        )

    def _cache_data(self) -> None:
        """Load all data into memory for faster access."""
        logger.info(f"Caching {len(self.feature_files)} feature files into memory...")
        self.cached_data = []

        for idx, file_path in enumerate(self.feature_files):
            if idx % 100 == 0:
                logger.info(f"Cached {idx}/{len(self.feature_files)} files")

            data = np.load(file_path)
            # Convert to torch tensors
            sample = {
                "phoneme_features": torch.from_numpy(data["phoneme_features"]).float(),
                "f0": torch.from_numpy(data["f0"]).float().unsqueeze(-1),
                "spk_id": torch.from_numpy(data["spk_id"]).long(),
                "target_sp": torch.from_numpy(data["target_sp"]).float(),
            }
            self.cached_data.append(sample)

        logger.info(f"Finished caching {len(self.cached_data)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - phoneme_features: (T, D_ph)
                - f0: (T, 1)
                - spk_id: scalar
                - target_sp: (T, D_sp)
        """
        if self.cache_in_memory and self.cached_data is not None:
            sample = self.cached_data[idx].copy()
        else:
            # Load from disk
            file_path = self.feature_files[idx]
            data = np.load(file_path)
            sample = {
                "phoneme_features": torch.from_numpy(data["phoneme_features"]).float(),
                "f0": torch.from_numpy(data["f0"]).float().unsqueeze(-1),
                "spk_id": torch.from_numpy(data["spk_id"]).long(),
                "target_sp": torch.from_numpy(data["target_sp"]).float(),
            }

        # Apply transforms
        if self.transform:
            sample["phoneme_features"] = self.transform(sample["phoneme_features"])
            sample["f0"] = self.transform(sample["f0"])

        if self.target_transform:
            sample["target_sp"] = self.target_transform(sample["target_sp"])

        # Apply data augmentation (if enabled and in training mode)
        if self.augment and self.split == "train":
            sample = self._augment_sample(sample)

        return sample

    def _augment_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to a sample.

        Args:
            sample: Input sample dictionary

        Returns:
            Augmented sample dictionary
        """
        # Example augmentations:
        # 1. Time stretching (simplified)
        # 2. Adding noise to features
        # 3. Speaker mixup (advanced)

        # For now, just add small Gaussian noise to phoneme features
        if random.random() < 0.5:  # 50% chance
            noise = torch.randn_like(sample["phoneme_features"]) * 0.01
            sample["phoneme_features"] = sample["phoneme_features"] + noise

        # Add small noise to F0 (avoid zeros)
        if random.random() < 0.3:
            noise = torch.randn_like(sample["f0"]) * 0.005
            sample["f0"] = torch.clamp(sample["f0"] + noise, min=0.0)

        return sample

    def get_speaker_ids(self) -> List[int]:
        """
        Get list of unique speaker IDs in the dataset.

        Returns:
            List of unique speaker IDs
        """
        speaker_ids = set()
        for file_path in self.feature_files:
            data = np.load(file_path)
            speaker_ids.add(int(data["spk_id"]))
        return sorted(list(speaker_ids))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        if self.cache_in_memory and self.cached_data is not None:
            data_source = self.cached_data
        else:
            # Compute on-the-fly (slower but doesn't require caching)
            data_source = []
            for file_path in self.feature_files:
                data = np.load(file_path)
                data_source.append({
                    "phoneme_features": torch.from_numpy(data["phoneme_features"]).float(),
                    "f0": torch.from_numpy(data["f0"]).float().unsqueeze(-1),
                    "spk_id": torch.from_numpy(data["spk_id"]).long(),
                    "target_sp": torch.from_numpy(data["target_sp"]).float(),
                })

        # Compute statistics
        total_frames = 0
        phoneme_means = []
        phonome_stds = []
        f0_means = []
        f0_stds = []
        sp_means = []
        sp_stds = []

        for sample in data_source:
            total_frames += sample["phoneme_features"].shape[0]
            phoneme_means.append(sample["phoneme_features"].mean(dim=0))
            phonome_stds.append(sample["phoneme_features"].std(dim=0))
            f0_means.append(sample["f0"].mean())
            f0_stds.append(sample["f0"].std())
            sp_means.append(sample["target_sp"].mean(dim=0))
            sp_stds.append(sample["target_sp"].std(dim=0))

        stats = {
            "num_samples": len(data_source),
            "total_frames": total_frames,
            "avg_frames_per_sample": total_frames / len(data_source) if len(data_source) > 0 else 0,
            "phoneme_feature_mean": torch.stack(phoneme_means).mean(dim=0) if phoneme_means else torch.zeros(self.phoneme_feature_dim),
            "phoneme_feature_std": torch.stack(phonome_stds).mean(dim=0) if phonome_stds else torch.ones(self.phoneme_feature_dim),
            "f0_mean": torch.stack(f0_means).mean() if f0_means else torch.tensor(0.0),
            "f0_std": torch.stack(f0_stds).mean() if f0_stds else torch.tensor(1.0),
            "target_sp_mean": torch.stack(sp_means).mean(dim=0) if sp_means else torch.zeros(self.spectral_envelope_dim),
            "target_sp_std": torch.stack(sp_stds).mean(dim=0) if sp_stds else torch.ones(self.spectral_envelope_dim),
            "num_speakers": len(self.get_speaker_ids()),
        }

        return stats


def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir: Directory containing preprocessed features
        batch_size: Batch size for data loaders
        num_workers: Number of worker threads for data loading
        pin_memory: Whether to pin memory for GPU transfer
        **dataset_kwargs: Additional arguments passed to CantioAIDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = CantioAIDataset(
        data_dir=data_dir,
        split="train",
        **dataset_kwargs
    )
    val_dataset = CantioAIDataset(
        data_dir=data_dir,
        split="val",
        **dataset_kwargs
    )
    test_dataset = CantioAIDataset(
        data_dir=data_dir,
        split="test",
        **dataset_kwargs
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader