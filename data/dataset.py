import os
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image

from data.augmentation import get_val_transforms, get_train_transforms

class WildlifeDataset(Dataset):
    """ Wildlife dataset wrapper. """

    def __init__(self, root_dir: str, transform=None, split: str = 'train'):
        """
        :param root_dir: Path to the root directory of the dataset.
        :param transform: Transformation applied to each image
        :param split: 'train, 'val' or 'test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # Load dataset using ImageFolder structure
        self.dataset = datasets.ImageFolder(self.root_dir)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def stratified_split(dataset: WildlifeDataset,
                    train_size: float = 0.8,
                    val_size: float = 0.15,
                    test_size: float = 0.15,
                    random_seed: int = 7278) -> tuple[Subset[Any], Subset[Any], Subset[Any]]:
    """
    Perform a stratified split of a dataset, into train, validation and test sets.
    :param dataset: PyTorch dataset
    :param train_size: Fraction of dataset to use for training
    :param val_size: Fraction of dataset to use for validation
    :param test_size: Fraction of dataset to use for testing
    :param random_seed: Random seed for reproducibility
    :return: train_size, val_size, test_size
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    # Get all labels
    labels = [label for _, label in dataset]
    indices = np.arange(len(dataset))

    # First split -> Test and Training + Val
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=train_size,
        stratify=[labels[i] for i in indices],
        random_state=random_seed
    )

    # Second split -> Training + Val in to different splits
    val_size_adjusted = val_size / (train_size + val_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        stratify=[labels[i] for i in train_val_indices],
        random_state=random_seed
    )

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def get_data_loaders(data_path: str,
                     batch_size: int = 32,
                     num_workers: int = 4,
                     train_split: float = 0.7,
                     val_split: float = 0.15,
                     test_split: float = 0.15,
                     use_augmentation: bool = True,
                     random_seed: int = 7278,
                     data_fraction: float = 1.0) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test data loaders.
    :param data_path: Path to dataset
    :param batch_size: Batch size for data loaders
    :param num_workers: Number of workers for data loaders
    :param train_split: Fraction of dataset to use for training
    :param val_split: Fraction of dataset to use for validation
    :param test_split: Fraction of dataset to use for test
    :param use_augmentation: Whether to use data augmentation
    :param random_seed: random seed for reproducibility
    :param data_fraction: Fraction of dataset to be split
    :return: train_loader, val_loader, test_loader, num_classes
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Get transforms
    train_transforms = get_train_transforms(use_augmentation=use_augmentation)
    val_transform = get_val_transforms()

    # Load full dataset
    full_dataset = WildlifeDataset(data_path, transform=None)
    num_classes = len(full_dataset.classes)

    # Perform stratified split
    train_dataset, val_dataset, test_dataset = stratified_split(
        full_dataset, train_split, val_split, test_split, random_seed
    )

    # Apply data fraction to training set if needed
    if data_fraction < 1.0:
        train_labels = [full_dataset.dataset[i][1] for i in train_dataset.indices]
        n_samples = int(len(train_labels) * data_fraction)

        # Stratified sampling
        train_indices = train_dataset.indices
        selected_indices, _ = train_test_split(
            train_indices,
            train_size=n_samples,
            stratify=train_labels,
            random_state=random_seed
        )

    # Apply transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes