import os
import random
import shutil
import json
import csv
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any, NamedTuple

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
        :param root_dir: Path to the root directory of the dataset class.
        :param transform: Transformation applied to each image
        :param split: 'train, 'val' or 'test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # Load dataset using ImageFolder structure
        self.dataset = datasets.ImageFolder(self.root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

class SplitIndices(NamedTuple):
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

def stratified_split(dataset: WildlifeDataset,
                    train_size: float = 0.7,
                    val_size: float = 0.15,
                    test_size: float = 0.15,
                    random_seed: int = 7278) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified split of a dataset, into train, validation and test sets.
    :param dataset: PyTorch dataset
    :param train_size: Fraction of dataset to use for training
    :param val_size: Fraction of dataset to use for validation
    :param test_size: Fraction of dataset to use for testing
    :param random_seed: Random seed for reproducibility
    :return: train_indices, val_indices, test_indices
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    n = len(dataset)
    indices = np.arange(n)

    # Get all labels - handle both direct ImageFolder and wrapped datasets
    if hasattr(dataset, 'targets'):
        # Direct ImageFolder with targets attribute
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Wrapped dataset (e.g., Subset) with underlying dataset having targets
        labels = np.array(dataset.dataset.targets)
    else:
        # Fallback -> pull labels by indexing
        labels = np.array([dataset[i][1] for i in indices])

    # First split -> Test and Training + Val
    train_val_size = train_size + val_size
    
    # Handle case where test_size is 0 or very close to 0
    if test_size < 1e-6:
        test_indices = np.array([], dtype=int)
        train_val_indices = indices
    else:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels,
            random_state=random_seed
        )

    # Second split -> Training + Val into Traning AND Val
    # Handle case where val_size is 0 or very close to 0
    if val_size < 1e-6:
        train_indices = train_val_indices
        val_indices = np.array([], dtype=int)
    else:
        labels_train_val = labels[train_val_indices]
        val_size_adjusted = val_size / train_val_size
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size_adjusted,
            stratify=labels_train_val,
            random_state=random_seed
        )

    return train_indices, val_indices, test_indices

def _unwrap_to_imagefolder(ds: Any) -> datasets.ImageFolder:
    """ Follow .dataset links until reaches an ImageFolder object. """
    cur = ds
    while hasattr(cur, 'dataset'):
        cur = cur.dataset
    return cur

def _gather_samples(data_sample: Any) -> List[Tuple[str, int]]:
    """
    Gather a list of tuples from a dataset.
    :param data_sample: ImageFolder dataset to gather samples from.
    :return: list of filepath and labels from the dataset
    """
    base = _unwrap_to_imagefolder(data_sample)
    if hasattr(base, 'samples'):
        return list(base.samples)
    if hasattr(base, 'imgs'):
        return list(base.imgs)
    raise AttributeError('Dataset does not expose samples nor images for filepaths and labels')

def compute_class_counts(dataset: Any, indices: Optional[np.ndarray] = None) -> Dict[str, int]:
    """
    Compute class counts for a given dataset, possibly limited to a set of indices.
    :param dataset: dataset to compute counts from.
    :param indices: set of indices to compute counts for.
    :return: counts per class in the dataset.
    """
    base = _unwrap_to_imagefolder(dataset)
    class_names = get_class_names(dataset)
    samples = _gather_samples(dataset)

    if indices is not None:
        samples = [samples[i] for i in indices]

    counts = {name: 0 for name in class_names}
    for _, label in samples:
        # Translate visible names if not yet done
        name = class_names[label]
        if name in _ITALIAN_NAMES:
            name = _IT2EN.get(name, name)
        counts[name] = counts.get(name, 0) + 1

    return counts

def _save_link_or_copy(src: str, dst: str) -> None:
    """
    Create a hardlink if possible otherwise falls back and copy.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        # Try hardlink
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def materialize_split(base_dataset: Any,
                     indices: np.ndarray,
                     output_dir: str) -> None:
    """ Materialize a split to out_dir with structure {class}/{filename}. """
    samples = _gather_samples(base_dataset)
    base = _unwrap_to_imagefolder(base_dataset)
    class_names = list(base.classes)

    # Ensure English names
    out_class_names = [(_IT2EN.get(n, n) if n in _ITALIAN_NAMES else n) for n in class_names]

    for i in indices:
        scr_path, label = samples[i]
        cls = out_class_names[label]
        filename = os.path.basename(scr_path)
        dst = os.path.join(output_dir, cls, filename)
        _save_link_or_copy(scr_path, dst)


# I'm saving in both csv and json to see which goes better
def _write_counts(path: str, counts: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    csv_path = os.path.splitext(path)[0] + '.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['class', 'count'])
        for k, v in counts.items():
            w.writerow([k, v])

    json_path = os.path.splitext(path)[0] + '.json'
    with open(json_path, 'w', newline='') as jsonfile:
        json.dump(counts, jsonfile, indent=2)

# Translation Functions, and Get label Function

_IT2EN= {
        "cane": "dog",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "chicken",
        "gatto": "cat",
        "mucca": "cow",
        "pecora": "sheep",
        "ragno": "spider",
        "scoiattolo": "squirrel",
    }

_ITALIAN_NAMES = set(_IT2EN.keys())

def is_italian(dataset: Any) -> bool:
    """
    Function that checks if a dataset has its labels in Italian,
    based on the list of italian names for the Animals10.
    :param dataset: PyTorch dataset, but specially the Wildlife dataset, given the dictionary used.
    :return: True if that dataset's visible class names are italian
    """
    base = _unwrap_to_imagefolder(dataset)
    if not hasattr(base, 'classes'):
        return False
    names = set(map(str.lower, base.classes))
    overlap = names & _ITALIAN_NAMES
    return len(overlap) > 0

def translate_names(dataset: Any) -> Any:
    """
    Translate Italian class names to English class names, inside an ImageFolder object,
    keeping the original class order and indices stable, based on the list of italian names
    for the Animals10.
    :param dataset: PyTorch dataset, but specially the Wildlife dataset, given the dictionary used.
    :return: dataset with English class names, mantling original class order and indices stable
    """
    base = _unwrap_to_imagefolder(dataset)
    if hasattr(base, 'classes') and hasattr(base, 'class_to_idx'):
        base.classes = [_IT2EN.get(name,name) for name in base.classes]
        base.class_to_idx = {name: idx for idx, name in enumerate(base.classes)}
    return dataset

def get_class_names(dataset: Any) -> List[str]:
    """ Obtain the class names from a potentially nested dataset. """

    base = _unwrap_to_imagefolder(dataset)
    if not hasattr(base, 'classes'):
        raise AttributeError('Dataset does not contain a class attribute')
    names = list(base.classes)

    if is_italian(base):
        names = [_IT2EN.get(name, name) for name in names]

    return names


# Data Loaders
def get_data_loaders(data_path: str,
                     batch_size: int = 32,
                     num_workers: int = 4,
                     train_split: float = 0.7,
                     val_split: float = 0.15,
                     test_split: float = 0.15,
                     use_augmentation: bool = True,
                     random_seed: int = 7278,
                     data_fraction: float = 1.0,
                     save_processed_root: Optional[str] = None,
                     fixed_indices: Optional[SplitIndices] = None
                     ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test data loaders, using stratified splits.
    :param data_path: Path to dataset
    :param batch_size: Batch size for data loaders
    :param num_workers: Number of workers for data loaders
    :param train_split: Fraction of dataset to use for training
    :param val_split: Fraction of dataset to use for validation
    :param test_split: Fraction of dataset to use for test
    :param use_augmentation: Whether to use data augmentation
    :param random_seed: random seed for reproducibility
    :param data_fraction: Fraction of dataset to be split
    :param save_processed_root: Path to save processed data
    :param fixed_indices: Fixed indices to use for training
    :return: train_loader, val_loader, test_loader, num_classes
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Get transforms
    train_transforms = get_train_transforms(use_augmentation=use_augmentation)
    val_transforms = get_val_transforms()

    # Base dataset computing labels and class count
    base_full = WildlifeDataset(data_path, transform=None)
    num_classes = len(base_full.classes)

    # Perform stratified split
    if fixed_indices is None:
        train_idx, val_idx, test_idx = stratified_split(
            base_full, train_split, val_split, test_split, random_seed
        )
    else:
        train_idx = np.array(fixed_indices.train, dtype=int)
        val_idx = np.array(fixed_indices.val, dtype=int)
        test_idx = np.array(fixed_indices.test, dtype=int)

    # Apply data fraction to training set if needed
    if data_fraction < 1.0:
        # labels for the base train indices
        base = _unwrap_to_imagefolder(base_full)
        if hasattr(base, "targets"):
            labels_all = np.array(base.targets)
        else:
            # fallback (slow but safe)
            labels_all = np.array([base_full[i][1] for i in range(len(base_full))])

        labels_train = labels_all[train_idx]
        n_samples = max(1, int(len(train_idx) * data_fraction))
        train_idx, _ = train_test_split(
            train_idx,
            train_size=n_samples,
            stratify=labels_train,
            random_state=random_seed
        )

    # Create one WildlifeDateset per slip to keep transforms independent
    train_base = WildlifeDataset(data_path, transform=train_transforms, split='train')
    val_base = WildlifeDataset(data_path, transform=val_transforms, split='val')
    test_base = WildlifeDataset(data_path, transform=val_transforms, split='test')

    # Wrap with Subset using split-specific indices
    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(val_base, val_idx)
    test_dataset = Subset(test_base, test_idx)

    # If class names are still in Italian, translate them in-place for all bases
    if is_italian(train_base):
        translate_names(train_base)
        translate_names(val_base)
        translate_names(test_base)

    # Create data loaders
    persistent_workers = bool(num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    if save_processed_root is not None:
        config_tag = f'seed{random_seed}_data_frac{int(data_fraction * 100)}'
        processed_root = Path(save_processed_root) / config_tag

        train_dir = processed_root / 'train'
        val_dir = processed_root / 'val'
        test_dir = processed_root / 'test'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        original_base = WildlifeDataset(data_path, transform=None)
        materialize_split(original_base, train_idx, str(train_dir))
        materialize_split(original_base, val_idx, str(val_dir))
        materialize_split(original_base, test_idx, str(test_dir))

        counts_train = compute_class_counts(original_base, train_idx)
        counts_val = compute_class_counts(original_base, val_idx)
        counts_test = compute_class_counts(original_base, test_idx)
        _write_counts(str(processed_root / 'class_count_train'), counts_train)
        _write_counts(str(processed_root / 'class_count_val'), counts_val)
        _write_counts(str(processed_root / 'class_count_test'), counts_test)
        overall = {k: counts_train.get(k, 0) + counts_val.get(k, 0) + counts_test.get(k, 0)
                   for k in sorted(set(list(counts_train) + list(counts_val) + list(counts_test)))}
        _write_counts(str(processed_root / 'class_counts_overall'), overall)


    return train_loader, val_loader, test_loader, num_classes
