import torch
import torch.nn as nn
from numpy.distutils.lib2def import output_def
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
from typing import Dict, Tuple

def evaluate_model(
        model: nn.Model,
        dataloader: DataLoader,
        device: torch.device
) -> Dict[str, float]:
    """
    Evaluate a model on a given dataloader.
    :param model: PyTorch model.
    :param dataloader: Test dataloader.
    :param device: Device where the model will be evaluated.
    :return: Dictionary with metrics computed on the test set.
    """

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    return compute_metrics(all_predictions, all_targets)

def compute_metrics(
        targets: np.ndarray,
        predictions: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics on a given dataset.
    :param targets: Ground truth labels.
    :param predictions: Predicted labels.
    :return: Dictionary containing accuracy, precision, recall, and f1 score.
    """

    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets,
                                                               predictions,
                                                               average='weighted',
                                                               zero_division=0)

    return {'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100}

def get_confusion_matrix(
        targets: np.ndarray,
        predictions: np.ndarray,
        num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix on a given dataset.
    :param targets: Ground truth labels.
    :param predictions: Predicted labels.
    :param num_classes: Number of classes
    :return: Confusion Matrix
    """

    return confusion_matrix(targets, predictions, labels=range(num_classes))