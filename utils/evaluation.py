import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
from typing import Dict, Tuple

def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
):
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

    y_true = np.concatenate(all_targets, axis=0)
    y_predictions = np.concatenate(all_predictions, axis=0)

    return compute_metrics(y_true, y_predictions)

def compute_metrics(
        targets: np.ndarray,
        predictions: np.ndarray
):
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
            'f1-score': f1 * 100}

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