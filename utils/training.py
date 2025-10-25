import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int = 0,
) -> Tuple[float, float]:
    """
    Train the model on one epoch.
    :param model: PyTorch model.
    :param dataloader: Training dataloader.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param device: Device to train the model on.
    :param epoch: Current epoch.
    :return: average_loss, avg_accuracy.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward Pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backword Pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix_str(
            s = f'loss: {running_loss / (batch_idx + 1)}, accuracy: {100.0 * correct / total}'
        )

    avg_loss = running_loss / len(dataloader)
    avg_accuracy = 100.0 * correct / total

    return avg_loss, avg_accuracy

def validate(model: nn.Model,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    Validate model on validation or test set.
    :param model: PyTorch model.
    :param dataloader: Validation dataloader.
    :param criterion: Loss function.
    :param device: Device to train the model on.
    :return: average_loss, avg_accuracy.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(dataloader)
    avg_accuracy = 100.0 * correct / total

    return avg_loss, avg_accuracy

class EarlyStopping:
    """ Early stooping will stop training when validation loss stops  to improve. """

    def __init__(self, patience = 7, min_delta=0, mode: str = 'min'):
        """
        :param patience: Number of epochs to wait before stopping training.
        :param min_delta: Minimum change to qualify as improvement.
        :param mode: 'min' for loss, 'max' for accuracy.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """ Checks if early stooping is required. """

        # Training star case
        if self.best_score is None:
            self.best_score = score

        # Variation smaller than the accepted as improvement for mode == 'min'
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            # Apply the limit over the number of epochs without considerable improvement
            if self.counter >= self.patience:
                self.early_stop = True
        # Variation smaller than the accepted as improvement for mode == 'max'
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            # Apply the limit over the number of epochs without considerable improvement
            if self.counter >= self.patience:
                self.early_stop = True
        # If variation is above the considerable improvement value,
        # counter = 0, process continues.
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
