import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

def plot_training_curves(
        tarin_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        save_path: str = None
):
    """
    PLot training and validation curves.
    :param tarin_losses:
    :param val_losses:
    :param train_accuracies:
    :param val_accuracies:
    :param save_path:
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8)) #

    epochs = range(1, len(train_accuracies) + 1)

    # Plot losses
    ax1.plot(epochs, tarin_losses, 'b-', label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label="Validation Loss", linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12) # Check if the size is readable
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3) # Evaluate changes for better readability

    # Plot Accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

def plot_learning_curves(
        data_fractions: List[float],
        scratch_accuracies: List[float],
        finetune_accuracies: List[float],
        zeroshot_accuracies: List[float],
        save_path: str = None
):
    """
    Plot the learning curves for the three models together
    :param data_fractions: List of data fractions used.
    :param scratch_accuracies: Accuracies for training from scratch.
    :param finetune_accuracies: Accuracies for training for fine-tune.
    :param zeroshot_accuracies: Zero-shot accuracies.
    :param save_path: path to save the figure.
    """

    plt.figure(figsize=(12, 8))

    # Convert fractions to percentages for x-axis
    data_percentages = [f * 100 for f in data_fractions]

    # Plot learning curves
    plt.plot(data_percentages, scratch_accuracies, 'o-',
             label="Training from Scratch Accuracy", linewidth=2, markersize=8)
    plt.plot(data_percentages, finetune_accuracies, 's-',
             label="Fine-tune Training Accuracy", linewidth=2, markersize=8)
    plt.plot(data_percentages, zeroshot_accuracies, '--',
             label="Zero-shot Training Accuracy", linewidth=2, markersize=8)

    plt.xlabel('Training Data Size (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 105)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: str = None, show: bool = False):
    """
    Plot confusion matrix.
    :param cm: Confusion matrix.
    :param class_names: List of class names.
    :param save_path: path to save the figure.
    :param show: Show or not plot, default False (not displaying the plot).
    """

    fig = plt.figure(figsize=(12, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)