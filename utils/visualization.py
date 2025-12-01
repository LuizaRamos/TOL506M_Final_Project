import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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
        scratch_uncertainties: List[float],
        finetune_uncertainties: List[float],
        zeroshot_uncertainties: List[float],
        save_path: str = None
):
    """
    Plot the learning curves for the three models together
    :param data_fractions: List of data fractions used.
    :param scratch_accuracies: Accuracies for training from scratch.
    :param finetune_accuracies: Accuracies for training for fine-tune.
    :param zeroshot_accuracies: Zero-shot accuracies.
    :param scratch_uncertainties: Accuracies for training from scratch.
    :param finetune_uncertainties: Accuracies for training from fine-tune.
    :param zeroshot_uncertainties: Accuracies for training from zero-shot.
    :param save_path: path to save the figure.
    """

    # Convert fractions to percentages for x-axis
    data_percentages = [f * 100 for f in data_fractions]

    # Put everything into a long-form DataFrame for seaborn
    models = (
            ["Training from Scratch"] * len(data_percentages) +
            ["Fine-tune Training"] * len(data_percentages) +
            ["Zero-shot"] * len(data_percentages)
    )

    df = pd.DataFrame({
        "Training Data Size (%)": data_percentages * 3,
        "Test Accuracy (%)": (
                scratch_accuracies +
                finetune_accuracies +
                zeroshot_accuracies
        ),
        "Uncertainty": (
                scratch_uncertainties +
                finetune_uncertainties +
                zeroshot_uncertainties
        ),
        "Model": models
    })

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Line plot with seaborn
    ax = sns.lineplot(
        data=df,
        x="Training Data Size (%)",
        y="Test Accuracy (%)",
        hue="Model",
        palette=sns.color_palette("plasma", n_colors=3),
        marker="o",
        linewidth=2
    )

    # Add uncertainty bands (mean +/- uncertainty)
    for model_name, model_df in df.groupby("Model"):
        x = model_df["Training Data Size (%)"].values
        y = model_df["Test Accuracy (%)"].values
        dy = model_df["Uncertainty"].values

        # Make sure everything is numpy arrays
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dy = np.asarray(dy, dtype=float)

        plt.fill_between(
            x,
            y - dy,
            y + dy,
            alpha=0.2
        )

    plt.xlabel('Training Data Size (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Learning Curves with Uncertainty', fontsize=14, fontweight='bold')
    plt.xlim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", fontsize=12, loc='best')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()

def plot_learning_curve_for_metric(
    df: pd.DataFrame,
    metric_col: str,
    metric_unc_col: str,
    metric_pretty_name: str,
    methods_to_use: List[str],
    save_path: str = None
):
    """
    Plot learning curve (with uncertainty) for a given metric for:
    - Task 1 (scratch)
    - Task 2 (freeze=up_to_layer3)
    - Task 3 (zero shot)
    - linear probe
    :param df: Pandas DataFrame containing metric data.
    :param metric_col: Name of metric column to be considered.
    :param metric_unc_col: Name of uncertainty column to be considered.
    :param metric_pretty_name: Name of metric pretty name.
    :param methods_to_use: List of methods to use.
    :param save_path: path to save the figure.
    :return: prints the figure generated with plot
    """

    # Filter to the 4 methods we care about
    sub = df[df["Method"].isin(methods_to_use)].copy()
    sub["Model"] = sub["Method"].map(methods_to_use)

    # Convert fractions (0.1, 0.25, ...) to percentages
    sub["Training Data Size (%)"] = sub["data_fraction"] * 100.0

    # Build long-form dataframe for seaborn
    plot_df = sub[[
        "Training Data Size (%)",
        metric_col,
        metric_unc_col,
        "Model"
    ]].rename(
        columns={
            metric_col: f"{metric_pretty_name} (%)",
            metric_unc_col: "Uncertainty"
        }
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Line plot
    ax = sns.lineplot(
        data=plot_df,
        x="Training Data Size (%)",
        y=f"{metric_pretty_name} (%)",
        hue="Model",
        marker="o",
        linewidth=2
    )

    # Add uncertainty bands for each model
    for model_name, model_df in plot_df.groupby("Model"):
        x = model_df["Training Data Size (%)"].values.astype(float)
        y = model_df[f"{metric_pretty_name} (%)"].values.astype(float)
        dy = model_df["Uncertainty"].values.astype(float)

        plt.fill_between(
            x,
            y - dy,
            y + dy,
            alpha=0.2
        )

    plt.xlabel("Training Data Size (%)", fontsize=12)
    plt.ylabel(f"Test {metric_pretty_name} (%)", fontsize=12)
    plt.title(f"{metric_pretty_name} Learning Curves with Uncertainty", fontsize=14, fontweight="bold")
    plt.xlim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", fontsize=12, loc="best")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

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