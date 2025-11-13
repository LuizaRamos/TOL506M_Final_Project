import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from typing import Optional

from config import Config
from data.dataset import WildlifeDataset, SplitIndices, get_data_loaders
from models.resnet_scratch import ResNet18Scratch, ResNet34Scratch, ResNet50Scratch, ResNet101Scratch, ResNet152Scratch
from utils.training import train_epoch, validate, EarlyStopping
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_curves

def train_from_scratch(config: Config = None,
                       data_fraction: float = 1.0,
                       version_RestNet: int = 18,
                       save_model: bool = True,
                       fixed_indices: Optional[SplitIndices] = None):
    """
    Train ResNet-(version_RestNet)
    :param version_RestNet: 18, 34, 50, 101 or 152.
    :param config: Configuration object.
    :param data_fraction: Fraction of training data to use.
    :param save_model: Whether to save the model.
    :param fixed_indices: Fixed indices to use.
    :return: Dictionary with training history and metrics
    """

    if config is None:
        config = Config()

    torch.manual_seed(config.RANDOM_SEED)

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        data_path=str(config.DATA_PATH),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        use_augmentation=config.USE_AUGMENTATION,
        data_fraction=data_fraction,
        save_processed_root=str(config.PROCESSED_DIR),
        fixed_indices=fixed_indices
    )

    print(f"Training on {num_classes} classes.")
    print(f"Training samples: {len(train_loader)}.")
    print(f"Validation samples: {len(val_loader)}.")
    print(f"Test samples: {len(test_loader)}.")

    if version_RestNet == 18:
        model = ResNet18Scratch(num_classes=num_classes).to(config.DEVICE)
        print("Model ResNet-18 from scratch.")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    elif version_RestNet == 34:
        model = ResNet34Scratch(num_classes=num_classes).to(config.DEVICE)
        print("Model ResNet-34 from scratch.")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    elif version_RestNet == 50:
        model = ResNet50Scratch(num_classes=num_classes).to(config.DEVICE)
        print("Model ResNet-50 from scratch.")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    elif version_RestNet == 101:
        model = ResNet101Scratch(num_classes=num_classes).to(config.DEVICE)
        print("Model ResNet-101 from scratch.")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    elif version_RestNet == 152:
        model = ResNet152Scratch(num_classes=num_classes).to(config.DEVICE)
        print("Model ResNet-152 from scratch.")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        raise ValueError(f"Unsupported version: {version_RestNet}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.SCRATCH_LR,
        momentum=config.SCRATCH_MOMENTUM,
        weight_decay=config.SCRATCH_WEIGHT_DECAY
    )

    # Learning rate scheduler
    if config.LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.SCRATCH_EPOCHS
        )
    elif config.LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
        )
    else:
        scheduler = None

    # Early stop
    early_stopping = EarlyStopping(
        patience=config.SCRATCH_EARLY_STOPPING_PATIENCE,
        mode='max' # Considers accuracy instead of loss
    )

    # Training Loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0.0
    start_time = time.time()

    for epoch in range(1, config.SCRATCH_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.DEVICE)


        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\n Epoch {epoch}/{config.SCRATCH_EPOCHS}:")
        print(f" Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f" Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            if save_model:
                save_path = config.MODELS_DIR / f"task1_scratch_best_fraction_{data_fraction:.2f}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model (Val Acc: {best_val_accuracy:.2f}%) to {save_path}.")

        if scheduler:
            scheduler.step()

        # Early stop
        if early_stopping(val_acc):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    test_metrics = evaluate_model(model, test_loader, config.DEVICE)

    print(f"\n Test Results: ")
    print(f" Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f" Precision: {test_metrics['precision']:.2f}%")
    print(f" Recall: {test_metrics['recall']:.2f}%")
    print(f" F1-score: {test_metrics['f1-score']:.2f}%")

    # Plot training curves
    plot_path = config.PLOTS_DIR / f"task1_scratch_learning_curves_{data_fraction:.2f}.png"
    plot_training_curves(
        train_losses, val_losses, train_accuracies, val_accuracies,
        save_path=str(plot_path)
    )

    # Save results (JSON, not the model again)
    results = {
        "data_fraction": data_fraction,
        "training_time": training_time,
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    results_path = config.METRICS_DIR / f"task1_scratch_results_{data_fraction:.2f}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics to {results_path}")

    return results

if __name__ == "__main__":
    train_from_scratch()