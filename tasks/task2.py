import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time
import json

from config import Config
from data.dataset import get_data_loaders
from models.pretrained import get_pretrained_model
from utils.training import train_epoch, validate, EarlyStopping
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_curves

def fine_tune_pretrained(config: Config,
                         data_fraction: float = 1.0,
                         save_mode: bool = True):
    """
    Fine-tune a pretrained ResNet model
    :param config: Configuration object
    :param data_fraction: Fraction of data to be used
    :param save_mode: Whether to save the model or not
    :return: Dictionary with training history and metrics
    """

    if config is None:
        config=Config()

    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)

    # Prepare data loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        data_path=str(config.DATA_DIR / config.DATASET_NAME),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        use_augmentation=config.USE_AUGMENTATION,
        data_fraction=data_fraction
    )

    print(f"Training on {num_classes} classes.")
    print(f"Training samples: {len(train_loader)}.")
    print(f"Validation samples: {len(val_loader)}.")
    print(f"Test samples: {len(test_loader)}.")

    model = get_pretrained_model(
        num_classes=num_classes,
        model_name='resnet18',
        freeze_until_layer=config.FREEZE_UNTIL_LAYER,
    ).to(config.DEVICE)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}/{total_params:,}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.FINETUNE_LR,
        weight_decay=config.FINETUNE_WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.FINETUNE_EPOCHS
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.FINETUNE_EARLY_STOPPING_PATIENCE,
        mode='max'
    )

    # Training Loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0.0
    start_time = time.time()

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        # Train
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch
        )

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, config.DEVICE)

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"\n Epoch {epoch}/{config.FINETUNE_EPOCHS}:")
        print(f" Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f" Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if save_mode:
                save_path = config.MODELS_DIR / f"task2_fine_tuned_{config.DATASET_NAME}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model (Val Acc: {best_val_accuracy:.2f}%).")

        if scheduler:
            scheduler.step()

        # Early stop
        if early_stopping(val_accuracy):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, config.DEVICE)

    print(f"\n Test Results: ")
    print(f" Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f" Precision: {test_metrics['precision']:.2f}%")
    print(f" Recall: {test_metrics['recall']:.2f}%")
    print(f" F1-score: {test_metrics['f1-score']:.2f}%")

    # Plot training curve
    plot_path = config.PLOTS_DIR / f"task2_fine_tuned_{data_fraction:.2f}.png"
    plot_training_curves(
        train_losses, val_losses, train_accuracies, val_accuracies, save_path=str(plot_path)
    )

    # Save results
    results = {
        'data_fraction': data_fraction,
        'training_time': training_time,
        'best_val_accuracy': best_val_accuracy,
        'test_metrics': test_metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

    result_path = config.METRICS_DIR / f'task2_fine_tuned_{data_fraction:.2f}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    fine_tune_pretrained()