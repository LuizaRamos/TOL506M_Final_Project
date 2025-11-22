import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
import numpy as np
from tqdm.auto import tqdm
import time
import json
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data.dataset import SplitIndices, get_data_loaders, get_class_names
from utils.evaluation import compute_metrics, get_confusion_matrix
from utils.visualization import plot_confusion_matrix
from utils.training import EarlyStopping


def build_prompts_for_classes(class_names, config: Config):
    """
    Helper function that builds a list of prompts for a given set of classes.
    :param class_names: class names.
    :param config: config object.
    Returns:
      prompts_per_class: list[list[str]] list of prompts for each class
      flat_prompts: flattened list of all prompts
      class_indices: list of (start, end) indices in flat_prompts for each class.
    """
    prompts_per_class = []
    for cls in class_names:
        if config.ZERO_SHOT_PROMPTS and cls in config.ZERO_SHOT_PROMPTS:
            prompts = config.ZERO_SHOT_PROMPTS[cls]
        else:
            # fallback template
            prompts = [f"a photo of a {cls}"]
        prompts_per_class.append(prompts)

    flat_prompts = []
    class_indices = []
    idx = 0
    for prompts in prompts_per_class:
        start = idx
        flat_prompts.extend(prompts)
        idx += len(prompts)
        end = idx
        class_indices.append((start, end))

    return prompts_per_class, flat_prompts, class_indices


def zero_shot_classification(
    config: Config = None,
    fixed_indices: Optional[SplitIndices] = None,
    test_loader: Optional[DataLoader] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    train_linear_probe: Optional[bool] = None,
):
    """
    Task 3: Zero-Shot classification with SigLIP 2 + optional linear probe.

    Zero-shot part:
      - Uses SigLIP2 image & text encoders.
      - Uses prompts per class from config.ZERO_SHOT_PROMPTS.
      - NO training on your dataset (true zero-shot).

    Linear probe (optional, NOT zero-shot):
      - Trains a logistic regression head on frozen SigLIP2 embeddings.
      - Intended for Task 4 to compare data efficiency vs Tasks 1 & 2.

    :param config: Configuration object
    :param fixed_indices: Fixed indices to use (for reproducible splits)
    :param test_loader: Test loader with images in [0, 1], Resize+ToTensor only
    :param train_loader: Train loader (same preprocessing as test_loader)
    :param val_loader: Validation loader
    :param train_linear_probe: if True, trains linear probe on SigLIP features
    :return: Dict with zero-shot metrics + optional linear-probe metrics
    """

    if config is None:
        config = Config()

    # Default to config flag if not passed
    if train_linear_probe is None:
        train_linear_probe = config.SIGLIP_LINEAR_PROBE

    device = config.DEVICE

    # Load SigLIP 2 model + processor from config
    model_name = 'google/siglip2-base-patch16-224'
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    print(f"Model Loaded: {model_name}")

    # Data loaders
    need_loaders = (
        test_loader is None
        or (train_linear_probe and (train_loader is None or val_loader is None))
    )

    if need_loaders:
        train_, val_, test_, num_classes = get_data_loaders(
            data_path=str(config.DATA_PATH),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            train_split=config.TRAIN_SPLIT,
            val_split=config.VAL_SPLIT,
            test_split=config.TEST_SPLIT,
            use_augmentation=config.USE_AUGMENTATION,
            data_fraction=1.0,
            save_processed_root=str(config.PROCESSED_DIR),
            fixed_indices=fixed_indices,
        )
        if train_loader is None:
            train_loader = train_
        if val_loader is None:
            val_loader = val_
        if test_loader is None:
            test_loader = test_
    else:
        num_classes = config.NUM_CLASSES

    # Class names and prompts
    class_names = get_class_names(test_loader.dataset)
    print(f"Classes: {class_names}")

    prompts_per_class, flat_prompts, class_indices = build_prompts_for_classes(
        class_names, config
    )

    print("Using prompts per class:")
    for cls, prompts in zip(class_names, prompts_per_class):
        print(f"  {cls}: {prompts}")

    # Text encoding: average prompts per class
    text_inputs = processor(
        text=flat_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_features_all = model.get_text_features(**text_inputs)

    text_features_all = text_features_all / text_features_all.norm(
        p=2, dim=-1, keepdim=True
    )

    # Average prompt embeddings to get a single prototype per class
    class_text_features = []
    for (start, end) in class_indices:
        feats = text_features_all[start:end]  # (num_prompts_for_class, dim)
        class_text_features.append(feats.mean(dim=0))

    class_text_features = torch.stack(class_text_features, dim=0)  # (num_classes, dim)
    class_text_features = class_text_features / class_text_features.norm(
        p=2, dim=-1, keepdim=True
    )

    # ZERO-SHOT EVALUATION
    all_predictions = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Zero-shot evaluating"):
            images = images.to(device)
            all_labels.append(labels.cpu().numpy())

            # images already in [0,1]; let processor normalize/resize
            inputs = processor(
                images=list(images),
                return_tensors="pt",
                do_rescale=False,
            ).to(device)

            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

            # Cosine similarity logits
            logits_per_image = image_features @ class_text_features.T

            probs = logits_per_image.softmax(dim=-1)
            preds = probs.argmax(dim=-1)

            all_predictions.append(preds.cpu().numpy())

    zero_shot_inference_time = time.time() - start_time

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)

    zero_shot_metrics = compute_metrics(y_true, y_pred)
    zero_shot_conf_mat = get_confusion_matrix(y_true, y_pred, num_classes)

    print(f"\nZero-Shot Test Results (SigLIP2):")
    print(f"  Accuracy:  {zero_shot_metrics['accuracy']:.2f}%")
    print(f"  Precision: {zero_shot_metrics['precision']:.2f}%")
    print(f"  Recall:    {zero_shot_metrics['recall']:.2f}%")
    print(f"  F1-Score:  {zero_shot_metrics['f1_score']:.2f}%")

    # Plot zero-shot confusion matrix
    plot_path = config.PLOTS_DIR / "task3_zero_shot_classification.png"
    plot_confusion_matrix(zero_shot_conf_mat, class_names, plot_path, show=False)

    # print a plain-text confusion matrix (to avoid problem with GitHub rendition)
    print("\nZero-Shot Confusion Matrix (rows = true, cols = pred):")
    print(zero_shot_conf_mat)

    # --- OPTIONAL: LINEAR PROBE ON TOP OF SIGLIP FEATURES (NOT ZERO-SHOT) ---
    linear_probe_results = None

    if train_linear_probe:
        print("\nTraining linear probe (logistic regression) on SigLIP2 image embeddings...")

        def extract_features(loader):
            feats_list, labels_list = [], []
            with torch.no_grad():
                for images, labels in tqdm(loader, desc="Extracting features"):
                    images = images.to(device)
                    inputs = processor(
                        images=list(images),
                        return_tensors="pt",
                        do_rescale=False,
                    ).to(device)
                    img_feats = model.get_image_features(**inputs)
                    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
                    feats_list.append(img_feats.cpu())
                    labels_list.append(labels.cpu())
            feats = torch.cat(feats_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            return feats, labels

        train_feats, train_labels = extract_features(train_loader)
        val_feats, val_labels = extract_features(val_loader)
        test_feats, test_labels = extract_features(test_loader)

        embed_dim = train_feats.shape[1]
        linear_head = nn.Linear(embed_dim, num_classes).to(device)

        optimizer = torch.optim.Adam(
            linear_head.parameters(),
            lr=config.LINEAR_PROBE_LR,
            weight_decay=config.LINEAR_PROBE_WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(
            patience=config.LINEAR_PROBE_PATIENCE,
            mode="max",
        )

        # Wrap features in TensorDataset/DataLoader
        train_dataset = TensorDataset(train_feats, train_labels)
        val_dataset = TensorDataset(val_feats, val_labels)
        test_dataset = TensorDataset(test_feats, test_labels)

        feat_batch_size = config.BATCH_SIZE
        train_feat_loader = DataLoader(train_dataset, batch_size=feat_batch_size, shuffle=True)
        val_feat_loader = DataLoader(val_dataset, batch_size=feat_batch_size, shuffle=False)
        test_feat_loader = DataLoader(test_dataset, batch_size=feat_batch_size, shuffle=False)

        best_val_acc = 0.0
        start_lp = time.time()

        for epoch in range(1, config.LINEAR_PROBE_EPOCHS + 1):
            # Train
            linear_head.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for feats, labels in train_feat_loader:
                feats = feats.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = linear_head(feats)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = 100.0 * correct / total

            # Validate
            linear_head.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for feats, labels in val_feat_loader:
                    feats = feats.to(device)
                    labels = labels.to(device)
                    logits = linear_head(feats)
                    loss = criterion(logits, labels)
                    val_loss_sum += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss_sum / val_total
            val_acc = 100.0 * val_correct / val_total

            print(
                f"[Linear probe] Epoch {epoch}/{config.LINEAR_PROBE_EPOCHS} "
                f"- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if early_stopping(val_acc):
                print(f"Linear probe early stopping at epoch {epoch}.")
                break

        lp_training_time = time.time() - start_lp

        # Test linear probe
        linear_head.eval()
        all_lp_preds = []
        all_lp_labels = []

        with torch.no_grad():
            for feats, labels in test_feat_loader:
                feats = feats.to(device)
                logits = linear_head(feats)
                preds = logits.argmax(dim=1)
                all_lp_preds.append(preds.cpu().numpy())
                all_lp_labels.append(labels.numpy())

        lp_y_true = np.concatenate(all_lp_labels, axis=0)
        lp_y_pred = np.concatenate(all_lp_preds, axis=0)
        lp_metrics = compute_metrics(lp_y_true, lp_y_pred)
        lp_conf_mat = get_confusion_matrix(lp_y_true, lp_y_pred, num_classes)

        print(f"\nLinear Probe Test Results (SigLIP2 features):")
        print(f"  Accuracy:  {lp_metrics['accuracy']:.2f}%")
        print(f"  Precision: {lp_metrics['precision']:.2f}%")
        print(f"  Recall:    {lp_metrics['recall']:.2f}%")
        print(f"  F1-Score:  {lp_metrics['f1_score']:.2f}%")

        plot_path_lp = config.PLOTS_DIR / "task3_linear_probe_confusion_matrix.png"
        plot_confusion_matrix(lp_conf_mat, class_names, plot_path_lp)

        linear_probe_results = {
            "training_time": lp_training_time,
            "best_val_accuracy": best_val_acc,
            "test_metrics": lp_metrics,
            "confusion_matrix": lp_conf_mat.tolist(),
            "hyperparameters": {
                "lr": config.LINEAR_PROBE_LR,
                "weight_decay": config.LINEAR_PROBE_WEIGHT_DECAY,
                "epochs": config.LINEAR_PROBE_EPOCHS,
                "early_stopping_patience": config.LINEAR_PROBE_PATIENCE,
                "batch_size": feat_batch_size,
            },
        }

    # --- Save all results ---
    results = {
        "zero_shot": {
            "inference_time": zero_shot_inference_time,
            "test_metrics": zero_shot_metrics,
            "num_prompts_per_class": [len(p) for p in prompts_per_class],
            "prompts": {
                cls: prompts for cls, prompts in zip(class_names, prompts_per_class)
            },
            "confusion_matrix": zero_shot_conf_mat.tolist(),
            "hyperparameters": {
                "model_name": model_name,
                "batch_size": config.BATCH_SIZE,
            },
        },
        "linear_probe": linear_probe_results,
    }

    results_path = config.RESULTS_DIR / "task3_zero_shot_and_linear_probe.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    zero_shot_classification()
