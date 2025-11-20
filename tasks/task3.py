import torch
from transformers import AutoProcessor, SiglipModel
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import time
import json
from typing import Optional
from torch.utils.data import DataLoader

from config import Config
from data.dataset import SplitIndices, get_data_loaders, get_class_names
from utils.evaluation import compute_metrics, get_confusion_matrix
from utils.visualization import plot_confusion_matrix


def zero_shot_classification(
    config: Config = None,
    fixed_indices: Optional[SplitIndices] = None,
    test_loader: Optional[DataLoader] = None,
):
    """
    Perform Zero-Shot classification task using SigLIP
    :param config: Configuration object
    :param fixed_indices: Fixed indices to use.
    :param test_loader: Test data loader (should return images in [0, 1] without normalization)
    :return: Dictionary with evaluation metrics
    """

    if config is None:
        config = Config()

    device = config.DEVICE

    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name).to(device)

    print(f"Model Loaded: {model_name}")

    # If no loader is provided, fall back to project helper.
    # IMPORTANT: for best results in this task, pass a test_loader that uses
    # only Resize + ToTensor (no Normalize), as you did in the notebook.
    if test_loader is None:
        _, _, test_loader, num_classes = get_data_loaders(
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
    else:
        num_classes = config.NUM_CLASSES

    # Class names used as text prompts
    class_names = get_class_names(test_loader.dataset)
    print(f"Classes: {class_names}")

    # text encoding
    text_inputs = processor(
        text=class_names,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # image encoding and zero-shot predictions
    all_predictions = []
    all_labels = []

    start_time = time.time()
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # images: batch of tensors in [0, 1] (Resize + ToTensor, no Normalize)
            images = images.to(device)
            all_labels.append(labels.cpu().numpy())

            # Let the SigLIP processor handle resizing/normalization; avoid double rescale
            inputs = processor(
                images=list(images),       # list of tensors
                return_tensors="pt",
                do_rescale=False,          # already in [0, 1]
            ).to(device)

            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

            # Cosine similarity logits
            if hasattr(model, "logit_scale"):
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.T)
            else:
                logits_per_image = image_features @ text_features.T

            probs = logits_per_image.softmax(dim=-1)
            preds = probs.argmax(dim=-1)

            all_predictions.append(preds.cpu().numpy())

    inference_time = time.time() - start_time

    # metrics
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)

    # compute_metrics expects (targets, predictions)
    metrics = compute_metrics(y_true, y_pred)
    confusion_matrix = get_confusion_matrix(y_true, y_pred, num_classes)

    # reporting
    print(f"\nZero-Shot Test Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']:.2f}%")

    # Save results
    results = {
        "inference_time": inference_time,
        "test_metrics": metrics,
        "num_prompts_per_class": len(config.ZERO_SHOT_PROMPTS),
        "prompts": config.ZERO_SHOT_PROMPTS,
        "confusion_matrix": confusion_matrix.tolist(),  # make JSON-serializable
    }

    # Confusion Matrix plot
    plot_path = config.PLOTS_DIR / "task3_zero_shot_classification.png"
    plot_confusion_matrix(confusion_matrix, class_names, plot_path)

    results_path = config.RESULTS_DIR / "task3_zero_shot.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    zero_shot_classification()
