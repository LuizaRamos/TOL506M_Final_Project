import torch
from transformers import AutoProcessor
from transformers import AutoModelForZeroShotImageClassification as autoModel
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

def zero_shot_classification(config: Config = None,
                             fixed_indices: Optional[SplitIndices] = None,
                             test_loader: Optional[DataLoader] = None):
    """
    Perform Zero-Shot classification task using SigLIP2
    :param config: Configuration object
    :param fixed_indices: Fixed indices to use.
    :param test_loader: Test data loader
    :return: Dictionary with evaluation metrics
    """

    if config is None:
        config = Config()

    device = config.DEVICE

    model_name = 'google/siglip-base-patch16-224'
    processor = AutoProcessor.from_pretrained(model_name)
    model = autoModel.from_pretrained(model_name).to(device)

    print(f'Model Loaded: {model_name}')

    if test_loader is None:
        # Load test set data
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
            fixed_indices=fixed_indices
        )
    else:
        num_classes = config.NUM_CLASSES

    class_names = get_class_names(test_loader.dataset)
    print(f"Classes: {class_names}")

    # Generate text features for each class label
    text_inputs = processor(text = class_names, padding = True, return_tensors = 'pt').to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    all_predictions = []
    all_labels = []

    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images, labels = batch['image'], batch['label']
            images = images.to(device)
            all_labels.append(labels.to(device))

            inputs = processor(images=images, return_tensors='pt').to(device)

            if hasattr(model, 'get_text_features'):
                image_features = model.get_text_features(**inputs)
            else:
                raise AttributeError('Model does not have text features.')

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            logits_per_image = (image_features @ text_features.T) * model.config.temperature
            probs = logits_per_image.softmax(dim=-1)
            predictions = probs.argmax(probs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    inference_time = time.time() - start_time

    # Calculate metrics
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    confusion_matrix = get_confusion_matrix(np.array(all_predictions), np.array(all_labels), num_classes)

    # Report results
    print(f"\nZero-Shot Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall: {metrics['recall']:.2f}%")
    print(f"  F1-Score: {metrics['f1_score']:.2f}%")

    # Save results
    results = {
        'inference_time': inference_time,
        'test_metrics': metrics,
        'num_prompts_per_class': len(config.ZERO_SHOT_PROMPTS),
        'prompts': config.ZERO_SHOT_PROMPTS,
        'confusion_matrix': confusion_matrix
    }

    # Confusion Matrix
    plot_path = config.PLOTS_DIR / f"task3_zero_shot_classification.png"
    plot_confusion_matrix(confusion_matrix, class_names, plot_path)

    results_path = config.RESULTS_DIR / f'task3_zero_shot.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    zero_shot_classification()
