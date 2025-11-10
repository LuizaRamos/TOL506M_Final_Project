import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import json

from config import Config
from data.dataset import get_data_loaders, get_class_names
from utils import compute_metrics

def zero_shot_classification(config: Config = None):
    """
    Perform Zero-Shot classification task using SigLIP2
    :param config: Configuration object
    :return: Dictionary with evaluation metrics
    """

    if config is None:
        config = Config()

    # Load test set data
    _, _, test_loader, num_classes = get_data_loaders(
        data_path=str(config.DATA_DIR / config.DATASET_NAME),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        use_augmentation=False,
        random_seed=config.RANDOM_SEED
    )

    class_names = get_class_names(test_loader.dataset)

    print(f"Classes: {class_names}")

    device = config.DEVICE

    try:
        processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)
        model = AutoModel.from_pretrained(config.SIGLIP_MODEL).to(device)
        model.eval()
        print(f"Model loaded: {config.SIGLIP_MODEL}")
    except Exception as e:
        print(f"Error Loading SigLIP model: {e}")
        print("Falling back to CLIP...")
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()

    text_prompts = []
    for class_name in class_names:
        class_prompts = [template.format(class_name)
                         for template in config.ZERO_SHOT_PROMPTS]
        text_prompts.append(class_prompts)

    all_text_embeddings = []
    for class_prompts in text_prompts:
        inputs = processor(text=class_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                text_features = model.get_text_features(**inputs)
            else:
                text_features = model.get_text_features(**inputs).pooler_output
        text_embeddings = text_features.mean(dim=0)
        all_text_embeddings.append(text_embeddings)

    text_embeddings = torch.stack(all_text_embeddings)

    all_predictions = []
    all_targets = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # Process images
            # It is needed to denormalize and convert back to PIL
            images_pil = []
            for img_tensor in images:
                # Denormalizing
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)

                # Convert back to PIL
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                images_pil.append(Image.fromarray(img_np))

            # Processing with SigLIP or CLIP
            inputs = processor(images=images_pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if k != 'text'}

            # Gets image embeddings
            if hasattr(model, "get_text_features"):
                image_features = model.get_text_features(**inputs)
            else:
                image_features = model.get_text_features(**inputs).pooler_output

            # Compute similarities
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

            similarities = image_features @ text_features_norm.T
            predictions = similarities.argmax(dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    inference_time = time.time() - start_time
    print(f"Inference completed in: {inference_time:.2f} seconds.")

    # Compute results
    metrics = compute_metrics(np.array(all_targets), np.array(all_predictions))

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
        'prompts': config.ZERO_SHOT_PROMPTS
    }

    results_path = config.RESULTS_DIR / f'task3_zero_shot.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    zero_shot_classification()
