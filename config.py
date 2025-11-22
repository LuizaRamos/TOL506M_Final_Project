""" Configuration file for the wildlife classification """

import torch
from pathlib import Path

class Config:
    """ Configuration class for the classification """

    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_PATH = PROJECT_ROOT / 'data' # place holder
    DATA_DIR = PROJECT_ROOT / 'data'
    PROCESSED_DIR = DATA_DIR / 'processed'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    MODELS_DIR = RESULTS_DIR / 'models'
    PLOTS_DIR = RESULTS_DIR / 'plots'
    METRICS_DIR = RESULTS_DIR / 'metrics'

    # Create directories
    for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Dataset configuration
    DATASET_NAME = "wildlife" # Animals10
    NUM_CLASSES = 10
    IMAGE_SIZE = 224

    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Training configuration
    BATCH_SIZE = 64
    NUM_WORKERS = 2      # Amount recommended by Google Colab
    PIN_MEMORY = True
    NUM_EPOCHS = 10

    # Task 1: Training from Scratch
    SCRATCH_EPOCHS = 100
    SCRATCH_LR = 0.1
    SCRATCH_MOMENTUM = 0.9
    SCRATCH_WEIGHT_DECAY = 1e-4
    SCRATCH_EARLY_STOPPING_PATIENCE = 10

    # Task 2: Fine-tuning
    FINETUNE_EPOCHS = 50
    FINETUNE_LR = 1e-3
    FINETUNE_WEIGHT_DECAY = 1e-4
    FINETUNE_EARLY_STOPPING_PATIENCE = 10
    FREEZE_LAYERS = ['all_but_fc', 'up_to_layer1', 'up_to_layer2', 'up_to_layer3']

    # Learning rate scheduling
    LR_SCHEDULER = 'cosine' # other options to test: 'step' or 'cosine'
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1

    # Data augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5
    COLOR_JITTER = True
    RANDOM_CROP = True

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    RANDOM_SEED = 7278

    # Learning curves
    DATA_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

    # Zero-Shot configuration
    SIGLIP_MODEL = "google/siglip2-base-patch16-224"

    SIGLIP_LINEAR_PROBE = False  # default: pure zero-shot
    LINEAR_PROBE_LR = 1e-3
    LINEAR_PROBE_WEIGHT_DECAY = 1e-4
    LINEAR_PROBE_EPOCHS = 50
    LINEAR_PROBE_PATIENCE = 10

    ZERO_SHOT_PROMPTS = {
        "dog": [
            "a photo of a dog",
            "a photo of a pet dog",
            "a photo of a dog running outdoors",
            "a close-up photo of a dog",
        ],
        "spider": [
            "a close-up photo of a spider",
            "a photo of a spider on a web",
            "a photo of a spider in nature",
            "a macro photograph of a spider",
        ],
        "chicken": [
            "a photo of a chicken",
            "a photo of a chicken on a farm",
            "a photo of a hen or rooster",
            "a close-up photo of a chicken",
        ],
        "horse": [
            "a photo of a horse",
            "a photo of a horse running in a field",
            "a photo of a horse on a farm",
            "a close-up photo of a horse's head",
        ],
        "butterfly": [
            "a close-up photo of a butterfly",
            "a photo of a butterfly on a flower",
            "a photo of a colorful butterfly in nature",
            "a macro photograph of a butterfly",
        ],
        "cow": [
            "a photo of a cow",
            "a photo of a cow on a farm",
            "a photo of a cow grazing in a field",
            "a close-up photo of a cow",
        ],
        "squirrel": [
            "a photo of a squirrel",
            "a photo of a squirrel on a tree",
            "a photo of a squirrel eating a nut",
            "a close-up photo of a squirrel in a park",
        ],
        "sheep": [
            "a photo of a sheep",
            "a photo of sheep grazing in a field",
            "a photo of a flock of sheep",
            "a close-up photo of a sheep on a farm",
        ],
        "cat": [
            "a photo of a cat",
            "a photo of a pet cat indoors",
            "a photo of a cat lying on a couch",
            "a close-up photo of a cat's face",
        ],
        "elephant": [
            "a photo of an elephant",
            "a photo of an elephant in the wild",
            "a photo of an elephant in a savanna",
            "a close-up photo of an elephant",
        ],
    }

    # Logging
    LOG_INTERVAL = 10
    SAVE_CHECKPOINTS = True
