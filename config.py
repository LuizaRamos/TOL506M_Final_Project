""" Configuration file for the wildlife classification """

import torch
from pathlib import Path

class Config:
    """ Configuration class for the classification """

    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    MODELS_DIR = PROJECT_ROOT / 'models'
    PLOTS_DIR = PROJECT_ROOT / 'plots'
    METRICS_DIR = PROJECT_ROOT / 'metrics'

    # Create directories
    for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Dataset configuration
    DATASET_NAME = "animals10" # Others: "oxford_pet", "animal10n"
    NUM_CLASSES = 10
    IMAGE_SIZE = 224

    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Training configuration
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Task 1: Training from Scratch
    SCRATCH_EPOCHS = 100
    SCRATCH_LR = 1e-3 # maybe 0.01?
    SCRATCH_MOMENTUM = 0.9
    SCRATCH_WEIGHT_DECAY = 1e-4
    SCRATCH_EARLY_STOPPING_PATIENCE = 15

    # Task 2: Fine-tuning
    FINETUNE_EPOCHS = 50
    FINETUNE_LR = 1e-3
    FINETUNE_WEIGHT_DECAY = 1e-4
    FINETUNE_EARLY_STOPPING_PATIENCE = 10
    FREEZE_UNTIL_LAYER = None

    # Learning rate scheduling
    LR_SCHEDULER = 'cosine' # other options to test: 'step', 'cosine' 'plateu'
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
    DATA_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Zero-Shot configuration
    SIGLIP_MODEL = "google/siglip-base-patch16-224"
    ZERO_SHOT_PROMPTS = [
        "a photo of a {}",
        "an image of a {}",
        "a picture of a {}"
    ]

    # Logging
    LOG_INTERVAL = 10
    SAVE_CHECKPOINTS = True