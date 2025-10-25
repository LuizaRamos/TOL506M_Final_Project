import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import json

from config import Config
from data.dataset import get_data_loaders
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_curves

def zero_shot_classification(config: Config = None):
    """
    Perform Zero-Shot classification task using SigLIP2
    :param config: Configuration object
    :return: Dictionary with evaluation metrics
    """