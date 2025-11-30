import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_model(num_classes: int,
                         model_name: str = 'resnet18',
                         freeze_until_layer: str = None):
    """
    Load a pretrained model and modify for target task.

    :param num_classes: Number of output classes.
    :param model_name: Name of the pretrained model.
    :param freeze_until_layer: Freeze layers until a certain layer
            (None, 'all_but_fc', 'up_to_layer1', 'up_to_layer2', 'up_to_layer3').
    :return: Modified pretrained model.
    """

    # Load pretrained model
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    # Freeze layers according to strategy
    freeze_backbone(model, freeze_until_layer)

    # Replace final layer
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, num_classes)

    return model

def freeze_backbone(model: nn.Module, freeze_mode: str):
    """
    freeze_mode options:
        - None: train everything
        - 'all_but_fc': freeze entire backbone, train only final classifier
        - 'up_to_layer1': freeze conv1, bn1, layer1
        - 'up_to_layer2': freeze conv1, bn1, layer1, layer2
        - 'up_to_layer3': freeze conv1, bn1, layer1, layer2, layer3
    """

    if freeze_mode is None:
        # Unfreeze everything
        for p in model.parameters():
            p.requires_grad = True
        return

    if freeze_mode in ['all_but_fc']:
        freeze_modules = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']

    elif freeze_mode == 'up_to_layer1':
        freeze_modules = ['conv1', 'bn1', 'layer1']

    elif freeze_mode == 'up_to_layer2':
        freeze_modules = ['conv1', 'bn1', 'layer1', 'layer2']

    elif freeze_mode == 'up_to_layer3':
        freeze_modules = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']

    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    # Apply freezing
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in freeze_modules):
            param.requires_grad = False

