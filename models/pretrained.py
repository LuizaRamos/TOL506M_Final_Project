import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_model(num_classes: int,
                         model_name: str = 'resnet18',
                         freeze_until_layer: int = None):
    """
    Load a pretrained model and modify for target task.

    :param num_classes: Number of output classes.
    :param model_name: Name of the pretrained model.
    :param freeze_until_layer: Freeze layers until this index, None indicate to train all.
    :return: Modified pretrained model.
    """

    # Load pretrained model
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    # Freeze layer if freeze_until_layer != None
    if freeze_until_layer is not None:
        freeze_layer(model, freeze_until_layer)

    # Replace predicted classification layer
    if hasattr(model, 'fc'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    return model

def freeze_layer(model: nn.Module, freeze_until_layer: int):
    """
    Freeze model layers for transfer learning.

    :param model: Model to freeze.
    :param freeze_until_layer: Freeze layers until this index.
    """

    if freeze_until_layer is None:
        # Freeze all layers, except the final classifier one.
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    else:
        # Freeze specific layer
        layer_count = 0
        for name, param in model.named_parameters():
            if layer_count < freeze_until_layer:
                param.requires_grad = False
            layer_count += 1

def unfreeze_layers(model):
    """ Unfreeze layers until they are frozen. """
    for param in model.parameters():
        param.requires_grad = True