from torchvision import transforms

def get_train_transforms(image_size: int = 224, use_augmentation: bool = True):
    """
    Get training data transforms with augmentation.
    :param image_size: Target image size.
    :param use_augmentation: Whether to use augmentation.
    :return: composed transforms
    """
    if use_augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def get_val_transforms(image_size: int = 224):
    """
    Get validation data transforms with augmentation.
    :param image_size: Target image size.
    :return: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])