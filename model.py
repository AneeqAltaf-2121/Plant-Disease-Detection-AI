"""Model architecture and transfer learning setup."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model(num_classes: int, freeze_features: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model
