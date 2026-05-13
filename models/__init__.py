"""Model registry. Add a new model by importing it and adding to MODEL_REGISTRY."""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

from .lenet import LeNet
from .resnet18 import ResNet18, BasicBlock, ResNet
from .se_resnet18 import SEResNet18


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "lenet": LeNet,
    "resnet18": ResNet18,
    "se_resnet18": SEResNet18,
}


def build_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[key](in_channels=in_channels, num_classes=num_classes)


__all__ = [
    "LeNet",
    "ResNet18",
    "SEResNet18",
    "ResNet",
    "BasicBlock",
    "MODEL_REGISTRY",
    "build_model",
]
