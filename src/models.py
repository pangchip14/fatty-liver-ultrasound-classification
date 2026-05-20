from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torchvision import models


def _safe_weights(factory: Callable, weight_enum, pretrained: bool):
    if not pretrained:
        return factory(weights=None)
    try:
        return factory(weights=weight_enum.DEFAULT)
    except Exception as exc:
        print(f"Warning: pretrained weights unavailable, using random init. Reason: {exc}")
        return factory(weights=None)


def create_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = name.lower()

    if name == "resnet18":
        model = _safe_weights(models.resnet18, models.ResNet18_Weights, pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        model = _safe_weights(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights, pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = _safe_weights(models.efficientnet_b0, models.EfficientNet_B0_Weights, pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}")


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return model.layer4[-1]
    if name == "mobilenet_v3_small":
        return model.features[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported model: {name}")
