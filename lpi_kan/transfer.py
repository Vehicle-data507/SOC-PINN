"""Layer-freezing strategies used for target-vehicle adaptation."""

from __future__ import annotations

import copy
from typing import Literal

from torch import nn

from .model import EnhancedPhysicsInformedEfficientKAN, count_parameters

FreezeStrategy = Literal["output_only", "last_two", "none"]


def create_transfer_model(
    pretrained_model: EnhancedPhysicsInformedEfficientKAN,
    strategy: FreezeStrategy,
) -> EnhancedPhysicsInformedEfficientKAN:
    model = copy.deepcopy(pretrained_model)
    if strategy not in {"output_only", "last_two", "none"}:
        raise ValueError(f"Unsupported transfer strategy: {strategy}")

    for parameter in model.parameters():
        parameter.requires_grad = strategy == "none"

    if strategy != "none":
        layers_to_train = 1 if strategy == "output_only" else 2
        for layer in model.kan.layers[-layers_to_train:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        if isinstance(model.capacity, nn.Parameter):
            model.capacity.requires_grad = True

    return model


def transfer_parameter_summary(model: nn.Module) -> dict[str, int]:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
