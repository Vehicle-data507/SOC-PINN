"""Public core implementation of the LPI-KAN SOC predictor."""

from .data_protocol import SOCWindowDataset, create_sliding_windows, split_adaptation_files
from .model import (
    EfficientKAN,
    EfficientKANLinear,
    EnhancedPhysicsInformedEfficientKAN,
    PhysicsLosses,
    count_parameters,
)
from .trainer import LossWeights, SOCTrainer, TrainingConfig, compose_loss, set_seed
from .transfer import create_transfer_model, transfer_parameter_summary

__all__ = [
    "EfficientKAN",
    "EfficientKANLinear",
    "EnhancedPhysicsInformedEfficientKAN",
    "LossWeights",
    "PhysicsLosses",
    "SOCTrainer",
    "SOCWindowDataset",
    "TrainingConfig",
    "compose_loss",
    "count_parameters",
    "create_sliding_windows",
    "create_transfer_model",
    "set_seed",
    "split_adaptation_files",
    "transfer_parameter_summary",
]
