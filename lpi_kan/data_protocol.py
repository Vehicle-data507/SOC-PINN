"""Data-shape helpers; no vehicle data are distributed with this package."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def split_adaptation_files(
    adaptation_files: Sequence[str | Path],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Perform the file-level 80/20 adaptation split used in the study."""
    files = sorted((Path(path) for path in adaptation_files), key=str)
    train_files, validation_files = train_test_split(
        files, test_size=validation_fraction, random_state=seed
    )
    return list(train_files), list(validation_files)


def create_sliding_windows(
    features: np.ndarray,
    soc: np.ndarray,
    current: np.ndarray,
    input_steps: int = 30,
    prediction_steps: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create direct multi-output samples within one complete drive segment."""
    features = np.asarray(features, dtype=np.float32)
    soc = np.asarray(soc, dtype=np.float32).reshape(-1)
    current = np.asarray(current, dtype=np.float32).reshape(-1)

    if features.ndim != 2:
        raise ValueError("features must have shape [time, feature]")
    if not (len(features) == len(soc) == len(current)):
        raise ValueError("features, soc, and current must have equal time lengths")
    if stride < 1:
        raise ValueError("stride must be positive")

    total_steps = input_steps + prediction_steps
    if len(features) < total_steps:
        raise ValueError(f"A segment requires at least {total_steps} valid records")

    input_windows = []
    target_windows = []
    current_windows = []
    for start in range(0, len(features) - total_steps + 1, stride):
        split = start + input_steps
        end = split + prediction_steps
        input_windows.append(features[start:split])
        target_windows.append(soc[split:end])
        current_windows.append(current[split:end])

    return (
        np.stack(input_windows),
        np.stack(target_windows),
        np.stack(current_windows),
    )


class SOCWindowDataset(Dataset):
    """Dataset yielding input, target SOC, and future-current sequences."""

    def __init__(
        self,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        future_current: np.ndarray | torch.Tensor,
    ) -> None:
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.future_current = torch.as_tensor(future_current, dtype=torch.float32)
        if not (
            len(self.inputs) == len(self.targets) == len(self.future_current)
        ):
            raise ValueError("All dataset arrays must contain the same sample count")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return (
            self.inputs[index],
            self.targets[index],
            self.future_current[index],
        )
