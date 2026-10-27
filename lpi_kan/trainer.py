"""Compact training framework for the released LPI-KAN core model."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import EnhancedPhysicsInformedEfficientKAN, PhysicsLosses


PHYSICS_LOSS_SCALE = 1.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class LossWeights:
    """Weights used by the source-domain experiment."""

    coulomb: float = 50.0
    boundary: float = 0.01
    monotonic: float = 0.01


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dt: float = 10.0
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 50
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    minimum_improvement: float = 1e-6
    seed: int = 42
    loss_weights: LossWeights = field(default_factory=LossWeights)


def compose_loss(
    data_loss: torch.Tensor,
    physics_losses: PhysicsLosses,
    weights: LossWeights,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_physics = (
        weights.coulomb * physics_losses.coulomb
        + weights.boundary * physics_losses.boundary
        + weights.monotonic * physics_losses.monotonic
    )
    total = data_loss + PHYSICS_LOSS_SCALE * weighted_physics
    return total, weighted_physics


class SOCTrainer:
    """AdamW trainer with validation selection and early stopping."""

    def __init__(
        self,
        model: EnhancedPhysicsInformedEfficientKAN,
        device: str | torch.device | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        set_seed(self.config.seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.history: list[dict[str, float]] = []

    def _run_epoch(
        self,
        loader: Iterable,
        optimizer: AdamW | None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        totals = {
            "loss": 0.0,
            "data": 0.0,
            "physics": 0.0,
            "coulomb": 0.0,
            "boundary": 0.0,
            "monotonic": 0.0,
        }
        batches = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError(
                        "Each batch must be (input_window, target_soc, future_current)"
                    )
                inputs, targets, future_current = (
                    tensor.to(self.device) for tensor in batch
                )

                if training:
                    optimizer.zero_grad()

                predictions = self.model(inputs)
                data_loss = nn.functional.mse_loss(predictions, targets)
                physics_losses = self.model.physics_loss(
                    predictions, future_current, dt=self.config.dt
                )
                total_loss, weighted_physics = compose_loss(
                    data_loss, physics_losses, self.config.loss_weights
                )

                if training:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    optimizer.step()

                totals["loss"] += total_loss.item()
                totals["data"] += data_loss.item()
                totals["physics"] += weighted_physics.item()
                totals["coulomb"] += physics_losses.coulomb.item()
                totals["boundary"] += physics_losses.boundary.item()
                totals["monotonic"] += physics_losses.monotonic.item()
                batches += 1

        if batches == 0:
            raise ValueError("The data loader is empty")
        return {name: value / batches for name, value in totals.items()}

    def fit(
        self,
        train_loader: Iterable,
        validation_loader: Iterable,
        checkpoint_path: str | Path | None = None,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
        )

        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        stale_epochs = 0

        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(train_loader, optimizer)
            validation_metrics = self._run_epoch(validation_loader, None)
            scheduler.step(validation_metrics["loss"])

            record = {
                "epoch": float(epoch + 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in validation_metrics.items()},
            }
            self.history.append(record)

            improved = (
                validation_metrics["loss"]
                < best_loss - self.config.minimum_improvement
            )
            if improved:
                best_loss = validation_metrics["loss"]
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    self.save_checkpoint(
                        checkpoint_path, best_epoch, best_loss, optimizer
                    )
            else:
                stale_epochs += 1

            if verbose:
                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"train={train_metrics['loss']:.6f} | "
                    f"validation={validation_metrics['loss']:.6f} | "
                    f"capacity={self.model.capacity.item():.3f} Ah"
                )

            if stale_epochs >= self.config.early_stopping_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        validation_loss: float,
        optimizer: AdamW | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": self.model.state_dict(),
            "training_config": asdict(self.config),
            "history": self.history,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)

    @torch.no_grad()
    def evaluate(self, loader: Iterable) -> dict[str, float]:
        self.model.eval()
        predictions = []
        targets = []
        for batch in loader:
            inputs, target, _ = batch
            predictions.append(self.model(inputs.to(self.device)).cpu())
            targets.append(target.cpu())

        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        error = prediction - target
        mse = torch.mean(error**2).item()
        mae = torch.mean(torch.abs(error)).item()
        denominator = torch.sum((target - torch.mean(target)) ** 2)
        r2 = 1.0 - (torch.sum(error**2) / denominator).item()
        return {"mse": mse, "rmse": mse**0.5, "mae": mae, "r2": r2}
