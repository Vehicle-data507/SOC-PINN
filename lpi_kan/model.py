"""Core EfficientKAN and physics-informed SOC model used in the study.

The KAN layer is adapted from Blealtan/efficient-kan (MIT License).
The required license notice is retained at the end of this source file.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class PhysicsLosses(NamedTuple):
    coulomb: torch.Tensor
    boundary: torch.Tensor
    monotonic: torch.Tensor


class EfficientKANLinear(nn.Module):
    """Efficient KAN layer with a residual base path and B-spline basis."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * step
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                torch.rand(
                    self.grid_size + 1,
                    self.in_features,
                    self.out_features,
                )
                - 0.5
            ) * self.scale_noise / self.grid_size
            scale = 1.0 if self.enable_standalone_scale_spline else self.scale_spline
            coefficients = self.curve_to_coefficients(
                self.grid.T[self.spline_order : -self.spline_order], noise
            )
            self.spline_weight.copy_(scale * coefficients)

        if self.enable_standalone_scale_spline:
            nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5) * self.scale_spline
            )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected [batch, {self.in_features}], received {tuple(x.shape)}"
            )

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left = (
                (x - grid[:, : -(order + 1)])
                / (grid[:, order:-1] - grid[:, : -(order + 1)] + 1e-8)
                * bases[:, :, :-1]
            )
            right = (
                (grid[:, order + 1 :] - x)
                / (grid[:, order + 1 :] - grid[:, 1:-order] + 1e-8)
                * bases[:, :, 1:]
            )
            bases = left + right
        return bases.contiguous()

    def curve_to_coefficients(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("Invalid spline-grid shape")
        expected = (x.shape[0], self.in_features, self.out_features)
        if tuple(y.shape) != expected:
            raise ValueError(f"Expected curve samples {expected}, received {tuple(y.shape)}")

        design = self.b_splines(x).transpose(0, 1)
        targets = y.transpose(0, 1)
        solution = torch.linalg.lstsq(design, targets).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if not self.enable_standalone_scale_spline:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_features = self.b_splines(x).reshape(x.shape[0], -1)
        spline_weights = self.scaled_spline_weight.reshape(self.out_features, -1)
        return base_output + F.linear(spline_features, spline_weights)


class EfficientKAN(nn.Module):
    """Stack of EfficientKANLinear layers."""

    def __init__(
        self,
        layer_widths: Sequence[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
    ) -> None:
        super().__init__()
        if len(layer_widths) < 2:
            raise ValueError("layer_widths must contain at least input and output sizes")
        self.layers = nn.ModuleList(
            EfficientKANLinear(
                in_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
            )
            for in_features, out_features in zip(layer_widths, layer_widths[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class EnhancedPhysicsInformedEfficientKAN(nn.Module):
    """Direct multi-output SOC predictor with three physics constraints."""

    def __init__(
        self,
        input_size: int = 12,
        window_size: int = 30,
        output_size: int = 30,
        hidden_dims: Sequence[int] = (8, 8),
        grid_size: int = 5,
        spline_order: int = 3,
        capacity_init: float = 142.5,
        learnable_capacity: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.output_size = output_size

        widths = [input_size * window_size, *hidden_dims, output_size]
        self.kan = EfficientKAN(
            widths, grid_size=grid_size, spline_order=spline_order
        )

        capacity = torch.tensor(capacity_init, dtype=torch.float32)
        if learnable_capacity:
            self.capacity = nn.Parameter(capacity)
        else:
            self.register_buffer("capacity", capacity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.window_size, self.input_size)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}], "
                f"received {tuple(x.shape)}"
            )
        return self.kan(x.reshape(x.shape[0], -1))

    def physics_loss(
        self,
        soc_prediction: torch.Tensor,
        future_current: torch.Tensor,
        dt: float = 10.0,
    ) -> PhysicsLosses:
        """Return Coulomb-counting, SOC-boundary, and monotonicity losses."""
        if soc_prediction.shape != future_current.shape:
            raise ValueError("soc_prediction and future_current must have equal shapes")

        delta_soc_rate = (
            soc_prediction[:, 1:] - soc_prediction[:, :-1]
        ) / dt
        average_current = (
            future_current[:, 1:] + future_current[:, :-1]
        ) / 2.0
        theoretical_rate = -average_current / (self.capacity * 3600.0) * 100.0
        coulomb = torch.mean((delta_soc_rate - theoretical_rate) ** 2)

        boundary = torch.mean(
            torch.relu(-soc_prediction) + torch.relu(soc_prediction - 100.0)
        )

        delta_soc = soc_prediction[:, 1:] - soc_prediction[:, :-1]
        monotonic_violation = torch.relu(delta_soc * torch.sign(average_current))
        monotonic = torch.mean(monotonic_violation**2)
        return PhysicsLosses(coulomb, boundary, monotonic)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = (
        parameter
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
    return sum(parameter.numel() for parameter in parameters)


# Third-party license notice for the adapted B-spline KAN layer:
#
# MIT License
#
# Copyright (c) 2024 Huanqi Cao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
