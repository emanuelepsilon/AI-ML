from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import ExperimentConfig


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor
    gradient: torch.Tensor


def laplacian_torch(field: torch.Tensor, dx: float) -> torch.Tensor:
    result = torch.zeros_like(field)
    result[..., 1:-1, 1:-1] = (
        field[..., 1:-1, 2:]
        + field[..., 1:-1, :-2]
        + field[..., 2:, 1:-1]
        + field[..., :-2, 1:-1]
        - 4.0 * field[..., 1:-1, 1:-1]
    ) / (dx * dx)
    return result


def spatial_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    true_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_y = target[..., 1:, :] - target[..., :-1, :]
    return F.mse_loss(pred_x, true_x) + F.mse_loss(pred_y, true_y)


def physics_consistency_loss(
    prediction: torch.Tensor,
    velocity: torch.Tensor,
    damping: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    previous = prediction[:, 0:1]
    current = prediction[:, 1:2]
    next_field = prediction[:, 2:3]
    expected_next = (
        (2.0 - damping) * current
        - (1.0 - damping) * previous
        + config.dt**2 * velocity**2 * laplacian_torch(current, config.dx)
    )
    return F.mse_loss(next_field[..., 2:-2, 2:-2], expected_next[..., 2:-2, 2:-2])


def operator_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    velocity: torch.Tensor,
    damping: torch.Tensor,
    config: ExperimentConfig,
) -> LossBreakdown:
    amplitude = target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-4)
    weights = 1.0 + 2.0 * target.abs() / amplitude
    data_loss = torch.mean(weights * (prediction - target) ** 2)
    physics_loss = physics_consistency_loss(prediction, velocity, damping, config)
    gradient_loss = spatial_gradient_loss(prediction, target)
    total = (
        data_loss
        + config.physics_weight * physics_loss
        + config.gradient_weight * gradient_loss
    )
    return LossBreakdown(total, data_loss, physics_loss, gradient_loss)
