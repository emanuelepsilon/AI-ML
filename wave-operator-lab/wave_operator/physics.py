from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


@dataclass(frozen=True)
class AcousticScenario:
    velocity: np.ndarray
    initial_pressure: np.ndarray
    damping: np.ndarray
    description: str


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    result = np.zeros_like(field)
    result[1:-1, 1:-1] = (
        field[1:-1, 2:]
        + field[1:-1, :-2]
        + field[2:, 1:-1]
        + field[:-2, 1:-1]
        - 4.0 * field[1:-1, 1:-1]
    ) / (dx * dx)
    return result


def damping_map(size: int, width: int, strength: float) -> np.ndarray:
    index = np.arange(size)
    distance = np.minimum(index, size - 1 - index)
    edge = np.clip((width - distance) / max(width, 1), 0.0, 1.0) ** 2
    damping = strength * (edge[:, None] + edge[None, :])
    return np.clip(damping, 0.0, 0.18).astype(np.float32)


def _gaussian(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, sigma: float) -> np.ndarray:
    radius = ((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)
    return np.exp(-radius)


def create_scenario(config: ExperimentConfig, seed: int, ood: bool = False) -> AcousticScenario:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, config.grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)

    background = rng.uniform(0.92, 1.08)
    velocity = np.full_like(xx, background, dtype=np.float32)
    inclusion_count = int(rng.integers(2, 4)) if ood else int(rng.integers(1, 3))

    for _ in range(inclusion_count):
        cx, cy = rng.uniform(0.22, 0.78, size=2)
        sigma_x = rng.uniform(0.055, 0.13 if not ood else 0.18)
        sigma_y = rng.uniform(0.055, 0.13 if not ood else 0.18)
        contrast_limit = 0.34 if not ood else 0.52
        contrast = rng.uniform(-contrast_limit, contrast_limit)
        blob = np.exp(-(((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2) / 2.0)
        velocity += contrast * blob.astype(np.float32)

    if ood:
        angle = rng.uniform(-0.8, 0.8)
        offset = rng.uniform(0.38, 0.62)
        signed_distance = (xx - offset) * np.cos(angle) + (yy - 0.5) * np.sin(angle)
        layer = np.abs(signed_distance) < rng.uniform(0.025, 0.055)
        velocity[layer] += rng.choice([-0.28, 0.32])

    velocity = np.clip(velocity, 0.55, config.max_velocity).astype(np.float32)
    source_x, source_y = rng.uniform(0.20, 0.80, size=2)
    source_width = rng.uniform(0.035, 0.065)
    initial = _gaussian(xx, yy, source_x, source_y, source_width).astype(np.float32)
    initial -= 0.32 * _gaussian(xx, yy, source_x, source_y, source_width * 1.9).astype(np.float32)
    initial[[0, -1], :] = 0.0
    initial[:, [0, -1]] = 0.0

    if ood:
        boundary_width = int(
            rng.choice(
                [
                    max(2, config.sponge_width - 2),
                    min(config.grid_size // 3, config.sponge_width + 3),
                ]
            )
        )
        boundary_strength = config.sponge_strength * float(rng.choice([0.55, 1.55]))
        boundary_kind = "unseen absorbing boundary profile"
    else:
        boundary_width = int(
            np.clip(
                config.sponge_width + rng.integers(-1, 2),
                2,
                config.grid_size // 3,
            )
        )
        boundary_strength = config.sponge_strength * rng.uniform(0.85, 1.15)
        boundary_kind = "variable absorbing boundary profile"

    medium_kind = "layered multi-inclusion OOD medium" if ood else "smooth heterogeneous medium"
    return AcousticScenario(
        velocity=velocity,
        initial_pressure=initial,
        damping=damping_map(config.grid_size, boundary_width, boundary_strength),
        description=f"{medium_kind}, {boundary_kind}",
    )


def simulate(scenario: AcousticScenario, config: ExperimentConfig) -> np.ndarray:
    dt2 = config.dt * config.dt
    pressure_0 = scenario.initial_pressure.astype(np.float32).copy()
    acceleration_0 = scenario.velocity**2 * laplacian(pressure_0, config.dx)
    pressure_1 = pressure_0 + 0.5 * dt2 * acceleration_0
    pressure_1[[0, -1], :] = 0.0
    pressure_1[:, [0, -1]] = 0.0

    fields = np.zeros(
        (config.simulation_steps, config.grid_size, config.grid_size),
        dtype=np.float32,
    )
    fields[0] = pressure_0
    fields[1] = pressure_1
    previous, current = pressure_0, pressure_1

    for step in range(2, config.simulation_steps):
        next_field = (
            (2.0 - scenario.damping) * current
            - (1.0 - scenario.damping) * previous
            + dt2 * scenario.velocity**2 * laplacian(current, config.dx)
        )
        next_field[[0, -1], :] = 0.0
        next_field[:, [0, -1]] = 0.0
        fields[step] = next_field
        previous, current = current, next_field

    return fields


def discrete_energy(
    previous: np.ndarray,
    current: np.ndarray,
    velocity: np.ndarray,
    config: ExperimentConfig,
) -> float:
    time_derivative = (current - previous) / config.dt
    grad_y, grad_x = np.gradient(current, config.dx)
    density = time_derivative**2 + velocity**2 * (grad_x**2 + grad_y**2)
    return float(0.5 * density.sum() * config.dx * config.dx)
