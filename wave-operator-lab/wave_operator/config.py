from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 17
    grid_size: int = 32
    coarse_grid_size: int = 16
    simulation_steps: int = 72
    train_scenarios: int = 28
    validation_scenarios: int = 6
    test_scenarios: int = 6
    ood_scenarios: int = 6
    samples_per_scenario: int = 8
    cfl: float = 0.34
    max_velocity: float = 1.7
    sponge_width: int = 5
    sponge_strength: float = 0.075
    fno_width: int = 20
    fno_modes: int = 8
    fno_layers: int = 4
    batch_size: int = 14
    epochs: int = 24
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    physics_weight: float = 0.2
    gradient_weight: float = 0.05
    early_stopping_patience: int = 7
    device: str = "auto"

    @property
    def dx(self) -> float:
        return 1.0 / (self.grid_size - 1)

    @property
    def dt(self) -> float:
        return self.cfl * self.dx / ((2.0**0.5) * self.max_velocity)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(**data)
