from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .physics import AcousticScenario, create_scenario, damping_map, simulate


@dataclass(frozen=True)
class ScenarioRun:
    scenario: AcousticScenario
    fields: np.ndarray
    coarse_scenario: AcousticScenario
    coarse_fields: np.ndarray
    seed: int
    ood: bool


class WaveFieldDataset(Dataset):
    def __init__(self, samples: list[dict[str, np.ndarray | float | int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "features": torch.from_numpy(sample["features"]),
            "target": torch.from_numpy(sample["target"]),
            "velocity": torch.from_numpy(sample["velocity"]),
            "damping": torch.from_numpy(sample["damping"]),
            "time_index": torch.tensor(sample["time_index"], dtype=torch.long),
        }


def coordinate_channels(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    return xx, yy


def resize_fields(fields: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(fields.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    else:
        raise ValueError(f"Expected a 2D field or 3D field stack, received {tensor.ndim}D")
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=True)
    return resized[:, 0].numpy()


def coarse_config(config: ExperimentConfig) -> ExperimentConfig:
    coarse_dx = 1.0 / (config.coarse_grid_size - 1)
    matching_cfl = config.dt * (2.0**0.5) * config.max_velocity / coarse_dx
    return replace(
        config,
        grid_size=config.coarse_grid_size,
        cfl=matching_cfl,
        sponge_width=max(3, round(config.sponge_width * config.coarse_grid_size / config.grid_size)),
    )


def coarse_scenario(scenario: AcousticScenario, config: ExperimentConfig) -> AcousticScenario:
    coarse = coarse_config(config)
    velocity = resize_fields(scenario.velocity, coarse.grid_size)[0]
    initial = resize_fields(scenario.initial_pressure, coarse.grid_size)[0]
    initial[[0, -1], :] = 0.0
    initial[:, [0, -1]] = 0.0
    return AcousticScenario(
        velocity=velocity.astype(np.float32),
        initial_pressure=initial.astype(np.float32),
        damping=damping_map(coarse.grid_size, coarse.sponge_width, coarse.sponge_strength),
        description=f"{scenario.description}, coarse grid",
    )


def make_features(
    scenario: AcousticScenario,
    coarse_fields: np.ndarray,
    time_index: int,
    config: ExperimentConfig,
) -> np.ndarray:
    xx, yy = coordinate_channels(config.grid_size)
    coarse_triplet = coarse_fields[time_index - 1 : time_index + 2]
    upsampled_triplet = resize_fields(coarse_triplet, config.grid_size)
    normalized_velocity = (scenario.velocity - 1.0) / 0.6
    time_value = np.full_like(xx, time_index / (config.simulation_steps - 1), dtype=np.float32)
    return np.stack(
        [
            upsampled_triplet[0],
            upsampled_triplet[1],
            upsampled_triplet[2],
            normalized_velocity,
            scenario.initial_pressure,
            xx,
            yy,
            time_value,
        ],
        axis=0,
    ).astype(np.float32)


def scenario_samples(run: ScenarioRun, config: ExperimentConfig) -> list[dict[str, np.ndarray | float | int]]:
    centers = np.linspace(
        2,
        config.simulation_steps - 3,
        config.samples_per_scenario,
    ).round().astype(int)
    samples: list[dict[str, np.ndarray | float | int]] = []
    for center in centers:
        target = run.fields[center - 1 : center + 2]
        samples.append(
            {
                "features": make_features(run.scenario, run.coarse_fields, int(center), config),
                "target": target.astype(np.float32),
                "velocity": run.scenario.velocity[None, ...].astype(np.float32),
                "damping": run.scenario.damping[None, ...].astype(np.float32),
                "time_index": int(center),
            }
        )
    return samples


def build_runs(
    count: int,
    config: ExperimentConfig,
    seed_offset: int,
    ood: bool = False,
) -> list[ScenarioRun]:
    runs = []
    for index in range(count):
        seed = config.seed + seed_offset + index
        scenario = create_scenario(config, seed=seed, ood=ood)
        low_resolution_scenario = coarse_scenario(scenario, config)
        runs.append(
            ScenarioRun(
                scenario,
                simulate(scenario, config),
                low_resolution_scenario,
                simulate(low_resolution_scenario, coarse_config(config)),
                seed,
                ood,
            )
        )
    return runs


def build_dataset(
    count: int,
    config: ExperimentConfig,
    seed_offset: int,
    ood: bool = False,
) -> tuple[WaveFieldDataset, list[ScenarioRun]]:
    runs = build_runs(count, config, seed_offset=seed_offset, ood=ood)
    samples = [sample for run in runs for sample in scenario_samples(run, config)]
    return WaveFieldDataset(samples), runs
