from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .losses import physics_consistency_loss
from .data import ScenarioRun, coarse_config, make_features, resize_fields
from .physics import simulate


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-8))


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_flat = prediction.ravel()
    target_flat = target.ravel()
    if pred_flat.std() < 1.0e-9 or target_flat.std() < 1.0e-9:
        return 0.0
    return float(np.corrcoef(pred_flat, target_flat)[0, 1])


def _spectral_error(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_spectrum = np.abs(np.fft.rfft2(prediction))
    target_spectrum = np.abs(np.fft.rfft2(target))
    return _relative_l2(pred_spectrum, target_spectrum)


def _energy_proxy(
    previous: np.ndarray,
    current: np.ndarray,
    velocity: np.ndarray,
    config: ExperimentConfig,
) -> float:
    time_derivative = (current - previous) / config.dt
    grad_y, grad_x = np.gradient(current, config.dx)
    density = time_derivative**2 + velocity**2 * (grad_x**2 + grad_y**2)
    return float(0.5 * density.sum() * config.dx * config.dx)


@torch.inference_mode()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.eval()
    rows: list[dict[str, float]] = []
    physics_values: list[float] = []

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        velocity = batch["velocity"].to(device)
        damping = batch["damping"].to(device)
        prediction = model(features)
        physics_values.append(
            float(
                physics_consistency_loss(prediction, velocity, damping, config).cpu()
            )
        )

        prediction_all = prediction.cpu().numpy()
        target_all = target.cpu().numpy()
        pred_np = prediction_all[:, 1]
        target_np = target_all[:, 1]
        velocity_np = velocity[:, 0].cpu().numpy()
        times = batch["time_index"].cpu().numpy()
        for index, (predicted, expected, time_index) in enumerate(zip(pred_np, target_np, times)):
            predicted_energy = _energy_proxy(
                prediction_all[index, 0],
                prediction_all[index, 1],
                velocity_np[index],
                config,
            )
            expected_energy = _energy_proxy(
                target_all[index, 0],
                target_all[index, 1],
                velocity_np[index],
                config,
            )
            rows.append(
                {
                    "time_index": float(time_index),
                    "rmse": float(np.sqrt(np.mean((predicted - expected) ** 2))),
                    "relative_l2": _relative_l2(predicted, expected),
                    "correlation": _correlation(predicted, expected),
                    "spectral_error": _spectral_error(predicted, expected),
                    "energy_relative_error": abs(predicted_energy - expected_energy)
                    / max(expected_energy, 1.0e-8),
                }
            )

    summary = {
        "rmse": float(np.mean([row["rmse"] for row in rows])),
        "relative_l2": float(np.mean([row["relative_l2"] for row in rows])),
        "correlation": float(np.mean([row["correlation"] for row in rows])),
        "spectral_error": float(np.mean([row["spectral_error"] for row in rows])),
        "energy_relative_error": float(
            np.mean([row["energy_relative_error"] for row in rows])
        ),
        "physics_residual_mse": float(np.mean(physics_values)),
        "samples": float(len(rows)),
    }
    return summary, rows


def evaluate_interpolation_baseline(
    loader: DataLoader,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for batch in loader:
        prediction = batch["features"][:, 1].numpy()
        target = batch["target"][:, 1].numpy()
        times = batch["time_index"].numpy()
        for predicted, expected, time_index in zip(prediction, target, times):
            rows.append(
                {
                    "time_index": float(time_index),
                    "rmse": float(np.sqrt(np.mean((predicted - expected) ** 2))),
                    "relative_l2": _relative_l2(predicted, expected),
                    "correlation": _correlation(predicted, expected),
                    "spectral_error": _spectral_error(predicted, expected),
                }
            )
    summary = {
        "rmse": float(np.mean([row["rmse"] for row in rows])),
        "relative_l2": float(np.mean([row["relative_l2"] for row in rows])),
        "correlation": float(np.mean([row["correlation"] for row in rows])),
        "spectral_error": float(np.mean([row["spectral_error"] for row in rows])),
        "samples": float(len(rows)),
    }
    return summary, rows


@torch.inference_mode()
def benchmark_inference(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    repeats: int = 40,
) -> float:
    model.eval()
    sample = sample.to(device)
    for _ in range(5):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / repeats / sample.shape[0]


def benchmark_solver(run: ScenarioRun, config: ExperimentConfig, repeats: int = 20) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        simulate(run.scenario, config)
    return 1000.0 * (time.perf_counter() - started) / repeats


def benchmark_coarse_solver(run: ScenarioRun, config: ExperimentConfig, repeats: int = 20) -> float:
    low_resolution_config = coarse_config(config)
    started = time.perf_counter()
    for _ in range(repeats):
        simulate(run.coarse_scenario, low_resolution_config)
    return 1000.0 * (time.perf_counter() - started) / repeats


@torch.inference_mode()
def predict_sequence(
    model: torch.nn.Module,
    run: ScenarioRun,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    centers = list(range(1, config.simulation_steps - 1))
    feature_stack = np.stack(
        [make_features(run.scenario, run.coarse_fields, center, config) for center in centers]
    )
    predictions = []
    for start in range(0, len(feature_stack), 16):
        batch = torch.from_numpy(feature_stack[start : start + 16]).to(device)
        predictions.append(model(batch)[:, 1].cpu().numpy())
    sequence = resize_fields(run.coarse_fields, config.grid_size)
    sequence[1:-1] = np.concatenate(predictions)
    sequence[:, [0, -1], :] = 0.0
    sequence[:, :, [0, -1]] = 0.0
    return sequence


def evaluate_sequences(
    model: torch.nn.Module,
    runs: list[ScenarioRun],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    sampled_errors: list[float] = []
    late_errors: list[float] = []
    evaluation_steps = np.linspace(3, config.simulation_steps - 3, 12).round().astype(int)
    for run in runs:
        sequence = predict_sequence(model, run, config, device)
        sampled_errors.extend(
            _relative_l2(sequence[step], run.fields[step]) for step in evaluation_steps
        )
        late_errors.append(_relative_l2(sequence[-2], run.fields[-2]))
    return {
        "mean_relative_l2": float(np.mean(sampled_errors)),
        "late_time_relative_l2": float(np.mean(late_errors)),
        "scenarios": float(len(runs)),
    }


def group_by_time(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        time_index = int(row["time_index"])
        for key in (
            "rmse",
            "relative_l2",
            "correlation",
            "spectral_error",
            "energy_relative_error",
        ):
            grouped[time_index][key].append(row[key])
    return [
        {
            "time_index": float(time_index),
            **{key: float(np.mean(values)) for key, values in metrics.items()},
        }
        for time_index, metrics in sorted(grouped.items())
    ]
