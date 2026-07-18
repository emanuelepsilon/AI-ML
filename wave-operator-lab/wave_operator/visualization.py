from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from .config import ExperimentConfig
from .data import ScenarioRun
from .evaluation import predict_sequence


TEXT = "#dbe4ee"
GRID = (0.55, 0.60, 0.68, 0.24)


def _style_3d(ax) -> None:
    ax.set_facecolor("#000000")
    ax.figure.set_facecolor("#000000")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.015, 0.015, 0.015, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.zaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _to_image(figure: Figure) -> Image.Image:
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())).convert("RGB")
    plt.close(figure)
    return image


def _smooth(field: np.ndarray, passes: int = 2) -> np.ndarray:
    result = field.copy()
    for _ in range(passes):
        result = (
            4.0 * result
            + np.roll(result, 1, 0)
            + np.roll(result, -1, 0)
            + np.roll(result, 1, 1)
            + np.roll(result, -1, 1)
        ) / 8.0
    return result


@torch.inference_mode()
def predict_run(
    model: torch.nn.Module,
    run: ScenarioRun,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    return predict_sequence(model, run, config, device)


def _render_surface(
    ax,
    field: np.ndarray,
    title: str,
    angle: float,
    z_limit: float,
    smooth_passes: int,
    show_grid: bool = False,
) -> None:
    size = field.shape[0]
    axis = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(axis, axis)
    rendered = _smooth(field, passes=smooth_passes) if smooth_passes else field
    _style_3d(ax)
    ax.plot_surface(
        xx,
        yy,
        rendered,
        cmap="turbo",
        vmin=-z_limit,
        vmax=z_limit,
        linewidth=0.24 if show_grid else 0,
        edgecolor=(0.92, 0.95, 1.0, 0.22) if show_grid else "none",
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        rendered,
        zdir="z",
        offset=-z_limit,
        levels=12,
        cmap="turbo",
        linewidths=0.6,
        alpha=0.72,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(-z_limit, z_limit))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("pressure")
    ax.set_title(title, fontsize=12, weight="bold", pad=9)
    ax.view_init(elev=28, azim=angle)


def render_propagation_frame(
    truth: np.ndarray,
    step: int,
    angle: float,
    z_limit: float,
) -> Image.Image:
    figure = plt.figure(figsize=(8.8, 5.2), dpi=100, facecolor="#000000")
    ax = figure.add_subplot(1, 1, 1, projection="3d")
    _render_surface(
        ax,
        truth,
        "High-resolution finite-difference solution",
        angle,
        z_limit,
        smooth_passes=1,
    )
    figure.suptitle(
        f"Heterogeneous acoustic-wave propagation | time step {step:02d}",
        color=TEXT,
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Reference simulation · 32 x 32 grid",
        color="#94a3b8",
        fontsize=10,
        ha="center",
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.87)
    return _to_image(figure)


def render_predictor_frame(
    coarse_input: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    step: int,
    angle: float,
    z_limit: float,
) -> Image.Image:
    relative_l2 = float(
        np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1.0e-8)
    )
    correlation = float(np.corrcoef(prediction.ravel(), truth.ravel())[0, 1])
    figure = plt.figure(figsize=(10.4, 4.8), dpi=100, facecolor="#000000")
    coarse_ax = figure.add_subplot(1, 2, 1, projection="3d")
    prediction_ax = figure.add_subplot(1, 2, 2, projection="3d")
    _render_surface(
        coarse_ax,
        coarse_input,
        "Coarse solver input · 16 x 16",
        angle,
        z_limit,
        smooth_passes=0,
        show_grid=True,
    )
    _render_surface(
        prediction_ax,
        prediction,
        "Neural reconstruction · 32 x 32",
        angle,
        z_limit,
        smooth_passes=1,
    )
    figure.suptitle(
        f"CNN coarse-to-fine predictor | OOD case | time step {step:02d}",
        color=TEXT,
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.025,
        f"Validated against withheld ground truth · relative L2 {relative_l2:.3f} · correlation {correlation:.3f}",
        color="#94a3b8",
        fontsize=10,
        ha="center",
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.14, top=0.87, wspace=0.0)
    return _to_image(figure)


def save_gif(frames: list[Image.Image], path: Path, duration: int = 72) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration,
        disposal=2,
        loop=0,
    )


def create_demo_visuals(
    model: torch.nn.Module,
    run: ScenarioRun,
    config: ExperimentConfig,
    device: torch.device,
    assets_dir: Path,
) -> dict[str, float]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    time_indices = np.linspace(6, config.simulation_steps - 4, 48).round().astype(int).tolist()
    rollout = predict_run(model, run, config, device)
    predictions = rollout[time_indices]
    truth = run.fields[time_indices]
    coarse_inputs = run.coarse_fields[time_indices]
    z_limit = max(
        float(np.percentile(np.abs(np.concatenate((truth, predictions), axis=0)), 99.6)),
        0.08,
    )
    errors = [
        float(np.linalg.norm(predicted - expected) / max(np.linalg.norm(expected), 1.0e-8))
        for predicted, expected in zip(predictions, truth)
    ]
    propagation_frames = [
        render_propagation_frame(expected, step, -56 + frame * 0.12, z_limit)
        for frame, (step, expected) in enumerate(zip(time_indices, truth))
    ]
    predictor_frames = [
        render_predictor_frame(
            coarse,
            predicted,
            expected,
            step,
            -56 + frame * 0.12,
            z_limit,
        )
        for frame, (step, coarse, expected, predicted) in enumerate(
            zip(time_indices, coarse_inputs, truth, predictions)
        )
    ]
    save_gif(propagation_frames, assets_dir / "wave-propagation.gif")
    save_gif(predictor_frames, assets_dir / "neural-predictor.gif")
    return {
        "demo_mean_relative_l2": float(np.mean(errors)),
        "demo_final_relative_l2": float(errors[-1]),
        "demo_max_absolute_error": float(np.max(np.abs(predictions - truth))),
    }


def plot_training_curves(histories: dict[str, Path], destination: Path) -> None:
    plt.style.use("dark_background")
    figure, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    colors = {"fno": "#38bdf8", "cnn": "#facc15"}
    for name, path in histories.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(float(row["epoch"])) for row in rows]
        values = [float(row["validation_total"]) for row in rows]
        ax.plot(epochs, values, label=name.upper(), color=colors.get(name), linewidth=2.1)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation objective")
    ax.set_title("Surrogate model convergence")
    ax.grid(alpha=0.18)
    ax.legend()
    figure.tight_layout()
    figure.savefig(destination, facecolor="#000000")
    plt.close(figure)
