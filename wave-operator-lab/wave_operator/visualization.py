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


def render_comparison_frame(
    truth: np.ndarray,
    prediction: np.ndarray,
    step: int,
    angle: float,
    z_limit: float,
) -> Image.Image:
    size = truth.shape[0]
    axis = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(axis, axis)
    figure = plt.figure(figsize=(10.4, 4.8), dpi=100, facecolor="#000000")
    for panel, (field, title) in enumerate(
        ((truth, "Numerical ground truth"), (prediction, "Neural operator prediction")),
        start=1,
    ):
        ax = figure.add_subplot(1, 2, panel, projection="3d")
        _style_3d(ax)
        rendered = _smooth(field)
        ax.plot_surface(
            xx,
            yy,
            rendered,
            cmap="turbo",
            vmin=-z_limit,
            vmax=z_limit,
            linewidth=0,
            antialiased=True,
        )
        ax.contour(
            xx,
            yy,
            rendered,
            zdir="z",
            offset=-z_limit,
            levels=10,
            cmap="turbo",
            linewidths=0.65,
            alpha=0.72,
        )
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(-z_limit, z_limit))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("pressure")
        ax.set_title(title, fontsize=12, weight="bold", pad=10)
        ax.view_init(elev=28, azim=angle)
    figure.suptitle(
        f"Acoustic field reconstruction | time step {step:02d}",
        color=TEXT,
        fontsize=14,
        weight="bold",
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.88, wspace=0.0)
    return _to_image(figure)


def render_error_frame(
    truth: np.ndarray,
    prediction: np.ndarray,
    time_steps: list[int],
    error_history: list[float],
    step: int,
    angle: float,
    error_limit: float,
) -> Image.Image:
    error = np.abs(prediction - truth)
    size = truth.shape[0]
    axis = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(axis, axis)
    figure = plt.figure(figsize=(10.4, 4.8), dpi=100, facecolor="#000000")
    surface_ax = figure.add_subplot(1, 2, 1, projection="3d")
    _style_3d(surface_ax)
    surface_ax.plot_surface(
        xx,
        yy,
        _smooth(error),
        cmap="magma",
        vmin=0.0,
        vmax=error_limit,
        linewidth=0,
        antialiased=True,
    )
    surface_ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, error_limit))
    surface_ax.set_xlabel("x")
    surface_ax.set_ylabel("y")
    surface_ax.set_zlabel("absolute error")
    surface_ax.set_title("Spatial prediction error", fontsize=12, weight="bold", pad=10)
    surface_ax.view_init(elev=30, azim=angle)

    curve_ax = figure.add_subplot(1, 2, 2)
    curve_ax.set_facecolor("#06080c")
    curve_ax.plot(time_steps[: len(error_history)], error_history, color="#38bdf8", linewidth=2.2)
    curve_ax.scatter(time_steps[len(error_history) - 1], error_history[-1], color="#facc15", s=48, zorder=4)
    curve_ax.set_xlim(min(time_steps), max(time_steps))
    curve_ax.set_ylim(0.0, max(max(error_history) * 1.2, 0.1))
    curve_ax.set_xlabel("time step", color=TEXT)
    curve_ax.set_ylabel("relative L2 error", color=TEXT)
    curve_ax.set_title("Error through time", color=TEXT, fontsize=12, weight="bold")
    curve_ax.tick_params(colors=TEXT)
    curve_ax.grid(alpha=0.2)
    for spine in curve_ax.spines.values():
        spine.set_color("#4b5563")
    curve_ax.text(
        0.04,
        0.93,
        f"current error: {error_history[-1]:.3f}",
        transform=curve_ax.transAxes,
        color="#facc15",
        va="top",
        fontsize=11,
    )
    figure.suptitle(
        f"Out-of-distribution diagnostic | time step {step:02d}",
        color=TEXT,
        fontsize=14,
        weight="bold",
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.86, wspace=0.22)
    return _to_image(figure)


def save_gif(frames: list[Image.Image], path: Path, duration: int = 105) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration,
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
    time_indices = np.linspace(4, config.simulation_steps - 3, 24).round().astype(int).tolist()
    rollout = predict_run(model, run, config, device)
    predictions = rollout[time_indices]
    truth = run.fields[time_indices]
    z_limit = max(float(np.percentile(np.abs(truth), 99.5)), 0.08)
    errors = [
        float(np.linalg.norm(predicted - expected) / max(np.linalg.norm(expected), 1.0e-8))
        for predicted, expected in zip(predictions, truth)
    ]
    error_limit = max(float(np.percentile(np.abs(predictions - truth), 99.2)), 0.025)

    comparison_frames = [
        render_comparison_frame(expected, predicted, step, -58 + frame * 1.0, z_limit)
        for frame, (step, expected, predicted) in enumerate(zip(time_indices, truth, predictions))
    ]
    error_frames = [
        render_error_frame(
            expected,
            predicted,
            time_indices,
            errors[: frame + 1],
            step,
            -50 + frame * 0.9,
            error_limit,
        )
        for frame, (step, expected, predicted) in enumerate(zip(time_indices, truth, predictions))
    ]
    save_gif(comparison_frames, assets_dir / "wave-operator-comparison.gif")
    save_gif(error_frames, assets_dir / "wave-operator-error-analysis.gif")
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
