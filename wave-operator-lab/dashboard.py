from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from wave_operator.config import load_config
from wave_operator.data import ScenarioRun, build_runs
from wave_operator.training import load_trained_model
from wave_operator.visualization import predict_run


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "default.json"


@st.cache_resource
def load_model(model_name: str) -> tuple[torch.nn.Module, object, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_trained_model(
        ROOT / "artifacts" / f"{model_name}_best.pt",
        device,
    )
    return model, config, device


@st.cache_data
def load_scenario(seed: int, ood: bool) -> ScenarioRun:
    config = load_config(CONFIG_PATH)
    return build_runs(1, config, seed_offset=seed - config.seed, ood=ood)[0]


@st.cache_data
def load_prediction(seed: int, ood: bool, model_name: str) -> np.ndarray:
    model, config, device = load_model(model_name)
    run = load_scenario(seed, ood)
    return predict_run(model, run, config, device)


def field_figure(field: np.ndarray, title: str, limit: float, cmap: str = "turbo"):
    figure, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="#07090d")
    ax.set_facecolor("#07090d")
    image = ax.imshow(
        field,
        origin="lower",
        cmap=cmap,
        vmin=-limit if cmap != "magma" else 0.0,
        vmax=limit,
        interpolation="bilinear",
    )
    ax.set_title(title, color="#e5e7eb")
    ax.set_xticks([])
    ax.set_yticks([])
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def main() -> None:
    st.set_page_config(page_title="WaveOperator Lab", layout="wide")
    st.title("WaveOperator Lab")
    st.caption("Physics-informed neural operator diagnostics for heterogeneous acoustic media")
    controls, content = st.columns([0.22, 0.78])
    with controls:
        st.subheader("Scenario")
        model_name = st.selectbox(
            "Reconstruction model",
            options=("cnn", "fno"),
            format_func=lambda name: "CNN baseline" if name == "cnn" else "Fourier Neural Operator",
        )
        seed = st.number_input("Seed", min_value=1, max_value=100_000, value=30_017, step=1)
        ood = st.toggle("Harder unseen medium", value=True)
    model, config, device = load_model(model_name)
    with controls:
        time_index = st.slider("Time step", 2, config.simulation_steps - 2, 34)

    run = load_scenario(int(seed), ood)
    started = time.perf_counter()
    prediction = load_prediction(int(seed), ood, model_name)[time_index]
    latency_ms = 1000.0 * (time.perf_counter() - started)
    truth = run.fields[time_index]
    error = np.abs(prediction - truth)
    relative_l2 = float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1.0e-8))
    correlation = float(np.corrcoef(prediction.ravel(), truth.ravel())[0, 1])
    limit = max(float(np.percentile(np.abs(truth), 99.5)), 0.05)

    with content:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Relative L2", f"{relative_l2:.4f}")
        metric_columns[1].metric("Correlation", f"{correlation:.4f}")
        metric_columns[2].metric("Full sequence", f"{latency_ms:.2f} ms")
        metric_columns[3].metric("Medium", "OOD" if ood else "in-distribution")
        ground_truth, neural, diagnostic = st.columns(3)
        ground_truth.pyplot(field_figure(truth, "Numerical ground truth", limit), use_container_width=True)
        neural.pyplot(field_figure(prediction, "Neural prediction", limit), use_container_width=True)
        diagnostic.pyplot(
            field_figure(error, "Absolute error", max(float(error.max()), 1.0e-4), "magma"),
            use_container_width=True,
        )

    st.divider()
    medium, source = st.columns(2)
    medium.pyplot(
        field_figure(run.scenario.velocity, "Wave-speed field", float(run.scenario.velocity.max()), "viridis"),
        use_container_width=True,
    )
    source.pyplot(
        field_figure(
            run.scenario.initial_pressure,
            "Initial pressure field",
            max(float(np.abs(run.scenario.initial_pressure).max()), 1.0e-4),
        ),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
