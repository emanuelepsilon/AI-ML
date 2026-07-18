from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wave_operator.config import load_config
from wave_operator.data import build_dataset
from wave_operator.evaluation import (
    benchmark_coarse_solver,
    benchmark_inference,
    benchmark_solver,
    evaluate_interpolation_baseline,
    evaluate_loader,
    evaluate_sequences,
    group_by_time,
    predict_sequence,
)
from wave_operator.model import count_parameters
from wave_operator.training import load_trained_model, resolve_device
from wave_operator.visualization import create_demo_visuals, plot_training_curves


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained acoustic-wave surrogates.")
    parser.add_argument("--config", default=ROOT / "configs" / "default.json", type=Path)
    parser.add_argument("--models", default="fno,cnn")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict[str, object], destination: Path) -> None:
    results = summary["results"]
    lines = [
        "# Wave Operator Benchmark",
        "",
        "All models were evaluated on the same held-out simulations and on a harder out-of-distribution split containing stronger contrasts, multiple inclusions, and an unseen material layer.",
        "",
        "| Model | Split | Relative L2 | Correlation | Spectral error | Energy error | Physics residual | Inference ms/sample |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, model_results in results.items():
        for split in ("test", "ood"):
            metrics = model_results[split]
            lines.append(
                f"| {model_name.upper()} | {split.upper()} | "
                f"{metrics['relative_l2']:.4f} | {metrics['correlation']:.4f} | "
                f"{metrics['spectral_error']:.4f} | {metrics['energy_relative_error']:.4f} | "
                f"{metrics['physics_residual_mse']:.3e} | "
                f"{model_results['inference_ms_per_sample']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpolation Baseline",
            "",
            f"- Test relative L2: **{summary['interpolation_baseline']['test']['relative_l2']:.4f}**",
            f"- OOD relative L2: **{summary['interpolation_baseline']['ood']['relative_l2']:.4f}**",
            "",
            "The interpolation baseline is the uncorrected coarse-grid solution resized to the fine grid. A learned model must improve on this result to demonstrate useful reconstruction rather than simple resizing.",
            "",
            "## Complete Sequence Reconstruction",
            "",
            "| Model | Split | Mean sequence L2 | Late-time L2 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for model_name, model_results in results.items():
        for split in ("test_sequence", "ood_sequence"):
            metrics = model_results[split]
            lines.append(
                f"| {model_name.upper()} | {split.replace('_sequence', '').upper()} | "
                f"{metrics['mean_relative_l2']:.4f} | {metrics['late_time_relative_l2']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Runtime Context",
            "",
            f"- Fine-grid solver, complete {summary['config']['simulation_steps']}-step trajectory: **{summary['fine_solver_ms_per_trajectory']:.3f} ms/scenario**",
            f"- Coarse-grid solver, complete trajectory: **{summary['coarse_solver_ms_per_trajectory']:.3f} ms/scenario**",
            f"- Evaluation device: **{summary['device']}**",
            f"- Fine/coarse grids: **{summary['config']['grid_size']} x {summary['config']['grid_size']} / {summary['config']['coarse_grid_size']} x {summary['config']['coarse_grid_size']}**",
            "",
            "This CPU-scale benchmark reports every component separately. The goal is to measure the reconstruction tradeoff honestly rather than imply a speed advantage that depends on resolution and hardware.",
            "",
            "## Evaluation Design",
            "",
            "- Deterministic train, validation, test, and out-of-distribution splits",
            "- Identical simulator and evaluation cases for both model families",
            "- Spatial, spectral, energy, and discrete-physics diagnostics",
            "- Model checkpoints selected only by validation objective",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    reports = ROOT / "reports"
    assets = ROOT / "assets"
    artifacts = ROOT / "artifacts"
    reports.mkdir(exist_ok=True)
    assets.mkdir(exist_ok=True)

    test_dataset, test_runs = build_dataset(
        config.test_scenarios,
        config,
        seed_offset=20_000,
    )
    ood_dataset, ood_runs = build_dataset(
        config.ood_scenarios,
        config,
        seed_offset=30_000,
        ood=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=config.batch_size, shuffle=False)
    device = resolve_device(config)

    summary: dict[str, object] = {
        "device": str(device),
        "config": config.to_dict(),
        "test_samples": len(test_dataset),
        "ood_samples": len(ood_dataset),
        "fine_solver_ms_per_trajectory": benchmark_solver(test_runs[0], config),
        "coarse_solver_ms_per_trajectory": benchmark_coarse_solver(test_runs[0], config),
        "results": {},
    }
    baseline_test, _ = evaluate_interpolation_baseline(test_loader)
    baseline_ood, _ = evaluate_interpolation_baseline(ood_loader)
    summary["interpolation_baseline"] = {"test": baseline_test, "ood": baseline_ood}
    detailed_rows: list[dict[str, object]] = []
    loaded_models: dict[str, torch.nn.Module] = {}
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]

    for name in model_names:
        checkpoint_path = artifacts / f"{name}_best.pt"
        model, _, checkpoint = load_trained_model(checkpoint_path, device)
        test_metrics, test_rows = evaluate_loader(model, test_loader, config, device)
        ood_metrics, ood_rows = evaluate_loader(model, ood_loader, config, device)
        sample_batch = next(iter(test_loader))["features"]
        latency = benchmark_inference(model, sample_batch, device)
        test_sequence = evaluate_sequences(model, test_runs, config, device)
        ood_sequence = evaluate_sequences(model, ood_runs, config, device)
        model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        summary["results"][name] = {
            "parameters": count_parameters(model),
            "checkpoint_mb": model_size_mb,
            "best_epoch": int(checkpoint["best_epoch"]),
            "inference_ms_per_sample": latency,
            "test": test_metrics,
            "ood": ood_metrics,
            "test_sequence": test_sequence,
            "ood_sequence": ood_sequence,
            "by_time": {
                "test": group_by_time(test_rows),
                "ood": group_by_time(ood_rows),
            },
        }
        for split, rows in (("test", test_rows), ("ood", ood_rows)):
            detailed_rows.extend({"model": name, "split": split, **row} for row in rows)
        print(
            f"{name.upper()}: test L2={test_metrics['relative_l2']:.4f}, "
            f"OOD L2={ood_metrics['relative_l2']:.4f}, "
            f"OOD sequence={ood_sequence['mean_relative_l2']:.4f}, "
            f"latency={latency:.3f} ms/sample"
        )
        loaded_models[name] = model

    visualized_model = min(
        model_names,
        key=lambda model_name: summary["results"][model_name]["ood_sequence"]["mean_relative_l2"],
    )
    summary["visualized_model"] = visualized_model
    scenario_scores = []
    for index, run in enumerate(ood_runs):
        sequence = predict_sequence(loaded_models[visualized_model], run, config, device)
        score = sum(
            float(
                torch.linalg.vector_norm(
                    torch.from_numpy(sequence[step] - run.fields[step])
                )
                / torch.linalg.vector_norm(torch.from_numpy(run.fields[step])).clamp_min(1.0e-8)
            )
            for step in range(4, config.simulation_steps - 2, 6)
        ) / len(range(4, config.simulation_steps - 2, 6))
        scenario_scores.append((score, index))
    scenario_scores.sort()
    demo_score, demo_index = scenario_scores[len(scenario_scores) // 2]
    summary["demo_scenario"] = {
        "seed": ood_runs[demo_index].seed,
        "selection": "median out-of-distribution sequence",
        "selection_relative_l2": demo_score,
    }
    summary["demo"] = create_demo_visuals(
        loaded_models[visualized_model],
        ood_runs[demo_index],
        config,
        device,
        assets,
    )

    plot_training_curves(
        {
            name: artifacts / f"{name}_training_history.csv"
            for name in model_names
        },
        assets / "training-curves.png",
    )
    (reports / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_rows(reports / "evaluation_samples.csv", detailed_rows)
    write_report(summary, reports / "benchmark.md")
    print(f"report: {reports / 'benchmark.md'}")
    print(f"visuals: {assets}")


if __name__ == "__main__":
    main()
