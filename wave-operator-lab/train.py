from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from wave_operator.config import load_config
from wave_operator.data import build_dataset
from wave_operator.model import count_parameters
from wave_operator.training import resolve_device, train_model


ROOT = Path(__file__).resolve().parent


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train acoustic-wave surrogate models.")
    parser.add_argument("--config", default=ROOT / "configs" / "default.json", type=Path)
    parser.add_argument("--models", default="fno,cnn", help="Comma-separated model names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    print("Building deterministic acoustic simulation datasets...")
    train_dataset, _ = build_dataset(
        config.train_scenarios,
        config,
        seed_offset=0,
    )
    validation_dataset, _ = build_dataset(
        config.validation_scenarios,
        config,
        seed_offset=10_000,
    )
    print(f"training samples: {len(train_dataset)}")
    print(f"validation samples: {len(validation_dataset)}")
    print(f"device: {resolve_device(config)}")

    manifest: dict[str, object] = {
        "config": config.to_dict(),
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "models": {},
    }
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        checkpoint_path = artifacts / f"{name}_best.pt"
        result = train_model(name, config, train_dataset, validation_dataset, checkpoint_path)
        write_history(artifacts / f"{name}_training_history.csv", result.history)
        manifest["models"][name] = {
            "parameters": count_parameters(result.model),
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "duration_seconds": result.duration_seconds,
            "checkpoint": checkpoint_path.name,
        }

    (artifacts / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("\nTraining complete.")
    print(f"manifest: {artifacts / 'training_manifest.json'}")


if __name__ == "__main__":
    main()
