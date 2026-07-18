from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .losses import operator_loss
from .model import build_model, count_parameters


@dataclass
class TrainingResult:
    model: torch.nn.Module
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    duration_seconds: float


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: ExperimentConfig) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _epoch_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "data": 0.0, "physics": 0.0, "gradient": 0.0}
    samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        velocity = batch["velocity"].to(device)
        damping = batch["damping"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(features)
            loss = operator_loss(prediction, target, velocity, damping, config)
            if training:
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = features.shape[0]
        samples += batch_size
        totals["total"] += float(loss.total.detach().cpu()) * batch_size
        totals["data"] += float(loss.data.detach().cpu()) * batch_size
        totals["physics"] += float(loss.physics.detach().cpu()) * batch_size
        totals["gradient"] += float(loss.gradient.detach().cpu()) * batch_size

    return {key: value / max(samples, 1) for key, value in totals.items()}


def train_model(
    name: str,
    config: ExperimentConfig,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    checkpoint_path: Path,
) -> TrainingResult:
    set_reproducible_seed(config.seed + (0 if name == "fno" else 1000))
    device = resolve_device(config)
    model = build_model(name, config).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    print(f"\n{name.upper()} | parameters={count_parameters(model):,} | device={device}")
    for epoch in range(1, config.epochs + 1):
        train_metrics = _epoch_pass(model, train_loader, config, device, optimizer)
        validation_metrics = _epoch_pass(model, validation_loader, config, device, None)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{config.epochs}  "
            f"train={train_metrics['total']:.5e}  "
            f"validation={validation_metrics['total']:.5e}  "
            f"physics={validation_metrics['physics']:.5e}"
        )

        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                print(f"early stopping after epoch {epoch}")
                break

    duration = time.perf_counter() - started
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "model_state": best_state,
            "config": config.to_dict(),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        checkpoint_path,
    )
    return TrainingResult(model, history, best_epoch, best_loss, duration)


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ExperimentConfig, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    model = build_model(str(checkpoint["model_name"]), config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint
