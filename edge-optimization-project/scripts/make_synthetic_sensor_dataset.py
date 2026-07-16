#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def build_sample(label: int, window: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, window)
    if label == 1:
        amp = rng.uniform(0.9, 1.5)
        freq = rng.uniform(12, 22)
        noise = rng.normal(0, 0.18, size=(window, 3))
    else:
        amp = rng.uniform(0.08, 0.35)
        freq = rng.uniform(1.5, 6.0)
        noise = rng.normal(0, 0.06, size=(window, 3))

    x = amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, np.pi)) + noise[:, 0]
    y = 0.9 * amp * np.cos(2 * np.pi * (freq * 0.9) * t + rng.uniform(0, np.pi)) + noise[:, 1]
    z = 0.75 * amp * np.sin(2 * np.pi * (freq * 1.1) * t + rng.uniform(0, np.pi)) + noise[:, 2]
    return np.vstack([x, y, z]).T


def featurize(sample: np.ndarray) -> dict:
    x, y, z = sample[:, 0], sample[:, 1], sample[:, 2]
    mag = np.sqrt(x * x + y * y + z * z)
    return {
        "x_mean": float(np.mean(x)),
        "y_mean": float(np.mean(y)),
        "z_mean": float(np.mean(z)),
        "x_std": float(np.std(x)),
        "y_std": float(np.std(y)),
        "z_std": float(np.std(z)),
        "x_rms": float(np.sqrt(np.mean(x * x))),
        "y_rms": float(np.sqrt(np.mean(y * y))),
        "z_rms": float(np.sqrt(np.mean(z * z))),
        "x_ptp": float(np.ptp(x)),
        "y_ptp": float(np.ptp(y)),
        "z_ptp": float(np.ptp(z)),
        "mag_mean": float(np.mean(mag)),
        "mag_std": float(np.std(mag)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sensor_dataset.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.samples):
        label = int(rng.random() > 0.5)
        sample = build_sample(label, args.window, rng)
        row = featurize(sample)
        row["label"] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["label"].value_counts().sort_index().rename(index={0: "fan_off", 1: "fan_on"}))


if __name__ == "__main__":
    main()
