#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    import tensorflow as tf
except Exception as exc:
    raise RuntimeError(
        "TensorFlow failed to import. If you see a protobuf runtime mismatch, run:\n"
        "  conda install -c conda-forge protobuf\n"
        "or:\n"
        "  python -m pip install --upgrade protobuf\n"
        f"Original error: {exc}"
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sensor_dataset.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=args.seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=40,
        batch_size=64,
        verbose=0,
        callbacks=callbacks,
    )

    probs = model.predict(X_test_s, verbose=0).reshape(-1)
    pred = (probs >= 0.5).astype(int)
    y_test_i = y_test.astype(int)
    acc = accuracy_score(y_test_i, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test_i, pred, average="binary", zero_division=0)

    model.save(artifacts / "baseline_model.keras")
    joblib.dump(scaler, artifacts / "scaler.pkl")
    np.save(artifacts / "x_test.npy", X_test_s)
    np.save(artifacts / "y_test.npy", y_test_i)
    np.save(artifacts / "x_calib.npy", X_train_s[:300])

    metrics = {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}
    with open(artifacts / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Baseline trained.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
