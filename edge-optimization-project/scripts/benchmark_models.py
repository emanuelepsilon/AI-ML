#!/usr/bin/env python3

from pathlib import Path
import argparse
import time
import json
import numpy as np
import pandas as pd
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


def size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def benchmark_keras(model_path: Path, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    model = tf.keras.models.load_model(model_path)

    # Warm-up
    _ = model.predict(x_test[:16], verbose=0)

    t0 = time.perf_counter()
    probs = model.predict(x_test, verbose=0).reshape(-1)
    total_ms = (time.perf_counter() - t0) * 1000.0
    pred = (probs >= 0.5).astype(int)
    m = compute_metrics(y_test, pred)
    m["model"] = "keras_fp32"
    m["latency_ms_total"] = total_ms
    m["latency_ms_per_sample"] = total_ms / len(x_test)
    m["size_kb"] = size_kb(model_path)
    return m


def benchmark_tflite(model_path: Path, x_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    preds = []
    t0 = time.perf_counter()
    for i in range(len(x_test)):
        sample = x_test[i : i + 1]
        if input_details["dtype"] == np.int8:
            q = np.round(sample / in_scale + in_zero).astype(np.int8)
            interpreter.set_tensor(input_details["index"], q)
        else:
            interpreter.set_tensor(input_details["index"], sample.astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] == np.int8:
            prob = (out.astype(np.float32) - out_zero) * out_scale
            prob = float(prob.reshape(-1)[0])
        else:
            prob = float(out.reshape(-1)[0])
        preds.append(1 if prob >= 0.5 else 0)
    total_ms = (time.perf_counter() - t0) * 1000.0

    m = compute_metrics(y_test, np.array(preds, dtype=int))
    m["model"] = model_name
    m["latency_ms_total"] = total_ms
    m["latency_ms_per_sample"] = total_ms / len(x_test)
    m["size_kb"] = size_kb(model_path)
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    x_test = np.load(artifacts / "x_test.npy").astype(np.float32)
    y_test = np.load(artifacts / "y_test.npy").astype(int)

    results = [
        benchmark_keras(artifacts / "baseline_model.keras", x_test, y_test),
        benchmark_tflite(artifacts / "model_fp32.tflite", x_test, y_test, "tflite_fp32"),
        benchmark_tflite(artifacts / "model_float16.tflite", x_test, y_test, "tflite_float16"),
        benchmark_tflite(artifacts / "model_int8.tflite", x_test, y_test, "tflite_int8"),
    ]
    df = pd.DataFrame(results).sort_values("model")
    df.to_csv(reports / "benchmark_results.csv", index=False)

    best_f1 = df.sort_values("f1", ascending=False).iloc[0]
    best_speed = df.sort_values("latency_ms_per_sample", ascending=True).iloc[0]
    summary = {
        "best_f1_model": best_f1["model"],
        "best_f1": float(best_f1["f1"]),
        "fastest_model": best_speed["model"],
        "fastest_ms_per_sample": float(best_speed["latency_ms_per_sample"]),
    }
    with open(reports / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(reports / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"- Best F1: **{summary['best_f1_model']}** ({summary['best_f1']:.4f})\n")
        f.write(
            f"- Fastest: **{summary['fastest_model']}** ({summary['fastest_ms_per_sample']:.4f} ms/sample)\n"
        )

    print("Saved reports/benchmark_results.csv")
    print("Saved reports/benchmark_summary.md")


if __name__ == "__main__":
    main()
