# Edge Optimization Project (Fresh Start)

This folder is a clean, end-to-end pipeline for:

1. Creating a labeled sensor dataset (with ground truth)
2. Training a baseline neural network
3. Optimizing to TensorFlow Lite (float16 + int8)
4. Benchmarking quality + speed + model size

## Quick Run (PowerShell)

From this folder:

```powershell
.\run_all.ps1
```

## Files

- `scripts/make_synthetic_sensor_dataset.py`
  - Generates a realistic 3-axis sensor dataset with true labels (`fan_off`, `fan_on`).
- `scripts/train_baseline.py`
  - Trains baseline model and saves metrics/artifacts.
- `scripts/optimize_tflite.py`
  - Creates optimized TFLite models.
- `scripts/benchmark_models.py`
  - Compares baseline vs optimized models.
- `reports/benchmark_summary.md`
  - Final summary table.

## Replace with Real Dataset Later

Once you pick a Kaggle dataset:

1. Keep the same feature column format (numeric features + `label`).
2. Replace `data/sensor_dataset.csv`.
3. Re-run `.\run_all.ps1`.

