#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
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
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    model_path = artifacts / "baseline_model.keras"
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    (artifacts / "model_fp32.tflite").write_bytes(tflite_fp32)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter.convert()
    (artifacts / "model_float16.tflite").write_bytes(tflite_f16)

    x_calib = np.load(artifacts / "x_calib.npy").astype(np.float32)

    def rep_data():
        for i in range(len(x_calib)):
            yield [x_calib[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_i8 = converter.convert()
    (artifacts / "model_int8.tflite").write_bytes(tflite_i8)

    print("Saved artifacts/model_fp32.tflite")
    print("Saved artifacts/model_float16.tflite")
    print("Saved artifacts/model_int8.tflite")


if __name__ == "__main__":
    main()
