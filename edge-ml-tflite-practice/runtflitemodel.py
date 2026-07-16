import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


def dequantize_if_needed(arr: np.ndarray, quant):
    scale, zero = quant
    if scale and scale > 0:
        return (arr.astype(np.float32) - zero) * scale
    return arr.astype(np.float32)


def prepare_input(img: Image.Image, input_detail):
    in_h = int(input_detail["shape"][1])
    in_w = int(input_detail["shape"][2])
    in_dtype = input_detail["dtype"]
    in_scale, in_zero = input_detail.get("quantization", (0.0, 0))

    img_model = img.resize((in_w, in_h)).convert("RGB")
    x = np.asarray(img_model, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    if in_dtype == np.float32:
        x_in = x.astype(np.float32)
    elif in_dtype == np.uint8:
        if in_scale and in_scale > 0:
            x_in = np.clip(np.round(x / in_scale + in_zero), 0, 255).astype(np.uint8)
        else:
            x_in = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    elif in_dtype == np.int8:
        if not (in_scale and in_scale > 0):
            raise ValueError("INT8 model input but no valid quantization scale.")
        x_in = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
    else:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")

    return x_in


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .tflite model")
    p.add_argument("--image", required=True, help="Path to image")
    p.add_argument("--topn", type=int, default=5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--threads", type=int, default=1, help="CPU threads for inference")
    args = p.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    interpreter = Interpreter(model_path=str(model_path), num_threads=args.threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in0 = input_details[0]
    out0 = output_details[0]

    print("=== Model Info ===")
    print(f"Model file: {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1024:.2f} KB")
    print(f"Threads: {args.threads}")
    print(f"Input shape: {in0['shape']}")
    print(f"Input dtype: {in0['dtype']}")
    print(f"Input quant: {in0.get('quantization', None)}")
    print(f"Output shape: {out0['shape']}")
    print(f"Output dtype: {out0['dtype']}")
    print(f"Output quant: {out0.get('quantization', None)}")
    print()

    img = Image.open(image_path).convert("RGB")
    x_in = prepare_input(img, in0)

    for _ in range(args.warmup):
        interpreter.set_tensor(in0["index"], x_in)
        interpreter.invoke()
        _ = interpreter.get_tensor(out0["index"])

    t0 = time.perf_counter()
    for _ in range(args.runs):
        interpreter.set_tensor(in0["index"], x_in)
        interpreter.invoke()
        _ = interpreter.get_tensor(out0["index"])
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    per_ms = total_ms / args.runs
    throughput = 1000.0 / per_ms if per_ms > 0 else float("inf")

    print("=== Benchmark ===")
    print(f"Warmup runs: {args.warmup}")
    print(f"Timed runs : {args.runs}")
    print(f"Total time : {total_ms:.3f} ms")
    print(f"Latency    : {per_ms:.6f} ms/inference")
    print(f"Throughput : {throughput:.2f} inferences/s")
    print()

    interpreter.set_tensor(in0["index"], x_in)
    interpreter.invoke()
    y = interpreter.get_tensor(out0["index"])
    y = dequantize_if_needed(y, out0.get("quantization", (0.0, 0)))

    y = np.squeeze(y)
    if y.ndim == 1:
        top_idx = np.argsort(-y)[: args.topn]
        print("=== Top Predictions ===")
        for rank, idx in enumerate(top_idx, 1):
            print(f"{rank}. class={int(idx)} score={float(y[idx]):.6f}")
    else:
        print("Output is not 1D class scores after squeeze.")
        print(f"Output shape after squeeze: {y.shape}")


if __name__ == "__main__":
    main()
