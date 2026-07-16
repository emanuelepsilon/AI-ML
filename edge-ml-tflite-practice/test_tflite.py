import numpy as np
import tensorflow as tf

model_path = r"C:\Users\emanu\.cache\kagglehub\models\kaggle\yolo-v5\tfLite\tflite-tflite-model\1\1.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== INPUT DETAILS ===")
for item in input_details:
    print(item)

print("\n=== OUTPUT DETAILS ===")
for item in output_details:
    print(item)

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

x = np.zeros(input_shape, dtype=input_dtype)

interpreter.set_tensor(input_details[0]["index"], x)
interpreter.invoke()

y = interpreter.get_tensor(output_details[0]["index"])

print("\n=== TEST SUCCESS ===")
print("Input shape :", input_shape)
print("Input dtype :", input_dtype)
print("Output shape:", y.shape)
print("Output dtype:", y.dtype)