import kagglehub

# Download latest version
path = kagglehub.model_download("kaggle/yolo-v5/tfLite/tflite-tflite-model")

print("Path to model files:", path)