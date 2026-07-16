# Benchmark Summary

|   accuracy |   precision |   recall |   f1 | model          |   latency_ms_total |   latency_ms_per_sample |   size_kb |
|-----------:|------------:|---------:|-----:|:---------------|-------------------:|------------------------:|----------:|
|          1 |           1 |        1 |    1 | keras_fp32     |           176.207  |              0.146839   |  36.9375  |
|          1 |           1 |        1 |    1 | tflite_float16 |            10.2086 |              0.00850717 |   5.01562 |
|          1 |           1 |        1 |    1 | tflite_fp32    |            10.3945 |              0.00866208 |   6.10156 |
|          1 |           1 |        1 |    1 | tflite_int8    |            25.1743 |              0.0209786  |   4.69531 |

- Best F1: **keras_fp32** (1.0000)
- Fastest: **tflite_float16** (0.0085 ms/sample)
