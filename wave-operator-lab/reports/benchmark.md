# Wave Operator Benchmark

All models were evaluated on the same held-out simulations and on a harder out-of-distribution split containing stronger contrasts, multiple inclusions, and an unseen material layer.

| Model | Split | Relative L2 | Correlation | Spectral error | Energy error | Physics residual | Inference ms/sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO | TEST | 0.3750 | 0.9033 | 0.2859 | 0.3280 | 2.475e-05 | 2.506 |
| FNO | OOD | 0.4758 | 0.8687 | 0.3210 | 0.3650 | 3.610e-05 | 2.506 |
| CNN | TEST | 0.3541 | 0.9113 | 0.2834 | 1.0388 | 4.587e-05 | 1.338 |
| CNN | OOD | 0.4309 | 0.8871 | 0.3092 | 1.1450 | 5.949e-05 | 1.338 |

## Interpolation Baseline

- Test relative L2: **0.4585**
- OOD relative L2: **0.5039**

The interpolation baseline is the uncorrected coarse-grid solution resized to the fine grid. A learned model must improve on this result to demonstrate useful reconstruction rather than simple resizing.

## Complete Sequence Reconstruction

| Model | Split | Mean sequence L2 | Late-time L2 |
| --- | --- | ---: | ---: |
| FNO | TEST | 0.3713 | 0.5493 |
| FNO | OOD | 0.4676 | 0.6852 |
| CNN | TEST | 0.3432 | 0.5301 |
| CNN | OOD | 0.4190 | 0.6453 |

## Runtime Context

- Fine-grid solver, complete 72-step trajectory: **10.822 ms/scenario**
- Coarse-grid solver, complete trajectory: **4.164 ms/scenario**
- Evaluation device: **cpu**
- Fine/coarse grids: **32 x 32 / 16 x 16**

This CPU-scale benchmark reports every component separately. The goal is to measure the reconstruction tradeoff honestly rather than imply a speed advantage that depends on resolution and hardware.

## Evaluation Design

- Deterministic train, validation, test, and out-of-distribution splits
- Identical simulator and evaluation cases for both model families
- Spatial, spectral, energy, and discrete-physics diagnostics
- Model checkpoints selected only by validation objective
