# Wave Operator Benchmark

All models were evaluated on the same held-out simulations and on a harder out-of-distribution split containing stronger contrasts, multiple inclusions, and an unseen material layer.

| Model | Split | Relative L2 | Correlation | Spectral error | Energy error | Physics residual | Inference ms/sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO | TEST | 0.3759 | 0.9032 | 0.2868 | 0.3461 | 2.563e-05 | 3.435 |
| FNO | OOD | 0.4770 | 0.8680 | 0.3213 | 0.3618 | 3.596e-05 | 3.435 |
| CNN | TEST | 0.3542 | 0.9119 | 0.2853 | 1.0419 | 4.587e-05 | 1.281 |
| CNN | OOD | 0.4292 | 0.8874 | 0.3090 | 1.1375 | 5.905e-05 | 1.281 |

## Interpolation Baseline

- Test relative L2: **0.4613**
- OOD relative L2: **0.5086**

The interpolation baseline is the uncorrected coarse-grid solution resized to the fine grid. A learned model must improve on this result to demonstrate useful reconstruction rather than simple resizing.

## Complete Sequence Reconstruction

| Model | Split | Mean sequence L2 | Late-time L2 |
| --- | --- | ---: | ---: |
| FNO | TEST | 0.3717 | 0.5535 |
| FNO | OOD | 0.4683 | 0.6985 |
| CNN | TEST | 0.3432 | 0.5354 |
| CNN | OOD | 0.4175 | 0.6544 |

## Runtime Context

- Fine-grid solver, complete 72-step trajectory: **4.576 ms/scenario**
- Coarse-grid solver, complete trajectory: **3.759 ms/scenario**
- Evaluation device: **cpu**
- Fine/coarse grids: **32 x 32 / 16 x 16**

This CPU-scale benchmark reports every component separately. The goal is to measure the reconstruction tradeoff honestly rather than imply a speed advantage that depends on resolution and hardware.

## Evaluation Design

- Deterministic train, validation, test, and out-of-distribution splits
- Identical simulator and evaluation cases for both model families
- Spatial, spectral, energy, and discrete-physics diagnostics
- Model checkpoints selected only by validation objective
