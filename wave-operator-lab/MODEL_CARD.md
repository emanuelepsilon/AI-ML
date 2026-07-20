# Model Card

## Purpose

WaveOperator Lab studies whether learned residual correctors can reconstruct fine acoustic-wave fields from stable low-resolution simulations. It is a scientific machine-learning experiment, not a validated engineering acoustics package.

## Models

- **Fourier Neural Operator:** global spectral mixing with four Fourier blocks and 207,783 trainable parameters.
- **Convolutional baseline:** compact dilated residual CNN with 46,947 trainable parameters.

Both models receive three consecutive interpolated coarse fields, the fine-grid wave-speed map, the initial-pressure field, spatial coordinates, and normalized time. They predict three corrected fine-grid fields.

## Data

All samples come from the included finite-difference acoustic solver. Training media use smooth random inclusions and variable absorbing boundary profiles. OOD media increase contrast, add a layered feature and use boundary strengths outside the training range.

The default experiment contains:

- 28 training scenarios
- 6 validation scenarios
- 6 held-out test scenarios
- 6 OOD scenarios
- 8 temporal windows per scenario

Seeds and all numerical and training parameters are stored in [`configs/default.json`](configs/default.json).

## Selection and Evaluation

Checkpoints are selected by validation objective. Test and OOD data are not used for optimization. Evaluation includes field-space, frequency-space, energy, discrete-physics, size, and latency measurements.

## Limitations

- The default spatial grids are intentionally small enough for CPU reproduction.
- Media are synthetic and dimensionless.
- The solver models a scalar acoustic equation rather than full elastic or fluid dynamics.
- Results do not establish real-time performance at production resolutions.
- OOD evaluation expands the tested distribution but cannot cover arbitrary geometries or boundary conditions.

## Intended Use

Suitable for reproducible scientific-ML study, architecture comparison, physics-aware loss experiments, and extension to larger PDE datasets. It should not be used for safety-critical acoustic design without independent numerical and experimental validation.
