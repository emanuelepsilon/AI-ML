from __future__ import annotations

import unittest

import torch

from wave_operator.config import ExperimentConfig
from wave_operator.losses import operator_loss
from wave_operator.model import build_model


class ModelTests(unittest.TestCase):
    def test_models_preserve_grid_shape(self) -> None:
        config = ExperimentConfig(grid_size=24, fno_width=12, fno_modes=5, fno_layers=2)
        inputs = torch.randn(2, 8, 24, 24)
        for name in ("fno", "cnn"):
            output = build_model(name, config)(inputs)
            self.assertEqual(tuple(output.shape), (2, 3, 24, 24))

    def test_physics_aware_objective_is_differentiable(self) -> None:
        config = ExperimentConfig(grid_size=16, fno_width=12, fno_modes=4, fno_layers=2)
        model = build_model("fno", config)
        features = torch.randn(2, 8, 16, 16)
        prediction = model(features)
        target = torch.randn_like(prediction)
        velocity = torch.ones(2, 1, 16, 16)
        damping = torch.zeros(2, 1, 16, 16)
        loss = operator_loss(prediction, target, velocity, damping, config)
        loss.total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
