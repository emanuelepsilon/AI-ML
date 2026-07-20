from __future__ import annotations

import unittest

import numpy as np

from wave_operator.config import ExperimentConfig
from wave_operator.physics import create_scenario, laplacian, simulate


class PhysicsTests(unittest.TestCase):
    def test_laplacian_of_constant_is_zero(self) -> None:
        field = np.ones((12, 12), dtype=np.float32)
        result = laplacian(field, dx=0.1)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1.0e-6)

    def test_simulation_is_finite_and_respects_boundaries(self) -> None:
        config = ExperimentConfig(grid_size=20, simulation_steps=16, sponge_width=3)
        scenario = create_scenario(config, seed=12)
        fields = simulate(scenario, config)
        self.assertEqual(fields.shape, (16, 20, 20))
        self.assertTrue(np.isfinite(fields).all())
        np.testing.assert_allclose(fields[:, 0, :], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(fields[:, :, -1], 0.0, atol=1.0e-7)

    def test_ood_medium_differs_from_training_family(self) -> None:
        config = ExperimentConfig(grid_size=24, simulation_steps=12, sponge_width=3)
        standard = create_scenario(config, seed=44, ood=False)
        ood = create_scenario(config, seed=44, ood=True)
        self.assertGreater(float(np.mean(np.abs(standard.velocity - ood.velocity))), 0.01)
        self.assertGreater(float(np.mean(np.abs(standard.damping - ood.damping))), 0.001)

    def test_training_scenarios_vary_boundary_profiles(self) -> None:
        config = ExperimentConfig(grid_size=24, simulation_steps=12, sponge_width=4)
        first = create_scenario(config, seed=4, ood=False)
        second = create_scenario(config, seed=9, ood=False)
        self.assertGreater(float(np.mean(np.abs(first.damping - second.damping))), 0.0001)


if __name__ == "__main__":
    unittest.main()
