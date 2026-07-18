from __future__ import annotations

import unittest

from wave_operator.config import ExperimentConfig
from wave_operator.data import build_dataset, coarse_config


class DataPipelineTests(unittest.TestCase):
    def test_coarse_and_fine_solvers_share_physical_time_step(self) -> None:
        config = ExperimentConfig(grid_size=24, coarse_grid_size=12, simulation_steps=10)
        self.assertAlmostEqual(coarse_config(config).dt, config.dt, places=10)

    def test_paired_dataset_shapes(self) -> None:
        config = ExperimentConfig(
            grid_size=20,
            coarse_grid_size=10,
            simulation_steps=12,
            samples_per_scenario=3,
            sponge_width=3,
        )
        dataset, runs = build_dataset(1, config, seed_offset=0)
        sample = dataset[0]
        self.assertEqual(tuple(sample["features"].shape), (8, 20, 20))
        self.assertEqual(tuple(sample["target"].shape), (3, 20, 20))
        self.assertEqual(runs[0].coarse_fields.shape, (12, 10, 10))


if __name__ == "__main__":
    unittest.main()
