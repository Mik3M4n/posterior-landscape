from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from posterior_landscape.association import (
    AssociationFeature,
    analyze_h0_associations,
    detect_location_configurations,
    expand_location_configurations,
    make_h0_corner_figures,
    pairwise_indicator_associations,
)
from posterior_landscape.config import AssociationSettings, PlotSettings


class AssociationTests(unittest.TestCase):
    @staticmethod
    def _location_feature(mu1: np.ndarray, mu2: np.ndarray) -> AssociationFeature:
        number = mu1.size
        present = np.ones(number, dtype=bool)
        return AssociationFeature(
            identifier="V2DE1",
            kind="valley",
            morphology=present,
            location=present,
            region=present,
            mu1=mu1,
            mu2=mu2,
            region_probability=np.full(number, 0.02),
            relative_width_m1=np.full(number, 0.3),
            relative_width_m2=np.full(number, 0.4),
            contrast=np.full(number, 0.5),
            extent=np.full(number, 1.0),
            bounded_fraction=np.full(number, 0.8),
            fallback_fraction=np.zeros(number),
            deficit_probability=np.full(number, 0.01),
            match_distance=np.full(number, 0.2),
            match_margin=np.full(number, 1.0),
            match_ambiguous=np.zeros(number, dtype=bool),
        )

    def test_stable_location_modes_are_split_without_hard_coded_identity(self) -> None:
        rng = np.random.default_rng(18)
        first = rng.normal((np.log(14.5), np.log(8.5)), (0.055, 0.055), (900, 2))
        second = rng.normal((np.log(27.0), np.log(16.5)), (0.045, 0.045), (180, 2))
        locations = np.exp(np.vstack([first, second]))
        feature = self._location_feature(locations[:, 0], locations[:, 1])
        weights = np.full(locations.shape[0], 1.0 / locations.shape[0])

        detected = detect_location_configurations([feature], weights)["V2DE1"]
        self.assertTrue(detected.multimodal)
        self.assertEqual(detected.configuration_labels, ("A", "B"))
        self.assertGreaterEqual(detected.minimum_label_agreement, 0.95)
        self.assertAlmostEqual(detected.probabilities[0], 5.0 / 6.0, delta=0.03)
        self.assertAlmostEqual(detected.probabilities[1], 1.0 / 6.0, delta=0.03)

        expanded = expand_location_configurations([feature], {"V2DE1": detected})
        self.assertEqual([item.identifier for item in expanded], ["V2DE1", "V2DE1-A", "V2DE1-B"])
        self.assertTrue(expanded[0].suppress_continuous)
        self.assertEqual(expanded[1].record_scope, "location_configuration")
        self.assertFalse(np.any(expanded[1].region & expanded[2].region))

    def test_single_location_cloud_remains_unsplit(self) -> None:
        rng = np.random.default_rng(19)
        locations = np.exp(
            rng.normal((np.log(18.0), np.log(10.0)), (0.10, 0.09), (1000, 2))
        )
        feature = self._location_feature(locations[:, 0], locations[:, 1])
        weights = np.full(locations.shape[0], 1.0 / locations.shape[0])

        detected = detect_location_configurations([feature], weights)["V2DE1"]
        self.assertFalse(detected.multimodal)
        expanded = expand_location_configurations([feature], {"V2DE1": detected})
        self.assertEqual([item.identifier for item in expanded], ["V2DE1"])

    def test_joint_centroid_and_conditional_h0_associations(self) -> None:
        rng = np.random.default_rng(42)
        number = 240
        h0 = rng.uniform(55.0, 95.0, number)
        mu1 = 32.0 - 0.20 * h0 + rng.normal(0.0, 1.1, number)
        mu2 = 23.0 - 0.13 * h0 + rng.normal(0.0, 1.0, number)
        present = np.ones(number, dtype=bool)
        feature = AssociationFeature(
            identifier="P2D1",
            kind="peak",
            morphology=present,
            location=present,
            region=present,
            mu1=mu1,
            mu2=mu2,
            region_probability=rng.uniform(0.1, 0.3, number),
            relative_width_m1=rng.uniform(0.1, 0.4, number),
            relative_width_m2=rng.uniform(0.1, 0.4, number),
            contrast=rng.uniform(0.2, 1.0, number),
            extent=np.full(number, np.nan),
            bounded_fraction=np.ones(number),
            fallback_fraction=np.full(number, np.nan),
            deficit_probability=np.full(number, np.nan),
        )
        records = analyze_h0_associations(
            [feature],
            h0,
            np.repeat(np.arange(4), number // 4),
            AssociationSettings(
                knn=5,
                permutations=8,
                uncertainty_resamples=8,
                random_seed=7,
            ),
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertLess(record["mu1_rho"], -0.5)
        self.assertLess(record["mu2_rho"], -0.4)
        self.assertGreater(record["mu_vector_mi_bits"], 0.1)
        self.assertTrue(np.isfinite(record["mu2_given_mu1_cmi_bits"]))
        self.assertTrue(np.isfinite(record["cov_h0_mu1"]))
        self.assertEqual(record["mu_vector_n"], number)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            paths = make_h0_corner_figures(
                Path(temporary_text) / "corner",
                [feature],
                records,
                h0,
                PlotSettings(write_pdf=False, write_png=True),
            )
            self.assertEqual([path.name for path in paths], ["corner.png"])
            self.assertGreater(paths[0].stat().st_size, 0)

    def test_pairwise_location_support_reports_coexistence_without_merging(self) -> None:
        first = np.asarray([1, 1, 1, 0, 0], dtype=bool)
        second = np.asarray([1, 1, 0, 1, 0], dtype=bool)
        records = pairwise_indicator_associations(
            ["D2", "P3"],
            ["dip", "peak"],
            [first, second],
            np.full(5, 0.2),
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record["ID_A"], record["ID_B"]), ("D2", "P3"))
        self.assertAlmostEqual(record["P_A_and_B"], 0.4)
        self.assertAlmostEqual(record["P_B_given_A"], 2.0 / 3.0)
        self.assertAlmostEqual(record["P_A_given_B"], 2.0 / 3.0)
        self.assertTrue(np.isfinite(record["phi"]))


if __name__ == "__main__":
    unittest.main()
