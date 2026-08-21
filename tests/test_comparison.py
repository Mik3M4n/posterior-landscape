from __future__ import annotations

import csv
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from posterior_landscape.association import _circular_h0_nulls
from posterior_landscape.comparison import (
    _projection_on_one_dimensional_grid,
    _ProjectedFeature,
    _full_h0_information,
    _load_projected_features,
    _select_primary_candidates,
    _write_feature_family_comparison,
    _write_full_h0_draws,
)
from posterior_landscape.config import AssociationSettings


def _candidate(identifier: str, jaccard: float, support: float) -> dict[str, object]:
    feature = _ProjectedFeature(
        identifier=identifier,
        parent_identifier=identifier,
        kind="valley" if identifier.startswith("V") else "pit",
        selected=np.asarray([True]),
        bounds=np.asarray([[12.0, 20.0]]),
        mu1=np.asarray([16.0]),
        mu2=np.asarray([9.0]),
        projection=np.ones((1, 3)),
    )
    return {
        "one_id": "D2",
        "one_center": 16.0,
        "feature": feature,
        "jaccard": jaccard,
        "two_support": support,
    }


class ComparisonTests(unittest.TestCase):
    class _Group(dict):
        def __init__(self, *args, attrs=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.attrs = {} if attrs is None else attrs

    def test_projection_grid_alignment_preserves_geometric_gw_input(self) -> None:
        grid = np.geomspace(1.0, 100.0, 41)
        projection = np.arange(82, dtype=float).reshape(2, 41)
        aligned = _projection_on_one_dimensional_grid(projection, grid, grid)
        np.testing.assert_array_equal(aligned, projection)

    def test_projection_grid_alignment_interpolates_non_geometric_input(self) -> None:
        source = np.linspace(0.05, 1.0, 81)
        target = np.geomspace(source[0], source[-1], 201)
        projection = np.stack((source, source**2))
        aligned = _projection_on_one_dimensional_grid(
            projection, source, target
        )
        self.assertEqual(aligned.shape, (2, target.size))
        np.testing.assert_allclose(aligned[0], target)
        np.testing.assert_allclose(aligned[1], target**2, atol=4.0e-5)

    def test_secondary_projection_loader_uses_m2_arrays_and_bounds(self) -> None:
        number = 4
        feature = self._Group(
            {
                "present": np.ones(number, dtype=bool),
                "measurement_valid": np.ones(number, dtype=bool),
                "projected_bounds_mass": np.tile([2.0, 4.0, 1.0, 3.0], (number, 1)),
                "mass_centroid": np.tile([3.0, 2.0], (number, 1)),
                "projection_m2": np.arange(number * 5, dtype=float).reshape(number, 5),
            },
            attrs={"type": "peak"},
        )
        results = self._Group(
            {
                "features": self._Group({"P2D1": feature}),
                "events": self._Group(),
                "shoulders": self._Group(),
            }
        )
        loaded = _load_projected_features(results, "m2")
        self.assertEqual(len(loaded), 1)
        np.testing.assert_allclose(loaded[0].bounds[0], [1.0, 3.0])
        np.testing.assert_allclose(loaded[0].projection, feature["projection_m2"])

    def test_primary_match_ignores_better_overlap_below_minimum_support(self) -> None:
        low_support_pit = _candidate("D2D2", 0.417, 0.0176)
        supported_valley = _candidate("V2DE2-A", 0.397, 0.582)

        selected = _select_primary_candidates(
            [low_support_pit, supported_valley],
            ["D2"],
            minimum_support=0.05,
        )

        self.assertIs(selected["D2"], supported_valley)

    def test_no_supported_counterpart_is_not_forced(self) -> None:
        selected = _select_primary_candidates(
            [_candidate("D2D2", 0.417, 0.0176)],
            ["D2"],
            minimum_support=0.05,
        )

        self.assertNotIn("D2", selected)

    def test_full_h0_comparison_uses_one_common_draw_subset(self) -> None:
        rng = np.random.default_rng(21)
        number = 240
        h0 = rng.uniform(45.0, 95.0, number)
        one_mu = 35.0 - 0.20 * h0 + rng.normal(0.0, 1.0, number)
        two_mu1 = 34.0 - 0.18 * h0 + rng.normal(0.0, 1.1, number)
        two_mu2 = 25.0 - 0.12 * h0 + rng.normal(0.0, 1.0, number)
        chain_id = np.repeat(np.arange(4), number // 4)
        common = np.ones(number, dtype=bool)
        common[-40:] = False
        settings = AssociationSettings(
            knn=3,
            permutations=6,
            uncertainty_resamples=6,
            random_seed=9,
        )
        analysis_rng = np.random.default_rng(settings.random_seed)
        nulls = _circular_h0_nulls(
            h0, chain_id, settings.permutations, analysis_rng
        )

        record = _full_h0_information(
            h0,
            chain_id,
            one_mu,
            two_mu1,
            two_mu2,
            common,
            nulls,
            settings,
            analysis_rng,
        )

        for name in (
            "h0_one_d_mi_bits",
            "h0_full_2d_mi_bits",
            "h0_all_scales_mi_bits",
            "h0_full_2d_given_1d_cmi_bits",
            "h0_mu2_given_all_m1_cmi_bits",
            "h0_one_d_given_full_2d_cmi_bits",
        ):
            self.assertTrue(np.isfinite(record[name]), name)
        self.assertEqual(record["common_valid_n_yes"], 200)
        self.assertEqual(record["common_valid_n_no"], 40)
        self.assertEqual(len(record["common_draws_by_chain"].split(";")), 4)
        self.assertLessEqual(
            record["h0_mu2_given_all_m1_cmi_bits_lower"],
            record["h0_mu2_given_all_m1_cmi_bits_upper"],
        )

    def test_compact_full_h0_draw_export_is_aligned(self) -> None:
        number = 12
        data = {
            "P1": {
                "two_d_ID": "P2D1",
                "two_d_parent_ID": "P2D1",
                "two_d_type": "peak",
                "common": np.arange(number) % 2 == 0,
                "one_mu": np.arange(number, dtype=float),
                "two_mu1": np.arange(number, dtype=float) + 1.0,
                "two_mu2": np.arange(number, dtype=float) + 2.0,
            }
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            path = Path(temporary_text) / "draws.npz"
            _write_full_h0_draws(
                path,
                ["P1"],
                data,
                np.linspace(50.0, 90.0, number),
                np.repeat(np.arange(4), 3),
                np.full(number, 1.0 / number),
            )
            with np.load(path, allow_pickle=False) as output:
                self.assertEqual(output["common_valid"].shape, (1, number))
                self.assertEqual(output["two_d_mu2"].shape, (1, number))
                self.assertEqual(output["one_d_ID"].tolist(), ["P1"])

    def test_family_table_has_one_row_per_two_dimensional_identity(self) -> None:
        number = 12
        common = np.ones(number, dtype=bool)
        base = {
            "two_d_ID": "P2D1",
            "two_d_parent_ID": "P2D1",
            "two_d_type": "peak",
            "common": common,
            "two_mu1": np.linspace(9.0, 10.0, number),
            "two_mu2": np.linspace(7.0, 8.0, number),
        }
        primary_m1 = {
            "P1": {**base, "one_mu": np.linspace(8.8, 9.8, number)},
            "S1": {
                **base,
                "common": np.arange(number) < 6,
                "one_mu": np.linspace(9.1, 10.1, number),
            },
        }
        primary_m2 = {
            "P2": {**base, "one_mu": np.linspace(6.8, 7.8, number)}
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            paths = _write_feature_family_comparison(
                Path(temporary_text),
                {},
                primary_m1,
                primary_m2,
                {},
                {},
                np.full(number, 1.0 / number),
                None,
                [],
                SimpleNamespace(association=AssociationSettings()),
            )
            csv_path = next(path for path in paths if path.suffix == ".csv")
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["family_ID"], "P2D1")
            self.assertEqual(rows[0]["m1_1d_ID"], "P1")
            self.assertEqual(rows[0]["m1_candidate_IDs"], "P1;S1")


if __name__ == "__main__":
    unittest.main()
