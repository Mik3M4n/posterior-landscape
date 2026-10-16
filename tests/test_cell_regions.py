"""Finite-volume regressions for the production integration consumers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from posterior_landscape import topology
from posterior_landscape.config import InputSettings
from posterior_landscape.ensemble import PointTracker, feature_geometry_maps
from posterior_landscape.io import (
    _domain_mask,
    density_in_feature_measure,
    input_to_feature_factor,
    normalize_density,
    open_density_store,
    quadrature_weights,
    validate_store,
)


def _expected_edges(centres: np.ndarray) -> np.ndarray:
    """Independent reference for the specified midpoint/exterior construction."""

    return np.concatenate(
        (
            [centres[0] - (centres[1] - centres[0]) / 2.0],
            (centres[:-1] + centres[1:]) / 2.0,
            [centres[-1] + (centres[-1] - centres[-2]) / 2.0],
        )
    )


class CellRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.counter = 0

    def grid(
        self,
        first: np.ndarray,
        second: np.ndarray | None = None,
        *,
        mask: np.ndarray | None = None,
        location: str | None = "cell_centres",
        native: str = "log",
        input_measure: str = "log",
        feature_measure: str = "log",
        domain: str = "full",
        log_base: float = np.e,
    ):
        first = np.asarray(first, dtype=float)
        second = first.copy() if second is None else np.asarray(second, dtype=float)
        mask = np.ones((first.size, second.size), bool) if mask is None else mask
        self.counter += 1
        path = self.directory / f"field_{self.counter}.npz"
        np.savez(path, p=np.ones((1, first.size, second.size)), m1=first, m2=second,
                 mask=mask, weights=np.ones(1))
        options = dict(file=path, domain=domain, density_measure=input_measure,
                       grid_measure=native)
        if location is not None:
            options["grid_location"] = location
        settings = InputSettings(**options)
        with open_density_store(path, input_settings=settings) as store:
            return validate_store(store, log_base=log_base, geometry="log",
                                  feature_measure=feature_measure,
                                  input_settings=settings)

    def test_inclusive_pixelpop_mask_retains_full_diagonal_weights(self) -> None:
        spacing = 0.053
        centres = np.exp(np.arange(100) * spacing)
        mask = centres[None, :] <= centres[:, None]
        grid = self.grid(centres, mask=mask, domain="ordered")
        self.assertEqual(np.count_nonzero(grid.mask), 5050)
        self.assertEqual(np.count_nonzero(np.diag(grid.mask)), 100)
        np.testing.assert_array_equal(grid.mask, mask)
        np.testing.assert_allclose(grid.cell_weights[mask], spacing**2,
                                   rtol=2e-13, atol=1e-16)
        np.testing.assert_allclose(np.diag(grid.cell_weights), spacing**2,
                                   rtol=2e-13, atol=1e-16)
        density = np.where(mask, 1.0 / (5050 * spacing**2), 0.0)
        self.assertAlmostEqual(grid.integrate(density), 1.0, places=13)
        first, second = grid.marginals(density)
        self.assertAlmostEqual(float(first @ grid.weight1), 1.0, places=13)
        self.assertAlmostEqual(float(second @ grid.weight2), 1.0, places=13)
        nodal = self.grid(centres, mask=mask, domain="ordered", location="nodes")
        self.assertGreater(abs(nodal.integrate(density) - 1.0), 0.01)

    def test_ordered_mask_accepts_roundoff_and_rejects_real_violation(self) -> None:
        first = np.asarray([1.0, 2.0, 3.0])
        second = np.nextafter(first, np.inf)
        mask = np.tri(3, dtype=bool)
        np.testing.assert_array_equal(_domain_mask(first, second, mask, "ordered"),
                                      mask)
        np.testing.assert_array_equal(_domain_mask(first, second, None, "ordered"),
                                      mask)
        invalid = mask.copy()
        invalid[0, 1] = True
        with self.assertRaisesRegex(ValueError, "coordinate2 <= coordinate1"):
            _domain_mask(first, second, invalid, "ordered")
        visible_difference = first + 1e-5
        with self.assertRaisesRegex(ValueError, "coordinate2 <= coordinate1"):
            _domain_mask(first, visible_difference, mask, "ordered")

    def test_cell_measure_conversion_preserves_every_cell_before_normalization(self) -> None:
        for native in ("log", "linear"):
            for base in (np.e, 10.0):
                with self.subTest(native=native, base=base):
                    first = np.exp(np.asarray([0.2, 0.5, 1.0, 1.8]))
                    second = np.exp(np.asarray([0.1, 0.6, 1.2]))
                    grid = self.grid(first, second, native=native,
                                     input_measure="log", feature_measure="linear",
                                     log_base=base)
                    if native == "log":
                        edges1 = _expected_edges(np.log(first) / np.log(base))
                        edges2 = _expected_edges(np.log(second) / np.log(base))
                        physical1, physical2 = base**edges1, base**edges2
                        logarithmic1, logarithmic2 = edges1, edges2
                    else:
                        physical1, physical2 = _expected_edges(first), _expected_edges(second)
                        logarithmic1 = np.log(physical1) / np.log(base)
                        logarithmic2 = np.log(physical2) / np.log(base)
                    input_area = np.multiply.outer(np.diff(logarithmic1),
                                                   np.diff(logarithmic2))
                    target_area = np.multiply.outer(np.diff(physical1),
                                                    np.diff(physical2))
                    density = np.arange(1, 13, dtype=float).reshape(4, 3)
                    density /= np.sum(density * input_area)
                    converted = density_in_feature_measure(density, grid)
                    np.testing.assert_allclose(grid.input_cell_weights, input_area,
                                               rtol=2e-14, atol=1e-15)
                    np.testing.assert_allclose(grid.cell_weights, target_area,
                                               rtol=2e-14, atol=1e-15)
                    np.testing.assert_allclose(converted * target_area,
                                               density * input_area,
                                               rtol=2e-14, atol=1e-15)
                    np.testing.assert_allclose(grid.linear_weight1, np.diff(physical1),
                                               rtol=2e-14, atol=1e-15)
                    np.testing.assert_allclose(grid.linear_weight2, np.diff(physical2),
                                               rtol=2e-14, atol=1e-15)
                    self.assertAlmostEqual(grid.integrate(converted), 1.0, places=14)
                    normalized, factor = normalize_density(converted, grid)
                    self.assertAlmostEqual(factor, 1.0, places=14)
                    self.assertAlmostEqual(grid.integrate(normalized), 1.0, places=14)
                    reverse_grid = self.grid(first, second, native=native,
                                             input_measure="linear", feature_measure="log",
                                             log_base=base)
                    # Switching the density measure must retain the very same bins.
                    np.testing.assert_array_equal(reverse_grid.input_cell_weights,
                                                  grid.cell_weights)
                    np.testing.assert_array_equal(reverse_grid.cell_weights,
                                                  grid.input_cell_weights)
                    reversed_density = density_in_feature_measure(converted, reverse_grid)
                    np.testing.assert_allclose(reversed_density, density,
                                               rtol=2e-14, atol=1e-15)
                    self.assertAlmostEqual(reverse_grid.integrate(reversed_density),
                                           1.0, places=14)

    def test_constant_density_on_linear_native_rectangle(self) -> None:
        first = np.asarray([2.0, 4.0, 7.0])
        second = np.asarray([3.0, 5.0])
        grid = self.grid(first, second, native="linear", input_measure="linear",
                         feature_measure="linear")
        # Physical edges are [1, 3, 5.5, 8.5] and [2, 4, 6].
        np.testing.assert_array_equal(grid.weight1, np.asarray([2.0, 2.5, 3.0]))
        np.testing.assert_array_equal(grid.weight2, np.asarray([2.0, 2.0]))
        density = np.full(grid.shape, 1.0 / 30.0)
        self.assertAlmostEqual(grid.integrate(density), 1.0, places=14)
        first_marginal, second_marginal = grid.marginals(density)
        np.testing.assert_allclose(first_marginal, np.full(3, 4.0/30.0), rtol=1e-14)
        np.testing.assert_allclose(second_marginal, np.full(2, 7.5/30.0), rtol=1e-14)

    def test_nodal_measure_conversion_and_weights_remain_pointwise(self) -> None:
        first = np.exp(np.asarray([0.2, 0.5, 1.0, 1.8]))
        second = np.exp(np.asarray([0.1, 0.6, 1.2]))
        for base in (np.e, 10.0):
            reference = self.grid(first, second, location=None, native="linear",
                                  feature_measure="linear", log_base=base)
            alternate = self.grid(first, second, location="nodes", native="log",
                                  feature_measure="linear", log_base=base)
            expected_factor = 1.0 / (np.multiply.outer(first, second) * np.log(base)**2)
            np.testing.assert_array_equal(input_to_feature_factor(reference), expected_factor)
            np.testing.assert_array_equal(input_to_feature_factor(alternate), expected_factor)
            np.testing.assert_array_equal(reference.weight1, quadrature_weights(first))
            np.testing.assert_array_equal(reference.weight2, quadrature_weights(second))
            np.testing.assert_array_equal(reference.weight1, alternate.weight1)
            np.testing.assert_array_equal(reference.weight2, alternate.weight2)
            density = np.arange(1, 13, dtype=float).reshape(4, 3)
            np.testing.assert_array_equal(density_in_feature_measure(density, reference),
                                          density * expected_factor)
            np.testing.assert_array_equal(density_in_feature_measure(density, reference),
                                          density_in_feature_measure(density, alternate))

    def test_hpd_and_smoothing_use_finite_cell_probability(self) -> None:
        centres = np.exp(np.asarray([0.1, 0.3, 0.8, 1.6]))
        mask = centres[None, :] <= centres[:, None]
        grid = self.grid(centres, mask=mask, domain="ordered")
        values = np.where(mask, np.arange(1, 17, dtype=float).reshape(4, 4), 0.0)
        density, _ = normalize_density(values, grid)
        active = np.flatnonzero(mask)
        order = active[np.argsort(density.ravel()[active])[::-1]]
        mass = (density * grid.cell_weights).ravel()[order]
        for probability in (0.1, 0.5, 0.9):
            position = min(np.searchsorted(np.cumsum(mass), probability), len(order)-1)
            expected = density.ravel()[order[position]]
            self.assertEqual(topology.hpd_threshold(density, probability, grid), expected)
            self.assertEqual(topology._hpd_threshold_from_descending_order(
                density, probability, grid, order), expected)
        for scale in (0.0, 0.2):
            smoothed = topology.smooth_density(density, grid, scale)
            self.assertAlmostEqual(grid.integrate(smoothed), 1.0, places=13)
            np.testing.assert_array_equal(smoothed[~mask], 0.0)

    def test_density_and_deficit_region_moments_and_projections(self) -> None:
        centres = np.exp(np.asarray([0.2, 0.5, 1.0, 1.8]))
        mask = centres[None, :] <= centres[:, None]
        grid = self.grid(centres, mask=mask, domain="ordered")
        region = np.ones(grid.shape, bool)
        region[0, 0] = False
        density = np.arange(1, 17, dtype=float).reshape(4, 4)
        deficit = np.maximum(12.0 - density, 0.0)
        x, y = np.meshgrid(centres, centres, indexing="ij")
        for values in (density, deficit):
            with self.subTest(deficit=values is deficit):
                selected = region & mask
                measured = topology._region_measurements(values, region, grid)
                self.assertIsNotNone(measured)
                total, centroid, covariance, _, _, projected1, projected2 = measured
                masses = np.where(selected, values * grid.cell_weights, 0.0)
                expected_total = float(masses.sum())
                expected_centroid = np.asarray([(masses*x).sum(), (masses*y).sum()])/expected_total
                dx, dy = x-expected_centroid[0], y-expected_centroid[1]
                expected_covariance = np.asarray([(masses*dx*dx).sum(),
                                                  (masses*dx*dy).sum(),
                                                  (masses*dy*dy).sum()])/expected_total
                self.assertAlmostEqual(total, expected_total, places=13)
                np.testing.assert_allclose(centroid, expected_centroid, rtol=2e-14)
                np.testing.assert_allclose(covariance, expected_covariance, rtol=2e-14)
                self.assertAlmostEqual(float(projected1 @ grid.weight1), total, delta=total*1e-7)
                self.assertAlmostEqual(float(projected2 @ grid.weight2), total, delta=total*1e-7)
                self.assertAlmostEqual(float((projected1*centres) @ grid.weight1)/total,
                                       centroid[0], delta=abs(centroid[0])*1e-7)
                self.assertAlmostEqual(float((projected2*centres) @ grid.weight2)/total,
                                       centroid[1], delta=abs(centroid[1])*1e-7)

    def test_peak_and_pit_regions_use_native_bin_physical_area(self) -> None:
        centres = np.exp(np.asarray([0.2, 0.5, 1.0, 1.8]))
        grid = self.grid(centres)
        tree = topology._TreeResult(extrema={}, events=[], flow=np.empty(0, int),
                                    elder={}, global_extremum=0, order=np.empty(0, int))
        expected_region = np.zeros(grid.shape, bool)
        expected_region[1:3, 1:3] = True
        physical_widths = np.diff(np.exp(_expected_edges(np.log(centres))))
        expected_area = np.multiply.outer(physical_widths, physical_widths)[expected_region].sum()
        for kind in ("peak", "pit"):
            with self.subTest(kind=kind):
                values = np.ones(grid.shape)
                values[1:3, 1:3] = ([[8.0, 6.0], [7.0, 5.0]] if kind == "peak"
                                    else [[0.05, 0.15], [0.25, 0.35]])
                density, _ = normalize_density(values, grid)
                point = topology.PointFeature(kind=kind, index=(1, 1),
                                              value=float(density[1, 1]), persistence=1.0,
                                              boundary=False, saddle_index=(0, 0))
                topology._point_feature_geometry(point, 5, tree, density, grid)
                self.assertTrue(point.geometry_valid)
                self.assertTrue(point.measurement_valid)
                np.testing.assert_array_equal(point.feature_region_mask, expected_region)
                self.assertAlmostEqual(point.width_area, expected_area, places=13)
                self.assertAlmostEqual(point.feature_probability,
                                       float((density*grid.cell_weights)[expected_region].sum()),
                                       places=14)
                measurement_density = (density if kind == "peak" else
                                       np.maximum(density[0, 0]-density, 0.0))
                expected = topology._region_measurements(measurement_density,
                                                         expected_region, grid)
                np.testing.assert_allclose(point.mass_centroid, expected[1], rtol=2e-14)
                np.testing.assert_allclose(point.spatial_covariance_mass, expected[2], rtol=2e-14)
                np.testing.assert_array_equal(point.projection_m1, expected[5])
                np.testing.assert_array_equal(point.projection_m2, expected[6])
                if kind == "pit":
                    self.assertAlmostEqual(point.deficit_probability, expected[0], places=14)

    def test_location_density_and_hpd_use_physical_cell_area(self) -> None:
        centres = np.exp(np.asarray([0.2, 0.5, 1.0, 1.8]))
        grid = self.grid(centres)
        template = topology.PointFeature(kind="peak", index=(1, 1), value=1.0,
                                         persistence=1.0, boundary=False,
                                         identifier="P2D1")
        tracker = PointTracker(template, 3, grid.shape)
        tracker.present[:] = True
        tracker.geometry_valid[:] = True
        tracker.location[:] = [[grid.geometry1[i], grid.geometry2[j]]
                               for i, j in ((0, 0), (1, 2), (3, 3))]
        tracker.region_weight_sum[1:3, 1:3] = 0.8
        maps = feature_geometry_maps([tracker], grid, np.asarray([0.2, 0.3, 0.5]),
                                     credible_mass=0.9)["P2D1"]
        widths = np.diff(np.exp(_expected_edges(np.log(centres))))
        area = np.multiply.outer(widths, widths)
        np.testing.assert_allclose(maps.location_density * area,
                                   maps.location_probability, rtol=1e-7, atol=1e-9)
        self.assertAlmostEqual(float((maps.location_density*area).sum()), 1.0,
                               places=7)
        self.assertAlmostEqual(float(maps.location_probability.sum()), 1.0,
                               places=7)
        for probability, actual in ((0.5, maps.location_hpd_50),
                                    (0.9, maps.location_hpd_credible)):
            order = np.argsort(maps.location_density.ravel())[::-1]
            count = int(np.searchsorted(np.cumsum(maps.location_probability.ravel()[order]),
                                        probability))+1
            expected = np.zeros(grid.shape, bool)
            expected.ravel()[order[:count]] = True
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
