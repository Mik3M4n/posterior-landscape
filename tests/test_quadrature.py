"""Native-bin construction, measure conservation, and finite-cell CDFs."""

import math
import unittest

import numpy as np

from posterior_landscape.io import quadrature_weights
from posterior_landscape.quadrature import (
    SpatialQuadrature,
    build_axis_quadrature,
    cell_cdf,
    cell_interval_weights,
    cell_joint_tails,
    cell_percentile,
    conservative_remap,
    inferred_cell_edges,
)


class QuadratureTests(unittest.TestCase):
    def test_uniform_and_nonuniform_linear_native_cells(self):
        uniform = build_axis_quadrature([1., 3., 5.], "cell_centres", "linear")
        np.testing.assert_array_equal(uniform.edges("linear"), [0., 2., 4., 6.])
        np.testing.assert_array_equal(uniform.weights("linear"), [2., 2., 2.])
        axis = build_axis_quadrature([2., 3., 7., 9.], "cell_centres", "linear")
        np.testing.assert_array_equal(axis.edges("linear"), [1.5, 2.5, 5., 8., 10.])
        np.testing.assert_array_equal(axis.weights("linear"), [1., 2.5, 3., 2.])
        np.testing.assert_allclose(axis.edges("log"), np.log(axis.edges("linear")))

    def test_nonuniform_log_native_and_alternate_base(self):
        u = np.array([.2, .5, 1.2, 1.5])
        for base in (math.e, 10.):
            axis = build_axis_quadrature(base ** u, "cell_centres", "log", base)
            expected = np.array([.05, .35, .85, 1.35, 1.65])
            np.testing.assert_allclose(axis.edges("log"), expected, atol=5e-16)
            np.testing.assert_allclose(axis.edges("linear"), base ** expected)
            np.testing.assert_allclose(axis.weights("linear"), np.diff(base ** expected))
            np.testing.assert_allclose(axis.weights("log", math.e), np.diff(expected) * math.log(base))

    def test_exact_mass_conservation_before_normalization(self):
        for native in ("linear", "log"):
            axis = build_axis_quadrature([2., 3., 8., 14.], "cell_centres", native, 10.)
            f = np.array([[.8, 2., .5, 4.], [.1, .9, 3., .6]])
            original = axis.edges("linear")
            for source, target in (("linear", "log"), ("log", "linear")):
                old_mass = f * axis.weights(source)
                converted = f * axis.conversion_factor(source, target)
                np.testing.assert_allclose(converted * axis.weights(target), old_mass, rtol=2e-15)
                np.testing.assert_allclose(np.sum(converted * axis.weights(target), axis=1), old_mass.sum(axis=1), rtol=2e-15)
                np.testing.assert_array_equal(axis.edges("linear"), original)
                roundtrip = converted * axis.conversion_factor(target, source)
                np.testing.assert_allclose(roundtrip, f, rtol=2e-15)

    def test_nodal_weights_exact_and_grid_measure_inactive(self):
        x = np.array([1., 2., 5., 9.])
        expected = np.array([.5, 2., 3.5, 2.])
        np.testing.assert_array_equal(quadrature_weights(x), expected)
        for native in ("linear", "log"):
            axis = build_axis_quadrature(x, "nodes", native, 10.)
            np.testing.assert_array_equal(axis.weights("linear"), expected)
            np.testing.assert_array_equal(axis.weights("log"), quadrature_weights(np.log(x) / np.log(10.)))
            self.assertIsNone(axis.native_edges)
            with self.assertRaises(ValueError):
                axis.conversion_factor("log", "linear")

    def test_linear_cell_cdf_partial_intervals_and_percentiles(self):
        axis = build_axis_quadrature([1., 2., 3.], "cell_centres", "linear")
        f = np.ones((2, 3)) / 3.
        np.testing.assert_allclose(cell_cdf(f, axis, "linear", [.5, 1., 2., 3.5]), [[0., 1./6., .5, 1.]] * 2)
        np.testing.assert_allclose(cell_interval_weights(axis, "linear", 1., 2.25), [.5, .75, 0.])
        np.testing.assert_allclose(cell_percentile(f, axis, "linear", .5), [2., 2.])
        np.testing.assert_allclose(cell_percentile(f, axis, "linear", .5, lower=2.5), [3., 3.])
        np.testing.assert_allclose(cell_percentile(f, axis, "linear", 0.), [.5, .5])
        np.testing.assert_allclose(cell_percentile(f, axis, "linear", 1.), [3.5, 3.5])

    def test_log_cell_cdf_and_single_occupied_cell(self):
        axis = build_axis_quadrature(np.exp([0., 1., 2.]), "cell_centres", "log")
        f = np.array([0., 1., 0.])
        np.testing.assert_allclose(cell_percentile(f, axis, "log", .5), math.e)
        np.testing.assert_allclose(cell_cdf(f, axis, "log", [0., np.exp(.5), math.e, np.exp(1.5)]), [0., 0., .5, 1.])

    def test_joint_tails_integrate_full_diagonal_cell(self):
        axis = build_axis_quadrature([1., 3., 5.], "cell_centres", "linear")
        mask = axis.centres[None, :] <= axis.centres[:, None]
        spatial = SpatialQuadrature(axis, axis, mask)
        density = np.zeros((2, 3, 3))
        density[:, 1, 1] = .25  # Unit probability in full [2,4] x [2,4] bin.
        any_tail, both, straddle = cell_joint_tails(density, spatial, "linear", [2., 3., 4.])
        np.testing.assert_allclose(any_tail, [[1., .75, 0.]] * 2)
        np.testing.assert_allclose(both, [[1., .25, 0.]] * 2)
        np.testing.assert_allclose(straddle, [[0., .5, 0.]] * 2)
        # Half the full cell above coordinate2=coordinate1 is retained.
        np.testing.assert_allclose(both + straddle, any_tail)
        self.assertEqual(int(mask.sum()), 6)

    def test_conservative_remapping_preserves_probability_and_extent(self):
        source = build_axis_quadrature([1., 2., 3.], "cell_centres", "linear")
        target = build_axis_quadrature([1.25, 2.75], "cell_centres", "linear")
        f = np.array([[1., 2., 3.], [3., 0., 1.]])
        mapped = conservative_remap(f, source, target, "linear")
        np.testing.assert_allclose(mapped, [[4./3., 8./3.], [2., 2./3.]])
        np.testing.assert_allclose((mapped * target.weights("linear")).sum(axis=1), (f * source.weights("linear")).sum(axis=1))
        wrong = build_axis_quadrature([1., 3.], "cell_centres", "linear")
        with self.assertRaisesRegex(ValueError, "extent"):
            conservative_remap(f, source, wrong, "linear")

    def test_invalid_inputs_and_unavailable_log_bins(self):
        for invalid in ([1.], [1., 1.], [2., 1.], [1., np.nan]):
            with self.assertRaises(ValueError):
                inferred_cell_edges(invalid)
        with self.assertRaises(ValueError):
            build_axis_quadrature([-1., 2.], "cell_centres", "log")
        axis = build_axis_quadrature([.1, 1.], "cell_centres", "linear")
        with self.assertRaisesRegex(ValueError, "positive physical"):
            axis.weights("log")
        with self.assertRaises(ValueError):
            build_axis_quadrature([1., 2.], "cell_centres", "log", native_edges=[0., 0., 1.])


if __name__ == "__main__":
    unittest.main()
