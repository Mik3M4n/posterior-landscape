from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

from posterior_landscape import topology
from posterior_landscape.boundary import DomainSupport, boundary_contacts
from posterior_landscape.io import Grid, normalize_density
from posterior_landscape.quadrature import build_axis_quadrature
from posterior_landscape.topology import BranchFeature


def boundary_grid(number: int = 81, *, cells: bool = False, full: bool = False) -> Grid:
    mass = np.linspace(1.0, 9.0, number)
    axis = build_axis_quadrature(mass, "cell_centres" if cells else "nodes", "linear")
    mask = np.ones((number, number), dtype=bool) if full else mass[None, :] <= mass[:, None]
    return Grid(
        m1=mass, m2=mass.copy(), ell1=np.log(mass), ell2=np.log(mass),
        geometry1=mass.copy(), geometry2=mass.copy(), mask=mask,
        weight1=axis.weights("linear"), weight2=axis.weights("linear"),
        log_base=math.e, geometry="linear", feature_measure="linear",
        input_density_measure="linear", domain="full" if full else "ordered",
        ordered_domain=not full, grid_location=axis.grid_location, grid_measure="linear",
        axis1=axis, axis2=axis,
    )


def diagonal_density(grid: Grid) -> np.ndarray:
    x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
    along = 0.5 * (x + y)
    transverse = (x - y) / np.sqrt(2.0)
    peaks = (
        1.0 + 3.0 * np.exp(-0.5 * ((along - 3.0) / 0.5) ** 2)
        + 3.0 * np.exp(-0.5 * ((along - 7.0) / 0.5) ** 2)
    )
    return normalize_density(np.where(grid.mask, 0.1 + peaks * np.exp(-0.5 * (transverse / 0.4) ** 2), 0.0), grid)[0]


def diagonal_arm(grid: Grid, endpoint: float = 3.0, *, reverse: bool = False) -> BranchFeature:
    center = np.linspace(5.0, endpoint, 80)
    points = np.column_stack([center, center])
    if reverse:
        points = points[::-1]
    saddle = int(np.argmin(abs(grid.m1 - 5.0)))
    extremum = int(np.argmin(abs(grid.m1 - endpoint)))
    return BranchFeature(
        kind="ridge", saddle_index=(saddle, saddle), extremum_index=(extremum, extremum),
        points_geometry=points, points_mass=points.copy(), prominence=1.0,
        event_persistence=1.0, length=float(np.linalg.norm(points[-1] - points[0])),
        boundary=True, topology_points_geometry=points.copy(),
        saddle_boundary_type="mask", extremum_boundary_type="mask",
    )


class BoundaryRegionTests(unittest.TestCase):
    def test_nodal_ordered_boundary_is_a_zero_outward_extent(self) -> None:
        grid = boundary_grid()
        support = DomainSupport(grid)
        outward = support.ray_exit(np.array([5.0, 5.0]), np.array([-1.0, 1.0]) / np.sqrt(2.0))
        self.assertTrue(outward.bounded)
        self.assertEqual(outward.distance, 0.0)
        self.assertEqual(outward.boundary_type, "ordered_diagonal")
        self.assertFalse(support.contains(np.array([[4.99, 5.01]]))[0])

    def test_complete_diagonal_cell_has_its_nominal_outward_extent(self) -> None:
        grid = boundary_grid(cells=True)
        support = DomainSupport(grid)
        outward = support.ray_exit(np.array([5.0, 5.0]), np.array([-1.0, 1.0]) / np.sqrt(2.0))
        self.assertAlmostEqual(outward.distance, 0.05 * np.sqrt(2.0), places=12)
        self.assertEqual(outward.boundary_type, "ordered_cell_boundary")
        self.assertTrue(support.contains(np.array([[4.99, 5.01]]))[0])
        self.assertTrue(np.all(grid.mask.diagonal()))
        np.testing.assert_allclose(grid.cell_weights.diagonal(), 0.01)

    def test_support_ray_stops_before_a_hole_and_never_reenters(self) -> None:
        grid = boundary_grid(full=True)
        mask = grid.mask.copy()
        mask[40, :] = False
        grid = copy.copy(grid)
        object.__setattr__(grid, "mask", mask)
        exit_info = DomainSupport(grid).ray_exit(np.array([3.0, 5.0]), np.array([1.0, 0.0]))
        self.assertAlmostEqual(exit_info.distance, 1.95, places=12)
        self.assertEqual(exit_info.boundary_type, "mask")
        self.assertTrue(DomainSupport(grid).contains(np.array([[7.0, 5.0]]))[0])

    def test_nonuniform_log_cells_use_the_same_native_bins_in_both_geometries(self) -> None:
        native = np.array([0.0, 0.3, 1.0, 1.8, 2.3])
        mass = np.exp(native)
        axis = build_axis_quadrature(mass, "cell_centres", "log")
        mask = mass[None, :] <= mass[:, None]
        for geometry in ("log", "linear"):
            coordinates = native if geometry == "log" else mass
            grid = Grid(
                m1=mass, m2=mass.copy(), ell1=native, ell2=native.copy(),
                geometry1=coordinates, geometry2=coordinates.copy(), mask=mask,
                weight1=axis.weights("log"), weight2=axis.weights("log"),
                log_base=math.e, geometry=geometry, domain="ordered", ordered_domain=True,
                grid_location="cell_centres", grid_measure="log", axis1=axis, axis2=axis,
            )
            exit_info = DomainSupport(grid).ray_exit(
                np.repeat(coordinates[2], 2), np.array([-1.0, 1.0]) / np.sqrt(2.0)
            )
            edges = axis.edges(geometry)
            expected = min(coordinates[2] - edges[2], edges[3] - coordinates[2]) * np.sqrt(2.0)
            self.assertAlmostEqual(exit_info.distance, expected, places=12)
            self.assertEqual(exit_info.boundary_type, "ordered_cell_boundary")

    def test_node_outer_limit_and_ordered_boundary_contacts_are_distinct(self) -> None:
        grid = boundary_grid()
        self.assertEqual(boundary_contacts(np.array([5.0, 5.0]), grid), ("ordered_diagonal",))
        self.assertEqual(boundary_contacts(np.array([9.0, 5.0]), grid), ("upper_coordinate1",))

    def test_diagonal_ridge_uses_supported_one_sided_width(self) -> None:
        grid = boundary_grid()
        density = diagonal_density(grid)
        arm = diagonal_arm(grid)
        topology._branch_feature_geometry(arm, density, grid)
        self.assertTrue(arm.geometry_valid)
        self.assertTrue(arm.strength_valid)
        self.assertGreater(arm.width_median, 0.0)
        self.assertLess(arm.width_median, 0.60)
        self.assertGreater(arm.boundary_bounded_fraction, 0.95)
        self.assertTrue(np.all(arm.section_valid))
        outward = np.minimum(arm.left_width_profile, arm.right_width_profile)
        np.testing.assert_allclose(outward, 0.0, atol=1.0e-12)
        inward = np.maximum(arm.left_width_profile, arm.right_width_profile)
        np.testing.assert_allclose(arm.width_profile, inward)
        self.assertFalse(np.any(arm.feature_region_mask & ~grid.mask))
        self.assertAlmostEqual(arm.feature_probability, grid.integrate(np.where(arm.feature_region_mask, density, 0.0)))
        # At the peak, the existing saddle-level baseline fixes an analytic
        # transverse half-prominence crossing.  Verify a one-sided, not doubled,
        # width against that profile's known Gaussian form.
        peak = arm.points_geometry[-1, 0]
        amplitude = 1.0 + 3.0 * np.exp(-0.5 * ((peak - 3.0) / 0.5) ** 2) + 3.0 * np.exp(-0.5 * ((peak - 7.0) / 0.5) ** 2)
        base_amplitude = 1.0 + 6.0 * np.exp(-8.0)
        half_amplitude = 0.5 * (amplitude + base_amplitude)
        expected = 0.4 * np.sqrt(-2.0 * np.log(half_amplitude / amplitude))
        self.assertAlmostEqual(float(arm.width_profile[-1]), expected, delta=0.025)

    def test_full_cell_ridge_retains_complete_diagonal_cells(self) -> None:
        grid = boundary_grid(cells=True)
        density = diagonal_density(grid)
        arm = diagonal_arm(grid)
        topology._branch_feature_geometry(arm, density, grid)
        self.assertTrue(arm.geometry_valid)
        self.assertTrue(arm.strength_valid)
        self.assertGreater(float(np.nanmin(arm.width_profile)), 0.0)
        self.assertTrue(np.any(arm.section_left_boundary_bounded | arm.section_right_boundary_bounded))
        diagonal = np.diag(arm.feature_region_mask)
        self.assertGreater(np.count_nonzero(diagonal), 10)
        np.testing.assert_allclose(grid.cell_weights.diagonal()[diagonal], 0.01)
        self.assertAlmostEqual(arm.feature_probability, grid.integrate(np.where(arm.feature_region_mask, density, 0.0)))

    def test_reversing_the_normal_swaps_sides_without_changing_measurement(self) -> None:
        grid = boundary_grid()
        density = diagonal_density(grid)
        first, second = diagonal_arm(grid), diagonal_arm(grid, reverse=True)
        topology._branch_feature_geometry(first, density, grid)
        topology._branch_feature_geometry(second, density, grid)
        self.assertAlmostEqual(first.width_median, second.width_median, places=7)
        self.assertAlmostEqual(first.log_contrast, second.log_contrast, places=7)
        self.assertAlmostEqual(first.feature_probability, second.feature_probability, places=12)
        np.testing.assert_allclose(first.left_width_profile, second.right_width_profile[::-1], atol=1.0e-6)
        np.testing.assert_array_equal(first.feature_region_mask, second.feature_region_mask)

    def test_paired_diagonal_event_uses_supported_region_and_projection(self) -> None:
        grid = boundary_grid()
        density = diagonal_density(grid)
        arms = [diagonal_arm(grid, 3.0), diagonal_arm(grid, 7.0)]
        for arm in arms:
            topology._branch_feature_geometry(arm, density, grid)
        event = topology._curve_events(arms, density, grid)[0]
        self.assertTrue(event.region_valid)
        self.assertTrue(event.strength_valid)
        self.assertGreater(event.boundary_bounded_fraction, 0.95)
        self.assertFalse(np.any(event.window_mask & ~grid.mask))
        self.assertAlmostEqual(event.region_probability, grid.integrate(np.where(event.window_mask, density, 0.0)), places=12)
        self.assertAlmostEqual(float(event.projection_m1 @ grid.weight1), event.region_probability, places=7)
        self.assertAlmostEqual(float(event.projection_m2 @ grid.weight2), event.region_probability, places=7)
        self.assertTrue(np.isfinite(event.mass_centroid).all())
        self.assertTrue(np.isfinite(event.spatial_covariance_mass).all())

    def test_uniform_supported_profile_does_not_invent_contrast(self) -> None:
        grid = boundary_grid()
        density = normalize_density(grid.mask.astype(float), grid)[0]
        for amplitude in (1.0, 1.0e12):
            arm = diagonal_arm(grid)
            topology._branch_feature_geometry(arm, amplitude * density, grid)
            self.assertTrue(arm.geometry_valid)
            self.assertFalse(arm.strength_valid)
            self.assertTrue(math.isnan(arm.log_contrast))
            self.assertFalse(np.any(arm.section_strength_valid))
            self.assertGreater(arm.feature_probability, 0.0)

    def test_unsupported_correction_clears_a_previously_computed_window(self) -> None:
        grid = boundary_grid()
        arm = diagonal_arm(grid)
        arm.points_geometry[:, 1] += 0.01
        arm.geometry_valid = True
        arm.feature_region_mask = grid.mask.copy()
        arm.feature_probability = 1.0
        arm.width_left = arm.width_right = arm.width_median = 2.0
        arm.width_along_lower = arm.width_along_upper = 2.0
        arm.log_contrast = 1.0
        arm.section_valid = np.ones(40, dtype=bool)
        arm.log_contrast_profile = np.ones(40)
        sampled = arm.points_geometry[np.rint(np.linspace(0, 79, 40)).astype(int)]
        arm.half_left_geometry = sampled.copy()
        arm.half_right_geometry = sampled.copy()
        # Supply a completed old measurement as the first stage, then require
        # the real supported-domain stage to invalidate it.  This isolates the
        # migration failure: stale legacy masks/scalars must not be exported.
        with patch.object(topology, "_branch_feature_geometry_legacy", return_value=None):
            topology._branch_feature_geometry(arm, diagonal_density(grid), grid)
        self.assertFalse(arm.geometry_valid)
        self.assertFalse(arm.strength_valid)
        self.assertIsNone(arm.feature_region_mask)
        for key in ("feature_probability", "width_left", "width_right", "width_median", "width_along_lower", "width_along_upper", "log_contrast"):
            self.assertTrue(math.isnan(getattr(arm, key)), key)

    def test_legacy_diagonal_ridge_remains_unavailable(self) -> None:
        grid = boundary_grid()
        arm = diagonal_arm(grid)
        topology._branch_feature_geometry(arm, diagonal_density(grid), grid, ridge_region_policy="legacy_two_sided")
        self.assertFalse(arm.geometry_valid)
        self.assertEqual(arm.region_policy, "legacy_two_sided")

    def test_valid_interior_measurements_are_exactly_legacy(self) -> None:
        grid = boundary_grid(full=True)
        x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
        density = normalize_density(0.05 + (1.0 + 2.0 * np.exp(-0.5 * ((y - 6.0) / 0.4) ** 2)) * np.exp(-0.5 * ((x - 5.0) / 0.5) ** 2), grid)[0]
        points = np.column_stack([np.full(80, 5.0), np.linspace(3.0, 6.0, 80)])
        template = BranchFeature(kind="ridge", saddle_index=(40, 20), extremum_index=(40, 50), points_geometry=points, points_mass=points.copy(), prominence=1.0, event_persistence=1.0, length=3.0, boundary=False, topology_points_geometry=points.copy())
        legacy, supported = copy.deepcopy(template), copy.deepcopy(template)
        topology._branch_feature_geometry(legacy, density, grid, ridge_region_policy="legacy_two_sided")
        topology._branch_feature_geometry(supported, density, grid)
        self.assertTrue(legacy.geometry_valid)
        for key in ("feature_probability", "width_left", "width_right", "width_median", "width_along_lower", "width_along_upper", "log_contrast"):
            self.assertEqual(getattr(legacy, key), getattr(supported, key), key)
        for key in ("width_profile", "log_contrast_profile", "feature_region_mask"):
            np.testing.assert_array_equal(getattr(legacy, key), getattr(supported, key), err_msg=key)

    def test_lower_endpoint_peak_retains_domain_context(self) -> None:
        grid = boundary_grid()
        labels = boundary_contacts(np.array([1.0, 1.0]), grid)
        self.assertIn("ordered_diagonal", labels)
        self.assertIn("lower_coordinate1", labels)
        self.assertIn("lower_coordinate2", labels)
        x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
        density = normalize_density(np.where(grid.mask, 0.001 + np.exp(-((x - 1.0) ** 2 + (y - 1.0) ** 2)), 0.0), grid)[0]
        analysis = topology.analyze_field(density, grid, persistence_threshold=0.1, detect_shoulders=False, detect_plateaus=False, hessian_refinement=False)
        peak = next(point for point in analysis.points if point.kind == "peak")
        self.assertEqual(peak.index, (0, 0))
        self.assertTrue(peak.measurement_valid)
        self.assertGreater(peak.feature_probability, 0.0)


if __name__ == "__main__":
    unittest.main()
