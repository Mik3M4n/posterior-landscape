from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from posterior_landscape import topology
from posterior_landscape.ensemble import (
    PosteriorCatalogue,
    _branch_compatibility,
    assign_reference_identifiers,
    compute_ensemble_summary,
    feature_geometry_maps,
    remove_corner_boundary_pits,
    retain_scale_persistent_reference,
    weighted_quantile,
)
from posterior_landscape.io import (
    Grid,
    _domain_mask,
    normalize_density,
    open_density_store,
    quadrature_weights,
    validate_store,
)
from posterior_landscape.plotting import (
    _CurveArmFigureData,
    _curve_event_features,
)
from posterior_landscape.topology import (
    BranchFeature,
    CurveEventFeature,
    PointFeature,
    analyze_field,
    hpd_threshold,
)


def synthetic_grid(number: int = 41) -> Grid:
    ell = np.linspace(0.0, 2.0, number)
    mass = 10.0**ell
    mask = mass[None, :] < mass[:, None]
    weights = quadrature_weights(ell)
    return Grid(
        m1=mass,
        m2=mass.copy(),
        ell1=ell,
        ell2=ell.copy(),
        geometry1=ell.copy(),
        geometry2=ell.copy(),
        mask=mask,
        weight1=weights,
        weight2=weights.copy(),
        log_base=10.0,
        geometry="log",
    )


def synthetic_density(grid: Grid, shift: float = 0.0) -> np.ndarray:
    x, y = np.meshgrid(grid.geometry1, grid.geometry2, indexing="ij")
    first = np.exp(-0.5 * (((x - 0.62 - shift) / 0.13) ** 2 + ((y - 0.22) / 0.12) ** 2))
    second = 0.75 * np.exp(
        -0.5 * (((x - 1.52 + 0.3 * shift) / 0.16) ** 2 + ((y - 0.84) / 0.15) ** 2)
    )
    return normalize_density(np.where(grid.mask, first + second + 2e-5, 0.0), grid)[0]


def directional_shoulder_density(
    grid: Grid, *, center: float | None = 2.5
) -> np.ndarray:
    u, v = np.meshgrid(np.log(grid.m1), np.log(grid.m2), indexing="ij")
    common = 0.5 * (u + v)
    ratio = 0.5 * (u - v)
    log_density = -1.4 * common - 0.25 * ratio
    if center is not None:
        log_density += 0.9 * np.exp(-0.5 * ((common - center) / 0.34) ** 2)
    density = np.where(
        grid.mask,
        np.exp(log_density) * np.exp(-0.5 * ((ratio - 0.38) / 0.20) ** 2)
        + 1.0e-14,
        0.0,
    )
    return normalize_density(density, grid)[0]


def rectangular_linear_grid(number: int = 81) -> Grid:
    mass = np.linspace(1.0, 9.0, number)
    ell = np.log(mass)
    weights = quadrature_weights(ell)
    return Grid(
        m1=mass,
        m2=mass.copy(),
        ell1=ell,
        ell2=ell.copy(),
        geometry1=mass.copy(),
        geometry2=mass.copy(),
        mask=np.ones((number, number), dtype=bool),
        weight1=weights,
        weight2=weights.copy(),
        log_base=np.e,
        geometry="linear",
    )


class CoreTests(unittest.TestCase):
    @staticmethod
    def _test_branch(
        extremum: tuple[int, int],
        *,
        scale_count: int,
        saddle_boundary_type: str = "none",
        extremum_boundary_type: str = "none",
    ) -> BranchFeature:
        points = np.asarray([[0.5, 0.5], extremum], dtype=float)
        return BranchFeature(
            kind="ridge",
            saddle_index=(5, 5),
            extremum_index=extremum,
            points_geometry=points,
            points_mass=points.copy(),
            prominence=1.0,
            event_persistence=1.0,
            length=1.0,
            boundary=(
                saddle_boundary_type != "none"
                or extremum_boundary_type != "none"
            ),
            topology_points_geometry=points.copy(),
            scale_count=scale_count,
            scale_total=3,
            saddle_boundary_type=saddle_boundary_type,
            extremum_boundary_type=extremum_boundary_type,
        )

    def test_reference_catalogue_excludes_one_of_three_scale_peak(self) -> None:
        stable_first = PointFeature(
            "peak", (8, 7), 2.0, 1.0, False, scale_count=3, scale_total=3
        )
        unstable = PointFeature(
            "peak", (9, 7), 1.8, 0.3, False, scale_count=1, scale_total=3
        )
        stable_second = PointFeature(
            "peak", (16, 12), 1.0, 0.2, False, scale_count=2, scale_total=3
        )
        saddle = PointFeature(
            "saddle", (5, 5), 0.5, 0.2, False, scale_count=3, scale_total=3
        )
        branches = [
            self._test_branch((8, 7), scale_count=3),
            self._test_branch((9, 7), scale_count=1),
            self._test_branch((16, 12), scale_count=2),
        ]
        event_with_unstable_arm = CurveEventFeature(
            "ridge",
            (5, 5),
            (0, 1),
            np.zeros((3, 2)),
            np.zeros((0, 2)),
        )
        stable_event = CurveEventFeature(
            "ridge",
            (5, 5),
            (0, 2),
            np.zeros((3, 2)),
            np.zeros((0, 2)),
        )
        reference = type("Reference", (), {})()
        reference.points = [stable_first, unstable, stable_second, saddle]
        reference.branches = branches
        reference.events = [event_with_unstable_arm, stable_event]

        result = retain_scale_persistent_reference(reference)

        self.assertEqual(result["required"], 2)
        self.assertEqual(result["points_removed"], 1)
        self.assertEqual(result["branches_removed"], 1)
        self.assertEqual(
            [item.index for item in reference.points if item.kind == "peak"],
            [(8, 7), (16, 12)],
        )
        self.assertEqual(len(reference.events), 1)
        self.assertEqual(reference.events[0].arm_indices, (0, 1))

    def test_branch_matching_does_not_substitute_boundary_connectivity(self) -> None:
        interior = self._test_branch((8, 7), scale_count=3)
        boundary = self._test_branch(
            (8, 7), scale_count=3, extremum_boundary_type="mask"
        )
        compatible = _branch_compatibility([interior], [boundary])
        self.assertEqual(compatible.tolist(), [[False]])

    def test_corner_boundary_pit_removes_only_incident_valley(self) -> None:
        corner = PointFeature(
            "pit", (1, 1), 1.0, 1.0, True, boundary_type="outer+mask"
        )
        mask_edge = PointFeature(
            "pit", (2, 2), 0.9, 0.8, True, boundary_type="mask"
        )
        peak = PointFeature("peak", (8, 8), 2.0, 1.0, False)
        saddle_corner = PointFeature(
            "saddle", (3, 3), 0.5, 0.4, True, boundary_type="mask"
        )
        saddle_interior = PointFeature("saddle", (5, 5), 0.4, 0.3, False)
        corner_branch = BranchFeature(
            kind="valley",
            saddle_index=(3, 3),
            extremum_index=(1, 1),
            points_geometry=np.zeros((2, 2)),
            points_mass=np.zeros((2, 2)),
            prominence=1.0,
            event_persistence=1.0,
            length=1.0,
            boundary=True,
            topology_points_geometry=np.zeros((2, 2)),
            extremum_boundary_type="outer+mask",
        )
        retained_branch = BranchFeature(
            kind="valley",
            saddle_index=(5, 5),
            extremum_index=(2, 2),
            points_geometry=np.zeros((2, 2)),
            points_mass=np.zeros((2, 2)),
            prominence=1.0,
            event_persistence=1.0,
            length=1.0,
            boundary=True,
            topology_points_geometry=np.zeros((2, 2)),
            extremum_boundary_type="mask",
        )
        corner_event = CurveEventFeature(
            "valley", (3, 3), (0, 1), np.zeros((3, 2)), np.zeros((0, 2))
        )
        retained_event = CurveEventFeature(
            "valley", (5, 5), (1, 1), np.zeros((3, 2)), np.zeros((0, 2))
        )
        reference = type("Reference", (), {})()
        reference.points = [
            corner,
            mask_edge,
            peak,
            saddle_corner,
            saddle_interior,
        ]
        reference.branches = [corner_branch, retained_branch]
        reference.events = [corner_event, retained_event]

        result = remove_corner_boundary_pits(reference)

        self.assertEqual(result["corner_pits_removed"], 1)
        self.assertEqual(result["branches_removed"], 1)
        self.assertEqual(result["events_removed"], 1)
        self.assertNotIn(corner, reference.points)
        self.assertIn(mask_edge, reference.points)
        self.assertEqual(len(reference.branches), 1)
        self.assertEqual(len(reference.events), 1)

    def test_automatic_persistence_ignores_terminal_gap(self) -> None:
        threshold = topology.automatic_persistence_threshold(
            [0.01, 0.04, 0.05, 5.0], 1.0
        )
        self.assertAlmostEqual(threshold, 0.02)

    def test_automatic_persistence_requires_configured_log_gap(self) -> None:
        values = [0.01, 0.025, 0.03]
        conservative = topology.automatic_persistence_threshold(values, 1.0)
        permissive = topology.automatic_persistence_threshold(
            values, 1.0, persistence_gap_min_log=0.8
        )
        self.assertAlmostEqual(conservative, 0.00325)
        self.assertAlmostEqual(permissive, np.sqrt(0.01 * 0.025))

    def test_automatic_persistence_is_invariant_to_terminal_outlier(self) -> None:
        first = topology.automatic_persistence_threshold(
            [0.01, 0.04, 0.05, 5.0], 1.0
        )
        second = topology.automatic_persistence_threshold(
            [0.01, 0.04, 0.05, 500.0], 1.0
        )
        self.assertAlmostEqual(first, second)

    def test_automatic_persistence_scales_with_density(self) -> None:
        values = np.asarray([0.01, 0.04, 0.05, 5.0])
        first = topology.automatic_persistence_threshold(values, 1.0)
        second = topology.automatic_persistence_threshold(7.0 * values, 7.0)
        self.assertAlmostEqual(second, 7.0 * first)

    def test_automatic_persistence_tie_uses_lower_threshold(self) -> None:
        threshold = topology.automatic_persistence_threshold(
            [1.0, 2.0, 4.0, 100.0],
            100.0,
            persistence_gap_min_log=0.5,
        )
        self.assertAlmostEqual(threshold, np.sqrt(2.0))

    def test_global_extrema_are_excluded_from_threshold_calibration(self) -> None:
        common = dict(
            events=[],
            flow=np.empty(0, dtype=int),
            elder={},
            order=np.empty(0, dtype=int),
        )
        join_tree = topology._TreeResult(
            extrema={
                1: {"persistence": 100.0},
                2: {"persistence": 0.04},
            },
            global_extremum=1,
            **common,
        )
        split_tree = topology._TreeResult(
            extrema={
                3: {"persistence": 80.0},
                4: {"persistence": 0.03},
            },
            global_extremum=3,
            **common,
        )
        self.assertEqual(
            topology._finite_nonessential_persistences(join_tree, split_tree),
            [0.04, 0.03],
        )

    def test_ordered_domain_accepts_a_restricted_ordered_mask(self) -> None:
        first = np.asarray([1.0, 2.0, 3.0])
        second = np.asarray([0.5, 1.5, 2.5])
        ordered = second[None, :] < first[:, None]
        supplied = ordered.copy()
        supplied[-1, 0] = False
        np.testing.assert_array_equal(
            _domain_mask(first, second, supplied, "ordered"), supplied
        )
        outside = supplied.copy()
        outside[0, 1] = True
        with self.assertRaisesRegex(ValueError, "coordinate2 < coordinate1"):
            _domain_mask(first, second, outside, "ordered")

    def test_directional_power_law_has_no_shoulder(self) -> None:
        grid = synthetic_grid(81)
        analysis = analyze_field(
            directional_shoulder_density(grid, center=None),
            grid,
            scale=0.025,
            persistence_threshold=0.001,
            detect_plateaus=False,
            curve_points=48,
            shoulder_alpha_threshold=2.0,
        )
        self.assertEqual(analysis.shoulders, [])

    def test_directional_shoulder_is_bounded_and_thresholded(self) -> None:
        grid = synthetic_grid(81)
        density = directional_shoulder_density(grid)
        analysis = analyze_field(
            density,
            grid,
            scale=0.025,
            persistence_threshold=0.001,
            detect_plateaus=False,
            curve_points=48,
            shoulder_alpha_threshold=2.0,
        )
        self.assertEqual(len(analysis.shoulders), 1)
        shoulder = analysis.shoulders[0]
        self.assertTrue(shoulder.curve_available)
        self.assertTrue(shoulder.region_valid)
        self.assertEqual(shoulder.center_curve_geometry.shape, (48, 2))
        self.assertGreater(shoulder.slope_contrast, 0.0)
        self.assertGreater(shoulder.region_probability, 0.0)
        self.assertLess(shoulder.alpha_max, 2.0)

        rejected = analyze_field(
            density,
            grid,
            scale=0.025,
            persistence_threshold=0.001,
            detect_plateaus=False,
            curve_points=48,
            shoulder_alpha_threshold=0.1,
        )
        self.assertEqual(rejected.shoulders, [])

    def test_directional_shoulder_is_scale_stable_and_can_truncate(self) -> None:
        grid = synthetic_grid(81)
        locations = []
        for scale in (0.0, 0.025, 0.05):
            analysis = analyze_field(
                directional_shoulder_density(grid),
                grid,
                scale=scale,
                persistence_threshold=0.001,
                detect_plateaus=False,
                curve_points=48,
                shoulder_alpha_threshold=2.0,
            )
            self.assertEqual(len(analysis.shoulders), 1)
            locations.append(np.mean(analysis.shoulders[0].topology_geometry, axis=0))
        self.assertLess(np.max(np.ptp(np.asarray(locations), axis=0)), 0.12)

        truncated = analyze_field(
            directional_shoulder_density(grid, center=4.35),
            grid,
            scale=0.025,
            persistence_threshold=0.001,
            detect_plateaus=False,
            curve_points=48,
            shoulder_alpha_threshold=2.0,
        )
        self.assertEqual(len(truncated.shoulders), 1)
        self.assertTrue(truncated.shoulders[0].low_density_truncated)
        self.assertLess(truncated.shoulders[0].bounded_fraction, 1.0)

    def test_linear_feature_measure_converts_per_log_input(self) -> None:
        mass = np.exp(np.linspace(np.log(1.0), np.log(9.0), 41))
        x, y = np.meshgrid(mass, mass, indexing="ij")
        mask = y < x
        linear_grid = Grid(
            m1=mass,
            m2=mass.copy(),
            ell1=np.log(mass),
            ell2=np.log(mass),
            geometry1=np.log(mass),
            geometry2=np.log(mass),
            mask=mask,
            weight1=quadrature_weights(mass),
            weight2=quadrature_weights(mass),
            log_base=np.e,
            geometry="log",
            feature_measure="linear",
        )
        target = np.where(
            mask,
            np.exp(-0.5 * (((x - 5.0) / 1.2) ** 2 + ((y - 2.5) / 0.8) ** 2)),
            0.0,
        )
        target = normalize_density(target, linear_grid)[0]
        per_log = target * x * y

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            path = Path(temporary_text) / "linear_measure_input.npz"
            np.savez(
                path,
                p=per_log[None, :, :],
                m1=mass,
                m2=mass,
                mask=mask,
                log_base=np.e,
            )
            with open_density_store(path) as store:
                grid = validate_store(
                    store,
                    log_base=np.e,
                    geometry="log",
                    feature_measure="linear",
                )
                summary = compute_ensemble_summary(
                    store,
                    grid,
                    credible_mass=0.90,
                    batch_size=1,
                )

        np.testing.assert_allclose(grid.weight1, quadrature_weights(mass))
        np.testing.assert_allclose(summary.reference_density, target, rtol=1e-12)
        expected_marginal1, expected_marginal2 = grid.marginals(target)
        np.testing.assert_allclose(
            summary.marginal1_quantiles[1], expected_marginal1, rtol=1e-12
        )
        np.testing.assert_allclose(
            summary.marginal2_quantiles[1], expected_marginal2, rtol=1e-12
        )
        self.assertAlmostEqual(grid.integrate(summary.reference_density), 1.0)
        self.assertEqual(summary.tail_probability, 0.999)
        self.assertEqual(summary.tail_m1_scale.shape, (1,))
        self.assertEqual(summary.tail_m2_scale.shape, (1,))
        self.assertTrue(np.all(np.isfinite(summary.tail_m1_scale)))
        self.assertTrue(np.all(np.isfinite(summary.tail_m2_scale)))
        np.testing.assert_allclose(
            summary.tail_both_fraction_at_m1_scale
            + summary.tail_straddle_fraction_at_m1_scale,
            1.0,
            atol=1.0e-12,
        )

    def test_log_measure_normalization_and_marginals(self) -> None:
        grid = synthetic_grid()
        density = synthetic_density(grid)
        first, second = grid.marginals(density)
        self.assertAlmostEqual(grid.integrate(density), 1.0, places=12)
        self.assertAlmostEqual(float(first @ grid.weight1), 1.0, places=12)
        self.assertAlmostEqual(float(second @ grid.weight2), 1.0, places=12)

    def test_two_peak_merge_tree_and_separatrices(self) -> None:
        grid = synthetic_grid()
        analysis = analyze_field(
            synthetic_density(grid),
            grid,
            persistence_threshold=0.02,
            hessian_refinement=True,
            detect_plateaus=False,
            curve_points=48,
        )
        peaks = [item for item in analysis.points if item.kind == "peak"]
        ridges = [item for item in analysis.branches if item.kind == "ridge"]
        self.assertEqual(len(peaks), 2)
        self.assertGreaterEqual(len(ridges), 2)
        self.assertAlmostEqual(sum(item.region_mass for item in peaks), 1.0, places=10)
        for branch in analysis.branches:
            self.assertEqual(branch.points_geometry.shape, (48, 2))
            self.assertTrue(np.all(np.isfinite(branch.points_geometry)))
            self.assertGreaterEqual(branch.prominence, 0.0)
        valid_ridges = [item for item in ridges if item.geometry_valid]
        self.assertTrue(valid_ridges)
        self.assertTrue(
            any(item.fallback_width_fraction > 0.0 for item in valid_ridges)
        )

    def test_gaussian_peak_half_prominence_widths(self) -> None:
        grid = rectangular_linear_grid(101)
        x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
        density = np.exp(
            -0.5 * (((x - 5.0) / 1.0) ** 2 + ((y - 5.0) / 0.5) ** 2)
        )
        density = normalize_density(density, grid)[0]
        analysis = analyze_field(
            density,
            grid,
            persistence_threshold=0.1,
            hessian_refinement=False,
            detect_plateaus=False,
            curve_points=24,
        )
        peak = next(item for item in analysis.points if item.kind == "peak")
        expected_major = 2.0 * np.sqrt(2.0 * np.log(2.0))
        expected_minor = 0.5 * expected_major
        self.assertTrue(peak.geometry_valid)
        self.assertAlmostEqual(peak.width_major, expected_major, delta=0.04)
        self.assertAlmostEqual(peak.width_minor, expected_minor, delta=0.04)
        self.assertEqual(peak.width_reference, "zero_density")
        self.assertGreater(peak.feature_probability, 0.0)
        self.assertTrue(peak.measurement_valid)
        self.assertAlmostEqual(peak.mass_centroid[0], 5.0, delta=0.08)
        self.assertAlmostEqual(peak.mass_centroid[1], 5.0, delta=0.08)
        self.assertAlmostEqual(
            float(peak.projection_m1 @ grid.weight1),
            peak.feature_probability,
            places=7,
        )

    def test_peak_width_is_grid_convergent(self) -> None:
        widths: list[tuple[float, float]] = []
        for number in (51, 101):
            grid = rectangular_linear_grid(number)
            x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
            density = np.exp(
                -0.5 * (((x - 5.0) / 1.0) ** 2 + ((y - 5.0) / 0.5) ** 2)
            )
            analysis = analyze_field(
                normalize_density(density, grid)[0],
                grid,
                persistence_threshold=0.1,
                hessian_refinement=False,
                detect_plateaus=False,
                curve_points=24,
            )
            peak = next(item for item in analysis.points if item.kind == "peak")
            widths.append((peak.width_major, peak.width_minor))
        np.testing.assert_allclose(widths[0], widths[1], atol=0.03, rtol=0.06)

    def test_interior_valley_has_two_sided_half_depth_width(self) -> None:
        grid = rectangular_linear_grid(81)
        x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
        density = (
            1.0
            + 3.0 * np.exp(-0.5 * ((x - 3.0) / 0.5) ** 2)
            + 3.0 * np.exp(-0.5 * ((x - 7.0) / 0.5) ** 2)
        ) * (1.0 + 0.05 * np.exp(-0.5 * ((y - 5.0) / 2.0) ** 2))
        density = normalize_density(density, grid)[0]
        points = np.column_stack(
            [np.full(80, 5.0), np.linspace(2.0, 8.0, 80)]
        )
        valley = BranchFeature(
            kind="valley",
            saddle_index=(40, 20),
            extremum_index=(40, 60),
            points_geometry=points,
            points_mass=points.copy(),
            prominence=1.0,
            event_persistence=1.0,
            length=6.0,
            boundary=False,
            topology_points_geometry=points.copy(),
        )
        topology._branch_feature_geometry(valley, density, grid)
        self.assertTrue(valley.geometry_valid)
        self.assertAlmostEqual(valley.valid_width_fraction, 1.0)
        self.assertAlmostEqual(valley.width_median, 2.80, delta=0.12)
        self.assertGreater(valley.log_contrast, 1.0)
        self.assertGreater(valley.feature_probability, 0.0)

    def test_paired_valley_event_uses_deficit_weighted_window(self) -> None:
        grid = rectangular_linear_grid(81)
        x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
        density = (
            1.0
            + 3.0 * np.exp(-0.5 * ((x - 3.0) / 0.5) ** 2)
            + 3.0 * np.exp(-0.5 * ((x - 7.0) / 0.5) ** 2)
        ) * (1.0 + 0.05 * np.exp(-0.5 * ((y - 5.0) / 2.0) ** 2))
        density = normalize_density(density, grid)[0]

        arms: list[BranchFeature] = []
        for endpoint, extremum in ((2.0, (40, 10)), (8.0, (40, 70))):
            points = np.column_stack(
                [np.full(40, 5.0), np.linspace(5.0, endpoint, 40)]
            )
            arm = BranchFeature(
                kind="valley",
                saddle_index=(40, 40),
                extremum_index=extremum,
                points_geometry=points,
                points_mass=points.copy(),
                prominence=1.0,
                event_persistence=1.0,
                length=3.0,
                boundary=False,
                topology_points_geometry=points.copy(),
                full_length=3.0,
            )
            topology._branch_feature_geometry(arm, density, grid)
            self.assertTrue(arm.geometry_valid)
            arms.append(arm)

        events = topology._curve_events(arms, density, grid)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertTrue(event.region_valid)
        self.assertGreater(event.deficit_probability, 0.0)
        self.assertAlmostEqual(event.mass_centroid[0], 5.0, delta=0.16)
        self.assertAlmostEqual(
            float(event.projection_m1 @ grid.weight1),
            event.deficit_probability,
            places=7,
        )
        self.assertGreater(event.longest_bounded_fraction, 0.90)

    def test_hpd_threshold_contains_requested_mass(self) -> None:
        grid = synthetic_grid()
        density = synthetic_density(grid)
        threshold = hpd_threshold(density, 0.9, grid)
        enclosed = grid.integrate(np.where(density >= threshold, density, 0.0))
        self.assertGreaterEqual(enclosed, 0.9)

    def test_curve_smoothing_preserves_endpoints_and_reduces_roughness(self) -> None:
        grid = synthetic_grid(81)
        coordinate = np.linspace(0.55, 1.55, 96)
        oscillation = 0.035 * np.where(np.arange(coordinate.size) % 2, 1.0, -1.0)
        points = np.column_stack([coordinate, 0.28 + oscillation])
        smoothed = topology._smooth_curve(points, grid, smoothing_cells=2.0)
        np.testing.assert_allclose(smoothed[[0, -1]], points[[0, -1]])
        original_roughness = np.sum(np.linalg.norm(np.diff(points, n=2, axis=0), axis=1))
        smooth_roughness = np.sum(
            np.linalg.norm(np.diff(smoothed, n=2, axis=0), axis=1)
        )
        self.assertLess(smooth_roughness, 0.25 * original_roughness)

    def test_valleys_are_truncated_at_the_hpd_support_boundary(self) -> None:
        grid = synthetic_grid(51)
        analysis = analyze_field(
            synthetic_density(grid),
            grid,
            persistence_threshold=0.02,
            hessian_refinement=True,
            detect_plateaus=False,
            curve_points=64,
            support_mass=0.95,
            curve_smoothing_cells=2.0,
        )
        threshold = hpd_threshold(analysis.density, 0.95, grid)
        valleys = [item for item in analysis.branches if item.kind == "valley"]
        self.assertTrue(valleys)
        self.assertTrue(any(item.low_density_truncated for item in valleys))
        for valley in valleys:
            self.assertLessEqual(valley.length, valley.full_length + 1e-12)
            self.assertGreaterEqual(valley.retained_fraction, 0.0)
            self.assertLessEqual(valley.retained_fraction, 1.0)
            if valley.curve_available:
                values = topology._sample_field(
                    analysis.density, valley.points_geometry, grid
                )
                self.assertGreaterEqual(float(np.min(values)), threshold - 1e-9)
            else:
                self.assertAlmostEqual(valley.length, 0.0, places=12)
                self.assertFalse(valley.geometry_valid)

    def test_posterior_feature_maps_are_conditional_and_normalized(self) -> None:
        grid = synthetic_grid(31)
        analyses = [
            analyze_field(
                synthetic_density(grid, shift=shift),
                grid,
                persistence_threshold=0.02,
                hessian_refinement=False,
                detect_plateaus=False,
                curve_points=24,
            )
            for shift in (-0.02, 0.0, 0.02)
        ]
        reference = analyses[1]
        assign_reference_identifiers(reference, grid)
        weights = np.full(3, 1.0 / 3.0)
        catalogue = PosteriorCatalogue.from_reference(reference, 3, grid)
        for draw, analysis in enumerate(analyses):
            catalogue.ingest(
                draw,
                analysis,
                draw_weight=float(weights[draw]),
                discover=False,
            )
        trackers = [
            item
            for item in catalogue.point_trackers
            if item.template.kind == "peak"
        ]
        maps = feature_geometry_maps(
            trackers, grid, weights, credible_mass=0.90
        )
        for tracker in trackers:
            spatial = maps[tracker.template.identifier]
            self.assertAlmostEqual(float(spatial.location_probability.sum()), 1.0)
            self.assertTrue(np.any(spatial.location_hpd_50))
            self.assertTrue(np.any(spatial.location_hpd_credible))
            self.assertTrue(
                np.all(
                    spatial.location_hpd_50
                    <= spatial.location_hpd_credible
                )
            )
            self.assertGreaterEqual(
                float(spatial.region_inclusion_conditional.max()), 0.0
            )
            self.assertLessEqual(
                float(spatial.region_inclusion_conditional.max()), 1.0 + 1e-6
            )

    def test_paired_curve_event_has_coherent_pointwise_corridor(self) -> None:
        number_draws = 101
        number_points = 15
        offsets = np.linspace(-0.2, 0.2, number_draws)
        first_reference = np.column_stack(
            [np.linspace(1.0, 0.0, number_points), np.ones(number_points)]
        )
        second_reference = np.column_stack(
            [np.linspace(1.0, 2.0, number_points), np.ones(number_points)]
        )

        def curves(reference: np.ndarray) -> np.ndarray:
            result = np.broadcast_to(
                reference, (number_draws, number_points, 2)
            ).copy()
            result[:, :, 1] += offsets[:, None]
            return result

        topology_starts = np.broadcast_to(
            np.asarray([1.0, 1.0]), (number_draws, 2)
        ).copy()
        topology_starts[:, 1] += offsets
        present = np.ones(number_draws, dtype=bool)
        second_geometry = present.copy()
        second_geometry[81:] = False
        first = _CurveArmFigureData(
            identifier="V2D1",
            kind="valley",
            boundary_limited=False,
            reference_curve_geometry=first_reference,
            present=present,
            curve_available=present,
            geometry_valid=present,
            topology_start_geometry=topology_starts,
            curves_geometry=curves(first_reference),
        )
        second = _CurveArmFigureData(
            identifier="V2D2",
            kind="valley",
            boundary_limited=False,
            reference_curve_geometry=second_reference,
            present=present,
            curve_available=present,
            geometry_valid=second_geometry,
            topology_start_geometry=topology_starts,
            curves_geometry=curves(second_reference),
        )
        events = _curve_event_features(
            [first, second],
            weights=np.full(number_draws, 1.0 / number_draws),
            credible_mass=0.90,
            typical_spacing=0.1,
            geometry="linear",
            log_base=np.e,
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.identifier, "V2DE1")
        self.assertEqual(event.arm_identifiers, ("V2D1", "V2D2"))
        self.assertAlmostEqual(event.support, 1.0)
        self.assertAlmostEqual(event.curve_support, 1.0)
        self.assertAlmostEqual(event.geometry_support, 81.0 / number_draws)
        self.assertEqual(event.median_curve_mass.shape, (29, 2))
        self.assertTrue(
            np.all(
                event.pointwise_credible_lower_mass[:, 1]
                <= event.pointwise_50_lower_mass[:, 1]
            )
        )
        self.assertTrue(
            np.all(
                event.pointwise_credible_upper_mass[:, 1]
                >= event.pointwise_50_upper_mass[:, 1]
            )
        )
        self.assertGreaterEqual(event.simultaneous_inflation, 1.0)
        self.assertLess(event.simultaneous_inflation, 1.1)
        self.assertGreaterEqual(event.simultaneous_coverage, 0.89)

    def test_weighted_quantile(self) -> None:
        values = np.asarray([0.0, 10.0, 20.0])
        weights = np.asarray([0.1, 0.8, 0.1])
        result = weighted_quantile(values, (0.1, 0.5, 0.9), weights)
        np.testing.assert_allclose(result, (0.0, 10.0, 10.0))

    @unittest.skipIf(topology._numba is None, "Numba is not installed")
    def test_compiled_tree_matches_reference_tree(self) -> None:
        grid = synthetic_grid(31)
        density = synthetic_density(grid)
        compiled = analyze_field(
            density,
            grid,
            persistence_threshold=0.02,
            hessian_refinement=False,
            detect_plateaus=False,
            curve_points=32,
        )
        numba_module = topology._numba
        try:
            topology._numba = None
            reference = analyze_field(
                density,
                grid,
                persistence_threshold=0.02,
                hessian_refinement=False,
                detect_plateaus=False,
                curve_points=32,
            )
        finally:
            topology._numba = numba_module
        compiled_points = sorted((item.kind, item.index) for item in compiled.points)
        reference_points = sorted((item.kind, item.index) for item in reference.points)
        compiled_branches = sorted(
            (item.kind, item.saddle_index, item.extremum_index)
            for item in compiled.branches
        )
        reference_branches = sorted(
            (item.kind, item.saddle_index, item.extremum_index)
            for item in reference.branches
        )
        self.assertEqual(compiled_points, reference_points)
        self.assertEqual(compiled_branches, reference_branches)
        np.testing.assert_array_equal(compiled.peak_labels, reference.peak_labels)
        np.testing.assert_array_equal(compiled.pit_labels, reference.pit_labels)


if __name__ == "__main__":
    unittest.main()
