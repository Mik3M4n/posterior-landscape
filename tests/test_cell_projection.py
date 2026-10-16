from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from posterior_landscape.comparison import (
    _aligned_projection,
    _assert_quadrature_compatibility,
    _projection_capture,
    _projection_in_one_d_measure,
    _signature,
)
from posterior_landscape.config import (
    AnalysisSettings,
    AssociationSettings,
    InputSettings,
    load_settings,
)
from posterior_landscape.io import h5py
from posterior_landscape.pipeline import run
from posterior_landscape.quadrature import (
    QUADRATURE_SCHEMA_VERSION,
    build_axis_quadrature,
)


def _settings(location="cell_centres", measure="log"):
    return SimpleNamespace(
        input=InputSettings(
            file=Path("unused.npz"), grid_location=location, grid_measure=measure
        ),
        analysis=AnalysisSettings(),
        association=AssociationSettings(),
    )


def _state(axis, measure="log"):
    return {
        "m_grid": axis.centres,
        "CELL_QUADRATURE_AXIS": axis,
        "MASS_DENSITY_MEASURE": measure,
        "MASS_MEASURE_CELL_EDGES": axis.edges(measure, log_base=math.e),
        "MASS_MEASURE_CELL_WIDTHS": axis.weights(measure, log_base=math.e),
        "MASS_MEASURE_WIDTHS": axis.weights(measure, log_base=math.e),
        "grid_location": "cell_centres",
        "grid_measure": "log",
        "quadrature_schema_version": QUADRATURE_SCHEMA_VERSION,
    }


class CellProjectionTests(unittest.TestCase):
    def test_nodal_conversion_retains_centre_jacobian(self):
        grid = np.asarray([1.0, 2.5, 8.0])
        density = np.asarray([[2.0, 3.0, 5.0]])
        np.testing.assert_array_equal(
            _projection_in_one_d_measure(
                density, grid, two_d_measure="log", one_d_measure="linear",
                log_base=10.0,
            ),
            density / math.log(10.0) / grid[None, :],
        )
        np.testing.assert_array_equal(
            _projection_in_one_d_measure(
                density, grid, two_d_measure="linear", one_d_measure="log",
                log_base=10.0,
            ),
            density * grid[None, :],
        )

    def test_cell_projection_conversion_conserves_each_bin_mass(self):
        grid = np.exp([0.0, 0.5, 1.8, 3.0])
        axis = build_axis_quadrature(grid, "cell_centres", "log", 10.0)
        density = np.asarray([[1.0, 4.0, 2.0, 0.5], [0.2, 0.3, 7.0, 1.0]])
        converted = _projection_in_one_d_measure(
            density, grid, two_d_measure="log", one_d_measure="linear",
            log_base=10.0, cell_axis=axis,
        )
        np.testing.assert_allclose(
            converted * axis.weights("linear")[None, :],
            density * axis.weights("log", log_base=10.0)[None, :],
            rtol=3e-15,
        )
        natural = _projection_in_one_d_measure(
            density, grid, two_d_measure="log", one_d_measure="log",
            log_base=10.0, cell_axis=axis,
        )
        np.testing.assert_allclose(
            natural * axis.weights("log", log_base=math.e)[None, :],
            density * axis.weights("log", log_base=10.0)[None, :],
            rtol=3e-15,
        )

    def test_alignment_preserves_mass_on_saved_resampled_bins(self):
        source_grid = np.exp([0.0, 0.5, 1.8, 3.0])
        source_axis = build_axis_quadrature(
            source_grid, "cell_centres", "log", math.e
        )
        source_edges = source_axis.edges("log", log_base=math.e)
        target_edges = np.linspace(source_edges[0], source_edges[-1], 10)
        target_grid = np.exp(0.5 * (target_edges[:-1] + target_edges[1:]))
        target_axis = build_axis_quadrature(
            target_grid, "cell_centres", "log", math.e,
            native_edges=target_edges,
        )
        density = np.asarray([[1.0, 4.0, 2.0, 0.5], [0.2, 0.3, 7.0, 1.0]])
        aligned = _aligned_projection(
            density, source_grid, _state(target_axis), _settings(),
            two_d_measure="log", log_base=math.e,
        )
        np.testing.assert_allclose(
            aligned @ target_axis.weights("log", log_base=math.e),
            density @ source_axis.weights("log", log_base=math.e),
            rtol=3e-15,
        )

    def test_cell_capture_uses_fractional_native_measure_edges(self):
        grid = np.exp([0.5, 1.5, 2.5])
        axis = build_axis_quadrature(grid, "cell_centres", "log", math.e)
        state = _state(axis)
        density = np.asarray([[1.0, 2.0, 3.0]])
        capture = _projection_capture(
            density, np.exp([0.25]), np.exp([2.75]), state
        )
        np.testing.assert_allclose(capture, [(0.75 + 2.0 + 2.25) / 6.0])

    def test_nodes_ignore_grid_measure_numerically(self):
        source = np.asarray([1.0, 2.5, 8.0])
        target = np.geomspace(source[0], source[-1], 8)
        state = {"m_grid": target, "MASS_DENSITY_MEASURE": "linear"}
        density = np.asarray([[1.0, 4.0, 2.0]])
        first = _aligned_projection(
            density, source, state, _settings("nodes", "linear"),
            two_d_measure="log", log_base=10.0,
        )
        second = _aligned_projection(
            density, source, state, _settings("nodes", "log"),
            two_d_measure="log", log_base=10.0,
        )
        np.testing.assert_array_equal(first, second)

    def test_cache_signature_changes_with_each_grid_convention(self):
        class Results:
            attrs = {}

            def __contains__(self, item):
                return False

        results = Results()
        baseline = _settings("nodes", "linear")
        first = _signature(baseline, {}, {}, results)
        changed_location = _settings("cell_centres", "linear")
        changed_measure = _settings("nodes", "log")
        self.assertNotEqual(first, _signature(changed_location, {}, {}, results))
        self.assertNotEqual(first, _signature(changed_measure, {}, {}, results))

    def test_comparison_rejects_historical_nodal_products_in_cell_mode(self):
        results = SimpleNamespace(attrs={})
        with self.assertRaisesRegex(RuntimeError, "different input grid placement"):
            _assert_quadrature_compatibility(
                _settings(), results, ({}, {}), ({}, {})
            )

    @unittest.skipIf(h5py is None, "h5py is required for combined workflow")
    def test_combined_cell_workflow_and_comparison_resume(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as text:
            directory = Path(text)
            grid = np.geomspace(1.0, 80.0, 31)
            axis = build_axis_quadrature(grid, "cell_centres", "log", 10.0)
            u1, u2 = np.meshgrid(np.log(grid), np.log(grid), indexing="ij")
            mask = grid[None, :] <= grid[:, None]
            density = np.stack([
                np.where(
                    mask,
                    np.exp(-((u1 - math.log(10.0) - shift) / 0.25) ** 2
                           - ((u2 - math.log(7.0)) / 0.24) ** 2)
                    + 0.6 * np.exp(-((u1 - math.log(33.0)) / 0.3) ** 2
                                   - ((u2 - math.log(22.0)) / 0.29) ** 2)
                    + 0.002,
                    0.0,
                )
                for shift in np.linspace(-0.02, 0.02, 8)
            ])
            widths = np.multiply.outer(axis.weights("log"), axis.weights("log"))
            density /= np.sum(density * widths, axis=(1, 2))[:, None, None]
            np.savez(directory / "posterior.npz", p=density, m1=grid, m2=grid, mask=mask)
            settings_path = directory / "settings.ini"
            settings_path.write_text("""[input]
file = posterior.npz
grid_location = cell_centres
grid_measure = log
domain = ordered
density_measure = log
[analysis]
run = both
log_base = 10
geometry = log
feature_measure = log
scales = 0
persistence_threshold = 0.02
hessian_refinement = false
detect_plateaus = false
detect_shoulders = false
ordered_tails = false
curve_points = 16
[one_dimensional]
measure = linear
scales = 0.05
minimum_persistence = 1
[association]
enabled = off
[compute]
workers = 1
batch_size = 4
resume = true
[output]
directory = fresh_cells
[plot]
write_pdf = false
write_png = false
""")
            settings = load_settings(settings_path)
            output = run(settings)
            manifest_path = output / "one_two_dimensional_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["completed"])
            self.assertEqual(manifest["grid_location"], "cell_centres")
            self.assertEqual(manifest["projection_measure_conversion"], "conservative_cell_width_ratio")
            self.assertTrue((output / "one_two_dimensional_feature_comparison.csv").is_file())
            self.assertTrue((output / "one_two_dimensional_m2_feature_comparison.csv").is_file())
            run(settings)
            self.assertEqual(json.loads(manifest_path.read_text())["signature"], manifest["signature"])


if __name__ == "__main__":
    unittest.main()
