"""Selected spatial quadrature across streaming diagnostics and pipeline caches."""
from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from posterior_landscape.__main__ import main
from posterior_landscape.config import load_settings
from posterior_landscape.ensemble import compute_ensemble_summary, _marginal_percentile
from posterior_landscape.io import open_density_store, validate_store, h5py
from posterior_landscape.pipeline import (
    _completed_topology_compatibility, _fingerprint, _quadrature_output_arrays,
    _topology_sample_signature, run,
)
from posterior_landscape.quadrature import cell_cdf, cell_percentile
from posterior_landscape.validation import validate_configuration


class QuadraturePipelineTests(unittest.TestCase):
    def fixture(self, directory: Path, *, measure: str = "log"):
        centres = np.exp(np.arange(7, dtype=float) * 0.2 + 0.4)
        mask = centres[None, :] <= centres[:, None]
        first, second = np.meshgrid(np.log(centres), np.log(centres), indexing="ij")
        values = np.stack([
            np.exp(-((first - 1.1 - shift) / 0.28) ** 2
                   - ((second - 0.85) / 0.3) ** 2) + 0.03
            for shift in (-0.025, 0.0, 0.025, 0.05)
        ])
        values = np.where(mask, values, 0.0)
        values /= np.sum(values * mask * 0.2**2, axis=(1, 2))[:, None, None]
        np.savez(directory / "posterior.npz", p=values, m1=centres, m2=centres,
                 mask=mask, weights=np.ones(values.shape[0]))
        ini = directory / "settings.ini"
        ini.write_text(f"""[input]
file = posterior.npz
grid_location = cell_centres
grid_measure = log
domain = ordered
density_measure = log
[analysis]
run = 2d
log_base = e
geometry = log
feature_measure = {measure}
scales = 0
hessian_refinement = false
detect_shoulders = false
detect_plateaus = false
ordered_tails = true
curve_points = 16
[compute]
workers = 1
batch_size = 2
[association]
enabled = off
[output]
directory = result
[plot]
write_pdf = false
""", encoding="utf-8")
        return load_settings(ini)

    def test_all_draws_audit_reports_mass_and_preserved_triangle(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            settings = self.fixture(Path(name), measure="linear")
            report = validate_configuration(settings, all_draws=True)
            self.assertEqual(report["audited_draws"], 4)
            self.assertEqual(report["active_mask_cells"], 28)
            self.assertEqual(report["active_diagonal_cells"], 7)
            self.assertEqual(report["audit_scope"], "all_draws")
            for field in ("minimum", "percentile_5", "median", "percentile_95", "maximum"):
                self.assertAlmostEqual(report["raw_normalization"][field], 1.0, places=14)
            for field in ("maximum_input_normalization_error", "maximum_post_normalization_error",
                          "maximum_conversion_normalization_error", "maximum_marginal_normalization_error",
                          "maximum_marginal_conservation_error"):
                self.assertLess(report[field], 2e-14)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["validate", str(settings.source), "--all-draws", "--json"]), 0)
            self.assertEqual(json.loads(output.getvalue())["active_diagonal_cells"], 7)
            self.assertFalse(settings.output.directory.exists())

    def test_streaming_reference_marginals_conversion_and_cell_tails(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            settings = self.fixture(Path(name), measure="linear")
            with open_density_store(settings.input.file, input_settings=settings.input) as store:
                grid = validate_store(store, input_settings=settings.input, log_base=math.e,
                                      feature_measure="linear", geometry="log")
                summary = compute_ensemble_summary(store, grid, credible_mass=0.9,
                                                   batch_size=2, tail_probability=0.8)
                np.testing.assert_allclose(summary.normalization, 1.0, atol=2e-14)
                np.testing.assert_allclose(summary.feature_normalization, 1.0, atol=2e-14)
                self.assertLess(summary.maximum_marginal_normalization_error, 2e-14)
                self.assertAlmostEqual(grid.integrate(summary.reference_density), 1.0, places=14)
                # Use the same independently computed finite-cell marginal to check
                # CDF inversion and common-grid survival, without nodal interpolation.
                ratios = np.multiply.outer(grid.input_weight1 / grid.weight1,
                                           grid.input_weight2 / grid.weight2)
                first_draw = np.asarray(store.draws[0]) * ratios
                marginal, _ = grid.marginals(first_draw)
                expected = cell_percentile(marginal, grid.axis1, "linear", 0.8)
                self.assertAlmostEqual(summary.tail_m1_scale[0], float(expected), places=13)
                self.assertAlmostEqual(float(cell_cdf(marginal, grid.axis1, "linear", expected)), 0.8, places=13)

    def test_nodal_tails_and_outputs_do_not_depend_on_native_grid_measure(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            settings = self.fixture(Path(name))
            summaries = []
            for native in ("linear", "log"):
                with open_density_store(settings.input.file, input_settings=settings.input) as store:
                    nodal = replace(settings.input, grid_location="nodes", grid_measure=native)
                    grid = validate_store(store, input_settings=nodal, log_base=math.e,
                                          feature_measure="log", geometry="log")
                    original = np.asarray(store.draws[0]).copy()
                    summary = compute_ensemble_summary(store, grid, credible_mass=0.9, batch_size=2)
                    summaries.append(summary)
                    # Existing centre-CDF result must be retained exactly.
                    raw = original / summary.normalization[0]
                    marginal, _ = grid.marginals(raw)
                    expected, _ = _marginal_percentile(marginal[None], grid.m1, grid.weight1, 0.999)
                    np.testing.assert_array_equal(summary.tail_m1_scale[:1], expected)
            for name in vars(summaries[0]):
                np.testing.assert_array_equal(getattr(summaries[0], name), getattr(summaries[1], name))

    def test_diagonal_full_cell_joint_events_and_primary_conditioned_composition(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            directory = Path(name)
            settings = self.fixture(directory)
            centres = np.exp(np.array((0.5, 1.5)))
            mask = centres[None, :] <= centres[:, None]
            draws = np.zeros((2, 2, 2), dtype=float)
            draws[:, 0, 0] = 1.0  # Sole occupied full cell spans 0<u1,u2<1.
            np.savez(settings.input.file, p=draws, m1=centres, m2=centres, mask=mask)
            with open_density_store(settings.input.file, input_settings=settings.input) as store:
                grid = validate_store(store, input_settings=settings.input, log_base=math.e,
                                      feature_measure="log", geometry="log")
                summary = compute_ensemble_summary(store, grid, credible_mass=0.9,
                                                   batch_size=1, tail_probability=0.5)
                self.assertEqual(summary.tail_event_convention, "full_cell_overlap")
                # At the cell midpoint: OR=3/4, AND=1/4, XOR=1/2.
                np.testing.assert_allclose(summary.tail_any_quantiles[:, 0], 0.75, atol=2e-15)
                np.testing.assert_allclose(summary.tail_both_quantiles[:, 0], 0.25, atol=2e-15)
                np.testing.assert_allclose(summary.tail_straddle_quantiles[:, 0], 0.5, atol=2e-15)
                np.testing.assert_allclose(summary.tail_m1_scale, centres[0], atol=2e-15)
                np.testing.assert_allclose(summary.tail_both_at_m1_scale, 0.25, atol=2e-15)
                np.testing.assert_allclose(summary.tail_straddle_at_m1_scale, 0.25, atol=2e-15)
                # Composition is conditional on primary exceeding M, not OR.
                np.testing.assert_allclose(summary.tail_both_fraction_at_m1_scale, 0.5, atol=2e-15)
                np.testing.assert_allclose(summary.tail_straddle_fraction_at_m1_scale, 0.5, atol=2e-15)

    def test_resolved_conventions_change_fingerprint_and_topology_signature(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            settings = self.fixture(Path(name))
            with open_density_store(settings.input.file, input_settings=settings.input) as store:
                fingerprints = []
                signatures = []
                for location, measure in (("nodes", "linear"), ("nodes", "log"),
                                          ("cell_centres", "linear"), ("cell_centres", "log")):
                    changed = replace(settings, input=replace(settings.input, grid_location=location,
                                                              grid_measure=measure))
                    fingerprints.append(_fingerprint(changed, store.shape))
                    grid = validate_store(store, input_settings=changed.input, log_base=math.e, geometry="log")
                    signatures.append(_topology_sample_signature(store, grid))
                self.assertEqual(len(set(fingerprints)), 4)
                self.assertEqual(len(set(signatures)), 4)

    @unittest.skipIf(h5py is None, "h5py is required for numerical serialization and reuse")
    def test_hdf5_weights_edges_metadata_and_completed_guard(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            settings = self.fixture(Path(name))
            run(settings)
            directory = settings.output.directory
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["grid_location"], "cell_centres")
            self.assertEqual(manifest["active_diagonal_cells"], 7)
            self.assertEqual(manifest["global_tail_scale"]["tail_event_convention"], "full_cell_overlap")
            self.assertIn("exactly one", manifest["global_tail_scale"]["P_straddle"])
            self.assertLess(manifest["maximum_input_normalization_error"], 2e-14)
            resolved = load_settings(directory / "settings.ini")
            self.assertEqual(resolved.input.file, settings.input.file)
            self.assertEqual(resolved.input.grid_measure, "log")
            with open_density_store(settings.input.file, input_settings=settings.input) as store:
                grid = validate_store(store, input_settings=settings.input, log_base=math.e, geometry="log")
                with h5py.File(directory / "results.h5", "r") as output:
                    self.assertEqual(output.attrs["grid_location"], "cell_centres")
                    self.assertEqual(output["global_tail"].attrs["event_convention"], "full_cell_overlap")
                    self.assertIn("exactly one", output["global_tail"].attrs["definitions"])
                    self.assertIn("conditional on m1", output["global_tail"].attrs["composition_at_m1_definition"])
                    for key, value in _quadrature_output_arrays(grid).items():
                        np.testing.assert_array_equal(output["coordinates"][key], value)
                self.assertTrue(_completed_topology_compatibility(
                    directory, manifest, settings, store, grid)[0])
                changed = replace(settings, input=replace(settings.input, grid_location="nodes"))
                nodal = validate_store(store, input_settings=changed.input, log_base=math.e, geometry="log")
                self.assertFalse(_completed_topology_compatibility(
                    directory, manifest, changed, store, nodal)[0])
            # A completed old convention must fail instead of entering H0-only reuse.
            changed = replace(settings, input=replace(settings.input, grid_location="nodes"))
            with self.assertRaisesRegex(FileExistsError, "quadrature|interpretation"):
                run(changed)
            # Native measure alone is also part of compatibility semantics.
            changed = replace(settings, input=replace(settings.input, grid_measure="linear"))
            with self.assertRaisesRegex(FileExistsError, "quadrature|interpretation"):
                run(changed)
