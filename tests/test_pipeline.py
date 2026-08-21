from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from test_core import synthetic_density, synthetic_grid

from posterior_landscape.config import PlotSettings, load_settings
from posterior_landscape.ensemble import (
    PosteriorCatalogue,
    assign_reference_identifiers,
)
from posterior_landscape.io import h5py, write_hdf5
from posterior_landscape.one_dimensional import (
    _core_signature_matches,
    _external_parameter_source,
    _geometric_one_dimensional_input,
    cell_widths_from_centers,
)
from posterior_landscape.pipeline import (
    _load_checkpoint,
    _replot_completed_geometry_if_needed,
    _save_checkpoint,
    run,
)
from posterior_landscape.topology import analyze_field
from posterior_landscape.validation import validate_configuration


class PipelineTests(unittest.TestCase):
    def test_external_parameter_adapter_preserves_private_identifiers(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            generic_path = temporary / "generic.ini"
            generic_path.write_text(
                """[input]
file = posterior.npz

[association]
parameter_name = lambda
parameter_label = $\\lambda$

[output]
directory = result
""",
                encoding="utf-8",
            )
            source = """H0_FEATURE_ORDER = ()
H0_samples = params[\"H0\"]
required = (\"H0_samples\", \"H0_FEATURE_ORDER\")
message = \"H0 feature analysis\"
"""
            transformed = _external_parameter_source(
                source, load_settings(generic_path)
            )
            self.assertIn("'H0_samples'", transformed)
            self.assertIn("'H0_FEATURE_ORDER'", transformed)
            self.assertIn("params['lambda']", transformed)
            self.assertIn("lambda feature analysis", transformed)
            self.assertNotIn("lambda_samples", transformed)

            legacy_path = temporary / "legacy.ini"
            legacy_path.write_text(
                """[input]
file = posterior.npz

[output]
directory = result
""",
                encoding="utf-8",
            )
            self.assertEqual(
                _external_parameter_source(source, load_settings(legacy_path)),
                source,
            )

    def test_non_geometric_one_dimensional_input_is_resampled(self) -> None:
        masses = np.linspace(0.05, 1.0, 81)
        samples = np.stack(
            [
                np.exp(-0.5 * ((masses - center) / 0.08) ** 2)
                for center in (0.35, 0.55)
            ]
        )
        widths = cell_widths_from_centers(masses)
        target, resampled, target_widths, changed = (
            _geometric_one_dimensional_input(masses, samples, widths)
        )
        self.assertTrue(changed)
        self.assertEqual(resampled.shape, (2, target.size))
        self.assertTrue(
            np.allclose(
                np.diff(np.log(target)),
                np.median(np.diff(np.log(target))),
                rtol=1.0e-5,
                atol=1.0e-12,
            )
        )
        np.testing.assert_allclose(
            np.einsum("bi,i->b", resampled, target_widths), 1.0
        )
        np.testing.assert_allclose(
            target[np.argmax(resampled, axis=1)], (0.35, 0.55), atol=0.01
        )

    def test_geometric_one_dimensional_input_is_unchanged(self) -> None:
        masses = np.geomspace(0.05, 1.0, 81)
        samples = np.ones((2, masses.size))
        widths = cell_widths_from_centers(masses)
        target, returned, target_widths, changed = (
            _geometric_one_dimensional_input(masses, samples, widths)
        )
        self.assertFalse(changed)
        np.testing.assert_array_equal(target, masses)
        np.testing.assert_array_equal(returned, samples)
        np.testing.assert_array_equal(target_widths, widths)

    def test_legacy_gw_settings_resolve_to_v07_scientific_defaults(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = production.h5

[analysis]
run = both
geometry = log
feature_measure = log
log_base = e

[one_dimensional]
upper_tail_percentile = 0.9999

[output]
directory = result

[plot]
write_pdf = true
write_png = true
""",
                encoding="utf-8",
            )
            settings = load_settings(settings_path)
            self.assertEqual(settings.input.domain, "legacy")
            self.assertEqual(settings.input.density_dataset, "p")
            self.assertEqual(settings.input.coordinate1_dataset, "m1")
            self.assertEqual(settings.input.coordinate2_dataset, "m2")
            self.assertEqual(settings.input.density_measure, "log")
            self.assertTrue(settings.analysis.detect_shoulders)
            self.assertTrue(settings.analysis.ordered_tails)
            self.assertTrue(settings.one_dimensional.enabled)
            self.assertEqual(settings.one_dimensional.upper_tail_percentile, 0.9999)
            self.assertEqual(settings.association.dataset, "h0")
            self.assertEqual(settings.association.parameter_name, "H0")
            self.assertEqual(settings.association.knn, 5)
            self.assertEqual(settings.association.permutations, 100)
            self.assertEqual(settings.association.uncertainty_resamples, 100)
            self.assertEqual(settings.association.random_seed, 1729)
            self.assertEqual(settings.output.profile, "essential")
            self.assertTrue(settings.plot.write_pdf)
            self.assertFalse(settings.plot.write_png)

    def test_generic_full_rectangle_linear_measure_and_parameter(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            x1 = np.linspace(0.0, 1.0, 23)
            x2 = np.linspace(-0.2, 0.8, 17)
            xx1, xx2 = np.meshgrid(x1, x2, indexing="ij")
            parameter = np.linspace(-1.0, 1.0, 24)
            w1 = np.gradient(x1)
            w2 = np.gradient(x2)
            draws = []
            for value in parameter:
                density = (
                    np.exp(
                        -0.5
                        * (
                            ((xx1 - 0.42 - 0.05 * value) / 0.12) ** 2
                            + ((xx2 - 0.20) / 0.14) ** 2
                        )
                    )
                    + 0.55
                    * np.exp(
                        -0.5
                        * (
                            ((xx1 - 0.76) / 0.10) ** 2
                            + ((xx2 - 0.55 + 0.03 * value) / 0.11) ** 2
                        )
                    )
                    + 1.0e-6
                )
                density /= np.sum(density * np.multiply.outer(w1, w2))
                draws.append(density)
            np.savez(
                temporary / "generic.npz",
                density=np.stack(draws),
                x1=x1,
                x2=x2,
                lambda_draw=parameter,
                chain=np.repeat((0, 1), 12),
            )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = generic.npz
density_dataset = density
coordinate1_dataset = x1
coordinate2_dataset = x2
domain = full
density_measure = linear
coordinate1_name = x1
coordinate2_name = x2
coordinate1_label = x1
coordinate2_label = x2

[analysis]
run = 2d
geometry = linear
feature_measure = linear
scales = 0.0
persistence_threshold = auto
hessian_refinement = false
detect_plateaus = false
detect_shoulders = false
ordered_tails = false
curve_points = 16

[compute]
workers = 1
batch_size = 12

[association]
dataset = lambda_draw
chain_id_dataset = chain
parameter_name = lambda
parameter_label = lambda
knn = 3
permutations = 2
uncertainty_resamples = 2

[output]
directory = result
profile = essential

[plot]
write_pdf = false
""",
                encoding="utf-8",
            )
            settings = load_settings(settings_path)
            report = validate_configuration(settings)
            self.assertEqual(report["shape"], [23, 17])
            self.assertEqual(report["domain"], "full")
            self.assertFalse(report["ordered_domain"])
            self.assertTrue(report["external_parameter"]["available"])
            output = run(settings)
            self.assertTrue((output / "features.csv").is_file())
            association = output / "external_parameter_associations.csv"
            self.assertTrue(association.is_file())
            header = association.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("cov_parameter_mu1", header)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["input_density_measure"], "linear")
            self.assertFalse(manifest["ordered_tails_enabled"])
            self.assertFalse(manifest["shoulders_enabled"])
            numerical = output / ("results.h5" if h5py is not None else "results.npz")
            self.assertTrue(numerical.is_file())
            if h5py is None:
                with np.load(numerical, allow_pickle=False) as result:
                    np.testing.assert_array_equal(
                        result["external_parameter"], parameter
                    )

    def test_v061_one_dimensional_core_signature_is_resume_compatible(self) -> None:
        manifest = {
            "package_version": "0.6.1",
            "core_signature": "legacy",
        }
        self.assertTrue(
            _core_signature_matches(manifest, "current", "legacy")
        )
        self.assertFalse(
            _core_signature_matches(
                {"package_version": "0.5.0", "core_signature": "legacy"},
                "current",
                "legacy",
            )
        )

    @unittest.skipIf(h5py is None, "h5py is required for H0-only reuse")
    def test_added_h0_reuses_completed_draw_level_features(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            grid = synthetic_grid(21)
            number = 40
            shifts = np.linspace(-0.04, 0.04, number)
            draws = np.stack(
                [synthetic_density(grid, shift=value) for value in shifts]
            )
            input_path = write_hdf5(
                temporary / "posterior.h5",
                draws,
                grid.m1,
                grid.m2,
                mask=grid.mask,
                log_base=10.0,
            )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = posterior.h5

[analysis]
geometry = log
log_base = 10
feature_measure = log
scales = 0.0
persistence_threshold = 0.02
hessian_refinement = false
detect_plateaus = false
curve_points = 16

[compute]
workers = 1
batch_size = 20

[association]
knn = 3
permutations = 2
uncertainty_resamples = 2
random_seed = 8

[output]
directory = result

[plot]
write_pdf = false
write_png = false
""",
                encoding="utf-8",
            )

            output = run(load_settings(settings_path))
            first_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(first_manifest["h0_association"]["available"])
            self.assertIn(
                "No aligned dataset named 'h0'",
                (output / "run.log").read_text(encoding="utf-8"),
            )

            # Emulate a completed same-release archive lacking optional
            # provenance fields.
            first_manifest["package_version"] = "0.5.0"
            first_manifest.pop("topology_sample_signature")
            (output / "manifest.json").write_text(
                json.dumps(first_manifest), encoding="utf-8"
            )
            with h5py.File(output / "results.h5", "r+") as result:
                del result.attrs["topology_sample_signature"]
                result.attrs["package_version"] = "0.5.0"
                for group in result["features"].values():
                    if "reference_measurement_valid" in group.attrs:
                        del group.attrs["reference_measurement_valid"]

            h0 = 75.0 + 240.0 * shifts
            chain_id = np.repeat((0, 1), number // 2)
            with h5py.File(input_path, "r+") as input_file:
                input_file.create_dataset("h0", data=h0)
                input_file.create_dataset("chain_id", data=chain_id)

            with patch(
                "posterior_landscape.pipeline.analyze_field",
                side_effect=AssertionError("topology was recomputed"),
            ):
                self.assertEqual(run(load_settings(settings_path)), output)

            updated_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(updated_manifest["h0_association"]["available"])
            self.assertEqual(
                updated_manifest["last_update"],
                {"kind": "h0_association_only", "topology_recomputed": False},
            )
            self.assertTrue(
                (output / "external_parameter_associations.csv").is_file()
            )
            self.assertFalse(
                (output / "feature_location_configurations.csv").is_file()
            )
            with h5py.File(output / "results.h5", "r") as result:
                self.assertTrue(
                    np.array_equal(np.asarray(result["posterior/h0"]), h0)
                )
                self.assertIn("h0_associations", result)
                self.assertEqual(
                    result.attrs["location_configuration_schema"], "1.0"
                )
            run_log = (output / "run.log").read_text(encoding="utf-8")
            self.assertIn("posterior topology will not be recomputed", run_log)

    def test_aligned_h0_runs_the_optional_association_module(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            grid = synthetic_grid(21)
            number = 40
            shifts = np.linspace(-0.04, 0.04, number)
            draws = np.stack(
                [synthetic_density(grid, shift=value) for value in shifts]
            )
            h0 = 75.0 + 240.0 * shifts
            np.savez(
                temporary / "posterior.npz",
                p=draws,
                m1=grid.m1,
                m2=grid.m2,
                mask=grid.mask,
                h0=h0,
                chain_id=np.repeat((0, 1), number // 2),
                log_base=10.0,
            )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = posterior.npz

[analysis]
geometry = log
log_base = 10
feature_measure = log
scales = 0.0
persistence_threshold = 0.02
hessian_refinement = false
detect_plateaus = false
curve_points = 16

[compute]
workers = 1
batch_size = 20

[association]
knn = 3
permutations = 2
uncertainty_resamples = 2
random_seed = 8

[output]
directory = result

[plot]
write_pdf = false
write_png = false
""",
                encoding="utf-8",
            )

            output = run(load_settings(settings_path))
            association_path = output / "external_parameter_associations.csv"
            self.assertTrue(association_path.is_file())
            header = association_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("mu_vector_mi_bits", header)
            self.assertIn("mu2_given_mu1_cmi_bits", header)
            self.assertIn("cov_parameter_mu1", header)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["h0_association"]["available"])
            self.assertEqual(manifest["h0_association"]["number_chains"], 2)
            numerical = output / (
                "results.h5" if h5py is not None else "results.npz"
            )
            if h5py is not None:
                with h5py.File(numerical, "r") as result:
                    self.assertIn("posterior/h0", result)
                    self.assertIn("posterior/chain_id", result)
                    self.assertIn("events", result)
                    self.assertIn("h0_associations", result)

    def test_linear_feature_measure_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            grid = synthetic_grid(21)
            draws = np.stack(
                [
                    synthetic_density(grid, shift=value)
                    for value in (-0.01, 0.0, 0.01)
                ]
            )
            np.savez(
                temporary / "posterior.npz",
                p=draws,
                m1=grid.m1,
                m2=grid.m2,
                mask=grid.mask,
                log_base=10.0,
            )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = posterior.npz

[analysis]
geometry = log
log_base = 10
feature_measure = linear
scales = 0.0
persistence_threshold = auto
hessian_refinement = false
detect_plateaus = false
curve_points = 16

[compute]
workers = 1
batch_size = 3

[output]
directory = result

[plot]
write_pdf = false
write_png = false
""",
                encoding="utf-8",
            )

            output = run(load_settings(settings_path))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["feature_measure"], "linear")
            self.assertEqual(manifest["density_measure"], "dm1_dm2")
            numerical = output / ("results.h5" if h5py is not None else "results.npz")
            if h5py is not None:
                with h5py.File(numerical, "r") as result:
                    self.assertEqual(result.attrs["feature_measure"], "linear")
                    self.assertEqual(result.attrs["density_measure"], "dm1_dm2")
                    density = np.asarray(result["maps/reference_density"])
                    weight1 = np.asarray(result["coordinates/quadrature_m1"])
                    weight2 = np.asarray(result["coordinates/quadrature_m2"])
                    mask = np.asarray(result["coordinates/mask"], dtype=bool)
                    integral = np.sum(
                        density * np.multiply.outer(weight1, weight2) * mask
                    )
                    self.assertAlmostEqual(float(integral), 1.0)

    def test_completed_run_refreshes_stale_geometry_figure(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            directory = Path(temporary_text)
            (directory / "results.h5").write_bytes(b"placeholder")
            (directory / "feature_geometry.png").write_bytes(b"old figure")
            manifest = {
                "completed": True,
                "outputs": ["feature_geometry.png"],
                "geometry_figure": {"schema_version": "1.0"},
            }
            settings = SimpleNamespace(
                plot=PlotSettings(write_pdf=False, write_png=True)
            )
            with patch(
                "posterior_landscape.pipeline."
                "make_feature_geometry_figure_from_results",
                return_value=[directory / "feature_geometry.png"],
            ) as replot:
                _replot_completed_geometry_if_needed(
                    directory, manifest, settings  # type: ignore[arg-type]
                )
            replot.assert_called_once()
            updated = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["geometry_figure"]["schema_version"], "3.0")

    def test_essential_profile_does_not_recreate_removed_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            directory = Path(temporary_text)
            (directory / "results.h5").write_bytes(b"placeholder")
            settings = SimpleNamespace(
                output=SimpleNamespace(profile="essential"),
                plot=PlotSettings(write_pdf=True, write_png=False),
            )
            with patch(
                "posterior_landscape.pipeline."
                "make_feature_geometry_figure_from_results"
            ) as replot:
                _replot_completed_geometry_if_needed(
                    directory,
                    {"geometry_figure": {"schema_version": "1.0"}},
                    settings,  # type: ignore[arg-type]
                )
            replot.assert_not_called()

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            grid = synthetic_grid(23)
            reference = analyze_field(
                synthetic_density(grid),
                grid,
                persistence_threshold=0.02,
                hessian_refinement=False,
                detect_plateaus=False,
                curve_points=24,
            )
            assign_reference_identifiers(reference, grid)
            catalogue = PosteriorCatalogue.from_reference(reference, 4, grid)
            catalogue.ingest(0, reference, discover=False)
            path = Path(temporary_text) / "checkpoint.pkl.gz"
            _save_checkpoint(
                path,
                fingerprint="test-fingerprint",
                next_draw=1,
                catalogue=catalogue,
            )
            loaded = _load_checkpoint(path, "test-fingerprint")
            self.assertIsNotNone(loaded)
            next_draw, restored = loaded  # type: ignore[misc]
            self.assertEqual(next_draw, 1)
            self.assertTrue(restored.point_trackers[0].present[0])

    def test_one_command_outputs_and_idempotent_rerun(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            grid = synthetic_grid(27)
            draws = np.stack(
                [
                    synthetic_density(grid, shift=value)
                    for value in np.linspace(-0.025, 0.025, 6)
                ]
            )
            input_path = temporary / "posterior.npz"
            np.savez(
                input_path,
                p=draws,
                m1=grid.m1,
                m2=grid.m2,
                mask=grid.mask,
                log_base=10.0,
            )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = posterior.npz

[analysis]
geometry = log
log_base = 10
feature_measure = log
credible_mass = 0.90
scales = 0.0
persistence_threshold = 0.02
hessian_refinement = true
detect_plateaus = false
curve_points = 24

[compute]
workers = 1
batch_size = 3
resume = true

[output]
directory = result
overwrite = false

[plot]
display_probability = 0.99
hpd_contours = 0.50, 0.90
write_pdf = false
write_png = true
""",
                encoding="utf-8",
            )
            settings = load_settings(settings_path)
            self.assertEqual(settings.analysis.support_mass, 0.995)
            self.assertEqual(settings.analysis.curve_smoothing_cells, 2.0)
            self.assertEqual(settings.analysis.feature_measure, "log")
            output = run(settings)
            expected = {"features.csv", "manifest.json", "settings.ini", "run.log"}
            self.assertTrue(expected <= {item.name for item in output.iterdir()})
            numerical = output / ("results.h5" if h5py is not None else "results.npz")
            self.assertTrue(numerical.is_file())
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["completed"])
            self.assertEqual(manifest["package_version"], "0.8.6")
            self.assertEqual(manifest["global_tail_scale"]["percentile"], 0.999)
            self.assertEqual(manifest["feature_measure"], "log")
            self.assertEqual(manifest["density_measure"], "dlogm1_dlogm2")
            self.assertEqual(manifest["number_draws"], 6)
            self.assertEqual(
                manifest["features"],
                {
                    "peaks": 2,
                    "pits": 1,
                    "ridges": 2,
                    "valleys": 1,
                    "ridge_events": 1,
                    "valley_events": 0,
                    "shoulders": 0,
                    "plateaus_and_floors": 0,
                },
            )
            self.assertEqual(manifest["support_mass"], 0.995)
            self.assertEqual(
                manifest["geometry_figure"]["schema_version"], "3.0"
            )
            self.assertEqual(
                manifest["geometry_figure"]["curve_events"],
                "two_reference_arms_paired_by_common_saddle",
            )
            self.assertEqual(
                manifest["location_configurations"]["schema_version"], "1.0"
            )
            catalogue_header = (output / "features.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("status", catalogue_header)
            self.assertIn("Pmed(feature)", catalogue_header)
            self.assertIn("geometry_support", catalogue_header)
            self.assertIn("width_major", catalogue_header)
            self.assertIn("saddle_x1_lower", catalogue_header)
            self.assertIn("prominence_lower", catalogue_header)
            with (output / "features.csv").open(newline="", encoding="utf-8") as stream:
                rows = {row["ID"]: row for row in csv.DictReader(stream)}
            self.assertEqual(rows["P2D1"]["status"], "interior")
            self.assertAlmostEqual(float(rows["P2D1"]["x1"]), 4.1246263829013525)
            self.assertAlmostEqual(float(rows["P2D1"]["x2"]), 1.7012542798525887)
            self.assertAlmostEqual(
                float(rows["P2D1"]["prominence"]), 4.826760286893837
            )
            valley_rows = [row for row in rows.values() if row["type"] == "valley"]
            self.assertEqual(len(valley_rows), 1)
            self.assertEqual(valley_rows[0]["status"], "outer_drainage_low_density")
            self.assertAlmostEqual(float(valley_rows[0]["x1"]), 100.0)
            self.assertAlmostEqual(
                float(rows["R2DE1"]["mu1_median"]), 17.820982367425287
            )
            self.assertAlmostEqual(
                float(rows["R2DE1"]["mu2_median"]), 4.37395247989514
            )

            if h5py is not None:
                with h5py.File(numerical, "r") as result:
                    self.assertEqual(result.attrs["schema_version"], "2.1")
                    peak_group = result["features/P2D1"]
                    self.assertIn("width_major", peak_group)
                    self.assertIn("location_hpd_credible", peak_group)
                    self.assertIn("region_inclusion_conditional", peak_group)
                    self.assertIn("mass_centroid", peak_group)
                    self.assertIn("projection_m1", peak_group)
                    self.assertIn("location_configuration", peak_group)
                    self.assertIn("events", result)
                    self.assertIn("shoulders", result)
                    self.assertEqual(result.attrs["shoulder_alpha_threshold"], 2.0)
                    ridge_group = result["features/R2D1"]
                    self.assertIn("width_profiles", ridge_group)
                    self.assertIn("valid_width_fraction", ridge_group)

            # The same one-line invocation is safe after completion.
            self.assertEqual(run(load_settings(settings_path)), output)

    def test_full_profile_uses_subdirectory_and_pdf_csv_policy(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            grid = synthetic_grid(19)
            draws = np.stack(
                [synthetic_density(grid, shift=value) for value in (-0.01, 0.0, 0.01)]
            )
            np.savez(
                temporary / "posterior.npz",
                p=draws,
                m1=grid.m1,
                m2=grid.m2,
                mask=grid.mask,
                log_base=10.0,
            )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = posterior.npz
domain = ordered

[analysis]
run = 2d
geometry = log
log_base = 10
feature_measure = log
scales = 0.0
persistence_threshold = 0.02
hessian_refinement = false
detect_plateaus = false
detect_shoulders = false
ordered_tails = false
curve_points = 16

[compute]
workers = 1
batch_size = 3

[output]
directory = result
profile = full
full_subdirectory = diagnostics

[plot]
write_pdf = true
write_png = true
""",
                encoding="utf-8",
            )
            output_root = run(load_settings(settings_path))
            self.assertEqual(
                {path.name for path in output_root.iterdir()},
                {"diagnostics", "essential"},
            )
            output = output_root / "diagnostics"
            self.assertTrue((output / "features.csv").is_file())
            self.assertTrue(any(output.glob("*.pdf")))
            self.assertTrue(any(output.glob("*.csv")))
            self.assertFalse(any(output.glob("*.png")))
            self.assertFalse(any(output.glob("*.txt")))
            self.assertFalse(any(output.glob("*.tex")))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["output_profile"], "full")
            essential = output_root / "essential"
            self.assertTrue((essential / "features.csv").is_file())
            essential_manifest = json.loads(
                (essential / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(essential_manifest["output_profile"], "essential")
            self.assertEqual(
                essential_manifest["outputs"],
                sorted(path.name for path in essential.iterdir() if path.is_file()),
            )
            standalone_settings_path = temporary / "settings_essential.ini"
            standalone_settings_path.write_text(
                settings_path.read_text(encoding="utf-8")
                .replace("directory = result", "directory = result_standalone")
                .replace("profile = full", "profile = essential"),
                encoding="utf-8",
            )
            standalone = run(load_settings(standalone_settings_path))
            self.assertEqual(
                {path.name for path in essential.iterdir()},
                {path.name for path in standalone.iterdir()},
            )
            self.assertEqual(run(load_settings(settings_path)), output_root)

    def test_one_dimensional_analysis_is_an_explicit_generic_option(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            grid = synthetic_grid(41)
            draws = np.stack(
                [
                    synthetic_density(grid, shift=value)
                    for value in np.linspace(-0.02, 0.02, 24)
                ]
            )
            parameter = np.linspace(-1.0, 1.0, draws.shape[0])
            np.savez(
                temporary / "posterior.npz",
                p=draws,
                x1=grid.m1,
                x2=grid.m2,
                mask=grid.mask,
                lambda_draw=parameter,
                chain=np.repeat((0, 1), draws.shape[0] // 2),
                log_base=10.0,
            )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = posterior.npz
coordinate1_dataset = x1
coordinate2_dataset = x2
coordinate1_name = x1
coordinate2_name = x2
coordinate1_label = x1
coordinate2_label = x2
domain = ordered
density_measure = log

[analysis]
run = 1d
geometry = log
feature_measure = log
log_base = 10
detect_shoulders = false
ordered_tails = false

[one_dimensional]
enabled = true
measure = log
scales = 0.05
minimum_persistence = 1

[association]
enabled = on
dataset = lambda_draw
chain_id_dataset = chain
parameter_name = lambda
parameter_label = $\\lambda$
knn = 3
permutations = 2
uncertainty_resamples = 2

[compute]
workers = 1
batch_size = 6

[output]
directory = result
profile = essential

[plot]
write_pdf = true
""",
                encoding="utf-8",
            )
            settings = load_settings(settings_path)
            report = validate_configuration(settings)
            self.assertTrue(report["one_dimensional_enabled"])
            output = run(settings)
            self.assertTrue((output / "one_dimensional_manifest.json").is_file())
            self.assertTrue((output / "one_dimensional_m2_manifest.json").is_file())
            summaries = sorted(output.glob("*feature_posterior_summary.csv"))
            self.assertGreaterEqual(len(summaries), 2)
            self.assertTrue(any(output.glob("*x1_features_and_support*.pdf")))
            self.assertTrue(any(output.glob("*x2_features_and_support*.pdf")))
            parameter_summaries = sorted(
                output.glob("*external_parameter_feature_summary_table_revised.csv")
            )
            self.assertEqual(len(parameter_summaries), 2)
            self.assertIn(
                "lambda",
                parameter_summaries[0].read_text(encoding="utf-8"),
            )
            production_summaries = sorted(
                output.glob("*external_parameter_feature_dependence_main.csv")
            )
            self.assertEqual(len(production_summaries), 2)
            production_content = production_summaries[0].read_text(
                encoding="utf-8"
            )
            self.assertIn("feature location", production_content)
            self.assertNotIn("mass scale", production_content)
            self.assertNotIn("mass_scale", production_content)
            self.assertFalse(
                any(
                    output.glob(
                        "*external_parameter_feature_location_dependence_prod.pdf"
                    )
                )
            )
            self.assertFalse(any(output.glob("*.tex")))
            self.assertFalse(any(output.glob("*.txt")))
            self.assertFalse(any(output.glob("*.png")))

    @unittest.skipIf(h5py is None, "h5py is required for combined comparison")
    def test_generic_full_combined_parameter_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            temporary = Path(temporary_text)
            grid = synthetic_grid(31)
            number = 24
            parameter = np.linspace(-1.0, 1.0, number)
            draws = np.stack(
                [
                    synthetic_density(grid, shift=0.02 * value)
                    for value in parameter
                ]
            )
            input_path = write_hdf5(
                temporary / "posterior.h5",
                draws,
                grid.m1,
                grid.m2,
                mask=grid.mask,
                log_base=10.0,
                coordinate1_dataset="x1",
                coordinate2_dataset="x2",
            )
            with h5py.File(input_path, "r+") as archive:
                archive.create_dataset("lambda", data=parameter)
                archive.create_dataset(
                    "chain_id", data=np.repeat((0, 1), number // 2)
                )
            settings_path = temporary / "settings.ini"
            settings_path.write_text(
                """[input]
file = posterior.h5
coordinate1_dataset = x1
coordinate2_dataset = x2
domain = ordered
density_measure = log
coordinate1_name = x1
coordinate2_name = x2
coordinate1_label = $x_1$
coordinate2_label = $x_2$

[analysis]
run = both
geometry = log
feature_measure = log
log_base = 10
scales = 0.0
persistence_threshold = 0.02
hessian_refinement = false
detect_plateaus = false
detect_shoulders = false
ordered_tails = false
curve_points = 16

[one_dimensional]
enabled = true
measure = log
scales = 0.05
minimum_persistence = 1

[association]
enabled = on
dataset = lambda
chain_id_dataset = chain_id
parameter_name = lambda
parameter_label = $\\lambda$
knn = 3
permutations = 2
uncertainty_resamples = 2

[compute]
workers = 1
batch_size = 12
resume = true

[output]
directory = result
profile = full
full_subdirectory = diagnostics

[plot]
write_pdf = true
""",
                encoding="utf-8",
            )
            output_root = run(load_settings(settings_path))
            output = output_root / "diagnostics"
            self.assertTrue((output / "results.h5").is_file())
            self.assertTrue(
                (output / "one_two_dimensional_feature_comparison.csv").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "one_two_dimensional_full_external_parameter_comparison.csv"
                ).is_file()
            )
            self.assertTrue(
                (
                    output
                    / "one_two_dimensional_full_external_parameter_draws.npz"
                ).is_file()
            )
            self.assertTrue(
                (output / "one_two_dimensional_feature_family_draws.npz").is_file()
            )
            self.assertFalse(
                (
                    output_root
                    / "essential"
                    / "one_two_dimensional_full_external_parameter_draws.npz"
                ).exists()
            )
            self.assertTrue(any(output.glob("*.pdf")))
            self.assertEqual(
                len(
                    list(
                        output.glob(
                            "*external_parameter_feature_location_dependence_prod.pdf"
                        )
                    )
                ),
                2,
            )
            self.assertFalse(any(output.glob("*.tex")))
            self.assertFalse(any(output.glob("*.txt")))
            self.assertFalse(any(output.glob("*.png")))

            with patch(
                "posterior_landscape.comparison._full_h0_information",
                side_effect=AssertionError("combined association was recomputed"),
            ):
                self.assertEqual(run(load_settings(settings_path)), output_root)


if __name__ == "__main__":
    unittest.main()
