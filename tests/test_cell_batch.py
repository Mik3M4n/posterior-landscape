from __future__ import annotations

import configparser
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_batch_module():
    path = Path(__file__).resolve().parents[1] / "run_posterior_landscape_batch.py"
    specification = importlib.util.spec_from_file_location("cell_batch_test_module", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


batch = _load_batch_module()


def _complete(directory, metadata):
    for name, key in (
        ("manifest.json", "completed"),
        ("one_dimensional_manifest.json", "completed_core"),
        ("one_dimensional_m2_manifest.json", "completed_core"),
        ("one_two_dimensional_manifest.json", "completed"),
    ):
        (directory / name).write_text(json.dumps({key: True, **metadata}))
    for name in (
        "results.h5", "features.csv",
        "one_two_dimensional_feature_comparison.csv",
        "one_two_dimensional_m2_feature_comparison.csv",
        "one_two_dimensional_feature_families.csv",
    ):
        (directory / name).touch()


class CellBatchTests(unittest.TestCase):
    def test_cell_workflow_never_skips_legacy_nodal_products(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as text:
            directory = Path(text)
            settings = directory / "settings.ini"
            settings.write_text(
                "[input]\ngrid_location=cell_centres\ngrid_measure=log\n"
                "[analysis]\nrun=both\n[one_dimensional]\nenabled=true\n"
            )
            _complete(directory, {})
            complete, message = batch._workflow_completion_status(directory, settings)
            self.assertFalse(complete)
            self.assertIn("incompatible quadrature", message)

    def test_each_cell_workflow_manifest_must_match(self):
        metadata = {
            "grid_location": "cell_centres", "grid_measure": "log",
            "quadrature_schema_version": "1.0",
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as text:
            directory = Path(text)
            settings = directory / "settings.ini"
            settings.write_text(
                "[input]\ngrid_location=cell_centres\ngrid_measure=log\n"
                "[analysis]\nrun=both\n[one_dimensional]\nenabled=true\n"
            )
            _complete(directory, metadata)
            complete, message = batch._workflow_completion_status(directory, settings)
            self.assertTrue(complete, message)
            for name in (
                "manifest.json", "one_dimensional_manifest.json",
                "one_dimensional_m2_manifest.json", "one_two_dimensional_manifest.json",
            ):
                with self.subTest(name=name):
                    _complete(directory, metadata)
                    saved = json.loads((directory / name).read_text())
                    saved["grid_measure"] = "linear"
                    (directory / name).write_text(json.dumps(saved))
                    complete, message = batch._workflow_completion_status(
                        directory, settings
                    )
                    self.assertFalse(complete)
                    self.assertIn(name, message)

    def test_grid_measure_change_invalidates_batch_completion(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as text:
            directory = Path(text)
            settings = directory / "settings.ini"
            settings.write_text(
                "[input]\ngrid_location=nodes\ngrid_measure=log\n"
                "[analysis]\nrun=2d\n"
            )
            _complete(directory, {"grid_location": "nodes", "grid_measure": "linear"})
            complete, message = batch._workflow_completion_status(directory, settings)
            self.assertFalse(complete)
            self.assertIn("incompatible quadrature", message)

    def test_template_preserves_cell_options_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as text:
            directory = Path(text)
            run = directory / "run_1"
            run.mkdir()
            density_path = run / "density.h5"
            density_path.write_bytes(b"input-fixture")
            template = directory / "template.ini"
            template.write_text(
                "[input]\nfile=x\ngrid_location=cell_centres\n"
                "grid_measure=log\nweights_dataset=weights\n"
                "[output]\ndirectory=y\n"
            )
            comparison = batch.ComparisonConfig(
                enabled=False, baseline_substring="baseline",
                output_directory=directory / "comparison",
                maximum_log_centroid_distance=0.75,
                minimum_match_margin=0.1, write_pdf=False, write_png=False,
                figure_dpi=72,
            )
            config = batch.BatchConfig(
                ini_path=directory / "batch.ini", parent_directory=directory,
                run_glob="run_*", requested_draws_per_chain=-1,
                density_basename="density", template_settings=template,
                output_subdirectory="fresh_cells", posterior_landscape_root=None,
                continue_on_error=True, dry_run=False, include_substrings=(),
                exclude_substrings=(), status_csv=directory / "status.csv",
                resolved_settings_filename="resolved.ini", comparison=comparison,
            )
            density = batch.DensityInfo(
                path=density_path, requested_draws_per_chain=-1,
                actual_draws_per_chain=(2, 2), number_draws=4, grid=(4, 4),
                has_h0=False, cache_signature="test-density",
            )
            output, resolved_path = batch.prepare_run_files(run, density, config)
            parser = configparser.ConfigParser()
            parser.read(resolved_path)
            self.assertEqual(parser["input"]["grid_location"], "cell_centres")
            self.assertEqual(parser["input"]["grid_measure"], "log")
            self.assertEqual(parser["input"]["weights_dataset"], "weights")
            provenance = json.loads((output / "posterior_landscape.batch.json").read_text())
            self.assertEqual(provenance["grid_location"], "cell_centres")
            self.assertEqual(provenance["grid_measure"], "log")
            template.write_text(template.read_text().replace("grid_measure=log", "grid_measure=linear"))
            with self.assertRaisesRegex(batch.BatchError, "different input/settings"):
                batch.prepare_run_files(run, density, config)


if __name__ == "__main__":
    unittest.main()
