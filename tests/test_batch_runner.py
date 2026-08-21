from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _load_batch_module():
    path = Path(__file__).resolve().parents[1] / "run_posterior_landscape_batch.py"
    specification = importlib.util.spec_from_file_location(
        "posterior_landscape_batch_test_module", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


batch = _load_batch_module()


class BatchRunnerTests(unittest.TestCase):
    @staticmethod
    def _write_complete_workflow(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, key in (
            ("manifest.json", "completed"),
            ("one_dimensional_manifest.json", "completed_core"),
            ("one_dimensional_m2_manifest.json", "completed_core"),
            ("one_two_dimensional_manifest.json", "completed"),
        ):
            (directory / name).write_text(
                json.dumps({key: True}), encoding="utf-8"
            )
        for name in (
            "results.h5",
            "features.csv",
            "one_two_dimensional_feature_comparison.csv",
            "one_two_dimensional_m2_feature_comparison.csv",
            "one_two_dimensional_feature_families.csv",
        ):
            (directory / name).touch()

    def test_complete_combined_workflow_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            directory = Path(temporary_text)
            settings = directory / "posterior_landscape.resolved.ini"
            settings.write_text(
                "[analysis]\nrun = both\n"
                "[one_dimensional]\nenabled = true\n",
                encoding="utf-8",
            )
            self._write_complete_workflow(directory)

            complete, _ = batch._workflow_completion_status(directory, settings)
            self.assertTrue(complete)
            (directory / "one_two_dimensional_feature_families.csv").unlink()
            complete, message = batch._workflow_completion_status(
                directory, settings
            )
            self.assertFalse(complete)
            self.assertIn("one_two_dimensional_feature_families.csv", message)

    def test_full_profile_completion_uses_custom_full_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            control = Path(temporary_text) / "features"
            control.mkdir()
            settings = control / "posterior_landscape.resolved.ini"
            settings.write_text(
                "[analysis]\nrun = both\n"
                "[one_dimensional]\nenabled = true\n"
                "[output]\nprofile = full\nfull_subdirectory = retained_full\n",
                encoding="utf-8",
            )
            results = control / "retained_full"
            self._write_complete_workflow(results)

            parser, resolved_results = batch._resolved_output_layout(
                control, settings
            )
            complete, message = batch._workflow_completion_status(
                control, settings
            )
            self.assertEqual(parser.get("output", "profile"), "full")
            self.assertEqual(resolved_results, results)
            self.assertTrue(complete, message)

            (results / "results.h5").unlink()
            complete, message = batch._workflow_completion_status(
                control, settings
            )
            self.assertFalse(complete)
            self.assertIn("results.h5", message)

    def test_collect_aggregation_runs_uses_full_results_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            run_directory = root / "run_R0-4-1-long"
            control = run_directory / "features"
            control.mkdir(parents=True)
            settings = control / "resolved.ini"
            settings.write_text(
                "[analysis]\nrun = both\n"
                "[one_dimensional]\nenabled = true\n"
                "[output]\nprofile = full\n",
                encoding="utf-8",
            )
            results = control / "full"
            self._write_complete_workflow(results)
            density = batch.DensityInfo(
                path=run_directory / "density.h5",
                requested_draws_per_chain=-1,
                actual_draws_per_chain=(2, 2),
                number_draws=4,
                grid=(8, 8),
                has_h0=False,
                cache_signature="full-profile",
            )
            comparison = batch.ComparisonConfig(
                enabled=True,
                baseline_substring="R0-4-1-long",
                output_directory=root / "comparison",
                maximum_log_centroid_distance=0.75,
                minimum_match_margin=0.10,
                write_pdf=False,
                write_png=False,
                figure_dpi=72,
            )
            config = batch.BatchConfig(
                ini_path=root / "batch.ini",
                parent_directory=root,
                run_glob="run_*",
                requested_draws_per_chain=-1,
                density_basename="density",
                template_settings=root / "template.ini",
                output_subdirectory="features",
                posterior_landscape_root=None,
                continue_on_error=True,
                dry_run=False,
                include_substrings=(),
                exclude_substrings=(),
                status_csv=root / "status.csv",
                resolved_settings_filename="resolved.ini",
                comparison=comparison,
            )
            with (
                patch.object(
                    batch, "discover_run_directories", return_value=[run_directory]
                ),
                patch.object(batch, "select_density_file", return_value=density),
            ):
                runs = batch.collect_aggregation_runs(config)

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].control_directory, control.resolve())
            self.assertEqual(runs[0].output_directory, results.resolve())
            self.assertEqual(runs[0].status, "completed_existing")
            self.assertEqual(batch._select_baseline_run(runs, comparison), runs[0])

    def test_consolidated_feature_schema_is_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            directory = Path(temporary_text)
            (directory / "features.csv").write_text(
                "ID,type,support,location_support,geometry_support,"
                "mu1_lower,mu1_median,mu1_upper,mu2_lower,mu2_median,"
                "mu2_upper,feature_probability_median,"
                "relative_prominence_median,match_ambiguity_probability\n"
                "P2D2,peak,0.998,0.997,0.85,9,10,11,7,8,9,0.12,0.9,0.02\n",
                encoding="utf-8",
            )
            with patch.object(batch, "_h5_location_configurations", return_value={}):
                candidates = batch._standard_feature_candidates(directory)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["ID"], "P2D2")
            self.assertEqual(candidates[0]["feature_type"], "peak")
            self.assertAlmostEqual(candidates[0]["P_morph"], 0.998)
            self.assertAlmostEqual(candidates[0]["mu2_median"], 8.0)

    def test_multimodal_configuration_is_recovered_from_results_h5(self) -> None:
        class Dataset:
            def __init__(self, values):
                self.values = np.asarray(values)

            def __array__(self, dtype=None):
                return np.asarray(self.values, dtype=dtype)

        class Group(dict):
            def __init__(self, values, attrs=None):
                super().__init__(values)
                self.attrs = attrs or {}

        class Handle:
            def __init__(self):
                self.mapping = {
                    "posterior/weights": Dataset([0.25, 0.25, 0.25, 0.25]),
                    "features/P2D2": Group(
                        {
                            "location_configuration": Dataset([0, 0, 1, 1]),
                            "location_configuration_probability": Dataset(
                                [0.5, 0.5]
                            ),
                            "location_configuration_conditional_probability": Dataset(
                                [0.5, 0.5]
                            ),
                            "mass_centroid": Dataset(
                                [
                                    [10.0, 5.0],
                                    [12.0, 6.0],
                                    [20.0, 10.0],
                                    [22.0, 11.0],
                                ]
                            ),
                        },
                        attrs={
                            "location_configuration_status": "robust_multimodal",
                            "location_configuration_count": 2,
                            "location_configuration_names": "A,B",
                        },
                    ),
                }

            def __contains__(self, name):
                return name in self.mapping

            def __getitem__(self, name):
                return self.mapping[name]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

        fake_h5py = types.SimpleNamespace(File=lambda *_args, **_kwargs: Handle())
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            directory = Path(temporary_text)
            (directory / "results.h5").touch()
            with patch.dict(sys.modules, {"h5py": fake_h5py}):
                configurations = batch._h5_location_configurations(
                    directory,
                    [{"ID": "P2D2", "feature_type": "peak"}],
                )
        rows = configurations["P2D2"]
        self.assertEqual(
            [row["configuration_ID"] for row in rows],
            ["P2D2-A", "P2D2-B"],
        )
        self.assertLess(rows[0]["mu1_median"], rows[1]["mu1_median"])

    def test_complete_run_is_not_invoked_again(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            run_directory = root / "R0-4-1-long"
            output_directory = run_directory / "features"
            output_directory.mkdir(parents=True)
            density_path = run_directory / "density.h5"
            density_path.touch()
            template = root / "template.ini"
            template.write_text("[input]\nfile=x\n[output]\ndirectory=y\n")
            comparison = batch.ComparisonConfig(
                enabled=False,
                baseline_substring="R0-4-1-long",
                output_directory=root / "comparison",
                maximum_log_centroid_distance=0.75,
                minimum_match_margin=0.1,
                write_pdf=False,
                write_png=False,
                figure_dpi=72,
            )
            config = batch.BatchConfig(
                ini_path=root / "batch.ini",
                parent_directory=root,
                run_glob="*",
                requested_draws_per_chain=-1,
                density_basename="density",
                template_settings=template,
                output_subdirectory="features",
                posterior_landscape_root=None,
                continue_on_error=True,
                dry_run=False,
                include_substrings=(),
                exclude_substrings=(),
                status_csv=root / "status.csv",
                resolved_settings_filename="resolved.ini",
                comparison=comparison,
            )
            density = batch.DensityInfo(
                path=density_path,
                requested_draws_per_chain=-1,
                actual_draws_per_chain=(5, 5),
                number_draws=10,
                grid=(4, 4),
                has_h0=True,
                cache_signature="test",
            )
            settings_path = output_directory / "resolved.ini"
            settings_path.touch()
            with (
                patch.object(
                    batch, "discover_run_directories", return_value=[run_directory]
                ),
                patch.object(
                    batch,
                    "build_execution_environment",
                    return_value=({}, None),
                ),
                patch.object(batch, "select_density_file", return_value=density),
                patch.object(
                    batch,
                    "prepare_run_files",
                    return_value=(output_directory, settings_path),
                ),
                patch.object(
                    batch,
                    "_workflow_completion_status",
                    return_value=(True, "complete"),
                ),
                patch.object(batch.subprocess, "run") as execute,
            ):
                self.assertEqual(batch.run_batch(config), 0)
            execute.assert_not_called()

    def test_complete_full_profile_run_is_not_invoked_again(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            run_directory = root / "R0-4-1-long"
            control = run_directory / "features"
            control.mkdir(parents=True)
            settings = control / "resolved.ini"
            settings.write_text(
                "[analysis]\nrun = both\n"
                "[one_dimensional]\nenabled = true\n"
                "[output]\nprofile = full\nfull_subdirectory = full\n",
                encoding="utf-8",
            )
            self._write_complete_workflow(control / "full")
            density_path = run_directory / "density.h5"
            density_path.touch()
            template = root / "template.ini"
            template.write_text("[input]\nfile=x\n[output]\ndirectory=y\n")
            comparison = batch.ComparisonConfig(
                enabled=False,
                baseline_substring="R0-4-1-long",
                output_directory=root / "comparison",
                maximum_log_centroid_distance=0.75,
                minimum_match_margin=0.1,
                write_pdf=False,
                write_png=False,
                figure_dpi=72,
            )
            config = batch.BatchConfig(
                ini_path=root / "batch.ini",
                parent_directory=root,
                run_glob="*",
                requested_draws_per_chain=-1,
                density_basename="density",
                template_settings=template,
                output_subdirectory="features",
                posterior_landscape_root=None,
                continue_on_error=True,
                dry_run=False,
                include_substrings=(),
                exclude_substrings=(),
                status_csv=root / "status.csv",
                resolved_settings_filename="resolved.ini",
                comparison=comparison,
            )
            density = batch.DensityInfo(
                path=density_path,
                requested_draws_per_chain=-1,
                actual_draws_per_chain=(5, 5),
                number_draws=10,
                grid=(4, 4),
                has_h0=True,
                cache_signature="full-profile",
            )
            with (
                patch.object(
                    batch, "discover_run_directories", return_value=[run_directory]
                ),
                patch.object(
                    batch,
                    "build_execution_environment",
                    return_value=({}, None),
                ),
                patch.object(batch, "select_density_file", return_value=density),
                patch.object(
                    batch,
                    "prepare_run_files",
                    return_value=(control, settings),
                ),
                patch.object(batch.subprocess, "run") as execute,
            ):
                self.assertEqual(batch.run_batch(config), 0)
            execute.assert_not_called()

    def test_replacement_robustness_figures_render(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = batch.ComparisonConfig(
                enabled=True,
                baseline_substring="baseline",
                output_directory=root,
                maximum_log_centroid_distance=0.75,
                minimum_match_margin=0.1,
                write_pdf=False,
                write_png=True,
                figure_dpi=72,
            )
            runs = []
            plot_data = {}
            mass = np.geomspace(1.0, 200.0, 80)
            rng = np.random.default_rng(123)
            for index, name in enumerate(("baseline", "R1", "R2")):
                run_directory = root / name
                output_directory = run_directory / "features"
                density = batch.DensityInfo(
                    path=run_directory / "density.h5",
                    requested_draws_per_chain=-1,
                    actual_draws_per_chain=(25, 25),
                    number_draws=50,
                    grid=(80, 80),
                    has_h0=True,
                    cache_signature=name,
                )
                runs.append(
                    batch.AggregationRun(
                        run_directory=run_directory,
                        output_directory=output_directory,
                        density=density,
                        display_label="Baseline" if index == 0 else name,
                        display_order=index,
                        status="completed_existing",
                        message="",
                    )
                )
                center = np.exp(-0.5 * ((np.log(mass) - np.log(10 + index)) / 0.5) ** 2)
                quantiles = np.vstack((0.7 * center, center, 1.3 * center))
                plot_data[name] = {
                    "m1": mass,
                    "m2": mass,
                    "marginal1": quantiles,
                    "marginal2": quantiles,
                    "h0": rng.normal(65 + index, 12, 50),
                    "weights": np.full(50, 1.0 / 50.0),
                }
            baseline = runs[0]
            batch._plot_h0_posterior_robustness(
                runs, plot_data, baseline, root, comparison
            )
            batch._plot_marginal_robustness(
                runs, plot_data, baseline, root, comparison
            )
            self.assertTrue((root / "h0_posterior_robustness.png").is_file())
            self.assertTrue(
                (root / "marginal_reconstruction_robustness.png").is_file()
            )

    @staticmethod
    def _candidate(
        identifier: str,
        feature_type: str,
        *,
        x1: float = 10.0,
        x2: float = 5.0,
        end_x1: float | None = None,
        end_x2: float | None = None,
        boundary: str = "no",
    ) -> dict[str, object]:
        candidate: dict[str, object] = {
            "ID": identifier,
            "feature_type": feature_type,
            "P_morph": 0.8,
            "P_loc": 0.75,
            "P_reg": 0.7,
            "x1_median": x1,
            "x2_median": x2,
            "x1_reference": x1,
            "x2_reference": x2,
            "mu1_median": x1,
            "mu2_median": x2,
            "boundary": boundary,
        }
        if feature_type in {"ridge", "valley"}:
            candidate.update(
                saddle_x1_median=x1,
                saddle_x2_median=x2,
                topology_end_x1_median=end_x1 if end_x1 is not None else x1 * 0.8,
                topology_end_x2_median=end_x2 if end_x2 is not None else x2 * 0.8,
            )
        return candidate

    @staticmethod
    def _comparison(root: Path) -> object:
        return batch.ComparisonConfig(
            enabled=True,
            baseline_substring="baseline",
            output_directory=root / "comparison",
            maximum_log_centroid_distance=0.35,
            minimum_match_margin=0.10,
            write_pdf=False,
            write_png=False,
            figure_dpi=72,
        )

    @staticmethod
    def _run(root: Path, name: str, order: int) -> object:
        output = root / name / "features"
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.h5").touch()
        density = batch.DensityInfo(
            path=root / name / "density.h5",
            requested_draws_per_chain=-1,
            actual_draws_per_chain=(2, 2),
            number_draws=4,
            grid=(8, 8),
            has_h0=False,
            cache_signature=name,
        )
        return batch.AggregationRun(
            run_directory=root / name,
            output_directory=output,
            density=density,
            display_label="Baseline" if name == "baseline" else name,
            display_order=order,
            status="completed_existing",
            message="",
        )

    def test_boundary_valley_branch_is_a_primary_dynamic_candidate(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = self._comparison(root)
            baseline = self._candidate(
                "BASELINE_VALLEY",
                "valley",
                x1=12.0,
                x2=7.0,
                end_x1=9.0,
                end_x2=8.0,
                boundary="mask",
            )
            current = self._candidate(
                "OTHER_VALLEY",
                "valley",
                x1=12.2,
                x2=7.1,
                end_x1=9.1,
                end_x2=8.05,
                boundary="mask",
            )
            assignments = batch._assign_baseline_features(
                [baseline], [current], comparison
            )
        assignment = assignments["BASELINE_VALLEY"]
        self.assertEqual(assignment["match_status"], "matched")
        self.assertEqual(assignment["candidate"]["ID"], "OTHER_VALLEY")
        self.assertEqual(batch._candidate_match_geometry(current), "saddle_and_endpoint")

    def test_split_relation_is_reported_without_hardcoded_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = self._comparison(root)
            run = self._run(root, "variant", 1)
            baseline = self._candidate(
                "VALLEY_BASE",
                "valley",
                x1=14.0,
                x2=9.0,
                end_x1=10.0,
                end_x2=8.0,
            )
            first = self._candidate(
                "VALLEY_SPLIT_A",
                "valley",
                x1=14.1,
                x2=9.0,
                end_x1=10.1,
                end_x2=8.0,
            )
            second = self._candidate(
                "VALLEY_SPLIT_B",
                "valley",
                x1=13.9,
                x2=9.1,
                end_x1=9.9,
                end_x2=8.1,
            )
            assignments = batch._assign_baseline_features(
                [baseline], [first, second], comparison
            )
            relations = batch._split_merge_rows_for_run(
                run, [baseline], [first, second], comparison
            )
        self.assertEqual(assignments["VALLEY_BASE"]["structural_relation"], "split")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["relation_type"], "split")
        self.assertIn("VALLEY_SPLIT_A", relations[0]["related_IDs"])
        self.assertIn("VALLEY_SPLIT_B", relations[0]["related_IDs"])

    def test_merge_relation_is_reported_for_each_compatible_baseline_feature(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = self._comparison(root)
            run = self._run(root, "variant", 1)
            first = self._candidate("PEAK_A", "peak", x1=10.0, x2=6.0)
            second = self._candidate("PEAK_B", "peak", x1=10.5, x2=6.0)
            merged = self._candidate("PEAK_MERGED", "peak", x1=10.25, x2=6.0)
            assignments = batch._assign_baseline_features(
                [first, second], [merged], comparison
            )
            relations = batch._split_merge_rows_for_run(
                run, [first, second], [merged], comparison
            )
        self.assertEqual(
            {assignments["PEAK_A"]["structural_relation"], assignments["PEAK_B"]["structural_relation"]},
            {"merge"},
        )
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["relation_type"], "merge")
        self.assertIn("PEAK_A", relations[0]["related_IDs"])
        self.assertIn("PEAK_B", relations[0]["related_IDs"])

    def test_reference_free_families_keep_single_run_additions(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = self._comparison(root)
            baseline_run = self._run(root, "baseline", 0)
            variant_run = self._run(root, "variant", 1)
            baseline_peak = self._candidate("PEAK_BASE", "peak", x1=10.0, x2=6.0)
            matched_peak = self._candidate("PEAK_VARIANT", "peak", x1=10.1, x2=6.1)
            additional_valley = self._candidate(
                "VALLEY_ONLY_VARIANT",
                "valley",
                x1=25.0,
                x2=12.0,
                end_x1=20.0,
                end_x2=10.0,
                boundary="mask",
            )
            families = batch._build_feature_families(
                {
                    "baseline": [baseline_peak],
                    "variant": [matched_peak, additional_valley],
                },
                [baseline_run, variant_run],
                baseline_run,
                comparison,
            )
            additions = batch._additional_feature_rows(
                families,
                [baseline_peak],
                [baseline_run, variant_run],
                baseline_run,
            )
        self.assertEqual(len(families), 2)
        self.assertEqual(len(additions), 1)
        self.assertEqual(additions[0]["representative_ID"], "VALLEY_ONLY_VARIANT")
        self.assertEqual(additions[0]["number_runs_detected"], 1)

    def test_reference_free_families_do_not_chain_distant_members(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = self._comparison(root)
            runs = [self._run(root, name, index) for index, name in enumerate(("R1", "R2", "R3"))]
            candidates_by_run = {
                "R1": [self._candidate("A", "shoulder", x1=10.0, x2=6.0)],
                "R2": [self._candidate("B", "shoulder", x1=13.5, x2=6.0)],
                "R3": [self._candidate("C", "shoulder", x1=18.5, x2=6.0)],
            }
            families = batch._build_feature_families(
                candidates_by_run, runs, runs[0], comparison
            )
            family_members = [
                {str(member["ID"]) for member in family["members"]}
                for family in families
            ]

        self.assertEqual(sorted(map(len, family_members)), [1, 2])
        self.assertNotIn({"A", "B", "C"}, family_members)

    def test_dynamic_morphology_robustness_figures_render(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = batch.ComparisonConfig(
                enabled=True,
                baseline_substring="baseline",
                output_directory=root,
                maximum_log_centroid_distance=0.35,
                minimum_match_margin=0.10,
                write_pdf=False,
                write_png=True,
                figure_dpi=72,
            )
            baseline_run = self._run(root, "baseline", 0)
            variant_run = self._run(root, "variant", 1)
            baseline_features = [
                self._candidate("PEAK_BASE", "peak", x1=10.0, x2=6.0),
                self._candidate(
                    "VALLEY_BASE",
                    "valley",
                    x1=14.0,
                    x2=9.0,
                    end_x1=10.0,
                    end_x2=8.0,
                    boundary="mask",
                ),
            ]
            candidates_by_run = {
                "baseline": baseline_features,
                "variant": [
                    self._candidate("PEAK_VARIANT", "peak", x1=10.1, x2=6.1),
                    self._candidate(
                        "VALLEY_VARIANT",
                        "valley",
                        x1=14.1,
                        x2=9.1,
                        end_x1=10.1,
                        end_x2=8.1,
                        boundary="mask",
                    ),
                ],
            }
            feature_rows = []
            for run in (baseline_run, variant_run):
                assignments = (
                    batch._baseline_assignments(baseline_features)
                    if run is baseline_run
                    else batch._assign_baseline_features(
                        baseline_features,
                        candidates_by_run[run.run_directory.name],
                        comparison,
                    )
                )
                feature_rows.extend(
                    batch._baseline_feature_rows_for_run(
                        run, baseline_features, assignments
                    )
                )
            families = batch._build_feature_families(
                candidates_by_run,
                [baseline_run, variant_run],
                baseline_run,
                comparison,
            )
            family_rows = batch._family_summary_rows(
                families, [baseline_run, variant_run], baseline_run
            )
            family_matrix_rows = batch._feature_family_matrix_rows(
                families, [baseline_run, variant_run], baseline_run
            )

            batch._plot_baseline_feature_robustness(
                [baseline_run, variant_run],
                feature_rows,
                baseline_features,
                root,
                comparison,
            )
            batch._plot_feature_family_robustness(
                [baseline_run, variant_run],
                family_rows,
                family_matrix_rows,
                root,
                comparison,
            )

            self.assertTrue((root / "feature_robustness_matrix.png").is_file())
            self.assertTrue(
                (root / "feature_family_robustness_matrix.png").is_file()
            )

    def test_aggregate_dynamic_catalogue_accepts_baseline_valley_without_event(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            comparison = self._comparison(root)
            baseline_run = self._run(root, "baseline", 0)
            variant_run = self._run(root, "variant", 1)
            template = root / "template.ini"
            template.write_text("[input]\nfile=x\n[output]\ndirectory=y\n")
            config = batch.BatchConfig(
                ini_path=root / "batch.ini",
                parent_directory=root,
                run_glob="*",
                requested_draws_per_chain=-1,
                density_basename="density",
                template_settings=template,
                output_subdirectory="features",
                posterior_landscape_root=None,
                continue_on_error=True,
                dry_run=False,
                include_substrings=(),
                exclude_substrings=(),
                status_csv=root / "status.csv",
                resolved_settings_filename="resolved.ini",
                comparison=comparison,
            )
            baseline_valley = self._candidate(
                "VALLEY_BASE",
                "valley",
                x1=15.0,
                x2=8.0,
                end_x1=12.0,
                end_x2=7.9,
                boundary="mask",
            )
            variant_valley = self._candidate(
                "VALLEY_VARIANT",
                "valley",
                x1=15.2,
                x2=8.1,
                end_x1=12.1,
                end_x2=8.0,
                boundary="mask",
            )
            candidates = {
                baseline_run.output_directory: [baseline_valley],
                variant_run.output_directory: [variant_valley],
            }
            global_row = {
                "run": "",
                "display_label": "",
                "display_order": 0,
                "run_status": "completed_existing",
            }
            with (
                patch.object(
                    batch, "collect_aggregation_runs", return_value=[baseline_run, variant_run]
                ),
                patch.object(
                    batch,
                    "_standard_feature_candidates",
                    side_effect=lambda directory: candidates[directory],
                ),
                patch.object(
                    batch,
                    "_global_row_for_run",
                    side_effect=lambda run: ({**global_row, "run": run.run_directory.name}, None),
                ),
                patch.object(batch, "_plot_baseline_feature_robustness"),
                patch.object(batch, "_plot_feature_family_robustness"),
                patch.object(batch, "_plot_h0_posterior_robustness"),
                patch.object(batch, "_plot_marginal_robustness"),
            ):
                self.assertEqual(batch.aggregate_batch_results(config), 0)
            rows = (comparison.output_directory / "robustness_features.csv").read_text(
                encoding="utf-8"
            )
            additions = (comparison.output_directory / "robustness_additional_features.csv").read_text(
                encoding="utf-8"
            )
        self.assertIn("VALLEY_BASE", rows)
        self.assertIn("VALLEY_VARIANT", rows)
        self.assertIn("family_ID", additions)

    def test_aggregate_only_never_calls_run_batch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_text:
            root = Path(temporary_text)
            config = types.SimpleNamespace()
            with (
                patch.object(batch, "load_batch_config", return_value=config),
                patch.object(batch, "aggregate_batch_results", return_value=0) as aggregate,
                patch.object(batch, "run_batch") as run_batch,
            ):
                self.assertEqual(batch.main([str(root / "batch.ini"), "--aggregate-only"]), 0)
        aggregate.assert_called_once_with(config)
        run_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
