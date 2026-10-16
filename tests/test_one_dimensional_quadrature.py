from __future__ import annotations

import math
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from posterior_landscape.config import load_settings
from posterior_landscape.one_dimensional import (
    _cell_source,
    _component_mass_marginals,
    _core_signature_matches,
    _density_signature,
    _execute,
    _finite_cell_one_dimensional_input,
    cell_widths_from_centers,
    run_one_dimensional,
)
from posterior_landscape.quadrature import (
    QUADRATURE_SCHEMA_VERSION,
    build_axis_quadrature,
    cell_percentile,
)


class OneDimensionalQuadratureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.ini = directory / 'settings.ini'
        self.ini.write_text(
            '[input]\nfile = input.npz\ndomain = ordered\n'
            'grid_location = cell_centres\ngrid_measure = log\n'
            '[analysis]\nlog_base = 10\n'
            '[one_dimensional]\nenabled = true\nmeasure = log\n'
            '[output]\ndirectory = output\n',
            encoding='utf-8',
        )
        self.settings = load_settings(self.ini)

    @staticmethod
    def store(coordinates, samples, mask):
        samples = np.asarray(samples)
        count = samples.shape[0]

        def batches(batch_size):
            for start in range(0, count, batch_size):
                yield start, samples[start:start + batch_size]

        return SimpleNamespace(
            m1=np.asarray(coordinates), m2=np.asarray(coordinates),
            mask=np.asarray(mask), number_draws=count, draws=samples,
            weights=np.full(count, 1.0 / count), iter_batches=batches,
        )

    def test_direct_cell_marginals_preserve_mass_and_first_moments(self) -> None:
        coordinates = np.array([1.0, 1.7, 3.2, 7.0])
        axis = build_axis_quadrature(coordinates, 'cell_centres', 'log', 10.0)
        mask = coordinates[None, :] <= coordinates[:, None]
        source_weights = np.outer(axis.weights('log'), axis.weights('log')) * mask
        samples = np.stack([np.arange(1, 17).reshape(4, 4), np.eye(4) + 1.0])
        samples = np.where(mask, samples, 0.0)
        samples /= np.sum(samples * source_weights, axis=(1, 2))[:, None, None]
        store = self.store(coordinates, samples, mask)
        for measure in ('linear', 'log'):
            settings = replace(
                self.settings,
                one_dimensional=replace(self.settings.one_dimensional, measure=measure),
            )
            first, weights1, second, weights2 = _component_mass_marginals(store, settings)
            np.testing.assert_allclose(first @ weights1, 1.0, atol=3e-15)
            np.testing.assert_allclose(second @ weights2, 1.0, atol=3e-15)
            mass = samples * source_weights
            np.testing.assert_allclose(
                (first * coordinates) @ weights1,
                np.sum(mass * coordinates[:, None], axis=(1, 2)), atol=3e-15,
            )
            np.testing.assert_allclose(
                (second * coordinates) @ weights2,
                np.sum(mass * coordinates[None, :], axis=(1, 2)), atol=3e-15,
            )
            # The source uses log10; the independent logarithmic analysis uses ln.
            expected = np.sum(mass, axis=2) / axis.weights(measure, log_base=math.e)
            np.testing.assert_allclose(first, expected, rtol=2e-15)

    def test_nodal_marginals_keep_existing_notebook_jacobian(self) -> None:
        coordinates = np.geomspace(1.0, 30.0, 7)
        mask = coordinates[None, :] <= coordinates[:, None]
        samples = np.stack([np.arange(1, 50).reshape(7, 7), np.eye(7) + 1.0])
        store = self.store(coordinates, samples, mask)
        settings = replace(
            self.settings, input=replace(self.settings.input, grid_location='nodes')
        )
        first, weights1, second, weights2 = _component_mass_marginals(store, settings)
        widths = cell_widths_from_centers(coordinates)
        legacy = samples / (np.outer(coordinates, coordinates) * np.log(10.0) ** 2)
        legacy = np.where(mask, legacy, 0.0)
        legacy /= np.sum(legacy * (np.outer(widths, widths) * mask), axis=(1, 2))[:, None, None]
        np.testing.assert_array_equal(weights1, widths)
        np.testing.assert_array_equal(weights2, widths)
        np.testing.assert_allclose(first, np.einsum('bij,j->bi', legacy, widths), rtol=3e-15)
        np.testing.assert_allclose(second, np.einsum('bij,i->bj', legacy, widths), rtol=3e-15)
        alternate = replace(settings, input=replace(settings.input, grid_measure='linear'))
        for initial, changed in zip(
            (first, weights1, second, weights2), _component_mass_marginals(store, alternate)
        ):
            np.testing.assert_array_equal(initial, changed)

    def test_cell_input_diagnostics_record_raw_error_and_reject_invalid_active_values(self) -> None:
        coordinates = np.geomspace(1.0, 10.0, 5)
        axis = build_axis_quadrature(coordinates, 'cell_centres', 'log', 10.0)
        mask = coordinates[None, :] <= coordinates[:, None]
        samples = np.ones((2, 5, 5))
        area = np.outer(axis.weights('log'), axis.weights('log')) * mask
        samples /= np.sum(area)
        samples[1] *= 2.0
        diagnostics = {}
        first, weights1, second, weights2 = _component_mass_marginals(
            self.store(coordinates, samples, mask), self.settings, diagnostics=diagnostics,
        )
        np.testing.assert_allclose(diagnostics['input_normalization_factors'], [1.0, 2.0], atol=4e-16)
        np.testing.assert_allclose(diagnostics['input_conversion_normalization_factors'], 1.0, atol=4e-16)
        self.assertAlmostEqual(diagnostics['maximum_input_normalization_error'], 1.0)
        self.assertEqual(diagnostics['active_mask_cells'], 15)
        self.assertEqual(diagnostics['active_diagonal_cells'], 5)
        np.testing.assert_allclose(first @ weights1, 1.0, atol=4e-16)
        np.testing.assert_allclose(second @ weights2, 1.0, atol=4e-16)
        for bad in (-0.1, np.nan, np.inf):
            invalid = samples.copy()
            invalid[1, 2, 2] = bad
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                _component_mass_marginals(self.store(coordinates, invalid, mask), self.settings)

    def test_nonuniform_cell_remapping_retains_outer_bins_and_probability(self) -> None:
        axis = build_axis_quadrature(np.array([1.0, 2.0, 3.5, 8.0]), 'cell_centres', 'log')
        for measure in ('linear', 'log'):
            samples = np.array([[10.0, 0.0, 0.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
            samples /= (samples @ axis.weights(measure))[:, None]
            target, remapped, changed = _finite_cell_one_dimensional_input(axis, samples, measure)
            self.assertTrue(changed)
            np.testing.assert_allclose(target.edges('linear')[[0, -1]], axis.edges('linear')[[0, -1]], rtol=1e-15)
            np.testing.assert_allclose(remapped @ target.weights(measure), 1.0, atol=2e-15)
            self.assertGreater(remapped[0, 0], 0.0)
            self.assertGreater(remapped[0, -1], 0.0)

    def test_uniform_log_cell_input_is_not_resampled(self) -> None:
        axis = build_axis_quadrature(np.exp(np.arange(4)), 'cell_centres', 'log')
        samples = np.ones((2, 4)) / 4.0
        returned_axis, returned, changed = _finite_cell_one_dimensional_input(axis, samples, 'log')
        self.assertFalse(changed)
        self.assertIs(returned_axis, axis)
        self.assertIs(returned, samples)

    def test_discovery_uses_native_edges_and_edge_cdf(self) -> None:
        axis = build_axis_quadrature(np.exp(np.arange(4)), 'cell_centres', 'log')
        samples = np.ones((2, 4)) / 4.0
        namespace = {
            'm_grid': axis.centres, 'pm1_samples': samples,
            'dm': axis.weights('log'), 'MARGINAL_FLOOR': 1e-12,
            'marginal_color': 'black', 'marginal_fill': 'gray',
            'AXIS_LABEL_FS': 12, 'TICK_LABEL_FS': 10,
            'COORDINATE_NAME': 'm1', 'COORDINATE_UNIT': '',
            'COORDINATE_LABEL': '$m_1$', 'CELL_QUADRATURE_AXIS': axis,
            '_cell_percentile': cell_percentile,
        }
        source = _cell_source('cell01_discovery.py').split(
            '# Construct smoothed densities, slopes, and curvatures.'
        )[0]
        _execute(source, namespace, 'finite-cell discovery setup')
        np.testing.assert_allclose(namespace['MASS_MEASURE_WIDTHS'], 1.0, atol=4e-16)
        np.testing.assert_allclose(namespace['MASS_MEASURE_CELL_EDGES'], np.arange(5) - 0.5, atol=4e-16)
        np.testing.assert_allclose(namespace['selected_draw_norm_before'], 1.0, atol=3e-16)
        result = namespace['_posterior_mass_percentile'](samples, 0.25)
        np.testing.assert_allclose(result, np.exp(0.5), rtol=2e-15)
        # This lower limit lies inside the first bin and below its centre.
        conditioned = namespace['_posterior_mass_percentile'](samples, 0.5, lower_mass=np.exp(-0.25))
        np.testing.assert_allclose(conditioned, np.exp(1.625), rtol=2e-15)

    def test_both_cache_interpretation_fields_change_signature(self) -> None:
        coordinates = np.exp(np.arange(4))
        store = self.store(coordinates, np.ones((2, 4, 4)), np.ones((4, 4), dtype=bool))
        original = _density_signature(store, self.settings)
        for field, value in (('grid_location', 'nodes'), ('grid_measure', 'linear')):
            changed = replace(self.settings, input=replace(self.settings.input, **{field: value}))
            self.assertNotEqual(original, _density_signature(store, changed))
        finite = {
            'package_version': '0.8.7', 'core_signature': original,
            'grid_location': 'cell_centres', 'grid_measure': 'log',
            'quadrature_method': 'finite_volume_cell_widths',
            'quadrature_schema_version': QUADRATURE_SCHEMA_VERSION,
        }
        self.assertTrue(_core_signature_matches(finite, original, {}, grid_location='cell_centres', grid_measure='log'))
        self.assertFalse(_core_signature_matches(finite, original, {}, grid_location='nodes'))
        legacy = {'package_version': '0.6.1', 'core_signature': original}
        self.assertFalse(_core_signature_matches(legacy, original, {}, grid_location='cell_centres', grid_measure='log'))

    def test_complete_cell_workflow_records_both_axes_and_reuses_only_compatible_cache(self) -> None:
        directory = Path(self.temporary.name)
        coordinates = np.geomspace(1.0, 30.0, 41)
        axis = build_axis_quadrature(coordinates, 'cell_centres', 'log', 10.0)
        mask = coordinates[None, :] <= coordinates[:, None]
        first, second = np.meshgrid(np.log(coordinates), np.log(coordinates), indexing='ij')
        draws = np.stack([
            np.exp(-((first - 2.0 - offset) ** 2 + (second - 1.7) ** 2) / 0.3)
            + 0.5 * np.exp(-((first - 0.5) ** 2 + (second - 0.25) ** 2) / 0.12)
            for offset in (-0.02, 0.0, 0.02)
        ])
        area = np.outer(axis.weights('log'), axis.weights('log')) * mask
        draws = np.where(mask, draws, 0.0)
        draws /= np.sum(draws * area, axis=(1, 2))[:, None, None]
        np.savez(directory / 'input.npz', p=draws, m1=coordinates, m2=coordinates,
                 mask=mask, weights=np.full(3, 1.0 / 3.0))
        settings = replace(
            self.settings,
            one_dimensional=replace(self.settings.one_dimensional, scales=(0.1,), minimum_persistence=1),
            association=replace(self.settings.association, enabled='off'),
            plot=replace(self.settings.plot, write_pdf=False, write_png=False),
        )
        result = run_one_dimensional(settings)
        self.assertFalse(result.reused_core)
        self.assertFalse(result.reused_secondary_core)
        for coordinate, state in (('m1', result.state), ('m2', result.secondary_state)):
            name = 'one_dimensional_manifest.json' if coordinate == 'm1' else 'one_dimensional_m2_manifest.json'
            manifest = json.loads((result.directory / name).read_text())
            self.assertEqual(manifest['grid_location'], 'cell_centres')
            self.assertEqual(manifest['grid_measure'], 'log')
            self.assertEqual(manifest['quadrature_schema_version'], QUADRATURE_SCHEMA_VERSION)
            self.assertLess(manifest['maximum_conversion_normalization_error'], 1e-14)
            np.testing.assert_allclose(state['physical_cell_edges'], axis.edges('linear'), rtol=1e-15)
            self.assertIn('CELL_QUADRATURE_AXIS', state)
        reused = run_one_dimensional(settings)
        self.assertTrue(reused.reused_core)
        self.assertTrue(reused.reused_secondary_core)
        changed = replace(settings, input=replace(settings.input, grid_measure='linear'))
        with self.assertRaisesRegex(FileExistsError, 'different input or settings'):
            run_one_dimensional(changed)
