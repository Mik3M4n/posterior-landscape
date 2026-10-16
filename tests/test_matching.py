from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from posterior_landscape.ensemble import (
    PosteriorCatalogue, add_scale_persistence, assign_reference_identifiers,
    retain_scale_persistent_reference,
)
from posterior_landscape.io import Grid, quadrature_weights
from posterior_landscape.matching import aligned_event, branch_compatibility, branch_anchor_keys, support_component_labels
from posterior_landscape.topology import BranchFeature, CurveEventFeature, PointFeature


def grid() -> Grid:
    x = np.arange(1., 22.)
    return Grid(x, x, np.log(x), np.log(x), x, x,
                x[:, None] >= x[None, :], quadrature_weights(x), quadrature_weights(x),
                np.e, 'linear', feature_measure='linear', input_density_measure='linear', domain='ordered')


def point(kind, index, boundary='none', role=None):
    return PointFeature(kind, index, 1., 1., boundary != 'none', role=role, boundary_type=boundary)


def branch(saddle, extremum, *, kind='ridge', saddle_boundary='none', extremum_boundary='none'):
    xy = np.asarray([saddle, extremum], dtype=float) + 1.
    return BranchFeature(kind, saddle, extremum, xy.copy(), xy.copy(), 1., 1.,
                         float(np.linalg.norm(xy[1] - xy[0])), False, xy.copy(),
                         saddle_boundary_type=saddle_boundary, extremum_boundary_type=extremum_boundary)


def event(branches, arms=(0, 1)):
    first, second = [branches[i] for i in arms]
    topology = np.vstack([first.topology_points_geometry[0], first.topology_points_geometry[-1], second.topology_points_geometry[-1]])
    curve = np.vstack([first.points_geometry[::-1], second.points_geometry[1:]])
    return CurveEventFeature(first.kind, first.saddle_index, arms, topology, curve, curve_available=True)


def analysis(points, branches, events=()):
    return SimpleNamespace(points=points, branches=branches, events=list(events), shoulders=[], plateaus=[])


def pixelpop_like(main=(10, 9), boundary='none'):
    low, high = (2, 2), (18, 17)
    s1, s2 = (6, 5), (14, 12)
    points = [point('peak', low, 'mask'), point('peak', main, boundary), point('peak', high),
              point('saddle', s1, role='join'), point('saddle', s2, role='join')]
    arms = [branch(s1, low, extremum_boundary='mask'), branch(s1, main, extremum_boundary=boundary),
            branch(s2, main, extremum_boundary=boundary), branch(s2, high)]
    return analysis(points, arms, [event(arms, (0, 1)), event(arms, (2, 3))])


class MatchingTests(unittest.TestCase):
    def test_ordered_diagonal_transition_restores_original_four_arms(self):
        g = grid()
        coherent = pixelpop_like()
        legacy = deepcopy(coherent)
        scales = [pixelpop_like((9, 9), 'mask'), pixelpop_like((9, 9), 'mask')]
        diagnostics = add_scale_persistence(coherent, scales, g)
        add_scale_persistence(legacy, scales, g, matching_policy='legacy')
        self.assertEqual([b.scale_count for b in coherent.branches], [3, 3, 3, 3])
        self.assertEqual([b.scale_count for b in legacy.branches], [3, 1, 1, 3])
        self.assertEqual([p.scale_count for p in coherent.points if p.kind == 'peak'], [3, 3, 3])
        self.assertEqual([p.scale_count for p in legacy.points if p.kind == 'peak'], [3, 3, 3])
        retain_scale_persistent_reference(coherent)
        retain_scale_persistent_reference(legacy)
        self.assertEqual((len(coherent.branches), len(coherent.events)), (4, 2))
        self.assertEqual((len(legacy.branches), len(legacy.events)), (2, 0))
        self.assertEqual(diagnostics['accepted_extremum_boundary_transitions'], 4)

    def test_outer_or_mask_boundary_labels_do_not_veto_same_connection(self):
        g = grid()
        for label in ['mask', 'outer', 'outer+mask']:
            with self.subTest(label=label):
                template = analysis([point('peak', (10, 9)), point('saddle', (6, 5), role='join')], [branch((6, 5), (10, 9))])
                candidate = analysis([point('peak', (10, 9), label), point('saddle', (6, 5), label, 'join')], [branch((6, 5), (10, 9), saddle_boundary=label, extremum_boundary=label)])
                c = PosteriorCatalogue.from_reference(template, 1, g)
                c.ingest(0, candidate, discover=False)
                self.assertTrue(c.point_trackers[0].present[0])
                self.assertTrue(c.branch_trackers[0].present[0])
                self.assertEqual(c.matching_diagnostics['accepted_boundary_transitions'], 1)

    def test_saddle_transition_also_remains_matched(self):
        g = grid()
        reference = analysis([point('peak', (10, 9)), point('saddle', (6, 5), role='join')], [branch((6, 5), (10, 9))])
        candidate = analysis([point('peak', (10, 9)), point('saddle', (5, 5), 'mask', 'join')], [branch((5, 5), (10, 9), saddle_boundary='mask')])
        add_scale_persistence(reference, [candidate], g)
        self.assertEqual(reference.branches[0].scale_count, 2)

    def test_nearby_curve_cannot_substitute_different_peak_connection(self):
        g = grid()
        a, b, s = (10, 9), (11, 9), (6, 5)
        points = [point('peak', a), point('peak', b), point('saddle', s, role='join')]
        reference = analysis(deepcopy(points), [branch(s, a)])
        candidate = analysis(deepcopy(points), [branch(s, b)])
        coherent = PosteriorCatalogue.from_reference(reference, 1, g)
        legacy = PosteriorCatalogue.from_reference(reference, 1, g, matching_policy='legacy')
        coherent.ingest(0, candidate, discover=False)
        legacy.ingest(0, candidate, discover=False)
        self.assertFalse(coherent.branch_trackers[0].present[0])
        self.assertTrue(legacy.branch_trackers[0].present[0])
        np.testing.assert_array_equal([p.present[0] for p in coherent.point_trackers], [p.present[0] for p in legacy.point_trackers])

    def test_nearby_saddles_use_known_incident_peak_identity(self):
        g = grid()
        p, q, s, t = (10, 9), (12, 9), (6, 5), (7, 5)
        reference = analysis([point('peak', p), point('peak', q), point('saddle', s, role='join'), point('saddle', t, role='join')], [branch(s, p), branch(t, q)])
        candidate = analysis(deepcopy(reference.points), [branch(t, p), branch(s, q)])
        c = PosteriorCatalogue.from_reference(reference, 1, g)
        c.ingest(0, candidate, discover=False)
        self.assertEqual([tracker.present[0] for tracker in c.branch_trackers], [True, True])
        np.testing.assert_allclose(c.branch_trackers[0].topology_endpoints[0, :2], np.asarray(t) + 1.)

    def test_join_and_split_saddles_at_one_index_keep_distinct_ids(self):
        g = grid()
        s, p, d = (6, 5), (10, 9), (3, 2)
        reference = analysis([point('peak', p), point('pit', d), point('saddle', s, role='join'), point('saddle', s, role='split')], [branch(s, p), branch(s, d, kind='valley')])
        assign_reference_identifiers(reference, g)
        self.assertNotEqual(reference.branches[0].start_identifier, reference.branches[1].start_identifier)
        candidate = deepcopy(reference)
        candidate.points.reverse()
        c = PosteriorCatalogue.from_reference(reference, 1, g)
        c.ingest(0, candidate, discover=False)
        self.assertEqual([tracker.present[0] for tracker in c.branch_trackers], [True, True])

    def test_nearby_saddles_sharing_peak_preserve_both_known_attachments(self):
        g = grid()
        low, shared, high = (2, 2), (10, 9), (18, 17)
        left, right = (6, 5), (7, 5)
        points = [point('peak', p) for p in (low, shared, high)]
        points += [point('saddle', s, role='join') for s in (left, right)]
        reference = analysis(points, [branch(left, low), branch(left, shared), branch(right, shared), branch(right, high)])
        candidate = analysis(deepcopy(points), [branch(right, low), branch(right, shared), branch(left, shared), branch(left, high)])
        catalogue = PosteriorCatalogue.from_reference(reference, 1, g)
        catalogue.ingest(0, candidate, discover=False)
        self.assertTrue(all(tracker.present[0] for tracker in catalogue.branch_trackers))
        for index in (0, 1):
            np.testing.assert_array_equal(catalogue.branch_trackers[index].topology_endpoints[0, :2], np.asarray(right) + 1.)
        for index in (2, 3):
            np.testing.assert_array_equal(catalogue.branch_trackers[index].topology_endpoints[0, :2], np.asarray(left) + 1.)

    def test_reversed_event_arms_are_aligned_without_mutation(self):
        g = grid()
        reference = pixelpop_like((9, 9), 'mask')
        candidate = deepcopy(reference)
        candidate.branches = [candidate.branches[1], candidate.branches[0], *candidate.branches[2:]]
        candidate.events[0] = event(candidate.branches, (0, 1))
        original_topology = candidate.events[0].topology_geometry.copy()
        original_curve = candidate.events[0].curve_geometry.copy()
        c = PosteriorCatalogue.from_reference(reference, 1, g)
        c.ingest(0, candidate, discover=False)
        self.assertTrue(c.event_trackers[0].present[0])
        np.testing.assert_allclose(c.event_trackers[0].topology_geometry[0], reference.events[0].topology_geometry.reshape(-1))
        aligned = aligned_event(reference.events[0], candidate.events[0], {0: 1, 1: 0})
        np.testing.assert_allclose(aligned.curve_geometry, reference.events[0].curve_geometry)
        np.testing.assert_array_equal(candidate.events[0].topology_geometry, original_topology)
        np.testing.assert_array_equal(candidate.events[0].curve_geometry, original_curve)

    def test_legacy_event_can_recover_independently_of_missing_arm(self):
        g = grid()
        reference = pixelpop_like()
        candidate = pixelpop_like((9, 9), 'mask')
        coherent = PosteriorCatalogue.from_reference(reference, 1, g)
        legacy = PosteriorCatalogue.from_reference(reference, 1, g, matching_policy='legacy')
        coherent.ingest(0, candidate, discover=False)
        legacy.ingest(0, candidate, discover=False)
        self.assertTrue(all(t.present[0] for t in coherent.branch_trackers))
        self.assertTrue(all(t.present[0] for t in coherent.event_trackers))
        self.assertFalse(legacy.branch_trackers[1].present[0])
        self.assertTrue(legacy.event_trackers[0].present[0])

    def test_event_requires_original_pair_and_both_arms(self):
        g = grid()
        reference = pixelpop_like((9, 9), 'mask')
        candidate = deepcopy(reference)
        candidate.points = [p for p in candidate.points if not (p.kind == 'peak' and p.index == (9, 9))]
        candidate.branches = [candidate.branches[0], candidate.branches[3]]
        candidate.events = []
        c = PosteriorCatalogue.from_reference(reference, 1, g)
        c.ingest(0, candidate, discover=False)
        self.assertFalse(c.event_trackers[0].present[0])
        self.assertTrue(c.branch_trackers[0].present[0])
        self.assertFalse(c.branch_trackers[1].present[0])

    def test_event_is_not_created_from_two_arms_without_original_candidate_pair(self):
        g = grid()
        reference = pixelpop_like((9, 9), 'mask')
        candidate = deepcopy(reference)
        candidate.events = []
        catalogue = PosteriorCatalogue.from_reference(reference, 1, g)
        catalogue.ingest(0, candidate, discover=False)
        self.assertTrue(all(tracker.present[0] for tracker in catalogue.branch_trackers))
        self.assertFalse(any(tracker.present[0] for tracker in catalogue.event_trackers))

    def test_fullpop_like_four_peaks_six_arms_three_events_retained(self):
        g = grid()
        peaks = [(1, 1), (9, 2), (9, 9), (18, 18)]
        saddles = [(5, 5), (9, 5), (14, 14)]
        points = [point('peak', p, 'mask' if p[0] == p[1] else 'none') for p in peaks]
        points += [point('saddle', s, 'mask' if s[0] == s[1] else 'none', 'join') for s in saddles]
        arms = [branch(saddles[0], peaks[0], saddle_boundary='mask', extremum_boundary='mask'), branch(saddles[0], peaks[2], saddle_boundary='mask', extremum_boundary='mask'),
                branch(saddles[1], peaks[1]), branch(saddles[1], peaks[2], extremum_boundary='mask'),
                branch(saddles[2], peaks[2], saddle_boundary='mask', extremum_boundary='mask'), branch(saddles[2], peaks[3], saddle_boundary='mask', extremum_boundary='mask')]
        reference = analysis(points, arms, [event(arms, pair) for pair in [(0, 1), (2, 3), (4, 5)]])
        add_scale_persistence(reference, [deepcopy(reference), deepcopy(reference)], g)
        retain_scale_persistent_reference(reference)
        self.assertEqual((sum(p.kind == 'peak' for p in reference.points), len(reference.branches), len(reference.events)), (4, 6, 3))

    def test_each_endpoint_must_satisfy_its_own_distance_limit(self):
        template = branch((6, 5), (10, 9))
        candidate = branch((6, 5), (13, 9))
        mapping = dict(zip(branch_anchor_keys(template), branch_anchor_keys(candidate)))
        compatible = branch_compatibility([template], [candidate], mapping, maximum_distance=2.5)
        self.assertFalse(compatible[0, 0])

    def test_discovered_arm_reuses_existing_displaced_point_identity(self):
        g = grid()
        saddle, old_peak, new_peak = (6, 5), (10, 9), (11, 9)
        reference = analysis([point('peak', old_peak), point('saddle', saddle, role='join')], [])
        candidate = analysis([point('peak', new_peak), point('saddle', saddle, role='join')], [branch(saddle, new_peak)])
        catalogue = PosteriorCatalogue.from_reference(reference, 2, g)
        catalogue.ingest(0, candidate, discover=True)
        catalogue.ingest(1, candidate, discover=True)
        self.assertEqual(len(catalogue.point_trackers), 1)
        self.assertEqual(len(catalogue.branch_trackers), 1)
        np.testing.assert_array_equal(catalogue.branch_trackers[0].present, [True, True])

    def test_invalid_matching_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorCatalogue.from_reference(pixelpop_like(), 1, grid(), matching_policy='unknown')

    def test_close_disconnected_support_components_cannot_exchange_connections(self):
        original = grid()
        mask = np.zeros(original.shape, dtype=bool)
        mask[5:10, 2:4] = True
        mask[5:10, 5:7] = True
        g = replace(original, mask=mask)
        first, second = (6, 3), (6, 5)
        a, b = (8, 3), (8, 5)
        reference = analysis([point('peak', a, 'mask'), point('saddle', first, 'mask', 'join')], [branch(first, a, saddle_boundary='mask', extremum_boundary='mask')])
        candidate = analysis([point('peak', b, 'mask'), point('saddle', second, 'mask', 'join')], [branch(second, b, saddle_boundary='mask', extremum_boundary='mask')])
        coherent = PosteriorCatalogue.from_reference(reference, 1, g)
        legacy = PosteriorCatalogue.from_reference(reference, 1, g, matching_policy='legacy')
        coherent.ingest(0, candidate, discover=False)
        legacy.ingest(0, candidate, discover=False)
        self.assertTrue(coherent.point_trackers[0].present[0])
        self.assertTrue(legacy.point_trackers[0].present[0])
        self.assertFalse(coherent.branch_trackers[0].present[0])
        self.assertTrue(legacy.branch_trackers[0].present[0])

    def test_support_components_use_exact_six_neighbor_topology(self):
        original = grid()
        mask = np.zeros(original.shape, dtype=bool)
        mask[5, 5] = mask[6, 4] = True
        labels = support_component_labels(replace(original, mask=mask))
        self.assertNotEqual(labels[5, 5], labels[6, 4])
        mask[:] = False
        mask[5, 5] = mask[6, 6] = True
        labels = support_component_labels(replace(original, mask=mask))
        self.assertEqual(labels[5, 5], labels[6, 6])


if __name__ == '__main__':
    unittest.main()
