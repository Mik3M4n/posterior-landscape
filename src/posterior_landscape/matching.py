"""Coherent correspondence of critical points, connections, and curve events.

Boundary membership describes a feature; it is not its identity. The point
assignments supplied by the catalogue remain authoritative for ridge and
valley endpoints, including transitions onto or away from domain boundaries.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from .io import Grid
    from .topology import BranchFeature, CurveEventFeature, PointFeature

MATCHING_SCHEMA_VERSION = "2.0"
PointKey = tuple[str, str, tuple[int, int]]


def validate_matching_policy(policy: str) -> str:
    if policy not in {"coherent", "legacy"}:
        raise ValueError("matching_policy must be 'coherent' or 'legacy'.")
    return policy


def point_key(point: PointFeature) -> PointKey:
    return point.kind, (point.role or "") if point.kind == "saddle" else "", tuple(point.index)


def branch_anchor_keys(branch: BranchFeature) -> tuple[PointKey, PointKey]:
    role = "join" if branch.kind == "ridge" else "split"
    extremum = "peak" if branch.kind == "ridge" else "pit"
    return ("saddle", role, tuple(branch.saddle_index)), (extremum, "", tuple(branch.extremum_index))


def support_component_labels(grid: Grid) -> np.ndarray:
    """Label the fixed support using exactly the topology's triangulation."""

    from .topology import _OFFSETS

    structure = np.zeros((3, 3), dtype=bool)
    structure[1, 1] = True
    for first, second in _OFFSETS:
        structure[1 + first, 1 + second] = True
    labels, _ = ndimage.label(grid.mask, structure=structure)
    return labels


def _same_component(first: PointKey, second: PointKey, labels: np.ndarray) -> bool:
    original = int(labels[first[2]])
    return original > 0 and original == int(labels[second[2]])


def _position(point: PointFeature, grid: Grid) -> np.ndarray:
    return np.asarray([grid.geometry1[point.index[0]], grid.geometry2[point.index[1]]])


def match_saddles(
    templates: list[PointFeature],
    candidates: list[PointFeature],
    template_branches: list[BranchFeature],
    candidate_branches: list[BranchFeature],
    point_matches: dict[PointKey, PointKey],
    grid: Grid,
    *,
    maximum_distance: float,
    assignment: Callable,
    template_anchors: list[tuple[PointKey, PointKey]] | None = None,
    component_labels: np.ndarray | None = None,
) -> dict[PointKey, PointKey]:
    """Match join/split saddles separately, respecting known endpoint identity.

    Every incident extremum already matched must belong to the corresponding
    candidate saddle. An absent extremum imposes no attachment constraint;
    additional arms at a multiway saddle are allowed. Coordinates break ties
    using the existing assignment.
"""

    result: dict[PointKey, PointKey] = {}
    incident_templates: dict[PointKey, set[PointKey]] = {}
    incident_candidates: dict[PointKey, set[PointKey]] = {}
    for index, branch in enumerate(template_branches):
        saddle, extremum = template_anchors[index] if template_anchors is not None else branch_anchor_keys(branch)
        if extremum in point_matches:
            incident_templates.setdefault(saddle, set()).add(point_matches[extremum])
    for branch in candidate_branches:
        saddle, extremum = branch_anchor_keys(branch)
        incident_candidates.setdefault(saddle, set()).add(extremum)
    for role in ("join", "split", ""):
        first = [p for p in templates if p.kind == "saddle" and (p.role or "") == role]
        second = [p for p in candidates if p.kind == "saddle" and (p.role or "") == role]
        compatibility = np.ones((len(first), len(second)), dtype=bool)
        for row, template in enumerate(first):
            known = incident_templates.get(point_key(template), set())
            if known:
                for column, candidate in enumerate(second):
                    compatibility[row, column] = known.issubset(incident_candidates.get(point_key(candidate), set()))
            if component_labels is not None:
                for column, candidate in enumerate(second):
                    compatibility[row, column] &= _same_component(point_key(template), point_key(candidate), component_labels)
        matches, _ = assignment(
            [_position(p, grid) for p in first],
            [_position(p, grid) for p in second],
            maximum_distance=maximum_distance,
            compatibility=compatibility,
        )
        for row, column in matches:
            result[point_key(first[row])] = point_key(second[column])
    return result


def branch_compatibility(
    templates: list[BranchFeature],
    candidates: list[BranchFeature],
    point_matches: dict[PointKey, PointKey],
    *,
    maximum_distance: float,
    template_anchors: list[tuple[PointKey, PointKey]] | None = None,
    component_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Require the same mapped anchors and each endpoint's distance limit."""

    compatible = np.zeros((len(templates), len(candidates)), dtype=bool)
    for row, template in enumerate(templates):
        first, last = template_anchors[row] if template_anchors is not None else branch_anchor_keys(template)
        if first not in point_matches or last not in point_matches:
            continue
        expected = point_matches[first], point_matches[last]
        if component_labels is not None and not all(_same_component(a, b, component_labels) for a, b in zip((first, last), expected)):
            continue
        for column, candidate in enumerate(candidates):
            if template.kind != candidate.kind or branch_anchor_keys(candidate) != expected:
                continue
            a = np.asarray(template.topology_points_geometry)[[0, -1]]
            b = np.asarray(candidate.topology_points_geometry)[[0, -1]]
            compatible[row, column] = bool(np.all(np.linalg.norm(a - b, axis=1) <= maximum_distance))
    return compatible


def event_compatibility(
    templates: list[CurveEventFeature],
    candidates: list[CurveEventFeature],
    branch_matches: dict[int, int],
) -> np.ndarray:
    """Match an existing pair only when its two original arms are matched."""

    compatible = np.zeros((len(templates), len(candidates)), dtype=bool)
    for row, template in enumerate(templates):
        if any(index not in branch_matches for index in template.arm_indices):
            continue
        expected = {branch_matches[index] for index in template.arm_indices}
        if len(expected) != 2:
            continue
        for column, candidate in enumerate(candidates):
            compatible[row, column] = template.kind == candidate.kind and expected == set(candidate.arm_indices)
    return compatible


def aligned_event(
    template: CurveEventFeature,
    candidate: CurveEventFeature,
    branch_matches: dict[int, int],
) -> CurveEventFeature:
    """Return template-oriented geometry without mutating the analysis."""

    expected = tuple(branch_matches[index] for index in template.arm_indices)
    if candidate.arm_indices == expected:
        return candidate
    if candidate.arm_indices != expected[::-1]:
        raise ValueError("The event does not contain the template's matched arms.")
    aligned = deepcopy(candidate)
    aligned.arm_indices = candidate.arm_indices[::-1]
    aligned.arm_identifiers = candidate.arm_identifiers[::-1]
    aligned.topology_geometry = np.asarray(candidate.topology_geometry)[[0, 2, 1]].copy()
    aligned.curve_geometry = np.asarray(candidate.curve_geometry)[::-1].copy()
    return aligned
