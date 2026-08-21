"""Posterior summaries, feature matching, and curve uncertainty."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .io import DensityStore, Grid, input_to_feature_factor, quadrature_weights
from .topology import (
    BranchFeature,
    CurveEventFeature,
    FieldAnalysis,
    PlateauFeature,
    PointFeature,
    ShoulderFeature,
)


def weighted_quantile(
    values: np.ndarray,
    probabilities: float | Iterable[float],
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted quantiles along axis zero, with bounded temporary memory."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    requested = np.atleast_1d(np.asarray(probabilities, dtype=float))
    if values.shape[0] != weights.size:
        raise ValueError("weights must correspond to axis zero of values.")
    if np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("Quantile probabilities must lie between zero and one.")
    valid_weight = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if valid_weight.sum() <= 0.0:
        raise ValueError("No positive posterior weight is available.")

    # NumPy's implementation is substantially faster for equal weights.
    positive = valid_weight[valid_weight > 0.0]
    if positive.size == weights.size and np.allclose(positive, positive[0]):
        result = np.quantile(values, requested, axis=0)
        return result[0] if np.ndim(probabilities) == 0 else result

    trailing_shape = values.shape[1:]
    flattened = values.reshape(values.shape[0], -1)
    output = np.empty((requested.size, flattened.shape[1]), dtype=float)
    # Sorting 8,000 draws over a modest block avoids a second full-size array.
    for start in range(0, flattened.shape[1], 512):
        stop = min(start + 512, flattened.shape[1])
        block = flattened[:, start:stop]
        finite = np.isfinite(block)
        order = np.argsort(np.where(finite, block, np.inf), axis=0)
        sorted_values = np.take_along_axis(block, order, axis=0)
        sorted_weights = valid_weight[order] * np.take_along_axis(finite, order, axis=0)
        cumulative = np.cumsum(sorted_weights, axis=0)
        totals = cumulative[-1]
        if np.any(totals <= 0.0):
            raise ValueError("A requested quantile column contains no finite values.")
        cumulative /= totals
        for q_index, probability in enumerate(requested):
            locations = np.argmax(cumulative >= probability, axis=0)
            output[q_index, start:stop] = np.take_along_axis(
                sorted_values, locations[None, :], axis=0
            )[0]
    output = output.reshape((requested.size,) + trailing_shape)
    return output[0] if np.ndim(probabilities) == 0 else output


@dataclass
class EnsembleSummary:
    reference_density: np.ndarray
    normalization: np.ndarray
    feature_normalization: np.ndarray
    marginal1_quantiles: np.ndarray
    marginal2_quantiles: np.ndarray
    tail_mass_grid: np.ndarray
    tail_any_quantiles: np.ndarray
    tail_both_quantiles: np.ndarray
    tail_straddle_quantiles: np.ndarray
    tail_m1_scale: np.ndarray
    tail_m2_scale: np.ndarray
    tail_both_at_m1_scale: np.ndarray
    tail_straddle_at_m1_scale: np.ndarray
    tail_both_fraction_at_m1_scale: np.ndarray
    tail_straddle_fraction_at_m1_scale: np.ndarray
    tail_m1_boundary_limited: np.ndarray
    tail_m2_boundary_limited: np.ndarray
    tail_probability: float
    probabilities: tuple[float, float, float]
    maximum_normalization_error: float


def _marginal_percentile(
    density: np.ndarray,
    masses: np.ndarray,
    integration_weights: np.ndarray,
    probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Notebook-compatible draw-level marginal percentile and edge flag."""

    weighted = np.asarray(density, dtype=float) * np.asarray(
        integration_weights, dtype=float
    )[None, :]
    normalization = np.sum(weighted, axis=1)
    valid = np.isfinite(normalization) & (normalization > 0.0)
    cumulative = np.full_like(weighted, np.nan)
    cumulative[valid] = np.cumsum(weighted[valid], axis=1) / normalization[
        valid, None
    ]
    result = np.full(weighted.shape[0], np.nan)
    boundary_limited = np.zeros(weighted.shape[0], dtype=bool)
    for draw in np.flatnonzero(valid):
        usable = np.isfinite(cumulative[draw]) & (integration_weights > 0.0)
        if np.count_nonzero(usable) < 2:
            continue
        cdf_unique, unique_indices = np.unique(
            cumulative[draw, usable], return_index=True
        )
        mass_unique = np.asarray(masses, dtype=float)[usable][unique_indices]
        if cdf_unique[0] <= probability <= cdf_unique[-1]:
            result[draw] = np.interp(probability, cdf_unique, mass_unique)
            crossing = int(np.searchsorted(cumulative[draw], probability, side="left"))
            boundary_limited[draw] = crossing >= masses.size - 2
    return result, boundary_limited


def _survival_on_grid(
    density: np.ndarray,
    masses: np.ndarray,
    integration_weights: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Evaluate a marginal upper-tail probability on common thresholds."""

    weighted = np.asarray(density, dtype=float) * np.asarray(
        integration_weights, dtype=float
    )[None, :]
    normalization = np.sum(weighted, axis=1)
    cumulative = np.cumsum(weighted, axis=1) / normalization[:, None]
    result = np.empty((density.shape[0], thresholds.size), dtype=np.float64)
    for draw in range(density.shape[0]):
        cdf = np.interp(
            thresholds,
            masses,
            cumulative[draw],
            left=0.0,
            right=1.0,
        )
        result[draw] = np.clip(1.0 - cdf, 0.0, 1.0)
    return result


def _survival_at_draw_thresholds(
    density: np.ndarray,
    masses: np.ndarray,
    integration_weights: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Evaluate one marginal survival probability at one threshold per draw."""

    weighted = np.asarray(density, dtype=float) * np.asarray(
        integration_weights, dtype=float
    )[None, :]
    normalization = np.sum(weighted, axis=1)
    cumulative = np.cumsum(weighted, axis=1) / normalization[:, None]
    result = np.full(density.shape[0], np.nan)
    for draw, threshold in enumerate(np.asarray(thresholds, dtype=float)):
        if not np.isfinite(threshold):
            continue
        cdf = np.interp(
            threshold,
            masses,
            cumulative[draw],
            left=0.0,
            right=1.0,
        )
        result[draw] = float(np.clip(1.0 - cdf, 0.0, 1.0))
    return result


def compute_ensemble_summary(
    store: DensityStore,
    grid: Grid,
    *,
    credible_mass: float,
    batch_size: int,
    tail_probability: float = 0.999,
    ordered_tails: bool = True,
    tile_size: int = 32,
) -> EnsembleSummary:
    """Compute normalized marginal quantiles and the pointwise median field."""

    number = store.number_draws
    normalization = np.empty(number, dtype=float)
    feature_normalization = np.empty(number, dtype=float)
    marginal1 = np.empty((number, grid.m1.size), dtype=float)
    marginal2 = np.empty((number, grid.m2.size), dtype=float)
    cell_weights = grid.cell_weights
    input_weight1 = (
        grid.input_weight1
        if grid.input_weight1 is not None
        else quadrature_weights(grid.ell1)
    )
    input_weight2 = (
        grid.input_weight2
        if grid.input_weight2 is not None
        else quadrature_weights(grid.ell2)
    )
    input_cell_weights = np.multiply.outer(input_weight1, input_weight2) * grid.mask
    feature_factor = input_to_feature_factor(grid)

    for start, batch in store.iter_batches(batch_size):
        stop = start + batch.shape[0]
        valid = batch[:, grid.mask]
        if np.any(~np.isfinite(valid)) or np.any(valid < 0.0):
            bad = int(start + np.argwhere(~np.isfinite(valid) | (valid < 0.0))[0, 0])
            raise ValueError(f"Draw {bad} contains invalid density values.")
        batch = np.where(grid.mask[None, :, :], batch, 0.0)
        integrals = np.einsum(
            "bij,ij->b", batch, input_cell_weights, optimize=True
        )
        if np.any(~np.isfinite(integrals)) or np.any(integrals <= 0.0):
            raise ValueError(
                f"A draw in batch {start}:{stop} has invalid normalization."
            )
        normalization[start:stop] = integrals
        batch /= integrals[:, None, None]
        batch *= feature_factor[None, :, :]
        converted_integrals = np.einsum(
            "bij,ij->b", batch, cell_weights, optimize=True
        )
        if np.any(~np.isfinite(converted_integrals)) or np.any(
            converted_integrals <= 0.0
        ):
            raise ValueError(
                f"A draw in batch {start}:{stop} has invalid normalization "
                "after density-measure conversion."
            )
        feature_normalization[start:stop] = converted_integrals
        batch /= converted_integrals[:, None, None]
        marginal1[start:stop] = np.einsum(
            "bij,j->bi", batch, grid.weight2, optimize=True
        )
        marginal2[start:stop] = np.einsum(
            "bij,i->bj", batch, grid.weight1, optimize=True
        )

    alpha = 0.5 * (1.0 - credible_mass)
    probabilities = (alpha, 0.5, 1.0 - alpha)
    marginal1_quantiles = weighted_quantile(marginal1, probabilities, store.weights)
    marginal2_quantiles = weighted_quantile(marginal2, probabilities, store.weights)

    if ordered_tails:
        tail_m1_scale, tail_m1_boundary_limited = _marginal_percentile(
            marginal1, grid.m1, grid.weight1, tail_probability
        )
        tail_m2_scale, tail_m2_boundary_limited = _marginal_percentile(
            marginal2, grid.m2, grid.weight2, tail_probability
        )
        tail_mass_grid = np.unique(np.concatenate([grid.m1, grid.m2]))
        tail_any = _survival_on_grid(
            marginal1, grid.m1, grid.weight1, tail_mass_grid
        )
        tail_both = _survival_on_grid(
            marginal2, grid.m2, grid.weight2, tail_mass_grid
        )
        tail_straddle = np.maximum(tail_any - tail_both, 0.0)
        tail_any_quantiles = weighted_quantile(
            tail_any, probabilities, store.weights
        )
        tail_both_quantiles = weighted_quantile(
            tail_both, probabilities, store.weights
        )
        tail_straddle_quantiles = weighted_quantile(
            tail_straddle, probabilities, store.weights
        )
        tail_both_at_m1_scale = _survival_at_draw_thresholds(
            marginal2, grid.m2, grid.weight2, tail_m1_scale
        )
        target_tail = 1.0 - float(tail_probability)
        tail_both_at_m1_scale = np.clip(
            tail_both_at_m1_scale, 0.0, target_tail
        )
        tail_straddle_at_m1_scale = np.maximum(
            target_tail - tail_both_at_m1_scale, 0.0
        )
        tail_both_fraction_at_m1_scale = tail_both_at_m1_scale / target_tail
        tail_straddle_fraction_at_m1_scale = (
            tail_straddle_at_m1_scale / target_tail
        )
    else:
        tail_mass_grid = np.empty(0, dtype=float)
        tail_any_quantiles = np.empty((len(probabilities), 0), dtype=float)
        tail_both_quantiles = np.empty((len(probabilities), 0), dtype=float)
        tail_straddle_quantiles = np.empty((len(probabilities), 0), dtype=float)
        tail_m1_scale = np.full(number, np.nan)
        tail_m2_scale = np.full(number, np.nan)
        tail_both_at_m1_scale = np.full(number, np.nan)
        tail_straddle_at_m1_scale = np.full(number, np.nan)
        tail_both_fraction_at_m1_scale = np.full(number, np.nan)
        tail_straddle_fraction_at_m1_scale = np.full(number, np.nan)
        tail_m1_boundary_limited = np.zeros(number, dtype=bool)
        tail_m2_boundary_limited = np.zeros(number, dtype=bool)

    reference = np.zeros(grid.shape, dtype=float)
    for i0 in range(0, grid.shape[0], tile_size):
        i1 = min(i0 + tile_size, grid.shape[0])
        for j0 in range(0, grid.shape[1], tile_size):
            j1 = min(j0 + tile_size, grid.shape[1])
            tile = np.asarray(store.draws[:, i0:i1, j0:j1], dtype=float)
            tile /= normalization[:, None, None]
            tile *= feature_factor[None, i0:i1, j0:j1]
            tile /= feature_normalization[:, None, None]
            reference[i0:i1, j0:j1] = weighted_quantile(tile, 0.5, store.weights)
    reference = np.where(grid.mask, reference, 0.0)
    reference /= grid.integrate(reference)
    return EnsembleSummary(
        reference_density=reference,
        normalization=normalization,
        feature_normalization=feature_normalization,
        marginal1_quantiles=marginal1_quantiles,
        marginal2_quantiles=marginal2_quantiles,
        tail_mass_grid=tail_mass_grid,
        tail_any_quantiles=tail_any_quantiles,
        tail_both_quantiles=tail_both_quantiles,
        tail_straddle_quantiles=tail_straddle_quantiles,
        tail_m1_scale=tail_m1_scale,
        tail_m2_scale=tail_m2_scale,
        tail_both_at_m1_scale=tail_both_at_m1_scale,
        tail_straddle_at_m1_scale=tail_straddle_at_m1_scale,
        tail_both_fraction_at_m1_scale=tail_both_fraction_at_m1_scale,
        tail_straddle_fraction_at_m1_scale=tail_straddle_fraction_at_m1_scale,
        tail_m1_boundary_limited=tail_m1_boundary_limited,
        tail_m2_boundary_limited=tail_m2_boundary_limited,
        tail_probability=float(tail_probability),
        probabilities=probabilities,
        maximum_normalization_error=float(np.max(np.abs(normalization - 1.0))),
    )


def _point_geometry(feature: PointFeature, grid: Grid) -> np.ndarray:
    return np.asarray(
        [grid.geometry1[feature.index[0]], grid.geometry2[feature.index[1]]]
    )


def _plateau_geometry(feature: PlateauFeature, grid: Grid) -> np.ndarray:
    return np.asarray(
        [
            grid.geometry1[feature.center_index[0]],
            grid.geometry2[feature.center_index[1]],
        ]
    )


def _branch_topology_geometry(feature: BranchFeature) -> np.ndarray:
    points = feature.topology_points_geometry
    return np.concatenate([points[0], points[-1]])


def _event_topology_geometry(feature: CurveEventFeature) -> np.ndarray:
    return np.asarray(feature.topology_geometry, dtype=float).reshape(-1)


def _shoulder_topology_geometry(feature: ShoulderFeature) -> np.ndarray:
    return np.asarray(feature.topology_geometry, dtype=float).reshape(-1)


def _domain_length(grid: Grid) -> float:
    return float(
        np.hypot(
            grid.geometry1[-1] - grid.geometry1[0],
            grid.geometry2[-1] - grid.geometry2[0],
        )
    )


def _assignment(
    template_positions: list[np.ndarray],
    candidate_positions: list[np.ndarray],
    *,
    maximum_distance: float,
    compatibility: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    if not template_positions or not candidate_positions:
        return [], list(range(len(candidate_positions)))
    cost = np.empty((len(template_positions), len(candidate_positions)), dtype=float)
    for row, first in enumerate(template_positions):
        for column, second in enumerate(candidate_positions):
            cost[row, column] = np.linalg.norm(first - second)
    if compatibility is not None:
        compatibility = np.asarray(compatibility, dtype=bool)
        if compatibility.shape != cost.shape:
            raise ValueError("compatibility must match the assignment cost matrix.")
        cost = np.where(compatibility, cost, maximum_distance + 1.0)
    rows, columns = linear_sum_assignment(cost)
    matches = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if cost[row, column] <= maximum_distance
        and (compatibility is None or compatibility[row, column])
    ]
    matched_candidates = {column for _, column in matches}
    unmatched = [
        index
        for index in range(len(candidate_positions))
        if index not in matched_candidates
    ]
    return matches, unmatched


def _match_quality(
    template_positions: list[np.ndarray],
    candidate_positions: list[np.ndarray],
    row: int,
    column: int,
    *,
    resolution: float,
    compatibility: np.ndarray | None = None,
) -> tuple[float, float, bool]:
    """Distance and nearest-assignment margin for one accepted match."""

    selected = float(
        np.linalg.norm(template_positions[row] - candidate_positions[column])
    )
    alternatives = [
        float(np.linalg.norm(template_positions[row] - candidate_positions[index]))
        for index in range(len(candidate_positions))
        if index != column
        and (compatibility is None or compatibility[row, index])
    ]
    alternatives.extend(
        float(np.linalg.norm(template_positions[index] - candidate_positions[column]))
        for index in range(len(template_positions))
        if index != row
        and (compatibility is None or compatibility[index, column])
    )
    if not alternatives:
        return selected, math.nan, False
    margin = min(alternatives) - selected
    return selected, margin, bool(margin <= resolution)


def _boundary_role_compatible(first: str, second: str) -> bool:
    """Prevent an interior hierarchy endpoint from matching a boundary one."""

    return (first == "none") == (second == "none")


def _branch_compatibility(
    templates: list[BranchFeature], candidates: list[BranchFeature]
) -> np.ndarray:
    """Compatibility of immutable saddle--extremum branch identities."""

    return np.asarray(
        [
            [
                _boundary_role_compatible(
                    template.saddle_boundary_type,
                    candidate.saddle_boundary_type,
                )
                and _boundary_role_compatible(
                    template.extremum_boundary_type,
                    candidate.extremum_boundary_type,
                )
                for candidate in candidates
            ]
            for template in templates
        ],
        dtype=bool,
    )


def assign_reference_identifiers(analysis: FieldAnalysis, grid: Grid) -> None:
    prefixes = {"peak": "P2D", "pit": "D2D", "saddle": "X2D"}
    for kind, prefix in prefixes.items():
        features = [feature for feature in analysis.points if feature.kind == kind]
        features.sort(
            key=lambda item: (
                _point_geometry(item, grid)[0],
                _point_geometry(item, grid)[1],
            )
        )
        for number, feature in enumerate(features, start=1):
            feature.identifier = f"{prefix}{number}"
    for kind, prefix in (("ridge", "R2D"), ("valley", "V2D")):
        features = [feature for feature in analysis.branches if feature.kind == kind]
        features.sort(
            key=lambda item: (item.points_geometry[0, 0], item.points_geometry[0, 1])
        )
        for number, feature in enumerate(features, start=1):
            feature.identifier = f"{prefix}{number}"
    ordered_plateaus = sorted(
        analysis.plateaus,
        key=lambda item: (
            _plateau_geometry(item, grid)[0],
            _plateau_geometry(item, grid)[1],
        ),
    )
    for number, feature in enumerate(ordered_plateaus, start=1):
        feature.identifier = f"L2D{number}"

    for branch in analysis.branches:
        start = next(
            (
                point
                for point in analysis.points
                if point.kind == "saddle" and point.index == branch.saddle_index
            ),
            None,
        )
        endpoint_kind = "peak" if branch.kind == "ridge" else "pit"
        end = next(
            (
                point
                for point in analysis.points
                if point.kind == endpoint_kind and point.index == branch.extremum_index
            ),
            None,
        )
        branch.start_identifier = start.identifier if start is not None else ""
        branch.end_identifier = end.identifier if end is not None else ""

    for kind, prefix in (("ridge", "R2DE"), ("valley", "V2DE")):
        events = [event for event in analysis.events if event.kind == kind]
        events.sort(
            key=lambda event: (
                float(np.mean(event.topology_geometry[:, 0])),
                float(np.mean(event.topology_geometry[:, 1])),
            )
        )
        for number, event in enumerate(events, start=1):
            event.identifier = f"{prefix}{number}"
            event.arm_identifiers = tuple(
                analysis.branches[index].identifier for index in event.arm_indices
            )

    ordered_shoulders = sorted(
        analysis.shoulders,
        key=lambda feature: (
            float(np.mean(feature.topology_geometry[:, 0])),
            float(np.mean(feature.topology_geometry[:, 1])),
        ),
    )
    for number, feature in enumerate(ordered_shoulders, start=1):
        feature.identifier = f"S2D{number}"


def add_scale_persistence(
    reference: FieldAnalysis, other_scales: Iterable[FieldAnalysis], grid: Grid
) -> None:
    """Count reference features recovered at the other requested scales."""

    other_scales = list(other_scales)
    total = 1 + len(other_scales)
    maximum = max(3.0 * grid.typical_spacing, 0.08 * _domain_length(grid))
    for feature in reference.points:
        feature.scale_count = 1
        feature.scale_total = total
    for feature in reference.branches:
        feature.scale_count = 1
        feature.scale_total = total

    for analysis in other_scales:
        for kind in ("peak", "pit", "saddle"):
            templates = [
                feature for feature in reference.points if feature.kind == kind
            ]
            candidates = [
                feature for feature in analysis.points if feature.kind == kind
            ]
            matches, _ = _assignment(
                [_point_geometry(item, grid) for item in templates],
                [_point_geometry(item, grid) for item in candidates],
                maximum_distance=maximum,
            )
            for row, _ in matches:
                templates[row].scale_count += 1
        for kind in ("ridge", "valley"):
            templates = [
                feature for feature in reference.branches if feature.kind == kind
            ]
            candidates = [
                feature for feature in analysis.branches if feature.kind == kind
            ]
            compatibility = _branch_compatibility(templates, candidates)
            template_positions = [
                _branch_topology_geometry(item)
                for item in templates
            ]
            candidate_positions = [
                _branch_topology_geometry(item)
                for item in candidates
            ]
            matches, _ = _assignment(
                template_positions,
                candidate_positions,
                maximum_distance=math.sqrt(2.0) * maximum,
                compatibility=compatibility,
            )
            for row, _ in matches:
                templates[row].scale_count += 1


def retain_scale_persistent_reference(
    reference: FieldAnalysis,
    *,
    minimum_scales: int | None = None,
) -> dict[str, int]:
    """Remove reference candidates that fail the multiscale requirement.

    Branches keep the saddle--extremum identity assigned by the complete
    hierarchy.  Paired events are retained only when both of their original
    arms survive; the remaining arms are never paired again.
    """

    totals = [
        feature.scale_total
        for feature in [*reference.points, *reference.branches]
        if feature.kind in {"peak", "pit", "ridge", "valley"}
    ]
    total = max(totals, default=1)
    required = (
        int(minimum_scales)
        if minimum_scales is not None
        else max(1, int(math.ceil(2.0 * total / 3.0)))
    )
    if required < 1 or required > total:
        raise ValueError("minimum_scales must lie between one and scale_total.")

    old_points = list(reference.points)
    stable_extrema = {
        (point.kind, point.index)
        for point in old_points
        if point.kind in {"peak", "pit"} and point.scale_count >= required
    }
    old_branches = list(reference.branches)
    retained_indices = [
        index
        for index, branch in enumerate(old_branches)
        if branch.scale_count >= required
        and (
            ("peak" if branch.kind == "ridge" else "pit"),
            branch.extremum_index,
        )
        in stable_extrema
    ]
    index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(retained_indices)
    }
    reference.branches = [old_branches[index] for index in retained_indices]

    retained_events: list[CurveEventFeature] = []
    for event in reference.events:
        if not all(index in index_map for index in event.arm_indices):
            continue
        event.arm_indices = tuple(index_map[index] for index in event.arm_indices)
        retained_events.append(event)
    reference.events = retained_events

    used_saddles = {branch.saddle_index for branch in reference.branches}
    reference.points = [
        point
        for point in old_points
        if (
            point.kind in {"peak", "pit"}
            and (point.kind, point.index) in stable_extrema
        )
        or (point.kind == "saddle" and point.index in used_saddles)
    ]
    return {
        "required": required,
        "points_removed": sum(
            point.kind in {"peak", "pit"}
            and (point.kind, point.index) not in stable_extrema
            for point in old_points
        ),
        "branches_removed": len(old_branches) - len(reference.branches),
    }


def remove_corner_boundary_pits(reference: FieldAnalysis) -> dict[str, int]:
    """Remove pits at intersections of independent domain boundaries.

    A pit touching both the outer array boundary and the support/mask
    boundary is a corner artefact of the gridded domain. Single-boundary pits
    remain eligible because they can anchor a valley that extends into the
    physical support. Branches and paired events incident on an excluded
    corner pit are removed together, while all other hierarchy connectivity is
    preserved.
    """

    corner_pits = {
        point.index
        for point in reference.points
        if point.kind == "pit" and point.boundary_type == "outer+mask"
    }
    if not corner_pits:
        return {
            "corner_pits_removed": 0,
            "branches_removed": 0,
            "events_removed": 0,
        }

    old_branches = list(reference.branches)
    retained_indices = [
        index
        for index, branch in enumerate(old_branches)
        if not (
            branch.kind == "valley"
            and branch.extremum_index in corner_pits
        )
    ]
    index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(retained_indices)
    }
    reference.branches = [old_branches[index] for index in retained_indices]

    old_events = list(reference.events)
    retained_events: list[CurveEventFeature] = []
    for event in old_events:
        if not all(index in index_map for index in event.arm_indices):
            continue
        event.arm_indices = tuple(index_map[index] for index in event.arm_indices)
        retained_events.append(event)
    reference.events = retained_events

    used_saddles = {branch.saddle_index for branch in reference.branches}
    reference.points = [
        point
        for point in reference.points
        if (
            point.kind == "pit" and point.index not in corner_pits
        )
        or point.kind == "peak"
        or (point.kind == "saddle" and point.index in used_saddles)
    ]
    return {
        "corner_pits_removed": len(corner_pits),
        "branches_removed": len(old_branches) - len(reference.branches),
        "events_removed": len(old_events) - len(reference.events),
    }


@dataclass
class PointTracker:
    template: PointFeature
    number_draws: int
    grid_shape: tuple[int, int]
    is_reference: bool = True
    present: np.ndarray = field(init=False)
    location: np.ndarray = field(init=False)
    persistence: np.ndarray = field(init=False)
    region_mass: np.ndarray = field(init=False)
    value: np.ndarray = field(init=False)
    boundary: np.ndarray = field(init=False)
    outer_boundary: np.ndarray = field(init=False)
    mask_boundary: np.ndarray = field(init=False)
    base_level: np.ndarray = field(init=False)
    half_level: np.ndarray = field(init=False)
    feature_probability: np.ndarray = field(init=False)
    width_major: np.ndarray = field(init=False)
    width_minor: np.ndarray = field(init=False)
    width_orientation: np.ndarray = field(init=False)
    width_area: np.ndarray = field(init=False)
    log_contrast: np.ndarray = field(init=False)
    geometry_valid: np.ndarray = field(init=False)
    measurement_valid: np.ndarray = field(init=False)
    relative_prominence: np.ndarray = field(init=False)
    mass_centroid: np.ndarray = field(init=False)
    spatial_covariance_mass: np.ndarray = field(init=False)
    projected_bounds_mass: np.ndarray = field(init=False)
    relative_projected_widths: np.ndarray = field(init=False)
    deficit_probability: np.ndarray = field(init=False)
    projection_m1: np.ndarray = field(init=False, repr=False)
    projection_m2: np.ndarray = field(init=False, repr=False)
    region_weight_sum: np.ndarray = field(init=False, repr=False)
    match_distance: np.ndarray = field(init=False)
    match_margin: np.ndarray = field(init=False)
    match_ambiguous: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.present = np.zeros(self.number_draws, dtype=bool)
        self.location = np.full((self.number_draws, 2), np.nan)
        self.persistence = np.full(self.number_draws, np.nan)
        self.region_mass = np.full(self.number_draws, np.nan)
        self.value = np.full(self.number_draws, np.nan)
        self.boundary = np.zeros(self.number_draws, dtype=bool)
        self.outer_boundary = np.zeros(self.number_draws, dtype=bool)
        self.mask_boundary = np.zeros(self.number_draws, dtype=bool)
        self.base_level = np.full(self.number_draws, np.nan)
        self.half_level = np.full(self.number_draws, np.nan)
        self.feature_probability = np.full(self.number_draws, np.nan)
        self.width_major = np.full(self.number_draws, np.nan)
        self.width_minor = np.full(self.number_draws, np.nan)
        self.width_orientation = np.full(self.number_draws, np.nan)
        self.width_area = np.full(self.number_draws, np.nan)
        self.log_contrast = np.full(self.number_draws, np.nan)
        self.geometry_valid = np.zeros(self.number_draws, dtype=bool)
        self.measurement_valid = np.zeros(self.number_draws, dtype=bool)
        self.relative_prominence = np.full(self.number_draws, np.nan)
        self.mass_centroid = np.full((self.number_draws, 2), np.nan)
        self.spatial_covariance_mass = np.full((self.number_draws, 3), np.nan)
        self.projected_bounds_mass = np.full((self.number_draws, 4), np.nan)
        self.relative_projected_widths = np.full((self.number_draws, 2), np.nan)
        self.deficit_probability = np.full(self.number_draws, np.nan)
        self.projection_m1 = np.full(
            (self.number_draws, self.grid_shape[0]), np.nan, dtype=np.float32
        )
        self.projection_m2 = np.full(
            (self.number_draws, self.grid_shape[1]), np.nan, dtype=np.float32
        )
        self.region_weight_sum = np.zeros(self.grid_shape, dtype=np.float64)
        self.match_distance = np.full(self.number_draws, np.nan)
        self.match_margin = np.full(self.number_draws, np.nan)
        self.match_ambiguous = np.zeros(self.number_draws, dtype=bool)

    def record(
        self,
        draw: int,
        feature: PointFeature,
        grid: Grid,
        draw_weight: float,
        match_quality: tuple[float, float, bool] | None = None,
    ) -> None:
        self.present[draw] = True
        self.location[draw] = _point_geometry(feature, grid)
        self.persistence[draw] = feature.persistence
        self.region_mass[draw] = feature.region_mass
        self.value[draw] = feature.value
        self.boundary[draw] = feature.boundary
        self.outer_boundary[draw] = "outer" in feature.boundary_type
        self.mask_boundary[draw] = "mask" in feature.boundary_type
        self.base_level[draw] = feature.base_level
        self.half_level[draw] = feature.half_level
        self.feature_probability[draw] = feature.feature_probability
        self.width_major[draw] = feature.width_major
        self.width_minor[draw] = feature.width_minor
        if np.isfinite(feature.width_orientation):
            reference = self.template.width_orientation
            orientation = feature.width_orientation
            if np.isfinite(reference):
                orientation = reference + (
                    (orientation - reference + 0.5 * math.pi) % math.pi
                    - 0.5 * math.pi
                )
            self.width_orientation[draw] = orientation
        self.width_area[draw] = feature.width_area
        self.log_contrast[draw] = feature.log_contrast
        self.geometry_valid[draw] = feature.geometry_valid
        self.measurement_valid[draw] = feature.measurement_valid
        self.relative_prominence[draw] = feature.relative_prominence
        self.mass_centroid[draw] = feature.mass_centroid
        self.spatial_covariance_mass[draw] = feature.spatial_covariance_mass
        self.projected_bounds_mass[draw] = feature.projected_bounds_mass
        self.relative_projected_widths[draw] = feature.relative_projected_widths
        self.deficit_probability[draw] = feature.deficit_probability
        if feature.projection_m1.size == self.grid_shape[0]:
            self.projection_m1[draw] = feature.projection_m1
        if feature.projection_m2.size == self.grid_shape[1]:
            self.projection_m2[draw] = feature.projection_m2
        if feature.geometry_valid and feature.feature_region_mask is not None:
            self.region_weight_sum += float(draw_weight) * feature.feature_region_mask
        if match_quality is not None:
            self.match_distance[draw], self.match_margin[draw], self.match_ambiguous[draw] = (
                match_quality
            )


@dataclass
class BranchTracker:
    template: BranchFeature
    number_draws: int
    curve_points: int
    width_points: int
    grid_shape: tuple[int, int]
    is_reference: bool = True
    present: np.ndarray = field(init=False)
    endpoints: np.ndarray = field(init=False)
    topology_endpoints: np.ndarray = field(init=False)
    prominence: np.ndarray = field(init=False)
    length: np.ndarray = field(init=False)
    full_length: np.ndarray = field(init=False)
    retained_fraction: np.ndarray = field(init=False)
    boundary: np.ndarray = field(init=False)
    outer_boundary: np.ndarray = field(init=False)
    mask_boundary: np.ndarray = field(init=False)
    curve_available: np.ndarray = field(init=False)
    low_density_truncated: np.ndarray = field(init=False)
    curves: np.ndarray = field(init=False)
    feature_probability: np.ndarray = field(init=False)
    width_left: np.ndarray = field(init=False)
    width_right: np.ndarray = field(init=False)
    width_median: np.ndarray = field(init=False)
    width_along_lower: np.ndarray = field(init=False)
    width_along_upper: np.ndarray = field(init=False)
    valid_width_fraction: np.ndarray = field(init=False)
    fallback_width_fraction: np.ndarray = field(init=False)
    log_contrast: np.ndarray = field(init=False)
    geometry_valid: np.ndarray = field(init=False)
    left_width_profiles: np.ndarray = field(init=False, repr=False)
    right_width_profiles: np.ndarray = field(init=False, repr=False)
    width_profiles: np.ndarray = field(init=False, repr=False)
    log_contrast_profiles: np.ndarray = field(init=False, repr=False)
    region_weight_sum: np.ndarray = field(init=False, repr=False)
    match_distance: np.ndarray = field(init=False)
    match_margin: np.ndarray = field(init=False)
    match_ambiguous: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.present = np.zeros(self.number_draws, dtype=bool)
        self.endpoints = np.full((self.number_draws, 4), np.nan)
        self.topology_endpoints = np.full((self.number_draws, 4), np.nan)
        self.prominence = np.full(self.number_draws, np.nan)
        self.length = np.full(self.number_draws, np.nan)
        self.full_length = np.full(self.number_draws, np.nan)
        self.retained_fraction = np.full(self.number_draws, np.nan)
        self.boundary = np.zeros(self.number_draws, dtype=bool)
        self.outer_boundary = np.zeros(self.number_draws, dtype=bool)
        self.mask_boundary = np.zeros(self.number_draws, dtype=bool)
        self.curve_available = np.zeros(self.number_draws, dtype=bool)
        self.low_density_truncated = np.zeros(self.number_draws, dtype=bool)
        self.curves = np.full(
            (self.number_draws, self.curve_points, 2), np.nan, dtype=np.float32
        )
        self.feature_probability = np.full(self.number_draws, np.nan)
        self.width_left = np.full(self.number_draws, np.nan)
        self.width_right = np.full(self.number_draws, np.nan)
        self.width_median = np.full(self.number_draws, np.nan)
        self.width_along_lower = np.full(self.number_draws, np.nan)
        self.width_along_upper = np.full(self.number_draws, np.nan)
        self.valid_width_fraction = np.zeros(self.number_draws)
        self.fallback_width_fraction = np.zeros(self.number_draws)
        self.log_contrast = np.full(self.number_draws, np.nan)
        self.geometry_valid = np.zeros(self.number_draws, dtype=bool)
        profile_shape = (self.number_draws, self.width_points)
        self.left_width_profiles = np.full(profile_shape, np.nan, dtype=np.float32)
        self.right_width_profiles = np.full(profile_shape, np.nan, dtype=np.float32)
        self.width_profiles = np.full(profile_shape, np.nan, dtype=np.float32)
        self.log_contrast_profiles = np.full(
            profile_shape, np.nan, dtype=np.float32
        )
        self.region_weight_sum = np.zeros(self.grid_shape, dtype=np.float64)
        self.match_distance = np.full(self.number_draws, np.nan)
        self.match_margin = np.full(self.number_draws, np.nan)
        self.match_ambiguous = np.zeros(self.number_draws, dtype=bool)

    def record(
        self,
        draw: int,
        feature: BranchFeature,
        draw_weight: float,
        match_quality: tuple[float, float, bool] | None = None,
    ) -> None:
        self.present[draw] = True
        self.endpoints[draw] = np.concatenate(
            [feature.points_geometry[0], feature.points_geometry[-1]]
        )
        self.topology_endpoints[draw] = _branch_topology_geometry(feature)
        self.prominence[draw] = feature.prominence
        self.length[draw] = feature.length
        self.full_length[draw] = feature.full_length
        self.retained_fraction[draw] = feature.retained_fraction
        self.boundary[draw] = feature.boundary
        self.outer_boundary[draw] = "outer" in feature.boundary_type
        self.mask_boundary[draw] = "mask" in feature.boundary_type
        self.curve_available[draw] = feature.curve_available
        self.low_density_truncated[draw] = feature.low_density_truncated
        if feature.curve_available:
            self.curves[draw] = feature.points_geometry.astype(np.float32)
        self.feature_probability[draw] = feature.feature_probability
        self.width_left[draw] = feature.width_left
        self.width_right[draw] = feature.width_right
        self.width_median[draw] = feature.width_median
        self.width_along_lower[draw] = feature.width_along_lower
        self.width_along_upper[draw] = feature.width_along_upper
        self.valid_width_fraction[draw] = feature.valid_width_fraction
        self.fallback_width_fraction[draw] = feature.fallback_width_fraction
        self.log_contrast[draw] = feature.log_contrast
        self.geometry_valid[draw] = feature.geometry_valid
        if feature.width_profile.size == self.width_points:
            self.left_width_profiles[draw] = feature.left_width_profile
            self.right_width_profiles[draw] = feature.right_width_profile
            self.width_profiles[draw] = feature.width_profile
            self.log_contrast_profiles[draw] = feature.log_contrast_profile
        if feature.geometry_valid and feature.feature_region_mask is not None:
            self.region_weight_sum += float(draw_weight) * feature.feature_region_mask
        if match_quality is not None:
            self.match_distance[draw], self.match_margin[draw], self.match_ambiguous[draw] = (
                match_quality
            )


@dataclass
class EventTracker:
    """Posterior tracker for one paired ridge or valley event."""

    template: CurveEventFeature
    number_draws: int
    grid_shape: tuple[int, int]
    present: np.ndarray = field(init=False)
    curve_available: np.ndarray = field(init=False)
    region_valid: np.ndarray = field(init=False)
    topology_geometry: np.ndarray = field(init=False)
    region_probability: np.ndarray = field(init=False)
    deficit_probability: np.ndarray = field(init=False)
    mass_centroid: np.ndarray = field(init=False)
    spatial_covariance_mass: np.ndarray = field(init=False)
    projected_bounds_mass: np.ndarray = field(init=False)
    relative_projected_widths: np.ndarray = field(init=False)
    projection_m1: np.ndarray = field(init=False, repr=False)
    projection_m2: np.ndarray = field(init=False, repr=False)
    extent: np.ndarray = field(init=False)
    full_extent: np.ndarray = field(init=False)
    retained_fraction: np.ndarray = field(init=False)
    bounded_fraction: np.ndarray = field(init=False)
    longest_bounded_fraction: np.ndarray = field(init=False)
    width_median: np.ndarray = field(init=False)
    fallback_fraction: np.ndarray = field(init=False)
    log_contrast: np.ndarray = field(init=False)
    relative_contrast: np.ndarray = field(init=False)
    boundary: np.ndarray = field(init=False)
    low_density_truncated: np.ndarray = field(init=False)
    region_weight_sum: np.ndarray = field(init=False, repr=False)
    match_distance: np.ndarray = field(init=False)
    match_margin: np.ndarray = field(init=False)
    match_ambiguous: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        number = self.number_draws
        self.present = np.zeros(number, dtype=bool)
        self.curve_available = np.zeros(number, dtype=bool)
        self.region_valid = np.zeros(number, dtype=bool)
        self.topology_geometry = np.full((number, 6), np.nan)
        self.region_probability = np.full(number, np.nan)
        self.deficit_probability = np.full(number, np.nan)
        self.mass_centroid = np.full((number, 2), np.nan)
        self.spatial_covariance_mass = np.full((number, 3), np.nan)
        self.projected_bounds_mass = np.full((number, 4), np.nan)
        self.relative_projected_widths = np.full((number, 2), np.nan)
        self.projection_m1 = np.full(
            (number, self.grid_shape[0]), np.nan, dtype=np.float32
        )
        self.projection_m2 = np.full(
            (number, self.grid_shape[1]), np.nan, dtype=np.float32
        )
        self.extent = np.full(number, np.nan)
        self.full_extent = np.full(number, np.nan)
        self.retained_fraction = np.full(number, np.nan)
        self.bounded_fraction = np.zeros(number)
        self.longest_bounded_fraction = np.zeros(number)
        self.width_median = np.full(number, np.nan)
        self.fallback_fraction = np.zeros(number)
        self.log_contrast = np.full(number, np.nan)
        self.relative_contrast = np.full(number, np.nan)
        self.boundary = np.zeros(number, dtype=bool)
        self.low_density_truncated = np.zeros(number, dtype=bool)
        self.region_weight_sum = np.zeros(self.grid_shape, dtype=np.float64)
        self.match_distance = np.full(number, np.nan)
        self.match_margin = np.full(number, np.nan)
        self.match_ambiguous = np.zeros(number, dtype=bool)

    def record(
        self,
        draw: int,
        feature: CurveEventFeature,
        draw_weight: float,
        match_quality: tuple[float, float, bool] | None = None,
    ) -> None:
        self.present[draw] = True
        self.curve_available[draw] = feature.curve_available
        self.region_valid[draw] = feature.region_valid
        self.topology_geometry[draw] = _event_topology_geometry(feature)
        self.region_probability[draw] = feature.region_probability
        self.deficit_probability[draw] = feature.deficit_probability
        self.mass_centroid[draw] = feature.mass_centroid
        self.spatial_covariance_mass[draw] = feature.spatial_covariance_mass
        self.projected_bounds_mass[draw] = feature.projected_bounds_mass
        self.relative_projected_widths[draw] = feature.relative_projected_widths
        if feature.projection_m1.size == self.grid_shape[0]:
            self.projection_m1[draw] = feature.projection_m1
        if feature.projection_m2.size == self.grid_shape[1]:
            self.projection_m2[draw] = feature.projection_m2
        self.extent[draw] = feature.extent
        self.full_extent[draw] = feature.full_extent
        self.retained_fraction[draw] = feature.retained_fraction
        self.bounded_fraction[draw] = feature.bounded_fraction
        self.longest_bounded_fraction[draw] = feature.longest_bounded_fraction
        self.width_median[draw] = feature.width_median
        self.fallback_fraction[draw] = feature.fallback_fraction
        self.log_contrast[draw] = feature.log_contrast
        self.relative_contrast[draw] = feature.relative_contrast
        self.boundary[draw] = feature.boundary
        self.low_density_truncated[draw] = feature.low_density_truncated
        if feature.region_valid and feature.window_mask is not None:
            self.region_weight_sum += float(draw_weight) * feature.window_mask
        if match_quality is not None:
            self.match_distance[draw], self.match_margin[draw], self.match_ambiguous[draw] = (
                match_quality
            )


@dataclass
class ShoulderTracker:
    """Posterior tracker for one directional slope-change front."""

    template: ShoulderFeature
    number_draws: int
    curve_points: int
    grid_shape: tuple[int, int]
    present: np.ndarray = field(init=False)
    curve_available: np.ndarray = field(init=False)
    region_valid: np.ndarray = field(init=False)
    topology_geometry: np.ndarray = field(init=False)
    curves: np.ndarray = field(init=False, repr=False)
    region_probability: np.ndarray = field(init=False)
    mass_centroid: np.ndarray = field(init=False)
    spatial_covariance_mass: np.ndarray = field(init=False)
    projected_bounds_mass: np.ndarray = field(init=False)
    relative_projected_widths: np.ndarray = field(init=False)
    projection_m1: np.ndarray = field(init=False, repr=False)
    projection_m2: np.ndarray = field(init=False, repr=False)
    extent: np.ndarray = field(init=False)
    full_extent: np.ndarray = field(init=False)
    retained_fraction: np.ndarray = field(init=False)
    bounded_fraction: np.ndarray = field(init=False)
    longest_bounded_fraction: np.ndarray = field(init=False)
    width_median: np.ndarray = field(init=False)
    alpha_max: np.ndarray = field(init=False)
    slope_pre: np.ndarray = field(init=False)
    slope_post: np.ndarray = field(init=False)
    slope_contrast: np.ndarray = field(init=False)
    boundary: np.ndarray = field(init=False)
    low_density_truncated: np.ndarray = field(init=False)
    region_weight_sum: np.ndarray = field(init=False, repr=False)
    match_distance: np.ndarray = field(init=False)
    match_margin: np.ndarray = field(init=False)
    match_ambiguous: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        number = self.number_draws
        self.present = np.zeros(number, dtype=bool)
        self.curve_available = np.zeros(number, dtype=bool)
        self.region_valid = np.zeros(number, dtype=bool)
        self.topology_geometry = np.full((number, 6), np.nan)
        self.curves = np.full(
            (number, self.curve_points, 2), np.nan, dtype=np.float32
        )
        self.region_probability = np.full(number, np.nan)
        self.mass_centroid = np.full((number, 2), np.nan)
        self.spatial_covariance_mass = np.full((number, 3), np.nan)
        self.projected_bounds_mass = np.full((number, 4), np.nan)
        self.relative_projected_widths = np.full((number, 2), np.nan)
        self.projection_m1 = np.full(
            (number, self.grid_shape[0]), np.nan, dtype=np.float32
        )
        self.projection_m2 = np.full(
            (number, self.grid_shape[1]), np.nan, dtype=np.float32
        )
        self.extent = np.full(number, np.nan)
        self.full_extent = np.full(number, np.nan)
        self.retained_fraction = np.full(number, np.nan)
        self.bounded_fraction = np.zeros(number)
        self.longest_bounded_fraction = np.zeros(number)
        self.width_median = np.full(number, np.nan)
        self.alpha_max = np.full(number, np.nan)
        self.slope_pre = np.full(number, np.nan)
        self.slope_post = np.full(number, np.nan)
        self.slope_contrast = np.full(number, np.nan)
        self.boundary = np.zeros(number, dtype=bool)
        self.low_density_truncated = np.zeros(number, dtype=bool)
        self.region_weight_sum = np.zeros(self.grid_shape, dtype=np.float64)
        self.match_distance = np.full(number, np.nan)
        self.match_margin = np.full(number, np.nan)
        self.match_ambiguous = np.zeros(number, dtype=bool)

    def record(
        self,
        draw: int,
        feature: ShoulderFeature,
        draw_weight: float,
        match_quality: tuple[float, float, bool] | None = None,
    ) -> None:
        self.present[draw] = True
        self.curve_available[draw] = feature.curve_available
        self.region_valid[draw] = feature.region_valid
        self.topology_geometry[draw] = _shoulder_topology_geometry(feature)
        if feature.curve_available:
            self.curves[draw] = feature.center_curve_geometry.astype(np.float32)
        self.region_probability[draw] = feature.region_probability
        self.mass_centroid[draw] = feature.mass_centroid
        self.spatial_covariance_mass[draw] = feature.spatial_covariance_mass
        self.projected_bounds_mass[draw] = feature.projected_bounds_mass
        self.relative_projected_widths[draw] = feature.relative_projected_widths
        if feature.projection_m1.size == self.grid_shape[0]:
            self.projection_m1[draw] = feature.projection_m1
        if feature.projection_m2.size == self.grid_shape[1]:
            self.projection_m2[draw] = feature.projection_m2
        self.extent[draw] = feature.extent
        self.full_extent[draw] = feature.full_extent
        self.retained_fraction[draw] = feature.retained_fraction
        self.bounded_fraction[draw] = feature.bounded_fraction
        self.longest_bounded_fraction[draw] = feature.longest_bounded_fraction
        self.width_median[draw] = feature.width_median
        self.alpha_max[draw] = feature.alpha_max
        self.slope_pre[draw] = feature.slope_pre
        self.slope_post[draw] = feature.slope_post
        self.slope_contrast[draw] = feature.slope_contrast
        self.boundary[draw] = feature.boundary
        self.low_density_truncated[draw] = feature.low_density_truncated
        if feature.region_valid and feature.window_mask is not None:
            self.region_weight_sum += float(draw_weight) * feature.window_mask
        if match_quality is not None:
            self.match_distance[draw], self.match_margin[draw], self.match_ambiguous[draw] = (
                match_quality
            )


@dataclass
class PlateauTracker:
    template: PlateauFeature
    number_draws: int
    is_reference: bool = True
    present: np.ndarray = field(init=False)
    location: np.ndarray = field(init=False)
    area: np.ndarray = field(init=False)
    probability_mass: np.ndarray = field(init=False)
    contrast: np.ndarray = field(init=False)
    boundary: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.present = np.zeros(self.number_draws, dtype=bool)
        self.location = np.full((self.number_draws, 2), np.nan)
        self.area = np.full(self.number_draws, np.nan)
        self.probability_mass = np.full(self.number_draws, np.nan)
        self.contrast = np.full(self.number_draws, np.nan)
        self.boundary = np.zeros(self.number_draws, dtype=bool)

    def record(self, draw: int, feature: PlateauFeature, grid: Grid) -> None:
        self.present[draw] = True
        self.location[draw] = _plateau_geometry(feature, grid)
        self.area[draw] = feature.area
        self.probability_mass[draw] = feature.probability_mass
        self.contrast[draw] = feature.contrast
        self.boundary[draw] = feature.boundary


@dataclass
class PosteriorCatalogue:
    number_draws: int
    grid: Grid
    point_trackers: list[PointTracker]
    branch_trackers: list[BranchTracker]
    event_trackers: list[EventTracker]
    shoulder_trackers: list[ShoulderTracker]
    plateau_trackers: list[PlateauTracker]
    topology_signatures: list[tuple[int, int, int, int, int, int]]

    @classmethod
    def from_reference(
        cls, reference: FieldAnalysis, number_draws: int, grid: Grid
    ) -> PosteriorCatalogue:
        return cls(
            number_draws=number_draws,
            grid=grid,
            point_trackers=[
                PointTracker(feature, number_draws, grid.shape)
                for feature in reference.points
                if feature.kind in {"peak", "pit"}
            ],
            branch_trackers=[
                BranchTracker(
                    feature,
                    number_draws,
                    feature.points_geometry.shape[0],
                    feature.width_profile.size,
                    grid.shape,
                )
                for feature in reference.branches
            ],
            event_trackers=[
                EventTracker(feature, number_draws, grid.shape)
                for feature in reference.events
            ],
            shoulder_trackers=[
                ShoulderTracker(
                    feature,
                    number_draws,
                    feature.center_curve_geometry.shape[0],
                    grid.shape,
                )
                for feature in reference.shoulders
            ],
            plateau_trackers=[
                PlateauTracker(feature, number_draws) for feature in reference.plateaus
            ],
            topology_signatures=[(0, 0, 0, 0, 0, 0)] * number_draws,
        )

    @property
    def maximum_point_distance(self) -> float:
        return max(4.0 * self.grid.typical_spacing, 0.10 * _domain_length(self.grid))

    def ingest(
        self,
        draw: int,
        analysis: FieldAnalysis,
        *,
        draw_weight: float = 1.0,
        discover: bool = True,
    ) -> None:
        candidates_by_kind = {
            kind: [item for item in analysis.points if item.kind == kind]
            for kind in ("peak", "pit")
        }
        for kind in ("peak", "pit"):
            tracker_indices = [
                index
                for index, tracker in enumerate(self.point_trackers)
                if tracker.template.kind == kind
            ]
            trackers = [self.point_trackers[index] for index in tracker_indices]
            candidates = candidates_by_kind[kind]
            template_positions = [
                _point_geometry(item.template, self.grid) for item in trackers
            ]
            candidate_positions = [
                _point_geometry(item, self.grid) for item in candidates
            ]
            matches, unmatched = _assignment(
                template_positions,
                candidate_positions,
                maximum_distance=self.maximum_point_distance,
            )
            for row, column in matches:
                trackers[row].record(
                    draw,
                    candidates[column],
                    self.grid,
                    draw_weight,
                    _match_quality(
                        template_positions,
                        candidate_positions,
                        row,
                        column,
                        resolution=self.grid.typical_spacing,
                    ),
                )
            if discover:
                for column in unmatched:
                    tracker = PointTracker(
                        candidates[column], self.number_draws, self.grid.shape, False
                    )
                    tracker.record(
                        draw, candidates[column], self.grid, draw_weight
                    )
                    self.point_trackers.append(tracker)

        for kind in ("ridge", "valley"):
            trackers = [
                tracker
                for tracker in self.branch_trackers
                if tracker.template.kind == kind
            ]
            candidates = [item for item in analysis.branches if item.kind == kind]
            compatibility = _branch_compatibility(
                [tracker.template for tracker in trackers], candidates
            )
            template_positions = [
                _branch_topology_geometry(item.template) for item in trackers
            ]
            candidate_positions = [
                _branch_topology_geometry(item) for item in candidates
            ]
            matches, unmatched = _assignment(
                template_positions,
                candidate_positions,
                maximum_distance=math.sqrt(2.0) * self.maximum_point_distance,
                compatibility=compatibility,
            )
            for row, column in matches:
                trackers[row].record(
                    draw,
                    candidates[column],
                    draw_weight,
                    _match_quality(
                        template_positions,
                        candidate_positions,
                        row,
                        column,
                        resolution=math.sqrt(2.0) * self.grid.typical_spacing,
                        compatibility=compatibility,
                    ),
                )
            if discover:
                for column in unmatched:
                    tracker = BranchTracker(
                        candidates[column],
                        self.number_draws,
                        candidates[column].points_geometry.shape[0],
                        candidates[column].width_profile.size,
                        self.grid.shape,
                        False,
                    )
                    tracker.record(draw, candidates[column], draw_weight)
                    self.branch_trackers.append(tracker)

        trackers = self.shoulder_trackers
        candidates = analysis.shoulders
        template_positions = [
            _shoulder_topology_geometry(item.template) for item in trackers
        ]
        candidate_positions = [
            _shoulder_topology_geometry(item) for item in candidates
        ]
        matches, _ = _assignment(
            template_positions,
            candidate_positions,
            maximum_distance=math.sqrt(3.0) * self.maximum_point_distance,
        )
        for row, column in matches:
            trackers[row].record(
                draw,
                candidates[column],
                draw_weight,
                _match_quality(
                    template_positions,
                    candidate_positions,
                    row,
                    column,
                    resolution=math.sqrt(3.0) * self.grid.typical_spacing,
                ),
            )

        for kind in ("ridge", "valley"):
            trackers = [
                tracker for tracker in self.event_trackers if tracker.template.kind == kind
            ]
            candidates = [item for item in analysis.events if item.kind == kind]
            template_positions = [
                _event_topology_geometry(item.template) for item in trackers
            ]
            candidate_positions = [
                _event_topology_geometry(item) for item in candidates
            ]
            matches, _ = _assignment(
                template_positions,
                candidate_positions,
                maximum_distance=math.sqrt(3.0) * self.maximum_point_distance,
            )
            for row, column in matches:
                trackers[row].record(
                    draw,
                    candidates[column],
                    draw_weight,
                    _match_quality(
                        template_positions,
                        candidate_positions,
                        row,
                        column,
                        resolution=math.sqrt(3.0) * self.grid.typical_spacing,
                    ),
                )

        for kind in ("plateau", "depression_floor"):
            trackers = [
                tracker
                for tracker in self.plateau_trackers
                if tracker.template.kind == kind
            ]
            candidates = [item for item in analysis.plateaus if item.kind == kind]
            matches, unmatched = _assignment(
                [_plateau_geometry(item.template, self.grid) for item in trackers],
                [_plateau_geometry(item, self.grid) for item in candidates],
                maximum_distance=self.maximum_point_distance,
            )
            for row, column in matches:
                trackers[row].record(draw, candidates[column], self.grid)
            if discover:
                for column in unmatched:
                    tracker = PlateauTracker(
                        candidates[column], self.number_draws, False
                    )
                    tracker.record(draw, candidates[column], self.grid)
                    self.plateau_trackers.append(tracker)

        self.topology_signatures[draw] = (
            sum(item.kind == "peak" for item in analysis.points),
            sum(item.kind == "pit" for item in analysis.points),
            sum(item.kind == "ridge" for item in analysis.branches),
            sum(item.kind == "valley" for item in analysis.branches),
            len(analysis.shoulders),
            len(analysis.plateaus),
        )

    def support(
        self,
        tracker: PointTracker | BranchTracker | EventTracker | ShoulderTracker | PlateauTracker,
        weights: np.ndarray,
    ) -> float:
        return float(np.sum(weights[tracker.present]))

    def retained(
        self, weights: np.ndarray, minimum_support: float
    ) -> tuple[list[PointTracker], list[BranchTracker], list[PlateauTracker]]:
        points = [
            item
            for item in self.point_trackers
            if self.support(item, weights) >= minimum_support
        ]
        branches = [
            item
            for item in self.branch_trackers
            if self.support(item, weights) >= minimum_support
        ]
        plateaus = [
            item
            for item in self.plateau_trackers
            if self.support(item, weights) >= minimum_support
        ]
        if any(
            not item.template.identifier for item in [*points, *branches, *plateaus]
        ):
            self._assign_new_identifiers(points, branches, plateaus)
        return points, branches, plateaus

    def retained_events(
        self, weights: np.ndarray, minimum_support: float
    ) -> list[EventTracker]:
        return [
            item
            for item in self.event_trackers
            if self.support(item, weights) >= minimum_support
        ]

    def retained_shoulders(
        self, weights: np.ndarray, minimum_support: float
    ) -> list[ShoulderTracker]:
        return [
            item
            for item in self.shoulder_trackers
            if self.support(item, weights) >= minimum_support
        ]

    def _assign_new_identifiers(
        self,
        points: list[PointTracker],
        branches: list[BranchTracker],
        plateaus: list[PlateauTracker],
    ) -> None:
        for kind, prefix in (("peak", "P2D"), ("pit", "D2D")):
            members = [item for item in points if item.template.kind == kind]
            members.sort(
                key=lambda item: tuple(_point_geometry(item.template, self.grid))
            )
            for number, tracker in enumerate(members, start=1):
                tracker.template.identifier = f"{prefix}{number}"
        for kind, prefix in (("ridge", "R2D"), ("valley", "V2D")):
            members = [item for item in branches if item.template.kind == kind]
            members.sort(key=lambda item: tuple(item.template.points_geometry[0]))
            for number, tracker in enumerate(members, start=1):
                tracker.template.identifier = f"{prefix}{number}"
        ordered = sorted(
            plateaus,
            key=lambda item: tuple(_plateau_geometry(item.template, self.grid)),
        )
        for number, tracker in enumerate(ordered, start=1):
            tracker.template.identifier = f"L2D{number}"

    def topology_frequencies(
        self, weights: np.ndarray
    ) -> list[tuple[tuple[int, ...], float, int]]:
        counts = Counter(self.topology_signatures)
        weight_sums: Counter[tuple[int, ...]] = Counter()
        for signature, weight in zip(self.topology_signatures, weights):
            weight_sums[signature] += float(weight)
        return [
            (signature, float(weight_sums[signature]), int(count))
            for signature, count in counts.most_common()
        ]


def probability_maps(
    branch_trackers: Iterable[BranchTracker],
    grid: Grid,
    weights: np.ndarray,
    *,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior probabilities of lying within radius of any ridge/valley."""

    trackers = list(branch_trackers)
    ridge = np.zeros(grid.shape, dtype=float)
    valley = np.zeros(grid.shape, dtype=float)
    iterations = max(1, math.ceil(radius / grid.typical_spacing))
    structure = ndimage.generate_binary_structure(2, 2)

    def nearest(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
        right = np.clip(np.searchsorted(coordinates, values), 1, coordinates.size - 1)
        left = right - 1
        choose_left = np.abs(values - coordinates[left]) <= np.abs(
            coordinates[right] - values
        )
        return np.where(choose_left, left, right)

    for draw, draw_weight in enumerate(weights):
        masks = {
            "ridge": np.zeros(grid.shape, dtype=bool),
            "valley": np.zeros(grid.shape, dtype=bool),
        }
        for tracker in trackers:
            if not tracker.present[draw] or not tracker.curve_available[draw]:
                continue
            points = tracker.curves[draw]
            ii = nearest(grid.geometry1, points[:, 0])
            jj = nearest(grid.geometry2, points[:, 1])
            masks[tracker.template.kind][ii, jj] = True
        ridge_mask = (
            ndimage.binary_dilation(
                masks["ridge"], structure=structure, iterations=iterations
            )
            & grid.mask
        )
        valley_mask = (
            ndimage.binary_dilation(
                masks["valley"], structure=structure, iterations=iterations
            )
            & grid.mask
        )
        ridge += float(draw_weight) * ridge_mask
        valley += float(draw_weight) * valley_mask
    return ridge, valley


@dataclass
class FeatureGeometryMaps:
    """Posterior spatial summaries for one registered feature."""

    location_probability: np.ndarray
    location_density: np.ndarray
    location_hpd_50: np.ndarray
    location_hpd_credible: np.ndarray
    region_inclusion_conditional: np.ndarray
    region_inclusion_unconditional: np.ndarray


def _location_hpd_mask(
    probability: np.ndarray,
    density: np.ndarray,
    mask: np.ndarray,
    credible_mass: float,
) -> np.ndarray:
    valid = mask & np.isfinite(density) & (probability > 0.0)
    result = np.zeros(mask.shape, dtype=bool)
    if not np.any(valid):
        return result
    flat = np.flatnonzero(valid)
    order = flat[np.argsort(density.ravel()[flat])[::-1]]
    cumulative = np.cumsum(probability.ravel()[order])
    location = min(
        int(np.searchsorted(cumulative, credible_mass, side="left")),
        order.size - 1,
    )
    result.ravel()[order[: location + 1]] = True
    return result


def _smooth_location_probability(probability: np.ndarray, grid: Grid) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(probability, sigma=0.75, mode="constant")
    smoothed = np.where(grid.mask, smoothed, 0.0)
    total = float(np.sum(smoothed))
    return smoothed / total if total > 0.0 else smoothed


def feature_geometry_maps(
    trackers: Iterable[PointTracker | BranchTracker],
    grid: Grid,
    weights: np.ndarray,
    *,
    credible_mass: float,
) -> dict[str, FeatureGeometryMaps]:
    """Build conditional location HPDs and finite-region inclusion maps."""

    physical_area = np.multiply.outer(
        quadrature_weights(grid.m1), quadrature_weights(grid.m2)
    )
    physical_area = np.maximum(physical_area, np.finfo(float).tiny)

    def nearest(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
        right = np.clip(np.searchsorted(coordinates, values), 1, coordinates.size - 1)
        left = right - 1
        choose_left = np.abs(values - coordinates[left]) <= np.abs(
            coordinates[right] - values
        )
        return np.where(choose_left, left, right)

    result: dict[str, FeatureGeometryMaps] = {}
    for tracker in trackers:
        location = np.zeros(grid.shape, dtype=float)
        if isinstance(tracker, PointTracker):
            selected = tracker.present
            denominator = float(np.sum(weights[selected]))
            if denominator > 0.0:
                positions = tracker.location[selected]
                ii = nearest(grid.geometry1, positions[:, 0])
                jj = nearest(grid.geometry2, positions[:, 1])
                np.add.at(location, (ii, jj), weights[selected] / denominator)
        else:
            selected = tracker.present & tracker.curve_available
            denominator = float(np.sum(weights[selected]))
            if denominator > 0.0:
                for draw in np.flatnonzero(selected):
                    points = tracker.curves[draw]
                    finite = np.all(np.isfinite(points), axis=1)
                    if not np.any(finite):
                        continue
                    points = points[finite]
                    ii = nearest(grid.geometry1, points[:, 0])
                    jj = nearest(grid.geometry2, points[:, 1])
                    contribution = float(weights[draw]) / (
                        denominator * points.shape[0]
                    )
                    np.add.at(location, (ii, jj), contribution)
        location = _smooth_location_probability(location, grid)
        location_density = np.divide(
            location,
            physical_area,
            out=np.zeros_like(location),
            where=grid.mask,
        )
        geometry_event = tracker.present & tracker.geometry_valid
        geometry_support = float(np.sum(weights[geometry_event]))
        unconditional = np.where(grid.mask, tracker.region_weight_sum, 0.0)
        conditional = (
            unconditional / geometry_support
            if geometry_support > 0.0
            else np.zeros(grid.shape, dtype=float)
        )
        result[tracker.template.identifier] = FeatureGeometryMaps(
            location_probability=location.astype(np.float32),
            location_density=location_density.astype(np.float32),
            location_hpd_50=_location_hpd_mask(
                location, location_density, grid.mask, 0.50
            ),
            location_hpd_credible=_location_hpd_mask(
                location, location_density, grid.mask, credible_mass
            ),
            region_inclusion_conditional=conditional.astype(np.float32),
            region_inclusion_unconditional=unconditional.astype(np.float32),
        )
    return result
