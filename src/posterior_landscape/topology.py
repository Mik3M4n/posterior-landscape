"""Single-field topology and ridge/valley geometry on a triangular grid."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded
from scipy.signal import find_peaks

from .io import Grid, normalize_density, quadrature_weights

try:
    import numba as _numba
except ImportError:  # pragma: no cover - the reference implementation remains valid.
    _numba = None


# Six-neighbour connectivity corresponding to a consistently triangulated grid.
_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1))


if _numba is not None:

    @_numba.njit(cache=True)
    def _find_compiled(parent: np.ndarray, item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    @_numba.njit(cache=True)
    def _build_tree_compiled(
        values: np.ndarray,
        mask: np.ndarray,
        order: np.ndarray,
        coordinate1: np.ndarray,
        coordinate2: np.ndarray,
        descending: bool,
    ):
        """Array-only union-find kernel; converted to feature objects in Python."""

        n1, n2 = mask.shape
        size = values.size
        number_valid = order.size
        parent = np.full(size, -1, dtype=np.int64)
        extremum = np.full(size, -1, dtype=np.int64)
        active = np.zeros(size, dtype=np.bool_)
        flow = np.full(size, -1, dtype=np.int64)
        is_extremum = np.zeros(size, dtype=np.bool_)
        extrema_saddle = np.full(size, -1, dtype=np.int64)
        extrema_persistence = np.full(size, np.nan)
        elder = np.full(size, -1, dtype=np.int64)

        event_saddle = np.full(number_valid, -1, dtype=np.int64)
        event_persistence = np.zeros(number_valid)
        event_offsets = np.zeros(number_valid + 1, dtype=np.int64)
        branch_extremum = np.full(6 * number_valid, -1, dtype=np.int64)
        branch_seed = np.full(6 * number_valid, -1, dtype=np.int64)
        branch_prominence = np.zeros(6 * number_valid)
        event_count = 0
        branch_count = 0

        di_values = np.asarray((-1, 1, 0, 0, -1, 1), dtype=np.int64)
        dj_values = np.asarray((0, 0, -1, 1, -1, 1), dtype=np.int64)

        for order_position in range(number_valid):
            vertex = int(order[order_position])
            i = vertex // n2
            j = vertex - i * n2
            active[vertex] = True
            parent[vertex] = vertex
            extremum[vertex] = vertex

            roots = np.empty(6, dtype=np.int64)
            seeds = np.empty(6, dtype=np.int64)
            root_count = 0
            flow_seed = -1
            flow_score = -np.inf if descending else np.inf

            for offset_index in range(6):
                ni = i + di_values[offset_index]
                nj = j + dj_values[offset_index]
                if ni < 0 or ni >= n1 or nj < 0 or nj >= n2:
                    continue
                neighbour = ni * n2 + nj
                if not mask[ni, nj] or not active[neighbour]:
                    continue
                root = _find_compiled(parent, neighbour)
                distance = math.hypot(
                    coordinate1[ni] - coordinate1[i],
                    coordinate2[nj] - coordinate2[j],
                )
                score = (values[neighbour] - values[vertex]) / distance
                better_flow = (
                    score > flow_score
                    or (
                        score == flow_score and (flow_seed < 0 or neighbour < flow_seed)
                    )
                    if descending
                    else score < flow_score
                    or (
                        score == flow_score and (flow_seed < 0 or neighbour < flow_seed)
                    )
                )
                if better_flow:
                    flow_score = score
                    flow_seed = neighbour

                existing = -1
                for candidate_index in range(root_count):
                    if roots[candidate_index] == root:
                        existing = candidate_index
                        break
                if existing < 0:
                    roots[root_count] = root
                    seeds[root_count] = neighbour
                    root_count += 1
                else:
                    old_seed = seeds[existing]
                    old_i = old_seed // n2
                    old_j = old_seed - old_i * n2
                    old_distance = math.hypot(
                        coordinate1[old_i] - coordinate1[i],
                        coordinate2[old_j] - coordinate2[j],
                    )
                    old_slope = abs(values[old_seed] - values[vertex]) / old_distance
                    new_slope = abs(values[neighbour] - values[vertex]) / distance
                    if new_slope > old_slope or (
                        new_slope == old_slope and neighbour < old_seed
                    ):
                        seeds[existing] = neighbour

            if root_count == 0:
                is_extremum[vertex] = True
                extrema_persistence[vertex] = np.inf
                continue
            flow[vertex] = flow_seed

            # Insertion-sort the at-most-six components by elder-rule priority.
            for candidate_index in range(1, root_count):
                candidate_root = roots[candidate_index]
                candidate_seed = seeds[candidate_index]
                candidate_extremum = extremum[candidate_root]
                location = candidate_index
                while location > 0:
                    previous_root = roots[location - 1]
                    previous_extremum = extremum[previous_root]
                    candidate_value = values[candidate_extremum]
                    previous_value = values[previous_extremum]
                    before = (
                        candidate_value > previous_value
                        or (
                            candidate_value == previous_value
                            and candidate_extremum < previous_extremum
                        )
                        if descending
                        else candidate_value < previous_value
                        or (
                            candidate_value == previous_value
                            and candidate_extremum < previous_extremum
                        )
                    )
                    if not before:
                        break
                    roots[location] = previous_root
                    seeds[location] = seeds[location - 1]
                    location -= 1
                roots[location] = candidate_root
                seeds[location] = candidate_seed

            survivor_root = roots[0]
            survivor_extremum = extremum[survivor_root]
            if root_count > 1:
                event_saddle[event_count] = vertex
                event_offsets[event_count] = branch_count
                maximum_loser_persistence = 0.0
                for root_index in range(root_count):
                    root = roots[root_index]
                    component_extremum = extremum[root]
                    prominence = (
                        values[component_extremum] - values[vertex]
                        if descending
                        else values[vertex] - values[component_extremum]
                    )
                    prominence = max(prominence, 0.0)
                    branch_extremum[branch_count] = component_extremum
                    branch_seed[branch_count] = seeds[root_index]
                    branch_prominence[branch_count] = prominence
                    branch_count += 1
                    if root_index > 0:
                        extrema_saddle[component_extremum] = vertex
                        extrema_persistence[component_extremum] = prominence
                        elder[component_extremum] = survivor_extremum
                        maximum_loser_persistence = max(
                            maximum_loser_persistence, prominence
                        )
                event_persistence[event_count] = maximum_loser_persistence
                event_count += 1
                event_offsets[event_count] = branch_count

            parent[vertex] = survivor_root
            for root_index in range(1, root_count):
                parent[roots[root_index]] = survivor_root
            extremum[survivor_root] = survivor_extremum

        return (
            flow,
            is_extremum,
            extrema_saddle,
            extrema_persistence,
            elder,
            event_saddle[:event_count],
            event_persistence[:event_count],
            event_offsets[: event_count + 1],
            branch_extremum[:branch_count],
            branch_seed[:branch_count],
            branch_prominence[:branch_count],
        )


@dataclass
class PointFeature:
    kind: str
    index: tuple[int, int]
    value: float
    persistence: float
    boundary: bool
    region_mass: float = math.nan
    saddle_index: tuple[int, int] | None = None
    role: str | None = None
    scale_count: int = 1
    scale_total: int = 1
    identifier: str = ""
    boundary_type: str = "none"
    base_level: float = math.nan
    half_level: float = math.nan
    feature_probability: float = math.nan
    width_major: float = math.nan
    width_minor: float = math.nan
    width_orientation: float = math.nan
    width_area: float = math.nan
    log_contrast: float = math.nan
    relative_prominence: float = math.nan
    geometry_valid: bool = False
    measurement_valid: bool = False
    width_reference: str = "unavailable"
    mass_centroid: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan), repr=False
    )
    spatial_covariance_mass: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan), repr=False
    )
    projected_bounds_mass: np.ndarray = field(
        default_factory=lambda: np.full(4, np.nan), repr=False
    )
    relative_projected_widths: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan), repr=False
    )
    deficit_probability: float = math.nan
    projection_m1: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    projection_m2: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    feature_region_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class BranchFeature:
    kind: str
    saddle_index: tuple[int, int]
    extremum_index: tuple[int, int]
    points_geometry: np.ndarray
    points_mass: np.ndarray
    prominence: float
    event_persistence: float
    length: float
    boundary: bool
    topology_points_geometry: np.ndarray = field(repr=False)
    full_length: float = math.nan
    retained_fraction: float = 1.0
    low_density_truncated: bool = False
    curve_available: bool = True
    boundary_type: str = "none"
    saddle_boundary_type: str = "none"
    extremum_boundary_type: str = "none"
    identifier: str = ""
    start_identifier: str = ""
    end_identifier: str = ""
    scale_count: int = 1
    scale_total: int = 1
    feature_probability: float = math.nan
    width_left: float = math.nan
    width_right: float = math.nan
    width_median: float = math.nan
    width_along_lower: float = math.nan
    width_along_upper: float = math.nan
    valid_width_fraction: float = 0.0
    fallback_width_fraction: float = 0.0
    log_contrast: float = math.nan
    geometry_valid: bool = False
    width_sample_fraction: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    left_width_profile: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    right_width_profile: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    width_profile: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    log_contrast_profile: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    width_center_geometry: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    half_left_geometry: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    half_right_geometry: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    shoulder_left_geometry: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    shoulder_right_geometry: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    shoulder_levels: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    section_valid: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool), repr=False
    )
    section_fallback: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool), repr=False
    )
    background_density: np.ndarray | None = field(default=None, repr=False)
    feature_region_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class CurveEventFeature:
    """Two branches born at one saddle, treated as one scientific feature."""

    kind: str
    saddle_index: tuple[int, int]
    arm_indices: tuple[int, int]
    topology_geometry: np.ndarray = field(repr=False)
    curve_geometry: np.ndarray = field(repr=False)
    curve_available: bool = False
    region_valid: bool = False
    identifier: str = ""
    arm_identifiers: tuple[str, str] = ("", "")
    boundary: bool = False
    low_density_truncated: bool = False
    region_probability: float = math.nan
    deficit_probability: float = math.nan
    mass_centroid: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan), repr=False
    )
    spatial_covariance_mass: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan), repr=False
    )
    projected_bounds_mass: np.ndarray = field(
        default_factory=lambda: np.full(4, np.nan), repr=False
    )
    relative_projected_widths: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan), repr=False
    )
    projection_m1: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    projection_m2: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    extent: float = math.nan
    full_extent: float = math.nan
    retained_fraction: float = math.nan
    bounded_fraction: float = 0.0
    longest_bounded_fraction: float = 0.0
    width_median: float = math.nan
    fallback_fraction: float = 0.0
    log_contrast: float = math.nan
    relative_contrast: float = math.nan
    window_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class ShoulderFeature:
    """Directional change-of-slope front and its draw-specific transition band."""

    kind: str
    topology_geometry: np.ndarray = field(repr=False)
    center_curve_geometry: np.ndarray = field(repr=False)
    onset_curve_geometry: np.ndarray = field(repr=False)
    end_curve_geometry: np.ndarray = field(repr=False)
    alpha_threshold: float = 2.0
    curve_available: bool = True
    region_valid: bool = False
    identifier: str = ""
    scale_count: int = 1
    scale_total: int = 1
    boundary: bool = False
    low_density_truncated: bool = False
    boundary_type: str = "none"
    alpha_max: float = math.nan
    slope_pre: float = math.nan
    slope_post: float = math.nan
    slope_contrast: float = math.nan
    region_probability: float = math.nan
    mass_centroid: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan), repr=False
    )
    spatial_covariance_mass: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan), repr=False
    )
    projected_bounds_mass: np.ndarray = field(
        default_factory=lambda: np.full(4, np.nan), repr=False
    )
    relative_projected_widths: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan), repr=False
    )
    projection_m1: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    projection_m2: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    extent: float = math.nan
    full_extent: float = math.nan
    retained_fraction: float = math.nan
    bounded_fraction: float = 0.0
    longest_bounded_fraction: float = 0.0
    width_median: float = math.nan
    window_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class PlateauFeature:
    kind: str
    center_index: tuple[int, int]
    center_mass: tuple[float, float]
    area: float
    probability_mass: float
    contrast: float
    boundary: bool
    region_mask: np.ndarray = field(repr=False)
    identifier: str = ""


@dataclass
class FieldAnalysis:
    density: np.ndarray = field(repr=False)
    scale: float
    persistence_threshold: float
    persistence_threshold_diagnostics: "PersistenceThresholdDiagnostics"
    points: list[PointFeature]
    branches: list[BranchFeature]
    events: list[CurveEventFeature]
    shoulders: list[ShoulderFeature]
    plateaus: list[PlateauFeature]
    peak_labels: np.ndarray = field(repr=False)
    pit_labels: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class PersistenceThresholdDiagnostics:
    """Auditable ingredients of one automatic persistence selection."""

    selection_mode: str
    threshold: float
    numerical_floor: float
    gap_min_log: float
    terminal_gap_log: float | None
    selected_gap_log: float | None
    selected_gap_lower: float | None
    selected_gap_upper: float | None
    gap_resolved: bool
    fallback_used: bool

    def as_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "selection_mode": self.selection_mode,
            "threshold": self.threshold,
            "numerical_floor": self.numerical_floor,
            "gap_min_log": self.gap_min_log,
            "terminal_gap_log": self.terminal_gap_log,
            "selected_gap_log": self.selected_gap_log,
            "selected_gap_lower": self.selected_gap_lower,
            "selected_gap_upper": self.selected_gap_upper,
            "gap_resolved": self.gap_resolved,
            "fallback_used": self.fallback_used,
        }


@dataclass
class _TreeResult:
    extrema: dict[int, dict[str, float | int | None]]
    events: list[dict[str, object]]
    flow: np.ndarray
    elder: dict[int, int]
    global_extremum: int
    order: np.ndarray = field(repr=False)


def _structure() -> np.ndarray:
    result = np.zeros((3, 3), dtype=bool)
    result[1, 1] = True
    for di, dj in _OFFSETS:
        result[di + 1, dj + 1] = True
    return result


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & ~ndimage.binary_erosion(
        mask, structure=_structure(), border_value=0
    )


def _boundary_type(
    index: tuple[int, int], grid: Grid, boundary: np.ndarray
) -> str:
    """Classify a critical point as interior, mask-edge, or array-edge."""

    if not boundary[index]:
        return "none"
    i, j = index
    touches_outer = i in {0, grid.shape[0] - 1} or j in {0, grid.shape[1] - 1}
    touches_mask = False
    for di, dj in _OFFSETS:
        ni, nj = i + di, j + dj
        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
            touches_mask |= not bool(grid.mask[ni, nj])
    if touches_outer and touches_mask:
        return "outer+mask"
    if touches_outer:
        return "outer"
    return "mask"


def _flat(index: tuple[int, int], shape: tuple[int, int]) -> int:
    return int(np.ravel_multi_index(index, shape))


def _index(flat_index: int, shape: tuple[int, int]) -> tuple[int, int]:
    result = np.unravel_index(int(flat_index), shape)
    return int(result[0]), int(result[1])


def _neighbours(flat_index: int, mask: np.ndarray) -> Iterable[int]:
    shape = mask.shape
    i, j = _index(flat_index, shape)
    for di, dj in _OFFSETS:
        ni, nj = i + di, j + dj
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and mask[ni, nj]:
            yield _flat((ni, nj), shape)


def _distance(
    first: int,
    second: int,
    shape: tuple[int, int],
    coordinate1: np.ndarray,
    coordinate2: np.ndarray,
) -> float:
    i1, j1 = _index(first, shape)
    i2, j2 = _index(second, shape)
    return float(
        np.hypot(coordinate1[i1] - coordinate1[i2], coordinate2[j1] - coordinate2[j2])
    )


def _build_tree(field: np.ndarray, grid: Grid, *, descending: bool) -> _TreeResult:
    """Build a zero-dimensional merge tree and a monotone flow forest."""

    shape = field.shape
    size = field.size
    values = np.asarray(field, dtype=float).ravel()
    valid = np.flatnonzero(grid.mask.ravel())
    order = valid[np.lexsort((valid, -values[valid] if descending else values[valid]))]

    if _numba is not None:
        (
            flow,
            extrema_flags,
            extrema_saddles,
            extrema_persistences,
            elder_array,
            event_saddles,
            event_persistences,
            event_offsets,
            branch_extrema,
            branch_seeds,
            branch_prominences,
        ) = _build_tree_compiled(  # type: ignore[name-defined]
            values,
            np.asarray(grid.mask, dtype=bool),
            np.asarray(order, dtype=np.int64),
            np.asarray(grid.geometry1, dtype=float),
            np.asarray(grid.geometry2, dtype=float),
            descending,
        )
        extrema: dict[int, dict[str, float | int | None]] = {}
        for flat_index in np.flatnonzero(extrema_flags):
            flat_index = int(flat_index)
            saddle = int(extrema_saddles[flat_index])
            extrema[flat_index] = {
                "birth": float(values[flat_index]),
                "saddle": saddle if saddle >= 0 else None,
                "persistence": float(extrema_persistences[flat_index]),
            }
        events: list[dict[str, object]] = []
        for event_index, saddle in enumerate(event_saddles):
            start = int(event_offsets[event_index])
            stop = int(event_offsets[event_index + 1])
            branches = [
                {
                    "extremum": int(branch_extrema[index]),
                    "seed": int(branch_seeds[index]),
                    "prominence": float(branch_prominences[index]),
                }
                for index in range(start, stop)
            ]
            events.append(
                {
                    "saddle": int(saddle),
                    "branches": branches,
                    "persistence": float(event_persistences[event_index]),
                }
            )
        global_extremum = int(order[0])
        dynamic_range = float(np.nanmax(values[valid]) - np.nanmin(values[valid]))
        extrema[global_extremum]["persistence"] = dynamic_range
        elder = {
            int(index): int(elder_array[index])
            for index in np.flatnonzero(elder_array >= 0)
        }
        return _TreeResult(
            extrema=extrema,
            events=events,
            flow=flow,
            elder=elder,
            global_extremum=global_extremum,
            order=order,
        )

    parent = np.full(size, -1, dtype=np.int64)
    extremum = np.full(size, -1, dtype=np.int64)
    active = np.zeros(size, dtype=bool)
    flow = np.full(size, -1, dtype=np.int64)
    extrema: dict[int, dict[str, float | int | None]] = {}
    elder: dict[int, int] = {}
    events: list[dict[str, object]] = []

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def birth_priority(root: int) -> tuple[float, int]:
        representative = int(extremum[root])
        value = values[representative]
        return ((-value if descending else value), representative)

    for vertex in order:
        vertex = int(vertex)
        active[vertex] = True
        parent[vertex] = vertex
        extremum[vertex] = vertex

        root_to_seed: dict[int, int] = {}
        for neighbour in _neighbours(vertex, grid.mask):
            if active[neighbour]:
                root = find(neighbour)
                candidate = root_to_seed.get(root)
                if candidate is None:
                    root_to_seed[root] = neighbour
                else:
                    old_distance = _distance(
                        vertex,
                        candidate,
                        shape,
                        grid.geometry1,
                        grid.geometry2,
                    )
                    new_distance = _distance(
                        vertex,
                        neighbour,
                        shape,
                        grid.geometry1,
                        grid.geometry2,
                    )
                    old_slope = abs(values[candidate] - values[vertex]) / old_distance
                    new_slope = abs(values[neighbour] - values[vertex]) / new_distance
                    if new_slope > old_slope:
                        root_to_seed[root] = neighbour

        roots = sorted(root_to_seed, key=birth_priority)
        if not roots:
            extrema[vertex] = {
                "birth": float(values[vertex]),
                "saddle": None,
                "persistence": math.inf,
            }
            continue

        # A flow pointer always enters one already-processed component.
        candidate_seeds = list(root_to_seed.values())
        if descending:
            flow[vertex] = max(
                candidate_seeds,
                key=lambda item: (
                    (values[item] - values[vertex])
                    / _distance(vertex, item, shape, grid.geometry1, grid.geometry2),
                    -item,
                ),
            )
        else:
            flow[vertex] = min(
                candidate_seeds,
                key=lambda item: (
                    (values[item] - values[vertex])
                    / _distance(vertex, item, shape, grid.geometry1, grid.geometry2),
                    item,
                ),
            )

        survivor_root = roots[0]
        survivor_extremum = int(extremum[survivor_root])
        event_persistences: list[float] = []
        if len(roots) > 1:
            branches: list[dict[str, int | float]] = []
            for root in roots:
                component_extremum = int(extremum[root])
                prominence = (
                    values[component_extremum] - values[vertex]
                    if descending
                    else values[vertex] - values[component_extremum]
                )
                branches.append(
                    {
                        "extremum": component_extremum,
                        "seed": int(root_to_seed[root]),
                        "prominence": float(max(0.0, prominence)),
                    }
                )
                if root != survivor_root:
                    extrema[component_extremum]["saddle"] = vertex
                    extrema[component_extremum]["persistence"] = float(
                        max(0.0, prominence)
                    )
                    elder[component_extremum] = survivor_extremum
                    event_persistences.append(float(max(0.0, prominence)))
            events.append(
                {
                    "saddle": vertex,
                    "branches": branches,
                    "persistence": max(event_persistences, default=0.0),
                }
            )

        parent[vertex] = survivor_root
        for root in roots[1:]:
            parent[root] = survivor_root
        extremum[survivor_root] = survivor_extremum

    if not extrema:
        raise ValueError("No valid extrema were found.")
    global_extremum = min(
        extrema,
        key=lambda item: ((-values[item] if descending else values[item]), item),
    )
    dynamic_range = float(np.nanmax(values[valid]) - np.nanmin(values[valid]))
    extrema[global_extremum]["persistence"] = dynamic_range
    return _TreeResult(
        extrema=extrema,
        events=events,
        flow=flow,
        elder=elder,
        global_extremum=global_extremum,
        order=order,
    )


def _persistence_threshold_diagnostics(
    persistences: Iterable[float],
    dynamic_range: float,
    *,
    persistence_gap_min_log: float = 1.0,
    persistence_threshold: str | float = "auto",
) -> PersistenceThresholdDiagnostics:
    """Resolve the pooled persistence threshold used by the v0.8 catalogue.

    The strongest finite nonessential feature cannot define the catalogue
    resolution by itself: the terminal gap, which would leave only that
    feature above threshold, is excluded from automatic calibration.
    """

    gap_min_log = float(persistence_gap_min_log)
    if not np.isfinite(gap_min_log) or gap_min_log <= 0.0:
        raise ValueError("persistence_gap_min_log must be finite and positive.")
    if not np.isfinite(dynamic_range) or dynamic_range < 0.0:
        raise ValueError("dynamic_range must be finite and nonnegative.")

    values = np.asarray(
        [value for value in persistences if np.isfinite(value) and value > 0.0],
        dtype=float,
    )
    values.sort()
    floor = float(
        max(np.finfo(float).eps * max(dynamic_range, 1.0), dynamic_range * 1e-4)
    )
    positive = values[values > floor]
    terminal_gap_log = (
        float(np.log(positive[-1]) - np.log(positive[-2]))
        if positive.size >= 2
        else None
    )

    if not isinstance(persistence_threshold, str):
        threshold = float(persistence_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("persistence_threshold must be finite and nonnegative.")
        return PersistenceThresholdDiagnostics(
            selection_mode="explicit",
            threshold=threshold,
            numerical_floor=floor,
            gap_min_log=gap_min_log,
            terminal_gap_log=terminal_gap_log,
            selected_gap_log=None,
            selected_gap_lower=None,
            selected_gap_upper=None,
            gap_resolved=False,
            fallback_used=False,
        )
    if persistence_threshold.strip().lower() != "auto":
        raise ValueError("persistence_threshold must be 'auto' or nonnegative.")

    fallback = floor
    if values.size >= 3 and positive.size >= 3:
        fallback = float(max(floor, np.quantile(positive, 0.10) * 0.25))
        # The final adjacent gap is deliberately omitted: selecting it would
        # leave only one finite nonessential feature above the threshold.
        log_gaps = np.log(positive[1:-1]) - np.log(positive[:-2])
        location = int(np.argmax(log_gaps))
        selected_gap = float(log_gaps[location])
        if selected_gap >= gap_min_log:
            lower = float(positive[location])
            upper = float(positive[location + 1])
            threshold = float(np.sqrt(lower * upper))
            return PersistenceThresholdDiagnostics(
                selection_mode="automatic_gap",
                threshold=threshold,
                numerical_floor=floor,
                gap_min_log=gap_min_log,
                terminal_gap_log=terminal_gap_log,
                selected_gap_log=selected_gap,
                selected_gap_lower=lower,
                selected_gap_upper=upper,
                gap_resolved=True,
                fallback_used=False,
            )

    return PersistenceThresholdDiagnostics(
        selection_mode="automatic_fallback",
        threshold=fallback,
        numerical_floor=floor,
        gap_min_log=gap_min_log,
        terminal_gap_log=terminal_gap_log,
        selected_gap_log=None,
        selected_gap_lower=None,
        selected_gap_upper=None,
        gap_resolved=False,
        fallback_used=True,
    )


def automatic_persistence_threshold(
    persistences: Iterable[float],
    dynamic_range: float,
    *,
    persistence_gap_min_log: float = 1.0,
) -> float:
    """Choose the automatic threshold from eligible nonterminal log gaps."""

    return _persistence_threshold_diagnostics(
        persistences,
        dynamic_range,
        persistence_gap_min_log=persistence_gap_min_log,
    ).threshold


def _finite_nonessential_persistences(*trees: _TreeResult) -> list[float]:
    """Return finite-pair persistences without either global extremum."""

    return [
        float(item["persistence"] or 0.0)
        for tree in trees
        for index, item in tree.extrema.items()
        if index != tree.global_extremum
    ]


def _surviving_extrema(tree: _TreeResult, threshold: float) -> set[int]:
    return {
        index
        for index, information in tree.extrema.items()
        if index == tree.global_extremum
        or float(information["persistence"] or 0.0) >= threshold
    }


def _representative(index: int, tree: _TreeResult, surviving: set[int]) -> int:
    seen: set[int] = set()
    while index not in surviving and index in tree.elder and index not in seen:
        seen.add(index)
        index = tree.elder[index]
    return index


def _flow_labels(tree: _TreeResult, grid: Grid, surviving: set[int]) -> np.ndarray:
    labels = np.full(grid.mask.size, -1, dtype=np.int64)
    valid = np.flatnonzero(grid.mask.ravel())
    for start in valid:
        current = int(start)
        trail: list[int] = []
        seen: set[int] = set()
        while tree.flow[current] >= 0 and current not in seen:
            if labels[current] >= 0:
                current = int(labels[current])
                break
            seen.add(current)
            trail.append(current)
            current = int(tree.flow[current])
        if labels[current] >= 0:
            endpoint = int(labels[current])
        else:
            endpoint = _representative(current, tree, surviving)
        labels[current] = endpoint
        for item in trail:
            labels[item] = endpoint
    return labels.reshape(grid.shape)


def _trace(flow: np.ndarray, seed: int, target: int, size: int) -> list[int]:
    result = [int(seed)]
    current = int(seed)
    seen = {current}
    while current != target and flow[current] >= 0 and len(result) <= size:
        current = int(flow[current])
        if current in seen:
            break
        seen.add(current)
        result.append(current)
    if result[-1] != target:
        result.append(int(target))
    return result


def _resample_curve(points: np.ndarray, number: int) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.repeat(points[:1], number, axis=0)
    keep = np.concatenate(
        [[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.0]
    )
    points = points[keep]
    if points.shape[0] <= 1:
        return np.repeat(points[:1], number, axis=0)
    distance = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    targets = np.linspace(0.0, distance[-1], number)
    return np.column_stack(
        [np.interp(targets, distance, points[:, axis]) for axis in range(2)]
    )


def _nearest_coordinate_indices(
    coordinates: np.ndarray, values: np.ndarray
) -> np.ndarray:
    right = np.clip(np.searchsorted(coordinates, values), 1, coordinates.size - 1)
    left = right - 1
    choose_left = np.abs(values - coordinates[left]) <= np.abs(
        coordinates[right] - values
    )
    return np.where(choose_left, left, right)


def _inside_grid_mask(points: np.ndarray, grid: Grid) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    inside_bounds = (
        (points[:, 0] >= grid.geometry1[0])
        & (points[:, 0] <= grid.geometry1[-1])
        & (points[:, 1] >= grid.geometry2[0])
        & (points[:, 1] <= grid.geometry2[-1])
    )
    result = np.zeros(points.shape[0], dtype=bool)
    if np.any(inside_bounds):
        subset = points[inside_bounds]
        ii = _nearest_coordinate_indices(grid.geometry1, subset[:, 0])
        jj = _nearest_coordinate_indices(grid.geometry2, subset[:, 1])
        result[inside_bounds] = grid.mask[ii, jj]
    return result


def _smooth_curve(
    points: np.ndarray, grid: Grid, smoothing_cells: float
) -> np.ndarray:
    """Penalized, endpoint-constrained smoothing in analysis coordinates.

    The penalty is on discrete curvature. Its characteristic length is set in
    grid-cell units, so changing ``curve_points`` does not change the physical
    amount of smoothing.
    """

    points = np.asarray(points, dtype=float)
    number = points.shape[0]
    if smoothing_cells <= 0.0 or number < 5:
        return points.copy()
    full_length = float(
        np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    )
    if full_length <= 0.0:
        return points.copy()
    sample_spacing = full_length / (number - 1)
    smoothing_length = smoothing_cells * grid.typical_spacing
    smoothing_samples = smoothing_length / max(sample_spacing, np.finfo(float).eps)
    if smoothing_samples < 0.25:
        return points.copy()

    # Minimize ||q-p||^2 + lambda ||D2 q||^2 with q[0] and q[-1] fixed.
    regularization = float(smoothing_samples**4)
    unknown = number - 2
    diagonal = np.full(unknown, 1.0 + 6.0 * regularization)
    diagonal[[0, -1]] = 1.0 + 5.0 * regularization
    first = np.full(max(0, unknown - 1), -4.0 * regularization)
    second = np.full(max(0, unknown - 2), regularization)
    banded = np.zeros((5, unknown), dtype=float)
    banded[2] = diagonal
    if unknown > 1:
        banded[1, 1:] = first
        banded[3, :-1] = first
    if unknown > 2:
        banded[0, 2:] = second
        banded[4, :-2] = second

    right_hand = points[1:-1].copy()
    right_hand[0] += 2.0 * regularization * points[0]
    if unknown > 1:
        right_hand[1] -= regularization * points[0]
        right_hand[-2] -= regularization * points[-1]
    right_hand[-1] += 2.0 * regularization * points[-1]

    result = points.copy()
    result[1:-1] = solve_banded((2, 2), banded, right_hand)
    # Convex triangular domains remain valid automatically. For an arbitrary
    # non-convex input mask, retain original samples wherever smoothing exits.
    invalid = ~_inside_grid_mask(result, grid)
    invalid[[0, -1]] = False
    result[invalid] = points[invalid]
    return result


def _sample_field(
    field: np.ndarray, points: np.ndarray, grid: Grid
) -> np.ndarray:
    """Bilinearly interpolate a gridded field at analysis-coordinate points."""

    points = np.asarray(points, dtype=float)
    x = np.clip(points[:, 0], grid.geometry1[0], grid.geometry1[-1])
    y = np.clip(points[:, 1], grid.geometry2[0], grid.geometry2[-1])
    i1 = np.clip(np.searchsorted(grid.geometry1, x), 1, grid.geometry1.size - 1)
    j1 = np.clip(np.searchsorted(grid.geometry2, y), 1, grid.geometry2.size - 1)
    i0 = i1 - 1
    j0 = j1 - 1
    dx = grid.geometry1[i1] - grid.geometry1[i0]
    dy = grid.geometry2[j1] - grid.geometry2[j0]
    tx = np.divide(x - grid.geometry1[i0], dx, out=np.zeros_like(x), where=dx > 0)
    ty = np.divide(y - grid.geometry2[j0], dy, out=np.zeros_like(y), where=dy > 0)
    weights = (
        (1.0 - tx) * (1.0 - ty),
        tx * (1.0 - ty),
        (1.0 - tx) * ty,
        tx * ty,
    )
    indices = ((i0, j0), (i1, j0), (i0, j1), (i1, j1))
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    for weight, (ii, jj) in zip(weights, indices):
        valid = grid.mask[ii, jj]
        numerator += weight * field[ii, jj] * valid
        denominator += weight * valid
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-12,
    )


def _truncate_valley_to_support(
    points: np.ndarray,
    field: np.ndarray,
    threshold: float,
    grid: Grid,
    number: int,
) -> tuple[np.ndarray, float, bool, bool]:
    """Keep the contiguous HPD-supported valley segment from its saddle.

    Returns the fixed-size visible curve, retained/full length fraction,
    whether truncation occurred, and whether at least one grid cell of curve
    remains. The full topological curve is left untouched elsewhere.
    """

    values = _sample_field(field, points, grid)
    inside = (values >= threshold) & _inside_grid_mask(points, grid)
    full_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if not inside[0] or full_length <= 0.0:
        return np.repeat(points[:1], number, axis=0), 0.0, True, False

    outside = np.flatnonzero(~inside)
    if outside.size == 0:
        return _resample_curve(points, number), 1.0, False, True

    stop = int(outside[0])
    if stop == 0:
        return np.repeat(points[:1], number, axis=0), 0.0, True, False
    retained = points[:stop].copy()
    # Locate the last supported point by bisection. Density is bilinear inside
    # grid cells, so a linear interpolation of endpoint values can otherwise
    # put the nominal crossing slightly outside the HPD region.
    supported_point = points[stop - 1].copy()
    unsupported_point = points[stop].copy()
    for _ in range(32):
        midpoint = 0.5 * (supported_point + unsupported_point)
        midpoint_supported = bool(_inside_grid_mask(midpoint[None, :], grid)[0])
        midpoint_value = float(_sample_field(field, midpoint[None, :], grid)[0])
        if midpoint_supported and midpoint_value >= threshold:
            supported_point = midpoint
        else:
            unsupported_point = midpoint
    crossing = supported_point
    retained = np.vstack([retained, crossing])
    retained_length = float(
        np.sum(np.linalg.norm(np.diff(retained, axis=0), axis=1))
    )
    retained_fraction = retained_length / full_length
    available = retained_length >= grid.typical_spacing
    if not available:
        return np.repeat(points[:1], number, axis=0), retained_fraction, True, False
    return (
        _resample_curve(retained, number),
        retained_fraction,
        True,
        True,
    )


def _derivatives(
    field: np.ndarray, grid: Grid
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g1, g2 = np.gradient(field, grid.geometry1, grid.geometry2, edge_order=2)
    h11, h12a = np.gradient(g1, grid.geometry1, grid.geometry2, edge_order=2)
    h12b, h22 = np.gradient(g2, grid.geometry1, grid.geometry2, edge_order=2)
    return g1, g2, h11, 0.5 * (h12a + h12b), h22


def _refine_curve(
    points: np.ndarray,
    kind: str,
    derivatives: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    grid: Grid,
) -> np.ndarray:
    """One bounded Newton snap toward a transverse Hessian ridge/valley."""

    if points.shape[0] < 3:
        return points
    g1, g2, h11, h12, h22 = derivatives
    result = points.copy()
    clamp = 0.75 * grid.typical_spacing
    for position in range(1, points.shape[0] - 1):
        x, y = points[position]
        i = int(np.clip(np.searchsorted(grid.geometry1, x), 1, len(grid.geometry1) - 1))
        j = int(np.clip(np.searchsorted(grid.geometry2, y), 1, len(grid.geometry2) - 1))
        i -= int(abs(grid.geometry1[i - 1] - x) < abs(grid.geometry1[i] - x))
        j -= int(abs(grid.geometry2[j - 1] - y) < abs(grid.geometry2[j] - y))
        hessian = np.asarray([[h11[i, j], h12[i, j]], [h12[i, j], h22[i, j]]])
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        normal_index = 0 if kind == "ridge" else 1
        curvature = float(eigenvalues[normal_index])
        if (kind == "ridge" and curvature >= 0.0) or (
            kind == "valley" and curvature <= 0.0
        ):
            continue
        normal = eigenvectors[:, normal_index]
        gradient = np.asarray([g1[i, j], g2[i, j]])
        displacement = -float(gradient @ normal) / curvature
        displacement = float(np.clip(displacement, -clamp, clamp))
        candidate = result[position] + displacement * normal
        ci = int(
            np.clip(
                np.searchsorted(grid.geometry1, candidate[0]),
                0,
                len(grid.geometry1) - 1,
            )
        )
        cj = int(
            np.clip(
                np.searchsorted(grid.geometry2, candidate[1]),
                0,
                len(grid.geometry2) - 1,
            )
        )
        if grid.mask[ci, cj]:
            result[position] = candidate
    return result


def _geometry_to_mass(points: np.ndarray, grid: Grid) -> np.ndarray:
    if grid.geometry == "log":
        return np.power(grid.log_base, points)
    return points.copy()


def _mass_to_geometry(points: np.ndarray, grid: Grid) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if grid.geometry == "log":
        return np.log(points) / np.log(grid.log_base)
    return points.copy()


def smooth_density(density: np.ndarray, grid: Grid, scale: float) -> np.ndarray:
    if scale <= 0.0:
        return normalize_density(density, grid)[0]
    spacing1 = float(np.median(np.diff(grid.geometry1)))
    spacing2 = float(np.median(np.diff(grid.geometry2)))
    sigma = (scale / spacing1, scale / spacing2)
    numerator = ndimage.gaussian_filter(
        np.where(grid.mask, density, 0.0), sigma=sigma, mode="nearest"
    )
    denominator = ndimage.gaussian_filter(
        grid.mask.astype(float), sigma=sigma, mode="nearest"
    )
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    return normalize_density(smoothed, grid)[0]


def hpd_threshold(density: np.ndarray, probability: float, grid: Grid) -> float:
    values = density[grid.mask]
    weights = (density * grid.cell_weights)[grid.mask]
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(weights[order])
    location = int(np.searchsorted(cumulative, probability, side="left"))
    location = min(location, len(order) - 1)
    return float(values[order[location]])


def _hpd_threshold_from_descending_order(
    density: np.ndarray, probability: float, grid: Grid, order: np.ndarray
) -> float:
    """HPD threshold reusing the join tree's already-sorted vertices."""

    flattened = np.asarray(density, dtype=float).ravel()
    probability_weights = (density * grid.cell_weights).ravel()[order]
    cumulative = np.cumsum(probability_weights)
    location = int(np.searchsorted(cumulative, probability, side="left"))
    location = min(location, order.size - 1)
    return float(flattened[int(order[location])])


def _adjacent_saddle_level(
    flat_index: int,
    tree: _TreeResult,
    field: np.ndarray,
    *,
    descending: bool,
) -> float | None:
    """Closest merge level involving an essential/global extremum."""

    levels: list[float] = []
    for event in tree.events:
        branches = event["branches"]
        has_extremum = any(
            int(item["extremum"]) == flat_index for item in branches  # type: ignore[union-attr]
        )
        if has_extremum:
            saddle = _index(int(event["saddle"]), field.shape)
            levels.append(float(field[saddle]))
    if not levels:
        return None
    return max(levels) if descending else min(levels)


def _region_measurements(
    weight_density: np.ndarray,
    region: np.ndarray,
    grid: Grid,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None:
    """Integrate one adaptive window in the selected density measure."""

    selected = np.asarray(region, dtype=bool) & grid.mask
    values = np.where(selected, np.asarray(weight_density, dtype=float), 0.0)
    cell_mass = values * grid.cell_weights
    total = float(np.sum(cell_mass))
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        return None

    m1, m2 = np.meshgrid(grid.m1, grid.m2, indexing="ij")
    centroid = np.asarray(
        [np.sum(cell_mass * m1) / total, np.sum(cell_mass * m2) / total],
        dtype=float,
    )
    delta1 = m1 - centroid[0]
    delta2 = m2 - centroid[1]
    covariance = np.asarray(
        [
            np.sum(cell_mass * delta1 * delta1) / total,
            np.sum(cell_mass * delta1 * delta2) / total,
            np.sum(cell_mass * delta2 * delta2) / total,
        ],
        dtype=float,
    )
    indices = np.argwhere(selected)
    bounds = np.asarray(
        [
            grid.m1[int(indices[:, 0].min())],
            grid.m1[int(indices[:, 0].max())],
            grid.m2[int(indices[:, 1].min())],
            grid.m2[int(indices[:, 1].max())],
        ],
        dtype=float,
    )
    relative_widths = np.asarray(
        [
            (bounds[1] - bounds[0]) / centroid[0],
            (bounds[3] - bounds[2]) / centroid[1],
        ],
        dtype=float,
    )
    projection_m1 = np.einsum(
        "ij,j->i", values, grid.weight2, optimize=True
    ).astype(np.float32)
    projection_m2 = np.einsum(
        "ij,i->j", values, grid.weight1, optimize=True
    ).astype(np.float32)
    return (
        total,
        centroid,
        covariance,
        bounds,
        relative_widths,
        projection_m1,
        projection_m2,
    )


def _point_feature_geometry(
    point: PointFeature,
    flat_index: int,
    tree: _TreeResult,
    field: np.ndarray,
    grid: Grid,
) -> None:
    """Attach a connected half-prominence/depth region to a point feature."""

    descending = point.kind == "peak"
    if point.saddle_index is not None:
        base_level = float(field[point.saddle_index])
        reference = "persistence_saddle"
    else:
        adjacent = _adjacent_saddle_level(
            flat_index, tree, field, descending=descending
        )
        if adjacent is not None:
            base_level = adjacent
            reference = "adjacent_saddle"
        elif descending:
            base_level = 0.0
            reference = "zero_density"
        else:
            base_level = float(np.max(field[grid.mask]))
            reference = "field_maximum"

    point.base_level = base_level
    point.width_reference = reference
    contrast = point.value - base_level if descending else base_level - point.value
    denominator = point.value if descending else base_level
    if np.isfinite(denominator) and denominator > 0.0:
        point.relative_prominence = float(np.clip(contrast / denominator, 0.0, 1.0))
    dynamic_range = float(np.ptp(field[grid.mask]))
    tolerance = max(dynamic_range * 1e-10, np.finfo(float).eps)
    if not np.isfinite(contrast) or contrast <= tolerance:
        return

    half_level = point.value - 0.5 * contrast if descending else point.value + 0.5 * contrast
    point.half_level = float(half_level)
    candidate = (
        (field >= half_level) if descending else (field <= half_level)
    ) & grid.mask
    labels, _ = ndimage.label(candidate, structure=_structure())
    label = int(labels[point.index])
    if label <= 0:
        return
    region = labels == label
    if np.count_nonzero(region) < 2:
        return

    physical_weights = np.multiply.outer(
        quadrature_weights(grid.m1), quadrature_weights(grid.m2)
    )
    weights = np.where(region, physical_weights, 0.0)
    area = float(np.sum(weights))
    if not np.isfinite(area) or area <= 0.0:
        return
    x, y = np.meshgrid(grid.m1, grid.m2, indexing="ij")
    center_x = float(np.sum(weights * x) / area)
    center_y = float(np.sum(weights * y) / area)
    dx = x - center_x
    dy = y - center_y
    covariance = np.asarray(
        [
            [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
            [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
        ],
        dtype=float,
    ) / area
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    major_vector = eigenvectors[:, order[0]]
    width_major = 4.0 * math.sqrt(float(eigenvalues[0]))
    width_minor = 4.0 * math.sqrt(float(eigenvalues[1]))
    if width_major <= 0.0 or not np.isfinite(width_major + width_minor):
        return
    orientation = math.atan2(float(major_vector[1]), float(major_vector[0]))
    orientation = (orientation + 0.5 * math.pi) % math.pi - 0.5 * math.pi

    epsilon = max(float(np.max(field[grid.mask])) * 1e-12, np.finfo(float).tiny)
    if descending and base_level > epsilon:
        point.log_contrast = float(math.log(point.value / base_level))
    elif not descending and point.value > epsilon:
        point.log_contrast = float(math.log(base_level / point.value))
    point.feature_probability = float(
        np.sum(field[region] * grid.cell_weights[region])
    )
    point.width_major = width_major
    point.width_minor = width_minor
    point.width_orientation = orientation
    point.width_area = area
    point.geometry_valid = True
    point.feature_region_mask = region

    density_summary = _region_measurements(field, region, grid)
    if density_summary is None:
        return
    if point.kind == "peak":
        (
            _,
            point.mass_centroid,
            point.spatial_covariance_mass,
            point.projected_bounds_mass,
            point.relative_projected_widths,
            point.projection_m1,
            point.projection_m2,
        ) = density_summary
        point.measurement_valid = True
        return

    # A boundary pit is a global drainage feature, not a finite localized dip.
    if point.boundary_type != "none":
        return
    deficit_density = np.where(region, np.maximum(base_level - field, 0.0), 0.0)
    deficit_summary = _region_measurements(deficit_density, region, grid)
    if deficit_summary is None:
        return
    (
        point.deficit_probability,
        point.mass_centroid,
        point.spatial_covariance_mass,
        point.projected_bounds_mass,
        point.relative_projected_widths,
        point.projection_m1,
        point.projection_m2,
    ) = deficit_summary
    point.measurement_valid = True


def _ray_limit(
    points: np.ndarray, directions: np.ndarray, grid: Grid
) -> np.ndarray:
    """Distance from points to the rectangular analysis-coordinate bounds."""

    result = np.full(points.shape[0], np.inf, dtype=float)
    bounds = (
        (grid.geometry1[0], grid.geometry1[-1]),
        (grid.geometry2[0], grid.geometry2[-1]),
    )
    for axis, (lower, upper) in enumerate(bounds):
        direction = directions[:, axis]
        positive = direction > 1e-14
        negative = direction < -1e-14
        candidate = np.full(points.shape[0], np.inf, dtype=float)
        candidate[positive] = (upper - points[positive, axis]) / direction[positive]
        candidate[negative] = (lower - points[negative, axis]) / direction[negative]
        result = np.minimum(result, candidate)
    result[~np.isfinite(result)] = 0.0
    return np.maximum(result, 0.0)


def _profile_half_crossing(
    values: np.ndarray,
    distances: np.ndarray,
    *,
    kind: str,
    tolerance: float,
    fallback_level: float | None = None,
) -> tuple[float, float, float, bool] | None:
    """Return half crossing, shoulder distance/level, and fallback status."""

    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < 5 or finite[0] != 0:
        return None
    gaps = np.flatnonzero(np.diff(finite) != 1)
    stop = int(finite[gaps[0]] + 1) if gaps.size else int(finite[-1] + 1)
    raw = np.asarray(values[:stop], dtype=float)
    distance = np.asarray(distances[:stop], dtype=float)
    if raw.size < 5 or distance[-1] <= 0.0:
        return None
    smooth = ndimage.gaussian_filter1d(raw, sigma=0.65, mode="nearest")
    differences = np.diff(smooth)
    if kind == "valley":
        candidates = np.flatnonzero(
            (differences[:-1] > 0.0) & (differences[1:] <= 0.0)
        ) + 1
        significant = [
            int(item) for item in candidates if smooth[item] - smooth[0] > tolerance
        ]
    else:
        candidates = np.flatnonzero(
            (differences[:-1] < 0.0) & (differences[1:] >= 0.0)
        ) + 1
        significant = [
            int(item) for item in candidates if smooth[0] - smooth[item] > tolerance
        ]
    if significant:
        shoulder_index = significant[0]
        shoulder = float(smooth[shoulder_index])
        used_fallback = False
    elif fallback_level is not None and (
        (kind == "ridge" and smooth[0] - fallback_level > tolerance)
        or (kind == "valley" and fallback_level - smooth[0] > tolerance)
    ):
        shoulder_index = smooth.size - 1
        shoulder = float(fallback_level)
        used_fallback = True
    else:
        return None
    half_level = 0.5 * (float(smooth[0]) + shoulder)
    section = smooth[: shoulder_index + 1]
    crossed = (
        np.flatnonzero(section >= half_level)
        if kind == "valley"
        else np.flatnonzero(section <= half_level)
    )
    crossed = crossed[crossed > 0]
    if crossed.size == 0:
        return None
    right = int(crossed[0])
    left = right - 1
    denominator = float(smooth[right] - smooth[left])
    fraction = 0.0 if abs(denominator) <= tolerance else (
        half_level - float(smooth[left])
    ) / denominator
    crossing = float(distance[left] + fraction * (distance[right] - distance[left]))
    if not np.isfinite(crossing) or crossing <= 0.0:
        return None
    shoulder_distance = float(distance[shoulder_index])
    return crossing, shoulder_distance, shoulder, used_fallback


def _nearest_indices(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(coordinates, values), 1, coordinates.size - 1)
    left = right - 1
    return np.where(
        np.abs(values - coordinates[left]) <= np.abs(coordinates[right] - values),
        left,
        right,
    )


def _rasterize_width_sections(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    """Rasterize transverse sections without retaining a mask per draw."""

    result = np.zeros(grid.shape, dtype=bool)

    def mark_segment(first: np.ndarray, second: np.ndarray) -> None:
        distance = float(np.linalg.norm(second - first))
        number = max(2, min(256, int(math.ceil(2.0 * distance / grid.typical_spacing)) + 1))
        fraction = np.linspace(0.0, 1.0, number)
        samples = first[None, :] + fraction[:, None] * (second - first)[None, :]
        inside = _inside_grid_mask(samples, grid)
        if not np.any(inside):
            return
        samples = samples[inside]
        ii = _nearest_indices(grid.geometry1, samples[:, 0])
        jj = _nearest_indices(grid.geometry2, samples[:, 1])
        result[ii, jj] = True

    indices = np.flatnonzero(valid)
    for index in indices:
        mark_segment(left[index], right[index])
    for first, second in zip(indices[:-1], indices[1:]):
        if second != first + 1:
            continue
        separation = max(
            float(np.linalg.norm(left[second] - left[first])),
            float(np.linalg.norm(right[second] - right[first])),
        )
        number = max(2, min(32, int(math.ceil(separation / grid.typical_spacing)) + 1))
        for fraction in np.linspace(0.0, 1.0, number)[1:-1]:
            first_side = (1.0 - fraction) * left[first] + fraction * left[second]
            second_side = (1.0 - fraction) * right[first] + fraction * right[second]
            mark_segment(first_side, second_side)
    return result & grid.mask


def _rasterize_valley_background(
    half_left: np.ndarray,
    half_right: np.ndarray,
    shoulder_left: np.ndarray,
    shoulder_right: np.ndarray,
    shoulder_levels: np.ndarray,
    valid: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    """Interpolate the transverse log-density background inside a valley."""

    background_sum = np.zeros(grid.shape, dtype=float)
    background_count = np.zeros(grid.shape, dtype=np.int32)
    epsilon = np.finfo(float).tiny

    def mark_section(
        first_half: np.ndarray,
        second_half: np.ndarray,
        first_shoulder: np.ndarray,
        second_shoulder: np.ndarray,
        levels: np.ndarray,
    ) -> None:
        distance = float(np.linalg.norm(second_half - first_half))
        number = max(
            2,
            min(256, int(math.ceil(2.0 * distance / grid.typical_spacing)) + 1),
        )
        fraction = np.linspace(0.0, 1.0, number)
        samples = first_half[None, :] + fraction[:, None] * (
            second_half - first_half
        )[None, :]
        inside = _inside_grid_mask(samples, grid)
        if not np.any(inside):
            return
        samples = samples[inside]
        shoulder_vector = second_shoulder - first_shoulder
        denominator = float(shoulder_vector @ shoulder_vector)
        if denominator <= np.finfo(float).eps:
            return
        transverse_fraction = np.clip(
            ((samples - first_shoulder[None, :]) @ shoulder_vector) / denominator,
            0.0,
            1.0,
        )
        log_levels = np.log(np.maximum(np.asarray(levels, dtype=float), epsilon))
        background = np.exp(
            (1.0 - transverse_fraction) * log_levels[0]
            + transverse_fraction * log_levels[1]
        )
        ii = _nearest_indices(grid.geometry1, samples[:, 0])
        jj = _nearest_indices(grid.geometry2, samples[:, 1])
        np.add.at(background_sum, (ii, jj), background)
        np.add.at(background_count, (ii, jj), 1)

    indices = np.flatnonzero(valid)
    for index in indices:
        mark_section(
            half_left[index],
            half_right[index],
            shoulder_left[index],
            shoulder_right[index],
            shoulder_levels[index],
        )
    for first, second in zip(indices[:-1], indices[1:]):
        if second != first + 1:
            continue
        separation = max(
            float(np.linalg.norm(half_left[second] - half_left[first])),
            float(np.linalg.norm(half_right[second] - half_right[first])),
        )
        number = max(
            2,
            min(32, int(math.ceil(separation / grid.typical_spacing)) + 1),
        )
        for fraction in np.linspace(0.0, 1.0, number)[1:-1]:
            interpolate = lambda values: (  # noqa: E731 - compact local geometry
                (1.0 - fraction) * values[first] + fraction * values[second]
            )
            mark_section(
                interpolate(half_left),
                interpolate(half_right),
                interpolate(shoulder_left),
                interpolate(shoulder_right),
                interpolate(shoulder_levels),
            )

    result = np.full(grid.shape, np.nan, dtype=np.float32)
    measured = (background_count > 0) & grid.mask
    result[measured] = (
        background_sum[measured] / background_count[measured]
    ).astype(np.float32)
    return result


def _branch_feature_geometry(
    branch: BranchFeature,
    field: np.ndarray,
    grid: Grid,
) -> None:
    """Measure a two-sided transverse half-prominence/depth width profile."""

    if not branch.curve_available or branch.points_geometry.shape[0] < 3:
        return
    number = min(40, branch.points_geometry.shape[0])
    sample_indices = np.rint(
        np.linspace(0, branch.points_geometry.shape[0] - 1, number)
    ).astype(int)
    centers = branch.points_geometry[sample_indices]
    full_tangent = np.gradient(branch.points_geometry, axis=0)
    tangents = full_tangent[sample_indices]
    tangent_norm = np.linalg.norm(tangents, axis=1)
    usable = tangent_norm > np.finfo(float).eps
    tangents[usable] /= tangent_norm[usable, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    normals[~usable] = 0.0

    radial_points = 40
    fractions = np.linspace(0.0, 1.0, radial_points) ** 1.5
    profile_values: list[np.ndarray] = []
    profile_distances: list[np.ndarray] = []
    for sign in (-1.0, 1.0):
        directions = sign * normals
        limits = 0.999 * _ray_limit(centers, directions, grid)
        distances = limits[:, None] * fractions[None, :]
        samples = centers[:, None, :] + distances[:, :, None] * directions[:, None, :]
        flattened = samples.reshape(-1, 2)
        inside = _inside_grid_mask(flattened, grid)
        values = np.full(flattened.shape[0], np.nan, dtype=float)
        values[inside] = _sample_field(field, flattened[inside], grid)
        profile_values.append(values.reshape(number, radial_points))
        profile_distances.append(distances)

    dynamic_range = float(np.ptp(field[grid.mask]))
    tolerance = max(dynamic_range * 1e-6, np.finfo(float).eps)
    ridge_base = (
        float(field[branch.saddle_index]) if branch.kind == "ridge" else None
    )
    left_geometry = np.full((number, 2), np.nan)
    right_geometry = np.full((number, 2), np.nan)
    left_shoulder_geometry = np.full((number, 2), np.nan)
    right_shoulder_geometry = np.full((number, 2), np.nan)
    shoulder_levels = np.full((number, 2), np.nan)
    left_width = np.full(number, np.nan)
    right_width = np.full(number, np.nan)
    contrast = np.full(number, np.nan)
    used_fallback = np.zeros(number, dtype=bool)
    epsilon = max(float(np.max(field[grid.mask])) * 1e-12, np.finfo(float).tiny)
    center_mass = _geometry_to_mass(centers, grid)

    for position in range(number):
        if not usable[position]:
            continue
        left_result = _profile_half_crossing(
            profile_values[0][position],
            profile_distances[0][position],
            kind=branch.kind,
            tolerance=tolerance,
            fallback_level=ridge_base,
        )
        right_result = _profile_half_crossing(
            profile_values[1][position],
            profile_distances[1][position],
            kind=branch.kind,
            tolerance=tolerance,
            fallback_level=ridge_base,
        )
        if left_result is None or right_result is None:
            continue
        (
            left_distance,
            left_shoulder_distance,
            left_level,
            left_fallback,
        ) = left_result
        (
            right_distance,
            right_shoulder_distance,
            right_level,
            right_fallback,
        ) = right_result
        used_fallback[position] = left_fallback or right_fallback
        left_geometry[position] = centers[position] - left_distance * normals[position]
        right_geometry[position] = centers[position] + right_distance * normals[position]
        left_shoulder_geometry[position] = (
            centers[position] - left_shoulder_distance * normals[position]
        )
        right_shoulder_geometry[position] = (
            centers[position] + right_shoulder_distance * normals[position]
        )
        shoulder_levels[position] = (left_level, right_level)
        side_mass = _geometry_to_mass(
            np.vstack([left_geometry[position], right_geometry[position]]), grid
        )
        left_width[position] = float(np.linalg.norm(center_mass[position] - side_mass[0]))
        right_width[position] = float(np.linalg.norm(side_mass[1] - center_mass[position]))
        center_value = max(
            float(_sample_field(field, centers[position : position + 1], grid)[0]),
            epsilon,
        )
        side_geometric = math.sqrt(max(left_level, epsilon) * max(right_level, epsilon))
        contrast[position] = (
            math.log(center_value / side_geometric)
            if branch.kind == "ridge"
            else math.log(side_geometric / center_value)
        )

    valid = (
        np.isfinite(left_width)
        & np.isfinite(right_width)
        & (left_width > 0.0)
        & (right_width > 0.0)
        & (contrast > 0.0)
    )
    branch.width_sample_fraction = np.linspace(0.0, 1.0, number).astype(np.float32)
    branch.left_width_profile = left_width.astype(np.float32)
    branch.right_width_profile = right_width.astype(np.float32)
    branch.width_profile = (left_width + right_width).astype(np.float32)
    branch.log_contrast_profile = contrast.astype(np.float32)
    branch.width_center_geometry = centers.astype(np.float32)
    branch.half_left_geometry = left_geometry.astype(np.float32)
    branch.half_right_geometry = right_geometry.astype(np.float32)
    branch.shoulder_left_geometry = left_shoulder_geometry.astype(np.float32)
    branch.shoulder_right_geometry = right_shoulder_geometry.astype(np.float32)
    branch.shoulder_levels = shoulder_levels.astype(np.float32)
    branch.section_valid = valid
    branch.section_fallback = used_fallback
    branch.valid_width_fraction = float(np.mean(valid))
    if np.any(valid):
        branch.fallback_width_fraction = float(np.mean(used_fallback[valid]))
    if np.count_nonzero(valid) < 2:
        return

    combined = left_width[valid] + right_width[valid]
    branch.width_left = float(np.median(left_width[valid]))
    branch.width_right = float(np.median(right_width[valid]))
    branch.width_median = float(np.median(combined))
    branch.width_along_lower = float(np.quantile(combined, 0.10))
    branch.width_along_upper = float(np.quantile(combined, 0.90))
    branch.log_contrast = float(np.median(contrast[valid]))
    region = _rasterize_width_sections(left_geometry, right_geometry, valid, grid)
    if not np.any(region):
        return
    branch.feature_probability = float(
        np.sum(field[region] * grid.cell_weights[region])
    )
    branch.feature_region_mask = region
    if branch.kind == "valley":
        branch.background_density = _rasterize_valley_background(
            left_geometry,
            right_geometry,
            left_shoulder_geometry,
            right_shoulder_geometry,
            shoulder_levels,
            valid,
            grid,
        )
    branch.geometry_valid = True


def _event_branch_pairs(
    branches: list[BranchFeature],
) -> list[tuple[int, int]]:
    """Return deterministic dominant binary pairs at each curve saddle."""

    grouped: dict[tuple[str, tuple[int, int]], list[int]] = {}
    for index, branch in enumerate(branches):
        grouped.setdefault((branch.kind, branch.saddle_index), []).append(index)

    result: list[tuple[int, int]] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        remaining = sorted(
            grouped[key],
            key=lambda index: tuple(branches[index].topology_points_geometry[-1]),
        )
        while remaining:
            anchor = remaining.pop(0)
            if not remaining:
                break
            endpoint = branches[anchor].topology_points_geometry[-1]
            partner = max(
                remaining,
                key=lambda index: float(
                    np.linalg.norm(
                        branches[index].topology_points_geometry[-1] - endpoint
                    )
                ),
            )
            remaining.remove(partner)
            result.append((anchor, partner))
    return result


def _join_event_curves(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    joined = np.concatenate([first[::-1], second[1:]], axis=0)
    saddle = first.shape[0] - 1
    joined[saddle] = 0.5 * (first[0] + second[0])
    return joined


def _curve_nodal_weights(points: np.ndarray) -> np.ndarray:
    """Trapezoidal arclength weights for samples along one joined curve."""

    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)
    if points.shape[0] == 1:
        return np.ones(1, dtype=float)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    result = np.empty(points.shape[0], dtype=float)
    result[0] = 0.5 * segments[0]
    result[-1] = 0.5 * segments[-1]
    if result.size > 2:
        result[1:-1] = 0.5 * (segments[:-1] + segments[1:])
    if not np.isfinite(result).all() or np.sum(result) <= 0.0:
        return np.ones(points.shape[0], dtype=float)
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return math.nan
    values = np.asarray(values, dtype=float)[finite]
    weights = np.asarray(weights, dtype=float)[finite]
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    location = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[order[min(location, order.size - 1)]])


def _longest_true_fraction(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=bool)
    if values.size == 0 or not np.any(values):
        return 0.0
    weights = np.asarray(weights, dtype=float)
    if weights.shape != values.shape or np.sum(weights) <= 0.0:
        weights = np.ones(values.size, dtype=float)
    padded = np.concatenate([[False], values, [False]])
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    runs = [
        float(np.sum(weights[start:stop]))
        for start, stop in zip(changes[::2], changes[1::2])
    ]
    return float(max(runs) / np.sum(weights))


def _curve_events(
    branches: list[BranchFeature], field: np.ndarray, grid: Grid
) -> list[CurveEventFeature]:
    """Measure paired ridge/valley windows on one posterior realization."""

    result: list[CurveEventFeature] = []
    for first_index, second_index in _event_branch_pairs(branches):
        first = branches[first_index]
        second = branches[second_index]
        endpoint_order = sorted(
            (first_index, second_index),
            key=lambda index: tuple(branches[index].topology_points_geometry[-1]),
        )
        first_index, second_index = endpoint_order
        first = branches[first_index]
        second = branches[second_index]
        saddle = 0.5 * (
            first.topology_points_geometry[0]
            + second.topology_points_geometry[0]
        )
        topology_geometry = np.vstack(
            [
                saddle,
                first.topology_points_geometry[-1],
                second.topology_points_geometry[-1],
            ]
        )
        curve_available = bool(first.curve_available and second.curve_available)
        curve_geometry = (
            _join_event_curves(first.points_geometry, second.points_geometry)
            if curve_available
            else np.empty((0, 2), dtype=float)
        )
        visible_extent = sum(
            float(np.sum(np.linalg.norm(np.diff(branch.points_mass, axis=0), axis=1)))
            for branch in (first, second)
        )
        full_extent = sum(
            float(
                np.sum(
                    np.linalg.norm(
                        np.diff(
                            _geometry_to_mass(
                                branch.topology_points_geometry, grid
                            ),
                            axis=0,
                        ),
                        axis=1,
                    )
                )
            )
            for branch in (first, second)
        )
        event = CurveEventFeature(
            kind=first.kind,
            saddle_index=first.saddle_index,
            arm_indices=(first_index, second_index),
            topology_geometry=topology_geometry,
            curve_geometry=curve_geometry,
            curve_available=curve_available,
            boundary=bool(first.boundary or second.boundary),
            low_density_truncated=bool(
                first.low_density_truncated or second.low_density_truncated
            ),
            extent=visible_extent,
            full_extent=full_extent,
        )
        if event.full_extent > 0.0:
            event.retained_fraction = float(event.extent / event.full_extent)

        valid_sections = np.concatenate(
            [first.section_valid[::-1], second.section_valid[1:]]
        )
        section_centers = np.concatenate(
            [first.width_center_geometry[::-1], second.width_center_geometry[1:]],
            axis=0,
        )
        section_weights = _curve_nodal_weights(
            _geometry_to_mass(section_centers, grid)
        )
        if section_weights.shape != valid_sections.shape:
            section_weights = np.ones(valid_sections.size, dtype=float)
        total_weight = float(np.sum(section_weights))
        event.bounded_fraction = (
            float(np.sum(section_weights[valid_sections]) / total_weight)
            if valid_sections.size and total_weight > 0.0
            else 0.0
        )
        event.longest_bounded_fraction = _longest_true_fraction(
            valid_sections, section_weights
        )
        widths = np.concatenate(
            [first.width_profile[::-1], second.width_profile[1:]]
        )
        contrasts = np.concatenate(
            [first.log_contrast_profile[::-1], second.log_contrast_profile[1:]]
        )
        fallbacks = np.concatenate(
            [first.section_fallback[::-1], second.section_fallback[1:]]
        )
        finite_width = np.isfinite(widths) & (widths > 0.0)
        finite_contrast = np.isfinite(contrasts) & (contrasts > 0.0)
        if np.any(finite_width):
            event.width_median = _weighted_median(
                widths[finite_width], section_weights[finite_width]
            )
        if np.any(finite_contrast):
            event.log_contrast = _weighted_median(
                contrasts[finite_contrast], section_weights[finite_contrast]
            )
            event.relative_contrast = float(-np.expm1(-event.log_contrast))
        if np.any(valid_sections):
            valid_weight = float(np.sum(section_weights[valid_sections]))
            event.fallback_fraction = (
                float(
                    np.sum(
                        section_weights[valid_sections]
                        * fallbacks[valid_sections]
                    )
                    / valid_weight
                )
                if valid_weight > 0.0
                else 0.0
            )

        if not (
            curve_available
            and first.geometry_valid
            and second.geometry_valid
            and first.feature_region_mask is not None
            and second.feature_region_mask is not None
        ):
            result.append(event)
            continue
        window = (
            np.asarray(first.feature_region_mask, dtype=bool)
            | np.asarray(second.feature_region_mask, dtype=bool)
        ) & grid.mask
        if not np.any(window):
            result.append(event)
            continue

        deficit_density: np.ndarray | None = None
        if event.kind == "valley":
            backgrounds = [first.background_density, second.background_density]
            if any(item is None for item in backgrounds):
                result.append(event)
                continue
            stacked = np.stack([np.asarray(item, dtype=float) for item in backgrounds])
            finite = np.isfinite(stacked)
            count = np.sum(finite, axis=0)
            background = np.divide(
                np.nansum(stacked, axis=0),
                count,
                out=np.full(grid.shape, np.nan),
                where=count > 0,
            )
            window &= np.isfinite(background)
            deficit_density = np.where(
                window, np.maximum(background - field, 0.0), 0.0
            )

        density_summary = _region_measurements(field, window, grid)
        primary_summary = (
            _region_measurements(deficit_density, window, grid)
            if deficit_density is not None
            else density_summary
        )
        if density_summary is None or primary_summary is None:
            result.append(event)
            continue
        event.region_probability = density_summary[0]
        if deficit_density is not None:
            event.deficit_probability = primary_summary[0]
        (
            _,
            event.mass_centroid,
            event.spatial_covariance_mass,
            event.projected_bounds_mass,
            event.relative_projected_widths,
            event.projection_m1,
            event.projection_m2,
        ) = primary_summary
        event.window_mask = window
        event.region_valid = True
        result.append(event)

    return result


def _natural_log_points_to_geometry(points: np.ndarray, grid: Grid) -> np.ndarray:
    """Convert points expressed in (ln m1, ln m2) to analysis coordinates."""

    points = np.asarray(points, dtype=float)
    if grid.geometry == "log":
        return points / math.log(grid.log_base)
    return np.exp(points)


def _robust_noise(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return 0.0
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def _contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=bool), (1, 1), constant_values=False)
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _longest_candidate_run(
    candidates: list[dict[str, float | int | bool]],
    *,
    bounded_only: bool,
) -> list[dict[str, float | int | bool]]:
    selected = [
        item
        for item in sorted(candidates, key=lambda value: int(value["slice"]))
        if (bool(item["bounded"]) or not bounded_only)
    ]
    if not selected:
        return []
    runs: list[list[dict[str, float | int | bool]]] = [[selected[0]]]
    for item in selected[1:]:
        if int(item["slice"]) - int(runs[-1][-1]["slice"]) <= 2:
            runs[-1].append(item)
        else:
            runs.append([item])
    return max(runs, key=len)


def _shoulder_slice_candidates(
    t_values: np.ndarray,
    log_values: np.ndarray,
    valid: np.ndarray,
    *,
    alpha_threshold: float,
    slice_index: int,
) -> list[dict[str, float | int | bool]]:
    """Find 1D shoulder landmarks along one fixed-mass-ratio slice."""

    result: list[dict[str, float | int | bool]] = []
    for start, stop in _contiguous_true_runs(valid):
        if stop - start < 15:
            continue
        t = np.asarray(t_values[start:stop], dtype=float)
        logq = np.asarray(log_values[start:stop], dtype=float)
        spacing = float(np.median(np.diff(t)))
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue

        # A light, fixed-grid regularization is used only for derivatives. The
        # population density and all region integrals remain untouched.
        logq_derivative = ndimage.gaussian_filter1d(logq, sigma=0.8, mode="nearest")
        alpha = np.gradient(logq_derivative, t, edge_order=2)
        kappa = np.gradient(alpha, t, edge_order=2)
        alpha_noise = _robust_noise(np.diff(alpha, n=2))
        kappa_noise = _robust_noise(np.diff(kappa))
        alpha_prominence = max(5.0 * alpha_noise, 1.0e-3)
        curvature_floor = max(3.0 * kappa_noise, 1.0e-3)

        alpha_peaks, properties = find_peaks(
            alpha,
            prominence=alpha_prominence,
            distance=max(3, int(round(0.02 * alpha.size))),
        )
        positive_kappa, _ = find_peaks(kappa)
        negative_kappa, _ = find_peaks(-kappa)
        for position, alpha_peak in enumerate(alpha_peaks):
            if alpha_peak < 3 or alpha_peak >= alpha.size - 4:
                continue
            if not float(alpha[alpha_peak]) < alpha_threshold:
                continue
            left_options = positive_kappa[
                (positive_kappa < alpha_peak)
                & (kappa[positive_kappa] > curvature_floor)
            ]
            right_options = negative_kappa[
                (negative_kappa > alpha_peak)
                & (-kappa[negative_kappa] > curvature_floor)
            ]
            if left_options.size == 0 or right_options.size == 0:
                continue
            left = int(left_options[-1])
            center = int(right_options[0])
            if center <= alpha_peak or left >= alpha_peak:
                continue

            crossings = np.flatnonzero(
                (kappa[center:-1] < 0.0) & (kappa[center + 1 :] >= 0.0)
            )
            bounded = crossings.size > 0
            end = int(center + crossings[0] + 1) if bounded else int(alpha.size - 1)
            if end <= center:
                bounded = False
                end = int(alpha.size - 1)

            slope_pre = float(
                (logq_derivative[alpha_peak] - logq_derivative[left])
                / max(t[alpha_peak] - t[left], np.finfo(float).eps)
            )
            slope_post = float(
                (logq_derivative[end] - logq_derivative[center])
                / max(t[end] - t[center], np.finfo(float).eps)
            )
            contrast = slope_pre - slope_post
            prominence = float(properties["prominences"][position])
            signal_floor = max(3.0 * alpha_noise, 1.0e-3)
            if not np.isfinite(contrast) or contrast <= signal_floor:
                continue

            result.append(
                {
                    "slice": int(slice_index),
                    "t_left": float(t[left]),
                    "t_on": float(t[alpha_peak]),
                    "t_center": float(t[center]),
                    "t_end": float(t[end]),
                    "alpha_max": float(alpha[alpha_peak]),
                    "slope_pre": slope_pre,
                    "slope_post": slope_post,
                    "contrast": float(contrast),
                    "score": float(prominence + max(contrast, 0.0)),
                    "bounded": bool(bounded),
                }
            )
    return result


def _track_shoulder_candidates(
    candidates: list[dict[str, float | int | bool]],
    *,
    link_distance: float,
) -> list[list[dict[str, float | int | bool]]]:
    """Greedily connect compatible slice detections into transition fronts."""

    tracks: list[list[dict[str, float | int | bool]]] = []
    for candidate in sorted(
        candidates, key=lambda item: (int(item["slice"]), float(item["t_center"]))
    ):
        best_index = -1
        best_distance = math.inf
        for track_index, track in enumerate(tracks):
            previous = track[-1]
            gap = int(candidate["slice"]) - int(previous["slice"])
            if gap < 1 or gap > 2:
                continue
            distance = abs(
                float(candidate["t_center"]) - float(previous["t_center"])
            ) / gap
            if distance <= link_distance and distance < best_distance:
                best_index = track_index
                best_distance = distance
        if best_index < 0:
            tracks.append([candidate])
        else:
            tracks[best_index].append(candidate)
    return tracks


def _shoulder_curve(
    candidates: list[dict[str, float | int | bool]],
    r_values: np.ndarray,
    key: str,
    curve_points: int,
    grid: Grid,
    smoothing_cells: float,
) -> np.ndarray:
    r = np.asarray([r_values[int(item["slice"])] for item in candidates], dtype=float)
    t = np.asarray([float(item[key]) for item in candidates], dtype=float)
    order = np.argsort(r)
    r = r[order]
    t = t[order]
    target_r = np.linspace(float(r[0]), float(r[-1]), curve_points)
    target_t = np.interp(target_r, r, t)
    natural_points = np.column_stack([target_t + target_r, target_t - target_r])
    geometry = _natural_log_points_to_geometry(natural_points, grid)
    return _smooth_curve(geometry, grid, smoothing_cells).astype(np.float32)


def _shoulders(
    field: np.ndarray,
    grid: Grid,
    support_threshold: float,
    *,
    alpha_threshold: float,
    curve_points: int,
    curve_smoothing_cells: float,
) -> list[ShoulderFeature]:
    """Extract common-mass slope-change fronts from ``ln(field)``.

    The directional coordinate is ``t=(ln m1+ln m2)/2`` at fixed
    ``r=(ln m1-ln m2)/2``. Hence ``d/dt = d/dln(m1)+d/dln(m2)`` and the
    configurable threshold applies directly to the directional logarithmic
    slope alpha. For a density per ``dln(m1)dln(m2)``, alpha < 2 means that
    the corresponding linear-mass density is still decreasing.
    """

    if not np.isfinite(alpha_threshold):
        raise ValueError("shoulder_alpha_threshold must be finite.")
    supported = grid.mask & np.isfinite(field) & (field >= support_threshold)
    if np.count_nonzero(supported) < 64:
        return []

    u = np.log(np.asarray(grid.m1, dtype=float))
    v = np.log(np.asarray(grid.m2, dtype=float))
    t_grid = 0.5 * (u[:, None] + v[None, :])
    r_grid = 0.5 * (u[:, None] - v[None, :])
    valid_values = np.where(grid.mask, np.maximum(field, np.finfo(float).tiny), np.nan)
    nearest = ndimage.distance_transform_edt(
        ~grid.mask, return_distances=False, return_indices=True
    )
    filled = valid_values.copy()
    filled[~grid.mask] = valid_values[tuple(nearest[:, ~grid.mask])]
    log_field = np.log(filled)

    log_interpolator = RegularGridInterpolator(
        (u, v), log_field, bounds_error=False, fill_value=np.nan
    )
    density_interpolator = RegularGridInterpolator(
        (u, v), np.where(grid.mask, field, 0.0), bounds_error=False, fill_value=0.0
    )
    mask_interpolator = RegularGridInterpolator(
        (u, v), grid.mask.astype(float), method="nearest", bounds_error=False,
        fill_value=0.0,
    )

    supported_t = t_grid[supported]
    supported_r = r_grid[supported]
    r_low, r_high = float(np.min(supported_r)), float(np.max(supported_r))
    t_low, t_high = float(np.min(supported_t)), float(np.max(supported_t))
    if r_high <= r_low or t_high <= t_low:
        return []
    number_r = min(80, max(24, min(grid.shape) // 3))
    number_t = min(240, max(96, max(grid.shape)))
    # Avoid the first/last ratio slice, where the triangular and outer domain
    # boundaries can create one-sided derivative extrema.
    r_values = np.linspace(r_low, r_high, number_r + 2)[1:-1]
    t_values = np.linspace(t_low, t_high, number_t)
    candidates: list[dict[str, float | int | bool]] = []
    for slice_index, r_value in enumerate(r_values):
        sample = np.column_stack([t_values + r_value, t_values - r_value])
        log_values = np.asarray(log_interpolator(sample), dtype=float)
        density_values = np.asarray(density_interpolator(sample), dtype=float)
        valid = (
            np.asarray(mask_interpolator(sample), dtype=float) > 0.5
        ) & np.isfinite(log_values) & (density_values >= support_threshold)
        candidates.extend(
            _shoulder_slice_candidates(
                t_values,
                log_values,
                valid,
                alpha_threshold=alpha_threshold,
                slice_index=slice_index,
            )
        )
    if not candidates:
        return []

    t_spacing = float(np.median(np.diff(t_values)))
    r_spacing = float(np.median(np.diff(r_values)))
    tracks = _track_shoulder_candidates(
        candidates, link_distance=max(5.0 * t_spacing, 2.5 * r_spacing)
    )
    minimum_slices = max(4, int(math.ceil(0.06 * number_r)))
    result: list[ShoulderFeature] = []
    for track in tracks:
        located = _longest_candidate_run(track, bounded_only=False)
        if len(located) < minimum_slices:
            continue
        bounded = _longest_candidate_run(located, bounded_only=True)
        center_curve = _shoulder_curve(
            located,
            r_values,
            "t_center",
            curve_points,
            grid,
            curve_smoothing_cells,
        )
        topology = center_curve[[0, curve_points // 2, -1]].astype(np.float32)
        mass_center = _geometry_to_mass(center_curve, grid)
        extent = float(
            np.sum(np.linalg.norm(np.diff(mass_center, axis=0), axis=1))
        )
        bounded_fraction = float(
            np.mean([bool(item["bounded"]) for item in located])
        )
        longest_fraction = float(len(bounded) / len(located))
        low_density_truncated = len(bounded) < len(located)

        empty_curve = np.empty((0, 2), dtype=np.float32)
        onset_curve = empty_curve
        end_curve = empty_curve
        region = np.zeros(grid.shape, dtype=bool)
        region_valid = len(bounded) >= minimum_slices
        width_median = math.nan
        retained_fraction = 0.0
        if region_valid:
            onset_curve = _shoulder_curve(
                bounded,
                r_values,
                "t_left",
                curve_points,
                grid,
                curve_smoothing_cells,
            )
            end_curve = _shoulder_curve(
                bounded,
                r_values,
                "t_end",
                curve_points,
                grid,
                curve_smoothing_cells,
            )
            bounded_r = np.asarray(
                [r_values[int(item["slice"])] for item in bounded], dtype=float
            )
            left_t = np.asarray([float(item["t_left"]) for item in bounded])
            end_t = np.asarray([float(item["t_end"]) for item in bounded])
            order = np.argsort(bounded_r)
            bounded_r = bounded_r[order]
            left_t = left_t[order]
            end_t = end_t[order]
            inside_r = (r_grid >= bounded_r[0]) & (r_grid <= bounded_r[-1])
            interpolated_left = np.interp(r_grid, bounded_r, left_t)
            interpolated_end = np.interp(r_grid, bounded_r, end_t)
            region = (
                supported
                & inside_r
                & (t_grid >= interpolated_left)
                & (t_grid <= interpolated_end)
            )
            retained_fraction = float(
                (bounded_r[-1] - bounded_r[0])
                / max(
                    r_values[int(located[-1]["slice"])]
                    - r_values[int(located[0]["slice"])],
                    np.finfo(float).eps,
                )
            )
            onset_mass = _geometry_to_mass(onset_curve, grid)
            end_mass = _geometry_to_mass(end_curve, grid)
            width_median = float(
                np.median(np.linalg.norm(end_mass - onset_mass, axis=1))
            )

        feature = ShoulderFeature(
            kind="shoulder",
            topology_geometry=topology,
            center_curve_geometry=center_curve,
            onset_curve_geometry=onset_curve,
            end_curve_geometry=end_curve,
            alpha_threshold=float(alpha_threshold),
            curve_available=True,
            region_valid=bool(region_valid),
            boundary=bool(low_density_truncated),
            low_density_truncated=bool(low_density_truncated),
            boundary_type="support" if low_density_truncated else "none",
            alpha_max=float(np.median([float(item["alpha_max"]) for item in located])),
            slope_pre=float(
                np.median([float(item["slope_pre"]) for item in bounded])
            ) if bounded else math.nan,
            slope_post=float(
                np.median([float(item["slope_post"]) for item in bounded])
            ) if bounded else math.nan,
            slope_contrast=float(
                np.median([float(item["contrast"]) for item in bounded])
            ) if bounded else math.nan,
            extent=extent,
            full_extent=extent,
            retained_fraction=retained_fraction,
            bounded_fraction=bounded_fraction,
            longest_bounded_fraction=longest_fraction,
            width_median=width_median,
            window_mask=region if region_valid else None,
        )
        if region_valid:
            measurements = _region_measurements(field, region, grid)
            if measurements is None:
                feature.region_valid = False
                feature.window_mask = None
            else:
                (
                    feature.region_probability,
                    feature.mass_centroid,
                    feature.spatial_covariance_mass,
                    feature.projected_bounds_mass,
                    feature.relative_projected_widths,
                    feature.projection_m1,
                    feature.projection_m2,
                ) = measurements
        result.append(feature)

    result.sort(
        key=lambda item: float(np.nanmedian(item.topology_geometry[:, 0]))
    )
    return result


def _persistent_shoulders(
    feature_sets: list[list[ShoulderFeature]], grid: Grid
) -> list[ShoulderFeature]:
    """Retain shoulder fronts recovered at multiple requested scales."""

    if not feature_sets:
        return []
    total = len(feature_sets)
    minimum_count = 1 if total == 1 else 2
    domain_length = float(
        np.hypot(
            grid.geometry1[-1] - grid.geometry1[0],
            grid.geometry2[-1] - grid.geometry2[0],
        )
    )
    maximum_distance = math.sqrt(3.0) * max(
        3.0 * grid.typical_spacing, 0.08 * domain_length
    )
    clusters: list[list[tuple[int, ShoulderFeature]]] = []
    for scale_index, features in enumerate(feature_sets):
        if not clusters:
            clusters.extend([(scale_index, feature)] for feature in features)
            continue
        cluster_positions = [
            np.mean(
                [item.topology_geometry.reshape(-1) for _, item in cluster], axis=0
            )
            for cluster in clusters
        ]
        feature_positions = [feature.topology_geometry.reshape(-1) for feature in features]
        pairs = sorted(
            (
                float(np.linalg.norm(first - second)),
                cluster_index,
                feature_index,
            )
            for cluster_index, first in enumerate(cluster_positions)
            for feature_index, second in enumerate(feature_positions)
        )
        used_clusters: set[int] = set()
        used_features: set[int] = set()
        for distance, cluster_index, feature_index in pairs:
            if distance > maximum_distance:
                break
            if cluster_index in used_clusters or feature_index in used_features:
                continue
            clusters[cluster_index].append((scale_index, features[feature_index]))
            used_clusters.add(cluster_index)
            used_features.add(feature_index)
        for feature_index, feature in enumerate(features):
            if feature_index not in used_features:
                clusters.append([(scale_index, feature)])

    result: list[ShoulderFeature] = []
    for cluster in clusters:
        scale_count = len({scale_index for scale_index, _ in cluster})
        if scale_count < minimum_count:
            continue
        positions = np.asarray(
            [feature.topology_geometry.reshape(-1) for _, feature in cluster]
        )
        pairwise = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        representative = cluster[int(np.argmin(np.sum(pairwise, axis=1)))][1]
        representative.scale_count = scale_count
        representative.scale_total = total
        result.append(representative)
    result.sort(
        key=lambda item: float(np.nanmedian(item.topology_geometry[:, 0]))
    )
    return result


def _plateaus(
    field: np.ndarray,
    grid: Grid,
    boundary: np.ndarray,
) -> list[PlateauFeature]:
    g1, g2, h11, h12, h22 = _derivatives(field, grid)
    valid_values = field[grid.mask]
    dynamic_range = float(valid_values.max() - valid_values.min())
    if dynamic_range <= 0.0:
        return []
    domain_scale = float(
        np.hypot(
            grid.geometry1[-1] - grid.geometry1[0],
            grid.geometry2[-1] - grid.geometry2[0],
        )
    )
    gradient = np.hypot(g1, g2) * domain_scale / dynamic_range
    curvature = (
        np.sqrt(h11 * h11 + 2.0 * h12 * h12 + h22 * h22)
        * domain_scale**2
        / dynamic_range
    )
    display = grid.mask & (field >= hpd_threshold(field, 0.995, grid))
    if np.count_nonzero(display) < 16:
        return []
    gradient_cut = float(np.quantile(gradient[display], 0.15))
    curvature_cut = float(np.quantile(curvature[display], 0.15))
    flat = display & (gradient <= gradient_cut) & (curvature <= curvature_cut)
    labels, number = ndimage.label(flat, structure=_structure())
    minimum_pixels = max(9, int(0.0005 * np.count_nonzero(grid.mask)))
    result: list[PlateauFeature] = []
    for label_value in range(1, number + 1):
        region = labels == label_value
        if np.count_nonzero(region) < minimum_pixels:
            continue
        collar = ndimage.binary_dilation(region, structure=_structure(), iterations=2)
        collar &= grid.mask & ~region
        if not np.any(collar):
            continue
        contrast = float(np.median(field[region]) - np.median(field[collar]))
        if abs(contrast) < dynamic_range * 1e-3:
            continue
        weighted_location = np.argwhere(region)
        center = np.rint(weighted_location.mean(axis=0)).astype(int)
        kind = "plateau" if contrast > 0.0 else "depression_floor"
        result.append(
            PlateauFeature(
                kind=kind,
                center_index=(int(center[0]), int(center[1])),
                center_mass=(float(grid.m1[center[0]]), float(grid.m2[center[1]])),
                area=float(np.sum(grid.cell_weights[region])),
                probability_mass=float(
                    np.sum(field[region] * grid.cell_weights[region])
                ),
                contrast=contrast,
                boundary=bool(np.any(region & boundary)),
                region_mask=region,
            )
        )
    return result


def analyze_field(
    density: np.ndarray,
    grid: Grid,
    *,
    scale: float = 0.0,
    persistence_threshold: str | float = "auto",
    persistence_gap_min_log: float = 1.0,
    hessian_refinement: bool = True,
    detect_plateaus: bool = True,
    curve_points: int = 160,
    support_mass: float = 0.995,
    curve_smoothing_cells: float = 2.0,
    shoulder_alpha_threshold: float = 2.0,
    shoulder_scales: tuple[float, ...] | None = None,
    detect_shoulders: bool = True,
) -> FieldAnalysis:
    """Extract a simplified topological catalogue from one density field."""

    if not 0.0 < support_mass < 1.0:
        raise ValueError("support_mass must lie strictly between zero and one.")
    if not np.isfinite(curve_smoothing_cells) or curve_smoothing_cells < 0.0:
        raise ValueError("curve_smoothing_cells must be finite and nonnegative.")
    if not np.isfinite(shoulder_alpha_threshold):
        raise ValueError("shoulder_alpha_threshold must be finite.")

    field = smooth_density(np.asarray(density, dtype=float), grid, scale)
    join_tree = _build_tree(field, grid, descending=True)
    support_threshold = _hpd_threshold_from_descending_order(
        field, support_mass, grid, join_tree.order
    )
    split_tree = _build_tree(field, grid, descending=False)
    valid_values = field[grid.mask]
    dynamic_range = float(valid_values.max() - valid_values.min())
    finite_nonessential_persistence = _finite_nonessential_persistences(
        join_tree, split_tree
    )
    threshold_diagnostics = _persistence_threshold_diagnostics(
        finite_nonessential_persistence,
        dynamic_range,
        persistence_gap_min_log=persistence_gap_min_log,
        persistence_threshold=persistence_threshold,
    )
    threshold = threshold_diagnostics.threshold

    peak_surviving = _surviving_extrema(join_tree, threshold)
    pit_surviving = _surviving_extrema(split_tree, threshold)
    peak_labels = _flow_labels(join_tree, grid, peak_surviving)
    pit_labels = _flow_labels(split_tree, grid, pit_surviving)
    boundary = boundary_mask(grid.mask)

    points: list[PointFeature] = []
    point_lookup: dict[tuple[str, int], PointFeature] = {}
    for kind, tree, surviving, labels in (
        ("peak", join_tree, peak_surviving, peak_labels),
        ("pit", split_tree, pit_surviving, pit_labels),
    ):
        for flat_index in sorted(surviving):
            information = tree.extrema[flat_index]
            index = _index(flat_index, grid.shape)
            region = labels == flat_index
            point = PointFeature(
                kind=kind,
                index=index,
                value=float(field[index]),
                persistence=float(information["persistence"] or 0.0),
                boundary=bool(boundary[index]),
                region_mass=float(np.sum(field[region] * grid.cell_weights[region])),
                saddle_index=(
                    _index(int(information["saddle"]), grid.shape)
                    if information["saddle"] is not None
                    else None
                ),
                boundary_type=_boundary_type(index, grid, boundary),
            )
            _point_feature_geometry(point, flat_index, tree, field, grid)
            points.append(point)
            point_lookup[(kind, flat_index)] = point

    derivatives = _derivatives(field, grid) if hessian_refinement else None
    branches: list[BranchFeature] = []
    saddle_seen: set[tuple[str, int]] = set()
    for kind, tree, surviving in (
        ("ridge", join_tree, peak_surviving),
        ("valley", split_tree, pit_surviving),
    ):
        endpoint_kind = "peak" if kind == "ridge" else "pit"
        for event in tree.events:
            event_persistence = float(event["persistence"])
            if event_persistence < threshold:
                continue
            saddle_flat = int(event["saddle"])
            saddle_index = _index(saddle_flat, grid.shape)
            saddle_key = (kind, saddle_flat)
            if saddle_key not in saddle_seen:
                points.append(
                    PointFeature(
                        kind="saddle",
                        index=saddle_index,
                        value=float(field[saddle_index]),
                        persistence=event_persistence,
                        boundary=bool(boundary[saddle_index]),
                        role="join" if kind == "ridge" else "split",
                        boundary_type=_boundary_type(saddle_index, grid, boundary),
                    )
                )
                saddle_seen.add(saddle_key)
            for branch_information in event["branches"]:  # type: ignore[index]
                extremum_flat = int(branch_information["extremum"])
                if extremum_flat not in surviving:
                    continue
                traced = [saddle_flat] + _trace(
                    tree.flow,
                    int(branch_information["seed"]),
                    extremum_flat,
                    field.size,
                )
                indices = np.asarray([_index(item, grid.shape) for item in traced])
                geometry_points = np.column_stack(
                    [grid.geometry1[indices[:, 0]], grid.geometry2[indices[:, 1]]]
                )
                geometry_points = _resample_curve(geometry_points, curve_points)
                if derivatives is not None:
                    geometry_points = _refine_curve(
                        geometry_points, kind, derivatives, grid
                    )
                topology_points = _smooth_curve(
                    geometry_points, grid, curve_smoothing_cells
                )
                full_length = float(
                    np.sum(np.linalg.norm(np.diff(topology_points, axis=0), axis=1))
                )
                if kind == "valley":
                    (
                        visible_points,
                        retained_fraction,
                        low_density_truncated,
                        curve_available,
                    ) = _truncate_valley_to_support(
                        topology_points,
                        field,
                        support_threshold,
                        grid,
                        curve_points,
                    )
                else:
                    visible_points = topology_points
                    retained_fraction = 1.0
                    low_density_truncated = False
                    curve_available = True
                length = float(
                    np.sum(np.linalg.norm(np.diff(visible_points, axis=0), axis=1))
                )
                extremum_index = _index(extremum_flat, grid.shape)
                boundary_types = {
                    _boundary_type(saddle_index, grid, boundary),
                    _boundary_type(extremum_index, grid, boundary),
                }
                touches_outer = any("outer" in item for item in boundary_types)
                touches_mask = any("mask" in item for item in boundary_types)
                if touches_outer and touches_mask:
                    branch_boundary_type = "outer+mask"
                elif touches_outer:
                    branch_boundary_type = "outer"
                elif touches_mask:
                    branch_boundary_type = "mask"
                else:
                    branch_boundary_type = "none"
                branch = BranchFeature(
                    kind=kind,
                    saddle_index=saddle_index,
                    extremum_index=extremum_index,
                    points_geometry=visible_points,
                    points_mass=_geometry_to_mass(visible_points, grid),
                    prominence=float(branch_information["prominence"]),
                    event_persistence=event_persistence,
                    length=length,
                    boundary=bool(
                        boundary[saddle_index] or boundary[extremum_index]
                    ),
                    topology_points_geometry=topology_points,
                    full_length=full_length,
                    retained_fraction=retained_fraction,
                    low_density_truncated=low_density_truncated,
                    curve_available=curve_available,
                    boundary_type=branch_boundary_type,
                    saddle_boundary_type=_boundary_type(
                        saddle_index, grid, boundary
                    ),
                    extremum_boundary_type=_boundary_type(
                        extremum_index, grid, boundary
                    ),
                )
                _branch_feature_geometry(branch, field, grid)
                endpoint = point_lookup.get((endpoint_kind, extremum_flat))
                if endpoint is not None:
                    branch.end_identifier = endpoint.identifier
                branches.append(branch)

    events = _curve_events(branches, field, grid)
    # Background grids are only needed while constructing the paired event.
    # Do not send them back through the worker process or retain one per draw.
    for branch in branches:
        branch.background_density = None
    shoulders = (
        _shoulders(
            field,
            grid,
            support_threshold,
            alpha_threshold=shoulder_alpha_threshold,
            curve_points=curve_points,
            curve_smoothing_cells=curve_smoothing_cells,
        )
        if detect_shoulders
        else []
    )
    if detect_shoulders and shoulder_scales is not None:
        requested_scales = tuple(dict.fromkeys(float(value) for value in shoulder_scales))
        if any(not np.isfinite(value) or value < 0.0 for value in requested_scales):
            raise ValueError("shoulder_scales must be finite and nonnegative.")
        feature_sets: list[list[ShoulderFeature]] = []
        for shoulder_scale in requested_scales:
            if math.isclose(shoulder_scale, scale, rel_tol=0.0, abs_tol=1.0e-14):
                feature_sets.append(shoulders)
                continue
            shoulder_field = smooth_density(
                np.asarray(density, dtype=float), grid, shoulder_scale
            )
            shoulder_support = hpd_threshold(shoulder_field, support_mass, grid)
            feature_sets.append(
                _shoulders(
                    shoulder_field,
                    grid,
                    shoulder_support,
                    alpha_threshold=shoulder_alpha_threshold,
                    curve_points=curve_points,
                    curve_smoothing_cells=curve_smoothing_cells,
                )
            )
        shoulders = _persistent_shoulders(feature_sets, grid)
    plateaus = _plateaus(field, grid, boundary) if detect_plateaus else []
    return FieldAnalysis(
        density=field,
        scale=float(scale),
        persistence_threshold=threshold,
        persistence_threshold_diagnostics=threshold_diagnostics,
        points=points,
        branches=branches,
        events=events,
        shoulders=shoulders,
        plateaus=plateaus,
        peak_labels=peak_labels,
        pit_labels=pit_labels,
    )
