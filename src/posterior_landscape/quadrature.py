"""Spatial quadrature and conservative operations on finite-volume cells.

Nodal quadrature deliberately retains the historical trapezoidal rule.
Finite cells are constructed once in their native coordinate; every other
measure uses transformed edges of those same bins.  The Boolean spatial mask
selects complete cells, including ordered-domain diagonal cells.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


QUADRATURE_SCHEMA_VERSION = "1.0"


def nodal_weights(coordinates: np.ndarray) -> np.ndarray:
    """The unchanged 0.8.6 trapezoidal nodal rule."""

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("Each coordinate grid must be one-dimensional with >=2 nodes.")
    differences = np.diff(coordinates)
    if np.any(~np.isfinite(coordinates)) or np.any(differences <= 0.0):
        raise ValueError("Coordinate grids must be finite and strictly increasing.")
    result = np.empty_like(coordinates)
    result[0] = 0.5 * differences[0]
    result[-1] = 0.5 * differences[-1]
    if coordinates.size > 2:
        result[1:-1] = 0.5 * (coordinates[2:] - coordinates[:-2])
    return result


def inferred_cell_edges(centres: np.ndarray) -> np.ndarray:
    """Midpoint edges with half-spacing extrapolation in the native coordinate."""

    centres = np.asarray(centres, dtype=float)
    nodal_weights(centres)  # Shared finite/strictly-increasing validation.
    edges = np.empty(centres.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges


def _check_base(base: float) -> float:
    base = float(base)
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError("An increasing logarithmic grid requires log_base > 1.")
    return base


@dataclass(frozen=True)
class AxisQuadrature:
    centres: np.ndarray
    grid_location: str
    grid_measure: str
    log_base: float
    native_centres: np.ndarray
    native_edges: np.ndarray | None
    linear_edges: np.ndarray | None
    logarithmic_edges: np.ndarray | None
    edges_source: str = "not_applicable"

    @property
    def native_widths(self) -> np.ndarray | None:
        return None if self.native_edges is None else np.diff(self.native_edges)

    def edges(self, measure: str, log_base: float | None = None) -> np.ndarray:
        if self.grid_location != "cell_centres":
            raise ValueError("Finite-cell edges are only defined for cell_centres.")
        if measure == "linear":
            return np.asarray(self.linear_edges, dtype=float).copy()
        if measure != "log":
            raise ValueError("Integration measure must be 'linear' or 'log'.")
        if self.logarithmic_edges is None:
            raise ValueError("Logarithmic integration requires positive physical cell edges.")
        base = self.log_base if log_base is None else _check_base(log_base)
        return self.logarithmic_edges * (math.log(self.log_base) / math.log(base))

    def weights(self, measure: str, log_base: float | None = None) -> np.ndarray:
        if measure not in {"linear", "log"}:
            raise ValueError("Integration measure must be 'linear' or 'log'.")
        if self.grid_location == "cell_centres":
            return np.diff(self.edges(measure, log_base))
        if measure == "linear":
            return nodal_weights(self.centres)
        base = self.log_base if log_base is None else _check_base(log_base)
        if np.any(self.centres <= 0.0):
            raise ValueError("Logarithmic integration requires positive coordinates.")
        return nodal_weights(np.log(self.centres) / np.log(base))

    def conversion_factor(
        self,
        from_measure: str,
        to_measure: str,
        *,
        source_log_base: float | None = None,
        target_log_base: float | None = None,
    ) -> np.ndarray:
        """Convert density while preserving each finite cell's probability."""

        if self.grid_location != "cell_centres":
            raise ValueError("Width-ratio conversion is restricted to cell_centres.")
        return self.weights(from_measure, source_log_base) / self.weights(
            to_measure, target_log_base
        )

    def metadata(self) -> dict[str, object]:
        return {
            "grid_location": self.grid_location,
            "grid_measure": self.grid_measure,
            "log_base": self.log_base,
            "quadrature_schema_version": QUADRATURE_SCHEMA_VERSION,
            "quadrature_method": (
                "finite_volume_cell_widths"
                if self.grid_location == "cell_centres"
                else "trapezoidal_nodal"
            ),
            "edges_source": self.edges_source,
        }


def build_axis_quadrature(
    centres: np.ndarray,
    grid_location: str = "nodes",
    grid_measure: str = "linear",
    log_base: float = math.e,
    *,
    native_edges: np.ndarray | None = None,
) -> AxisQuadrature:
    """Construct an axis; explicit edges are for internal remapped/saved bins."""

    centres = np.asarray(centres, dtype=float).copy()
    nodal_weights(centres)
    if grid_location not in {"nodes", "cell_centres"}:
        raise ValueError("grid_location must be 'nodes' or 'cell_centres'.")
    if grid_measure not in {"linear", "log"}:
        raise ValueError("grid_measure must be 'linear' or 'log'.")
    if grid_location == "nodes":
        if native_edges is not None:
            raise ValueError("Supplied cell edges require cell_centres.")
        # grid_measure has no numerical role whatsoever in nodal mode.
        return AxisQuadrature(
            centres, grid_location, grid_measure, float(log_base), centres.copy(),
            None, None, None,
        )
    if grid_measure == "log":
        _check_base(log_base)
        if np.any(centres <= 0.0):
            raise ValueError("Logarithmic native cells require positive coordinates.")
        native = np.log(centres) / np.log(log_base)
    else:
        native = centres.copy()
    source = "inferred" if native_edges is None else "supplied"
    edges = (
        inferred_cell_edges(native)
        if native_edges is None
        else np.asarray(native_edges, dtype=float).copy()
    )
    if (
        edges.shape != (centres.size + 1,)
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or np.any(native < edges[:-1])
        or np.any(native > edges[1:])
    ):
        raise ValueError("Cell edges must be finite, increasing, and contain their centres.")
    physical = np.power(log_base, edges) if grid_measure == "log" else edges.copy()
    if np.any(~np.isfinite(physical)) or np.any(np.diff(physical) <= 0.0):
        raise ValueError("Physical cell edges must be finite and strictly increasing.")
    logarithmic = None
    if np.all(physical > 0.0):
        _check_base(log_base)
        logarithmic = edges.copy() if grid_measure == "log" else np.log(physical) / np.log(log_base)
    return AxisQuadrature(
        centres, grid_location, grid_measure, float(log_base), native, edges,
        physical, logarithmic, source,
    )


@dataclass(frozen=True)
class SpatialQuadrature:
    axis1: AxisQuadrature
    axis2: AxisQuadrature
    mask: np.ndarray

    def weights(self, measure: str, log_base: float | None = None) -> np.ndarray:
        return np.multiply.outer(
            self.axis1.weights(measure, log_base),
            self.axis2.weights(measure, log_base),
        ) * self.mask

    def conversion_factor(
        self, from_measure: str, to_measure: str, *,
        source_log_base: float | None = None, target_log_base: float | None = None,
    ) -> np.ndarray:
        return np.multiply.outer(
            self.axis1.conversion_factor(
                from_measure, to_measure, source_log_base=source_log_base,
                target_log_base=target_log_base,
            ),
            self.axis2.conversion_factor(
                from_measure, to_measure, source_log_base=source_log_base,
                target_log_base=target_log_base,
            ),
        )

    def signature(self) -> str:
        digest = hashlib.sha256(b"posterior-landscape-quadrature-v1")
        for axis in (self.axis1, self.axis2):
            digest.update(json.dumps(axis.metadata(), sort_keys=True).encode())
            for array in (axis.centres, axis.native_edges, axis.linear_edges,
                          axis.logarithmic_edges):
                if array is not None:
                    data = np.ascontiguousarray(array, dtype="<f8")
                    digest.update(str(data.shape).encode())
                    digest.update(data.tobytes())
        digest.update(np.ascontiguousarray(self.mask, dtype=np.uint8).tobytes())
        return digest.hexdigest()


def _coordinate(values: np.ndarray, measure: str, base: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if measure == "linear":
        return values
    # Nonpositive thresholds lie below all valid logarithmic cells.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(values > 0.0, np.log(values) / np.log(base), -np.inf)


def cell_interval_weights(
    axis: AxisQuadrature, measure: str, lower: np.ndarray | float,
    upper: np.ndarray | float, *, log_base: float | None = None,
) -> np.ndarray:
    """Widths of intersections with intervals whose bounds are physical."""

    base = axis.log_base if log_base is None else log_base
    edges = axis.edges(measure, log_base)
    lo = _coordinate(np.asarray(lower), measure, base)[..., None]
    hi = _coordinate(np.asarray(upper), measure, base)[..., None]
    return np.maximum(0.0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))


def cell_cdf(
    density: np.ndarray, axis: AxisQuadrature, measure: str,
    thresholds: np.ndarray | float, *, log_base: float | None = None,
) -> np.ndarray:
    """Normalized finite-cell CDF at common physical thresholds."""

    values = np.asarray(density, dtype=float)
    if values.shape[-1] != axis.centres.size:
        raise ValueError("Density shape does not agree with the quadrature axis.")
    requested = np.asarray(thresholds, dtype=float)
    partial = cell_interval_weights(axis, measure, -np.inf, requested, log_base=log_base)
    total = np.sum(values * axis.weights(measure, log_base), axis=-1)
    result = np.einsum("...i,qi->...q", values, partial.reshape(-1, axis.centres.size))
    result = np.divide(result, total[..., None], out=np.full_like(result, np.nan), where=total[..., None] > 0.0)
    result = np.clip(result, 0.0, 1.0).reshape(values.shape[:-1] + requested.shape)
    return result


def cell_percentile(
    density: np.ndarray, axis: AxisQuadrature, measure: str,
    probability: float, *, lower: np.ndarray | float | None = None,
    log_base: float | None = None,
) -> np.ndarray:
    """Invert the piecewise-constant density in its selected measure."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must lie between zero and one.")
    values = np.asarray(density, dtype=float)
    if values.shape[-1] != axis.centres.size:
        raise ValueError("Density shape does not agree with the quadrature axis.")
    edges = axis.edges(measure, log_base)
    base = axis.log_base if log_base is None else log_base
    lower_coordinate = np.full(values.shape[:-1], edges[0]) if lower is None else _coordinate(
        np.broadcast_to(np.asarray(lower, dtype=float), values.shape[:-1]), measure, base,
    )
    left = np.maximum(edges[:-1], lower_coordinate[..., None])
    widths = np.maximum(edges[1:] - left, 0.0)
    masses = values * widths
    flat = masses.reshape(-1, axis.centres.size)
    left_flat = np.broadcast_to(left, masses.shape).reshape(flat.shape)
    widths_flat = np.broadcast_to(widths, masses.shape).reshape(flat.shape)
    result = np.full(flat.shape[0], np.nan)
    for row in range(flat.shape[0]):
        total = float(np.sum(flat[row]))
        if not np.isfinite(total) or total <= 0.0:
            continue
        active = np.flatnonzero(flat[row] > 0.0)
        if probability == 0.0:
            value = left_flat[row, active[0]]
        elif probability == 1.0:
            value = edges[active[-1] + 1]
        else:
            cumulative = np.cumsum(flat[row])
            target = probability * total
            index = min(int(np.searchsorted(cumulative, target, side="left")), flat.shape[1] - 1)
            before = cumulative[index - 1] if index else 0.0
            fraction = np.clip((target - before) / flat[row, index], 0.0, 1.0)
            value = left_flat[row, index] + fraction * widths_flat[row, index]
        result[row] = base ** value if measure == "log" else value
    return result.reshape(values.shape[:-1])


def cell_joint_tails(
    density: np.ndarray, quadrature: SpatialQuadrature, measure: str,
    thresholds: np.ndarray | float, *, log_base: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Any/both/exactly-one tails integrated over complete nominal cells.

    Centre ordering does not imply pointwise ordering inside a full diagonal
    bin.  The two marginal survivals therefore cannot replace these joint
    integrals in finite-volume mode.  Only axis interval overlaps are used;
    no cell is clipped against the ordered-domain boundary.
    """

    values = np.asarray(density, dtype=float)
    if values.shape[-2:] != quadrature.mask.shape:
        raise ValueError("Density shape does not agree with the spatial quadrature.")
    values = np.where(quadrature.mask, values, 0.0)
    requested = np.asarray(thresholds, dtype=float)
    first = cell_interval_weights(
        quadrature.axis1, measure, requested, np.inf, log_base=log_base,
    ).reshape(-1, quadrature.axis1.centres.size)
    second = cell_interval_weights(
        quadrature.axis2, measure, requested, np.inf, log_base=log_base,
    ).reshape(-1, quadrature.axis2.centres.size)
    weight1 = quadrature.axis1.weights(measure, log_base)
    weight2 = quadrature.axis2.weights(measure, log_base)
    total = np.einsum("...ij,i,j->...", values, weight1, weight2, optimize=True)
    both = np.einsum("...ij,qi,qj->...q", values, first, second, optimize=True)
    # Directly sum the two disjoint straddling rectangles. This avoids
    # cancellation for very small upper tails.
    straddle = (
        np.einsum("...ij,qi,qj->...q", values, first, weight2 - second, optimize=True)
        + np.einsum("...ij,qi,qj->...q", values, weight1 - first, second, optimize=True)
    )
    shape = values.shape[:-2] + requested.shape
    normalized = []
    for event in (both + straddle, both, straddle):
        probability = np.divide(
            event, total[..., None], out=np.full_like(event, np.nan),
            where=total[..., None] > 0.0,
        )
        normalized.append(np.clip(probability, 0.0, 1.0).reshape(shape))
    return tuple(normalized)


def conservative_remap(
    density: np.ndarray, source_axis: AxisQuadrature, target_axis: AxisQuadrature,
    measure: str, *, log_base: float | None = None,
) -> np.ndarray:
    """Remap cell-constant densities by overlaps in the selected measure."""

    values = np.asarray(density, dtype=float)
    if values.shape[-1] != source_axis.centres.size:
        raise ValueError("Density shape does not agree with the source axis.")
    source = source_axis.edges(measure, log_base)
    target = target_axis.edges(measure, log_base)
    scale = max(1.0, float(np.max(np.abs(source))))
    tol = 32.0 * np.finfo(float).eps * scale
    if abs(source[0] - target[0]) > tol or abs(source[-1] - target[-1]) > tol:
        raise ValueError("Conservative remapping requires the same outer cell extent.")
    overlap = np.maximum(
        0.0, np.minimum(source[1:, None], target[None, 1:])
        - np.maximum(source[:-1, None], target[None, :-1]),
    )
    return np.einsum("...i,ij->...j", values, overlap) / np.diff(target)
