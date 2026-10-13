"""Boundary identities and first-exit geometry for supported region profiles.

This module does not alter topology connectivity or spatial quadrature.  Nodal
ordered grids have a continuous ordered boundary.  Finite-volume grids instead
occupy the union of their complete active bins, including diagonal bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .io import Grid


RIDGE_REGION_SCHEMA_VERSION = "2.0"


def _roundoff(coordinates: np.ndarray) -> float:
    return 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(coordinates))))


def _node_edges(coordinates: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [[coordinates[0]], 0.5 * (coordinates[:-1] + coordinates[1:]), [coordinates[-1]]]
    )


@dataclass(frozen=True)
class SupportRayExit:
    distance: float
    boundary_type: str
    bounded: bool


class DomainSupport:
    """Closed supported domain used only for transverse measurement profiles."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.finite_cells = grid.grid_location == "cell_centres"
        self.full_mask = bool(np.all(grid.mask))
        self.ordered = bool(
            grid.ordered_domain and grid.domain != "full" and not self.full_mask
        )
        if self.finite_cells:
            quadrature = grid.quadrature
            self.edges = (
                quadrature.axis1.edges(grid.geometry),
                quadrature.axis2.edges(grid.geometry),
            )
        else:
            self.edges = (_node_edges(grid.geometry1), _node_edges(grid.geometry2))
        self.tolerance = _roundoff(np.concatenate(self.edges))
        self.simple_ordered_nodes = bool(
            self.ordered and not self.finite_cells
            and np.array_equal(grid.mask, grid.m2[None, :] <= grid.m1[:, None])
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        inside = np.isfinite(points).all(axis=1)
        candidates = []
        for axis, edges in enumerate(self.edges):
            values = points[:, axis]
            inside &= (values >= edges[0] - self.tolerance) & (
                values <= edges[-1] + self.tolerance
            )
            right = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)
            left = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, edges.size - 2)
            candidates.append((left, right))
        selected = np.zeros(points.shape[0], dtype=bool)
        for first in candidates[0]:
            for second in candidates[1]:
                selected |= self.grid.mask[first, second]
        inside &= selected
        if self.ordered and not self.finite_cells:
            inside &= points[:, 1] <= points[:, 0] + self.tolerance
        return inside

    def _exit_type(self, point: np.ndarray, outside: np.ndarray) -> str:
        labels = []
        for axis, edges in enumerate(self.edges):
            if abs(float(point[axis] - edges[0])) <= self.tolerance:
                labels.append(f"lower_coordinate{axis + 1}")
            if abs(float(point[axis] - edges[-1])) <= self.tolerance:
                labels.append(f"upper_coordinate{axis + 1}")
        if self.ordered and not self.finite_cells and abs(float(point[0] - point[1])) <= self.tolerance:
            labels.append("ordered_diagonal")
        if not labels:
            if self.ordered and self.finite_cells:
                indices = [
                    int(np.clip(np.searchsorted(edges, outside[axis], side="right") - 1, 0, edges.size - 2))
                    for axis, edges in enumerate(self.edges)
                ]
                if self.grid.m2[indices[1]] > self.grid.m1[indices[0]]:
                    labels.append("ordered_cell_boundary")
            if not labels:
                labels.append("mask")
        return "+".join(labels)

    def ray_exit(self, point: np.ndarray, direction: np.ndarray) -> SupportRayExit:
        """First support exit; later supported intervals cannot repair a hole."""

        point = np.asarray(point, dtype=float)
        direction = np.asarray(direction, dtype=float)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) <= np.finfo(float).eps:
            return SupportRayExit(0.0, "unavailable", False)
        if not self.contains(point[None, :])[0]:
            return SupportRayExit(0.0, "outside_support", False)
        if self.full_mask or self.simple_ordered_nodes:
            limits = []
            for axis, edges in enumerate(self.edges):
                if direction[axis] > 1.0e-14:
                    limits.append(float((edges[-1] - point[axis]) / direction[axis]))
                elif direction[axis] < -1.0e-14:
                    limits.append(float((edges[0] - point[axis]) / direction[axis]))
            if self.simple_ordered_nodes and direction[1] - direction[0] > 1.0e-14:
                limits.append(float((point[0] - point[1]) / (direction[1] - direction[0])))
            distance = max(0.0, min(limits))
            boundary = point + distance * direction
            outside = boundary + max(self.tolerance * 4.0, 1.0e-10) * direction
            return SupportRayExit(distance, self._exit_type(boundary, outside), True)
        breakpoints = [np.asarray([0.0])]
        for axis, edges in enumerate(self.edges):
            if abs(float(direction[axis])) > 1.0e-14:
                distances = (edges - point[axis]) / direction[axis]
                breakpoints.append(distances[distances > self.tolerance])
        if self.ordered and not self.finite_cells:
            outward = float(direction[1] - direction[0])
            if outward > 1.0e-14:
                distance = float((point[0] - point[1]) / outward)
                if distance > self.tolerance:
                    breakpoints.append(np.asarray([distance]))
        distances = np.unique(np.concatenate(breakpoints))
        # The tail midpoint lies beyond the rectangular support, guaranteeing
        # a terminating interval without an arbitrary profile sampling step.
        extent = max(float(np.ptp(edges)) for edges in self.edges)
        tail = float(distances[-1]) + max(extent, 1.0) / np.linalg.norm(direction)
        distances = np.append(distances, tail)
        middle = 0.5 * (distances[:-1] + distances[1:])
        samples = point[None, :] + middle[:, None] * direction[None, :]
        unsupported = np.flatnonzero(~self.contains(samples))
        if not unsupported.size:
            return SupportRayExit(0.0, "unavailable", False)
        index = int(unsupported[0])
        distance = float(distances[index])
        boundary = point + distance * direction
        return SupportRayExit(distance, self._exit_type(boundary, samples[index]), True)


def support_contains(points: np.ndarray, grid: Grid) -> np.ndarray:
    return DomainSupport(grid).contains(points)


def support_ray_exit(point: np.ndarray, direction: np.ndarray, grid: Grid) -> SupportRayExit:
    return DomainSupport(grid).ray_exit(point, direction)


def boundary_contacts(point_geometry: np.ndarray, grid: Grid) -> tuple[str, ...]:
    """Descriptive critical-point labels, never an automatic rejection rule."""

    point = np.asarray(point_geometry, dtype=float)
    coordinates = (np.asarray(grid.geometry1), np.asarray(grid.geometry2))
    tolerance = _roundoff(np.concatenate(coordinates))
    labels = []
    indices = []
    for axis, centres in enumerate(coordinates):
        if abs(float(point[axis] - centres[0])) <= tolerance:
            labels.append(f"lower_coordinate{axis + 1}")
        if abs(float(point[axis] - centres[-1])) <= tolerance:
            labels.append(f"upper_coordinate{axis + 1}")
        indices.append(int(np.argmin(np.abs(centres - point[axis]))))
    ordered = bool(grid.ordered_domain and grid.domain != "full" and not np.all(grid.mask))
    if ordered and abs(float(point[0] - point[1])) <= tolerance:
        labels.append("ordered_diagonal")
    i, j = indices
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and not grid.mask[ni, nj]:
            if not (ordered and grid.m2[nj] > grid.m1[ni]):
                labels.append("mask")
                break
    return tuple(dict.fromkeys(labels))
