"""Publication-style landscape and feature-confidence figures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import PlotSettings
from .ensemble import (
    BranchTracker,
    EnsembleSummary,
    EventTracker,
    FeatureGeometryMaps,
    PlateauTracker,
    PointTracker,
    ShoulderTracker,
    weighted_quantile,
)
from .io import Grid, h5py
from .topology import FieldAnalysis, hpd_threshold

_KIND_COLORS = {
    "peak": "#ef7d22",
    "pit": "#2774ae",
    "ridge": "#169c91",
    "valley": "#386cb0",
    "shoulder": "#8e44ad",
    "plateau": "#d4a72c",
    "depression_floor": "#6f5aa8",
}


def make_global_tail_figure(
    output_stem: Path,
    *,
    summary: EnsembleSummary,
    weights: np.ndarray,
    settings: PlotSettings,
) -> list[Path]:
    """Plot the any/both/straddling ordered-component tail probabilities."""

    if not (settings.write_pdf or settings.write_png):
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.8, 4.1))
    colors = {
        "any": "#5C176D",
        "both": "#2774ae",
        "straddle": "#d97706",
    }
    labels = {
        "any": r"$P_{\rm any}(M)=P(m_1>M)$",
        "both": r"$P_{\rm both}(M)=P(m_2>M)$",
        "straddle": r"$P_{\rm straddle}(M)$",
    }
    floor = 1e-8
    for name, quantiles in (
        ("any", summary.tail_any_quantiles),
        ("both", summary.tail_both_quantiles),
        ("straddle", summary.tail_straddle_quantiles),
    ):
        lower, median, upper = np.maximum(np.asarray(quantiles), floor)
        axis.fill_between(
            summary.tail_mass_grid,
            lower,
            upper,
            color=colors[name],
            alpha=0.18,
            linewidth=0.0,
        )
        axis.plot(
            summary.tail_mass_grid,
            median,
            color=colors[name],
            lw=1.7,
            label=labels[name],
        )
    m1_interval = weighted_quantile(
        summary.tail_m1_scale, summary.probabilities, weights
    )
    m2_interval = weighted_quantile(
        summary.tail_m2_scale, summary.probabilities, weights
    )
    axis.axvspan(m1_interval[0], m1_interval[2], color=colors["any"], alpha=0.10)
    axis.axvline(m1_interval[1], color=colors["any"], ls="-.", lw=1.1)
    axis.axvspan(m2_interval[0], m2_interval[2], color=colors["both"], alpha=0.10)
    axis.axvline(m2_interval[1], color=colors["both"], ls=":", lw=1.2)
    axis.axhline(
        1.0 - summary.tail_probability,
        color="0.35",
        ls="--",
        lw=0.9,
        label=rf"configured tail $={1.0-summary.tail_probability:g}$",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(summary.tail_mass_grid[0], summary.tail_mass_grid[-1])
    axis.set_ylim(floor, 1.0)
    axis.set_xlabel(r"threshold mass $M$")
    axis.set_ylabel("upper-tail probability")
    axis.grid(alpha=0.18, which="both", lw=0.5)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    written: list[Path] = []
    if settings.write_pdf:
        path = output_stem.with_suffix(".pdf")
        figure.savefig(path, bbox_inches="tight")
        written.append(path)
    if settings.write_png:
        path = output_stem.with_suffix(".png")
        figure.savefig(path, dpi=220, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def make_feature_projection_figure(
    output_stem: Path,
    *,
    grid: Grid,
    summary: EnsembleSummary,
    point_trackers: list[PointTracker],
    event_trackers: list[EventTracker],
    shoulder_trackers: list[ShoulderTracker] | None = None,
    weights: np.ndarray,
    credible_mass: float,
    settings: PlotSettings,
    coordinate: str = "m1",
    location_configurations: dict[
        str, tuple[tuple[str, np.ndarray], ...]
    ] | None = None,
) -> list[Path]:
    """Compare draw-adaptive feature projections with one full marginal."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    configurations = location_configurations or {}
    if coordinate not in {"m1", "m2"}:
        raise ValueError("coordinate must be 'm1' or 'm2'.")
    projection_name = f"projection_{coordinate}"
    masses = grid.m1 if coordinate == "m1" else grid.m2
    coordinate_name = (
        grid.coordinate1_name if coordinate == "m1" else grid.coordinate2_name
    )
    coordinate_label = (
        grid.coordinate1_label if coordinate == "m1" else grid.coordinate2_label
    )
    full_quantiles = (
        summary.marginal1_quantiles
        if coordinate == "m1"
        else summary.marginal2_quantiles
    )
    entries: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []

    def add_entries(
        identifier: str,
        kind: str,
        projections: np.ndarray,
        selected: np.ndarray,
        projection_kind: str,
    ) -> None:
        split = configurations.get(identifier)
        if split:
            for label, configuration_mask in split:
                entries.append(
                    (
                        f"{identifier}-{label}",
                        kind,
                        projections,
                        selected & np.asarray(configuration_mask, dtype=bool),
                        projection_kind,
                    )
                )
        else:
            entries.append(
                (identifier, kind, projections, selected, projection_kind)
            )

    for tracker in point_trackers:
        selected = tracker.present & tracker.measurement_valid
        add_entries(
            tracker.template.identifier,
            tracker.template.kind,
            getattr(tracker, projection_name),
            selected,
            "deficit" if tracker.template.kind == "pit" else "density",
        )
    for tracker in event_trackers:
        selected = tracker.present & tracker.region_valid
        add_entries(
            tracker.template.identifier,
            tracker.template.kind,
            getattr(tracker, projection_name),
            selected,
            "deficit" if tracker.template.kind == "valley" else "density",
        )
    for tracker in shoulder_trackers or []:
        selected = tracker.present & tracker.region_valid
        add_entries(
            tracker.template.identifier,
            "shoulder",
            getattr(tracker, projection_name),
            selected,
            "density",
        )
    entries = [item for item in entries if np.any(item[3])]
    if not entries:
        return []

    columns = 3
    rows = int(np.ceil(len(entries) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0, 3.05 * rows),
        sharex=True,
        squeeze=False,
    )
    alpha = 0.5 * (1.0 - credible_mass)
    full_lower, full_median, full_upper = full_quantiles
    for axis, (identifier, kind, projections, selected, projection_kind) in zip(
        axes.ravel(), entries
    ):
        selected_weights = weights[selected]
        selected_weights = selected_weights / selected_weights.sum()
        lower, median, upper = weighted_quantile(
            projections[selected],
            (alpha, 0.5, 1.0 - alpha),
            selected_weights,
        )
        color = _KIND_COLORS[kind]
        axis.fill_between(
            masses, full_lower, full_upper, color="0.75", alpha=0.28, lw=0.0
        )
        axis.plot(masses, full_median, color="0.35", lw=1.0, label="full marginal")
        axis.fill_between(masses, lower, upper, color=color, alpha=0.22, lw=0.0)
        axis.plot(masses, median, color=color, lw=1.5, label=projection_kind)
        axis.set_title(
            f"{identifier}: {projection_kind} projection  "
            f"P(reg)={np.sum(weights[selected]):.0%}",
            fontsize=9,
        )
        axis.set_yscale("log")
        if grid.geometry == "log":
            axis.set_xscale("log")
        positive = np.concatenate([full_upper[full_upper > 0.0], upper[upper > 0.0]])
        if positive.size:
            axis.set_ylim(
                max(float(np.min(positive)) * 0.5, 1e-12),
                float(np.max(positive)) * 1.8,
            )
        axis.grid(alpha=0.18, lw=0.5)
        axis.set_xlabel(coordinate_label)
        projection_measure = (
            f"projected density / dlog({coordinate_name})"
            if grid.feature_measure == "log"
            else f"projected density / d{coordinate_name}"
        )
        axis.set_ylabel(projection_measure)
    for axis in axes.ravel()[len(entries) :]:
        axis.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Posterior draw-adaptive {coordinate_name} feature projections",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.005,
        "Bands are conditional on a valid draw-specific window; "
        "pits and valleys show missing-probability density.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.97))
    written: list[Path] = []
    if settings.write_pdf:
        path = output_stem.with_suffix(".pdf")
        figure.savefig(path, bbox_inches="tight")
        written.append(path)
    if settings.write_png:
        path = output_stem.with_suffix(".png")
        figure.savefig(path, dpi=220, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


@dataclass(frozen=True)
class _GeometryFigureFeature:
    """Small plotting-only representation of one registered feature."""

    identifier: str
    kind: str
    support: float
    geometry_support: float
    boundary_limited: bool
    center_mass: np.ndarray
    location_hpd_50: np.ndarray
    location_hpd_credible: np.ndarray
    region_inclusion_unconditional: np.ndarray
    reference_region: np.ndarray | None


@dataclass(frozen=True)
class _CurveArmFigureData:
    """Posterior arrays for one saddle-to-extremum branch."""

    identifier: str
    kind: str
    boundary_limited: bool
    reference_curve_geometry: np.ndarray
    present: np.ndarray
    curve_available: np.ndarray
    geometry_valid: np.ndarray
    topology_start_geometry: np.ndarray
    curves_geometry: np.ndarray


@dataclass(frozen=True)
class _CurveEventFigure:
    """Two arms joined at one saddle with curve-aware posterior corridors."""

    identifier: str
    kind: str
    arm_identifiers: tuple[str, str]
    support: float
    curve_support: float
    geometry_support: float
    boundary_limited: bool
    median_curve_mass: np.ndarray
    pointwise_50_lower_mass: np.ndarray
    pointwise_50_upper_mass: np.ndarray
    pointwise_credible_lower_mass: np.ndarray
    pointwise_credible_upper_mass: np.ndarray
    simultaneous_inflation: float
    simultaneous_coverage: float


def _stable_color(kind: str, ordinal: int) -> tuple[float, float, float, float]:
    """Return recognizable kind colors while separating repeated features."""

    from matplotlib import colors

    base = np.asarray(colors.to_rgba(_KIND_COLORS[kind]))
    # Alternate gently toward white/black without obscuring semantic color.
    cycle = (0.0, 0.18, -0.12, 0.30, -0.22)
    amount = cycle[ordinal % len(cycle)]
    if amount >= 0.0:
        base[:3] += amount * (1.0 - base[:3])
    else:
        base[:3] *= 1.0 + amount
    return tuple(float(value) for value in base)


def _mass_coordinates(points: np.ndarray, grid: Grid) -> np.ndarray:
    bounded = np.asarray(points, dtype=float).copy()
    bounded[..., 0] = np.clip(bounded[..., 0], grid.geometry1[0], grid.geometry1[-1])
    bounded[..., 1] = np.clip(bounded[..., 1], grid.geometry2[0], grid.geometry2[-1])
    return np.power(grid.log_base, bounded) if grid.geometry == "log" else bounded


def _conditional_quantile(
    values: np.ndarray, present: np.ndarray, weights: np.ndarray, probability: float
) -> np.ndarray:
    selected_weights = weights[present]
    if selected_weights.size == 0:
        return np.full(values.shape[1:] or (), np.nan)
    selected_weights = selected_weights / selected_weights.sum()
    return weighted_quantile(values[present], probability, selected_weights)


def _display_probability(probability: np.ndarray, grid: Grid) -> np.ndarray:
    """Lightly smooth a probability raster for contours without changing output data."""

    numerator = ndimage.gaussian_filter(
        np.where(grid.mask, probability, 0.0), sigma=0.75, mode="nearest"
    )
    denominator = ndimage.gaussian_filter(
        grid.mask.astype(float), sigma=0.75, mode="nearest"
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=(denominator > 1e-12) & grid.mask,
    )


def _point_position(
    tracker: PointTracker, weights: np.ndarray, grid: Grid
) -> np.ndarray:
    geometry = _conditional_quantile(tracker.location, tracker.present, weights, 0.5)
    if np.any(~np.isfinite(geometry)):
        index = tracker.template.index
        geometry = np.asarray([grid.geometry1[index[0]], grid.geometry2[index[1]]])
    return _mass_coordinates(np.asarray(geometry)[None, :], grid)[0]


def _plateau_position(
    tracker: PlateauTracker, weights: np.ndarray, grid: Grid
) -> np.ndarray:
    geometry = _conditional_quantile(tracker.location, tracker.present, weights, 0.5)
    if np.any(~np.isfinite(geometry)):
        index = tracker.template.center_index
        geometry = np.asarray([grid.geometry1[index[0]], grid.geometry2[index[1]]])
    return _mass_coordinates(np.asarray(geometry)[None, :], grid)[0]


def _feature_contributions(
    top,
    right,
    reference: FieldAnalysis,
    summary: EnsembleSummary,
    grid: Grid,
    points: Iterable[PointTracker],
) -> None:
    """Draw reference Morse-basin contributions to both marginals."""

    kind_number = {"peak": 0, "pit": 0}
    for tracker in points:
        feature = tracker.template
        if not tracker.is_reference or feature.kind not in kind_number:
            continue
        labels = (
            reference.peak_labels if feature.kind == "peak" else reference.pit_labels
        )
        label = int(np.ravel_multi_index(feature.index, grid.shape))
        region = labels == label
        if not np.any(region):
            continue
        contribution = np.where(region, summary.reference_density, 0.0)
        marginal1, marginal2 = grid.marginals(contribution)
        color = _stable_color(feature.kind, kind_number[feature.kind])
        kind_number[feature.kind] += 1
        style = "-" if feature.kind == "peak" else "--"
        top.plot(grid.m1, marginal1, color=color, lw=1.0, ls=style, alpha=0.85)
        right.plot(marginal2, grid.m2, color=color, lw=1.0, ls=style, alpha=0.85)


def make_landscape_figure(
    output_stem: Path,
    *,
    grid: Grid,
    summary: EnsembleSummary,
    reference: FieldAnalysis,
    point_trackers: list[PointTracker],
    branch_trackers: list[BranchTracker],
    event_trackers: list[EventTracker] | None = None,
    plateau_trackers: list[PlateauTracker],
    shoulder_trackers: list[ShoulderTracker] | None = None,
    weights: np.ndarray,
    ridge_probability: np.ndarray,
    valley_probability: np.ndarray,
    geometry_maps: dict[str, FeatureGeometryMaps],
    settings: PlotSettings,
) -> list[Path]:
    """Write the central density, marginals, and posterior feature geometry."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8.4, 7.7), constrained_layout=False)
    layout = figure.add_gridspec(
        2,
        3,
        width_ratios=(5.0, 1.35, 0.16),
        height_ratios=(1.35, 5.0),
        left=0.10,
        right=0.96,
        bottom=0.09,
        top=0.96,
        hspace=0.04,
        wspace=0.04,
    )
    top = figure.add_subplot(layout[0, 0])
    joint = figure.add_subplot(layout[1, 0], sharex=top)
    right = figure.add_subplot(layout[1, 1], sharey=joint)
    figure.add_subplot(layout[0, 1]).axis("off")
    figure.add_subplot(layout[0, 2]).axis("off")
    colorbar_axis = figure.add_subplot(layout[1, 2])

    density = summary.reference_density
    positive = density[grid.mask & (density > 0.0)]
    if positive.size == 0:
        raise ValueError("The reference density has no positive value to plot.")
    display_threshold = hpd_threshold(density, settings.display_probability, grid)
    upper = float(np.nanmax(positive))
    lower = max(float(display_threshold), upper * 1e-7, np.finfo(float).tiny)
    shown = np.ma.masked_where(
        ~grid.mask,
        np.where(density > 0.0, density, np.finfo(float).tiny),
    )
    color_map = plt.get_cmap(settings.joint_cmap).copy()
    color_map.set_bad("white")
    color_map.set_under("black")
    mesh = joint.pcolormesh(
        grid.m1,
        grid.m2,
        shown.T,
        shading="auto",
        cmap=color_map,
        norm=LogNorm(vmin=lower, vmax=upper),
        rasterized=True,
    )

    contour_values = [
        hpd_threshold(density, probability, grid)
        for probability in settings.hpd_contours
    ]
    ordered = sorted(
        zip(contour_values, settings.hpd_contours), key=lambda item: item[0]
    )
    distinct_levels: list[float] = []
    level_labels: dict[float, str] = {}
    for value, probability in ordered:
        if distinct_levels and np.isclose(value, distinct_levels[-1]):
            continue
        distinct_levels.append(value)
        level_labels[value] = f"{100.0 * probability:g}% HPD"
    if distinct_levels:
        contour = joint.contour(
            grid.m1,
            grid.m2,
            density.T,
            levels=distinct_levels,
            colors="white",
            linewidths=0.8,
            alpha=0.72,
        )
        joint.clabel(contour, fmt=level_labels, fontsize=7, inline=True)

    def fill_region(mask: np.ndarray, color, alpha: float, zorder: int) -> None:
        if np.any(mask) and np.any(~mask):
            joint.contourf(
                grid.m1,
                grid.m2,
                mask.T.astype(float),
                levels=[0.5, 1.5],
                colors=[to_rgba(color, alpha)],
                antialiased=True,
                zorder=zorder,
            )

    kind_number: dict[str, int] = {}
    for tracker in branch_trackers:
        if not tracker.template.curve_available:
            continue
        kind = tracker.template.kind
        ordinal = kind_number.get(kind, 0)
        kind_number[kind] = ordinal + 1
        color = _stable_color(kind, ordinal)
        spatial = geometry_maps[tracker.template.identifier]
        if tracker.template.feature_region_mask is not None:
            fill_region(
                tracker.template.feature_region_mask & (density >= display_threshold),
                color,
                alpha=0.035,
                zorder=4,
            )
        fill_region(spatial.location_hpd_credible, color, alpha=0.065, zorder=4)
        fill_region(spatial.location_hpd_50, color, alpha=0.075, zorder=5)
        center_mass = _mass_coordinates(tracker.template.points_geometry, grid)
        joint.plot(
            center_mass[:, 0],
            center_mass[:, 1],
            color=color,
            lw=1.65,
            ls="-" if kind == "ridge" else "--",
            zorder=6,
        )
        midpoint = center_mass[len(center_mass) // 2]
        joint.annotate(
            tracker.template.identifier,
            midpoint,
            xytext=(3, 3),
            textcoords="offset points",
            color=color,
            fontsize=7,
            weight="bold",
            zorder=8,
        )

    kind_number = {}
    for tracker in point_trackers:
        kind = tracker.template.kind
        ordinal = kind_number.get(kind, 0)
        kind_number[kind] = ordinal + 1
        color = _stable_color(kind, ordinal)
        spatial = geometry_maps[tracker.template.identifier]
        if kind == "peak" and tracker.template.feature_region_mask is not None:
            fill_region(
                tracker.template.feature_region_mask, color, alpha=0.045, zorder=4
            )
        fill_region(spatial.location_hpd_credible, color, alpha=0.075, zorder=5)
        fill_region(spatial.location_hpd_50, color, alpha=0.090, zorder=6)
        position = _point_position(tracker, weights, grid)
        marker = "^" if kind == "peak" else "v"
        joint.scatter(
            position[0],
            position[1],
            marker=marker,
            s=45,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=9,
        )
        joint.annotate(
            tracker.template.identifier,
            position,
            xytext=(4, -7 if kind == "peak" else 4),
            textcoords="offset points",
            color=color,
            fontsize=7,
            weight="bold",
            zorder=10,
        )

    for ordinal, tracker in enumerate(shoulder_trackers or []):
        color = _stable_color("shoulder", ordinal)
        inclusion = tracker.region_weight_sum
        available_levels = [level for level in (0.10, 0.50, 0.90) if np.nanmax(inclusion) >= level]
        if available_levels:
            joint.contour(
                grid.m1,
                grid.m2,
                inclusion.T,
                levels=available_levels,
                colors=[color],
                linewidths=[0.55, 0.8, 1.0][: len(available_levels)],
                alpha=0.55,
                zorder=5,
            )
        center = _mass_coordinates(tracker.template.center_curve_geometry, grid)
        joint.plot(center[:, 0], center[:, 1], color=color, lw=1.6, ls="-.", zorder=7)
        for boundary_curve in (
            tracker.template.onset_curve_geometry,
            tracker.template.end_curve_geometry,
        ):
            if boundary_curve.size:
                boundary_mass = _mass_coordinates(boundary_curve, grid)
                joint.plot(
                    boundary_mass[:, 0],
                    boundary_mass[:, 1],
                    color=color,
                    lw=0.85,
                    ls=":",
                    alpha=0.9,
                    zorder=7,
                )
        midpoint = center[len(center) // 2]
        joint.annotate(
            tracker.template.identifier,
            midpoint,
            xytext=(4, 3),
            textcoords="offset points",
            color=color,
            fontsize=7,
            weight="bold",
            zorder=10,
        )

    kind_number = {}
    for tracker in plateau_trackers:
        kind = tracker.template.kind
        ordinal = kind_number.get(kind, 0)
        kind_number[kind] = ordinal + 1
        color = _stable_color(kind, ordinal)
        position = _plateau_position(tracker, weights, grid)
        if tracker.is_reference and np.any(tracker.template.region_mask):
            joint.contour(
                grid.m1,
                grid.m2,
                tracker.template.region_mask.T.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=1.1,
                linestyles=[":"],
            )
        joint.scatter(
            *position,
            marker="s",
            s=32,
            facecolor="none",
            edgecolor=color,
            linewidth=1.2,
            zorder=9,
        )
        joint.annotate(
            tracker.template.identifier,
            position,
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=7,
            weight="bold",
            zorder=10,
        )

    q1 = summary.marginal1_quantiles
    q2 = summary.marginal2_quantiles
    top.fill_between(grid.m1, q1[0], q1[2], color="0.45", alpha=0.24, lw=0)
    top.plot(grid.m1, q1[1], color="black", lw=1.6)
    right.fill_betweenx(grid.m2, q2[0], q2[2], color="0.45", alpha=0.24, lw=0)
    right.plot(q2[1], grid.m2, color="black", lw=1.6)
    # The adaptive projections use physical grid coordinates.  Plotting them
    # explicitly here avoids the normalization mismatch of the separate
    # comparison panels and keeps both marginal axes aligned with the map.
    entries: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for tracker in point_trackers:
        entries.append(
            (
                tracker.template.identifier,
                tracker.template.kind,
                tracker.projection_m1,
                tracker.projection_m2,
                tracker.present & tracker.measurement_valid,
            )
        )
    for tracker in event_trackers or []:
        entries.append(
            (
                tracker.template.identifier,
                tracker.template.kind,
                tracker.projection_m1,
                tracker.projection_m2,
                tracker.present & tracker.region_valid,
            )
        )
    for tracker in shoulder_trackers or []:
        entries.append(
            (
                tracker.template.identifier,
                "shoulder",
                tracker.projection_m1,
                tracker.projection_m2,
                tracker.present & tracker.region_valid,
            )
        )
    projection_ordinals: dict[str, int] = {}
    projection_styles = {
        "peak": "-",
        "pit": ":",
        "ridge": "-",
        "valley": "--",
        "shoulder": "-.",
    }
    for identifier, kind, first, second, selected in entries:
        if not np.any(selected):
            continue
        selected_weights = weights[selected]
        selected_weights = selected_weights / np.sum(selected_weights)
        median1 = weighted_quantile(first[selected], 0.5, selected_weights)
        median2 = weighted_quantile(second[selected], 0.5, selected_weights)
        ordinal = projection_ordinals.get(kind, 0)
        projection_ordinals[kind] = ordinal + 1
        color = _stable_color(kind, ordinal)
        style = projection_styles[kind]
        top.plot(grid.m1, median1, color=color, lw=1.0, ls=style, alpha=0.88)
        right.plot(median2, grid.m2, color=color, lw=1.0, ls=style, alpha=0.88)

    tail_color = "#4d4d4d"
    has_ordered_tails = summary.tail_mass_grid.size > 0
    if has_ordered_tails:
        tail_m1 = weighted_quantile(
            summary.tail_m1_scale, summary.probabilities, weights
        )
        tail_m2 = weighted_quantile(
            summary.tail_m2_scale, summary.probabilities, weights
        )
        top.axvspan(tail_m1[0], tail_m1[2], color=tail_color, alpha=0.13, lw=0)
        top.axvline(tail_m1[1], color=tail_color, lw=1.1, ls="-.")
        joint.axvspan(tail_m1[0], tail_m1[2], color=tail_color, alpha=0.055, lw=0)
        joint.axvline(tail_m1[1], color=tail_color, lw=0.8, ls="-.", alpha=0.8)
        right.axhspan(tail_m2[0], tail_m2[2], color=tail_color, alpha=0.13, lw=0)
        right.axhline(tail_m2[1], color=tail_color, lw=1.1, ls=":")
        joint.axhspan(tail_m2[0], tail_m2[2], color=tail_color, alpha=0.055, lw=0)
        joint.axhline(tail_m2[1], color=tail_color, lw=0.8, ls=":", alpha=0.8)

    if grid.geometry == "log":
        for axis in (joint, top):
            axis.set_xscale("log")
        for axis in (joint, right):
            axis.set_yscale("log")
    joint.set_xlim(grid.m1[0], grid.m1[-1])
    joint.set_ylim(grid.m2[0], grid.m2[-1])
    joint.set_xlabel(grid.coordinate1_label)
    joint.set_ylabel(grid.coordinate2_label)
    if grid.feature_measure == "linear":
        top.set_ylabel(f"density / d{grid.coordinate1_name}")
        right.set_xlabel(f"density / d{grid.coordinate2_name}")
        density_label = (
            f"density / (d{grid.coordinate1_name} d{grid.coordinate2_name})"
        )
    else:
        top.set_ylabel(f"density / dlog({grid.coordinate1_name})")
        right.set_xlabel(f"density / dlog({grid.coordinate2_name})")
        density_label = (
            "density / "
            f"(dlog({grid.coordinate1_name}) dlog({grid.coordinate2_name}))"
        )
    top.tick_params(labelbottom=False)
    right.tick_params(labelleft=False)
    top.grid(alpha=0.18, lw=0.5)
    right.grid(alpha=0.18, lw=0.5)
    joint.set_facecolor("white")

    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label(density_label, fontsize=9)

    legend = [
        Line2D([], [], marker="^", ls="", color=_KIND_COLORS["peak"], label="peak"),
        Line2D([], [], marker="v", ls="", color=_KIND_COLORS["pit"], label="pit"),
        Line2D(
            [],
            [],
            color=_KIND_COLORS["ridge"],
            lw=1.7,
            label="topological ridge",
        ),
        Line2D([], [], color=_KIND_COLORS["valley"], lw=1.7, ls="--", label="valley"),
        Line2D(
            [],
            [],
            color=_KIND_COLORS["shoulder"],
            lw=1.7,
            ls="-.",
            label="slope-change front",
        ),
        Patch(
            facecolor=to_rgba("0.25", 0.10),
            edgecolor="none",
            label="location credible region",
        ),
        Patch(
            facecolor=to_rgba("0.25", 0.04),
            edgecolor="none",
            label="half-prominence/depth region",
        ),
        Line2D(
            [],
            [],
            marker="s",
            markerfacecolor="none",
            ls="",
            color=_KIND_COLORS["plateau"],
            label="plateau/floor",
        ),
    ]
    if has_ordered_tails:
        legend.insert(
            5,
            Line2D(
                [],
                [],
                color=tail_color,
                lw=1.1,
                ls="-.",
                label=f"{100.0 * summary.tail_probability:g}% global tail scales",
            ),
        )
    joint.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.82, ncol=2)

    written: list[Path] = []
    if settings.write_pdf:
        pdf = output_stem.with_suffix(".pdf")
        figure.savefig(pdf, dpi=220, bbox_inches="tight")
        written.append(pdf)
    if settings.write_png:
        png = output_stem.with_suffix(".png")
        figure.savefig(png, dpi=220, bbox_inches="tight")
        written.append(png)
    plt.close(figure)
    return written


def _feature_sort_key(
    feature: _GeometryFigureFeature | _CurveEventFigure,
) -> tuple[int, int, str]:
    order = {"peak": 0, "pit": 1, "ridge": 2, "valley": 3}
    digits = "".join(
        character for character in feature.identifier if character.isdigit()
    )
    number = int(digits) if digits else 0
    return order.get(feature.kind, 99), number, feature.identifier


def _geometry_figure_features(
    *,
    grid: Grid,
    point_trackers: list[PointTracker],
    weights: np.ndarray,
    geometry_maps: dict[str, FeatureGeometryMaps],
) -> list[_GeometryFigureFeature]:
    """Convert live point trackers to the compact data needed by the figure."""

    result: list[_GeometryFigureFeature] = []
    for tracker in point_trackers:
        if tracker.template.kind not in {"peak", "pit"}:
            continue
        spatial = geometry_maps[tracker.template.identifier]
        result.append(
            _GeometryFigureFeature(
                identifier=tracker.template.identifier,
                kind=tracker.template.kind,
                support=float(np.sum(weights[tracker.present])),
                geometry_support=float(
                    np.sum(weights[tracker.present & tracker.geometry_valid])
                ),
                boundary_limited=tracker.template.boundary_type != "none",
                center_mass=_point_position(tracker, weights, grid)[None, :],
                location_hpd_50=spatial.location_hpd_50,
                location_hpd_credible=spatial.location_hpd_credible,
                region_inclusion_unconditional=(
                    spatial.region_inclusion_unconditional
                ),
                reference_region=tracker.template.feature_region_mask,
            )
        )
    return sorted(result, key=_feature_sort_key)


def _curve_arm_data_from_trackers(
    branch_trackers: list[BranchTracker],
) -> list[_CurveArmFigureData]:
    result: list[_CurveArmFigureData] = []
    for tracker in branch_trackers:
        template = tracker.template
        if template.kind not in {"ridge", "valley"}:
            continue
        result.append(
            _CurveArmFigureData(
                identifier=template.identifier,
                kind=template.kind,
                boundary_limited=(
                    template.boundary_type != "none"
                    or template.extremum_boundary_type != "none"
                    or template.low_density_truncated
                ),
                reference_curve_geometry=np.asarray(
                    template.points_geometry, dtype=float
                ),
                present=np.asarray(tracker.present, dtype=bool),
                curve_available=np.asarray(tracker.curve_available, dtype=bool),
                geometry_valid=np.asarray(tracker.geometry_valid, dtype=bool),
                topology_start_geometry=np.asarray(
                    tracker.topology_endpoints[:, :2], dtype=float
                ),
                curves_geometry=np.asarray(tracker.curves, dtype=float),
            )
        )
    return result


def _paired_curve_arm_groups(
    arms: list[_CurveArmFigureData], typical_spacing: float
) -> list[tuple[_CurveArmFigureData, _CurveArmFigureData]]:
    """Pair the two reference arms born at each merge/split saddle."""

    groups: list[tuple[_CurveArmFigureData, _CurveArmFigureData]] = []
    tolerance = max(0.75 * typical_spacing, 1e-10)
    for kind in ("ridge", "valley"):
        remaining = sorted(
            [arm for arm in arms if arm.kind == kind],
            key=lambda arm: arm.identifier,
        )
        while remaining:
            anchor = remaining.pop(0)
            start = anchor.reference_curve_geometry[0]
            matches = [
                arm
                for arm in remaining
                if np.linalg.norm(arm.reference_curve_geometry[0] - start)
                <= tolerance
            ]
            if not matches:
                continue
            # Binary saddles are expected. In a grid-degenerate multiway event,
            # select the arm whose endpoint is farthest from the anchor so the
            # displayed curve spans the dominant connector; all arms remain in
            # the numerical branch diagnostics.
            partner = max(
                matches,
                key=lambda arm: np.linalg.norm(
                    arm.reference_curve_geometry[-1]
                    - anchor.reference_curve_geometry[-1]
                ),
            )
            remaining.remove(partner)
            groups.append(
                tuple(sorted((anchor, partner), key=lambda arm: arm.identifier))
            )
    groups.sort(
        key=lambda pair: (
            0 if pair[0].kind == "ridge" else 1,
            float(np.mean([arm.reference_curve_geometry[0, 0] for arm in pair])),
            float(np.mean([arm.reference_curve_geometry[0, 1] for arm in pair])),
        )
    )
    return groups


def _join_curve_arms(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Join saddle-to-endpoint arms into endpoint-to-saddle-to-endpoint curves."""

    joined = np.concatenate([first[..., ::-1, :], second[..., 1:, :]], axis=-2)
    saddle_index = first.shape[-2] - 1
    joined[..., saddle_index, :] = 0.5 * (
        first[..., 0, :] + second[..., 0, :]
    )
    return joined


def _curve_normals(curve: np.ndarray) -> np.ndarray:
    tangent = np.gradient(curve, axis=0)
    length = np.linalg.norm(tangent, axis=1)
    valid = length > 1e-12
    if not np.all(valid):
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            tangent[:] = (1.0, 0.0)
            length[:] = 1.0
        else:
            missing = np.flatnonzero(~valid)
            nearest = valid_indices[
                np.argmin(np.abs(missing[:, None] - valid_indices[None, :]), axis=1)
            ]
            tangent[missing] = tangent[nearest]
            length[missing] = length[nearest]
    tangent = tangent / length[:, None]
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _curve_event_summary(
    pair: tuple[_CurveArmFigureData, _CurveArmFigureData],
    *,
    identifier: str,
    weights: np.ndarray,
    credible_mass: float,
    typical_spacing: float,
    geometry: str,
    log_base: float,
    configuration_mask: np.ndarray | None = None,
) -> _CurveEventFigure:
    """Build a pointwise transverse corridor for one paired saddle event."""

    first, second = pair
    event_present = first.present & second.present
    starts = np.stack(
        [first.topology_start_geometry, second.topology_start_geometry], axis=1
    )
    finite_starts = np.all(np.isfinite(starts), axis=(1, 2))
    saddle_distance = np.linalg.norm(starts[:, 0] - starts[:, 1], axis=1)
    event_present &= finite_starts & (saddle_distance <= 1.5 * typical_spacing)
    if configuration_mask is not None:
        event_present &= np.asarray(configuration_mask, dtype=bool)
    curve_present = (
        event_present
        & first.curve_available
        & second.curve_available
        & np.all(np.isfinite(first.curves_geometry), axis=(1, 2))
        & np.all(np.isfinite(second.curves_geometry), axis=(1, 2))
    )
    geometry_present = (
        curve_present & first.geometry_valid & second.geometry_valid
    )
    support = float(np.sum(weights[event_present]))
    curve_support = float(np.sum(weights[curve_present]))
    geometry_support = float(np.sum(weights[geometry_present]))

    reference = _join_curve_arms(
        first.reference_curve_geometry, second.reference_curve_geometry
    )
    selected = np.flatnonzero(curve_present)
    if selected.size:
        curves = _join_curve_arms(
            first.curves_geometry[selected], second.curves_geometry[selected]
        )
        selected_weights = weights[selected]
        selected_weights = selected_weights / selected_weights.sum()
        median = weighted_quantile(curves, 0.5, selected_weights)
        normals = _curve_normals(median)
        displacement = np.einsum(
            "dpc,pc->dp", curves - median[None, :, :], normals
        )
        tail = 0.5 * (1.0 - credible_mass)
        quantiles = weighted_quantile(
            displacement,
            (tail, 0.25, 0.75, 1.0 - tail),
            selected_weights,
        )
        credible_lower, inner_lower, inner_upper, credible_upper = quantiles
        credible_lower = np.minimum(credible_lower, 0.0)
        credible_upper = np.maximum(credible_upper, 0.0)
        inner_lower = np.minimum(inner_lower, 0.0)
        inner_upper = np.maximum(inner_upper, 0.0)

        epsilon = max(1e-12, 1e-6 * typical_spacing)
        positive_scale = np.maximum(credible_upper, epsilon)
        negative_scale = np.maximum(-credible_lower, epsilon)
        standardized = np.where(
            displacement >= 0.0,
            displacement / positive_scale[None, :],
            -displacement / negative_scale[None, :],
        )
        maximum_deviation = np.max(standardized, axis=1)
        simultaneous_inflation = max(
            1.0,
            float(
                weighted_quantile(
                    maximum_deviation, credible_mass, selected_weights
                )
            ),
        )
        simultaneous_coverage = float(
            np.sum(
                selected_weights[
                    maximum_deviation <= simultaneous_inflation + 1e-12
                ]
            )
        )
    else:
        median = reference
        normals = _curve_normals(median)
        credible_lower = np.zeros(median.shape[0])
        credible_upper = np.zeros(median.shape[0])
        inner_lower = np.zeros(median.shape[0])
        inner_upper = np.zeros(median.shape[0])
        simultaneous_inflation = np.nan
        simultaneous_coverage = np.nan

    def boundary(offset: np.ndarray) -> np.ndarray:
        geometry_points = median + offset[:, None] * normals
        return (
            np.power(log_base, geometry_points)
            if geometry == "log"
            else geometry_points
        )

    median_mass = np.power(log_base, median) if geometry == "log" else median
    return _CurveEventFigure(
        identifier=identifier,
        kind=first.kind,
        arm_identifiers=(first.identifier, second.identifier),
        support=support,
        curve_support=curve_support,
        geometry_support=geometry_support,
        boundary_limited=first.boundary_limited or second.boundary_limited,
        median_curve_mass=median_mass,
        pointwise_50_lower_mass=boundary(inner_lower),
        pointwise_50_upper_mass=boundary(inner_upper),
        pointwise_credible_lower_mass=boundary(credible_lower),
        pointwise_credible_upper_mass=boundary(credible_upper),
        simultaneous_inflation=simultaneous_inflation,
        simultaneous_coverage=simultaneous_coverage,
    )


def _curve_event_features(
    arms: list[_CurveArmFigureData],
    *,
    weights: np.ndarray,
    credible_mass: float,
    typical_spacing: float,
    geometry: str,
    log_base: float,
    location_configurations: dict[
        str, tuple[tuple[str, np.ndarray], ...]
    ] | None = None,
) -> list[_CurveEventFigure]:
    events: list[_CurveEventFigure] = []
    counters = {"ridge": 0, "valley": 0}
    configurations = location_configurations or {}
    for pair in _paired_curve_arm_groups(arms, typical_spacing):
        kind = pair[0].kind
        counters[kind] += 1
        prefix = "R2DE" if kind == "ridge" else "V2DE"
        identifier = f"{prefix}{counters[kind]}"
        split = configurations.get(identifier)
        if split:
            for label, configuration_mask in split:
                events.append(
                    _curve_event_summary(
                        pair,
                        identifier=f"{identifier}-{label}",
                        weights=weights,
                        credible_mass=credible_mass,
                        typical_spacing=typical_spacing,
                        geometry=geometry,
                        log_base=log_base,
                        configuration_mask=configuration_mask,
                    )
                )
        else:
            events.append(
                _curve_event_summary(
                    pair,
                    identifier=identifier,
                    weights=weights,
                    credible_mass=credible_mass,
                    typical_spacing=typical_spacing,
                    geometry=geometry,
                    log_base=log_base,
                )
            )
    return events


def _masked_probability_for_plot(
    values: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Smooth only for drawing while respecting the triangular support mask."""

    numerator = ndimage.gaussian_filter(
        np.where(mask, values, 0.0), sigma=0.75, mode="nearest"
    )
    denominator = ndimage.gaussian_filter(
        mask.astype(float), sigma=0.75, mode="nearest"
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=(denominator > 1e-12) & mask,
    )


def _array_hpd_threshold(
    density: np.ndarray,
    mask: np.ndarray,
    weight1: np.ndarray,
    weight2: np.ndarray,
    probability: float,
) -> float:
    """HPD threshold used when rebuilding a figure from results.h5."""

    cell_mass = density * np.multiply.outer(weight1, weight2)
    valid = np.flatnonzero(mask & np.isfinite(density) & (density > 0.0))
    if valid.size == 0:
        return np.inf
    order = valid[np.argsort(density.ravel()[valid])[::-1]]
    cumulative = np.cumsum(cell_mass.ravel()[order])
    target = probability * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), order.size - 1)
    return float(density.ravel()[order[index]])


def _write_geometry_confidence_figure(
    output_stem: Path,
    *,
    m1: np.ndarray,
    m2: np.ndarray,
    mask: np.ndarray,
    density: np.ndarray,
    display_threshold: float,
    geometry: str,
    coordinate1_label: str,
    coordinate2_label: str,
    features: list[_GeometryFigureFeature],
    curve_events: list[_CurveEventFigure],
    credible_mass: float,
    settings: PlotSettings,
) -> list[Path]:
    """Draw spatial uncertainty and exact feature-support summaries."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12.2, 10.0), constrained_layout=False)
    layout = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.975,
        bottom=0.125,
        top=0.95,
        hspace=0.20,
        wspace=0.17,
    )
    point_axis = figure.add_subplot(layout[0, 0])
    ridge_axis = figure.add_subplot(layout[0, 1], sharex=point_axis, sharey=point_axis)
    valley_axis = figure.add_subplot(layout[1, 0], sharex=point_axis, sharey=point_axis)
    support_axis = figure.add_subplot(layout[1, 1])

    positive = density[mask & (density > 0.0)]
    upper = float(np.max(positive)) if positive.size else 1.0
    lower = max(float(display_threshold), upper * 1e-7, np.finfo(float).tiny)
    shown = np.ma.masked_where(
        (~mask) | (density < display_threshold) | (density <= 0.0), density
    )

    kind_counts: dict[str, int] = {}
    feature_colors: dict[str, tuple[float, float, float, float]] = {}
    plotted_objects: list[_GeometryFigureFeature | _CurveEventFigure] = [
        *features,
        *curve_events,
    ]
    for feature in sorted(plotted_objects, key=_feature_sort_key):
        ordinal = kind_counts.get(feature.kind, 0)
        kind_counts[feature.kind] = ordinal + 1
        feature_colors[feature.identifier] = _stable_color(feature.kind, ordinal)

    def draw_mask_contour(axis, region: np.ndarray, color, style: str, width: float):
        if np.any(region) and np.any(~region):
            axis.contour(
                m1,
                m2,
                region.T.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=[width],
                linestyles=[style],
                zorder=6,
            )

    def draw_background(axis) -> None:
        axis.pcolormesh(
            m1,
            m2,
            shown.T,
            shading="auto",
            cmap="Greys",
            norm=LogNorm(vmin=lower, vmax=upper),
            alpha=0.52,
            rasterized=True,
            zorder=0,
        )

    def configure_spatial_axis(axis, title: str) -> None:
        axis.set_title(title, fontsize=10)
        axis.set_xlim(m1[0], m1[-1])
        axis.set_ylim(m2[0], m2[-1])
        axis.set_xlabel(coordinate1_label)
        axis.set_ylabel(coordinate2_label)
        axis.set_facecolor("#f2f2f2")
        if geometry == "log":
            axis.set_xscale("log")
            axis.set_yscale("log")

    def draw_point_panel(axis, kinds: set[str], title: str) -> None:
        draw_background(axis)
        selected = [feature for feature in features if feature.kind in kinds]
        if not selected:
            axis.text(
                0.5,
                0.5,
                "No retained features",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )
        for feature in selected:
            color = feature_colors[feature.identifier]
            probability = _masked_probability_for_plot(
                feature.region_inclusion_unconditional, mask
            )
            for level, alpha in ((0.10, 0.055), (0.50, 0.095), (0.90, 0.16)):
                if float(np.max(probability)) >= level:
                    axis.contourf(
                        m1,
                        m2,
                        probability.T,
                        levels=[level, 1.000001],
                        colors=[to_rgba(color, alpha)],
                        antialiased=True,
                        zorder=2,
                    )
            if feature.reference_region is not None:
                draw_mask_contour(
                    axis,
                    feature.reference_region,
                    color,
                    style=":",
                    width=0.85,
                )
            draw_mask_contour(
                axis,
                feature.location_hpd_credible,
                color,
                style="--",
                width=1.05,
            )
            draw_mask_contour(
                axis,
                feature.location_hpd_50,
                color,
                style="-",
                width=1.15,
            )
            marker = "^" if feature.kind == "peak" else "v"
            axis.scatter(
                feature.center_mass[0, 0],
                feature.center_mass[0, 1],
                marker=marker,
                s=48,
                facecolor=color,
                edgecolor="white",
                linewidth=0.75,
                zorder=9,
            )
            label_position = feature.center_mass[0]
            suffix = "†" if feature.boundary_limited else ""
            axis.annotate(
                feature.identifier + suffix,
                label_position,
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
                fontsize=7,
                weight="bold",
                zorder=10,
            )
        configure_spatial_axis(axis, title)

    def arm_short(identifier: str) -> str:
        return identifier[0] + identifier.rsplit("D", 1)[-1]

    def fill_corridor(axis, lower_curve, upper_curve, color, alpha, zorder) -> None:
        polygon = np.concatenate([upper_curve, lower_curve[::-1]], axis=0)
        axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=to_rgba(color, alpha),
            edgecolor="none",
            zorder=zorder,
        )

    def draw_curve_event_panel(axis, kind: str, title: str) -> None:
        draw_background(axis)
        selected = [event for event in curve_events if event.kind == kind]
        if not selected:
            axis.text(
                0.5,
                0.5,
                "No paired events",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )
        offsets = ((5, 5), (5, -13), (-38, 5), (-38, -13))
        for ordinal, event in enumerate(selected):
            color = feature_colors[event.identifier]
            fill_corridor(
                axis,
                event.pointwise_credible_lower_mass,
                event.pointwise_credible_upper_mass,
                color,
                alpha=0.16,
                zorder=2,
            )
            fill_corridor(
                axis,
                event.pointwise_50_lower_mass,
                event.pointwise_50_upper_mass,
                color,
                alpha=0.30,
                zorder=3,
            )
            for curve, style, width in (
                (event.pointwise_credible_lower_mass, ":", 0.95),
                (event.pointwise_credible_upper_mass, ":", 0.95),
                (event.pointwise_50_lower_mass, "-", 0.85),
                (event.pointwise_50_upper_mass, "-", 0.85),
            ):
                axis.plot(
                    curve[:, 0],
                    curve[:, 1],
                    color=color,
                    lw=width,
                    ls=style,
                    zorder=5,
                )
            axis.plot(
                event.median_curve_mass[:, 0],
                event.median_curve_mass[:, 1],
                color=color,
                lw=2.1,
                ls="-" if kind == "ridge" else "--",
                zorder=7,
            )
            center = event.median_curve_mass[len(event.median_curve_mass) // 2]
            arms = "+".join(arm_short(item) for item in event.arm_identifiers)
            suffix = "†" if event.boundary_limited else ""
            axis.annotate(
                f"{event.identifier}{suffix} ({arms})",
                center,
                xytext=offsets[ordinal % len(offsets)],
                textcoords="offset points",
                color=color,
                fontsize=7,
                weight="bold",
                zorder=9,
            )
        configure_spatial_axis(axis, title)

    # Boundary pits commonly have enormous half-depth regions that hide the
    # interior peak geometry. Keep their confidence in the exact bar panel and
    # reserve the spatial point panel for the scientifically useful peak windows.
    draw_point_panel(point_axis, {"peak"}, "Peaks")
    draw_curve_event_panel(ridge_axis, "ridge", "Paired ridge events")
    draw_curve_event_panel(valley_axis, "valley", "Paired valley events")

    ordered_features = sorted(plotted_objects, key=_feature_sort_key)
    positions = np.arange(len(ordered_features), dtype=float)
    for position, feature in zip(positions, ordered_features):
        color = feature_colors[feature.identifier]
        support_axis.barh(
            position,
            feature.support,
            height=0.66,
            color=to_rgba(color, 0.28),
            edgecolor="none",
            zorder=2,
        )
        support_axis.barh(
            position,
            feature.geometry_support,
            height=0.32,
            color=to_rgba(color, 0.88),
            edgecolor="none",
            zorder=3,
        )
        if isinstance(feature, _CurveEventFigure):
            support_axis.scatter(
                feature.curve_support,
                position,
                marker="D",
                s=18,
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
                zorder=4,
            )
            inflation = (
                f"  S×{feature.simultaneous_inflation:.2g}"
                if np.isfinite(feature.simultaneous_inflation)
                else ""
            )
            value_label = (
                f"{100.0 * feature.support:.0f}/"
                f"{100.0 * feature.curve_support:.0f}/"
                f"{100.0 * feature.geometry_support:.0f}{inflation}"
            )
        else:
            value_label = (
                f"{100.0 * feature.support:.0f}/"
                f"{100.0 * feature.geometry_support:.0f}"
            )
        support_axis.text(
            1.015,
            position,
            value_label,
            va="center",
            ha="left",
            fontsize=7,
        )
    support_axis.set_yticks(positions)
    support_axis.set_yticklabels(
        [
            (
                feature.identifier
                + ("†" if feature.boundary_limited else "")
                + (
                    " ("
                    + "+".join(
                        arm_short(item) for item in feature.arm_identifiers
                    )
                    + ")"
                    if isinstance(feature, _CurveEventFigure)
                    else ""
                )
            )
            for feature in ordered_features
        ],
        fontsize=7,
    )
    support_axis.invert_yaxis()
    support_axis.set_xlim(0.0, 1.28)
    support_axis.set_xticks((0.0, 0.25, 0.50, 0.75, 1.0))
    support_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    support_axis.grid(axis="x", alpha=0.22, lw=0.6)
    support_axis.axvline(0.5, color="0.45", lw=0.7, ls=":", zorder=1)
    support_axis.axvline(0.9, color="0.45", lw=0.7, ls="--", zorder=1)
    support_axis.set_title("Detection and finite-geometry confidence", fontsize=10)
    support_axis.set_xlabel(
        "posterior probability   "
        "(points: topology/geometry; events: topology/curve/geometry)"
    )
    credible_label = f"pointwise {100.0 * credible_mass:g}% curve corridor"
    figure.legend(
        handles=[
            Patch(
                facecolor=to_rgba("0.25", 0.11),
                label="peak-region inclusion (10/50/90%)",
            ),
            Line2D([], [], color="0.25", lw=1.15, label="peak 50% location HPD"),
            Line2D(
                [],
                [],
                color="0.25",
                lw=1.05,
                ls="--",
                label=f"peak {100.0 * credible_mass:g}% location HPD",
            ),
            Line2D(
                [],
                [],
                color="0.25",
                lw=0.85,
                ls=":",
                label="peak median-field width boundary",
            ),
            Patch(
                facecolor=to_rgba("0.25", 0.30),
                label="pointwise 50% curve corridor",
            ),
            Patch(facecolor=to_rgba("0.25", 0.16), label=credible_label),
            Line2D([], [], color="0.25", lw=2.1, label="posterior median event curve"),
            Patch(facecolor=to_rgba("0.25", 0.28), label="topology support"),
            Patch(
                facecolor=to_rgba("0.25", 0.88),
                label="finite geometry support",
            ),
            Line2D(
                [],
                [],
                marker="D",
                markerfacecolor="white",
                markeredgecolor="0.25",
                ls="",
                label="paired-curve support",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.047),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Peak shading is unconditional finite-region inclusion. Curve "
            "corridors are pointwise and conditional on both matched arms being "
            "available. S× is the inflation required for a simultaneous "
            f"{100.0 * credible_mass:g}% whole-curve band (not drawn). "
            "A/B labels are mutually exclusive posterior location configurations. "
            "† boundary-limited or truncated."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
    )

    written: list[Path] = []
    if settings.write_pdf:
        pdf = output_stem.with_suffix(".pdf")
        figure.savefig(pdf, dpi=220, bbox_inches="tight")
        written.append(pdf)
    if settings.write_png:
        png = output_stem.with_suffix(".png")
        figure.savefig(png, dpi=220, bbox_inches="tight")
        written.append(png)
    plt.close(figure)
    return written


def make_shoulder_geometry_figure(
    output_stem: Path,
    *,
    grid: Grid,
    summary: EnsembleSummary,
    shoulder_trackers: list[ShoulderTracker],
    weights: np.ndarray,
    credible_mass: float,
    settings: PlotSettings,
) -> list[Path]:
    """Visualize shoulder fronts, transition bands, and support hierarchy."""

    if not shoulder_trackers:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure, (map_axis, support_axis) = plt.subplots(
        1, 2, figsize=(11.0, max(4.7, 0.58 * len(shoulder_trackers) + 3.0)),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
    )
    density = summary.reference_density
    positive = density[grid.mask & (density > 0.0)]
    display_threshold = hpd_threshold(density, settings.display_probability, grid)
    shown = np.ma.masked_where(
        (~grid.mask) | (density < display_threshold) | (density <= 0.0), density
    )
    upper = float(np.max(positive))
    lower = max(float(display_threshold), upper * 1.0e-7, np.finfo(float).tiny)
    map_axis.pcolormesh(
        grid.m1,
        grid.m2,
        shown.T,
        shading="auto",
        cmap="Greys",
        norm=LogNorm(vmin=lower, vmax=upper),
        rasterized=True,
    )
    tail = 0.5 * (1.0 - credible_mass)
    for ordinal, tracker in enumerate(shoulder_trackers):
        color = _stable_color("shoulder", ordinal)
        inclusion = tracker.region_weight_sum
        levels = [level for level in (0.10, 0.50, 0.90) if np.max(inclusion) >= level]
        if levels:
            map_axis.contour(
                grid.m1,
                grid.m2,
                inclusion.T,
                levels=levels,
                colors=[color],
                linewidths=[0.65, 1.0, 1.25][: len(levels)],
                linestyles=[":", "-", "--"][: len(levels)],
                alpha=0.75,
            )
        selected = tracker.present & tracker.curve_available
        if np.any(selected):
            selected_weights = weights[selected]
            selected_weights /= selected_weights.sum()
            lower_curve, median_curve, upper_curve = weighted_quantile(
                tracker.curves[selected],
                (tail, 0.5, 1.0 - tail),
                selected_weights,
            )
            lower_mass = _mass_coordinates(lower_curve, grid)
            median_mass = _mass_coordinates(median_curve, grid)
            upper_mass = _mass_coordinates(upper_curve, grid)
            polygon = np.vstack([lower_mass, upper_mass[::-1]])
            map_axis.fill(
                polygon[:, 0], polygon[:, 1], color=color, alpha=0.16, lw=0.0
            )
            map_axis.plot(
                median_mass[:, 0], median_mass[:, 1], color=color, lw=2.0
            )
        reference_center = _mass_coordinates(
            tracker.template.center_curve_geometry, grid
        )
        map_axis.plot(
            reference_center[:, 0],
            reference_center[:, 1],
            color=color,
            lw=1.0,
            ls="--",
        )
        for boundary in (
            tracker.template.onset_curve_geometry,
            tracker.template.end_curve_geometry,
        ):
            if boundary.size:
                boundary_mass = _mass_coordinates(boundary, grid)
                map_axis.plot(
                    boundary_mass[:, 0],
                    boundary_mass[:, 1],
                    color=color,
                    lw=0.9,
                    ls=":",
                )
        midpoint = reference_center[len(reference_center) // 2]
        map_axis.annotate(
            tracker.template.identifier,
            midpoint,
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=8,
            weight="bold",
        )

    if grid.geometry == "log":
        map_axis.set_xscale("log")
        map_axis.set_yscale("log")
    map_axis.set_xlim(grid.m1[0], grid.m1[-1])
    map_axis.set_ylim(grid.m2[0], grid.m2[-1])
    map_axis.set_xlabel(grid.coordinate1_label)
    map_axis.set_ylabel(grid.coordinate2_label)
    map_axis.set_title("Directional slope-change fronts")
    map_axis.grid(alpha=0.16, lw=0.5)

    y = np.arange(len(shoulder_trackers), dtype=float)
    identifiers = [tracker.template.identifier for tracker in shoulder_trackers]
    morph = np.asarray([float(np.sum(weights[t.present])) for t in shoulder_trackers])
    location = np.asarray(
        [float(np.sum(weights[t.present & t.curve_available])) for t in shoulder_trackers]
    )
    region = np.asarray(
        [float(np.sum(weights[t.present & t.region_valid])) for t in shoulder_trackers]
    )
    support_axis.barh(y, morph, height=0.66, color="#d7c2e2", label="morphology")
    support_axis.barh(y, location, height=0.45, color="#ae7fc4", label="front")
    support_axis.barh(y, region, height=0.24, color=_KIND_COLORS["shoulder"], label="finite region")
    for position, values in enumerate(zip(morph, location, region)):
        support_axis.text(
            min(1.02, max(values) + 0.018),
            position,
            "/".join(f"{100.0 * value:.0f}" for value in values),
            va="center",
            fontsize=8,
        )
    support_axis.set_yticks(y, identifiers)
    support_axis.invert_yaxis()
    support_axis.set_xlim(0.0, 1.15)
    support_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    support_axis.axvline(0.5, color="0.5", lw=0.7, ls=":")
    support_axis.axvline(0.9, color="0.5", lw=0.7, ls="--")
    support_axis.grid(axis="x", alpha=0.2, lw=0.6)
    support_axis.set_xlabel("posterior probability (morphology / front / finite region)")
    support_axis.set_title("Shoulder confidence hierarchy")
    support_axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.legend(
        handles=[
            Patch(color=_KIND_COLORS["shoulder"], alpha=0.16, label=f"pointwise {100 * credible_mass:g}% front corridor"),
            Line2D([], [], color=_KIND_COLORS["shoulder"], lw=2.0, label="posterior median front"),
            Line2D([], [], color=_KIND_COLORS["shoulder"], lw=1.0, ls=":", label="median-field onset/end"),
            Line2D([], [], color=_KIND_COLORS["shoulder"], lw=1.0, ls="--", label="median-field front / 90% inclusion"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    written: list[Path] = []
    if settings.write_pdf:
        pdf = output_stem.with_suffix(".pdf")
        figure.savefig(pdf, bbox_inches="tight")
        written.append(pdf)
    if settings.write_png:
        png = output_stem.with_suffix(".png")
        figure.savefig(png, dpi=220, bbox_inches="tight")
        written.append(png)
    plt.close(figure)
    return written


def make_feature_geometry_figure(
    output_stem: Path,
    *,
    grid: Grid,
    summary: EnsembleSummary,
    point_trackers: list[PointTracker],
    branch_trackers: list[BranchTracker],
    weights: np.ndarray,
    geometry_maps: dict[str, FeatureGeometryMaps],
    credible_mass: float,
    settings: PlotSettings,
    location_configurations: dict[
        str, tuple[tuple[str, np.ndarray], ...]
    ] | None = None,
) -> list[Path]:
    """Write the dedicated uncertainty and finite-feature-region figure."""

    features = _geometry_figure_features(
        grid=grid,
        point_trackers=point_trackers,
        weights=weights,
        geometry_maps=geometry_maps,
    )
    curve_events = _curve_event_features(
        _curve_arm_data_from_trackers(branch_trackers),
        weights=weights,
        credible_mass=credible_mass,
        typical_spacing=grid.typical_spacing,
        geometry=grid.geometry,
        log_base=grid.log_base,
        location_configurations=location_configurations,
    )
    threshold = hpd_threshold(
        summary.reference_density, settings.display_probability, grid
    )
    return _write_geometry_confidence_figure(
        output_stem,
        m1=grid.m1,
        m2=grid.m2,
        mask=grid.mask,
        density=summary.reference_density,
        display_threshold=threshold,
        geometry=grid.geometry,
        coordinate1_label=grid.coordinate1_label,
        coordinate2_label=grid.coordinate2_label,
        features=features,
        curve_events=curve_events,
        credible_mass=credible_mass,
        settings=settings,
    )


def make_feature_geometry_figure_from_results(
    results_path: Path,
    output_stem: Path,
    *,
    settings: PlotSettings,
) -> list[Path]:
    """Rebuild the geometry figure from a completed HDF5 run without reanalysis."""

    if h5py is None:
        raise RuntimeError("h5py is required to replot a completed HDF5 result.")

    def text_attribute(value) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    with h5py.File(results_path, "r") as result:
        coordinates = result["coordinates"]
        m1 = np.asarray(coordinates["m1"])
        m2 = np.asarray(coordinates["m2"])
        mask = np.asarray(coordinates["mask"], dtype=bool)
        coordinate1_label = text_attribute(
            coordinates.attrs.get("coordinate1_label", r"$m_1$")
        )
        coordinate2_label = text_attribute(
            coordinates.attrs.get("coordinate2_label", r"$m_2$")
        )
        weight1 = np.asarray(coordinates["quadrature_m1"])
        weight2 = np.asarray(coordinates["quadrature_m2"])
        density = np.asarray(result["maps/reference_density"])
        weights = np.asarray(result["posterior/weights"])
        geometry = text_attribute(result.attrs["analysis_geometry"])
        log_base = float(result.attrs["log_base"])
        credible_mass = float(result.attrs["credible_mass"])
        geometry1 = np.asarray(
            coordinates["log_m1"] if geometry == "log" else coordinates["m1"]
        )
        geometry2 = np.asarray(
            coordinates["log_m2"] if geometry == "log" else coordinates["m2"]
        )
        typical_spacing = float(
            np.sqrt(
                np.median(np.diff(geometry1)) * np.median(np.diff(geometry2))
            )
        )

        def to_mass(points: np.ndarray) -> np.ndarray:
            return np.power(log_base, points) if geometry == "log" else points

        features: list[_GeometryFigureFeature] = []
        curve_arms: list[_CurveArmFigureData] = []
        location_configurations: dict[
            str, tuple[tuple[str, np.ndarray], ...]
        ] = {}
        if "events" in result:
            for identifier, group in result["events"].items():
                if (
                    text_attribute(
                        group.attrs.get("location_configuration_status", "")
                    )
                    != "robust_multimodal"
                    or "location_configuration" not in group
                ):
                    continue
                labels = np.asarray(group["location_configuration"], dtype=int)
                names = tuple(
                    item
                    for item in text_attribute(
                        group.attrs.get("location_configuration_names", "")
                    ).split(",")
                    if item
                )
                location_configurations[str(identifier)] = tuple(
                    (label, labels == index) for index, label in enumerate(names)
                )
        for identifier, group in result["features"].items():
            kind = text_attribute(group.attrs["type"])
            if kind not in {"peak", "pit", "ridge", "valley"}:
                continue
            status = text_attribute(group.attrs.get("status", "interior"))
            boundary_limited = (
                "boundary" in status
                or "drainage" in status
                or "truncated" in status
                or "low_density" in status
            )
            if kind in {"ridge", "valley"}:
                curve_arms.append(
                    _CurveArmFigureData(
                        identifier=str(identifier),
                        kind=kind,
                        boundary_limited=boundary_limited,
                        reference_curve_geometry=np.asarray(
                            group["reference_curve_geometry"]
                        ),
                        present=np.asarray(group["present"], dtype=bool),
                        curve_available=np.asarray(
                            group["curve_available"], dtype=bool
                        ),
                        geometry_valid=np.asarray(
                            group["geometry_valid"], dtype=bool
                        ),
                        topology_start_geometry=np.asarray(
                            group["topology_endpoints_geometry"]
                        )[:, :2],
                        curves_geometry=np.asarray(group["curves_geometry"]),
                    )
                )
                continue

            present = np.asarray(group["present"], dtype=bool)
            locations = np.asarray(group["location_geometry"])
            if np.any(present):
                selected_weights = weights[present]
                selected_weights = selected_weights / selected_weights.sum()
                center_geometry = weighted_quantile(
                    locations[present], 0.5, selected_weights
                )
            else:
                center_geometry = np.full(2, np.nan)
            center_mass = to_mass(np.asarray(center_geometry)[None, :])
            reference_region = (
                np.asarray(group["reference_feature_region_mask"], dtype=bool)
                if "reference_feature_region_mask" in group
                else None
            )
            features.append(
                _GeometryFigureFeature(
                    identifier=str(identifier),
                    kind=kind,
                    support=float(group.attrs["support"]),
                    geometry_support=float(group.attrs["geometry_support"]),
                    boundary_limited=boundary_limited,
                    center_mass=center_mass,
                    location_hpd_50=np.asarray(group["location_hpd_50"], dtype=bool),
                    location_hpd_credible=np.asarray(
                        group["location_hpd_credible"], dtype=bool
                    ),
                    region_inclusion_unconditional=np.asarray(
                        group["region_inclusion_unconditional"]
                    ),
                    reference_region=reference_region,
                )
            )
    curve_events = _curve_event_features(
        curve_arms,
        weights=weights,
        credible_mass=credible_mass,
        typical_spacing=typical_spacing,
        geometry=geometry,
        log_base=log_base,
        location_configurations=location_configurations,
    )
    threshold = _array_hpd_threshold(
        density,
        mask,
        weight1,
        weight2,
        settings.display_probability,
    )
    return _write_geometry_confidence_figure(
        output_stem,
        m1=m1,
        m2=m2,
        mask=mask,
        density=density,
        display_threshold=threshold,
        geometry=geometry,
        coordinate1_label=coordinate1_label,
        coordinate2_label=coordinate2_label,
        features=sorted(features, key=_feature_sort_key),
        curve_events=curve_events,
        credible_mass=credible_mass,
        settings=settings,
    )
