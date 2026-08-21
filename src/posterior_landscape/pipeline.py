"""One-command orchestration and flat, human-readable outputs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import math
import multiprocessing
import os
import pickle
import shutil
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import __version__
from .association import (
    AssociationFeature,
    LocationConfigurationResult,
    analyze_h0_associations,
    detect_location_configurations,
    expand_location_configurations,
    make_h0_corner_figures,
    make_location_configuration_figures,
    location_configuration_masks,
    pairwise_indicator_associations,
)
from .config import AnalysisSettings, Settings, load_settings
from .ensemble import (
    BranchTracker,
    EnsembleSummary,
    EventTracker,
    FeatureGeometryMaps,
    PlateauTracker,
    PointTracker,
    PosteriorCatalogue,
    ShoulderTracker,
    add_scale_persistence,
    assign_reference_identifiers,
    compute_ensemble_summary,
    feature_geometry_maps,
    probability_maps,
    remove_corner_boundary_pits,
    retain_scale_persistent_reference,
    weighted_quantile,
)
from .io import (
    Grid,
    density_in_feature_measure,
    h5py,
    open_density_store,
    quadrature_weights,
    validate_store,
    write_json,
)
from .plotting import (
    make_feature_projection_figure,
    make_feature_geometry_figure,
    make_feature_geometry_figure_from_results,
    make_global_tail_figure,
    make_landscape_figure,
    make_shoulder_geometry_figure,
)
from .topology import (
    BranchFeature,
    FieldAnalysis,
    PlateauFeature,
    PointFeature,
    ShoulderFeature,
    analyze_field,
)

_WORKER_GRID: Grid | None = None
_WORKER_OPTIONS: dict[str, Any] | None = None

_MISSING_H0_WARNING = (
    "WARNING: No aligned top-level dataset named 'h0' was found. "
    "Continuing without H0 association analysis; this is valid for a "
    "fixed-H0 run."
)
_H0_DISABLED_MESSAGE = "H0 association analysis disabled by [association] enabled=off."


def _h0_unavailable_message(settings: Settings) -> str:
    if settings.association.enabled == "off":
        return "External-parameter association analysis is disabled."
    return (
        "WARNING: No aligned dataset named "
        f"'{settings.association.dataset}' was found. Continuing without "
        f"association analysis for {settings.association.parameter_name}."
    )


def _configure_worker(grid: Grid, options: dict[str, Any]) -> None:
    global _WORKER_GRID, _WORKER_OPTIONS
    _WORKER_GRID = grid
    _WORKER_OPTIONS = options


def _analyze_worker(item: tuple[int, np.ndarray]) -> tuple[int, FieldAnalysis]:
    if _WORKER_GRID is None or _WORKER_OPTIONS is None:  # pragma: no cover
        raise RuntimeError("Analysis worker was not initialized.")
    index, density = item
    feature_density = density_in_feature_measure(density, _WORKER_GRID)
    return index, analyze_field(feature_density, _WORKER_GRID, **_WORKER_OPTIONS)


def _density_measure_name(feature_measure: str) -> str:
    return "dm1_dm2" if feature_measure == "linear" else "dlogm1_dlogm2"


def _analysis_options(
    settings: AnalysisSettings, persistence_threshold: float
) -> dict[str, Any]:
    return {
        "scale": settings.scales[0],
        "persistence_threshold": persistence_threshold,
        "persistence_gap_min_log": settings.persistence_gap_min_log,
        "hessian_refinement": settings.hessian_refinement,
        "detect_plateaus": settings.detect_plateaus,
        "curve_points": settings.curve_points,
        "support_mass": settings.support_mass,
        "curve_smoothing_cells": settings.curve_smoothing_cells,
        "shoulder_alpha_threshold": settings.shoulder_alpha_threshold,
        "shoulder_scales": settings.scales,
        "detect_shoulders": settings.detect_shoulders,
    }


def _automatic_workers(requested: int | None) -> int:
    if requested is not None:
        return requested
    return max(1, min(4, os.cpu_count() or 1))


def _automatic_batch_size(requested: int | None, grid: Grid, workers: int) -> int:
    if requested is not None:
        return requested
    bytes_per_draw = max(1, int(np.prod(grid.shape)) * 8)
    memory_limited = max(1, (128 * 1024**2) // bytes_per_draw)
    return max(workers, min(64, memory_limited))


def _fingerprint(settings: Settings, input_shape: tuple[int, ...]) -> str:
    stat = settings.input.file.stat()
    digest = hashlib.sha256()
    digest.update(settings.source.read_bytes())
    digest.update(str(settings.input.file).encode())
    digest.update(str(input_shape).encode())
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    digest.update(__version__.encode())
    return digest.hexdigest()


def _configure_logging(directory: Path) -> logging.Logger:
    logger = logging.getLogger("posterior_landscape")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_logging(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


_GENERATED_NAMES = {
    ".posterior_landscape.checkpoint.pkl.gz",
    ".posterior_landscape.checkpoint.pkl.tmp",
    "catalogue.csv",
    "catalogue.txt",
    "curve_event_catalogue.csv",
    "curve_event_catalogue.txt",
    "shoulder_catalogue.csv",
    "shoulder_catalogue.txt",
    "shoulder_geometry.pdf",
    "shoulder_geometry.png",
    "feature_geometry.pdf",
    "feature_geometry.png",
    "feature_location_configurations.csv",
    "feature_location_configurations.txt",
    "feature_location_configurations.pdf",
    "feature_location_configurations.png",
    "feature_projections.pdf",
    "feature_projections.png",
    "feature_projections_m2.pdf",
    "feature_projections_m2.png",
    "global_tail_probabilities.pdf",
    "global_tail_probabilities.png",
    "global_tail_probability_curves.csv",
    "global_tail_scales.csv",
    "global_tail_scales.txt",
    "global_tail_scales_draws.npz",
    "h0_feature_associations.csv",
    "h0_feature_associations.txt",
    "external_parameter_associations.csv",
    "h0_feature_corner.pdf",
    "h0_feature_corner.png",
    "external_parameter_corner.pdf",
    "landscape.pdf",
    "landscape.png",
    "manifest.json",
    "manifest.json.partial",
    "posterior_intervals.csv",
    "results.h5",
    "results.h5.partial",
    "results.npz",
    "ridge_valley_diagnostics.csv",
    "ridge_valley_diagnostics.txt",
    "run.log",
    "settings.ini",
    "topology_frequencies.csv",
    "two_dimensional_feature_cooccurrence.csv",
    "two_dimensional_feature_cooccurrence.txt",
}

_GEOMETRY_FIGURE_SCHEMA = "3.0"
_LOCATION_CONFIGURATION_SCHEMA = "1.0"


def _geometry_figure_manifest() -> dict[str, Any]:
    """Describe the conditioning and correspondence used in the figure."""

    return {
        "schema_version": _GEOMETRY_FIGURE_SCHEMA,
        "peak_region_probability": "unconditional",
        "peak_location_regions": "conditional_on_topological_match",
        "curve_events": "two_reference_arms_paired_by_common_saddle",
        "multimodal_curve_events": "separate_location_configuration_corridors",
        "curve_corridors": (
            "pointwise_transverse_and_conditional_on_both_matched_curves"
        ),
        "simultaneous_band": "reported_as_corridor_inflation_factor",
        "inclusion_levels": [0.10, 0.50, 0.90],
    }


def _location_configuration_manifest(
    results: dict[str, LocationConfigurationResult],
) -> dict[str, Any]:
    multimodal = [
        identifier for identifier, result in results.items() if result.multimodal
    ]
    return {
        "schema_version": _LOCATION_CONFIGURATION_SCHEMA,
        "method": "stable_disconnected_90pct_log_centroid_hpd",
        "bandwidths_bins": list((1.00, 1.15, 1.40, 1.80, 2.20)),
        "minimum_component_probability": 0.02,
        "minimum_label_agreement": 0.95,
        "minimum_configuration_draws": 50,
        "multimodal_families": multimodal,
    }


def _clear_generated_outputs(directory: Path, protected: set[Path]) -> None:
    for name in _GENERATED_NAMES:
        path = directory / name
        if path.resolve() not in protected and path.is_file():
            path.unlink()


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    next_draw: int,
    catalogue: PosteriorCatalogue,
) -> None:
    temporary = path.with_suffix(".tmp")
    payload = {
        "version": __version__,
        "fingerprint": fingerprint,
        "next_draw": int(next_draw),
        "catalogue": catalogue,
    }
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _load_checkpoint(
    path: Path, fingerprint: str
) -> tuple[int, PosteriorCatalogue] | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if payload.get("version") != __version__:
        raise ValueError("The checkpoint was written by another package version.")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(
            "The existing checkpoint belongs to different input or settings. "
            "Choose another output directory or set overwrite=true."
        )
    return int(payload["next_draw"]), payload["catalogue"]


def _conditional_quantile(
    values: np.ndarray,
    present: np.ndarray,
    weights: np.ndarray,
    probabilities: float | Iterable[float],
) -> np.ndarray:
    selected = np.asarray(present, dtype=bool)
    if not np.any(selected):
        requested = np.atleast_1d(np.asarray(probabilities, dtype=float))
        shape = () if values.ndim == 1 else values.shape[1:]
        result = np.full((requested.size,) + shape, np.nan)
        return result[0] if np.ndim(probabilities) == 0 else result
    selected_weights = weights[selected]
    selected_weights = selected_weights / selected_weights.sum()
    return weighted_quantile(values[selected], probabilities, selected_weights)


def _geometry_to_mass(values: np.ndarray, grid: Grid) -> np.ndarray:
    return np.power(grid.log_base, values) if grid.geometry == "log" else values


def _tracker_support(
    tracker: PointTracker | BranchTracker | EventTracker | ShoulderTracker | PlateauTracker,
    weights: np.ndarray,
) -> float:
    return float(np.sum(weights[tracker.present]))


def _geometry_support(
    tracker: PointTracker | BranchTracker, weights: np.ndarray
) -> float:
    return float(np.sum(weights[tracker.present & tracker.geometry_valid]))


def _event_location_support(
    tracker: EventTracker | ShoulderTracker, weights: np.ndarray
) -> float:
    return _event_probability(tracker.present & tracker.curve_available, weights)


def _event_region_support(
    tracker: EventTracker | ShoulderTracker, weights: np.ndarray
) -> float:
    return _event_probability(tracker.present & tracker.region_valid, weights)


def _event_probability(event: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights[np.asarray(event, dtype=bool)]))


def _boundary_label(
    template: PointFeature | BranchFeature | ShoulderFeature | PlateauFeature,
) -> str:
    boundary_type = getattr(template, "boundary_type", "none")
    if boundary_type != "none":
        return str(boundary_type)
    return "yes" if template.boundary else "no"


def _feature_status(template: PointFeature | BranchFeature | PlateauFeature) -> str:
    if isinstance(template, BranchFeature):
        if template.kind == "ridge":
            return "topological_ridge"
        endpoint = template.extremum_boundary_type
        if "outer" in endpoint:
            base = "outer_drainage"
        elif "mask" in endpoint:
            base = "mask_boundary"
        else:
            base = "interior"
        if not template.curve_available:
            return f"{base}_low_density"
        if template.low_density_truncated:
            return f"{base}_truncated"
        return f"{base}_full"
    boundary = _boundary_label(template)
    return "interior" if boundary == "no" else f"{boundary}_boundary"


def _catalogue_records(
    points: list[PointTracker],
    branches: list[BranchTracker],
    plateaus: list[PlateauTracker],
    weights: np.ndarray,
    grid: Grid,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tracker in points:
        position = _conditional_quantile(
            tracker.location, tracker.present, weights, 0.5
        )
        position = _geometry_to_mass(np.asarray(position), grid)
        geometry_present = tracker.present & tracker.geometry_valid
        records.append(
            {
                "ID": tracker.template.identifier,
                "type": tracker.template.kind,
                "status": _feature_status(tracker.template),
                "m1": float(position[0]),
                "m2": float(position[1]),
                "Pmed(region)": float(
                    _conditional_quantile(
                        tracker.region_mass, tracker.present, weights, 0.5
                    )
                ),
                "Pmed(feature)": float(
                    _conditional_quantile(
                        tracker.feature_probability,
                        geometry_present,
                        weights,
                        0.5,
                    )
                ),
                "prominence": float(
                    _conditional_quantile(
                        tracker.persistence, tracker.present, weights, 0.5
                    )
                ),
                "support": _tracker_support(tracker, weights),
                "geometry_support": _geometry_support(tracker, weights),
                "width_major": float(
                    _conditional_quantile(
                        tracker.width_major, geometry_present, weights, 0.5
                    )
                ),
                "width_minor": float(
                    _conditional_quantile(
                        tracker.width_minor, geometry_present, weights, 0.5
                    )
                ),
                "orientation_deg": math.degrees(
                    float(
                        _conditional_quantile(
                            tracker.width_orientation,
                            geometry_present,
                            weights,
                            0.5,
                        )
                    )
                ),
                "width_median": math.nan,
                "valid_fraction": (
                    _geometry_support(tracker, weights)
                    / max(_tracker_support(tracker, weights), np.finfo(float).eps)
                ),
                "log_contrast": float(
                    _conditional_quantile(
                        tracker.log_contrast, geometry_present, weights, 0.5
                    )
                ),
                "match_ambiguity_probability": _event_probability(
                    tracker.present & tracker.match_ambiguous, weights
                ),
                "persist.": f"{tracker.template.scale_count}/{tracker.template.scale_total}",
                "parts": "-",
                "boundary": _boundary_label(tracker.template),
            }
        )
    for tracker in branches:
        endpoints = _conditional_quantile(
            tracker.topology_endpoints, tracker.present, weights, 0.5
        )
        endpoints = _geometry_to_mass(np.asarray(endpoints).reshape(2, 2), grid)
        geometry_present = tracker.present & tracker.geometry_valid
        records.append(
            {
                "ID": tracker.template.identifier,
                "type": tracker.template.kind,
                "status": _feature_status(tracker.template),
                "m1": float(endpoints[0, 0]),
                "m2": float(endpoints[0, 1]),
                "Pmed(region)": math.nan,
                "Pmed(feature)": float(
                    _conditional_quantile(
                        tracker.feature_probability,
                        geometry_present,
                        weights,
                        0.5,
                    )
                ),
                "prominence": float(
                    _conditional_quantile(
                        tracker.prominence, tracker.present, weights, 0.5
                    )
                ),
                "support": _tracker_support(tracker, weights),
                "geometry_support": _geometry_support(tracker, weights),
                "width_major": math.nan,
                "width_minor": math.nan,
                "orientation_deg": math.nan,
                "width_median": float(
                    _conditional_quantile(
                        tracker.width_median, geometry_present, weights, 0.5
                    )
                ),
                "valid_fraction": float(
                    _conditional_quantile(
                        tracker.valid_width_fraction,
                        tracker.present,
                        weights,
                        0.5,
                    )
                ),
                "log_contrast": float(
                    _conditional_quantile(
                        tracker.log_contrast, geometry_present, weights, 0.5
                    )
                ),
                "match_ambiguity_probability": _event_probability(
                    tracker.present & tracker.match_ambiguous, weights
                ),
                "persist.": f"{tracker.template.scale_count}/{tracker.template.scale_total}",
                "parts": (
                    f"{tracker.template.start_identifier or '-'}->"
                    f"{tracker.template.end_identifier or '-'}"
                ),
                "boundary": _boundary_label(tracker.template),
            }
        )
    for tracker in plateaus:
        position = _conditional_quantile(
            tracker.location, tracker.present, weights, 0.5
        )
        position = _geometry_to_mass(np.asarray(position), grid)
        records.append(
            {
                "ID": tracker.template.identifier,
                "type": tracker.template.kind,
                "status": _feature_status(tracker.template),
                "m1": float(position[0]),
                "m2": float(position[1]),
                "Pmed(region)": float(
                    _conditional_quantile(
                        tracker.probability_mass, tracker.present, weights, 0.5
                    )
                ),
                "Pmed(feature)": float(
                    _conditional_quantile(
                        tracker.probability_mass, tracker.present, weights, 0.5
                    )
                ),
                "prominence": abs(
                    float(
                        _conditional_quantile(
                            tracker.contrast, tracker.present, weights, 0.5
                        )
                    )
                ),
                "support": _tracker_support(tracker, weights),
                "geometry_support": _tracker_support(tracker, weights),
                "width_major": math.nan,
                "width_minor": math.nan,
                "orientation_deg": math.nan,
                "width_median": math.nan,
                "valid_fraction": 1.0,
                "log_contrast": math.nan,
                "match_ambiguity_probability": math.nan,
                "persist.": "-",
                "parts": str(int(np.count_nonzero(tracker.template.region_mask))),
                "boundary": _boundary_label(tracker.template),
            }
        )
    order = {
        "peak": 0,
        "pit": 1,
        "ridge": 2,
        "valley": 3,
        "plateau": 4,
        "depression_floor": 5,
    }
    return sorted(
        records, key=lambda item: (order[item["type"]], item["m1"], item["m2"])
    )


def _number(value: Any, width: int = 13) -> str:
    if value is None or not np.isfinite(float(value)):
        return f"{'-':>{width}}"
    return f"{float(value):>{width}.6g}"


def _write_catalogue(
    directory: Path,
    records: list[dict[str, Any]],
    grid: Grid,
) -> tuple[Path, Path]:
    fields = [
        "ID",
        "type",
        "status",
        "m1",
        "m2",
        "Pmed(region)",
        "Pmed(feature)",
        "prominence",
        "support",
        "geometry_support",
        "width_major",
        "width_minor",
        "orientation_deg",
        "width_median",
        "valid_fraction",
        "log_contrast",
        "match_ambiguity_probability",
        "persist.",
        "parts",
        "boundary",
    ]
    csv_path = directory / "catalogue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    text_path = directory / "catalogue.txt"
    coordinate1 = grid.coordinate1_name
    coordinate2 = grid.coordinate2_name
    lines = [
        "Fixed two-dimensional feature catalogue",
        (
            f"{'ID':>8}  {'type':>16}  {'status':>28}  {coordinate1:>13}  "
            f"{coordinate2:>13}  {'Pmed(feature)':>13}  {'prominence':>13}  "
            f"{'support':>9}  {'geom.P':>9}  {'width A':>11}  {'width B':>11}  "
            f"{'valid':>8}  {'boundary':>11}"
        ),
        "-" * 198,
    ]
    for record in records:
        if record["type"] == "peak":
            width_a = record["width_major"]
            width_b = record["width_minor"]
        elif record["type"] in {"ridge", "valley"}:
            width_a = record["width_median"]
            width_b = math.nan
        else:
            width_a = math.nan
            width_b = math.nan
        lines.append(
            f"{record['ID']:>8}  {record['type']:>16}  "
            f"{record['status']:>28}  "
            f"{_number(record['m1'])}  {_number(record['m2'])}  "
            f"{_number(record['Pmed(feature)'])}  {_number(record['prominence'])}  "
            f"{record['support']:>8.1%}  {record['geometry_support']:>8.1%}  "
            f"{_number(width_a, 11)}  "
            f"{_number(width_b, 11)}  "
            f"{record['valid_fraction']:>7.1%}  {record['boundary']:>11}"
        )
    lines.extend(
        [
            "",
            (
                "Prominence is density persistence for peaks/pits and branches, "
                "and collar contrast for plateaus/floors."
            ),
            (
                "Support is posterior probability of a successful match to the "
                "fixed median-field feature."
            ),
            (
                "geom.P is P(match and valid finite geometry). Width A/B are "
                "major/minor half-prominence widths for points; width A is the "
                "median transverse width for curves."
            ),
            (
                "Valid is the finite-geometry fraction among point matches, or "
                "the median valid arclength fraction for a ridge/valley."
            ),
            (
                "Boundary is 'mask' for an internal support edge, 'outer' for "
                "a numerical grid limit, or 'outer+mask' when both apply."
            ),
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, csv_path


def _write_branch_diagnostics(
    directory: Path,
    branches: list[BranchTracker],
    weights: np.ndarray,
    grid: Grid,
) -> tuple[Path, Path]:
    fields = [
        "ID",
        "type",
        "status",
        "saddle_m1",
        "saddle_m2",
        "extremum_m1",
        "extremum_m2",
        "curve_end_m1",
        "curve_end_m2",
        "extent",
        "full_extent",
        "retained_fraction",
        "prominence",
        "support",
        "geometry_support",
        "curve_support",
        "truncation_probability",
        "feature_probability",
        "width_left",
        "width_right",
        "width_median",
        "width_along_p10",
        "width_along_p90",
        "valid_width_fraction",
        "fallback_width_fraction",
        "match_distance",
        "match_margin",
        "match_ambiguity_probability",
        "log_contrast",
        "parts",
        "boundary",
    ]
    records: list[dict[str, Any]] = []
    for tracker in branches:
        topology_endpoints = _conditional_quantile(
            tracker.topology_endpoints, tracker.present, weights, 0.5
        )
        topology_endpoints = _geometry_to_mass(
            np.asarray(topology_endpoints).reshape(2, 2), grid
        )
        curve_present = tracker.present & tracker.curve_available
        curve_endpoints = _conditional_quantile(
            tracker.endpoints, curve_present, weights, 0.5
        )
        curve_endpoints = _geometry_to_mass(
            np.asarray(curve_endpoints).reshape(2, 2), grid
        )
        geometry_present = tracker.present & tracker.geometry_valid
        records.append(
            {
                "ID": tracker.template.identifier,
                "type": tracker.template.kind,
                "status": _feature_status(tracker.template),
                "saddle_m1": float(topology_endpoints[0, 0]),
                "saddle_m2": float(topology_endpoints[0, 1]),
                "extremum_m1": float(topology_endpoints[1, 0]),
                "extremum_m2": float(topology_endpoints[1, 1]),
                "curve_end_m1": float(curve_endpoints[1, 0]),
                "curve_end_m2": float(curve_endpoints[1, 1]),
                "extent": float(
                    _conditional_quantile(tracker.length, curve_present, weights, 0.5)
                ),
                "full_extent": float(
                    _conditional_quantile(
                        tracker.full_length, tracker.present, weights, 0.5
                    )
                ),
                "retained_fraction": float(
                    _conditional_quantile(
                        tracker.retained_fraction, tracker.present, weights, 0.5
                    )
                ),
                "prominence": float(
                    _conditional_quantile(
                        tracker.prominence, tracker.present, weights, 0.5
                    )
                ),
                "support": _tracker_support(tracker, weights),
                "geometry_support": _geometry_support(tracker, weights),
                "curve_support": _event_probability(curve_present, weights),
                "truncation_probability": _event_probability(
                    tracker.present & tracker.low_density_truncated, weights
                ),
                "feature_probability": float(
                    _conditional_quantile(
                        tracker.feature_probability,
                        geometry_present,
                        weights,
                        0.5,
                    )
                ),
                "width_left": float(
                    _conditional_quantile(
                        tracker.width_left, geometry_present, weights, 0.5
                    )
                ),
                "width_right": float(
                    _conditional_quantile(
                        tracker.width_right, geometry_present, weights, 0.5
                    )
                ),
                "width_median": float(
                    _conditional_quantile(
                        tracker.width_median, geometry_present, weights, 0.5
                    )
                ),
                "width_along_p10": float(
                    _conditional_quantile(
                        tracker.width_along_lower,
                        geometry_present,
                        weights,
                        0.5,
                    )
                ),
                "width_along_p90": float(
                    _conditional_quantile(
                        tracker.width_along_upper,
                        geometry_present,
                        weights,
                        0.5,
                    )
                ),
                "valid_width_fraction": float(
                    _conditional_quantile(
                        tracker.valid_width_fraction,
                        tracker.present,
                        weights,
                        0.5,
                    )
                ),
                "fallback_width_fraction": float(
                    _conditional_quantile(
                        tracker.fallback_width_fraction,
                        tracker.present,
                        weights,
                        0.5,
                    )
                ),
                "match_distance": float(
                    _conditional_quantile(
                        tracker.match_distance, tracker.present, weights, 0.5
                    )
                ),
                "match_margin": float(
                    _conditional_quantile(
                        tracker.match_margin,
                        tracker.present & np.isfinite(tracker.match_margin),
                        weights,
                        0.5,
                    )
                ),
                "match_ambiguity_probability": _event_probability(
                    tracker.present & tracker.match_ambiguous, weights
                ),
                "log_contrast": float(
                    _conditional_quantile(
                        tracker.log_contrast, geometry_present, weights, 0.5
                    )
                ),
                "parts": (
                    f"{tracker.template.start_identifier or '-'}->"
                    f"{tracker.template.end_identifier or '-'}"
                ),
                "boundary": _boundary_label(tracker.template),
            }
        )
    csv_path = directory / "ridge_valley_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    text_path = directory / "ridge_valley_diagnostics.txt"
    lines = [
        "Ridge and valley curve diagnostics",
        (
            f"{'ID':>8}  {'type':>8}  {'status':>28}  "
            f"{'saddle m1':>12}  {'saddle m2':>12}  "
            f"{'curve end1':>12}  {'curve end2':>12}  {'extent':>11}  "
            f"{'retained':>9}  {'support':>9}  {'curve P':>9}  "
            f"{'geom.P':>9}  {'width':>11}  {'valid':>8}  {'boundary':>11}"
        ),
        "-" * 210,
    ]
    for record in records:
        lines.append(
            f"{record['ID']:>8}  {record['type']:>8}  {record['status']:>28}  "
            f"{_number(record['saddle_m1'], 12)}  {_number(record['saddle_m2'], 12)}  "
            f"{_number(record['curve_end_m1'], 12)}  {_number(record['curve_end_m2'], 12)}  "
            f"{_number(record['extent'], 11)}  {record['retained_fraction']:>8.1%}  "
            f"{record['support']:>8.1%}  {record['curve_support']:>8.1%}  "
            f"{record['geometry_support']:>8.1%}  "
            f"{_number(record['width_median'], 11)}  "
            f"{record['valid_width_fraction']:>7.1%}  {record['boundary']:>11}"
        )
    lines.extend(
        [
            "",
            (
                "Support is P(topological match); curve P is P(match and at "
                "least one HPD-supported grid cell)."
            ),
            (
                "geom.P is P(match and at least two valid two-sided transverse "
                "width sections); widths are reported in physical mass units."
            ),
            (
                "fallback_width_fraction is the fraction of valid ridge sections "
                "using the merge-saddle level because no local side minimum exists."
            ),
            (
                "Valley extent and curve end use only the contiguous "
                "high-density segment from the saddle."
            ),
            "The CSV also retains the full topological extremum and full extent.",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, csv_path


def _write_event_catalogue(
    directory: Path,
    events: list[EventTracker],
    weights: np.ndarray,
    credible_mass: float,
) -> tuple[Path, Path]:
    """Write the scientific paired-event catalogue and conditional intervals."""

    alpha = 0.5 * (1.0 - credible_mass)
    probabilities = (alpha, 0.5, 1.0 - alpha)
    scalar_fields = (
        "mu1",
        "mu2",
        "region_probability",
        "deficit_probability",
        "relative_width_m1",
        "relative_width_m2",
        "extent",
        "full_extent",
        "retained_fraction",
        "bounded_fraction",
        "longest_bounded_fraction",
        "width_median",
        "fallback_fraction",
        "log_contrast",
        "relative_contrast",
        "match_distance",
        "match_margin",
    )
    fields = [
        "ID",
        "type",
        "arms",
        "projection_weight",
        "P_morph",
        "P_loc",
        "P_reg",
        "truncation_probability",
        "boundary_probability",
        "match_ambiguity_probability",
    ]
    for name in scalar_fields:
        fields.extend(f"{name}_{suffix}" for suffix in ("lower", "median", "upper"))

    records: list[dict[str, Any]] = []
    for tracker in events:
        valid = tracker.present & tracker.region_valid
        record: dict[str, Any] = {
            "ID": tracker.template.identifier,
            "type": f"{tracker.template.kind}_event",
            "arms": "+".join(tracker.template.arm_identifiers),
            "projection_weight": (
                "deficit" if tracker.template.kind == "valley" else "density"
            ),
            "P_morph": _tracker_support(tracker, weights),
            "P_loc": _event_location_support(tracker, weights),
            "P_reg": _event_region_support(tracker, weights),
            "truncation_probability": _event_probability(
                tracker.present & tracker.low_density_truncated, weights
            ),
            "boundary_probability": _event_probability(
                tracker.present & tracker.boundary, weights
            ),
            "match_ambiguity_probability": _event_probability(
                tracker.present & tracker.match_ambiguous, weights
            ),
        }
        values_by_name = {
            "mu1": tracker.mass_centroid[:, 0],
            "mu2": tracker.mass_centroid[:, 1],
            "region_probability": tracker.region_probability,
            "deficit_probability": tracker.deficit_probability,
            "relative_width_m1": tracker.relative_projected_widths[:, 0],
            "relative_width_m2": tracker.relative_projected_widths[:, 1],
            "extent": tracker.extent,
            "full_extent": tracker.full_extent,
            "retained_fraction": tracker.retained_fraction,
            "bounded_fraction": tracker.bounded_fraction,
            "longest_bounded_fraction": tracker.longest_bounded_fraction,
            "width_median": tracker.width_median,
            "fallback_fraction": tracker.fallback_fraction,
            "log_contrast": tracker.log_contrast,
            "relative_contrast": tracker.relative_contrast,
            "match_distance": tracker.match_distance,
            "match_margin": tracker.match_margin,
        }
        for name, values in values_by_name.items():
            selected = valid & np.isfinite(values)
            quantiles = _conditional_quantile(
                values, selected, weights, probabilities
            )
            for suffix, value in zip(("lower", "median", "upper"), quantiles):
                record[f"{name}_{suffix}"] = float(value)
        records.append(record)

    csv_path = directory / "curve_event_catalogue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    text_path = directory / "curve_event_catalogue.txt"
    lines = [
        "Paired ridge and valley event catalogue",
        (
            f"{'ID':>8}  {'type':>12}  {'arms':>13}  {'P morph':>8}  "
            f"{'P loc':>8}  {'P reg':>8}  {'mu1':>11}  {'mu2':>11}  "
            f"{'P(feature)':>11}  {'D(feature)':>11}  {'bounded':>8}  "
            f"{'longest':>8}  {'width':>10}"
        ),
        "-" * 150,
    ]
    for record in records:
        lines.append(
            f"{record['ID']:>8}  {record['type']:>12}  {record['arms']:>13}  "
            f"{record['P_morph']:>7.1%}  {record['P_loc']:>7.1%}  "
            f"{record['P_reg']:>7.1%}  {_number(record['mu1_median'], 11)}  "
            f"{_number(record['mu2_median'], 11)}  "
            f"{_number(record['region_probability_median'], 11)}  "
            f"{_number(record['deficit_probability_median'], 11)}  "
            f"{record['bounded_fraction_median']:>7.1%}  "
            f"{record['longest_bounded_fraction_median']:>7.1%}  "
            f"{_number(record['width_median_median'], 10)}"
        )
    lines.extend(
        [
            "",
            "P morph / P loc / P reg are paired topology, paired-curve, and finite-window support.",
            "Valley centroids and projections use the log-interpolated transverse deficit.",
            "Bounded is total measurable arclength; longest is its largest contiguous fraction.",
            "All numerical lower/median/upper intervals are in the CSV and conditional on P reg.",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, csv_path


def _write_shoulder_catalogue(
    directory: Path,
    shoulders: list[ShoulderTracker],
    weights: np.ndarray,
    credible_mass: float,
) -> tuple[Path, Path]:
    """Write directional shoulder support, geometry, and strength summaries."""

    tail = 0.5 * (1.0 - credible_mass)
    probabilities = (tail, 0.5, 1.0 - tail)
    scalar_fields = (
        "mu1",
        "mu2",
        "region_probability",
        "relative_width_m1",
        "relative_width_m2",
        "extent",
        "retained_fraction",
        "bounded_fraction",
        "longest_bounded_fraction",
        "width_median",
        "alpha_max",
        "slope_pre",
        "slope_post",
        "slope_contrast",
        "match_distance",
        "match_margin",
    )
    fields = [
        "ID",
        "type",
        "alpha_threshold",
        "P_morph",
        "P_loc",
        "P_reg",
        "truncation_probability",
        "boundary_probability",
        "match_ambiguity_probability",
        "scale_count",
        "scale_total",
    ]
    for name in scalar_fields:
        fields.extend(f"{name}_{suffix}" for suffix in ("lower", "median", "upper"))

    records: list[dict[str, Any]] = []
    for tracker in shoulders:
        location = tracker.present & tracker.curve_available
        valid = tracker.present & tracker.region_valid
        record: dict[str, Any] = {
            "ID": tracker.template.identifier,
            "type": "shoulder",
            "alpha_threshold": tracker.template.alpha_threshold,
            "P_morph": _tracker_support(tracker, weights),
            "P_loc": _event_location_support(tracker, weights),
            "P_reg": _event_region_support(tracker, weights),
            "truncation_probability": _event_probability(
                tracker.present & tracker.low_density_truncated, weights
            ),
            "boundary_probability": _event_probability(
                tracker.present & tracker.boundary, weights
            ),
            "match_ambiguity_probability": _event_probability(
                tracker.present & tracker.match_ambiguous, weights
            ),
            "scale_count": tracker.template.scale_count,
            "scale_total": tracker.template.scale_total,
        }
        values_by_name = {
            "mu1": tracker.mass_centroid[:, 0],
            "mu2": tracker.mass_centroid[:, 1],
            "region_probability": tracker.region_probability,
            "relative_width_m1": tracker.relative_projected_widths[:, 0],
            "relative_width_m2": tracker.relative_projected_widths[:, 1],
            "extent": tracker.extent,
            "retained_fraction": tracker.retained_fraction,
            "bounded_fraction": tracker.bounded_fraction,
            "longest_bounded_fraction": tracker.longest_bounded_fraction,
            "width_median": tracker.width_median,
            "alpha_max": tracker.alpha_max,
            "slope_pre": tracker.slope_pre,
            "slope_post": tracker.slope_post,
            "slope_contrast": tracker.slope_contrast,
            "match_distance": tracker.match_distance,
            "match_margin": tracker.match_margin,
        }
        for name, values in values_by_name.items():
            conditioning = location if name in {"extent", "alpha_max", "match_distance", "match_margin"} else valid
            selected = conditioning & np.isfinite(values)
            quantiles = _conditional_quantile(values, selected, weights, probabilities)
            for suffix, value in zip(("lower", "median", "upper"), quantiles):
                record[f"{name}_{suffix}"] = float(value)
        records.append(record)

    csv_path = directory / "shoulder_catalogue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    text_path = directory / "shoulder_catalogue.txt"
    lines = [
        "Directional two-dimensional shoulder catalogue",
        (
            f"{'ID':>8}  {'P morph':>8}  {'P loc':>8}  {'P region':>8}  "
            f"{'mu1':>11}  {'mu2':>11}  {'P(feature)':>11}  "
            f"{'width':>11}  {'Delta S':>11}  {'persist.':>9}"
        ),
        "-" * 125,
    ]
    for record in records:
        lines.append(
            f"{record['ID']:>8}  {record['P_morph']:>7.1%}  "
            f"{record['P_loc']:>7.1%}  {record['P_reg']:>7.1%}  "
            f"{_number(record['mu1_median'], 11)}  "
            f"{_number(record['mu2_median'], 11)}  "
            f"{_number(record['region_probability_median'], 11)}  "
            f"{_number(record['width_median_median'], 11)}  "
            f"{_number(record['slope_contrast_median'], 11)}  "
            f"{record['scale_count']}/{record['scale_total']: <7}"
        )
    lines.extend(
        [
            "",
            "P morph is a compatible directional slope-change; P loc also requires a front.",
            "P region additionally requires finite onset and end fronts.",
            "Continuous summaries are conditional on their stated finite geometry.",
            "Delta S is the pre-transition minus post-transition directional log slope.",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, csv_path


def _write_location_configurations(
    directory: Path,
    features: list[AssociationFeature],
    results: dict[str, LocationConfigurationResult],
    weights: np.ndarray,
    credible_mass: float,
) -> tuple[Path, Path]:
    """Write finite automatic summaries of posterior location multimodality."""

    alpha = 0.5 * (1.0 - credible_mass)
    probabilities = (alpha, 0.5, 1.0 - alpha)
    scalar_names = (
        "mu1",
        "mu2",
        "region_probability",
        "deficit_probability",
        "relative_width_m1",
        "relative_width_m2",
        "contrast",
        "extent",
        "bounded_fraction",
        "fallback_fraction",
        "match_distance",
        "match_margin",
    )
    fields = [
        "family_ID",
        "configuration_ID",
        "configuration",
        "role",
        "type",
        "status",
        "P_family_region",
        "P_configuration",
        "P_configuration_given_region",
        "draw_count",
        "minimum_bandwidth_agreement",
        "match_ambiguity_probability",
    ]
    for name in scalar_names:
        fields.extend(f"{name}_{suffix}" for suffix in ("lower", "median", "upper"))

    records: list[dict[str, Any]] = []
    for feature in features:
        result = results[feature.identifier]
        family_probability = float(np.sum(weights[feature.region]))
        configurations = (
            tuple(enumerate(result.configuration_labels))
            if result.multimodal
            else ((0, ""),)
        )
        for index, label in configurations:
            selected = (
                result.labels == index
                if result.multimodal
                else np.asarray(feature.region, dtype=bool)
            )
            configuration_probability = float(np.sum(weights[selected]))
            configuration_id = (
                f"{feature.identifier}-{label}"
                if result.multimodal
                else feature.identifier
            )
            record: dict[str, Any] = {
                "family_ID": feature.identifier,
                "configuration_ID": configuration_id,
                "configuration": label,
                "role": (
                    "main" if result.multimodal and index == 0 else
                    "alternative" if result.multimodal else "single"
                ),
                "type": feature.kind,
                "status": result.status,
                "P_family_region": family_probability,
                "P_configuration": configuration_probability,
                "P_configuration_given_region": (
                    configuration_probability / family_probability
                    if family_probability > 0.0
                    else math.nan
                ),
                "draw_count": int(np.count_nonzero(selected)),
                "minimum_bandwidth_agreement": result.minimum_label_agreement,
                "match_ambiguity_probability": math.nan,
            }
            values = {
                "mu1": feature.mu1,
                "mu2": feature.mu2,
                "region_probability": feature.region_probability,
                "deficit_probability": feature.deficit_probability,
                "relative_width_m1": feature.relative_width_m1,
                "relative_width_m2": feature.relative_width_m2,
                "contrast": feature.contrast,
                "extent": feature.extent,
                "bounded_fraction": feature.bounded_fraction,
                "fallback_fraction": feature.fallback_fraction,
                "match_distance": feature.match_distance,
                "match_margin": feature.match_margin,
            }
            for name, source in values.items():
                source_values = (
                    np.asarray(source, dtype=float)
                    if source is not None
                    else np.full(weights.size, np.nan)
                )
                valid = selected & np.isfinite(source_values)
                interval = _conditional_quantile(
                    source_values, valid, weights, probabilities
                )
                for suffix, value in zip(("lower", "median", "upper"), interval):
                    record[f"{name}_{suffix}"] = float(value)
            if feature.match_ambiguous is not None and configuration_probability > 0.0:
                ambiguous = np.asarray(feature.match_ambiguous, dtype=bool)
                record["match_ambiguity_probability"] = float(
                    np.sum(weights[selected & ambiguous]) / configuration_probability
                )
            records.append(record)

    csv_path = directory / "feature_location_configurations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    text_path = directory / "feature_location_configurations.txt"
    multimodal_records = [
        record for record in records if record["status"] == "robust_multimodal"
    ]
    lines = [
        "Robust posterior feature-location configurations",
        (
            f"{'ID':>12}  {'type':>8}  {'role':>11}  {'P(all)':>8}  "
            f"{'P(|region)':>10}  {'mu1':>11}  {'mu2':>11}  "
            f"{'match d':>10}  {'ambig.':>8}"
        ),
        "-" * 105,
    ]
    if not multimodal_records:
        lines.append("No robust multimodal feature locations were detected.")
    for record in multimodal_records:
        lines.append(
            f"{record['configuration_ID']:>12}  {record['type']:>8}  "
            f"{record['role']:>11}  {record['P_configuration']:>7.1%}  "
            f"{record['P_configuration_given_region']:>9.1%}  "
            f"{_number(record['mu1_median'], 11)}  "
            f"{_number(record['mu2_median'], 11)}  "
            f"{_number(record['match_distance_median'], 10)}  "
            f"{record['match_ambiguity_probability']:>7.1%}"
        )
    lines.extend(
        [
            "",
            "A is the most probable configuration; later letters are alternatives.",
            "Configurations are mutually exclusive draw-level realizations, "
            "not coexisting features.",
            "Continuous intervals are conditional on that configuration's valid adaptive region.",
            "Unimodal and unstable families remain unsplit and are retained in the CSV.",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, csv_path


def _association_features(
    points: list[PointTracker],
    events: list[EventTracker],
    shoulders: list[ShoulderTracker],
) -> list[AssociationFeature]:
    result: list[AssociationFeature] = []
    for tracker in points:
        if tracker.template.kind == "pit" and not tracker.template.measurement_valid:
            continue
        region = tracker.present & tracker.measurement_valid
        result.append(
            AssociationFeature(
                identifier=tracker.template.identifier,
                kind=tracker.template.kind,
                morphology=tracker.present,
                location=tracker.present,
                region=region,
                mu1=tracker.mass_centroid[:, 0],
                mu2=tracker.mass_centroid[:, 1],
                region_probability=tracker.feature_probability,
                relative_width_m1=tracker.relative_projected_widths[:, 0],
                relative_width_m2=tracker.relative_projected_widths[:, 1],
                contrast=tracker.relative_prominence,
                extent=np.full(tracker.number_draws, np.nan),
                bounded_fraction=np.where(region, 1.0, np.nan),
                fallback_fraction=np.full(tracker.number_draws, np.nan),
                deficit_probability=tracker.deficit_probability,
                match_distance=tracker.match_distance,
                match_margin=tracker.match_margin,
                match_ambiguous=tracker.match_ambiguous,
            )
        )
    for tracker in events:
        result.append(
            AssociationFeature(
                identifier=tracker.template.identifier,
                kind=tracker.template.kind,
                morphology=tracker.present,
                location=tracker.present & tracker.curve_available,
                region=tracker.present & tracker.region_valid,
                mu1=tracker.mass_centroid[:, 0],
                mu2=tracker.mass_centroid[:, 1],
                region_probability=tracker.region_probability,
                relative_width_m1=tracker.relative_projected_widths[:, 0],
                relative_width_m2=tracker.relative_projected_widths[:, 1],
                contrast=tracker.relative_contrast,
                extent=tracker.extent,
                bounded_fraction=tracker.bounded_fraction,
                fallback_fraction=tracker.fallback_fraction,
                deficit_probability=tracker.deficit_probability,
                match_distance=tracker.match_distance,
                match_margin=tracker.match_margin,
                match_ambiguous=tracker.match_ambiguous,
            )
        )
    for tracker in shoulders:
        number = tracker.number_draws
        result.append(
            AssociationFeature(
                identifier=tracker.template.identifier,
                kind="shoulder",
                morphology=tracker.present,
                location=tracker.present & tracker.curve_available,
                region=tracker.present & tracker.region_valid,
                mu1=tracker.mass_centroid[:, 0],
                mu2=tracker.mass_centroid[:, 1],
                region_probability=tracker.region_probability,
                relative_width_m1=tracker.relative_projected_widths[:, 0],
                relative_width_m2=tracker.relative_projected_widths[:, 1],
                contrast=tracker.slope_contrast,
                extent=tracker.extent,
                bounded_fraction=tracker.bounded_fraction,
                fallback_fraction=np.full(number, np.nan),
                deficit_probability=np.full(number, np.nan),
                match_distance=tracker.match_distance,
                match_margin=tracker.match_margin,
                match_ambiguous=tracker.match_ambiguous,
            )
        )
    order = {"peak": 0, "pit": 1, "ridge": 2, "valley": 3, "shoulder": 4}
    return sorted(result, key=lambda item: (order[item.kind], item.identifier))


def _hdf5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _result_array(group: Any, name: str, number_draws: int) -> np.ndarray:
    if name not in group:
        raise ValueError(
            f"Completed results are missing '{group.name}/{name}', which is "
            "required for H0-only post-processing."
        )
    result = np.asarray(group[name])
    if result.ndim == 0 or result.shape[0] != number_draws:
        raise ValueError(
            f"Completed result '{group.name}/{name}' does not contain one "
            "entry per posterior draw."
        )
    return result


def _global_tail_association_feature(
    summary: EnsembleSummary,
) -> AssociationFeature:
    """Represent the configured marginal tail pair as one global mass scale."""

    number = summary.tail_m1_scale.size
    finite = np.isfinite(summary.tail_m1_scale) & np.isfinite(summary.tail_m2_scale)
    missing = np.full(number, np.nan)
    label = f"GT{100.0 * summary.tail_probability:g}"
    return AssociationFeature(
        identifier=label,
        kind="global_tail_scale",
        morphology=finite,
        location=finite,
        region=finite,
        mu1=summary.tail_m1_scale,
        mu2=summary.tail_m2_scale,
        region_probability=summary.tail_both_fraction_at_m1_scale,
        relative_width_m1=missing.copy(),
        relative_width_m2=missing.copy(),
        contrast=missing.copy(),
        extent=missing.copy(),
        bounded_fraction=np.where(finite, 1.0, np.nan),
        fallback_fraction=missing.copy(),
        deficit_probability=summary.tail_straddle_fraction_at_m1_scale,
        record_scope="global_tail_scale",
    )


def _association_features_from_results(
    path: Path, number_draws: int
) -> list[AssociationFeature]:
    """Reconstruct association inputs without rerunning any topology."""

    if h5py is None:  # pragma: no cover - h5py is a required dependency.
        raise RuntimeError("H0-only post-processing requires results.h5 and h5py.")
    result: list[AssociationFeature] = []
    with h5py.File(path, "r") as output:
        if "features" not in output or "events" not in output:
            raise ValueError(
                "Completed results.h5 predates the draw-adaptive feature archive "
                "needed for H0-only post-processing."
            )
        for identifier in sorted(output["features"]):
            group = output["features"][identifier]
            kind = _hdf5_text(group.attrs.get("type", ""))
            if kind not in {"peak", "pit"}:
                continue
            reference_measurement_valid = bool(
                group.attrs.get(
                    "reference_measurement_valid",
                    kind != "pit"
                    or _hdf5_text(
                        group.attrs.get("reference_boundary_type", "none")
                    )
                    == "none",
                )
            )
            if kind == "pit" and not reference_measurement_valid:
                continue
            present = _result_array(group, "present", number_draws).astype(bool)
            measurement_valid = _result_array(
                group, "measurement_valid", number_draws
            ).astype(bool)
            region = present & measurement_valid
            centroid = _result_array(group, "mass_centroid", number_draws)
            widths = _result_array(
                group, "relative_projected_widths", number_draws
            )
            result.append(
                AssociationFeature(
                    identifier=identifier,
                    kind=kind,
                    morphology=present,
                    location=present.copy(),
                    region=region,
                    mu1=centroid[:, 0],
                    mu2=centroid[:, 1],
                    region_probability=_result_array(
                        group, "feature_probability", number_draws
                    ),
                    relative_width_m1=widths[:, 0],
                    relative_width_m2=widths[:, 1],
                    contrast=_result_array(
                        group, "relative_prominence", number_draws
                    ),
                    extent=np.full(number_draws, np.nan),
                    bounded_fraction=np.where(region, 1.0, np.nan),
                    fallback_fraction=np.full(number_draws, np.nan),
                    deficit_probability=_result_array(
                        group, "deficit_probability", number_draws
                    ),
                    match_distance=_result_array(
                        group, "match_distance", number_draws
                    ),
                    match_margin=_result_array(group, "match_margin", number_draws),
                    match_ambiguous=_result_array(
                        group, "match_ambiguous", number_draws
                    ).astype(bool),
                )
            )

        for identifier in sorted(output["events"]):
            group = output["events"][identifier]
            kind = _hdf5_text(group.attrs.get("type", ""))
            if kind not in {"ridge", "valley"}:
                continue
            present = _result_array(group, "present", number_draws).astype(bool)
            curve_available = _result_array(
                group, "curve_available", number_draws
            ).astype(bool)
            region_valid = _result_array(
                group, "region_valid", number_draws
            ).astype(bool)
            centroid = _result_array(group, "mass_centroid", number_draws)
            widths = _result_array(
                group, "relative_projected_widths", number_draws
            )
            result.append(
                AssociationFeature(
                    identifier=identifier,
                    kind=kind,
                    morphology=present,
                    location=present & curve_available,
                    region=present & region_valid,
                    mu1=centroid[:, 0],
                    mu2=centroid[:, 1],
                    region_probability=_result_array(
                        group, "region_probability", number_draws
                    ),
                    relative_width_m1=widths[:, 0],
                    relative_width_m2=widths[:, 1],
                    contrast=_result_array(
                        group, "relative_contrast", number_draws
                    ),
                    extent=_result_array(group, "extent", number_draws),
                    bounded_fraction=_result_array(
                        group, "bounded_fraction", number_draws
                    ),
                    fallback_fraction=_result_array(
                        group, "fallback_fraction", number_draws
                    ),
                    deficit_probability=_result_array(
                        group, "deficit_probability", number_draws
                    ),
                    match_distance=_result_array(
                        group, "match_distance", number_draws
                    ),
                    match_margin=_result_array(group, "match_margin", number_draws),
                    match_ambiguous=_result_array(
                        group, "match_ambiguous", number_draws
                    ).astype(bool),
                )
            )
        if "shoulders" in output:
            for identifier in sorted(output["shoulders"]):
                group = output["shoulders"][identifier]
                present = _result_array(group, "present", number_draws).astype(bool)
                curve_available = _result_array(
                    group, "curve_available", number_draws
                ).astype(bool)
                region_valid = _result_array(
                    group, "region_valid", number_draws
                ).astype(bool)
                centroid = _result_array(group, "mass_centroid", number_draws)
                widths = _result_array(
                    group, "relative_projected_widths", number_draws
                )
                result.append(
                    AssociationFeature(
                        identifier=identifier,
                        kind="shoulder",
                        morphology=present,
                        location=present & curve_available,
                        region=present & region_valid,
                        mu1=centroid[:, 0],
                        mu2=centroid[:, 1],
                        region_probability=_result_array(
                            group, "region_probability", number_draws
                        ),
                        relative_width_m1=widths[:, 0],
                        relative_width_m2=widths[:, 1],
                        contrast=_result_array(
                            group, "slope_contrast", number_draws
                        ),
                        extent=_result_array(group, "extent", number_draws),
                        bounded_fraction=_result_array(
                            group, "bounded_fraction", number_draws
                        ),
                        fallback_fraction=np.full(number_draws, np.nan),
                        deficit_probability=np.full(number_draws, np.nan),
                        match_distance=_result_array(
                            group, "match_distance", number_draws
                        ),
                        match_margin=_result_array(
                            group, "match_margin", number_draws
                        ),
                        match_ambiguous=_result_array(
                            group, "match_ambiguous", number_draws
                        ).astype(bool),
                    )
                )
        if "global_tail" in output:
            tail = output["global_tail"]
            m1_scale = _result_array(tail, "m1_scale", number_draws)
            m2_scale = _result_array(tail, "m2_scale", number_draws)
            finite = np.isfinite(m1_scale) & np.isfinite(m2_scale)
            missing = np.full(number_draws, np.nan)
            probability = float(tail.attrs.get("probability", 0.999))
            result.append(
                AssociationFeature(
                    identifier=f"GT{100.0 * probability:g}",
                    kind="global_tail_scale",
                    morphology=finite,
                    location=finite,
                    region=finite,
                    mu1=m1_scale,
                    mu2=m2_scale,
                    region_probability=_result_array(
                        tail, "both_fraction_at_m1_scale", number_draws
                    ),
                    relative_width_m1=missing.copy(),
                    relative_width_m2=missing.copy(),
                    contrast=missing.copy(),
                    extent=missing.copy(),
                    bounded_fraction=np.where(finite, 1.0, np.nan),
                    fallback_fraction=missing.copy(),
                    deficit_probability=_result_array(
                        tail, "straddle_fraction_at_m1_scale", number_draws
                    ),
                    record_scope="global_tail_scale",
                )
            )
    order = {
        "peak": 0,
        "pit": 1,
        "ridge": 2,
        "valley": 3,
        "shoulder": 4,
        "global_tail_scale": 5,
    }
    return sorted(result, key=lambda item: (order[item.kind], item.identifier))


def _write_h0_associations(
    directory: Path, records: list[dict[str, Any]]
) -> tuple[Path]:
    """Write neutral public association columns; calculations remain unchanged."""

    csv_path = directory / "external_parameter_associations.csv"
    public_records = [
        {key.replace("h0", "parameter"): value for key, value in record.items()}
        for record in records
    ]
    fields = list(public_records[0]) if public_records else ["ID", "type"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_records)
    return (csv_path,)


def _write_pairwise_feature_associations(
    directory: Path,
    features: list[AssociationFeature],
    weights: np.ndarray,
    *,
    h0: np.ndarray | None,
    chain_id: np.ndarray | None,
    settings: Settings,
    stem: str = "two_dimensional_feature_cooccurrence",
) -> tuple[Path, Path]:
    """Write general pairwise location-support diagnostics."""

    records = pairwise_indicator_associations(
        [feature.identifier for feature in features],
        [feature.kind for feature in features],
        [feature.location for feature in features],
        weights,
        h0=h0,
        chain_id=chain_id,
        permutations=(settings.association.permutations if h0 is not None else 0),
        random_seed=settings.association.random_seed,
    )
    csv_path = directory / f"{stem}.csv"
    fields = list(records[0]) if records else ["ID_A", "type_A", "ID_B", "type_B"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    text_path = directory / f"{stem}.txt"
    lines = [
        "Pairwise feature location-support associations",
        (
            f"{'feature A':>12}  {'feature B':>12}  {'P(A&B)':>8}  "
            f"{'P(B|A)':>8}  {'P(A|B)':>8}  {'phi':>8}  {'lift':>8}  "
            f"{'H0 r_rb':>8}  {'p_perm':>8}"
        ),
        "-" * 112,
    ]
    for record in records:
        lines.append(
            f"{record['ID_A']:>12}  {record['ID_B']:>12}  "
            f"{record['P_A_and_B']:>8.1%}  {record['P_B_given_A']:>8.1%}  "
            f"{record['P_A_given_B']:>8.1%}  {_number(record['phi'], 8)}  "
            f"{_number(record['lift'], 8)}  "
            f"{_number(record['joint_occurrence_h0_rrb'], 8)}  "
            f"{_number(record['joint_occurrence_h0_p_perm'], 8)}"
        )
    lines.extend(
        [
            "",
            "Indicators are draw-level location support for the fixed catalogue families.",
            "P(A&B) is coexistence, phi is the Bernoulli correlation, and lift is",
            "P(A&B)/[P(A)P(B)]. These diagnostics do not automatically merge labels.",
            "When H0 is available, r_rb/p_perm compare joint occurrence with non-occurrence",
            "using the same within-chain circular-shift calibration as other H0 tests.",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, csv_path


def _write_global_tail_products(
    directory: Path,
    summary: EnsembleSummary,
    weights: np.ndarray,
    *,
    h0: np.ndarray | None,
    chain_id: np.ndarray | None,
    h0_record: dict[str, Any] | None,
) -> list[Path]:
    """Write the global any/both/straddling tail scales and curves."""

    probabilities = summary.probabilities
    m1_interval = weighted_quantile(summary.tail_m1_scale, probabilities, weights)
    m2_interval = weighted_quantile(summary.tail_m2_scale, probabilities, weights)
    difference = summary.tail_m1_scale - summary.tail_m2_scale
    ratio = summary.tail_m1_scale / summary.tail_m2_scale
    difference_interval = weighted_quantile(difference, probabilities, weights)
    ratio_interval = weighted_quantile(ratio, probabilities, weights)
    both_interval = weighted_quantile(
        summary.tail_both_fraction_at_m1_scale, probabilities, weights
    )
    straddle_interval = weighted_quantile(
        summary.tail_straddle_fraction_at_m1_scale, probabilities, weights
    )
    suffixes = ("lower", "median", "upper")
    record: dict[str, Any] = {
        "type": "global_tail_scale",
        "percentile": summary.tail_probability,
        "tail_probability": 1.0 - summary.tail_probability,
        "m1_boundary_limited_probability": float(
            np.sum(weights[np.asarray(summary.tail_m1_boundary_limited, dtype=bool)])
        ),
        "m2_boundary_limited_probability": float(
            np.sum(weights[np.asarray(summary.tail_m2_boundary_limited, dtype=bool)])
        ),
    }
    for name, values in (
        ("m1_scale", m1_interval),
        ("m2_scale", m2_interval),
        ("m1_minus_m2", difference_interval),
        ("m1_over_m2", ratio_interval),
        ("both_fraction_at_m1_scale", both_interval),
        ("straddle_fraction_at_m1_scale", straddle_interval),
    ):
        for suffix, value in zip(suffixes, values):
            record[f"{name}_{suffix}"] = float(value)
    if h0_record is not None:
        for name in (
            "mu1_rho",
            "mu1_mi_bits",
            "mu2_rho",
            "mu2_mi_bits",
            "mu_vector_mi_bits",
            "mu2_given_mu1_cmi_bits",
            "mu2_given_mu1_cmi_lower",
            "mu2_given_mu1_cmi_upper",
            "region_probability_rho",
            "region_probability_mi_bits",
            "deficit_probability_rho",
            "deficit_probability_mi_bits",
        ):
            record[f"h0_{name}"] = h0_record.get(name, math.nan)

    summary_path = directory / "global_tail_scales.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)

    curve_path = directory / "global_tail_probability_curves.csv"
    curve_fields = [
        "mass",
        "P_any_lower",
        "P_any_median",
        "P_any_upper",
        "P_both_lower",
        "P_both_median",
        "P_both_upper",
        "P_straddle_lower",
        "P_straddle_median",
        "P_straddle_upper",
    ]
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(curve_fields)
        for index, mass in enumerate(summary.tail_mass_grid):
            writer.writerow(
                [
                    float(mass),
                    *summary.tail_any_quantiles[:, index].tolist(),
                    *summary.tail_both_quantiles[:, index].tolist(),
                    *summary.tail_straddle_quantiles[:, index].tolist(),
                ]
            )

    draw_path = directory / "global_tail_scales_draws.npz"
    payload: dict[str, Any] = {
        "posterior_weights": np.asarray(weights, dtype=float),
        "percentile": np.asarray(summary.tail_probability),
        "m1_scale": summary.tail_m1_scale,
        "m2_scale": summary.tail_m2_scale,
        "P_both_at_m1_scale": summary.tail_both_at_m1_scale,
        "P_straddle_at_m1_scale": summary.tail_straddle_at_m1_scale,
        "both_fraction_at_m1_scale": summary.tail_both_fraction_at_m1_scale,
        "straddle_fraction_at_m1_scale": (
            summary.tail_straddle_fraction_at_m1_scale
        ),
        "m1_boundary_limited": summary.tail_m1_boundary_limited,
        "m2_boundary_limited": summary.tail_m2_boundary_limited,
    }
    if h0 is not None:
        payload["h0"] = np.asarray(h0, dtype=float)
    if chain_id is not None:
        payload["chain_id"] = np.asarray(chain_id)
    np.savez_compressed(draw_path, **payload)

    label = f"{100.0 * summary.tail_probability:g}%"
    text_path = directory / "global_tail_scales.txt"
    lines = [
        f"Global ordered-component tail scales ({label} quantile)",
        "",
        (
            f"m1: {m1_interval[1]:.6g} "
            f"[{m1_interval[0]:.6g}, {m1_interval[2]:.6g}]"
        ),
        (
            f"m2: {m2_interval[1]:.6g} "
            f"[{m2_interval[0]:.6g}, {m2_interval[2]:.6g}]"
        ),
        (
            "tail composition at the draw-specific m1 scale: "
            f"both={both_interval[1]:.1%} "
            f"[{both_interval[0]:.1%}, {both_interval[2]:.1%}], "
            f"straddling={straddle_interval[1]:.1%} "
            f"[{straddle_interval[0]:.1%}, {straddle_interval[2]:.1%}]"
        ),
        "",
        "P_any(M)=P(m1>M); P_both(M)=P(m2>M); ",
        "P_straddle(M)=P_any(M)-P_both(M).",
        "These are global tail scales, not local topological features or hard maxima.",
    ]
    if h0_record is not None:
        lines.extend(
            [
                "",
                (
                    "H0 association: "
                    f"rho(m1)={h0_record['mu1_rho']:.4g}, "
                    f"rho(m2)={h0_record['mu2_rho']:.4g}, "
                    f"I(vector)={h0_record['mu_vector_mi_bits']:.4g} bits, "
                    f"I(m2|m1)={h0_record['mu2_given_mu1_cmi_bits']:.4g} bits."
                ),
            ]
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [text_path, summary_path, curve_path, draw_path]


def _load_global_tail_summary(path: Path) -> Any | None:
    """Load only the saved tail arrays needed for H0-only refreshes."""

    if h5py is None:
        return None
    with h5py.File(path, "r") as output:
        if "global_tail" not in output:
            return None
        tail = output["global_tail"]
        posterior = output["posterior"]
        probabilities = tuple(
            float(value)
            for value in posterior.attrs.get(
                "quantile_probabilities", (0.05, 0.5, 0.95)
            )
        )
        return SimpleNamespace(
            probabilities=probabilities,
            tail_mass_grid=np.asarray(tail["mass_grid"]),
            tail_any_quantiles=np.asarray(tail["P_any_quantiles"]),
            tail_both_quantiles=np.asarray(tail["P_both_quantiles"]),
            tail_straddle_quantiles=np.asarray(tail["P_straddle_quantiles"]),
            tail_m1_scale=np.asarray(tail["m1_scale"]),
            tail_m2_scale=np.asarray(tail["m2_scale"]),
            tail_both_at_m1_scale=np.asarray(tail["P_both_at_m1_scale"]),
            tail_straddle_at_m1_scale=np.asarray(
                tail["P_straddle_at_m1_scale"]
            ),
            tail_both_fraction_at_m1_scale=np.asarray(
                tail["both_fraction_at_m1_scale"]
            ),
            tail_straddle_fraction_at_m1_scale=np.asarray(
                tail["straddle_fraction_at_m1_scale"]
            ),
            tail_m1_boundary_limited=np.asarray(tail["m1_boundary_limited"]),
            tail_m2_boundary_limited=np.asarray(tail["m2_boundary_limited"]),
            tail_probability=float(tail.attrs["probability"]),
        )


def _interval_record(
    identifier: str,
    kind: str,
    support: float,
    location: np.ndarray,
    scalar: np.ndarray,
    scalar_name: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ID": identifier,
        "type": kind,
        "support": support,
        "location_support": support,
        "measurement_support": math.nan,
        "geometry_support": math.nan,
        "m1_lower": location[0, 0],
        "m1_median": location[1, 0],
        "m1_upper": location[2, 0],
        "m2_lower": location[0, 1],
        "m2_median": location[1, 1],
        "m2_upper": location[2, 1],
        "quantity": scalar_name,
        "quantity_lower": scalar[0],
        "quantity_median": scalar[1],
        "quantity_upper": scalar[2],
    }
    for prefix in (
        "saddle_m1",
        "saddle_m2",
        "topology_end_m1",
        "topology_end_m2",
        "prominence",
        "density_value",
        "full_extent",
        "retained_fraction",
        "feature_probability",
        "width_major",
        "width_minor",
        "orientation_deg",
        "width_area",
        "log_contrast",
        "width_left",
        "width_right",
        "width_median",
        "width_along_p10",
        "width_along_p90",
        "valid_width_fraction",
        "fallback_width_fraction",
        "match_distance",
        "match_margin",
        "relative_prominence",
        "mu1",
        "mu2",
        "projected_m1_lower",
        "projected_m1_upper",
        "projected_m2_lower",
        "projected_m2_upper",
        "relative_projected_width_m1",
        "relative_projected_width_m2",
        "cov_m1_m1",
        "cov_m1_m2",
        "cov_m2_m2",
        "deficit_probability",
    ):
        for suffix in ("lower", "median", "upper"):
            record[f"{prefix}_{suffix}"] = math.nan
    record.update(
        {
            "curve_support": math.nan,
            "truncation_probability": math.nan,
            "outer_boundary_probability": math.nan,
            "mask_boundary_probability": math.nan,
            "match_ambiguity_probability": math.nan,
        }
    )
    return record


def _set_interval_triplet(
    record: dict[str, Any], prefix: str, values: np.ndarray
) -> None:
    for suffix, value in zip(("lower", "median", "upper"), values):
        record[f"{prefix}_{suffix}"] = float(value)


def _write_intervals(
    directory: Path,
    points: list[PointTracker],
    branches: list[BranchTracker],
    plateaus: list[PlateauTracker],
    weights: np.ndarray,
    grid: Grid,
    credible_mass: float,
) -> Path:
    alpha = 0.5 * (1.0 - credible_mass)
    q = (alpha, 0.5, 1.0 - alpha)
    records: list[dict[str, Any]] = []
    for tracker in points:
        location = _conditional_quantile(tracker.location, tracker.present, weights, q)
        location = _geometry_to_mass(location, grid)
        scalar = _conditional_quantile(tracker.region_mass, tracker.present, weights, q)
        record = _interval_record(
            tracker.template.identifier,
            tracker.template.kind,
            _tracker_support(tracker, weights),
            location,
            scalar,
            "region_probability",
        )
        geometry_present = tracker.present & tracker.geometry_valid
        measurement_present = tracker.present & tracker.measurement_valid
        record["geometry_support"] = _geometry_support(tracker, weights)
        record["measurement_support"] = _event_probability(
            measurement_present, weights
        )
        for prefix, values in (
            ("feature_probability", tracker.feature_probability),
            ("width_major", tracker.width_major),
            ("width_minor", tracker.width_minor),
            ("orientation_deg", np.degrees(tracker.width_orientation)),
            ("width_area", tracker.width_area),
            ("log_contrast", tracker.log_contrast),
        ):
            _set_interval_triplet(
                record,
                prefix,
                _conditional_quantile(values, geometry_present, weights, q),
            )
        _set_interval_triplet(
            record,
            "prominence",
            _conditional_quantile(tracker.persistence, tracker.present, weights, q),
        )
        _set_interval_triplet(
            record,
            "density_value",
            _conditional_quantile(tracker.value, tracker.present, weights, q),
        )
        point_measurements = (
            ("relative_prominence", tracker.relative_prominence),
            ("mu1", tracker.mass_centroid[:, 0]),
            ("mu2", tracker.mass_centroid[:, 1]),
            ("projected_m1_lower", tracker.projected_bounds_mass[:, 0]),
            ("projected_m1_upper", tracker.projected_bounds_mass[:, 1]),
            ("projected_m2_lower", tracker.projected_bounds_mass[:, 2]),
            ("projected_m2_upper", tracker.projected_bounds_mass[:, 3]),
            (
                "relative_projected_width_m1",
                tracker.relative_projected_widths[:, 0],
            ),
            (
                "relative_projected_width_m2",
                tracker.relative_projected_widths[:, 1],
            ),
            ("cov_m1_m1", tracker.spatial_covariance_mass[:, 0]),
            ("cov_m1_m2", tracker.spatial_covariance_mass[:, 1]),
            ("cov_m2_m2", tracker.spatial_covariance_mass[:, 2]),
            ("deficit_probability", tracker.deficit_probability),
        )
        for prefix, values in point_measurements:
            _set_interval_triplet(
                record,
                prefix,
                _conditional_quantile(values, measurement_present, weights, q),
            )
        record["outer_boundary_probability"] = _event_probability(
            tracker.present & tracker.outer_boundary, weights
        )
        record["mask_boundary_probability"] = _event_probability(
            tracker.present & tracker.mask_boundary, weights
        )
        _set_interval_triplet(
            record,
            "match_distance",
            _conditional_quantile(
                tracker.match_distance, tracker.present, weights, q
            ),
        )
        finite_margin = tracker.present & np.isfinite(tracker.match_margin)
        _set_interval_triplet(
            record,
            "match_margin",
            _conditional_quantile(
                tracker.match_margin, finite_margin, weights, q
            ),
        )
        record["match_ambiguity_probability"] = _event_probability(
            tracker.present & tracker.match_ambiguous, weights
        )
        records.append(record)
    for tracker in branches:
        curve_present = tracker.present & tracker.curve_available
        endpoints = _conditional_quantile(
            tracker.endpoints, curve_present, weights, q
        ).reshape(3, 2, 2)
        location = _geometry_to_mass(endpoints[:, 1, :], grid)
        scalar = _conditional_quantile(tracker.length, curve_present, weights, q)
        record = _interval_record(
            tracker.template.identifier,
            tracker.template.kind,
            _tracker_support(tracker, weights),
            location,
            scalar,
            "curve_extent",
        )
        geometry_present = tracker.present & tracker.geometry_valid
        record["geometry_support"] = _geometry_support(tracker, weights)
        record["location_support"] = _event_probability(curve_present, weights)
        record["measurement_support"] = record["geometry_support"]
        for prefix, values in (
            ("feature_probability", tracker.feature_probability),
            ("width_left", tracker.width_left),
            ("width_right", tracker.width_right),
            ("width_median", tracker.width_median),
            ("width_along_p10", tracker.width_along_lower),
            ("width_along_p90", tracker.width_along_upper),
            ("log_contrast", tracker.log_contrast),
        ):
            _set_interval_triplet(
                record,
                prefix,
                _conditional_quantile(values, geometry_present, weights, q),
            )
        _set_interval_triplet(
            record,
            "valid_width_fraction",
            _conditional_quantile(
                tracker.valid_width_fraction, tracker.present, weights, q
            ),
        )
        _set_interval_triplet(
            record,
            "fallback_width_fraction",
            _conditional_quantile(
                tracker.fallback_width_fraction, tracker.present, weights, q
            ),
        )
        topology_endpoints = _conditional_quantile(
            tracker.topology_endpoints, tracker.present, weights, q
        ).reshape(3, 2, 2)
        topology_endpoints = _geometry_to_mass(topology_endpoints, grid)
        _set_interval_triplet(record, "saddle_m1", topology_endpoints[:, 0, 0])
        _set_interval_triplet(record, "saddle_m2", topology_endpoints[:, 0, 1])
        _set_interval_triplet(
            record, "topology_end_m1", topology_endpoints[:, 1, 0]
        )
        _set_interval_triplet(
            record, "topology_end_m2", topology_endpoints[:, 1, 1]
        )
        _set_interval_triplet(
            record,
            "prominence",
            _conditional_quantile(tracker.prominence, tracker.present, weights, q),
        )
        _set_interval_triplet(
            record,
            "full_extent",
            _conditional_quantile(tracker.full_length, tracker.present, weights, q),
        )
        _set_interval_triplet(
            record,
            "retained_fraction",
            _conditional_quantile(
                tracker.retained_fraction, tracker.present, weights, q
            ),
        )
        record["curve_support"] = _event_probability(curve_present, weights)
        record["truncation_probability"] = _event_probability(
            tracker.present & tracker.low_density_truncated, weights
        )
        record["outer_boundary_probability"] = _event_probability(
            tracker.present & tracker.outer_boundary, weights
        )
        record["mask_boundary_probability"] = _event_probability(
            tracker.present & tracker.mask_boundary, weights
        )
        _set_interval_triplet(
            record,
            "match_distance",
            _conditional_quantile(
                tracker.match_distance, tracker.present, weights, q
            ),
        )
        finite_margin = tracker.present & np.isfinite(tracker.match_margin)
        _set_interval_triplet(
            record,
            "match_margin",
            _conditional_quantile(
                tracker.match_margin, finite_margin, weights, q
            ),
        )
        record["match_ambiguity_probability"] = _event_probability(
            tracker.present & tracker.match_ambiguous, weights
        )
        records.append(record)
    for tracker in plateaus:
        location = _conditional_quantile(tracker.location, tracker.present, weights, q)
        location = _geometry_to_mass(location, grid)
        scalar = _conditional_quantile(
            tracker.probability_mass, tracker.present, weights, q
        )
        record = _interval_record(
            tracker.template.identifier,
            tracker.template.kind,
            _tracker_support(tracker, weights),
            location,
            scalar,
            "region_probability",
        )
        record["measurement_support"] = _tracker_support(tracker, weights)
        _set_interval_triplet(
            record,
            "prominence",
            _conditional_quantile(
                np.abs(tracker.contrast), tracker.present, weights, q
            ),
        )
        records.append(record)
    path = directory / "posterior_intervals.csv"
    fields = (
        list(records[0])
        if records
        else list(
            _interval_record(
                "", "", math.nan, np.full((3, 2), math.nan), np.full(3, math.nan), ""
            )
        )
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path


def _write_topology_frequencies(
    directory: Path, catalogue: PosteriorCatalogue, weights: np.ndarray
) -> Path:
    path = directory / "topology_frequencies.csv"
    fields = [
        "peaks",
        "pits",
        "ridges",
        "valleys",
        "shoulders",
        "plateaus",
        "posterior_probability",
        "draw_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for signature, probability, count in catalogue.topology_frequencies(weights):
            writer.writerow(dict(zip(fields, (*signature, probability, count))))
    return path


def _dataset(group: Any, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.dtype.kind in {"U", "O"}:
        values = values.astype("S")
    options: dict[str, Any] = {}
    if values.ndim > 0 and values.size >= 128:
        options = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset(name, data=values, **options)


def _store_location_configuration(
    group: Any, result: LocationConfigurationResult
) -> None:
    """Store draw labels and conservative mode-detection diagnostics."""

    for name in (
        "location_configuration",
        "location_configuration_probability",
        "location_configuration_conditional_probability",
        "location_configuration_component_count",
    ):
        if name in group:
            del group[name]
    _dataset(group, "location_configuration", result.labels)
    _dataset(
        group,
        "location_configuration_probability",
        np.asarray(result.probabilities),
    )
    _dataset(
        group,
        "location_configuration_conditional_probability",
        np.asarray(result.conditional_probabilities),
    )
    _dataset(
        group,
        "location_configuration_component_count",
        np.asarray(result.component_counts, dtype=np.int16),
    )
    group.attrs["location_configuration_status"] = result.status
    group.attrs["location_configuration_count"] = len(
        result.configuration_labels
    )
    group.attrs["location_configuration_names"] = ",".join(
        result.configuration_labels
    )
    group.attrs["location_configuration_minimum_agreement"] = (
        result.minimum_label_agreement
    )


def _write_numerical_results(
    directory: Path,
    *,
    fingerprint: str,
    topology_sample_signature: str,
    grid: Grid,
    weights: np.ndarray,
    summary: EnsembleSummary,
    reference: FieldAnalysis,
    points: list[PointTracker],
    branches: list[BranchTracker],
    events: list[EventTracker],
    shoulders: list[ShoulderTracker],
    plateaus: list[PlateauTracker],
    ridge_probability: np.ndarray,
    valley_probability: np.ndarray,
    geometry_maps: dict[str, FeatureGeometryMaps],
    credible_mass: float,
    support_mass: float,
    curve_smoothing_cells: float,
    shoulder_alpha_threshold: float,
    h0: np.ndarray | None,
    chain_id: np.ndarray | None,
    h0_records: list[dict[str, Any]],
    location_configurations: dict[str, LocationConfigurationResult],
) -> Path:
    if h5py is None:  # Useful for lightweight source-tree testing.
        path = directory / "results.npz"
        payload: dict[str, Any] = dict(
            x1=grid.m1,
            x2=grid.m2,
            m1=grid.m1,
            m2=grid.m2,
            mask=grid.mask,
            posterior_weights=weights,
            reference_density=summary.reference_density,
            input_normalization=summary.normalization,
            feature_normalization=summary.feature_normalization,
            marginal1_quantiles=summary.marginal1_quantiles,
            marginal2_quantiles=summary.marginal2_quantiles,
            tail_mass_grid=summary.tail_mass_grid,
            tail_any_quantiles=summary.tail_any_quantiles,
            tail_both_quantiles=summary.tail_both_quantiles,
            tail_straddle_quantiles=summary.tail_straddle_quantiles,
            tail_m1_scale=summary.tail_m1_scale,
            tail_m2_scale=summary.tail_m2_scale,
            tail_both_at_m1_scale=summary.tail_both_at_m1_scale,
            tail_straddle_at_m1_scale=summary.tail_straddle_at_m1_scale,
            tail_both_fraction_at_m1_scale=(
                summary.tail_both_fraction_at_m1_scale
            ),
            tail_straddle_fraction_at_m1_scale=(
                summary.tail_straddle_fraction_at_m1_scale
            ),
            tail_m1_boundary_limited=summary.tail_m1_boundary_limited,
            tail_m2_boundary_limited=summary.tail_m2_boundary_limited,
            tail_probability=summary.tail_probability,
            ridge_probability=ridge_probability,
            valley_probability=valley_probability,
            credible_mass=credible_mass,
            support_mass=support_mass,
            persistence_threshold=reference.persistence_threshold,
            persistence_threshold_selection_mode=(
                reference.persistence_threshold_diagnostics.selection_mode
            ),
            persistence_gap_min_log=(
                reference.persistence_threshold_diagnostics.gap_min_log
            ),
            persistence_gap_resolved=(
                reference.persistence_threshold_diagnostics.gap_resolved
            ),
            persistence_fallback_used=(
                reference.persistence_threshold_diagnostics.fallback_used
            ),
            persistence_numerical_floor=(
                reference.persistence_threshold_diagnostics.numerical_floor
            ),
            persistence_terminal_gap_log=(
                np.nan
                if reference.persistence_threshold_diagnostics.terminal_gap_log
                is None
                else reference.persistence_threshold_diagnostics.terminal_gap_log
            ),
            persistence_selected_gap_log=(
                np.nan
                if reference.persistence_threshold_diagnostics.selected_gap_log
                is None
                else reference.persistence_threshold_diagnostics.selected_gap_log
            ),
            persistence_selected_gap_lower=(
                np.nan
                if reference.persistence_threshold_diagnostics.selected_gap_lower
                is None
                else reference.persistence_threshold_diagnostics.selected_gap_lower
            ),
            persistence_selected_gap_upper=(
                np.nan
                if reference.persistence_threshold_diagnostics.selected_gap_upper
                is None
                else reference.persistence_threshold_diagnostics.selected_gap_upper
            ),
            curve_smoothing_cells=curve_smoothing_cells,
            shoulder_alpha_threshold=shoulder_alpha_threshold,
            input_density_measure=grid.input_density_measure,
            density_measure=_density_measure_name(grid.feature_measure),
            feature_measure=grid.feature_measure,
            topology_sample_signature=topology_sample_signature,
        )
        if h0 is not None:
            payload["external_parameter"] = h0
            payload["h0"] = h0
            payload["chain_id"] = chain_id
        np.savez_compressed(path, **payload)
        return path

    path = directory / "results.h5"
    temporary = directory / "results.h5.partial"
    with h5py.File(temporary, "w") as output:
        output.attrs["schema_version"] = "2.1"
        output.attrs["package_version"] = __version__
        output.attrs["completed"] = True
        output.attrs["fingerprint"] = fingerprint
        output.attrs["topology_sample_signature"] = topology_sample_signature
        output.attrs["input_density_measure"] = grid.input_density_measure
        output.attrs["density_measure"] = _density_measure_name(
            grid.feature_measure
        )
        output.attrs["feature_measure"] = grid.feature_measure
        output.attrs["log_base"] = grid.log_base
        output.attrs["analysis_geometry"] = grid.geometry
        output.attrs["credible_mass"] = credible_mass
        output.attrs["support_mass"] = support_mass
        output.attrs["curve_smoothing_cells"] = curve_smoothing_cells
        output.attrs["shoulder_alpha_threshold"] = shoulder_alpha_threshold
        output.attrs["shoulder_direction"] = "d_dlnm1_plus_d_dlnm2"
        output.attrs["shoulder_window"] = (
            "positive_kappa_max_to_post_transition_kappa_zero"
        )
        output.attrs["persistence_threshold"] = reference.persistence_threshold
        threshold_diagnostics = reference.persistence_threshold_diagnostics
        output.attrs["persistence_gap_min_log"] = threshold_diagnostics.gap_min_log
        output.attrs["persistence_threshold_selection_mode"] = (
            threshold_diagnostics.selection_mode
        )
        output.attrs["persistence_threshold_gap_resolved"] = (
            threshold_diagnostics.gap_resolved
        )
        output.attrs["persistence_threshold_fallback_used"] = (
            threshold_diagnostics.fallback_used
        )
        output.attrs["persistence_threshold_numerical_floor"] = (
            threshold_diagnostics.numerical_floor
        )
        for name, value in (
            ("terminal_gap_log", threshold_diagnostics.terminal_gap_log),
            ("selected_gap_log", threshold_diagnostics.selected_gap_log),
            ("selected_gap_lower", threshold_diagnostics.selected_gap_lower),
            ("selected_gap_upper", threshold_diagnostics.selected_gap_upper),
        ):
            if value is not None:
                output.attrs[f"persistence_threshold_{name}"] = value
        output.attrs["width_definition"] = "half_prominence_or_half_depth"
        output.attrs["width_fraction"] = 0.5
        output.attrs["width_units"] = "coordinate"
        output.attrs["event_pairing"] = "common_saddle_two_arm"
        output.attrs["valley_measurement_weight"] = (
            "log_interpolated_side_shoulder_deficit"
        )
        output.attrs["adaptive_windows_may_overlap"] = True
        output.attrs["location_configuration_schema"] = "1.0"
        output.attrs["ordered_tails_enabled"] = summary.tail_mass_grid.size > 0
        if summary.tail_mass_grid.size > 0:
            output.attrs["global_tail_schema"] = "1.0"
            output.attrs["global_tail_probability"] = summary.tail_probability
        output.attrs["location_configuration_method"] = (
            "stable_disconnected_90pct_log_centroid_hpd"
        )
        output.attrs["external_association_available"] = h0 is not None
        output.attrs["h0_association_available"] = h0 is not None
        coordinates = output.create_group("coordinates")
        coordinates.attrs["coordinate1_name"] = grid.coordinate1_name
        coordinates.attrs["coordinate2_name"] = grid.coordinate2_name
        coordinates.attrs["coordinate1_label"] = grid.coordinate1_label
        coordinates.attrs["coordinate2_label"] = grid.coordinate2_label
        coordinates.attrs["coordinate1_unit"] = grid.coordinate1_unit
        coordinates.attrs["coordinate2_unit"] = grid.coordinate2_unit
        for name, values in (
            ("x1", grid.m1),
            ("x2", grid.m2),
            ("m1", grid.m1),
            ("m2", grid.m2),
            ("log_m1", grid.ell1),
            ("log_m2", grid.ell2),
            ("mask", grid.mask),
            ("quadrature_m1", grid.weight1),
            ("quadrature_m2", grid.weight2),
        ):
            _dataset(coordinates, name, values)
        posterior = output.create_group("posterior")
        for name, values in (
            ("weights", weights),
            ("input_normalization", summary.normalization),
            ("feature_normalization", summary.feature_normalization),
            ("marginal1_quantiles", summary.marginal1_quantiles),
            ("marginal2_quantiles", summary.marginal2_quantiles),
        ):
            _dataset(posterior, name, values)
        if h0 is not None:
            _dataset(posterior, "external_parameter", h0)
            _dataset(posterior, "h0", h0)
            _dataset(posterior, "chain_id", chain_id)
        posterior.attrs["quantile_probabilities"] = summary.probabilities
        if summary.tail_mass_grid.size > 0:
            tail = output.create_group("global_tail")
            tail.attrs["type"] = "global_tail_scale"
            tail.attrs["probability"] = summary.tail_probability
            tail.attrs["definitions"] = (
                "P_any=P(m1>M); P_both=P(m2>M); "
                "P_straddle=P_any-P_both"
            )
            for name, values in (
                ("mass_grid", summary.tail_mass_grid),
                ("P_any_quantiles", summary.tail_any_quantiles),
                ("P_both_quantiles", summary.tail_both_quantiles),
                ("P_straddle_quantiles", summary.tail_straddle_quantiles),
                ("m1_scale", summary.tail_m1_scale),
                ("m2_scale", summary.tail_m2_scale),
                ("P_both_at_m1_scale", summary.tail_both_at_m1_scale),
                ("P_straddle_at_m1_scale", summary.tail_straddle_at_m1_scale),
                (
                    "both_fraction_at_m1_scale",
                    summary.tail_both_fraction_at_m1_scale,
                ),
                (
                    "straddle_fraction_at_m1_scale",
                    summary.tail_straddle_fraction_at_m1_scale,
                ),
                ("m1_boundary_limited", summary.tail_m1_boundary_limited),
                ("m2_boundary_limited", summary.tail_m2_boundary_limited),
            ):
                _dataset(tail, name, values)
        maps = output.create_group("maps")
        for name, values in (
            ("reference_density", summary.reference_density),
            ("ridge_probability", ridge_probability),
            ("valley_probability", valley_probability),
        ):
            _dataset(maps, name, values)
        features = output.create_group("features")
        for tracker in points:
            group = features.create_group(tracker.template.identifier)
            group.attrs["type"] = tracker.template.kind
            group.attrs["status"] = _feature_status(tracker.template)
            group.attrs["reference_boundary_type"] = tracker.template.boundary_type
            group.attrs["reference_measurement_valid"] = (
                tracker.template.measurement_valid
            )
            group.attrs["support"] = _tracker_support(tracker, weights)
            group.attrs["morphology_support"] = _tracker_support(
                tracker, weights
            )
            group.attrs["location_support"] = _tracker_support(tracker, weights)
            group.attrs["region_support"] = _event_probability(
                tracker.present & tracker.measurement_valid, weights
            )
            group.attrs["geometry_support"] = _geometry_support(tracker, weights)
            group.attrs["scale_count"] = tracker.template.scale_count
            group.attrs["scale_total"] = tracker.template.scale_total
            group.attrs["width_reference"] = tracker.template.width_reference
            group.attrs["width_units"] = "mass"
            group.attrs["projection_weight"] = (
                "deficit" if tracker.template.kind == "pit" else "density"
            )
            group.attrs["relative_strength"] = "relative_prominence"
            if tracker.template.identifier in location_configurations:
                _store_location_configuration(
                    group, location_configurations[tracker.template.identifier]
                )
            for name, values in (
                ("present", tracker.present),
                ("location_geometry", tracker.location),
                ("persistence", tracker.persistence),
                ("region_probability", tracker.region_mass),
                ("density_value", tracker.value),
                ("boundary", tracker.boundary),
                ("outer_boundary", tracker.outer_boundary),
                ("mask_boundary", tracker.mask_boundary),
                ("base_level", tracker.base_level),
                ("half_level", tracker.half_level),
                ("feature_probability", tracker.feature_probability),
                ("width_major", tracker.width_major),
                ("width_minor", tracker.width_minor),
                ("width_orientation_radians", tracker.width_orientation),
                ("width_area", tracker.width_area),
                ("log_contrast", tracker.log_contrast),
                ("geometry_valid", tracker.geometry_valid),
                ("measurement_valid", tracker.measurement_valid),
                ("relative_prominence", tracker.relative_prominence),
                ("mass_centroid", tracker.mass_centroid),
                ("spatial_covariance_mass", tracker.spatial_covariance_mass),
                ("projected_bounds_mass", tracker.projected_bounds_mass),
                ("relative_projected_widths", tracker.relative_projected_widths),
                ("deficit_probability", tracker.deficit_probability),
                ("projection_m1", tracker.projection_m1),
                ("projection_m2", tracker.projection_m2),
                ("match_distance", tracker.match_distance),
                ("match_margin", tracker.match_margin),
                ("match_ambiguous", tracker.match_ambiguous),
            ):
                _dataset(group, name, values)
            if tracker.template.feature_region_mask is not None:
                _dataset(
                    group,
                    "reference_feature_region_mask",
                    tracker.template.feature_region_mask,
                )
            spatial = geometry_maps[tracker.template.identifier]
            for name, values in (
                ("location_probability", spatial.location_probability),
                ("location_density_mass_coordinates", spatial.location_density),
                ("location_hpd_50", spatial.location_hpd_50),
                ("location_hpd_credible", spatial.location_hpd_credible),
                (
                    "region_inclusion_conditional",
                    spatial.region_inclusion_conditional,
                ),
                (
                    "region_inclusion_unconditional",
                    spatial.region_inclusion_unconditional,
                ),
            ):
                _dataset(group, name, values)
        event_group = output.create_group("events")
        for tracker in events:
            group = event_group.create_group(tracker.template.identifier)
            group.attrs["type"] = tracker.template.kind
            group.attrs["arms"] = "+".join(tracker.template.arm_identifiers)
            group.attrs["projection_weight"] = (
                "deficit" if tracker.template.kind == "valley" else "density"
            )
            group.attrs["relative_strength"] = "relative_contrast"
            group.attrs["morphology_support"] = _tracker_support(tracker, weights)
            group.attrs["location_support"] = _event_location_support(tracker, weights)
            group.attrs["region_support"] = _event_region_support(tracker, weights)
            if tracker.template.identifier in location_configurations:
                _store_location_configuration(
                    group, location_configurations[tracker.template.identifier]
                )
            for name, values in (
                ("present", tracker.present),
                ("curve_available", tracker.curve_available),
                ("region_valid", tracker.region_valid),
                ("topology_geometry", tracker.topology_geometry),
                ("region_probability", tracker.region_probability),
                ("deficit_probability", tracker.deficit_probability),
                ("mass_centroid", tracker.mass_centroid),
                ("spatial_covariance_mass", tracker.spatial_covariance_mass),
                ("projected_bounds_mass", tracker.projected_bounds_mass),
                ("relative_projected_widths", tracker.relative_projected_widths),
                ("projection_m1", tracker.projection_m1),
                ("projection_m2", tracker.projection_m2),
                ("extent", tracker.extent),
                ("full_extent", tracker.full_extent),
                ("retained_fraction", tracker.retained_fraction),
                ("bounded_fraction", tracker.bounded_fraction),
                ("longest_bounded_fraction", tracker.longest_bounded_fraction),
                ("width_median", tracker.width_median),
                ("fallback_fraction", tracker.fallback_fraction),
                ("log_contrast", tracker.log_contrast),
                ("relative_contrast", tracker.relative_contrast),
                ("boundary", tracker.boundary),
                ("low_density_truncated", tracker.low_density_truncated),
                ("match_distance", tracker.match_distance),
                ("match_margin", tracker.match_margin),
                ("match_ambiguous", tracker.match_ambiguous),
                ("reference_topology_geometry", tracker.template.topology_geometry),
                ("reference_curve_geometry", tracker.template.curve_geometry),
            ):
                _dataset(group, name, values)
            _dataset(
                group,
                "region_inclusion_unconditional",
                tracker.region_weight_sum,
            )
            support = _event_region_support(tracker, weights)
            conditional = (
                tracker.region_weight_sum / support
                if support > 0.0
                else np.zeros_like(tracker.region_weight_sum)
            )
            _dataset(group, "region_inclusion_conditional", conditional)

        shoulder_group = output.create_group("shoulders")
        for tracker in shoulders:
            group = shoulder_group.create_group(tracker.template.identifier)
            group.attrs["type"] = "shoulder"
            group.attrs["alpha_threshold"] = tracker.template.alpha_threshold
            group.attrs["projection_weight"] = "density"
            group.attrs["relative_strength"] = "slope_contrast"
            group.attrs["morphology_support"] = _tracker_support(tracker, weights)
            group.attrs["location_support"] = _event_location_support(
                tracker, weights
            )
            group.attrs["region_support"] = _event_region_support(tracker, weights)
            group.attrs["scale_count"] = tracker.template.scale_count
            group.attrs["scale_total"] = tracker.template.scale_total
            if tracker.template.identifier in location_configurations:
                _store_location_configuration(
                    group, location_configurations[tracker.template.identifier]
                )
            for name, values in (
                ("present", tracker.present),
                ("curve_available", tracker.curve_available),
                ("region_valid", tracker.region_valid),
                ("topology_geometry", tracker.topology_geometry),
                ("curves_geometry", tracker.curves),
                ("region_probability", tracker.region_probability),
                ("mass_centroid", tracker.mass_centroid),
                ("spatial_covariance_mass", tracker.spatial_covariance_mass),
                ("projected_bounds_mass", tracker.projected_bounds_mass),
                ("relative_projected_widths", tracker.relative_projected_widths),
                ("projection_m1", tracker.projection_m1),
                ("projection_m2", tracker.projection_m2),
                ("extent", tracker.extent),
                ("full_extent", tracker.full_extent),
                ("retained_fraction", tracker.retained_fraction),
                ("bounded_fraction", tracker.bounded_fraction),
                ("longest_bounded_fraction", tracker.longest_bounded_fraction),
                ("width_median", tracker.width_median),
                ("alpha_max", tracker.alpha_max),
                ("slope_pre", tracker.slope_pre),
                ("slope_post", tracker.slope_post),
                ("slope_contrast", tracker.slope_contrast),
                ("boundary", tracker.boundary),
                ("low_density_truncated", tracker.low_density_truncated),
                ("match_distance", tracker.match_distance),
                ("match_margin", tracker.match_margin),
                ("match_ambiguous", tracker.match_ambiguous),
                (
                    "reference_topology_geometry",
                    tracker.template.topology_geometry,
                ),
                (
                    "reference_center_curve_geometry",
                    tracker.template.center_curve_geometry,
                ),
                (
                    "reference_onset_curve_geometry",
                    tracker.template.onset_curve_geometry,
                ),
                (
                    "reference_end_curve_geometry",
                    tracker.template.end_curve_geometry,
                ),
            ):
                _dataset(group, name, values)
            _dataset(
                group,
                "region_inclusion_unconditional",
                tracker.region_weight_sum,
            )
            support = _event_region_support(tracker, weights)
            conditional = (
                tracker.region_weight_sum / support
                if support > 0.0
                else np.zeros_like(tracker.region_weight_sum)
            )
            _dataset(group, "region_inclusion_conditional", conditional)

        if h0 is not None:
            association_group = output.create_group("h0_associations")
            for record in h0_records:
                group = association_group.create_group(str(record["ID"]))
                group.attrs["type"] = str(record["type"])
                for name, value in record.items():
                    if name in {"ID", "type"}:
                        continue
                    group.attrs[name] = value
        for tracker in branches:
            group = features.create_group(tracker.template.identifier)
            group.attrs["type"] = tracker.template.kind
            group.attrs["status"] = _feature_status(tracker.template)
            group.attrs["reference_boundary_type"] = tracker.template.boundary_type
            group.attrs["reference_extremum_boundary_type"] = (
                tracker.template.extremum_boundary_type
            )
            group.attrs["support"] = _tracker_support(tracker, weights)
            group.attrs["geometry_support"] = _geometry_support(tracker, weights)
            group.attrs["scale_count"] = tracker.template.scale_count
            group.attrs["scale_total"] = tracker.template.scale_total
            group.attrs["start_feature"] = tracker.template.start_identifier
            group.attrs["end_feature"] = tracker.template.end_identifier
            group.attrs["width_units"] = "mass"
            for name, values in (
                ("present", tracker.present),
                ("endpoints_geometry", tracker.endpoints),
                ("topology_endpoints_geometry", tracker.topology_endpoints),
                ("prominence", tracker.prominence),
                ("extent", tracker.length),
                ("full_extent", tracker.full_length),
                ("retained_fraction", tracker.retained_fraction),
                ("boundary", tracker.boundary),
                ("outer_boundary", tracker.outer_boundary),
                ("mask_boundary", tracker.mask_boundary),
                ("curve_available", tracker.curve_available),
                ("low_density_truncated", tracker.low_density_truncated),
                ("curves_geometry", tracker.curves),
                ("feature_probability", tracker.feature_probability),
                ("width_left", tracker.width_left),
                ("width_right", tracker.width_right),
                ("width_median", tracker.width_median),
                ("width_along_p10", tracker.width_along_lower),
                ("width_along_p90", tracker.width_along_upper),
                ("valid_width_fraction", tracker.valid_width_fraction),
                ("fallback_width_fraction", tracker.fallback_width_fraction),
                ("log_contrast", tracker.log_contrast),
                ("geometry_valid", tracker.geometry_valid),
                ("left_width_profiles", tracker.left_width_profiles),
                ("right_width_profiles", tracker.right_width_profiles),
                ("width_profiles", tracker.width_profiles),
                ("log_contrast_profiles", tracker.log_contrast_profiles),
                ("match_distance", tracker.match_distance),
                ("match_margin", tracker.match_margin),
                ("match_ambiguous", tracker.match_ambiguous),
            ):
                _dataset(group, name, values)
            _dataset(
                group,
                "reference_curve_geometry",
                tracker.template.points_geometry,
            )
            _dataset(
                group,
                "reference_topology_curve_geometry",
                tracker.template.topology_points_geometry,
            )
            _dataset(
                group,
                "width_sample_fraction",
                tracker.template.width_sample_fraction,
            )
            if tracker.template.feature_region_mask is not None:
                _dataset(
                    group,
                    "reference_feature_region_mask",
                    tracker.template.feature_region_mask,
                )
            spatial = geometry_maps[tracker.template.identifier]
            for name, values in (
                ("location_probability", spatial.location_probability),
                ("location_density_mass_coordinates", spatial.location_density),
                ("location_hpd_50", spatial.location_hpd_50),
                ("location_hpd_credible", spatial.location_hpd_credible),
                (
                    "region_inclusion_conditional",
                    spatial.region_inclusion_conditional,
                ),
                (
                    "region_inclusion_unconditional",
                    spatial.region_inclusion_unconditional,
                ),
            ):
                _dataset(group, name, values)
        for tracker in plateaus:
            group = features.create_group(tracker.template.identifier)
            group.attrs["type"] = tracker.template.kind
            group.attrs["status"] = _feature_status(tracker.template)
            group.attrs["support"] = _tracker_support(tracker, weights)
            for name, values in (
                ("present", tracker.present),
                ("location_geometry", tracker.location),
                ("area", tracker.area),
                ("region_probability", tracker.probability_mass),
                ("contrast", tracker.contrast),
                ("boundary", tracker.boundary),
                ("reference_region_mask", tracker.template.region_mask),
            ):
                _dataset(group, name, values)
    temporary.replace(path)
    return path


def _read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sample_draw_indices(number_draws: int, sample_draws: int = 16) -> np.ndarray:
    return np.unique(
        np.linspace(0, number_draws - 1, min(sample_draws, number_draws)).astype(
            int
        )
    )


def _topology_sample_signature(store: Any, grid: Grid) -> str:
    """Hash topology inputs at deterministic representative posterior draws."""

    digest = hashlib.sha256(b"posterior-landscape-topology-sample-v1")

    def update(values: np.ndarray, dtype: Any) -> None:
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())

    digest.update(str(store.shape).encode("ascii"))
    update(store.m1, "<f8")
    update(store.m2, "<f8")
    update(store.mask, np.uint8)
    update(store.weights, "<f8")
    indices = _sample_draw_indices(store.number_draws)
    update(indices, "<i8")
    for index in indices:
        update(store.draws[int(index)], "<f8")
    return digest.hexdigest()


def _release_series(version: Any) -> tuple[str, str] | None:
    pieces = _hdf5_text(version).split(".")
    return (pieces[0], pieces[1]) if len(pieces) >= 2 else None


def _completed_topology_compatibility(
    directory: Path,
    manifest: dict[str, Any],
    settings: Settings,
    store: Any,
    grid: Grid,
) -> tuple[bool, str, str]:
    """Check whether a completed same-release catalogue can be reused safely."""

    results_path = directory / "results.h5"
    settings_copy = directory / "settings.ini"
    if h5py is None or not results_path.is_file():
        return False, "", "completed results.h5 is unavailable"
    if not settings_copy.is_file():
        return False, "", "the completed run has no saved settings.ini"
    try:
        previous_settings = load_settings(settings_copy)
    except (OSError, ValueError):
        return False, "", "the completed INI settings cannot be read"
    previous_analysis = dict(vars(previous_settings.analysis))
    current_analysis = dict(vars(settings.analysis))
    previous_analysis.pop("run", None)
    current_analysis.pop("run", None)
    if previous_analysis != current_analysis:
        return False, "", "the two-dimensional analysis settings changed"
    try:
        previous_input = Path(str(manifest["input_file"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError):
        return False, "", "the completed manifest has no usable input path"
    if previous_input != settings.input.file.resolve():
        return False, "", "the input path differs from the completed run"
    if int(manifest.get("number_draws", -1)) != store.number_draws:
        return False, "", "the number of posterior draws changed"
    if list(manifest.get("grid_shape", [])) != list(grid.shape):
        return False, "", "the density grid shape changed"
    compatible_releases = {
        ("0", "5"),
        ("0", "6"),
        ("0", "7"),
        ("0", "8"),
    }
    if _release_series(manifest.get("package_version", "")) not in compatible_releases:
        return False, "", "the completed run is not topology-compatible"

    with h5py.File(results_path, "r") as output:
        schema_version = _hdf5_text(output.attrs.get("schema_version", ""))
        if schema_version != "2.1":
            return (
                False,
                "",
                "results.h5 predates saved m2 projections and global-tail arrays",
            )
        expected_attributes = {
            "feature_measure": grid.feature_measure,
            "analysis_geometry": grid.geometry,
        }
        for name, expected in expected_attributes.items():
            if _hdf5_text(output.attrs.get(name, "")) != expected:
                return False, "", f"results.h5 attribute '{name}' changed"
        if not np.isclose(
            float(output.attrs.get("log_base", np.nan)), grid.log_base
        ):
            return False, "", "the logarithm base changed"
        if "coordinates" not in output or "posterior" not in output:
            return False, "", "results.h5 lacks coordinates or posterior metadata"
        coordinates = output["coordinates"]
        for name, current in (
            ("m1", grid.m1),
            ("m2", grid.m2),
            ("mask", grid.mask),
        ):
            if name not in coordinates or not np.array_equal(
                np.asarray(coordinates[name]), current
            ):
                return False, "", f"the saved {name} grid changed"
        posterior = output["posterior"]
        if "weights" not in posterior or not np.allclose(
            np.asarray(posterior["weights"]),
            store.weights,
            rtol=1e-12,
            atol=1e-15,
        ):
            return False, "", "the posterior draw weights changed"
        if "input_normalization" not in posterior:
            return False, "", "saved input normalizations are unavailable"
        saved_normalization = np.asarray(posterior["input_normalization"])
        if saved_normalization.shape != (store.number_draws,):
            return False, "", "saved input normalizations have the wrong shape"
        stored_signature = manifest.get(
            "topology_sample_signature",
            output.attrs.get("topology_sample_signature", ""),
        )

    current_signature = _topology_sample_signature(store, grid)
    if stored_signature:
        if _hdf5_text(stored_signature) != current_signature:
            return False, "", "the sampled density fields or topology inputs changed"
    else:
        # Early archives did not store the representative density signature.
        # Their saved per-draw integrals provide a conservative compatibility
        # check; current outputs use the stronger signature above.
        input_cell_weights = (
            np.multiply.outer(
                quadrature_weights(grid.ell1), quadrature_weights(grid.ell2)
            )
            * grid.mask
        )
        indices = _sample_draw_indices(store.number_draws)
        current_normalization = np.asarray(
            [
                np.sum(
                    np.where(grid.mask, np.asarray(store.draws[int(index)]), 0.0)
                    * input_cell_weights
                )
                for index in indices
            ]
        )
        if not np.allclose(
            current_normalization,
            saved_normalization[indices],
            rtol=1e-9,
            atol=1e-12,
        ):
            return False, "", "the sampled density normalizations changed"
    return True, current_signature, ""


def _arrays_identical(first: np.ndarray, second: np.ndarray) -> bool:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        return False
    if first.dtype.kind in {"S", "U", "O"} or second.dtype.kind in {
        "S",
        "U",
        "O",
    }:
        return np.array_equal(first.astype(str), second.astype(str))
    return np.array_equal(first, second)


def _stored_h0_is_current(
    directory: Path,
    manifest: dict[str, Any],
    settings: Settings,
    h0: np.ndarray,
    chain_id: np.ndarray,
) -> bool:
    results_path = directory / "results.h5"
    association = manifest.get("h0_association", {})
    if not association.get("available"):
        return False
    configuration = manifest.get("location_configurations", {})
    if configuration.get("schema_version") != _LOCATION_CONFIGURATION_SCHEMA:
        return False
    expected_settings = {
        "knn": settings.association.knn,
        "permutations": settings.association.permutations,
        "uncertainty_resamples": settings.association.uncertainty_resamples,
        "random_seed": settings.association.random_seed,
    }
    if any(
        association.get(name) != value
        for name, value in expected_settings.items()
    ):
        return False
    # The compact profile deliberately removes the standalone location table;
    # its complete numerical content remains in results.h5.
    required_files = ("external_parameter_associations.csv",)
    if not all((directory / name).is_file() for name in required_files):
        return False
    if h5py is None or not results_path.is_file():
        return False
    with h5py.File(results_path, "r") as output:
        if (
            "posterior/h0" not in output
            or "posterior/chain_id" not in output
            or "h0_associations" not in output
        ):
            return False
        return _arrays_identical(np.asarray(output["posterior/h0"]), h0) and (
            _arrays_identical(np.asarray(output["posterior/chain_id"]), chain_id)
        )


def _h0_manifest(settings: Settings, chain_id: np.ndarray) -> dict[str, Any]:
    return {
        "available": True,
        "dataset": settings.association.dataset,
        "parameter_name": settings.association.parameter_name,
        "parameter_label": settings.association.parameter_label,
        "parameter_unit": settings.association.parameter_unit,
        "knn": settings.association.knn,
        "permutations": settings.association.permutations,
        "uncertainty_resamples": settings.association.uncertainty_resamples,
        "random_seed": settings.association.random_seed,
        "number_chains": int(np.unique(chain_id).size),
        "null_calibration": "within_chain_circular_h0_shift",
        "continuous_conditioning": (
            "valid_draw_adaptive_region_or_location_configuration"
        ),
        "multimodal_parent_continuous_summary": "omitted",
    }


def _replace_h0_results(
    path: Path,
    *,
    fingerprint: str,
    topology_sample_signature: str,
    h0: np.ndarray,
    chain_id: np.ndarray,
    records: list[dict[str, Any]],
    location_configurations: dict[str, LocationConfigurationResult],
) -> None:
    """Atomically add or replace H0-only products in completed results.h5."""

    if h5py is None:  # pragma: no cover - h5py is a required dependency.
        raise RuntimeError("H0-only post-processing requires h5py.")
    temporary = path.with_name("results.h5.partial")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copy2(path, temporary)
        with h5py.File(temporary, "r+") as output:
            posterior = output.require_group("posterior")
            for name, values in (("h0", h0), ("chain_id", chain_id)):
                if name in posterior:
                    del posterior[name]
                _dataset(posterior, name, values)
            if "h0_associations" in output:
                del output["h0_associations"]
            associations = output.create_group("h0_associations")
            for record in records:
                group = associations.create_group(str(record["ID"]))
                group.attrs["type"] = str(record["type"])
                for name, value in record.items():
                    if name not in {"ID", "type"}:
                        group.attrs[name] = value
            for identifier, result in location_configurations.items():
                if identifier in output.get("events", {}):
                    target = output["events"][identifier]
                elif identifier in output.get("shoulders", {}):
                    target = output["shoulders"][identifier]
                elif identifier in output.get("features", {}):
                    target = output["features"][identifier]
                else:  # pragma: no cover - guarded by result reconstruction.
                    continue
                _store_location_configuration(target, result)
            output.attrs["package_version"] = __version__
            output.attrs["fingerprint"] = fingerprint
            output.attrs["topology_sample_signature"] = topology_sample_signature
            output.attrs["h0_association_available"] = True
            output.attrs["location_configuration_schema"] = "1.0"
            output.attrs["location_configuration_method"] = (
                "stable_disconnected_90pct_log_centroid_hpd"
            )
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_manifest_atomic(directory: Path, manifest: dict[str, Any]) -> None:
    temporary = directory / "manifest.json.partial"
    write_json(temporary, manifest)
    temporary.replace(directory / "manifest.json")


def _replot_completed_geometry_if_needed(
    directory: Path,
    manifest: dict[str, Any],
    settings: Settings,
) -> None:
    """Add or refresh the geometry figure from a completed HDF5 run."""

    output = getattr(settings, "output", None)
    if getattr(output, "profile", "full") == "essential":
        # This diagnostic is intentionally absent from the compact profile.
        return
    expected: list[Path] = []
    if settings.plot.write_pdf:
        expected.append(directory / "feature_geometry.pdf")
    if settings.plot.write_png:
        expected.append(directory / "feature_geometry.png")
    if not expected:
        return
    missing = [path for path in expected if not path.is_file()]
    figure_metadata = manifest.get("geometry_figure", {})
    stale = figure_metadata.get("schema_version") != _GEOMETRY_FIGURE_SCHEMA
    if not missing and not stale:
        return
    results_path = directory / "results.h5"
    if not results_path.is_file():
        return
    if stale:
        print(
            "Writing the updated paired-event confidence figure from results.h5."
        )
    else:
        print("Writing the missing feature-geometry confidence figure from results.h5.")
    written = make_feature_geometry_figure_from_results(
        results_path,
        directory / "feature_geometry",
        settings=settings.plot,
    )
    outputs = list(manifest.get("outputs", []))
    for path in written:
        if path.name not in outputs:
            outputs.append(path.name)
    manifest["outputs"] = outputs
    manifest["geometry_figure"] = _geometry_figure_manifest()
    write_json(directory / "manifest.json", manifest)


def _run_h0_only_update(
    directory: Path,
    manifest: dict[str, Any],
    settings: Settings,
    store: Any,
    *,
    fingerprint: str,
    topology_sample_signature: str,
) -> Path:
    """Run H0 associations from saved draw-level features only."""

    if store.h0 is None or store.chain_id is None:  # pragma: no cover
        raise ValueError("H0-only post-processing requires aligned h0 samples.")
    results_path = directory / "results.h5"
    logger = _configure_logging(directory)
    try:
        logger.info(
            "Reusing the completed feature catalogue and draw-level geometry; "
            "posterior topology will not be recomputed."
        )
        saved_features = _association_features_from_results(
            results_path, store.number_draws
        )
        tail_features = [
            feature
            for feature in saved_features
            if feature.kind == "global_tail_scale"
        ]
        family_features = [
            feature
            for feature in saved_features
            if feature.kind != "global_tail_scale"
        ]
        with h5py.File(results_path, "r") as completed:
            weights = np.asarray(completed["posterior/weights"], dtype=float)
            credible_mass = float(completed.attrs.get("credible_mass", 0.90))
        location_configurations = detect_location_configurations(
            family_features, weights
        )
        features = expand_location_configurations(
            family_features, location_configurations
        )
        features.extend(tail_features)
        configuration_paths = list(
            _write_location_configurations(
                directory,
                family_features,
                location_configurations,
                weights,
                credible_mass,
            )
        )
        cooccurrence_paths = list(
            _write_pairwise_feature_associations(
                directory,
                family_features,
                weights,
                h0=store.h0,
                chain_id=store.chain_id,
                settings=settings,
            )
        )
        logger.info(
            "Computing %s associations for %d saved features "
            "(%d permutations; k=%d).",
            settings.association.parameter_name,
            len(features),
            settings.association.permutations,
            settings.association.knn,
        )
        records = analyze_h0_associations(
            features,
            store.h0,
            store.chain_id,
            settings.association,
        )
        association_paths = list(_write_h0_associations(directory, records))
        tail_summary = _load_global_tail_summary(results_path)
        tail_paths: list[Path] = []
        if tail_summary is not None:
            tail_record = next(
                (
                    record
                    for record in records
                    if record.get("type") == "global_tail_scale"
                ),
                None,
            )
            tail_paths = _write_global_tail_products(
                directory,
                tail_summary,
                weights,
                h0=store.h0,
                chain_id=store.chain_id,
                h0_record=tail_record,
            )
        figure_paths: list[Path] = []
        if settings.plot.write_pdf or settings.plot.write_png:
            figure_paths.extend(
                make_location_configuration_figures(
                    directory / "feature_location_configurations",
                    family_features,
                    location_configurations,
                    weights,
                    settings.plot,
                )
            )
            logger.info(
                "Writing %s-feature corner summaries.",
                settings.association.parameter_name,
            )
            figure_paths.extend(
                make_h0_corner_figures(
                    directory / "external_parameter_corner",
                    features,
                    records,
                    store.h0,
                    settings.plot,
                    settings.association.parameter_name,
                    settings.association.parameter_label,
                    parameter_unit=settings.association.parameter_unit,
                    coordinate1_name=settings.input.coordinate1_name,
                    coordinate2_name=settings.input.coordinate2_name,
                    coordinate1_label=settings.input.coordinate1_label,
                    coordinate2_label=settings.input.coordinate2_label,
                    coordinate1_unit=settings.input.coordinate1_unit,
                    coordinate2_unit=settings.input.coordinate2_unit,
                )
            )
        logger.info(
            "Adding the %s products to the existing results.h5 archive.",
            settings.association.parameter_name,
        )
        _replace_h0_results(
            results_path,
            fingerprint=fingerprint,
            topology_sample_signature=topology_sample_signature,
            h0=store.h0,
            chain_id=store.chain_id,
            records=records,
            location_configurations=location_configurations,
        )
        if settings.plot.write_pdf or settings.plot.write_png:
            logger.info("Refreshing configuration-aware feature geometry.")
            figure_paths.extend(
                make_feature_geometry_figure_from_results(
                    results_path,
                    directory / "feature_geometry",
                    settings=settings.plot,
                )
            )

        updated = dict(manifest)
        updated["package_version"] = __version__
        updated["fingerprint"] = fingerprint
        updated["topology_sample_signature"] = topology_sample_signature
        updated["h0_association"] = _h0_manifest(settings, store.chain_id)
        updated["location_configurations"] = _location_configuration_manifest(
            location_configurations
        )
        updated["geometry_figure"] = _geometry_figure_manifest()
        updated["last_update"] = {
            "kind": "h0_association_only",
            "topology_recomputed": False,
        }
        outputs = list(updated.get("outputs", []))
        written = [
            *configuration_paths,
            *cooccurrence_paths,
            *association_paths,
            *tail_paths,
            *figure_paths,
            results_path,
            directory / "run.log",
        ]
        for path in written:
            if path.exists() and path.name not in outputs:
                outputs.append(path.name)
        updated["outputs"] = outputs
        _write_manifest_atomic(directory, updated)
        logger.info(
            "%s-only update complete. No density summaries or topology were "
            "recomputed. Results written to %s",
            settings.association.parameter_name,
            directory,
        )
        return directory
    finally:
        _close_logging(logger)


def _run_two_dimensional(settings: Settings) -> Path:
    """Run or resume the existing two-dimensional analysis."""

    directory = settings.output.directory
    directory.mkdir(parents=True, exist_ok=True)
    had_existing_output = any(directory.iterdir())

    with open_density_store(
        settings.input.file,
        input_settings=settings.input,
        association_settings=settings.association,
    ) as store:
        grid = validate_store(
            store,
            log_base=settings.analysis.log_base,
            geometry=settings.analysis.geometry,
            feature_measure=settings.analysis.feature_measure,
            input_settings=settings.input,
        )
        if settings.analysis.detect_shoulders and (
            np.any(grid.m1 <= 0.0) or np.any(grid.m2 <= 0.0)
        ):
            raise ValueError(
                "analysis.detect_shoulders=true requires positive coordinates."
            )
        if settings.analysis.ordered_tails and not grid.ordered_domain:
            raise ValueError(
                "analysis.ordered_tails=true requires an ordered domain with "
                "coordinate2 < coordinate1 everywhere inside the mask."
            )
        if settings.association.enabled == "off":
            store.h0 = None
            store.chain_id = None
        if store.h0 is not None and not np.allclose(
            store.weights, np.full(store.number_draws, 1.0 / store.number_draws)
        ):
            raise ValueError(
                "The external-parameter association module currently requires "
                "equal-weight posterior draws. Resample first or disable it."
            )
        fingerprint = _fingerprint(settings, store.shape)
        existing_manifest = _read_manifest(directory)
        if (
            existing_manifest
            and existing_manifest.get("completed")
            and not settings.output.overwrite
        ):
            h0_current = (
                store.h0 is not None
                and store.chain_id is not None
                and _stored_h0_is_current(
                    directory,
                    existing_manifest,
                    settings,
                    store.h0,
                    store.chain_id,
                )
            )
            if existing_manifest.get("fingerprint") == fingerprint and (
                store.h0 is None or h0_current
            ):
                if store.h0 is None:
                    print(_h0_unavailable_message(settings))
                _replot_completed_geometry_if_needed(
                    directory, existing_manifest, settings
                )
                print(f"Output is already complete: {directory}")
                return directory
            compatible, topology_sample_signature, reason = (
                _completed_topology_compatibility(
                    directory,
                    existing_manifest,
                    settings,
                    store,
                    grid,
                )
            )
            if compatible:
                if store.h0 is None:
                    print(_h0_unavailable_message(settings))
                    _replot_completed_geometry_if_needed(
                        directory, existing_manifest, settings
                    )
                    print(f"Output is already complete: {directory}")
                    return directory
                if h0_current:
                    _replot_completed_geometry_if_needed(
                        directory, existing_manifest, settings
                    )
                    print(f"Output is already complete: {directory}")
                    return directory
                print(
                    f"Aligned {settings.association.parameter_name} samples were "
                    "added or changed; reusing the completed feature results and "
                    f"running only the {settings.association.parameter_name} "
                    "analysis."
                )
                return _run_h0_only_update(
                    directory,
                    existing_manifest,
                    settings,
                    store,
                    fingerprint=fingerprint,
                    topology_sample_signature=topology_sample_signature,
                )
            raise FileExistsError(
                "The output directory contains a completed run for different "
                "input or settings, so its topology cannot be reused "
                f"({reason}). Set overwrite=true or choose another directory."
            )
        if had_existing_output and not (
            settings.output.overwrite or settings.compute.resume
        ):
            raise FileExistsError(
                "Output directory is not empty. Enable resume, set overwrite=true, "
                "or choose another directory."
            )

        protected = {settings.source.resolve(), settings.input.file.resolve()}
        for name in _GENERATED_NAMES:
            if (directory / name).resolve() == settings.input.file.resolve():
                raise ValueError(
                    f"Input file collides with generated output name: {name}"
                )
        if settings.output.overwrite:
            _clear_generated_outputs(directory, protected)
        logger = _configure_logging(directory)
        topology_sample_signature = _topology_sample_signature(store, grid)
        settings_copy = directory / "settings.ini"
        if settings_copy.resolve() != settings.source.resolve():
            shutil.copy2(settings.source, settings_copy)

        workers = min(_automatic_workers(settings.compute.workers), store.number_draws)
        batch_size = _automatic_batch_size(settings.compute.batch_size, grid, workers)
        logger.info(
            "Input: %d posterior draws on a %d x %d grid; %d workers, batch %d.",
            store.number_draws,
            grid.shape[0],
            grid.shape[1],
            workers,
            batch_size,
        )
        logger.info(
            "Input measure: %s; feature measure: %s; analysis geometry: %s.",
            grid.input_density_measure,
            grid.feature_measure,
            grid.geometry,
        )
        if settings.analysis.detect_shoulders:
            logger.info(
                "Directional shoulders: alpha_parallel < %.6g; required at %d/%d "
                "analysis scales.",
                settings.analysis.shoulder_alpha_threshold,
                1 if len(settings.analysis.scales) == 1 else 2,
                len(settings.analysis.scales),
            )
        if (
            settings.analysis.detect_shoulders
            and
            grid.feature_measure == "linear"
            and math.isclose(settings.analysis.shoulder_alpha_threshold, 2.0)
        ):
            logger.warning(
                "shoulder_alpha_threshold=2 has the per-log-density meaning. "
                "For a density analyzed in the linear coordinate measure, "
                "zero is the analogous "
                "decreasing-density threshold; keep 2 only if intentional."
            )
        if store.h0 is not None:
            logger.info(
                "Aligned %s samples found (%d chains): joint feature-vector "
                "associations will be computed.",
                settings.association.parameter_name,
                np.unique(store.chain_id).size,
            )
        else:
            logger.warning(_h0_unavailable_message(settings))
        if any(scale > 0.0 for scale in settings.analysis.scales):
            for label, coordinates in (
                (grid.coordinate1_name, grid.geometry1),
                (grid.coordinate2_name, grid.geometry2),
            ):
                differences = np.diff(coordinates)
                spacing_ratio = float(differences.max() / differences.min())
                if spacing_ratio > 1.10:
                    logger.warning(
                        "%s spacing varies by a factor %.3g in %s geometry; "
                        "Gaussian scale values are approximate. Choose the "
                        "matching geometry or use scales=0.",
                        label,
                        spacing_ratio,
                        grid.geometry,
                    )
        logger.info("Computing normalized posterior median and marginal intervals.")
        summary = compute_ensemble_summary(
            store,
            grid,
            credible_mass=settings.analysis.credible_mass,
            tail_probability=settings.one_dimensional.upper_tail_percentile,
            ordered_tails=settings.analysis.ordered_tails,
            batch_size=batch_size,
        )

        logger.info("Extracting the fixed catalogue from the posterior-median field.")
        reference_analyses = [
            analyze_field(
                summary.reference_density,
                grid,
                scale=scale,
                persistence_threshold=settings.analysis.persistence_threshold,
                persistence_gap_min_log=settings.analysis.persistence_gap_min_log,
                hessian_refinement=settings.analysis.hessian_refinement,
                detect_plateaus=settings.analysis.detect_plateaus,
                curve_points=settings.analysis.curve_points,
                support_mass=settings.analysis.support_mass,
                curve_smoothing_cells=settings.analysis.curve_smoothing_cells,
                shoulder_alpha_threshold=settings.analysis.shoulder_alpha_threshold,
                shoulder_scales=(
                    settings.analysis.scales if scale_index == 0 else (scale,)
                ),
                detect_shoulders=settings.analysis.detect_shoulders,
            )
            for scale_index, scale in enumerate(settings.analysis.scales)
        ]
        reference = reference_analyses[0]
        add_scale_persistence(reference, reference_analyses[1:], grid)
        scale_filter = retain_scale_persistent_reference(reference)
        boundary_filter = remove_corner_boundary_pits(reference)
        assign_reference_identifiers(reference, grid)
        logger.info(
            (
                "Reference multiscale filter: required %d/%d scales; "
                "removed %d point extrema and %d hierarchy branches."
            ),
            scale_filter["required"],
            len(reference_analyses),
            scale_filter["points_removed"],
            scale_filter["branches_removed"],
        )
        if boundary_filter["corner_pits_removed"]:
            logger.info(
                (
                    "Reference corner-boundary filter: removed %d corner pit(s), "
                    "%d incident branch(es), and %d paired event(s)."
                ),
                boundary_filter["corner_pits_removed"],
                boundary_filter["branches_removed"],
                boundary_filter["events_removed"],
            )
        threshold_diagnostics = reference.persistence_threshold_diagnostics
        if threshold_diagnostics.selection_mode == "automatic_gap":
            logger.info(
                (
                    "Automatic persistence threshold %.6g from log gap %.6g "
                    "between %.6g and %.6g (required G >= %.6g; terminal "
                    "gap %.6g excluded)."
                ),
                threshold_diagnostics.threshold,
                threshold_diagnostics.selected_gap_log,
                threshold_diagnostics.selected_gap_lower,
                threshold_diagnostics.selected_gap_upper,
                threshold_diagnostics.gap_min_log,
                threshold_diagnostics.terminal_gap_log,
            )
        elif threshold_diagnostics.selection_mode == "automatic_fallback":
            logger.info(
                (
                    "Automatic persistence threshold unresolved: no eligible "
                    "nonterminal log gap met G >= %.6g; using the v0.8 "
                    "fallback %.6g."
                ),
                threshold_diagnostics.gap_min_log,
                threshold_diagnostics.threshold,
            )
        else:
            logger.info(
                "Using configured persistence threshold %.6g.",
                threshold_diagnostics.threshold,
            )
        logger.info(
            (
                "Reference threshold %.6g: %d peaks, %d pits, %d ridges, "
                "%d valleys, %d shoulders, %d plateaus/floors."
            ),
            reference.persistence_threshold,
            sum(item.kind == "peak" for item in reference.points),
            sum(item.kind == "pit" for item in reference.points),
            sum(item.kind == "ridge" for item in reference.branches),
            sum(item.kind == "valley" for item in reference.branches),
            len(reference.shoulders),
            len(reference.plateaus),
        )
        logger.info(
            "Reference paired events: %d ridges and %d valleys.",
            sum(item.kind == "ridge" for item in reference.events),
            sum(item.kind == "valley" for item in reference.events),
        )
        reference_valleys = [
            item for item in reference.branches if item.kind == "valley"
        ]
        if reference_valleys:
            logger.info(
                "Valley support mask: %.3f probability; %d/%d reference valleys "
                "truncated and %d retain less than one grid cell.",
                settings.analysis.support_mass,
                sum(item.low_density_truncated for item in reference_valleys),
                len(reference_valleys),
                sum(not item.curve_available for item in reference_valleys),
            )
        logger.info(
            "Reference finite geometry: %d/%d peaks, %d/%d ridges, %d/%d valleys.",
            sum(item.geometry_valid for item in reference.points if item.kind == "peak"),
            sum(item.kind == "peak" for item in reference.points),
            sum(item.geometry_valid for item in reference.branches if item.kind == "ridge"),
            sum(item.kind == "ridge" for item in reference.branches),
            sum(item.geometry_valid for item in reference.branches if item.kind == "valley"),
            sum(item.kind == "valley" for item in reference.branches),
        )

        checkpoint_path = directory / ".posterior_landscape.checkpoint.pkl.gz"
        checkpoint = (
            _load_checkpoint(checkpoint_path, fingerprint)
            if settings.compute.resume and not settings.output.overwrite
            else None
        )
        if checkpoint is None:
            next_draw = 0
            catalogue = PosteriorCatalogue.from_reference(
                reference, store.number_draws, grid
            )
        else:
            next_draw, catalogue = checkpoint
            logger.info("Resuming posterior feature extraction at draw %d.", next_draw)

        options = _analysis_options(settings.analysis, reference.persistence_threshold)
        executor: ProcessPoolExecutor | None = None
        if workers > 1:
            context = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_configure_worker,
                initargs=(grid, options),
            )
        else:
            _configure_worker(grid, options)

        checkpoint_stride = max(256, batch_size * 4)
        last_checkpoint = next_draw
        checkpoint_time = time.monotonic()
        try:
            for start in range(next_draw, store.number_draws, batch_size):
                stop = min(start + batch_size, store.number_draws)
                batch = store.batch(start, stop)
                payload = (
                    (start + offset, batch[offset]) for offset in range(stop - start)
                )
                results = (
                    executor.map(_analyze_worker, payload, chunksize=1)
                    if executor is not None
                    else map(_analyze_worker, payload)
                )
                for draw, analysis in results:
                    # A fixed catalogue is intentional: it makes posterior IDs stable
                    # and matches the comparison catalogue supplied by the user.
                    catalogue.ingest(
                        draw,
                        analysis,
                        draw_weight=float(store.weights[draw]),
                        discover=False,
                    )
                logger.info(
                    "Posterior topology: %d/%d draws.", stop, store.number_draws
                )
                now = time.monotonic()
                if settings.compute.resume and (
                    stop - last_checkpoint >= checkpoint_stride
                    or now - checkpoint_time >= 300.0
                ):
                    _save_checkpoint(
                        checkpoint_path,
                        fingerprint=fingerprint,
                        next_draw=stop,
                        catalogue=catalogue,
                    )
                    last_checkpoint = stop
                    checkpoint_time = now
        finally:
            if executor is not None:
                executor.shutdown(cancel_futures=True)

        points, branches, plateaus = catalogue.retained(
            store.weights, settings.analysis.minimum_support
        )
        events = catalogue.retained_events(
            store.weights, settings.analysis.minimum_support
        )
        shoulders = catalogue.retained_shoulders(
            store.weights, settings.analysis.minimum_support
        )
        radius = (
            settings.analysis.ridge_probability_radius
            if settings.analysis.ridge_probability_radius is not None
            else 1.5 * grid.typical_spacing
        )
        logger.info("Computing ridge and valley posterior probability maps.")
        ridge_probability, valley_probability = probability_maps(
            branches, grid, store.weights, radius=radius
        )
        logger.info("Computing feature credible regions and finite-region maps.")
        geometry_maps = feature_geometry_maps(
            [*points, *branches],
            grid,
            store.weights,
            credible_mass=settings.analysis.credible_mass,
        )

        records = _catalogue_records(points, branches, plateaus, store.weights, grid)
        catalogue_paths = _write_catalogue(directory, records, grid)
        diagnostic_paths = _write_branch_diagnostics(
            directory, branches, store.weights, grid
        )
        event_paths = _write_event_catalogue(
            directory,
            events,
            store.weights,
            settings.analysis.credible_mass,
        )
        shoulder_paths = _write_shoulder_catalogue(
            directory,
            shoulders,
            store.weights,
            settings.analysis.credible_mass,
        )
        intervals_path = _write_intervals(
            directory,
            points,
            branches,
            plateaus,
            store.weights,
            grid,
            settings.analysis.credible_mass,
        )
        topology_path = _write_topology_frequencies(directory, catalogue, store.weights)
        family_features = _association_features(points, events, shoulders)
        location_configurations = detect_location_configurations(
            family_features, store.weights
        )
        multimodal_identifiers = [
            identifier
            for identifier, result in location_configurations.items()
            if result.multimodal
        ]
        logger.info(
            "Posterior location configurations: %d robust multimodal "
            "families%s.",
            len(multimodal_identifiers),
            (
                " (" + ", ".join(multimodal_identifiers) + ")"
                if multimodal_identifiers
                else ""
            ),
        )
        configuration_paths = list(
            _write_location_configurations(
                directory,
                family_features,
                location_configurations,
                store.weights,
                settings.analysis.credible_mass,
            )
        )
        cooccurrence_paths = list(
            _write_pairwise_feature_associations(
                directory,
                family_features,
                store.weights,
                h0=store.h0,
                chain_id=store.chain_id,
                settings=settings,
            )
        )
        association_features = expand_location_configurations(
            family_features, location_configurations
        )
        if settings.analysis.ordered_tails:
            association_features.append(_global_tail_association_feature(summary))
        h0_records: list[dict[str, Any]] = []
        association_paths: list[Path] = []
        if store.h0 is not None:
            logger.info(
                "Computing %s associations (%d permutations; k=%d).",
                settings.association.parameter_name,
                settings.association.permutations,
                settings.association.knn,
            )
            h0_records = analyze_h0_associations(
                association_features,
                store.h0,
                store.chain_id,
                settings.association,
            )
            association_paths.extend(
                _write_h0_associations(directory, h0_records)
            )
        tail_h0_record = next(
            (
                record
                for record in h0_records
                if record.get("type") == "global_tail_scale"
            ),
            None,
        )
        tail_paths = (
            _write_global_tail_products(
                directory,
                summary,
                store.weights,
                h0=store.h0,
                chain_id=store.chain_id,
                h0_record=tail_h0_record,
            )
            if settings.analysis.ordered_tails
            else []
        )
        numerical_path = _write_numerical_results(
            directory,
            fingerprint=fingerprint,
            topology_sample_signature=topology_sample_signature,
            grid=grid,
            weights=store.weights,
            summary=summary,
            reference=reference,
            points=points,
            branches=branches,
            events=events,
            shoulders=shoulders,
            plateaus=plateaus,
            ridge_probability=ridge_probability,
            valley_probability=valley_probability,
            geometry_maps=geometry_maps,
            credible_mass=settings.analysis.credible_mass,
            support_mass=settings.analysis.support_mass,
            curve_smoothing_cells=settings.analysis.curve_smoothing_cells,
            shoulder_alpha_threshold=settings.analysis.shoulder_alpha_threshold,
            h0=store.h0,
            chain_id=store.chain_id,
            h0_records=h0_records,
            location_configurations=location_configurations,
        )
        figure_paths: list[Path] = []
        if settings.plot.write_pdf or settings.plot.write_png:
            logger.info("Writing the landscape and feature-confidence figures.")
            figure_paths.extend(
                make_landscape_figure(
                    directory / "landscape",
                    grid=grid,
                    summary=summary,
                    reference=reference,
                    point_trackers=points,
                    branch_trackers=branches,
                    event_trackers=events,
                    plateau_trackers=plateaus,
                    shoulder_trackers=shoulders,
                    weights=store.weights,
                    ridge_probability=ridge_probability,
                    valley_probability=valley_probability,
                    geometry_maps=geometry_maps,
                    settings=settings.plot,
                )
            )
            figure_paths.extend(
                make_feature_projection_figure(
                    directory / "feature_projections",
                    grid=grid,
                    summary=summary,
                    point_trackers=points,
                    event_trackers=events,
                    shoulder_trackers=shoulders,
                    weights=store.weights,
                    credible_mass=settings.analysis.credible_mass,
                    settings=settings.plot,
                    location_configurations=location_configuration_masks(
                        location_configurations
                    ),
                )
            )
            if settings.analysis.ordered_tails:
                figure_paths.extend(
                    make_global_tail_figure(
                        directory / "global_tail_probabilities",
                        summary=summary,
                        weights=store.weights,
                        settings=settings.plot,
                    )
                )
            figure_paths.extend(
                make_feature_projection_figure(
                    directory / "feature_projections_m2",
                    grid=grid,
                    summary=summary,
                    point_trackers=points,
                    event_trackers=events,
                    shoulder_trackers=shoulders,
                    weights=store.weights,
                    credible_mass=settings.analysis.credible_mass,
                    settings=settings.plot,
                    coordinate="m2",
                    location_configurations=location_configuration_masks(
                        location_configurations
                    ),
                )
            )
            figure_paths.extend(
                make_location_configuration_figures(
                    directory / "feature_location_configurations",
                    family_features,
                    location_configurations,
                    store.weights,
                    settings.plot,
                )
            )
            if store.h0 is not None:
                figure_paths.extend(
                    make_h0_corner_figures(
                        directory / "external_parameter_corner",
                        association_features,
                        h0_records,
                        store.h0,
                        settings.plot,
                        settings.association.parameter_name,
                        settings.association.parameter_label,
                        parameter_unit=settings.association.parameter_unit,
                        coordinate1_name=grid.coordinate1_name,
                        coordinate2_name=grid.coordinate2_name,
                        coordinate1_label=grid.coordinate1_label,
                        coordinate2_label=grid.coordinate2_label,
                        coordinate1_unit=grid.coordinate1_unit,
                        coordinate2_unit=grid.coordinate2_unit,
                    )
                )
            figure_paths.extend(
                make_shoulder_geometry_figure(
                    directory / "shoulder_geometry",
                    grid=grid,
                    summary=summary,
                    shoulder_trackers=shoulders,
                    weights=store.weights,
                    credible_mass=settings.analysis.credible_mass,
                    settings=settings.plot,
                )
            )
            figure_paths.extend(
                make_feature_geometry_figure(
                    directory / "feature_geometry",
                    grid=grid,
                    summary=summary,
                    point_trackers=points,
                    branch_trackers=branches,
                    weights=store.weights,
                    geometry_maps=geometry_maps,
                    credible_mass=settings.analysis.credible_mass,
                    settings=settings.plot,
                    location_configurations=location_configuration_masks(
                        location_configurations
                    ),
                )
            )

        output_paths = [
            *catalogue_paths,
            *diagnostic_paths,
            *event_paths,
            *shoulder_paths,
            intervals_path,
            topology_path,
            *configuration_paths,
            *cooccurrence_paths,
            *association_paths,
            *tail_paths,
            numerical_path,
            *figure_paths,
            directory / "settings.ini",
            directory / "run.log",
        ]
        manifest = {
            "completed": True,
            "package_version": __version__,
            "fingerprint": fingerprint,
            "topology_sample_signature": topology_sample_signature,
            "input_file": str(settings.input.file),
            "number_draws": store.number_draws,
            "grid_shape": list(grid.shape),
            "input_density_measure": grid.input_density_measure,
            "density_measure": _density_measure_name(grid.feature_measure),
            "feature_measure": grid.feature_measure,
            "domain": grid.domain,
            "ordered_domain": grid.ordered_domain,
            "coordinates": {
                "coordinate1": {
                    "name": grid.coordinate1_name,
                    "label": grid.coordinate1_label,
                    "unit": grid.coordinate1_unit,
                },
                "coordinate2": {
                    "name": grid.coordinate2_name,
                    "label": grid.coordinate2_label,
                    "unit": grid.coordinate2_unit,
                },
            },
            "log_base": grid.log_base,
            "analysis_geometry": grid.geometry,
            "persistence_threshold": reference.persistence_threshold,
            "persistence_threshold_selection": (
                reference.persistence_threshold_diagnostics.as_dict()
            ),
            "support_mass": settings.analysis.support_mass,
            "curve_smoothing_cells": settings.analysis.curve_smoothing_cells,
            "shoulder_alpha_threshold": settings.analysis.shoulder_alpha_threshold,
            "shoulders_enabled": settings.analysis.detect_shoulders,
            "ordered_tails_enabled": settings.analysis.ordered_tails,
            "ridge_probability_radius": radius,
            "width_definition": "half_prominence_or_half_depth",
            "width_fraction": 0.5,
            "width_units": "mass",
            "adaptive_measurement": {
                "point_peak_weight": "density",
                "finite_point_pit_weight": "saddle_deficit",
                "curve_event_pairing": "common_saddle_two_arm",
                "ridge_event_weight": "density",
                "valley_event_weight": "log_interpolated_side_shoulder_deficit",
                "shoulder_direction": "d_dlnm1_plus_d_dlnm2",
                "shoulder_region_weight": "density",
                "windows_may_overlap": True,
            },
            "global_tail_scale": (
                {
                    "enabled": True,
                    "percentile": summary.tail_probability,
                    "label": "global_tail_scale",
                    "coordinate1_scale": "first marginal percentile",
                    "coordinate2_scale": "second marginal percentile",
                    "P_any": "P(coordinate1 > threshold)",
                    "P_both": "P(coordinate2 > threshold)",
                    "P_straddle": "P(coordinate1 > threshold, coordinate2 <= threshold)",
                }
                if settings.analysis.ordered_tails
                else {"enabled": False}
            ),
            "feature_cooccurrence": {
                "indicator": "draw_level_location_support",
                "scope": "all_fixed_catalogue_family_pairs",
                "statistics": [
                    "P(A_and_B)",
                    "P(B_given_A)",
                    "P(A_given_B)",
                    "covariance",
                    "phi",
                    "lift",
                ],
                "automatic_label_merging": False,
            },
            "geometry_figure": _geometry_figure_manifest(),
            "location_configurations": _location_configuration_manifest(
                location_configurations
            ),
            "maximum_input_normalization_error": summary.maximum_normalization_error,
            "features": {
                "peaks": sum(item.template.kind == "peak" for item in points),
                "pits": sum(item.template.kind == "pit" for item in points),
                "ridges": sum(item.template.kind == "ridge" for item in branches),
                "valleys": sum(item.template.kind == "valley" for item in branches),
                "ridge_events": sum(
                    item.template.kind == "ridge" for item in events
                ),
                "valley_events": sum(
                    item.template.kind == "valley" for item in events
                ),
                "shoulders": len(shoulders),
                "plateaus_and_floors": len(plateaus),
            },
            "h0_association": (
                _h0_manifest(settings, store.chain_id)
                if store.h0 is not None
                else {
                    "available": False,
                    "knn": settings.association.knn,
                    "permutations": settings.association.permutations,
                    "uncertainty_resamples": (
                        settings.association.uncertainty_resamples
                    ),
                    "random_seed": settings.association.random_seed,
                    "number_chains": 0,
                    "null_calibration": "within_chain_circular_h0_shift",
                    "continuous_conditioning": "valid_draw_adaptive_region",
                }
            ),
            "external_parameter_association": (
                _h0_manifest(settings, store.chain_id)
                if store.h0 is not None
                else {
                    "available": False,
                    "dataset": settings.association.dataset,
                    "parameter_name": settings.association.parameter_name,
                    "parameter_label": settings.association.parameter_label,
                    "parameter_unit": settings.association.parameter_unit,
                }
            ),
            "reference_finite_geometry": {
                "peaks": sum(
                    item.template.kind == "peak" and item.template.geometry_valid
                    for item in points
                ),
                "ridges": sum(
                    item.template.kind == "ridge" and item.template.geometry_valid
                    for item in branches
                ),
                "valleys": sum(
                    item.template.kind == "valley" and item.template.geometry_valid
                    for item in branches
                ),
                "shoulders": sum(
                    item.template.region_valid for item in shoulders
                ),
            },
            "outputs": [path.name for path in output_paths if path.exists()],
        }
        write_json(directory / "manifest.json", manifest)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        logger.info("Complete. Results written to %s", directory)
        print((directory / "catalogue.txt").read_text(encoding="utf-8"))
        if shoulders:
            print((directory / "shoulder_catalogue.txt").read_text(encoding="utf-8"))
        _close_logging(logger)
        return directory


def _consolidate_feature_tables(directory: Path) -> Path | None:
    """Merge the human-facing feature tables into one sparse CSV catalogue."""

    sources = [
        directory / "catalogue.csv",
        directory / "curve_event_catalogue.csv",
        directory / "shoulder_catalogue.csv",
    ]
    interval_path = directory / "posterior_intervals.csv"
    intervals: dict[str, dict[str, str]] = {}
    if interval_path.is_file():
        with interval_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                intervals[str(row.get("ID", ""))] = dict(row)
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not source.is_file():
            continue
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                merged = dict(row)
                identifier = str(merged.get("ID", ""))
                for key, value in intervals.get(identifier, {}).items():
                    merged.setdefault(key, value)
                rows.append(
                    {
                        key.replace("m1", "x1").replace("m2", "x2"): value
                        for key, value in merged.items()
                    }
                )
    if not rows:
        return None
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    destination = directory / "features.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _apply_output_policy(
    directory: Path,
    profile: str,
    parameter_name: str = "H0",
    coordinate1_name: str = "m1",
    coordinate2_name: str = "m2",
) -> None:
    """Keep PDF figures and CSV tables, with a compact essential profile."""

    for path in list(directory.iterdir()):
        if not path.is_file():
            continue
        neutral_name = path.name.replace("H0", "external_parameter").replace(
            "h0", "external_parameter"
        )
        if parameter_name.lower() != "h0":
            neutral_name = neutral_name.replace(
                f"_{parameter_name}_feature", "_external_parameter_feature"
            )
        if (coordinate1_name, coordinate2_name) != ("m1", "m2"):
            neutral_name = neutral_name.replace(
                "mass_scale", "feature_location"
            )
            neutral_name = neutral_name.replace(
                "external_parameter_feature_class_feature_location",
                "external_parameter_feature_class_location",
            ).replace(
                "external_parameter_feature_feature_location",
                "external_parameter_feature_location",
            )
            neutral_name = neutral_name.replace(
                "_pm1_", f"_{coordinate1_name}_"
            ).replace("_pm2_", f"_{coordinate2_name}_")
        if neutral_name != path.name:
            destination = path.with_name(neutral_name)
            if destination.exists():
                destination.unlink()
            path.rename(destination)
    for path in directory.glob("*.csv"):
        content = path.read_text(encoding="utf-8")
        neutral = content.replace("H0", "parameter").replace("h0", "parameter")
        if (coordinate1_name, coordinate2_name) != ("m1", "m2"):
            neutral = neutral.replace("mass scale", "feature location").replace(
                "mass_scale", "feature_location"
            )
        if neutral != content:
            path.write_text(neutral, encoding="utf-8")
    for path in directory.glob("*external_parameter*.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        if rows and "parameter_name" not in fields:
            fields.append("parameter_name")
            for row in rows:
                row["parameter_name"] = parameter_name
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    for path in list(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".png", ".tex"}:
            path.unlink()
    if profile == "essential":
        for name in (
            "global_tail_scales_draws.npz",
            "one_two_dimensional_full_h0_draws.npz",
            "one_two_dimensional_full_external_parameter_draws.npz",
            "one_two_dimensional_feature_family_draws.npz",
        ):
            path = directory / name
            if path.is_file():
                path.unlink()
        csv_keep = {
            "features.csv",
            "external_parameter_associations.csv",
            "one_two_dimensional_feature_comparison.csv",
            "one_two_dimensional_m2_feature_comparison.csv",
            "one_two_dimensional_feature_families.csv",
        }
        for path in list(directory.glob("*.csv")):
            keep_one_d_parameter = any(
                token in path.name
                for token in (
                    "external_parameter_feature_summary_table_revised.csv",
                    "external_parameter_feature_dependence_main.csv",
                )
            )
            if (
                path.name not in csv_keep
                and not path.name.endswith("feature_posterior_summary.csv")
                and not keep_one_d_parameter
            ):
                path.unlink()
        for path in list(directory.glob("*.pdf")):
            if path.name != "landscape.pdf" and "features_and_support" not in path.name:
                path.unlink()
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_profile"] = profile
        manifest["outputs"] = sorted(
            path.name for path in directory.iterdir() if path.is_file()
        )
        write_json(manifest_path, manifest)


def _materialize_essential_view(
    full_directory: Path,
    essential_directory: Path,
    parameter_name: str,
    coordinate1_name: str,
    coordinate2_name: str,
) -> None:
    """Create the compact essential view from one completed full run.

    The full directory is the source of every product.  Copying it before
    applying the essential policy guarantees that an essential view made by a
    full run has the same compact files as a standalone essential run, while
    preserving all draw-level products in ``full``.
    """

    if essential_directory == full_directory:
        raise ValueError(
            "output.full_subdirectory must differ from the essential directory."
        )
    if essential_directory.exists():
        shutil.rmtree(essential_directory)
    shutil.copytree(full_directory, essential_directory)
    _apply_output_policy(
        essential_directory,
        "essential",
        parameter_name,
        coordinate1_name,
        coordinate2_name,
    )


def run(settings: Settings) -> Path:
    """Run the selected 1D, 2D, or combined workflow with one command."""

    requested_directory = settings.output.directory
    requested_profile = settings.output.profile
    if requested_profile == "full":
        effective_directory = requested_directory / settings.output.full_subdirectory
        settings = replace(
            settings,
            output=replace(settings.output, directory=effective_directory),
        )
    one_d_result = None
    run_one_d = settings.one_dimensional.enabled
    run_two_d = settings.analysis.run in {"2d", "both"}
    if run_one_d:
        from .one_dimensional import run_one_dimensional

        one_d_result = run_one_dimensional(settings)

    if run_two_d:
        _run_two_dimensional(settings)

    if run_one_d and run_two_d:
        from .comparison import run_one_two_dimensional_comparison

        if one_d_result is None:  # pragma: no cover - defensive invariant.
            raise RuntimeError("The combined workflow lacks its 1D result.")
        run_one_two_dimensional_comparison(
            settings,
            one_d_result.state,
            one_d_result.secondary_state,
        )

    _consolidate_feature_tables(settings.output.directory)
    _apply_output_policy(
        settings.output.directory,
        requested_profile,
        settings.association.parameter_name,
        settings.input.coordinate1_name,
        settings.input.coordinate2_name,
    )
    if requested_profile == "full":
        _materialize_essential_view(
            settings.output.directory,
            requested_directory / "essential",
            settings.association.parameter_name,
            settings.input.coordinate1_name,
            settings.input.coordinate2_name,
        )
    return requested_directory
