"""Notebook-faithful one-dimensional feature analysis.

The scientific implementation lives in ``_one_d_cells``.  Those files are
the authoritative notebook cells supplied for this project.  This module only
constructs their inputs, substitutes explicit INI settings, caches their
draw-level products, and makes catalogue ordering data-driven.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import __version__
from .association import pairwise_indicator_associations
from .config import Settings
from .io import DensityStore, open_density_store


_MISSING_H0_WARNING = (
    "WARNING: No aligned top-level dataset named 'h0' was found. "
    "Continuing without H0 association analysis; this is valid for a "
    "fixed-H0 run."
)

_STATE_NAME = "one_dimensional_state.pkl.gz"
_MANIFEST_NAME = "one_dimensional_manifest.json"
_SECONDARY_STATE_NAME = "one_dimensional_m2_state.pkl.gz"
_SECONDARY_MANIFEST_NAME = "one_dimensional_m2_manifest.json"
_NOTEBOOK_RANDOM_SEED = 12345


@dataclass(frozen=True)
class OneDimensionalResult:
    """Reusable products from the independent component-mass analyses."""

    directory: Path
    state: dict[str, Any]
    secondary_state: dict[str, Any]
    reused_core: bool
    reused_secondary_core: bool
    h0_analyzed: bool


def cell_widths_from_centers(x: np.ndarray) -> np.ndarray:
    """Notebook quadrature rule, copied without a numerical change."""

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D grid with at least two points")
    edges = np.empty(x.size + 1, dtype=np.float64)
    if np.all(x > 0):
        edges[1:-1] = np.sqrt(x[:-1] * x[1:])
        edges[0] = x[0] ** 2 / edges[1]
        edges[-1] = x[-1] ** 2 / edges[-2]
    else:
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * (x[1] - x[0])
        edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.diff(edges)


def _cell_source(name: str) -> str:
    return (
        resources.files("posterior_landscape._one_d_cells")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


def _coordinate_source(source: str, coordinate: str) -> str:
    """Adapt only user-facing string literals for the secondary coordinate.

    Variable names and executable expressions are deliberately untouched, so
    the supplied notebook algorithm remains exactly the same in both runs.
    """

    if coordinate == "m1":
        return source
    if coordinate != "m2":
        raise ValueError("coordinate must be 'm1' or 'm2'.")

    class _Strings(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if not isinstance(node.value, str):
                return node
            value = node.value
            for first, second in (
                ("_pm1_", "_pm2_"),
                ("dP_dlnm1", "dP_dlnm2"),
                ("dP_dm1", "dP_dm2"),
                ("dP/dln(m1)", "dP/dln(m2)"),
                ("dP/dm1", "dP/dm2"),
                ("m1 >", "m2 >"),
                ("p(m1", "p(m2"),
                ("primary-mass", "secondary-mass"),
                ("Primary-mass", "Secondary-mass"),
                ("primary mass", "secondary mass"),
                ("Primary mass", "Secondary mass"),
                ("m_1", "m_2"),
            ):
                value = value.replace(first, second)
            return ast.copy_location(ast.Constant(value=value), node)

    tree = _Strings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _external_parameter_source(source: str, settings: Settings) -> str:
    """Adapt H0-only notebook strings for a configured scalar parameter.

    Executable variable names are deliberately unchanged. For the legacy H0
    configuration the source is returned byte-for-byte, preserving the v0.7
    notebook path.
    """

    if settings.association.parameter_name == "H0":
        return source
    parameter_name = settings.association.parameter_name
    parameter_label = settings.association.parameter_label
    label_core = (
        parameter_label[1:-1]
        if parameter_label.startswith("$") and parameter_label.endswith("$")
        else parameter_label
    )

    tree = ast.parse(source)
    executable_identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }

    class _Strings(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if not isinstance(node.value, str):
                return node
            # Some notebook cells validate their inputs through strings passed
            # to globals().  Those strings name the unchanged private H0_*
            # implementation variables and must remain synchronized with the
            # executable identifiers.  Public prose, labels, table columns,
            # and filenames are still adapted to the configured parameter.
            if node.value in executable_identifiers:
                return node
            value = node.value.replace("Hubble constant", parameter_name)
            value = value.replace("H_0", label_core).replace("H0", parameter_name)
            return ast.copy_location(ast.Constant(value=value), node)

    tree = _Strings().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _replace_top_level_assignments(
    source: str, replacements: dict[str, str]
) -> str:
    """Replace selected notebook configuration assignments by source text."""

    tree = ast.parse(source)
    spans: list[tuple[int, int, str]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        for name in names:
            if name in replacements:
                spans.append((node.lineno - 1, node.end_lineno or node.lineno, name))
                break
    missing = set(replacements) - {name for _, _, name in spans}
    if missing:
        raise RuntimeError(
            "Notebook configuration assignment not found: "
            + ", ".join(sorted(missing))
        )
    lines = source.splitlines(keepends=True)
    for start, stop, name in sorted(spans, reverse=True):
        newline = "\n" if lines[start].endswith("\n") else ""
        lines[start:stop] = [f"{name} = {replacements[name]}{newline}"]
    return "".join(lines)


def _execute(source: str, namespace: dict[str, Any], name: str) -> None:
    exec(compile(source, name, "exec"), namespace, namespace)


def _savefig_factory(settings: Settings):
    def savefig(filename: str | Path) -> None:
        import matplotlib.pyplot as plt

        requested = Path(filename)
        requested.parent.mkdir(parents=True, exist_ok=True)
        stem = requested.with_suffix("")
        if settings.plot.write_pdf:
            plt.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        if settings.plot.write_png:
            plt.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")

    return savefig


def _label_with_unit(label: str, unit: str) -> str:
    """Return a plotting label with an optional plain-text unit suffix."""

    return f"{label} [{unit}]" if unit else label


def _external_parameter_axis_label(settings: Settings) -> str:
    """Preserve the legacy H0 axis while supporting any configured scalar."""

    if settings.association.parameter_name == "H0":
        return r"$H_0\,[{\rm km\,s^{-1}\,Mpc^{-1}}]$"
    return _label_with_unit(
        settings.association.parameter_label,
        settings.association.parameter_unit,
    )


def _update_hash(digest: Any, values: np.ndarray, dtype: Any) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _density_signature(
    store: DensityStore,
    settings: Settings,
    *,
    package_version: str = __version__,
) -> str:
    digest = hashlib.sha256(b"posterior-landscape-one-dimensional-v1")
    digest.update(package_version.encode("ascii"))
    digest.update(repr(settings.one_dimensional).encode("utf-8"))
    digest.update(str(settings.analysis.log_base).encode("ascii"))
    _update_hash(digest, store.m1, "<f8")
    _update_hash(digest, store.m2, "<f8")
    _update_hash(digest, store.mask, np.uint8)
    _update_hash(digest, store.weights, "<f8")
    indices = np.unique(
        np.linspace(0, store.number_draws - 1, min(16, store.number_draws)).astype(
            int
        )
    )
    _update_hash(digest, indices, "<i8")
    for index in indices:
        _update_hash(digest, store.draws[int(index)], "<f8")
    return digest.hexdigest()


def _h0_signature(store: DensityStore, settings: Settings) -> str | None:
    if store.h0 is None or store.chain_id is None:
        return None
    digest = hashlib.sha256(b"posterior-landscape-one-dimensional-h0-v1")
    _update_hash(digest, store.h0, "<f8")
    _update_hash(digest, store.chain_id, "<i8")
    digest.update(repr(settings.association).encode("utf-8"))
    digest.update(str(_NOTEBOOK_RANDOM_SEED).encode("ascii"))
    return digest.hexdigest()


def _core_signature_matches(
    manifest: dict[str, Any] | None,
    current_signature: str,
    legacy_signatures: dict[str, str] | str,
) -> bool:
    """Accept scientifically identical earlier 1D cores after this patch."""

    if not manifest:
        return False
    if isinstance(legacy_signatures, str):
        legacy_signatures = {"0.6.1": legacy_signatures}
    stored = manifest.get("core_signature")
    return bool(
        stored == current_signature
        or (
            manifest.get("package_version") in legacy_signatures
            and stored == legacy_signatures[manifest.get("package_version")]
        )
    )


def _write_pickle_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=4) as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read_pickle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid one-dimensional state file.")
    return payload


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _clear_generated_one_d_outputs(directory: Path) -> None:
    exact = {
        _STATE_NAME,
        _MANIFEST_NAME,
        _SECONDARY_STATE_NAME,
        _SECONDARY_MANIFEST_NAME,
    }
    prefixes = (
        "fullpop_dP_dlnm1_",
        "fullpop_dP_dm1_",
        "fullpop_dP_dlnm2_",
        "fullpop_dP_dm2_",
        "one_dimensional_m1_",
        "one_dimensional_m2_",
        "one_two_dimensional_",
    )
    for path in directory.iterdir():
        if path.is_file() and (path.name in exact or path.name.startswith(prefixes)):
            path.unlink()


def _component_mass_marginals(
    store: DensityStore, settings: Settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover both linear component-mass marginals in one input pass."""

    dm1 = cell_widths_from_centers(store.m1)
    dm2 = cell_widths_from_centers(store.m2)
    if np.any(store.m1 <= 0.0) or np.any(store.m2 <= 0.0):
        raise ValueError(
            "The independent 1D finder currently requires positive coordinates "
            "because smoothing and matching are performed logarithmically."
        )
    log_jacobian = np.multiply.outer(store.m1, store.m2) * float(
        np.log(settings.analysis.log_base) ** 2
    )
    cell_area = np.multiply.outer(dm1, dm2) * store.mask
    primary = np.empty((store.number_draws, store.m1.size), dtype=np.float64)
    secondary = np.empty((store.number_draws, store.m2.size), dtype=np.float64)
    batch_size = settings.compute.batch_size or 32
    batch_size = max(1, min(int(batch_size), store.number_draws))
    for start, batch in store.iter_batches(batch_size):
        stop = start + batch.shape[0]
        converted = (
            batch / log_jacobian[None, :, :]
            if settings.input.density_measure == "log"
            else batch
        )
        physical = np.where(store.mask[None, :, :], converted, 0.0)
        normalization = np.einsum(
            "bij,ij->b", physical, cell_area, optimize=True
        )
        if np.any(~np.isfinite(normalization)) or np.any(normalization <= 0.0):
            raise RuntimeError(
                "The two-dimensional input contains a draw with invalid "
                "linear-mass normalization."
            )
        physical /= normalization[:, None, None]
        primary[start:stop] = np.einsum(
            "bij,j->bi", physical, dm2, optimize=True
        )
        secondary[start:stop] = np.einsum(
            "bij,i->bj", physical, dm1, optimize=True
        )
        print(
            f"One-dimensional marginals: {stop}/{store.number_draws} draws.",
            flush=True,
        )
    return primary, dm1, secondary, dm2


def _geometric_one_dimensional_input(
    masses: np.ndarray,
    samples: np.ndarray,
    widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return the unchanged input or a probability-preserving log-grid copy.

    The notebook-derived 1D finder uses constant steps in ``ln(m)``.  The 2D
    package, however, accepts any strictly increasing positive coordinate
    grid.  Resample only at this interface when the input is not already
    geometric; the legacy GW path therefore remains byte-for-byte unchanged.
    """

    masses = np.asarray(masses, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    log_masses = np.log(masses)
    differences = np.diff(log_masses)
    spacing = float(np.median(differences))
    if np.allclose(differences, spacing, rtol=1.0e-5, atol=1.0e-12):
        return masses, samples, widths, False

    # Match the finest logarithmic resolution already present in the input.
    # This avoids coarsening the high-coordinate end of a uniform linear grid.
    number = max(
        masses.size,
        int(math.ceil((log_masses[-1] - log_masses[0]) / np.min(differences)))
        + 1,
    )
    target = np.geomspace(masses[0], masses[-1], number)
    resampled = np.empty((samples.shape[0], number), dtype=np.float64)
    for index, values in enumerate(samples):
        resampled[index] = np.interp(target, masses, values)
    resampled = np.where(
        np.isfinite(resampled) & (resampled >= 0.0), resampled, 0.0
    )
    target_widths = cell_widths_from_centers(target)
    normalization = np.einsum(
        "bi,i->b", resampled, target_widths, optimize=True
    )
    if np.any(~np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise RuntimeError(
            "Resampling onto the one-dimensional logarithmic grid produced "
            "an invalid marginal normalization."
        )
    resampled /= normalization[:, None]
    return target, resampled, target_widths, True


def _base_namespace(
    settings: Settings,
    directory: Path,
    pm1_samples: np.ndarray,
    dm1: np.ndarray,
) -> dict[str, Any]:
    import matplotlib

    try:
        from IPython.display import display
    except ImportError:
        display = print

    matplotlib.use("Agg", force=True)
    return {
        "__name__": "posterior_landscape.one_dimensional_notebook",
        "m_grid": np.asarray([], dtype=float),
        "pm1_samples": pm1_samples,
        "dm": dm1,
        "fin": str(directory),
        "MARGINAL_FLOOR": settings.one_dimensional.density_floor,
        "marginal_color": "#5C176D",
        "marginal_fill": "#D77A8A",
        "AXIS_LABEL_FS": 15,
        "TICK_LABEL_FS": 11.5,
        "MARGINAL_LINEWIDTH": 2.3,
        "MARGINAL_FILL_ALPHA": 0.48,
        "M_GRID_MIN": math.nan,
        "M_GRID_MAX": math.nan,
        "RANDOM_SEED": _NOTEBOOK_RANDOM_SEED,
        "_savefig": _savefig_factory(settings),
        "display": display,
    }


def _notebook_core(
    masses: np.ndarray,
    samples: np.ndarray,
    widths: np.ndarray,
    settings: Settings,
    directory: Path,
    *,
    coordinate: str,
) -> dict[str, Any]:
    namespace = _base_namespace(settings, directory, samples, widths)
    namespace["m_grid"] = np.asarray(masses, dtype=float)
    namespace["M_GRID_MIN"] = float(masses[0])
    namespace["M_GRID_MAX"] = float(masses[-1])

    coordinate_name = (
        settings.input.coordinate1_name
        if coordinate == "m1"
        else settings.input.coordinate2_name
    )
    coordinate_label = (
        settings.input.coordinate1_label
        if coordinate == "m1"
        else settings.input.coordinate2_label
    )
    coordinate_unit = (
        settings.input.coordinate1_unit
        if coordinate == "m1"
        else settings.input.coordinate2_unit
    )
    legacy_mass_coordinate = coordinate_name in {"m1", "m2"}
    if legacy_mass_coordinate:
        coordinate_index = "1" if coordinate == "m1" else "2"
        coordinate_xlabel = rf"$m_{coordinate_index}\,[M_\odot]$"
        coordinate_output_tag = f"pm{coordinate_index}"
        coordinate_plain_unit = "Msun"
        coordinate_value_heading = "mass"
        coordinate_log_argument = "m"
        feature_scale_channel_label = "mass scale"
        feature_scale_axis_label = r"Feature mass scale $[M_\odot]$"
        feature_collection_label = "mass features"
    else:
        coordinate_xlabel = _label_with_unit(coordinate_label, coordinate_unit)
        coordinate_output_tag = coordinate_name
        coordinate_plain_unit = coordinate_unit
        coordinate_value_heading = coordinate_name
        coordinate_log_argument = coordinate_name
        feature_scale_channel_label = "location"
        feature_scale_axis_label = f"Feature location {coordinate_xlabel}"
        feature_collection_label = "one-dimensional features"
    namespace.update(
        {
            "COORDINATE_NAME": coordinate_name,
            "COORDINATE_LABEL": coordinate_label,
            "COORDINATE_UNIT": coordinate_unit,
            "COORDINATE_PLAIN_UNIT": coordinate_plain_unit,
            "COORDINATE_UNIT_SUFFIX": (
                f" {coordinate_plain_unit}" if coordinate_plain_unit else ""
            ),
            "COORDINATE_VALUE_HEADING": coordinate_value_heading,
            "COORDINATE_LOG_ARGUMENT": coordinate_log_argument,
            "COORDINATE_XLABEL": coordinate_xlabel,
            "COORDINATE_OUTPUT_TAG": coordinate_output_tag,
            "FEATURE_SCALE_CHANNEL_LABEL": feature_scale_channel_label,
            "FEATURE_SCALE_AXIS_LABEL": feature_scale_axis_label,
            "FEATURE_COLLECTION_LABEL": feature_collection_label,
        }
    )

    one_d = settings.one_dimensional
    discovery = _replace_top_level_assignments(
        _coordinate_source(_cell_source("cell01_discovery.py"), coordinate),
        {
            "MASS_DENSITY_MEASURE": repr(one_d.measure),
            "FEATURE_SMOOTH_SCALES_LNM": repr(tuple(one_d.scales)),
            "FEATURE_MIN_SEPARATION_LNM": repr(one_d.minimum_separation_lnm),
            "FEATURE_MATCH_TOLERANCE_LNM": repr(one_d.match_tolerance_lnm),
            "FEATURE_MIN_PERSISTENCE": repr(one_d.minimum_persistence),
            "HIGH_MASS_PERCENTILE": repr(one_d.upper_tail_percentile),
        },
    )
    _execute(discovery, namespace, "one-dimensional discovery cell")
    if legacy_mass_coordinate:
        tail_tex_label = rf"$m_{{{namespace['percentile_label']}}}$"
    else:
        coordinate_label_core = (
            coordinate_label[1:-1]
            if coordinate_label.startswith("$") and coordinate_label.endswith("$")
            else coordinate_label
        )
        tail_tex_label = (
            rf"$({coordinate_label_core})_{{{namespace['percentile_label']}}}$"
        )
    if coordinate_name == "m2" and coordinate == "m2":
        if one_d.measure == "linear":
            namespace.update(
                {
                    "MASS_DENSITY_OUTPUT_TAG": "dP_dm2",
                    "MASS_DENSITY_PLAIN_LABEL": "dP/dm2",
                    "MASS_DENSITY_TEX_LABEL": r"$dP/dm_2$",
                    "MASS_DENSITY_YLABEL": r"$dP/dm_2\,[M_\odot^{-1}]$",
                }
            )
        else:
            namespace.update(
                {
                    "MASS_DENSITY_OUTPUT_TAG": "dP_dlnm2",
                    "MASS_DENSITY_PLAIN_LABEL": "dP/dln(m2)",
                    "MASS_DENSITY_TEX_LABEL": r"$dP/d\ln m_2$",
                    "MASS_DENSITY_YLABEL": r"$dP/d\ln m_2$",
                }
            )
    elif coordinate_name not in {"m1", "m2"}:
        if one_d.measure == "linear":
            unit = f" [{coordinate_unit}^-1]" if coordinate_unit else ""
            namespace.update(
                {
                    "MASS_DENSITY_OUTPUT_TAG": f"dP_d{coordinate_name}",
                    "MASS_DENSITY_PLAIN_LABEL": f"dP/d{coordinate_name}",
                    "MASS_DENSITY_TEX_LABEL": f"dP/d{coordinate_name}",
                    "MASS_DENSITY_YLABEL": f"dP/d{coordinate_name}{unit}",
                }
            )
        else:
            namespace.update(
                {
                    "MASS_DENSITY_OUTPUT_TAG": f"dP_dlog{coordinate_name}",
                    "MASS_DENSITY_PLAIN_LABEL": f"dP/dlog({coordinate_name})",
                    "MASS_DENSITY_TEX_LABEL": f"dP/dlog({coordinate_name})",
                    "MASS_DENSITY_YLABEL": f"dP/dlog({coordinate_label})",
                }
            )
    namespace["TAIL_TEX_LABEL"] = tail_tex_label
    _execute(
        _coordinate_source(_cell_source("cell02_bands.py"), coordinate),
        namespace,
        "one-dimensional bands cell",
    )
    _execute(
        _coordinate_source(_cell_source("cell03_draws.py"), coordinate),
        namespace,
        "one-dimensional draws cell",
    )
    _execute(
        _coordinate_source(_cell_source("cell04_figure.py"), coordinate),
        namespace,
        "one-dimensional figure cell",
    )
    summary_source = _replace_top_level_assignments(
        _coordinate_source(_cell_source("cell05_summaries.py"), coordinate),
        {"FEATURE_SUMMARY_ORDER": "tuple(feature_posterior_order)"},
    )
    _execute(summary_source, namespace, "one-dimensional summary cell")

    names = (
        "m_grid",
        "dm",
        "log_m_grid",
        "MASS_MEASURE_WIDTHS",
        "MASS_MEASURE_CELL_EDGES",
        "MASS_MEASURE_CELL_WIDTHS",
        "MASS_DENSITY_MEASURE",
        "MASS_DENSITY_OUTPUT_TAG",
        "MASS_DENSITY_PLAIN_LABEL",
        "MASS_DENSITY_TEX_LABEL",
        "MASS_DENSITY_YLABEL",
        "COORDINATE_NAME",
        "COORDINATE_LABEL",
        "COORDINATE_UNIT",
        "COORDINATE_PLAIN_UNIT",
        "COORDINATE_UNIT_SUFFIX",
        "COORDINATE_VALUE_HEADING",
        "COORDINATE_LOG_ARGUMENT",
        "COORDINATE_XLABEL",
        "COORDINATE_OUTPUT_TAG",
        "FEATURE_SCALE_CHANNEL_LABEL",
        "FEATURE_SCALE_AXIS_LABEL",
        "FEATURE_COLLECTION_LABEL",
        "TAIL_TEX_LABEL",
        "FEATURE_SMOOTH_SCALES_LNM",
        "FEATURE_REFERENCE_SMOOTH_LNM",
        "FEATURE_MIN_PERSISTENCE",
        "FEATURE_MATCH_TOLERANCE_LNM",
        "SHOULDER_ALPHA_MAX_SELECTED",
        "FEATURE_COLOR",
        "FEATURE_FILL",
        "PEAK_COLOR",
        "DIP_COLOR",
        "SHOULDER_COLOR",
        "M99_COLOR",
        "PEAK_BAND_ALPHA",
        "DIP_BAND_ALPHA",
        "SHOULDER_BAND_ALPHA",
        "HIGH_MASS_BAND_ALPHA",
        "FEATURE_AXIS_LABEL_FS",
        "FEATURE_TICK_LABEL_FS",
        "FEATURE_LEGEND_FS",
        "feature_bands",
        "feature_posterior_order",
        "feature_reference_lookup",
        "feature_draw_posteriors",
        "feature_posterior_summary_rows",
        "feature_posterior_summary_lookup",
        "m_high_percentile_samples",
        "m_high_q05",
        "m_high_q50",
        "m_high_q95",
        "percentile_label",
        "pm1_q05_feature",
        "pm1_q50_feature",
        "pm1_q95_feature",
        "reference_p",
    )
    state = {name: namespace[name] for name in names}
    state["coordinate"] = coordinate
    return state


def _h0_namespace(
    state: dict[str, Any],
    store: DensityStore,
    settings: Settings,
    directory: Path,
) -> dict[str, Any]:
    if store.h0 is None or store.chain_id is None:
        raise RuntimeError("H0 products requested without aligned H0 samples.")
    chain_id = np.asarray(store.chain_id)
    labels = np.unique(chain_id)
    groups = [np.flatnonzero(chain_id == label) for label in labels]
    counts = [indices.size for indices in groups]
    if not counts or len(set(counts)) != 1:
        raise ValueError(
            "The notebook-faithful H0 null calibration requires equal-length "
            "chains in chain_id."
        )
    order = np.concatenate(groups)
    try:
        from IPython.display import display
    except ImportError:
        display = print
    namespace = dict(state)
    coordinate = str(state.get("coordinate", "m1"))
    coordinate_name = str(
        state.get(
            "COORDINATE_NAME",
            settings.input.coordinate1_name
            if coordinate == "m1"
            else settings.input.coordinate2_name,
        )
    )
    coordinate_label = str(
        state.get(
            "COORDINATE_LABEL",
            settings.input.coordinate1_label
            if coordinate == "m1"
            else settings.input.coordinate2_label,
        )
    )
    coordinate_unit = str(
        state.get(
            "COORDINATE_UNIT",
            settings.input.coordinate1_unit
            if coordinate == "m1"
            else settings.input.coordinate2_unit,
        )
    )
    if coordinate_name in {"m1", "m2"}:
        feature_scale_channel_label = "mass scale"
        feature_scale_axis_label = r"Feature mass scale $[M_\odot]$"
        feature_collection_label = "mass features"
    else:
        feature_scale_channel_label = "location"
        feature_scale_axis_label = (
            f"Feature location {_label_with_unit(coordinate_label, coordinate_unit)}"
        )
        feature_collection_label = "one-dimensional features"
    namespace.update(
        {
            "FEATURE_SCALE_CHANNEL_LABEL": state.get(
                "FEATURE_SCALE_CHANNEL_LABEL", feature_scale_channel_label
            ),
            "FEATURE_SCALE_AXIS_LABEL": state.get(
                "FEATURE_SCALE_AXIS_LABEL", feature_scale_axis_label
            ),
            "FEATURE_COLLECTION_LABEL": state.get(
                "FEATURE_COLLECTION_LABEL", feature_collection_label
            ),
        }
    )
    reordered_posteriors: dict[str, dict[str, Any]] = {}
    for identifier, posterior in state["feature_draw_posteriors"].items():
        reordered_posteriors[identifier] = {
            name: (
                np.asarray(values)[order]
                if np.asarray(values).ndim > 0
                and np.asarray(values).shape[0] == order.size
                else values
            )
            for name, values in posterior.items()
        }
    parameter_values = np.asarray(store.h0, dtype=float)[order]
    namespace.update(
        {
            "__name__": "posterior_landscape.one_dimensional_h0_notebook",
            "fin": str(directory),
            # Keep the H0 alias for the unchanged legacy source and expose the
            # configured key for parameterized notebook strings.
            "params": {
                "H0": parameter_values,
                settings.association.parameter_name: parameter_values,
            },
            "feature_draw_posteriors": reordered_posteriors,
            "m_high_percentile_samples": np.asarray(
                state["m_high_percentile_samples"], dtype=float
            )[order],
            "trace": SimpleNamespace(
                posterior=SimpleNamespace(
                    sizes={"chain": len(groups), "draw": counts[0]}
                )
            ),
            "RANDOM_SEED": _NOTEBOOK_RANDOM_SEED,
            "_savefig": _savefig_factory(settings),
            "MARGINAL_FILL_ALPHA": 0.48,
            "EXTERNAL_PARAMETER_NAME": settings.association.parameter_name,
            "EXTERNAL_PARAMETER_LABEL": settings.association.parameter_label,
            "EXTERNAL_PARAMETER_UNIT": settings.association.parameter_unit,
            "EXTERNAL_PARAMETER_AXIS_LABEL": _external_parameter_axis_label(
                settings
            ),
            "display": display,
        }
    )
    return namespace


def _feature_label(identifier: str) -> str:
    prefix = identifier[:1]
    number = identifier[1:]
    return rf"${prefix}_{{{number}}}$"


def _run_h0(
    state: dict[str, Any],
    store: DensityStore,
    settings: Settings,
    directory: Path,
) -> None:
    namespace = _h0_namespace(state, store, settings, directory)
    coordinate = str(state.get("coordinate", "m1"))
    order = tuple(state["feature_posterior_order"])
    coordinate_name = str(state.get("COORDINATE_NAME", coordinate))
    tail_id = (
        f"m{state['percentile_label']}"
        if coordinate_name in {"m1", "m2"}
        else f"{coordinate_name}_{state['percentile_label']}"
    )
    labels = {identifier: _feature_label(identifier) for identifier in order}
    labels[tail_id] = str(
        state.get("TAIL_TEX_LABEL", rf"$m_{{{state['percentile_label']}}}$")
    )
    colors: dict[str, str] = {}
    for identifier in order:
        reference_type = state["feature_reference_lookup"][identifier][
            "reference_type"
        ]
        colors[identifier] = {
            "peak": state["PEAK_COLOR"],
            "dip": state["DIP_COLOR"],
            "shoulder": state["SHOULDER_COLOR"],
        }[reference_type]
    colors[tail_id] = "0.30"
    mixed = tuple(
        identifier
        for identifier in order
        if {
            value
            for value in np.asarray(
                state["feature_draw_posteriors"][identifier]["realized_type"],
                dtype=object,
            )
            if value in {"peak", "shoulder"}
        }
        == {"peak", "shoulder"}
    )

    source = _replace_top_level_assignments(
        _external_parameter_source(
            _coordinate_source(_cell_source("cell08_h0.py"), coordinate),
            settings,
        ),
        {
            "H0_FEATURE_ORDER": repr(order + (tail_id,)),
            "H0_FEATURE_LABELS": repr(labels),
            "H0_FEATURE_COLORS": repr(colors),
            "H0_MIXED_MORPHOLOGY_FEATURES": repr(mixed),
            "MI_N_NEIGHBORS": repr(settings.association.knn),
            "MI_N_PERMUTATIONS": repr(settings.association.permutations),
            "MI_RANDOM_SEED": repr(_NOTEBOOK_RANDOM_SEED),
        },
    )
    source = source.replace('"m99.99"', repr(tail_id)).replace(
        "'m99.99'", repr(tail_id)
    )
    _execute(source, namespace, "one-dimensional H0 analysis cell")
    insufficient = [
        identifier
        for identifier in order + (tail_id,)
        if np.count_nonzero(
            np.isfinite(
                np.asarray(
                    namespace["h0_feature_statistics"][identifier]["mass_scale"],
                    dtype=float,
                )
            )
        )
        < 20
    ]
    if insufficient:
        print(
            "WARNING: The notebook production external-parameter figure requires "
            "at least 20 finite feature-scale draws per feature; skipping only "
            "that figure/table "
            "because the following features are below the limit: "
            + ", ".join(insufficient)
        )
        return
    output_source = _external_parameter_source(
        _coordinate_source(_cell_source("cell09_h0_outputs.py"), coordinate),
        settings,
    )
    output_source = output_source.replace('"m99.99"', repr(tail_id)).replace(
        "'m99.99'", repr(tail_id)
    )
    _execute(output_source, namespace, "one-dimensional H0 output cell")


def _write_cooccurrence(
    state: dict[str, Any],
    store: DensityStore,
    settings: Settings,
    directory: Path,
) -> tuple[Path, Path]:
    """Write the same general pairwise location-support test for one axis."""

    order = list(state["feature_posterior_order"])
    h0_enabled = (
        settings.association.enabled != "off" and store.h0 is not None
    )
    records = pairwise_indicator_associations(
        order,
        [
            str(state["feature_reference_lookup"][identifier]["reference_type"])
            for identifier in order
        ],
        [
            np.asarray(
                state["feature_draw_posteriors"][identifier]["located"],
                dtype=bool,
            )
            for identifier in order
        ],
        store.weights,
        h0=(store.h0 if h0_enabled else None),
        chain_id=(store.chain_id if h0_enabled else None),
        permutations=(
            settings.association.permutations
            if h0_enabled
            else 0
        ),
        random_seed=settings.association.random_seed,
    )
    coordinate = str(state.get("coordinate", "m1"))
    stem = f"one_dimensional_{coordinate}_feature_cooccurrence"
    csv_path = directory / f"{stem}.csv"
    fields = list(records[0]) if records else ["ID_A", "ID_B"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    text_path = directory / f"{stem}.txt"
    lines = [
        f"Pairwise {coordinate} feature location-support associations",
        (
            f"{'A':>5}  {'B':>5}  {'P(A&B)':>8}  {'P(B|A)':>8}  "
            f"{'P(A|B)':>8}  {'phi':>8}  {'lift':>8}  {'H0 r_rb':>8}  "
            f"{'p_perm':>8}"
        ),
        "-" * 93,
    ]
    for record in records:
        lines.append(
            f"{record['ID_A']:>5}  {record['ID_B']:>5}  "
            f"{record['P_A_and_B']:>8.1%}  {record['P_B_given_A']:>8.1%}  "
            f"{record['P_A_given_B']:>8.1%}  {record['phi']:>8.3f}  "
            f"{record['lift']:>8.3f}  "
            f"{record['joint_occurrence_h0_rrb']:>8.3f}  "
            f"{record['joint_occurrence_h0_p_perm']:>8.3f}"
        )
    lines.extend(
        [
            "",
            "Indicators are draw-level location support. Coexistence and conditional",
            "probabilities are reported without automatically merging feature labels.",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, csv_path


def run_one_dimensional(settings: Settings) -> OneDimensionalResult:
    """Run or reuse the independent m1 and m2 notebook workflows."""

    directory = settings.output.directory
    directory.mkdir(parents=True, exist_ok=True)
    if settings.output.overwrite:
        _clear_generated_one_d_outputs(directory)

    state_paths = {
        "m1": directory / _STATE_NAME,
        "m2": directory / _SECONDARY_STATE_NAME,
    }
    manifest_paths = {
        "m1": directory / _MANIFEST_NAME,
        "m2": directory / _SECONDARY_MANIFEST_NAME,
    }
    with open_density_store(
        settings.input.file,
        input_settings=settings.input,
        association_settings=settings.association,
    ) as store:
        coordinate_names = {
            "m1": settings.input.coordinate1_name,
            "m2": settings.input.coordinate2_name,
        }
        if not np.allclose(
            store.weights,
            np.full(store.number_draws, 1.0 / store.number_draws),
            rtol=1.0e-10,
            atol=1.0e-14,
        ):
            raise ValueError(
                "The notebook-faithful 1D workflow requires equal-weight "
                "posterior draws. Resample the posterior before using 1d/both mode."
            )

        core_signature = _density_signature(store, settings)
        legacy_signatures = {
            version: _density_signature(store, settings, package_version=version)
            for version in ("0.6.1", "0.6.2", "0.8.0")
        }
        manifests = {
            coordinate: _read_manifest(manifest_paths[coordinate])
            for coordinate in ("m1", "m2")
        }
        reusable: dict[str, bool] = {}
        for coordinate in ("m1", "m2"):
            manifest = manifests[coordinate]
            matches = _core_signature_matches(
                manifest, core_signature, legacy_signatures
            )
            reusable[coordinate] = bool(
                manifest
                and manifest.get("completed_core")
                and matches
                and state_paths[coordinate].is_file()
                and not settings.output.overwrite
            )
            if (
                state_paths[coordinate].exists()
                and not reusable[coordinate]
                and not settings.output.overwrite
            ):
                raise FileExistsError(
                    f"The output directory contains {coordinate} one-dimensional "
                    "results for different input or settings. Set overwrite=true "
                    "or choose another output directory."
                )

        states: dict[str, dict[str, Any]] = {}
        for coordinate in ("m1", "m2"):
            if reusable[coordinate]:
                states[coordinate] = _read_pickle(state_paths[coordinate])
                states[coordinate].setdefault("coordinate", coordinate)
                print(
                    "Reusing completed one-dimensional "
                    f"{coordinate_names[coordinate]} feature results."
                )
                manifest = manifests[coordinate]
                if manifest and manifest.get("core_signature") != core_signature:
                    manifest = dict(manifest)
                    manifest["package_version"] = __version__
                    manifest["core_signature"] = core_signature
                    manifest["coordinate"] = coordinate
                    manifests[coordinate] = manifest
                    _write_manifest(manifest_paths[coordinate], manifest)

        if not all(reusable.values()):
            print(
                "Constructing the independent "
                f"{coordinate_names['m1']} and {coordinate_names['m2']} "
                "marginals from "
                f"{store.number_draws} posterior draws."
            )
            pm1, dm1, pm2, dm2 = _component_mass_marginals(store, settings)
            inputs = {
                "m1": (store.m1, pm1, dm1),
                "m2": (store.m2, pm2, dm2),
            }
            for coordinate in ("m1", "m2"):
                if reusable[coordinate]:
                    continue
                print(
                    "Running the independent one-dimensional "
                    f"{coordinate_names[coordinate]} "
                    "feature finder."
                )
                masses, samples, widths = inputs[coordinate]
                masses, samples, widths, resampled = (
                    _geometric_one_dimensional_input(masses, samples, widths)
                )
                if resampled:
                    print(
                        f"Resampled the {coordinate_names[coordinate]} marginal from "
                        f"{inputs[coordinate][0].size} positive input nodes to "
                        f"{masses.size} geometrically spaced analysis nodes."
                    )
                state = _notebook_core(
                    masses,
                    samples,
                    widths,
                    settings,
                    directory,
                    coordinate=coordinate,
                )
                states[coordinate] = state
                _write_pickle_atomic(state_paths[coordinate], state)
                manifest = {
                    "completed_core": True,
                    "package_version": __version__,
                    "core_signature": core_signature,
                    "coordinate": coordinate,
                    "coordinate_name": (
                        settings.input.coordinate1_name
                        if coordinate == "m1"
                        else settings.input.coordinate2_name
                    ),
                    "coordinate_label": (
                        settings.input.coordinate1_label
                        if coordinate == "m1"
                        else settings.input.coordinate2_label
                    ),
                    "input_file": str(settings.input.file),
                    "number_draws": store.number_draws,
                    "measure": settings.one_dimensional.measure,
                    "upper_tail_percentile": (
                        settings.one_dimensional.upper_tail_percentile
                    ),
                    "features": list(state["feature_posterior_order"]),
                    "notebook_random_seed": _NOTEBOOK_RANDOM_SEED,
                    "notebook_cells": {
                        "core": [
                            "cell01_discovery.py",
                            "cell02_bands.py",
                            "cell03_draws.py",
                            "cell04_figure.py",
                            "cell05_summaries.py",
                        ],
                        "h0": ["cell08_h0.py", "cell09_h0_outputs.py"],
                        "paper_specific_diagnostics_not_run": [
                            "cell06_diagnostics.py",
                            "cell07_shoulder_diagnostics.py",
                        ],
                    },
                    "h0_signature": None,
                    "h0_available": False,
                }
                manifests[coordinate] = manifest
                _write_manifest(manifest_paths[coordinate], manifest)

        for coordinate in ("m1", "m2"):
            _write_cooccurrence(
                states[coordinate], store, settings, directory
            )

        association_enabled = settings.association.enabled != "off"
        h0_analyzed = False
        if not association_enabled:
            print("External-parameter association analysis is disabled.")
        elif store.h0 is None:
            print(
                "WARNING: No aligned dataset named "
                f"'{settings.association.dataset}' was found. Continuing without "
                f"association analysis for {settings.association.parameter_name}."
            )
        else:
            current_h0_signature = _h0_signature(store, settings)
            for coordinate in ("m1", "m2"):
                manifest = manifests[coordinate] or {}
                if manifest.get("h0_signature") == current_h0_signature:
                    print(
                        "One-dimensional "
                        f"{coordinate_names[coordinate]} external-parameter "
                        "association results "
                        "are already complete."
                    )
                    h0_analyzed = True
                    continue
                if reusable[coordinate]:
                    print(
                        f"Aligned {settings.association.parameter_name} samples "
                        "were added or changed; running only the one-dimensional "
                        f"{coordinate_names[coordinate]} association analysis."
                    )
                _run_h0(states[coordinate], store, settings, directory)
                h0_analyzed = True
                manifest = dict(manifest)
                manifest["h0_signature"] = current_h0_signature
                manifest["h0_available"] = True
                manifests[coordinate] = manifest
                _write_manifest(manifest_paths[coordinate], manifest)

    return OneDimensionalResult(
        directory=directory,
        state=states["m1"],
        secondary_state=states["m2"],
        reused_core=reusable["m1"],
        reused_secondary_core=reusable["m2"],
        h0_analyzed=h0_analyzed,
    )
