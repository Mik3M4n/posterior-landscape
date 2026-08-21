"""Independent 1D versus projected 2D feature comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .association import (
    _binary_association,
    _circular_h0_nulls,
    ksg_conditional_mutual_information,
    ksg_mutual_information,
)
from .config import Settings
from .io import h5py


@dataclass
class _ProjectedFeature:
    identifier: str
    parent_identifier: str
    kind: str
    selected: np.ndarray
    bounds: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    projection: np.ndarray


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _weighted_quantile(
    values: np.ndarray, probabilities: tuple[float, ...], weights: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(finite):
        return np.full(len(probabilities), np.nan)
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0.0:
        return np.full(len(probabilities), np.nan)
    targets = np.asarray(probabilities) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return values[np.minimum(indices, values.size - 1)]


def _configuration_names(group: Any) -> tuple[str, ...]:
    raw = _text(group.attrs.get("location_configuration_names", ""))
    return tuple(item for item in raw.split(",") if item)


def _load_projected_features(
    results: Any, coordinate: str = "m1"
) -> list[_ProjectedFeature]:
    if coordinate not in {"m1", "m2"}:
        raise ValueError("coordinate must be 'm1' or 'm2'.")
    projection_name = f"projection_{coordinate}"
    bound_slice = slice(0, 2) if coordinate == "m1" else slice(2, 4)
    output: list[_ProjectedFeature] = []
    specifications = (
        ("features", "measurement_valid"),
        ("events", "region_valid"),
        ("shoulders", "region_valid"),
    )
    for category, validity_name in specifications:
        if category not in results:
            continue
        for identifier in results[category]:
            group = results[category][identifier]
            if not all(
                name in group
                for name in (
                    "present",
                    validity_name,
                    "projected_bounds_mass",
                    "mass_centroid",
                    projection_name,
                )
            ):
                continue
            kind = _text(group.attrs.get("type", "shoulder"))
            selected = np.asarray(group["present"], dtype=bool) & np.asarray(
                group[validity_name], dtype=bool
            )
            bounds = np.asarray(
                group["projected_bounds_mass"], dtype=float
            )[:, bound_slice]
            centroid = np.asarray(group["mass_centroid"], dtype=float)
            mu1 = centroid[:, 0]
            mu2 = centroid[:, 1]
            projection = np.asarray(group[projection_name], dtype=float)
            status = _text(
                group.attrs.get("location_configuration_status", "unimodal")
            )
            names = _configuration_names(group)
            if (
                status == "robust_multimodal"
                and names
                and "location_configuration" in group
            ):
                labels = np.asarray(group["location_configuration"], dtype=int)
                for index, name in enumerate(names):
                    output.append(
                        _ProjectedFeature(
                            identifier=f"{identifier}-{name}",
                            parent_identifier=identifier,
                            kind=kind,
                            selected=selected & (labels == index),
                            bounds=bounds,
                            mu1=mu1,
                            mu2=mu2,
                            projection=projection,
                        )
                    )
            else:
                output.append(
                    _ProjectedFeature(
                        identifier=identifier,
                        parent_identifier=identifier,
                        kind=kind,
                        selected=selected,
                        bounds=bounds,
                        mu1=mu1,
                        mu2=mu2,
                        projection=projection,
                    )
                )
    return output


def _compatible(one_d_type: str, two_d_type: str) -> bool:
    if one_d_type == "dip":
        return two_d_type in {"pit", "valley"}
    return two_d_type in {"peak", "ridge", "shoulder"}


def _log_interval_metrics(
    first_left: float,
    first_right: float,
    second_left: float,
    second_right: float,
) -> tuple[float, float, float]:
    if not np.all(
        np.isfinite([first_left, first_right, second_left, second_right])
    ) or min(first_left, second_left) <= 0.0:
        return math.nan, math.nan, math.nan
    a0, a1, b0, b1 = map(
        math.log, (first_left, first_right, second_left, second_right)
    )
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    first_width = a1 - a0
    second_width = b1 - b0
    return (
        intersection / union if union > 0.0 else math.nan,
        intersection / first_width if first_width > 0.0 else math.nan,
        intersection / second_width if second_width > 0.0 else math.nan,
    )


def _draw_interval_jaccard(
    first_left: np.ndarray,
    first_right: np.ndarray,
    second_left: np.ndarray,
    second_right: np.ndarray,
) -> np.ndarray:
    a0 = np.log(first_left)
    a1 = np.log(first_right)
    b0 = np.log(second_left)
    b1 = np.log(second_right)
    intersection = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan),
        where=union > 0.0,
    )


def _projection_in_one_d_measure(
    projection: np.ndarray,
    m_grid: np.ndarray,
    *,
    two_d_measure: str,
    one_d_measure: str,
    log_base: float,
) -> np.ndarray:
    values = np.asarray(projection, dtype=float)
    if two_d_measure == "log":
        values = values / math.log(log_base)
        if one_d_measure == "linear":
            values = values / m_grid[None, :]
    elif one_d_measure == "log":
        values = values * m_grid[None, :]
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)


def _projection_on_one_dimensional_grid(
    projection: np.ndarray,
    source_grid: np.ndarray,
    target_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate 2D projections onto the independent 1D analysis grid.

    Geometrically spaced GW inputs use identical source and target grids and
    are returned without a numerical change.  A distinct target occurs only
    when the 1D compatibility layer has resampled a positive non-geometric
    input grid.
    """

    values = np.asarray(projection, dtype=float)
    source = np.asarray(source_grid, dtype=float)
    target = np.asarray(target_grid, dtype=float)
    if np.array_equal(source, target):
        return values
    if (
        values.ndim != 2
        or values.shape[1] != source.size
        or target[0] < source[0]
        or target[-1] > source[-1]
    ):
        raise RuntimeError(
            "Cannot align the projected 2D density with the independent "
            "one-dimensional analysis grid."
        )
    aligned = np.empty((values.shape[0], target.size), dtype=float)
    for index, row in enumerate(values):
        aligned[index] = np.interp(target, source, row)
    return np.where(np.isfinite(aligned) & (aligned > 0.0), aligned, 0.0)


def _projection_capture(
    projection: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    state: dict[str, Any],
) -> np.ndarray:
    if state["MASS_DENSITY_MEASURE"] == "linear":
        lower = left
        upper = right
    else:
        lower = np.log(left)
        upper = np.log(right)
    edges = np.asarray(state["MASS_MEASURE_CELL_EDGES"], dtype=float)
    cell_widths = np.asarray(state["MASS_MEASURE_CELL_WIDTHS"], dtype=float)
    weights = np.asarray(state["MASS_MEASURE_WIDTHS"], dtype=float)
    overlap = np.maximum(
        0.0,
        np.minimum(edges[None, 1:], upper[:, None])
        - np.maximum(edges[None, :-1], lower[:, None]),
    )
    partial_weights = weights[None, :] * overlap / cell_widths[None, :]
    denominator = np.einsum("bi,i->b", projection, weights, optimize=True)
    numerator = np.einsum("bi,bi->b", projection, partial_weights, optimize=True)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0.0,
    )


def _one_d_mass_scale(reference_type: str, posterior: dict[str, Any]) -> np.ndarray:
    name = "mu_deficit" if reference_type == "dip" else "mu_hat"
    return np.asarray(posterior[name], dtype=float)


def _calibrated_mutual_information(
    h0: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    nulls: list[np.ndarray],
    *,
    k: int,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[float, float]:
    """Return null-calibrated I(H0; values) and its finite-sample offset."""

    values = np.asarray(values, dtype=float)
    selected = values[indices]
    matrix = selected[:, None] if selected.ndim == 1 else selected
    if indices.size <= max(k + 2, 20) or not np.any(
        np.ptp(matrix, axis=0) > 0.0
    ):
        return math.nan, math.nan
    raw = ksg_mutual_information(h0[indices], selected, k=k)
    def estimate_null(null: np.ndarray) -> float:
        return ksg_mutual_information(null[indices], selected, k=k)

    estimates = (
        list(executor.map(estimate_null, nulls))
        if executor is not None
        else [estimate_null(null) for null in nulls]
    )
    null_values = np.asarray(estimates, dtype=float)
    offset = float(np.nanmedian(null_values)) if null_values.size else 0.0
    return max(0.0, float(raw) - offset), offset


def _calibrated_conditional_information(
    h0: np.ndarray,
    values: np.ndarray,
    conditioning: np.ndarray,
    indices: np.ndarray,
    nulls: list[np.ndarray],
    *,
    k: int,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[float, float]:
    """Return null-calibrated I(H0; values | conditioning) and its offset."""

    values = np.asarray(values, dtype=float)
    conditioning = np.asarray(conditioning, dtype=float)
    selected_values = values[indices]
    selected_conditioning = conditioning[indices]
    if indices.size <= max(k + 2, 30):
        return math.nan, math.nan
    raw = ksg_conditional_mutual_information(
        h0[indices], selected_values, selected_conditioning, k=k
    )
    def estimate_null(null: np.ndarray) -> float:
        return ksg_conditional_mutual_information(
            null[indices], selected_values, selected_conditioning, k=k
        )

    estimates = (
        list(executor.map(estimate_null, nulls))
        if executor is not None
        else [estimate_null(null) for null in nulls]
    )
    null_values = np.asarray(estimates, dtype=float)
    offset = float(np.nanmedian(null_values)) if null_values.size else 0.0
    return max(0.0, float(raw) - offset), offset


def _chain_spearman_text(
    h0: np.ndarray,
    values: np.ndarray,
    common: np.ndarray,
    chain_id: np.ndarray,
) -> tuple[str, str]:
    """Compact exact per-chain draw counts and Spearman coefficients."""

    count_parts: list[str] = []
    rho_parts: list[str] = []
    for label in np.unique(chain_id):
        selected = common & (chain_id == label)
        number = int(np.count_nonzero(selected))
        count_parts.append(f"{label}:{number}")
        if number > 2 and np.ptp(values[selected]) > 0.0:
            rho = float(spearmanr(h0[selected], values[selected]).statistic)
            rho_parts.append(f"{label}:{rho:.8g}")
        else:
            rho_parts.append(f"{label}:nan")
    return ";".join(count_parts), ";".join(rho_parts)


def _full_h0_information(
    h0: np.ndarray,
    chain_id: np.ndarray,
    one_mu: np.ndarray,
    two_mu1: np.ndarray,
    two_mu2: np.ndarray,
    common: np.ndarray,
    nulls: list[np.ndarray],
    settings: Any,
    rng: np.random.Generator,
    workers: int = 1,
) -> dict[str, Any]:
    """Complete 1D-versus-2D H0 comparison on one identical draw subset."""

    common = (
        np.asarray(common, dtype=bool)
        & np.isfinite(h0)
        & np.isfinite(one_mu)
        & np.isfinite(two_mu1)
        & np.isfinite(two_mu2)
    )
    indices = np.flatnonzero(common)
    one_mu = np.asarray(one_mu, dtype=float)
    two_mu1 = np.asarray(two_mu1, dtype=float)
    two_mu2 = np.asarray(two_mu2, dtype=float)
    full_two_d = np.column_stack([two_mu1, two_mu2])
    all_m1 = np.column_stack([one_mu, two_mu1])
    all_scales = np.column_stack([one_mu, two_mu1, two_mu2])

    record: dict[str, Any] = {}
    pooled_values = (
        ("h0_one_d_rho", one_mu),
        ("h0_two_d_mu1_rho", two_mu1),
        ("h0_two_d_mu2_rho", two_mu2),
    )
    for name, values in pooled_values:
        record[name] = (
            float(spearmanr(h0[common], values[common]).statistic)
            if indices.size > 2 and np.ptp(values[common]) > 0.0
            else math.nan
        )

    information_values = (
        ("h0_one_d_mi_bits", one_mu),
        ("h0_two_d_mu1_mi_bits", two_mu1),
        ("h0_two_d_mu2_mi_bits", two_mu2),
        ("h0_full_2d_mi_bits", full_two_d),
        ("h0_all_m1_mi_bits", all_m1),
        ("h0_all_scales_mi_bits", all_scales),
    )
    conditional_values = (
        ("h0_full_2d_given_1d_cmi_bits", full_two_d, one_mu),
        ("h0_mu2_given_all_m1_cmi_bits", two_mu2, all_m1),
        ("h0_one_d_given_full_2d_cmi_bits", one_mu, full_two_d),
    )
    legacy_conditionals = (
        ("h0_two_d_mu1_given_one_d_cmi_bits", two_mu1, one_mu),
        ("h0_one_d_given_two_d_mu1_cmi_bits", one_mu, two_mu1),
    )
    offsets: dict[str, float] = {}
    resampled: dict[str, list[float]] = {
        name: [] for name, _, _ in conditional_values
    }
    subsets: list[np.ndarray] = []
    if indices.size > max(settings.knn + 2, 30):
        subset_size = max(settings.knn + 3, int(0.8 * indices.size))
        subsets = [
            rng.choice(indices, size=subset_size, replace=False)
            for _ in range(settings.uncertainty_resamples)
        ]

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for name, values in information_values:
            information, _ = _calibrated_mutual_information(
                h0,
                values,
                indices,
                nulls,
                k=settings.knn,
                executor=executor,
            )
            record[name] = information

        for name, values, conditioning in (
            *conditional_values,
            *legacy_conditionals,
        ):
            information, offset = _calibrated_conditional_information(
                h0,
                values,
                conditioning,
                indices,
                nulls,
                k=settings.knn,
                executor=executor,
            )
            record[name] = information
            offsets[name] = offset

        for name, values, conditioning in conditional_values:
            def estimate_subset(subset: np.ndarray) -> float:
                return ksg_conditional_mutual_information(
                    h0[subset], values[subset], conditioning[subset], k=settings.knn
                )

            estimates = list(executor.map(estimate_subset, subsets))
            resampled[name] = [
                max(0.0, estimate - offsets[name])
                for estimate in estimates
                if np.isfinite(estimate)
            ]
    for name, _, _ in conditional_values:
        if resampled[name]:
            lower, upper = np.quantile(resampled[name], (0.05, 0.95))
        else:
            lower = upper = math.nan
        record[f"{name}_lower"] = float(lower)
        record[f"{name}_upper"] = float(upper)

    record["h0_full_2d_given_1d_chain_rule_bits"] = (
        record["h0_all_scales_mi_bits"] - record["h0_one_d_mi_bits"]
    )
    record["h0_mu2_given_all_m1_chain_rule_bits"] = (
        record["h0_all_scales_mi_bits"] - record["h0_all_m1_mi_bits"]
    )
    record["h0_one_d_given_full_2d_chain_rule_bits"] = (
        record["h0_all_scales_mi_bits"] - record["h0_full_2d_mi_bits"]
    )

    finite_h0 = np.isfinite(h0)
    effect, probability, first, second = _binary_association(
        h0, common, finite_h0, nulls
    )
    record["common_valid_rrb"] = effect
    record["common_valid_p_perm"] = probability
    record["common_valid_n_yes"] = first
    record["common_valid_n_no"] = second

    counts, one_chain = _chain_spearman_text(h0, one_mu, common, chain_id)
    _, mu1_chain = _chain_spearman_text(h0, two_mu1, common, chain_id)
    _, mu2_chain = _chain_spearman_text(h0, two_mu2, common, chain_id)
    record["common_draws_by_chain"] = counts
    record["h0_one_d_rho_by_chain"] = one_chain
    record["h0_two_d_mu1_rho_by_chain"] = mu1_chain
    record["h0_two_d_mu2_rho_by_chain"] = mu2_chain
    return record


def _write_full_h0_text(
    path: Path, rows: list[dict[str, Any]], parameter_name: str = "H0"
) -> None:
    lines = [
        f"Full 1D versus 2D {parameter_name} association on identical "
        "common-valid draws",
        "",
        (
            "Conditional increments use direct null-calibrated nearest-neighbour "
            "conditional mutual information. Brackets give paired 90% resampling "
            "intervals."
        ),
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['one_d_ID']:>4s} -> {row['two_d_ID']:<12s}: "
            f"N={row['number_common_draws']}, P(common)={row['common_support']:.3f}, "
            f"selection r/p={_number(row['common_valid_rrb'])}/"
            f"{_number(row['common_valid_p_perm'])}"
        )
        lines.append(
            f"       {parameter_name} bits: "
            f"1D={_number(row['h0_one_d_mi_bits'])}, "
            f"full2D={_number(row['h0_full_2d_mi_bits'])}, "
            f"all={_number(row['h0_all_scales_mi_bits'])}"
        )
        lines.append(
            "       added full2D|1D="
            f"{_number(row['h0_full_2d_given_1d_cmi_bits'])} "
            f"[{_number(row['h0_full_2d_given_1d_cmi_bits_lower'])}, "
            f"{_number(row['h0_full_2d_given_1d_cmi_bits_upper'])}]; "
            "m2|all-m1="
            f"{_number(row['h0_mu2_given_all_m1_cmi_bits'])} "
            f"[{_number(row['h0_mu2_given_all_m1_cmi_bits_lower'])}, "
            f"{_number(row['h0_mu2_given_all_m1_cmi_bits_upper'])}]; "
            "1D|full2D="
            f"{_number(row['h0_one_d_given_full_2d_cmi_bits'])} "
            f"[{_number(row['h0_one_d_given_full_2d_cmi_bits_lower'])}, "
            f"{_number(row['h0_one_d_given_full_2d_cmi_bits_upper'])}]"
        )
    lines.extend(
        [
            "",
            "These are posterior association diagnostics, conditional on the same "
            "draws having both complete 1D and full 2D feature measurements.",
            "common_valid_rrb/p_perm tests whether membership in that common-valid "
            f"subset itself varies with {parameter_name}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_full_h0_draws(
    path: Path,
    order: list[str] | tuple[str, ...],
    data: dict[str, dict[str, Any]],
    h0: np.ndarray,
    chain_id: np.ndarray,
    weights: np.ndarray,
) -> None:
    identifiers = [identifier for identifier in order if identifier in data]
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("1.0"),
        one_d_ID=np.asarray(identifiers, dtype=str),
        two_d_ID=np.asarray(
            [data[identifier]["two_d_ID"] for identifier in identifiers], dtype=str
        ),
        two_d_parent_ID=np.asarray(
            [data[identifier]["two_d_parent_ID"] for identifier in identifiers],
            dtype=str,
        ),
        two_d_type=np.asarray(
            [data[identifier]["two_d_type"] for identifier in identifiers], dtype=str
        ),
        h0=np.asarray(h0, dtype=float),
        chain_id=np.asarray(chain_id),
        weights=np.asarray(weights, dtype=float),
        common_valid=np.stack(
            [
                np.asarray(data[identifier]["common"], dtype=bool)
                for identifier in identifiers
            ]
        ),
        one_d_mass_scale=np.stack(
            [
                np.asarray(data[identifier]["one_mu"], dtype=float)
                for identifier in identifiers
            ]
        ),
        two_d_mu1=np.stack(
            [
                np.asarray(data[identifier]["two_mu1"], dtype=float)
                for identifier in identifiers
            ]
        ),
        two_d_mu2=np.stack(
            [
                np.asarray(data[identifier]["two_mu2"], dtype=float)
                for identifier in identifiers
            ]
        ),
    )
    temporary.replace(path)


def _select_primary_candidates(
    candidates: list[dict[str, Any]],
    one_d_order: list[str] | tuple[str, ...],
    minimum_support: float,
    coordinate: str = "m1",
) -> dict[str, dict[str, Any]]:
    """Select primary counterparts only from sufficiently supported 2D regions."""

    selected: dict[str, dict[str, Any]] = {}
    for one_id in one_d_order:
        choices = [
            item
            for item in candidates
            if item["one_id"] == one_id
            and item["jaccard"] > 0.0
            and item["two_support"] >= minimum_support
        ]
        if choices:
            selected[one_id] = max(
                choices,
                key=lambda item: (
                    item["jaccard"],
                    -abs(
                        math.log(
                            np.nanmedian(
                                getattr(
                                    item["feature"],
                                    "mu1" if coordinate == "m1" else "mu2",
                                )[
                                    item["feature"].selected
                                ]
                            )
                            / item["one_center"]
                        )
                    ),
                ),
            )
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "---"
    return f"{value:.{digits}g}" if np.isfinite(value) else "---"


def _write_text(
    path: Path,
    rows: list[dict[str, Any]],
    one_d_order: list[str] | tuple[str, ...],
    minimum_support: float,
    coordinate: str = "m1",
    coordinate_name: str | None = None,
    parameter_name: str = "H0",
) -> None:
    coordinate_display = coordinate_name or coordinate
    lines = [
        "Independent 1D features versus draw-adaptive "
        f"{coordinate_display} projections of 2D features",
        "",
        "The primary 2D counterpart maximizes reference-interval Jaccard overlap",
        "within the compatible morphology family after requiring finite-region support",
        f"of at least {minimum_support:.3g}; secondary positive overlaps are retained.",
        "",
    ]
    for row in rows:
        if not row["primary_match"]:
            continue
        lines.append(
            f"{row['one_d_ID']:>4s} -> {row['two_d_ID']:<12s} "
            f"({row['relationship']}): P(common)={_number(row['common_support'])}, "
            f"Jref={_number(row['reference_jaccard'])}, "
            f"Jdraw={_number(row['draw_jaccard_median'])}, "
            f"capture={_number(row['projection_capture_median'])}, "
            f"rho(mu1D,mu2D)={_number(row['mass_scale_spearman'])}"
        )
        if np.isfinite(float(row["h0_joint_mi_bits"])):
            lines.append(
                f"       {parameter_name} bits: "
                f"1D={_number(row['h0_one_d_mi_bits'])}, "
                f"2D={_number(row['h0_two_d_mi_bits'])}, "
                f"joint={_number(row['h0_joint_mi_bits'])}, "
                f"2D|1D={_number(row['h0_two_d_given_one_d_bits'])}, "
                f"1D|2D={_number(row['h0_one_d_given_two_d_bits'])}"
            )
    matched = {row["one_d_ID"] for row in rows if row["primary_match"]}
    unmatched = [identifier for identifier in one_d_order if identifier not in matched]
    for identifier in unmatched:
        lines.append(
            f"{identifier:>4s} -> unmatched: no compatible 2D projected interval has "
            "both positive reference overlap and sufficient finite-region support."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(
    directory: Path,
    rows: list[dict[str, Any]],
    primary_data: dict[tuple[str, str], dict[str, Any]],
    state: dict[str, Any],
    settings: Settings,
    coordinate: str = "m1",
) -> list[Path]:
    if not (settings.plot.write_pdf or settings.plot.write_png):
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    coordinate_display = str(state.get("COORDINATE_NAME", coordinate))

    primary_rows = [row for row in rows if row["primary_match"]]
    one_ids = list(state["feature_posterior_order"])
    two_ids = []
    for row in rows:
        if row["two_d_ID"] not in two_ids:
            two_ids.append(row["two_d_ID"])
    matrix = np.full((len(one_ids), len(two_ids)), np.nan)
    for row in rows:
        matrix[one_ids.index(row["one_d_ID"]), two_ids.index(row["two_d_ID"])] = row[
            "reference_jaccard"
        ]

    figure, axes = plt.subplots(1, 2, figsize=(12.0, max(4.2, 0.45 * len(one_ids))))
    image = axes[0].imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(two_ids)), two_ids, rotation=45, ha="right")
    axes[0].set_yticks(range(len(one_ids)), one_ids)
    axes[0].set_xlabel("2D draw-adaptive feature")
    axes[0].set_ylabel("independent 1D feature")
    axes[0].set_title("Reference projected-interval overlap")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value) and value >= 0.03:
                axes[0].text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    fontsize=8,
                )
    figure.colorbar(
        image, ax=axes[0], label=f"Jaccard overlap in ln({coordinate_display})"
    )

    labels = [f"{row['one_d_ID']}→{row['two_d_ID']}" for row in primary_rows]
    x = np.arange(len(labels))
    width = 0.25
    if primary_rows and any(
        np.isfinite(float(row["h0_joint_mi_bits"])) for row in primary_rows
    ):
        axes[1].bar(
            x - width,
            [row["h0_one_d_mi_bits"] for row in primary_rows],
            width,
            label="1D scale",
        )
        axes[1].bar(
            x,
            [row["h0_two_d_mi_bits"] for row in primary_rows],
            width,
            label="2D projected scale",
        )
        axes[1].bar(
            x + width,
            [row["h0_joint_mi_bits"] for row in primary_rows],
            width,
            label="joint vector",
        )
        axes[1].set_ylabel("calibrated mutual information [bits]")
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].set_title(
            f"Association with {settings.association.parameter_label} "
            "on common valid draws"
        )
    else:
        axes[1].bar(
            x - 0.16,
            [row["draw_jaccard_median"] for row in primary_rows],
            0.32,
            label="moving-window overlap",
        )
        axes[1].bar(
            x + 0.16,
            [row["projection_capture_median"] for row in primary_rows],
            0.32,
            label="2D projection captured",
        )
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_ylabel("posterior median fraction")
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].set_title("Draw-level geometric agreement")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()

    written: list[Path] = []
    suffix = "" if coordinate == "m1" else f"_{coordinate}"
    stem = directory / f"one_two_dimensional_comparison{suffix}"
    if settings.plot.write_pdf:
        path = stem.with_suffix(".pdf")
        figure.savefig(path, bbox_inches="tight")
        written.append(path)
    if settings.plot.write_png:
        path = stem.with_suffix(".png")
        figure.savefig(path, dpi=220, bbox_inches="tight")
        written.append(path)
    plt.close(figure)

    if primary_rows:
        columns = 3
        plot_rows = int(math.ceil(len(primary_rows) / columns))
        figure, axes = plt.subplots(
            plot_rows,
            columns,
            figsize=(11.0, 3.0 * plot_rows),
            squeeze=False,
        )
        m_grid = np.asarray(state["m_grid"], dtype=float)
        for axis, row in zip(axes.ravel(), primary_rows):
            data = primary_data[(row["one_d_ID"], row["two_d_ID"])]
            low, median, high = data["projection_quantiles"]
            axis.fill_between(
                m_grid,
                state["pm1_q05_feature"],
                state["pm1_q95_feature"],
                color="#D77A8A",
                alpha=0.25,
                linewidth=0,
            )
            axis.plot(m_grid, state["pm1_q50_feature"], color="#5C176D", lw=1.5)
            axis.fill_between(m_grid, low, high, color="#2A9D8F", alpha=0.22)
            axis.plot(m_grid, median, color="#2A9D8F", lw=1.5)
            axis.axvspan(
                row["one_d_reference_left"],
                row["one_d_reference_right"],
                color="0.4",
                alpha=0.10,
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            positive = np.concatenate(
                [
                    np.asarray(state["pm1_q95_feature"])[
                        np.asarray(state["pm1_q95_feature"]) > 0.0
                    ],
                    high[high > 0.0],
                ]
            )
            if positive.size:
                axis.set_ylim(max(np.min(positive) * 0.4, 1e-8), np.max(positive) * 2)
            axis.set_title(
                f"{row['one_d_ID']} ↔ {row['two_d_ID']}  "
                f"capture={row['projection_capture_median']:.0%}",
                fontsize=9,
            )
            axis.set_xlabel(rf"$m_{coordinate[-1]}$")
            axis.set_ylabel("normalized density in selected measure")
            axis.grid(alpha=0.18, lw=0.5)
        for axis in axes.ravel()[len(primary_rows) :]:
            axis.axis("off")
        figure.suptitle(
            "Independent 1D spectrum and normalized draw-adaptive "
            "2D feature projections",
            fontsize=12,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        stem = directory / f"one_two_dimensional_projections{suffix}"
        if settings.plot.write_pdf:
            path = stem.with_suffix(".pdf")
            figure.savefig(path, bbox_inches="tight")
            written.append(path)
        if settings.plot.write_png:
            path = stem.with_suffix(".png")
            figure.savefig(path, dpi=220, bbox_inches="tight")
            written.append(path)
        plt.close(figure)
    return written


def _run_secondary_projection_comparison(
    directory: Path,
    results: Any,
    state: dict[str, Any],
    settings: Settings,
    weights: np.ndarray,
    h0: np.ndarray | None,
    chain_id: np.ndarray | None,
    nulls: list[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Path]]:
    """Match the independent m2 catalogue to the m2 views of 2D features."""

    coordinate = "m2"
    projection_grid = np.asarray(results["coordinates/m2"], dtype=float)
    m_grid = np.asarray(state["m_grid"], dtype=float)
    two_d_measure = _text(results.attrs.get("feature_measure", "log"))
    log_base = float(results.attrs.get("log_base", settings.analysis.log_base))
    projected = _load_projected_features(results, coordinate)
    projected.sort(
        key=lambda feature: np.nanmedian(feature.mu2[feature.selected])
        if np.any(feature.selected)
        else np.inf
    )

    candidates: list[dict[str, Any]] = []
    for one_id in state["feature_posterior_order"]:
        reference = state["feature_reference_lookup"][one_id]
        one_type = reference["reference_type"]
        one_band = reference["band"]
        for feature in projected:
            if not _compatible(one_type, feature.kind) or not np.any(feature.selected):
                continue
            selected_weights = weights[feature.selected]
            bounds = feature.bounds[feature.selected]
            left = _weighted_quantile(bounds[:, 0], (0.5,), selected_weights)[0]
            right = _weighted_quantile(bounds[:, 1], (0.5,), selected_weights)[0]
            jaccard, one_coverage, two_coverage = _log_interval_metrics(
                float(one_band["left"]),
                float(one_band["right"]),
                float(left),
                float(right),
            )
            candidates.append(
                {
                    "one_id": one_id,
                    "one_type": one_type,
                    "one_left": float(one_band["left"]),
                    "one_center": float(one_band["center"]),
                    "one_right": float(one_band["right"]),
                    "feature": feature,
                    "two_left": float(left),
                    "two_right": float(right),
                    "jaccard": jaccard,
                    "one_coverage": one_coverage,
                    "two_coverage": two_coverage,
                    "two_support": float(np.sum(weights[feature.selected])),
                }
            )
    primary_by_one = _select_primary_candidates(
        candidates,
        state["feature_posterior_order"],
        settings.analysis.minimum_support,
        coordinate=coordinate,
    )
    primary_counts: dict[str, int] = {}
    for item in primary_by_one.values():
        identifier = item["feature"].identifier
        primary_counts[identifier] = primary_counts.get(identifier, 0) + 1

    rows: list[dict[str, Any]] = []
    primary_data: dict[tuple[str, str], dict[str, Any]] = {}
    matched_data: dict[str, dict[str, Any]] = {}
    for item in candidates:
        one_id = item["one_id"]
        feature = item["feature"]
        is_primary = primary_by_one.get(one_id) is item
        positive_overlaps = sum(
            other["one_id"] == one_id and other["jaccard"] > 0.0
            for other in candidates
        )
        base = {
            "one_d_ID": one_id,
            "one_d_type": item["one_type"],
            "two_d_ID": feature.identifier,
            "two_d_parent_ID": feature.parent_identifier,
            "two_d_type": feature.kind,
            "primary_match": is_primary,
            "relationship": "unmatched",
            "overlapping_two_d_count": positive_overlaps,
            "one_d_reference_left": item["one_left"],
            "one_d_reference_center": item["one_center"],
            "one_d_reference_right": item["one_right"],
            "two_d_projected_left_median": item["two_left"],
            "two_d_projected_right_median": item["two_right"],
            "reference_jaccard": item["jaccard"],
            "reference_one_d_coverage": item["one_coverage"],
            "reference_two_d_coverage": item["two_coverage"],
            "one_d_bounded_support": math.nan,
            "two_d_region_support": item["two_support"],
            "common_support": math.nan,
            "number_common_draws": 0,
            "draw_jaccard_q05": math.nan,
            "draw_jaccard_median": math.nan,
            "draw_jaccard_q95": math.nan,
            "projection_capture_q05": math.nan,
            "projection_capture_median": math.nan,
            "projection_capture_q95": math.nan,
            "mass_scale_log_ratio_q05": math.nan,
            "mass_scale_log_ratio_median": math.nan,
            "mass_scale_log_ratio_q95": math.nan,
            "mass_scale_spearman": math.nan,
            "h0_one_d_rho": math.nan,
            "h0_one_d_mi_bits": math.nan,
            "h0_two_d_rho": math.nan,
            "h0_two_d_mi_bits": math.nan,
            "h0_joint_mi_bits": math.nan,
            "h0_two_d_given_one_d_bits": math.nan,
            "h0_one_d_given_two_d_bits": math.nan,
        }
        if not is_primary:
            base["relationship"] = (
                "unmatched"
                if one_id not in primary_by_one
                else "secondary overlap"
                if item["jaccard"] > 0.0
                else "no overlap"
            )
            rows.append(base)
            continue

        posterior = state["feature_draw_posteriors"][one_id]
        bounded = np.asarray(posterior["bounded"], dtype=bool)
        left = np.asarray(posterior["left"], dtype=float)
        right = np.asarray(posterior["right"], dtype=float)
        one_mu = _one_d_mass_scale(item["one_type"], posterior)
        two_mu = feature.mu2
        common = (
            bounded
            & feature.selected
            & np.isfinite(left)
            & np.isfinite(right)
            & (left > 0.0)
            & (right > left)
            & np.isfinite(one_mu)
            & (one_mu > 0.0)
            & np.isfinite(two_mu)
            & (two_mu > 0.0)
            & np.isfinite(feature.mu1)
            & (feature.mu1 > 0.0)
            & np.isfinite(feature.bounds[:, 0])
            & np.isfinite(feature.bounds[:, 1])
            & (feature.bounds[:, 0] > 0.0)
            & (feature.bounds[:, 1] > feature.bounds[:, 0])
        )
        indices = np.flatnonzero(common)
        common_weights = weights[common]
        jaccard = _draw_interval_jaccard(
            left[common],
            right[common],
            feature.bounds[common, 0],
            feature.bounds[common, 1],
        )
        converted = _projection_in_one_d_measure(
            _projection_on_one_dimensional_grid(
                feature.projection[common], projection_grid, m_grid
            ),
            m_grid,
            two_d_measure=two_d_measure,
            one_d_measure=state["MASS_DENSITY_MEASURE"],
            log_base=log_base,
        )
        capture = _projection_capture(converted, left[common], right[common], state)
        log_ratio = np.log(two_mu[common] / one_mu[common])
        j_q = _weighted_quantile(jaccard, (0.05, 0.5, 0.95), common_weights)
        c_q = _weighted_quantile(capture, (0.05, 0.5, 0.95), common_weights)
        r_q = _weighted_quantile(log_ratio, (0.05, 0.5, 0.95), common_weights)
        mass_rho = (
            float(spearmanr(one_mu[common], two_mu[common]).statistic)
            if indices.size > 2
            else math.nan
        )
        if h0 is not None and chain_id is not None:
            one_mi, _ = _calibrated_mutual_information(
                h0, one_mu, indices, nulls, k=settings.association.knn
            )
            two_mi, _ = _calibrated_mutual_information(
                h0, two_mu, indices, nulls, k=settings.association.knn
            )
            joint_mi, _ = _calibrated_mutual_information(
                h0,
                np.column_stack([one_mu, two_mu]),
                indices,
                nulls,
                k=settings.association.knn,
            )
            two_given_one, _ = _calibrated_conditional_information(
                h0,
                two_mu,
                one_mu,
                indices,
                nulls,
                k=settings.association.knn,
            )
            one_given_two, _ = _calibrated_conditional_information(
                h0,
                one_mu,
                two_mu,
                indices,
                nulls,
                k=settings.association.knn,
            )
            base.update(
                {
                    "h0_one_d_rho": float(
                        spearmanr(h0[common], one_mu[common]).statistic
                    ),
                    "h0_one_d_mi_bits": one_mi,
                    "h0_two_d_rho": float(
                        spearmanr(h0[common], two_mu[common]).statistic
                    ),
                    "h0_two_d_mi_bits": two_mi,
                    "h0_joint_mi_bits": joint_mi,
                    "h0_two_d_given_one_d_bits": two_given_one,
                    "h0_one_d_given_two_d_bits": one_given_two,
                }
            )

        normalized = converted.copy()
        norm = np.einsum(
            "bi,i->b",
            normalized,
            np.asarray(state["MASS_MEASURE_WIDTHS"]),
            optimize=True,
        )
        valid_norm = norm > 0.0
        normalized[valid_norm] /= norm[valid_norm, None]
        primary_data[(one_id, feature.identifier)] = {
            "projection_quantiles": (
                np.quantile(
                    normalized[valid_norm], (0.05, 0.5, 0.95), axis=0
                )
                if np.any(valid_norm)
                else np.full((3, m_grid.size), np.nan)
            )
        }
        merged = primary_counts.get(feature.identifier, 0) > 1
        split = positive_overlaps > 1
        base.update(
            {
                "relationship": (
                    "complex overlap"
                    if merged and split
                    else "many 1D to one 2D"
                    if merged
                    else "one 1D to many 2D"
                    if split
                    else "direct"
                ),
                "one_d_bounded_support": float(np.mean(bounded)),
                "common_support": float(np.sum(weights[common])),
                "number_common_draws": int(np.count_nonzero(common)),
                "draw_jaccard_q05": j_q[0],
                "draw_jaccard_median": j_q[1],
                "draw_jaccard_q95": j_q[2],
                "projection_capture_q05": c_q[0],
                "projection_capture_median": c_q[1],
                "projection_capture_q95": c_q[2],
                "mass_scale_log_ratio_q05": r_q[0],
                "mass_scale_log_ratio_median": r_q[1],
                "mass_scale_log_ratio_q95": r_q[2],
                "mass_scale_spearman": mass_rho,
            }
        )
        rows.append(base)
        matched_data[one_id] = {
            "two_d_ID": feature.identifier,
            "two_d_parent_ID": feature.parent_identifier,
            "two_d_type": feature.kind,
            "common": common,
            "one_mu": one_mu,
            "two_mu1": feature.mu1,
            "two_mu2": feature.mu2,
        }

    order = list(state["feature_posterior_order"])
    rows.sort(
        key=lambda row: (
            order.index(row["one_d_ID"]),
            not row["primary_match"],
            -float(row["reference_jaccard"]),
        )
    )
    csv_path = directory / "one_two_dimensional_m2_feature_comparison.csv"
    text_path = directory / "one_two_dimensional_m2_feature_comparison.txt"
    _write_csv(csv_path, rows)
    _write_text(
        text_path,
        rows,
        state["feature_posterior_order"],
        settings.analysis.minimum_support,
        coordinate="m2",
        coordinate_name=str(state.get("COORDINATE_NAME", "m2")),
        parameter_name=settings.association.parameter_name,
    )
    figures = _write_figures(
        directory,
        rows,
        primary_data,
        state,
        settings,
        coordinate="m2",
    )
    return rows, matched_data, [csv_path, text_path, *figures]


def _write_feature_family_comparison(
    directory: Path,
    results: Any,
    primary_m1: dict[str, dict[str, Any]],
    primary_m2: dict[str, dict[str, Any]],
    primary_state: dict[str, Any],
    secondary_state: dict[str, Any],
    weights: np.ndarray,
    h0: np.ndarray | None,
    nulls: list[np.ndarray],
    settings: Settings,
) -> list[Path]:
    """Treat m1/m2 projections as correlated views of each 2D identity."""

    families: list[dict[str, Any]] = []
    common_identities = sorted(
        {item["two_d_ID"] for item in primary_m1.values()}
        & {item["two_d_ID"] for item in primary_m2.values()}
    )
    for identity in common_identities:
        first_choices = [
            (one_id, item)
            for one_id, item in primary_m1.items()
            if item["two_d_ID"] == identity
        ]
        second_choices = [
            (one_id, item)
            for one_id, item in primary_m2.items()
            if item["two_d_ID"] == identity
        ]
        first_id, first, second_id, second = max(
            (
                (first_id, first, second_id, second)
                for first_id, first in first_choices
                for second_id, second in second_choices
            ),
            key=lambda choice: np.count_nonzero(
                np.asarray(choice[1]["common"], dtype=bool)
                & np.asarray(choice[3]["common"], dtype=bool)
            ),
        )
        families.append(
            {
                "family_ID": identity,
                "parent_ID": first["two_d_parent_ID"],
                "type": first["two_d_type"],
                "m1_1d_ID": first_id,
                "m2_1d_ID": second_id,
                "m1_candidate_IDs": ";".join(
                    sorted(item[0] for item in first_choices)
                ),
                "m2_candidate_IDs": ";".join(
                    sorted(item[0] for item in second_choices)
                ),
                "common": np.asarray(first["common"], dtype=bool)
                & np.asarray(second["common"], dtype=bool),
                "x1": np.asarray(first["one_mu"], dtype=float),
                "x2": np.asarray(second["one_mu"], dtype=float),
                "y1": np.asarray(first["two_mu1"], dtype=float),
                "y2": np.asarray(first["two_mu2"], dtype=float),
            }
        )

    if "global_tail" in results:
        tail = results["global_tail"]
        x1 = np.asarray(primary_state["m_high_percentile_samples"], dtype=float)
        x2 = np.asarray(secondary_state["m_high_percentile_samples"], dtype=float)
        y1 = np.asarray(tail["m1_scale"], dtype=float)
        y2 = np.asarray(tail["m2_scale"], dtype=float)
        finite = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y1) & np.isfinite(y2)
        percentile = float(tail.attrs.get("probability", 0.999))
        families.append(
            {
                "family_ID": "global_tail_scale",
                "parent_ID": "global_tail_scale",
                "type": "global_tail_scale",
                "m1_1d_ID": f"m1_{100.0 * percentile:g}",
                "m2_1d_ID": f"m2_{100.0 * percentile:g}",
                "m1_candidate_IDs": f"m1_{100.0 * percentile:g}",
                "m2_candidate_IDs": f"m2_{100.0 * percentile:g}",
                "common": finite,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
            }
        )

    rows: list[dict[str, Any]] = []
    for family in families:
        common = (
            family["common"]
            & np.isfinite(family["x1"])
            & np.isfinite(family["x2"])
            & np.isfinite(family["y1"])
            & np.isfinite(family["y2"])
        )
        indices = np.flatnonzero(common)
        x = np.column_stack([family["x1"], family["x2"]])
        y = np.column_stack([family["y1"], family["y2"]])
        all_scales = np.column_stack([x, y])
        row: dict[str, Any] = {
            "family_ID": family["family_ID"],
            "parent_ID": family["parent_ID"],
            "type": family["type"],
            "m1_1d_ID": family["m1_1d_ID"],
            "m2_1d_ID": family["m2_1d_ID"],
            "m1_candidate_IDs": family["m1_candidate_IDs"],
            "m2_candidate_IDs": family["m2_candidate_IDs"],
            "common_support": float(np.sum(weights[common])),
            "number_common_draws": int(indices.size),
        }
        variables = {
            "m1_1d": family["x1"],
            "m2_1d": family["x2"],
            "mu1_2d": family["y1"],
            "mu2_2d": family["y2"],
        }
        for first_name, first_values in variables.items():
            for second_name, second_values in variables.items():
                key = f"rho_{first_name}_{second_name}"
                row[key] = (
                    float(
                        spearmanr(
                            first_values[common], second_values[common]
                        ).statistic
                    )
                    if indices.size > 2
                    and np.ptp(first_values[common]) > 0.0
                    and np.ptp(second_values[common]) > 0.0
                    else math.nan
                )
        if h0 is not None:
            for name, values in (
                ("h0_1d_pair_mi_bits", x),
                ("h0_2d_vector_mi_bits", y),
                ("h0_all_scales_mi_bits", all_scales),
            ):
                row[name], _ = _calibrated_mutual_information(
                    h0,
                    values,
                    indices,
                    nulls,
                    k=settings.association.knn,
                )
            conditionals = (
                ("h0_2d_given_1d_pair_cmi_bits", y, x),
                ("h0_1d_pair_given_2d_cmi_bits", x, y),
                (
                    "h0_2d_mu2_given_both_1d_and_2d_mu1_cmi_bits",
                    family["y2"],
                    np.column_stack([x, family["y1"]]),
                ),
                (
                    "h0_1d_m2_given_1d_m1_and_full_2d_cmi_bits",
                    family["x2"],
                    np.column_stack([family["x1"], y]),
                ),
            )
            for name, values, conditioning in conditionals:
                row[name], _ = _calibrated_conditional_information(
                    h0,
                    values,
                    conditioning,
                    indices,
                    nulls,
                    k=settings.association.knn,
                )
        else:
            for name in (
                "h0_1d_pair_mi_bits",
                "h0_2d_vector_mi_bits",
                "h0_all_scales_mi_bits",
                "h0_2d_given_1d_pair_cmi_bits",
                "h0_1d_pair_given_2d_cmi_bits",
                "h0_2d_mu2_given_both_1d_and_2d_mu1_cmi_bits",
                "h0_1d_m2_given_1d_m1_and_full_2d_cmi_bits",
            ):
                row[name] = math.nan
        rows.append(row)

    csv_path = directory / "one_two_dimensional_feature_families.csv"
    text_path = directory / "one_two_dimensional_feature_families.txt"
    draw_path = directory / "one_two_dimensional_feature_family_draws.npz"
    _write_csv(csv_path, rows)
    lines = [
        "Correlated m1 and m2 views of common two-dimensional feature families",
        "",
        (
            f"{'2D family':>18}  {'1D m1':>8}  {'1D m2':>8}  {'Pcommon':>8}  "
            f"{'rho(1D1,1D2)':>14}  {'I(1D pair)':>11}  {'I(2D)':>8}  "
            f"{'I(2D|1D)':>10}"
        ),
        "-" * 112,
    ]
    for row in rows:
        lines.append(
            f"{row['family_ID']:>18}  {row['m1_1d_ID']:>8}  "
            f"{row['m2_1d_ID']:>8}  {row['common_support']:>8.1%}  "
            f"{_number(row['rho_m1_1d_m2_1d'], 4):>14}  "
            f"{_number(row['h0_1d_pair_mi_bits'], 4):>11}  "
            f"{_number(row['h0_2d_vector_mi_bits'], 4):>8}  "
            f"{_number(row['h0_2d_given_1d_pair_cmi_bits'], 4):>10}"
        )
    lines.extend(
        [
            "",
            "Each row is one physical 2D identity; the m1 and m2 catalogues are",
            "correlated projections, not independent features. Mutual information",
            "is therefore evaluated jointly and conditionally, never added across views.",
            "If an identity has multiple candidate 1D matches, the displayed pair",
            "maximizes common draw support and all candidate IDs remain in the CSV.",
            "The global-tail row is a global scale family, not a local morphology.",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if families:
        np.savez_compressed(
            draw_path,
            family_ID=np.asarray([item["family_ID"] for item in families], dtype=str),
            m1_1d_ID=np.asarray([item["m1_1d_ID"] for item in families], dtype=str),
            m2_1d_ID=np.asarray([item["m2_1d_ID"] for item in families], dtype=str),
            common_valid=np.stack([item["common"] for item in families]),
            m1_1d_scale=np.stack([item["x1"] for item in families]),
            m2_1d_scale=np.stack([item["x2"] for item in families]),
            mu1_2d=np.stack([item["y1"] for item in families]),
            mu2_2d=np.stack([item["y2"] for item in families]),
            posterior_weights=np.asarray(weights, dtype=float),
            **({"h0": np.asarray(h0, dtype=float)} if h0 is not None else {}),
        )
    return [path for path in (csv_path, text_path, draw_path) if path.exists()]


def _signature(
    settings: Settings,
    one_d_manifest: dict[str, Any],
    secondary_manifest: dict[str, Any],
    results: Any,
) -> str:
    digest = hashlib.sha256(b"posterior-landscape-1d-2d-comparison-v4")
    digest.update(str(one_d_manifest.get("core_signature", "")).encode())
    digest.update(str(secondary_manifest.get("core_signature", "")).encode())
    digest.update(_text(results.attrs.get("fingerprint", "")).encode())
    digest.update(_text(results.attrs.get("topology_sample_signature", "")).encode())
    digest.update(repr(settings.analysis.minimum_support).encode())
    digest.update(repr(settings.association).encode())
    if "posterior" in results and "h0" in results["posterior"]:
        h0 = np.ascontiguousarray(np.asarray(results["posterior/h0"], dtype="<f8"))
        digest.update(h0.tobytes())
    return digest.hexdigest()


def run_one_two_dimensional_comparison(
    settings: Settings,
    state: dict[str, Any],
    secondary_state: dict[str, Any],
) -> Path:
    """Compare independent m1/m2 catalogues with both views of 2D features."""

    if h5py is None:
        raise RuntimeError("Combined 1D/2D comparison requires h5py.")
    directory = settings.output.directory
    results_path = directory / "results.h5"
    if not results_path.is_file():
        raise RuntimeError("Combined mode requires the completed 2D results.h5 file.")
    one_manifest_path = directory / "one_dimensional_manifest.json"
    one_manifest = json.loads(one_manifest_path.read_text(encoding="utf-8"))
    secondary_manifest_path = directory / "one_dimensional_m2_manifest.json"
    secondary_manifest = json.loads(
        secondary_manifest_path.read_text(encoding="utf-8")
    )
    comparison_manifest_path = directory / "one_two_dimensional_manifest.json"

    with h5py.File(results_path, "r") as results:
        signature = _signature(
            settings, one_manifest, secondary_manifest, results
        )
        if comparison_manifest_path.is_file() and not settings.output.overwrite:
            previous = json.loads(comparison_manifest_path.read_text(encoding="utf-8"))
            required = [
                directory / "one_two_dimensional_feature_comparison.csv",
                directory / "one_two_dimensional_m2_feature_comparison.csv",
                directory / "one_two_dimensional_feature_families.csv",
            ]
            h0_expected = (
                settings.association.enabled != "off"
                and "posterior" in results
                and "h0" in results["posterior"]
            )
            if h0_expected:
                required.extend(
                    [
                        directory
                        / "one_two_dimensional_full_external_parameter_comparison.csv",
                    ]
                )
            if previous.get("signature") == signature and all(
                path.is_file() for path in required
            ):
                print("1D–2D comparison is already complete.")
                return directory

        weights = np.asarray(results["posterior/weights"], dtype=float)
        weights = weights / np.sum(weights)
        projection_grid = np.asarray(results["coordinates/m1"], dtype=float)
        m_grid = np.asarray(state["m_grid"], dtype=float)
        two_d_measure = _text(results.attrs.get("feature_measure", "log"))
        log_base = float(results.attrs.get("log_base", settings.analysis.log_base))
        projected = _load_projected_features(results)
        projected.sort(
            key=lambda feature: np.nanmedian(feature.mu1[feature.selected])
            if np.any(feature.selected)
            else np.inf
        )

        candidates: list[dict[str, Any]] = []
        for one_id in state["feature_posterior_order"]:
            reference = state["feature_reference_lookup"][one_id]
            one_type = reference["reference_type"]
            one_band = reference["band"]
            for feature in projected:
                if not _compatible(one_type, feature.kind) or not np.any(
                    feature.selected
                ):
                    continue
                selected_weights = weights[feature.selected]
                bounds = feature.bounds[feature.selected]
                left = _weighted_quantile(bounds[:, 0], (0.5,), selected_weights)[0]
                right = _weighted_quantile(bounds[:, 1], (0.5,), selected_weights)[0]
                jaccard, one_coverage, two_coverage = _log_interval_metrics(
                    float(one_band["left"]),
                    float(one_band["right"]),
                    float(left),
                    float(right),
                )
                candidates.append(
                    {
                        "one_id": one_id,
                        "one_type": one_type,
                        "one_left": float(one_band["left"]),
                        "one_center": float(one_band["center"]),
                        "one_right": float(one_band["right"]),
                        "feature": feature,
                        "two_left": float(left),
                        "two_right": float(right),
                        "jaccard": jaccard,
                        "one_coverage": one_coverage,
                        "two_coverage": two_coverage,
                        "two_support": float(np.sum(weights[feature.selected])),
                    }
                )

        primary_by_one = _select_primary_candidates(
            candidates,
            state["feature_posterior_order"],
            settings.analysis.minimum_support,
        )
        primary_counts: dict[str, int] = {}
        for item in primary_by_one.values():
            identifier = item["feature"].identifier
            primary_counts[identifier] = primary_counts.get(identifier, 0) + 1

        h0 = (
            np.asarray(results["posterior/h0"], dtype=float)
            if settings.association.enabled != "off"
            and "h0" in results["posterior"]
            else None
        )
        chain_id = (
            np.asarray(results["posterior/chain_id"])
            if h0 is not None and "chain_id" in results["posterior"]
            else None
        )
        rng = np.random.default_rng(settings.association.random_seed)
        nulls = (
            _circular_h0_nulls(
                h0,
                chain_id,
                settings.association.permutations,
                rng,
            )
            if h0 is not None and chain_id is not None
            else []
        )
        information_workers = settings.compute.workers or max(
            1, min(4, os.cpu_count() or 1)
        )

        rows: list[dict[str, Any]] = []
        primary_data: dict[tuple[str, str], dict[str, Any]] = {}
        full_h0_rows: list[dict[str, Any]] = []
        full_h0_data: dict[str, dict[str, Any]] = {}
        for item in candidates:
            one_id = item["one_id"]
            feature = item["feature"]
            is_primary = primary_by_one.get(one_id) is item
            positive_overlaps = sum(
                other["one_id"] == one_id and other["jaccard"] > 0.0
                for other in candidates
            )
            if not is_primary:
                relationship = (
                    "unmatched"
                    if one_id not in primary_by_one
                    else "secondary overlap"
                    if item["jaccard"] > 0.0
                    else "no overlap"
                )
                rows.append(
                    {
                        "one_d_ID": one_id,
                        "one_d_type": item["one_type"],
                        "two_d_ID": feature.identifier,
                        "two_d_parent_ID": feature.parent_identifier,
                        "two_d_type": feature.kind,
                        "primary_match": False,
                        "relationship": relationship,
                        "overlapping_two_d_count": positive_overlaps,
                        "one_d_reference_left": item["one_left"],
                        "one_d_reference_center": item["one_center"],
                        "one_d_reference_right": item["one_right"],
                        "two_d_projected_left_median": item["two_left"],
                        "two_d_projected_right_median": item["two_right"],
                        "reference_jaccard": item["jaccard"],
                        "reference_one_d_coverage": item["one_coverage"],
                        "reference_two_d_coverage": item["two_coverage"],
                        "one_d_bounded_support": math.nan,
                        "two_d_region_support": item["two_support"],
                        "common_support": math.nan,
                        "number_common_draws": 0,
                        "draw_jaccard_q05": math.nan,
                        "draw_jaccard_median": math.nan,
                        "draw_jaccard_q95": math.nan,
                        "projection_capture_q05": math.nan,
                        "projection_capture_median": math.nan,
                        "projection_capture_q95": math.nan,
                        "mass_scale_log_ratio_q05": math.nan,
                        "mass_scale_log_ratio_median": math.nan,
                        "mass_scale_log_ratio_q95": math.nan,
                        "mass_scale_spearman": math.nan,
                        "h0_one_d_rho": math.nan,
                        "h0_one_d_mi_bits": math.nan,
                        "h0_two_d_rho": math.nan,
                        "h0_two_d_mi_bits": math.nan,
                        "h0_joint_mi_bits": math.nan,
                        "h0_two_d_given_one_d_bits": math.nan,
                        "h0_one_d_given_two_d_bits": math.nan,
                    }
                )
                continue

            posterior = state["feature_draw_posteriors"][one_id]
            bounded = np.asarray(posterior["bounded"], dtype=bool)
            left = np.asarray(posterior["left"], dtype=float)
            right = np.asarray(posterior["right"], dtype=float)
            one_mu = _one_d_mass_scale(item["one_type"], posterior)
            common = (
                bounded
                & feature.selected
                & np.isfinite(left)
                & np.isfinite(right)
                & (left > 0.0)
                & (right > left)
                & np.isfinite(one_mu)
                & (one_mu > 0.0)
                & np.isfinite(feature.mu1)
                & (feature.mu1 > 0.0)
                & np.isfinite(feature.mu2)
                & (feature.mu2 > 0.0)
                & np.isfinite(feature.bounds[:, 0])
                & np.isfinite(feature.bounds[:, 1])
                & (feature.bounds[:, 0] > 0.0)
                & (feature.bounds[:, 1] > feature.bounds[:, 0])
            )
            selected_indices = np.flatnonzero(common)
            common_weights = weights[common]
            jaccard = _draw_interval_jaccard(
                left[common],
                right[common],
                feature.bounds[common, 0],
                feature.bounds[common, 1],
            )
            converted_projection = _projection_in_one_d_measure(
                _projection_on_one_dimensional_grid(
                    feature.projection[common], projection_grid, m_grid
                ),
                m_grid,
                two_d_measure=two_d_measure,
                one_d_measure=state["MASS_DENSITY_MEASURE"],
                log_base=log_base,
            )
            capture = _projection_capture(
                converted_projection, left[common], right[common], state
            )
            log_ratio = np.log(feature.mu1[common] / one_mu[common])
            j_q = _weighted_quantile(jaccard, (0.05, 0.5, 0.95), common_weights)
            c_q = _weighted_quantile(capture, (0.05, 0.5, 0.95), common_weights)
            r_q = _weighted_quantile(log_ratio, (0.05, 0.5, 0.95), common_weights)
            mass_rho = (
                float(spearmanr(one_mu[common], feature.mu1[common]).statistic)
                if selected_indices.size > 2
                else math.nan
            )
            full_h0_data[one_id] = {
                "two_d_ID": feature.identifier,
                "two_d_parent_ID": feature.parent_identifier,
                "two_d_type": feature.kind,
                "common": common,
                "one_mu": one_mu,
                "two_mu1": feature.mu1,
                "two_mu2": feature.mu2,
            }

            h0_one_rho = h0_one_mi = h0_two_rho = h0_two_mi = math.nan
            h0_joint = h0_two_given_one = h0_one_given_two = math.nan
            if h0 is not None and chain_id is not None:
                print(
                    "Full 1D–2D H0 comparison: "
                    f"{len(full_h0_rows) + 1}/{len(primary_by_one)} "
                    f"({one_id}->{feature.identifier}).",
                    flush=True,
                )
                full_information = _full_h0_information(
                    h0,
                    chain_id,
                    one_mu,
                    feature.mu1,
                    feature.mu2,
                    common,
                    nulls,
                    settings.association,
                    rng,
                    workers=information_workers,
                )
                h0_one_rho = full_information["h0_one_d_rho"]
                h0_one_mi = full_information["h0_one_d_mi_bits"]
                h0_two_rho = full_information["h0_two_d_mu1_rho"]
                h0_two_mi = full_information["h0_two_d_mu1_mi_bits"]
                h0_joint = full_information["h0_all_m1_mi_bits"]
                h0_two_given_one = full_information[
                    "h0_two_d_mu1_given_one_d_cmi_bits"
                ]
                h0_one_given_two = full_information[
                    "h0_one_d_given_two_d_mu1_cmi_bits"
                ]
                full_record: dict[str, Any] = {
                    "one_d_ID": one_id,
                    "one_d_type": item["one_type"],
                    "two_d_ID": feature.identifier,
                    "two_d_parent_ID": feature.parent_identifier,
                    "two_d_type": feature.kind,
                    "number_total_draws": int(h0.size),
                    "number_common_draws": int(np.count_nonzero(common)),
                    "common_support": float(np.sum(weights[common])),
                }
                full_record.update(full_information)
                full_h0_rows.append(full_record)

            normalized_projection = converted_projection.copy()
            norm = np.einsum(
                "bi,i->b",
                normalized_projection,
                np.asarray(state["MASS_MEASURE_WIDTHS"]),
                optimize=True,
            )
            valid_norm = norm > 0.0
            normalized_projection[valid_norm] /= norm[valid_norm, None]
            projection_quantiles = (
                np.quantile(
                    normalized_projection[valid_norm],
                    (0.05, 0.5, 0.95),
                    axis=0,
                )
                if np.any(valid_norm)
                else np.full((3, m_grid.size), np.nan)
            )
            primary_data[(one_id, feature.identifier)] = {
                "projection_quantiles": projection_quantiles
            }

            merged = primary_counts.get(feature.identifier, 0) > 1
            split = positive_overlaps > 1
            relationship = (
                "complex overlap"
                if merged and split
                else "many 1D to one 2D"
                if merged
                else "one 1D to many 2D"
                if split
                else "direct"
            )
            rows.append(
                {
                    "one_d_ID": one_id,
                    "one_d_type": item["one_type"],
                    "two_d_ID": feature.identifier,
                    "two_d_parent_ID": feature.parent_identifier,
                    "two_d_type": feature.kind,
                    "primary_match": True,
                    "relationship": relationship,
                    "overlapping_two_d_count": positive_overlaps,
                    "one_d_reference_left": item["one_left"],
                    "one_d_reference_center": item["one_center"],
                    "one_d_reference_right": item["one_right"],
                    "two_d_projected_left_median": item["two_left"],
                    "two_d_projected_right_median": item["two_right"],
                    "reference_jaccard": item["jaccard"],
                    "reference_one_d_coverage": item["one_coverage"],
                    "reference_two_d_coverage": item["two_coverage"],
                    "one_d_bounded_support": float(np.mean(bounded)),
                    "two_d_region_support": item["two_support"],
                    "common_support": float(np.sum(weights[common])),
                    "number_common_draws": int(np.count_nonzero(common)),
                    "draw_jaccard_q05": j_q[0],
                    "draw_jaccard_median": j_q[1],
                    "draw_jaccard_q95": j_q[2],
                    "projection_capture_q05": c_q[0],
                    "projection_capture_median": c_q[1],
                    "projection_capture_q95": c_q[2],
                    "mass_scale_log_ratio_q05": r_q[0],
                    "mass_scale_log_ratio_median": r_q[1],
                    "mass_scale_log_ratio_q95": r_q[2],
                    "mass_scale_spearman": mass_rho,
                    "h0_one_d_rho": h0_one_rho,
                    "h0_one_d_mi_bits": h0_one_mi,
                    "h0_two_d_rho": h0_two_rho,
                    "h0_two_d_mi_bits": h0_two_mi,
                    "h0_joint_mi_bits": h0_joint,
                    "h0_two_d_given_one_d_bits": h0_two_given_one,
                    "h0_one_d_given_two_d_bits": h0_one_given_two,
                }
            )

    rows.sort(
        key=lambda row: (
            list(state["feature_posterior_order"]).index(row["one_d_ID"]),
            not row["primary_match"],
            -float(row["reference_jaccard"]),
        )
    )
    full_h0_rows.sort(
        key=lambda row: list(state["feature_posterior_order"]).index(
            row["one_d_ID"]
        )
    )
    csv_path = directory / "one_two_dimensional_feature_comparison.csv"
    text_path = directory / "one_two_dimensional_feature_comparison.txt"
    _write_csv(csv_path, rows)
    _write_text(
        text_path,
        rows,
        state["feature_posterior_order"],
        settings.analysis.minimum_support,
        coordinate_name=str(state.get("COORDINATE_NAME", "m1")),
        parameter_name=settings.association.parameter_name,
    )
    figure_paths = _write_figures(
        directory, rows, primary_data, state, settings
    )
    full_h0_paths: list[Path] = []
    if full_h0_rows and h0 is not None and chain_id is not None:
        full_h0_csv_path = (
            directory
            / "one_two_dimensional_full_external_parameter_comparison.csv"
        )
        full_h0_text_path = (
            directory
            / "one_two_dimensional_full_external_parameter_comparison.txt"
        )
        full_h0_draw_path = (
            directory / "one_two_dimensional_full_external_parameter_draws.npz"
        )
        _write_csv(full_h0_csv_path, full_h0_rows)
        _write_full_h0_text(
            full_h0_text_path,
            full_h0_rows,
            settings.association.parameter_name,
        )
        _write_full_h0_draws(
            full_h0_draw_path,
            state["feature_posterior_order"],
            full_h0_data,
            h0,
            chain_id,
            weights,
        )
        full_h0_paths = [
            full_h0_csv_path,
            full_h0_text_path,
            full_h0_draw_path,
        ]
    with h5py.File(results_path, "r") as comparison_results:
        secondary_rows, secondary_data, secondary_paths = (
            _run_secondary_projection_comparison(
                directory,
                comparison_results,
                secondary_state,
                settings,
                weights,
                h0,
                chain_id,
                nulls,
            )
        )
        family_paths = _write_feature_family_comparison(
            directory,
            comparison_results,
            full_h0_data,
            secondary_data,
            state,
            secondary_state,
            weights,
            h0,
            nulls,
            settings,
        )
    manifest = {
        "completed": True,
        "signature": signature,
        "matching": (
            "maximum_positive_reference_interval_jaccard_within_morphology_family_"
            "after_finite_region_support_filter"
        ),
        "minimum_two_d_region_support": settings.analysis.minimum_support,
        "interval_coordinates": ["ln_m1", "ln_m2"],
        "moving_window_capture": (
            "fraction_of_2d_projection_inside_draw_specific_1d_window"
        ),
        "h0_conditioning": "common_valid_draws",
        "full_h0_comparison": {
            "available": bool(full_h0_rows),
            "variables": {
                "one_d": "draw_adaptive_1d_feature_mass_scale",
                "full_two_d": ["draw_adaptive_2d_mu1", "draw_adaptive_2d_mu2"],
            },
            "conditional_information_estimator": "direct_null_calibrated_knn",
            "chain_rule_differences": "diagnostic_only",
            "uncertainty": "paired_80_percent_subsamples_5th_to_95th_percentiles",
            "common_valid_selection_test": "rank_biserial_within_chain_circular_shift",
            "parallel_workers": information_workers,
            "knn": settings.association.knn,
            "permutations": settings.association.permutations,
            "uncertainty_resamples": settings.association.uncertainty_resamples,
            "random_seed": settings.association.random_seed,
        },
        "outputs": [
            csv_path.name,
            text_path.name,
            *[path.name for path in figure_paths],
            *[path.name for path in full_h0_paths],
            *[path.name for path in secondary_paths],
            *[path.name for path in family_paths],
        ],
    }
    temporary = comparison_manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(comparison_manifest_path)
    print(text_path.read_text(encoding="utf-8"))
    return directory
