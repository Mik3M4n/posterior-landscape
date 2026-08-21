"""Associations between adaptive two-dimensional features and an aligned scalar."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import rankdata, spearmanr

from .config import AssociationSettings, PlotSettings


@dataclass(frozen=True)
class AssociationFeature:
    identifier: str
    kind: str
    morphology: np.ndarray
    location: np.ndarray
    region: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    region_probability: np.ndarray
    relative_width_m1: np.ndarray
    relative_width_m2: np.ndarray
    contrast: np.ndarray
    extent: np.ndarray
    bounded_fraction: np.ndarray
    fallback_fraction: np.ndarray
    deficit_probability: np.ndarray
    match_distance: np.ndarray | None = None
    match_margin: np.ndarray | None = None
    match_ambiguous: np.ndarray | None = None
    parent_identifier: str = ""
    configuration: str = ""
    family_morphology: np.ndarray | None = None
    family_location: np.ndarray | None = None
    family_region: np.ndarray | None = None
    record_scope: str = "feature"
    suppress_continuous: bool = False


@dataclass(frozen=True)
class LocationConfigurationResult:
    """Stable posterior location configurations for one catalogue family."""

    identifier: str
    kind: str
    status: str
    labels: np.ndarray
    configuration_labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    conditional_probabilities: tuple[float, ...]
    component_counts: tuple[int, ...]
    minimum_label_agreement: float

    @property
    def multimodal(self) -> bool:
        return self.status == "robust_multimodal"


_LOCATION_BANDWIDTHS = (1.00, 1.15, 1.40, 1.80, 2.20)
_LOCATION_REFERENCE_BANDWIDTH = 1.15
_LOCATION_HPD_MASS = 0.90
_LOCATION_MINIMUM_COMPONENT_MASS = 0.02
_LOCATION_MINIMUM_LABEL_AGREEMENT = 0.95
_LOCATION_MINIMUM_DRAWS = 50
_LOCATION_HISTOGRAM_BINS = 80


def _weighted_quantile_1d(
    values: np.ndarray, probability: float, weights: np.ndarray
) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative.size == 0 or cumulative[-1] <= 0.0:
        return math.nan
    target = float(probability) * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, target, side="left")),
        ordered_values.size - 1,
    )
    return float(ordered_values[index])


def _location_hpd_threshold(values: np.ndarray, probability: float) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    target = float(probability) * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, target, side="left")), flat.size - 1
    )
    return float(flat[order[index]])


def _whiten_locations(
    values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    normalized = weights / np.sum(weights)
    center = np.sum(values * normalized[:, None], axis=0)
    centered = values - center
    covariance = (centered * normalized[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-10, np.finfo(float).eps)
    transform = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    )
    return centered @ transform


def _weighted_kmeans(
    values: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    *,
    maximum_iterations: int = 100,
) -> np.ndarray | None:
    centers = np.asarray(centers, dtype=float).copy()
    labels = np.full(values.shape[0], -1, dtype=np.int16)
    for _ in range(maximum_iterations):
        distance = np.sum(
            (values[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        updated = np.argmin(distance, axis=1).astype(np.int16)
        new_centers = centers.copy()
        for label in range(centers.shape[0]):
            selected = updated == label
            if not np.any(selected):
                return None
            selected_weights = weights[selected]
            new_centers[label] = np.average(
                values[selected], axis=0, weights=selected_weights
            )
        converged = np.array_equal(updated, labels) and np.allclose(
            new_centers, centers
        )
        labels = updated
        centers = new_centers
        if converged:
            break
    return labels


def _location_mode_fit(
    values: np.ndarray, weights: np.ndarray, bandwidth: float
) -> tuple[np.ndarray | None, tuple[float, ...], int]:
    lower = np.asarray(
        [_weighted_quantile_1d(values[:, axis], 0.005, weights) for axis in range(2)]
    )
    upper = np.asarray(
        [_weighted_quantile_1d(values[:, axis], 0.995, weights) for axis in range(2)]
    )
    if np.any(~np.isfinite(lower)) or np.any(upper <= lower):
        return None, (), 0
    histogram, first_edges, second_edges = np.histogram2d(
        values[:, 0],
        values[:, 1],
        bins=_LOCATION_HISTOGRAM_BINS,
        range=((lower[0], upper[0]), (lower[1], upper[1])),
        weights=weights,
    )
    smoothed = ndimage.gaussian_filter(
        histogram, float(bandwidth), mode="nearest"
    )
    if not np.any(smoothed > 0.0):
        return None, (), 0
    threshold = _location_hpd_threshold(smoothed, _LOCATION_HPD_MASS)
    component_labels, number = ndimage.label(
        smoothed >= threshold, structure=np.ones((3, 3), dtype=bool)
    )
    first_centers = 0.5 * (first_edges[:-1] + first_edges[1:])
    second_centers = 0.5 * (second_edges[:-1] + second_edges[1:])
    total = float(np.sum(smoothed))
    components: list[tuple[float, np.ndarray]] = []
    for component in range(1, number + 1):
        selected = component_labels == component
        component_mass = float(np.sum(smoothed[selected]) / total)
        if component_mass < _LOCATION_MINIMUM_COMPONENT_MASS:
            continue
        restricted = np.where(selected, smoothed, -np.inf)
        maximum = np.unravel_index(np.argmax(restricted), restricted.shape)
        components.append(
            (
                component_mass,
                np.asarray(
                    [first_centers[maximum[0]], second_centers[maximum[1]]]
                ),
            )
        )
    components.sort(key=lambda item: item[0], reverse=True)
    count = len(components)
    if count < 2:
        return None, tuple(item[0] for item in components), count
    labels = _weighted_kmeans(
        values, weights, np.vstack([item[1] for item in components])
    )
    return labels, tuple(item[0] for item in components), count


def _aligned_labels(
    reference: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    number: int,
) -> tuple[np.ndarray, float]:
    overlap = np.zeros((number, number), dtype=float)
    for first in range(number):
        for second in range(number):
            overlap[first, second] = float(
                np.sum(weights[(reference == first) & (candidate == second)])
            )
    rows, columns = linear_sum_assignment(-overlap)
    mapping = np.full(number, -1, dtype=int)
    mapping[columns] = rows
    aligned = mapping[candidate]
    agreement = float(np.sum(weights[aligned == reference]) / np.sum(weights))
    return aligned.astype(np.int16), agreement


def detect_location_configurations(
    features: list[AssociationFeature], weights: np.ndarray
) -> dict[str, LocationConfigurationResult]:
    """Conservatively split stable disconnected posterior location modes."""

    weights = np.asarray(weights, dtype=float)
    results: dict[str, LocationConfigurationResult] = {}
    for feature in features:
        finite = (
            np.asarray(feature.region, dtype=bool)
            & np.isfinite(feature.mu1)
            & np.isfinite(feature.mu2)
            & (feature.mu1 > 0.0)
            & (feature.mu2 > 0.0)
        )
        indices = np.flatnonzero(finite)
        labels_all = np.full(weights.size, -1, dtype=np.int16)
        valid_probability = float(np.sum(weights[finite]))
        if indices.size < _LOCATION_MINIMUM_DRAWS or valid_probability <= 0.0:
            labels_all[finite] = 0
            results[feature.identifier] = LocationConfigurationResult(
                feature.identifier,
                feature.kind,
                "insufficient_support",
                labels_all,
                ("A",),
                (valid_probability,),
                (1.0 if valid_probability > 0.0 else math.nan,),
                (),
                math.nan,
            )
            continue

        selected_weights = weights[indices]
        selected_weights = selected_weights / np.sum(selected_weights)
        whitened = _whiten_locations(
            np.log(np.column_stack([feature.mu1[finite], feature.mu2[finite]])),
            selected_weights,
        )
        fits: dict[float, np.ndarray | None] = {}
        counts: list[int] = []
        for bandwidth in _LOCATION_BANDWIDTHS:
            fitted, _, count = _location_mode_fit(
                whitened, selected_weights, bandwidth
            )
            fits[bandwidth] = fitted
            counts.append(count)

        reference = fits[_LOCATION_REFERENCE_BANDWIDTH]
        reference_count = counts[_LOCATION_BANDWIDTHS.index(
            _LOCATION_REFERENCE_BANDWIDTH
        )]
        status = "unimodal" if max(counts, default=0) <= 1 else "unstable"
        minimum_agreement = math.nan
        robust = (
            reference is not None
            and reference_count >= 2
            and all(count == reference_count for count in counts)
            and all(fit is not None for fit in fits.values())
        )
        if robust:
            cluster_probability = np.asarray(
                [
                    np.sum(selected_weights[reference == label])
                    for label in range(reference_count)
                ]
            )
            cluster_count = np.asarray(
                [np.count_nonzero(reference == label) for label in range(reference_count)]
            )
            robust = bool(
                np.all(cluster_probability >= _LOCATION_MINIMUM_COMPONENT_MASS)
                and np.all(cluster_count >= _LOCATION_MINIMUM_DRAWS)
            )
        if robust:
            ordering = sorted(
                range(reference_count),
                key=lambda label: (
                    -float(cluster_probability[label]),
                    float(np.median(feature.mu1[indices][reference == label])),
                    float(np.median(feature.mu2[indices][reference == label])),
                ),
            )
            remap = np.empty(reference_count, dtype=np.int16)
            for new, old in enumerate(ordering):
                remap[old] = new
            reference = remap[reference]
            agreements = []
            for bandwidth in _LOCATION_BANDWIDTHS:
                candidate = fits[bandwidth]
                assert candidate is not None
                _, agreement = _aligned_labels(
                    reference, candidate, selected_weights, reference_count
                )
                agreements.append(agreement)
            minimum_agreement = float(min(agreements))
            robust = minimum_agreement >= _LOCATION_MINIMUM_LABEL_AGREEMENT

        if not robust:
            labels_all[finite] = 0
            results[feature.identifier] = LocationConfigurationResult(
                feature.identifier,
                feature.kind,
                status,
                labels_all,
                ("A",),
                (valid_probability,),
                (1.0,),
                tuple(counts),
                minimum_agreement,
            )
            continue

        labels_all[indices] = reference
        configuration_labels = tuple(
            chr(ord("A") + index) if index < 26 else str(index + 1)
            for index in range(reference_count)
        )
        probabilities = tuple(
            float(np.sum(weights[labels_all == label]))
            for label in range(reference_count)
        )
        results[feature.identifier] = LocationConfigurationResult(
            feature.identifier,
            feature.kind,
            "robust_multimodal",
            labels_all,
            configuration_labels,
            probabilities,
            tuple(value / valid_probability for value in probabilities),
            tuple(counts),
            minimum_agreement,
        )
    return results


def location_configuration_masks(
    results: dict[str, LocationConfigurationResult],
) -> dict[str, tuple[tuple[str, np.ndarray], ...]]:
    """Return only robust configuration masks for plotting and summaries."""

    return {
        identifier: tuple(
            (label, result.labels == index)
            for index, label in enumerate(result.configuration_labels)
        )
        for identifier, result in results.items()
        if result.multimodal
    }


def expand_location_configurations(
    features: list[AssociationFeature],
    results: dict[str, LocationConfigurationResult],
) -> list[AssociationFeature]:
    """Replace a multimodal continuous summary by parent plus configurations."""

    expanded: list[AssociationFeature] = []
    for feature in features:
        result = results[feature.identifier]
        if not result.multimodal:
            expanded.append(feature)
            continue
        expanded.append(
            replace(
                feature,
                record_scope="multimodal_family",
                family_morphology=feature.morphology,
                family_location=feature.location,
                family_region=feature.region,
                suppress_continuous=True,
            )
        )
        for index, label in enumerate(result.configuration_labels):
            selected = result.labels == index
            expanded.append(
                replace(
                    feature,
                    identifier=f"{feature.identifier}-{label}",
                    morphology=selected,
                    location=selected,
                    region=selected,
                    parent_identifier=feature.identifier,
                    configuration=label,
                    family_morphology=feature.morphology,
                    family_location=feature.location,
                    family_region=feature.region,
                    record_scope="location_configuration",
                    suppress_continuous=False,
                )
            )
    return expanded


def _matrix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return result[:, None] if result.ndim == 1 else result


def _standardize(values: np.ndarray, *, whiten: bool = False) -> np.ndarray:
    values = _matrix(values)
    centered = values - np.mean(values, axis=0)
    if whiten and values.shape[1] > 1:
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(float(np.max(eigenvalues)) * 1e-12, np.finfo(float).eps)
        transform = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor)))
        result = centered @ transform
    else:
        scale = np.std(centered, axis=0, ddof=1)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        result = centered / scale
    row = np.arange(result.shape[0], dtype=float)[:, None] + 1.0
    column = np.arange(result.shape[1], dtype=float)[None, :] + 1.0
    return result + 1e-10 * np.sin(row * (column + 0.37))


def _neighbour_counts(values: np.ndarray, radii: np.ndarray) -> np.ndarray:
    tree = cKDTree(values)
    return np.asarray(
        tree.query_ball_point(values, radii, p=np.inf, return_length=True),
        dtype=int,
    ) - 1


def ksg_mutual_information(x: np.ndarray, y: np.ndarray, *, k: int = 5) -> float:
    """KSG-1 mutual information in bits using the maximum norm."""

    x = _standardize(x, whiten=True)
    y = _standardize(y, whiten=True)
    number = x.shape[0]
    if y.shape[0] != number or number <= k + 2:
        return math.nan
    joint = np.column_stack([x, y])
    distances = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, -1]
    radii = np.nextafter(distances, 0.0)
    nx = _neighbour_counts(x, radii)
    ny = _neighbour_counts(y, radii)
    estimate = digamma(k) + digamma(number) - np.mean(
        digamma(nx + 1) + digamma(ny + 1)
    )
    return float(estimate / np.log(2.0))


def ksg_conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """Nearest-neighbour estimate of I(x;y|z), in bits."""

    x = _standardize(x)
    y = _standardize(y)
    z = _standardize(z)
    number = x.shape[0]
    if y.shape[0] != number or z.shape[0] != number or number <= k + 2:
        return math.nan
    joint = np.column_stack([x, y, z])
    distances = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, -1]
    radii = np.nextafter(distances, 0.0)
    nxz = _neighbour_counts(np.column_stack([x, z]), radii)
    nyz = _neighbour_counts(np.column_stack([y, z]), radii)
    nz = _neighbour_counts(z, radii)
    estimate = digamma(k) + np.mean(
        digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1)
    )
    return float(estimate / np.log(2.0))


def _circular_h0_nulls(
    h0: np.ndarray,
    chain_id: np.ndarray,
    number: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    nulls: list[np.ndarray] = []
    groups = [np.flatnonzero(chain_id == label) for label in np.unique(chain_id)]
    for _ in range(number):
        shifted = np.asarray(h0, dtype=float).copy()
        for indices in groups:
            if indices.size > 1:
                offset = int(rng.integers(1, indices.size))
                shifted[indices] = np.roll(h0[indices], offset)
        nulls.append(shifted)
    return nulls


def _rank_biserial(h0: np.ndarray, indicator: np.ndarray) -> float:
    indicator = np.asarray(indicator, dtype=bool)
    first = np.asarray(h0, dtype=float)[indicator]
    second = np.asarray(h0, dtype=float)[~indicator]
    if first.size == 0 or second.size == 0:
        return math.nan
    ranks = rankdata(np.concatenate([first, second]), method="average")
    u = float(np.sum(ranks[: first.size]) - first.size * (first.size + 1) / 2.0)
    return float(2.0 * u / (first.size * second.size) - 1.0)


def _binary_association(
    h0: np.ndarray,
    indicator: np.ndarray,
    conditioning: np.ndarray,
    nulls: list[np.ndarray],
    *,
    minimum_group: int = 20,
) -> tuple[float, float, int, int]:
    selected = np.asarray(conditioning, dtype=bool)
    indicator = np.asarray(indicator, dtype=bool)[selected]
    values = np.asarray(h0, dtype=float)[selected]
    first = int(np.count_nonzero(indicator))
    second = int(indicator.size - first)
    if min(first, second) < minimum_group:
        return math.nan, math.nan, first, second
    observed = _rank_biserial(values, indicator)
    if not nulls:
        return observed, math.nan, first, second
    null_values = np.asarray(
        [_rank_biserial(null[selected], indicator) for null in nulls], dtype=float
    )
    probability = float(
        (1 + np.count_nonzero(np.abs(null_values) >= abs(observed)))
        / (null_values.size + 1)
    )
    return observed, probability, first, second


def pairwise_indicator_associations(
    identifiers: list[str],
    kinds: list[str],
    indicators: list[np.ndarray],
    weights: np.ndarray,
    *,
    h0: np.ndarray | None = None,
    chain_id: np.ndarray | None = None,
    permutations: int = 0,
    random_seed: int = 12345,
) -> list[dict[str, Any]]:
    """Summarize pairwise coexistence without declaring composite features.

    The indicators are normally draw-level location-support flags.  The
    returned probabilities are posterior-weighted.  H0 rank-biserial tests
    are optional and, when requested, use the same within-chain circular null
    calibration as the rest of the association module.
    """

    if not (len(identifiers) == len(kinds) == len(indicators)):
        raise ValueError("Pairwise indicator inputs must have the same length.")
    posterior_weights = np.asarray(weights, dtype=float).reshape(-1)
    total_weight = float(np.sum(posterior_weights))
    if (
        posterior_weights.size == 0
        or np.any(~np.isfinite(posterior_weights))
        or np.any(posterior_weights < 0.0)
        or not np.isfinite(total_weight)
        or total_weight <= 0.0
    ):
        raise ValueError(
            "Posterior weights must be finite, non-negative, and have "
            "positive total weight."
        )
    posterior_weights = posterior_weights / total_weight
    flags = [np.asarray(value, dtype=bool).reshape(-1) for value in indicators]
    if any(value.size != posterior_weights.size for value in flags):
        raise ValueError("Every indicator must contain one value per draw.")

    nulls: list[np.ndarray] = []
    h0_values: np.ndarray | None = None
    if h0 is not None or chain_id is not None:
        if h0 is None or chain_id is None:
            raise ValueError("Both h0 and chain_id are required for H0 tests.")
        h0_values = np.asarray(h0, dtype=float).reshape(-1)
        chains = np.asarray(chain_id).reshape(-1)
        if h0_values.size != posterior_weights.size or chains.size != h0_values.size:
            raise ValueError("h0 and chain_id must align with the indicators.")
        nulls = _circular_h0_nulls(
            h0_values,
            chains,
            int(permutations),
            np.random.default_rng(random_seed),
        )

    records: list[dict[str, Any]] = []
    for first_index, second_index in combinations(range(len(flags)), 2):
        first = flags[first_index]
        second = flags[second_index]
        joint = first & second
        p_first = float(np.sum(posterior_weights[first]))
        p_second = float(np.sum(posterior_weights[second]))
        p_joint = float(np.sum(posterior_weights[joint]))
        covariance = p_joint - p_first * p_second
        denominator = math.sqrt(
            max(p_first * (1.0 - p_first) * p_second * (1.0 - p_second), 0.0)
        )
        phi = covariance / denominator if denominator > 0.0 else math.nan
        if h0_values is not None:
            rrb, p_perm, n_yes, n_no = _binary_association(
                h0_values,
                joint,
                np.ones(joint.size, dtype=bool),
                nulls,
            )
        else:
            rrb = p_perm = math.nan
            n_yes = int(np.count_nonzero(joint))
            n_no = int(joint.size - n_yes)
        records.append(
            {
                "ID_A": identifiers[first_index],
                "type_A": kinds[first_index],
                "ID_B": identifiers[second_index],
                "type_B": kinds[second_index],
                "P_A": p_first,
                "P_B": p_second,
                "P_A_and_B": p_joint,
                "P_B_given_A": p_joint / p_first if p_first > 0.0 else math.nan,
                "P_A_given_B": p_joint / p_second if p_second > 0.0 else math.nan,
                "covariance": covariance,
                "phi": phi,
                "lift": (
                    p_joint / (p_first * p_second)
                    if p_first > 0.0 and p_second > 0.0
                    else math.nan
                ),
                "joint_occurrence_h0_rrb": rrb,
                "joint_occurrence_h0_p_perm": p_perm,
                "joint_occurrence_n_yes": n_yes,
                "joint_occurrence_n_no": n_no,
            }
        )
    return sorted(
        records,
        key=lambda record: (
            -abs(record["phi"]) if np.isfinite(record["phi"]) else math.inf,
            record["ID_A"],
            record["ID_B"],
        ),
    )


def _continuous_association(
    h0: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    nulls: list[np.ndarray],
    *,
    k: int,
) -> tuple[float, float, int, float]:
    values = np.asarray(values, dtype=float)
    selected = np.asarray(selected, dtype=bool) & np.isfinite(values)
    number = int(np.count_nonzero(selected))
    if number <= max(k + 2, 20) or np.ptp(values[selected]) <= 0.0:
        return math.nan, math.nan, number, math.nan
    rho = float(spearmanr(h0[selected], values[selected]).statistic)
    raw = ksg_mutual_information(h0[selected], values[selected], k=k)
    null_mi = np.asarray(
        [
            ksg_mutual_information(null[selected], values[selected], k=k)
            for null in nulls
        ],
        dtype=float,
    )
    offset = float(np.nanmedian(null_mi)) if null_mi.size else 0.0
    return rho, max(0.0, raw - offset), number, offset


def _joint_association(
    h0: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    selected: np.ndarray,
    nulls: list[np.ndarray],
    *,
    k: int,
) -> tuple[float, int, float]:
    selected = (
        np.asarray(selected, dtype=bool)
        & np.isfinite(mu1)
        & np.isfinite(mu2)
    )
    number = int(np.count_nonzero(selected))
    if number <= max(k + 2, 20):
        return math.nan, number, math.nan
    vector = np.column_stack([mu1[selected], mu2[selected]])
    raw = ksg_mutual_information(h0[selected], vector, k=k)
    null_mi = np.asarray(
        [ksg_mutual_information(null[selected], vector, k=k) for null in nulls]
    )
    offset = float(np.nanmedian(null_mi)) if null_mi.size else 0.0
    return max(0.0, raw - offset), number, offset


def _conditional_association(
    h0: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    selected: np.ndarray,
    nulls: list[np.ndarray],
    settings: AssociationSettings,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    selected = (
        np.asarray(selected, dtype=bool)
        & np.isfinite(mu1)
        & np.isfinite(mu2)
    )
    indices = np.flatnonzero(selected)
    number = indices.size
    if number <= max(settings.knn + 2, 30):
        return math.nan, math.nan, math.nan, int(number)
    raw = ksg_conditional_mutual_information(
        h0[indices], mu2[indices], mu1[indices], k=settings.knn
    )
    null_cmi = np.asarray(
        [
            ksg_conditional_mutual_information(
                null[indices], mu2[indices], mu1[indices], k=settings.knn
            )
            for null in nulls
        ]
    )
    offset = float(np.nanmedian(null_cmi)) if null_cmi.size else 0.0
    calibrated = max(0.0, raw - offset)
    resampled: list[float] = []
    subset_size = max(settings.knn + 3, int(0.8 * number))
    for _ in range(settings.uncertainty_resamples):
        subset = rng.choice(indices, size=subset_size, replace=False)
        estimate = ksg_conditional_mutual_information(
            h0[subset], mu2[subset], mu1[subset], k=settings.knn
        )
        if np.isfinite(estimate):
            resampled.append(max(0.0, estimate - offset))
    if resampled:
        lower, upper = np.quantile(resampled, (0.05, 0.95))
    else:
        lower = upper = math.nan
    return calibrated, float(lower), float(upper), int(number)


def analyze_h0_associations(
    features: list[AssociationFeature],
    h0: np.ndarray,
    chain_id: np.ndarray,
    settings: AssociationSettings,
) -> list[dict[str, Any]]:
    """Compute binary, scalar, vector, and coverage-sensitive associations."""

    h0 = np.asarray(h0, dtype=float)
    chain_id = np.asarray(chain_id)
    rng = np.random.default_rng(settings.random_seed)
    nulls = _circular_h0_nulls(
        h0, chain_id, settings.permutations, rng
    )
    records: list[dict[str, Any]] = []
    scalar_names = (
        "mu1",
        "mu2",
        "region_probability",
        "relative_width_m1",
        "relative_width_m2",
        "contrast",
        "extent",
        "bounded_fraction",
        "fallback_fraction",
        "deficit_probability",
    )
    logger = logging.getLogger("posterior_landscape")
    for feature_index, feature in enumerate(features, start=1):
        family_region = (
            np.asarray(feature.family_region, dtype=bool)
            if feature.family_region is not None
            else np.asarray(feature.region, dtype=bool)
        )
        family_morphology = (
            np.asarray(feature.family_morphology, dtype=bool)
            if feature.family_morphology is not None
            else np.asarray(feature.morphology, dtype=bool)
        )
        family_location = (
            np.asarray(feature.family_location, dtype=bool)
            if feature.family_location is not None
            else np.asarray(feature.location, dtype=bool)
        )
        family_morphology_probability = float(
            np.mean(family_morphology)
        )
        family_location_probability = float(
            np.mean(family_location)
        )
        family_region_probability = float(np.mean(family_region))
        configuration_probability = (
            float(np.mean(feature.region))
            if feature.record_scope == "location_configuration"
            else math.nan
        )
        record: dict[str, Any] = {
            "ID": feature.identifier,
            "type": feature.kind,
            "record_scope": feature.record_scope,
            "parent_ID": feature.parent_identifier,
            "configuration": feature.configuration,
            "P_morph": float(np.mean(feature.morphology)),
            "P_loc": float(np.mean(feature.location)),
            "P_reg": float(np.mean(feature.region)),
            "P_family_morph": family_morphology_probability,
            "P_family_loc": family_location_probability,
            "P_family_reg": family_region_probability,
            "P_config": configuration_probability,
            "P_config_given_reg": (
                configuration_probability / family_region_probability
                if np.isfinite(configuration_probability)
                and family_region_probability > 0.0
                else math.nan
            ),
        }
        binary_tests = (
            ("morph", feature.morphology, np.ones(h0.size, dtype=bool)),
            ("loc_given_morph", feature.location, feature.morphology),
            ("reg_given_loc", feature.region, feature.location),
        )
        for prefix, indicator, conditioning in binary_tests:
            if feature.record_scope == "location_configuration":
                effect = probability = math.nan
                first = second = 0
            else:
                effect, probability, first, second = _binary_association(
                    h0, indicator, conditioning, nulls
                )
            record[f"{prefix}_rrb"] = effect
            record[f"{prefix}_p_perm"] = probability
            record[f"{prefix}_n_yes"] = first
            record[f"{prefix}_n_no"] = second

        if feature.record_scope == "location_configuration":
            effect, probability, first, second = _binary_association(
                h0, feature.region, family_region, nulls
            )
        else:
            effect = probability = math.nan
            first = second = 0
        record["config_rrb"] = effect
        record["config_p_perm"] = probability
        record["config_n_yes"] = first
        record["config_n_no"] = second

        continuous_selection = (
            np.zeros(h0.size, dtype=bool)
            if feature.suppress_continuous
            else feature.region
        )

        for name in scalar_names:
            rho, information, number, _ = _continuous_association(
                h0,
                getattr(feature, name),
                continuous_selection,
                nulls,
                k=settings.knn,
            )
            record[f"{name}_rho"] = rho
            record[f"{name}_mi_bits"] = information
            record[f"{name}_n"] = number

        joint, joint_number, _ = _joint_association(
            h0,
            feature.mu1,
            feature.mu2,
            continuous_selection,
            nulls,
            k=settings.knn,
        )
        cmi, cmi_lower, cmi_upper, cmi_number = _conditional_association(
            h0,
            feature.mu1,
            feature.mu2,
            continuous_selection,
            nulls,
            settings,
            rng,
        )
        record["mu_vector_mi_bits"] = joint
        record["mu_vector_n"] = joint_number
        record["mu2_given_mu1_cmi_bits"] = cmi
        record["mu2_given_mu1_cmi_lower"] = cmi_lower
        record["mu2_given_mu1_cmi_upper"] = cmi_upper
        record["mu2_given_mu1_cmi_n"] = cmi_number

        vector_mask = (
            continuous_selection
            & np.isfinite(feature.mu1)
            & np.isfinite(feature.mu2)
        )
        if np.count_nonzero(vector_mask) > 2:
            vector = np.column_stack(
                [h0[vector_mask], feature.mu1[vector_mask], feature.mu2[vector_mask]]
            )
            covariance = np.cov(vector, rowvar=False)
            correlation = np.full((3, 3), np.nan)
            for row in range(3):
                for column in range(3):
                    if np.ptp(vector[:, row]) > 0.0 and np.ptp(
                        vector[:, column]
                    ) > 0.0:
                        correlation[row, column] = float(
                            spearmanr(
                                vector[:, row], vector[:, column]
                            ).statistic
                        )
        else:
            covariance = correlation = np.full((3, 3), np.nan)
        for row, first in enumerate(("h0", "mu1", "mu2")):
            for column, second in enumerate(("h0", "mu1", "mu2")):
                record[f"cov_{first}_{second}"] = float(covariance[row, column])
                record[f"rho_{first}_{second}"] = float(correlation[row, column])

        for threshold in (0.5, 0.8):
            selected = continuous_selection & (
                feature.bounded_fraction >= threshold
            )
            rho, information, number, _ = _continuous_association(
                h0, feature.mu1, selected, nulls, k=settings.knn
            )
            suffix = str(int(100 * threshold))
            record[f"mu1_coverage{suffix}_rho"] = rho
            record[f"mu1_coverage{suffix}_mi_bits"] = information
            record[f"mu1_coverage{suffix}_n"] = number
        records.append(record)
        logger.info(
            "%s association: %d/%d features (%s).",
            settings.parameter_name,
            feature_index,
            len(features),
            feature.identifier,
        )
    return records


def _hpd_contours(
    x: np.ndarray,
    y: np.ndarray,
    probabilities: tuple[float, float] = (0.5, 0.9),
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[finite]
    y = np.asarray(y, dtype=float)[finite]
    selected_weights = (
        None if weights is None else np.asarray(weights, dtype=float)[finite]
    )
    hist, edges_x, edges_y = np.histogram2d(
        x, y, bins=55, density=True, weights=selected_weights
    )
    hist = ndimage.gaussian_filter(hist, sigma=1.0, mode="nearest")
    ordered = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    thresholds = [
        float(ordered[min(np.searchsorted(cumulative, probability), ordered.size - 1)])
        for probability in probabilities
    ]
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    return centers_x, centers_y, hist.T, sorted(set(thresholds))


def _draw_joint(axis: Any, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if np.count_nonzero(np.isfinite(x) & np.isfinite(y)) < 30:
        axis.text(0.5, 0.5, "insufficient support", ha="center", va="center")
        return
    centers_x, centers_y, density, levels = _hpd_contours(x, y)
    if levels:
        axis.contour(centers_x, centers_y, density, levels=levels, colors=[color])
    axis.axvline(float(np.median(x)), color="0.45", ls=":", lw=0.7)
    axis.axhline(float(np.median(y)), color="0.45", ls=":", lw=0.7)


def make_h0_corner_figures(
    base: Path,
    features: list[AssociationFeature],
    records: list[dict[str, Any]],
    h0: np.ndarray,
    plot: PlotSettings,
    parameter_name: str = "H0",
    parameter_label: str = r"$H_0$",
    *,
    parameter_unit: str = "",
    coordinate1_name: str = "m1",
    coordinate2_name: str = "m2",
    coordinate1_label: str = r"$m_1$",
    coordinate2_label: str = r"$m_2$",
    coordinate1_unit: str = "",
    coordinate2_unit: str = "",
) -> list[Path]:
    """Write per-feature PDF corners and one compact PNG vector summary."""

    def with_unit(label: str, unit: str) -> str:
        return f"{label} [{unit}]" if unit else label

    def centroid_label(
        name: str, label: str, unit: str, index: int
    ) -> str:
        if name in {"m1", "m2"}:
            result = rf"$\widehat\mu_{index}$"
        else:
            core = (
                label[1:-1]
                if label.startswith("$") and label.endswith("$")
                else label
            )
            result = rf"$\widehat{{{core}}}$"
        return with_unit(result, unit)

    parameter_axis_label = with_unit(parameter_label, parameter_unit)
    first_centroid_label = centroid_label(
        coordinate1_name, coordinate1_label, coordinate1_unit, 1
    )
    second_centroid_label = centroid_label(
        coordinate2_name, coordinate2_label, coordinate2_unit, 2
    )

    pairs = [
        (feature, record)
        for feature, record in zip(features, records)
        if not feature.suppress_continuous
    ]
    if not pairs:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {
        "peak": "#e67e22",
        "pit": "#377eb8",
        "ridge": "#1b9e91",
        "valley": "#386cb0",
        "shoulder": "#8e44ad",
    }
    written: list[Path] = []
    if plot.write_pdf:
        path = base.with_suffix(".pdf")
        with PdfPages(path) as pdf:
            for feature, record in pairs:
                selected = feature.region & np.isfinite(feature.mu1) & np.isfinite(feature.mu2)
                variables = (h0[selected], feature.mu1[selected], feature.mu2[selected])
                labels = (
                    parameter_axis_label,
                    first_centroid_label,
                    second_centroid_label,
                )
                color = colors.get(feature.kind, "0.3")
                figure, axes = plt.subplots(3, 3, figsize=(7.2, 7.2))
                for row in range(3):
                    for column in range(3):
                        axis = axes[row, column]
                        if row < column:
                            axis.axis("off")
                        elif row == column:
                            axis.hist(
                                variables[row],
                                bins=35,
                                density=True,
                                color=color,
                                alpha=0.45,
                            )
                            axis.set_yticks([])
                        else:
                            _draw_joint(axis, variables[column], variables[row], color)
                        if row == 2 and not (row < column):
                            axis.set_xlabel(labels[column])
                        if column == 0 and row > 0:
                            axis.set_ylabel(labels[row])
                figure.suptitle(
                    f"{feature.identifier}: {parameter_name} and adaptive feature centroid\n"
                    f"I({parameter_name};mu1,mu2)={record['mu_vector_mi_bits']:.3f} bits; "
                    f"I({parameter_name};mu2|mu1)={record['mu2_given_mu1_cmi_bits']:.3f} bits"
                )
                figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
                pdf.savefig(figure)
                plt.close(figure)
        written.append(path)

    if plot.write_png:
        path = base.with_suffix(".png")
        number = max(1, len(pairs))
        figure, axes = plt.subplots(number, 3, figsize=(10.5, 2.65 * number), squeeze=False)
        for row, (feature, record) in enumerate(pairs):
            selected = feature.region & np.isfinite(feature.mu1) & np.isfinite(feature.mu2)
            color = colors.get(feature.kind, "0.3")
            pairs = (
                (
                    h0[selected],
                    feature.mu1[selected],
                    parameter_axis_label,
                    first_centroid_label,
                ),
                (
                    h0[selected],
                    feature.mu2[selected],
                    parameter_axis_label,
                    second_centroid_label,
                ),
                (
                    feature.mu1[selected],
                    feature.mu2[selected],
                    first_centroid_label,
                    second_centroid_label,
                ),
            )
            for column, (x, y, xlabel, ylabel) in enumerate(pairs):
                _draw_joint(axes[row, column], x, y, color)
                axes[row, column].set_xlabel(xlabel)
                axes[row, column].set_ylabel(ylabel)
            axes[row, 0].set_title(
                f"{feature.identifier}   I12={record['mu_vector_mi_bits']:.3f} bits; "
                f"I2|1={record['mu2_given_mu1_cmi_bits']:.3f}"
            )
        figure.suptitle(
            f"Compact {parameter_label} – {coordinate1_label} – "
            f"{coordinate2_label} corner summary"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def make_location_configuration_figures(
    base: Path,
    features: list[AssociationFeature],
    results: dict[str, LocationConfigurationResult],
    weights: np.ndarray,
    plot: PlotSettings,
) -> list[Path]:
    """Plot separate centroid HPDs for robust location configurations."""

    multimodal = [
        feature for feature in features if results[feature.identifier].multimodal
    ]
    if not multimodal:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    columns = min(3, len(multimodal))
    rows = int(math.ceil(len(multimodal) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 3.9 * rows),
        squeeze=False,
    )
    colors = plt.get_cmap("tab10")
    for axis, feature in zip(axes.ravel(), multimodal):
        result = results[feature.identifier]
        for index, label in enumerate(result.configuration_labels):
            selected = result.labels == index
            x = np.asarray(feature.mu1[selected], dtype=float)
            y = np.asarray(feature.mu2[selected], dtype=float)
            color = colors(index % 10)
            axis.scatter(x, y, s=5, alpha=0.10, color=color, rasterized=True)
            centers_x, centers_y, density, levels = _hpd_contours(
                np.log(x),
                np.log(y),
                probabilities=(0.5, 0.9),
                weights=np.asarray(weights, dtype=float)[selected],
            )
            if levels:
                axis.contour(
                    np.exp(centers_x),
                    np.exp(centers_y),
                    density,
                    levels=levels,
                    colors=[color],
                    linewidths=(1.6, 1.0),
                )
            axis.scatter(
                np.median(x),
                np.median(y),
                marker="*",
                s=90,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=(
                    f"{feature.identifier}-{label}: "
                    f"P={result.probabilities[index]:.1%}"
                ),
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\widehat\mu_1$")
        axis.set_ylabel(r"$\widehat\mu_2$")
        axis.set_title(f"{feature.identifier}: posterior location configurations")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.18, lw=0.5)
    for axis in axes.ravel()[len(multimodal) :]:
        axis.axis("off")
    figure.suptitle(
        "Robust multimodal feature locations (separate conditional HPDs)"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    written: list[Path] = []
    if plot.write_pdf:
        path = base.with_suffix(".pdf")
        figure.savefig(path, bbox_inches="tight")
        written.append(path)
    if plot.write_png:
        path = base.with_suffix(".png")
        figure.savefig(path, dpi=220, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written
