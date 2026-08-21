# ============================================================
# Draw-level posterior for 1D feature presence, morphology,
# location, edges, width, morphology-specific strength, and
# moving-band mass scale.
#
# Reference identities are taken from the posterior-median
# feature catalogue. In every posterior draw:
#   * peaks and shoulders are allowed to represent the same
#     positive/turnover feature family;
#   * dips form the suppression family;
#   * each candidate proposes only to its nearest compatible
#     posterior-median feature in ln(m);
#   * candidates outside the finite identity range of every
#     reference feature are rejected;
#   * if several candidates propose to the same feature, only
#     the closest one is retained;
#   * unmatched reference features are absent;
#   * unmatched draw-level candidates are ignored.
#
# Feature location and complete boundary support are recorded separately.
# Location intervals are conditional on a compatible location being found;
# edges, widths, band integrals, and strengths require complete boundaries.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    # The notebook environment normally provides tqdm.  Retain functional
    # execution without it rather than failing after the vectorized work.
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------------------------------------------
# Configuration inherited from the reference discovery.
# ------------------------------------------------------------

FEATURE_POSTERIOR_QUANTILES = (
    0.05,
    0.50,
    0.95,
)

# If a peak and a shoulder are found at the same mass scale in one
# realization, retain the peak: it is the stronger realization of the
# same positive/turnover structure. Reuse the already adopted cross-scale
# matching distance; no additional tuned scale is introduced.
FEATURE_PEAK_SHOULDER_DEDUP_LNM = FEATURE_MATCH_TOLERANCE_LNM

FEATURE_POSTERIOR_CSV = os.path.join(
    fin,
    f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_pm1_feature_posterior_summary.csv",
)

FEATURE_POSTERIOR_TEX = os.path.join(
    fin,
    f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_pm1_feature_posterior_table.tex",
)


# ------------------------------------------------------------
# Validate and normalize the posterior draws without changing order.
# ------------------------------------------------------------

_required_draw_feature_products = (
    "feature_bands",
    "pm1_feature_samples",
    "FEATURE_SMOOTH_SCALES_LNM",
    "FEATURE_REFERENCE_SMOOTH_LNM",
    "FEATURE_MIN_PERSISTENCE",
    "FEATURE_MATCH_TOLERANCE_LNM",
    "minimum_extrema_distance",
    "m_high_percentile_samples",
    "MASS_DENSITY_MEASURE",
    "MASS_DENSITY_OUTPUT_TAG",
    "MASS_DENSITY_PLAIN_LABEL",
    "MASS_DENSITY_TEX_LABEL",
    "SHOULDER_ALPHA_MAX_SELECTED",
    "MASS_MEASURE_CELL_EDGES",
    "MASS_MEASURE_CELL_WIDTHS",
    "MASS_MEASURE_WIDTHS",
)

_missing_draw_feature_products = [
    name
    for name in _required_draw_feature_products
    if name not in globals()
]

if _missing_draw_feature_products:
    raise RuntimeError(
        "Run the reference-discovery and median-band cells first. Missing: "
        + ", ".join(_missing_draw_feature_products)
    )

pm1_draw_feature_density = np.asarray(
    pm1_feature_samples,
    dtype=float,
)

if (
    pm1_draw_feature_density.ndim != 2
    or pm1_draw_feature_density.shape[1] != m_grid.size
):
    raise RuntimeError(
        "pm1_feature_samples must have shape "
        "(number of draws, len(m_grid))."
    )

pm1_draw_feature_density = np.where(
    np.isfinite(pm1_draw_feature_density)
    & (pm1_draw_feature_density >= 0.0),
    pm1_draw_feature_density,
    0.0,
)

_draw_feature_norm = np.einsum(
    "si,i->s",
    pm1_draw_feature_density,
    MASS_MEASURE_WIDTHS,
    optimize=True,
)

if np.any(
    (~np.isfinite(_draw_feature_norm))
    | (_draw_feature_norm <= 0.0)
):
    _bad_draws = np.flatnonzero(
        (~np.isfinite(_draw_feature_norm))
        | (_draw_feature_norm <= 0.0)
    )
    raise RuntimeError(
        "Invalid posterior marginal normalization in draws: "
        + ", ".join(str(int(i)) for i in _bad_draws[:20])
    )

pm1_draw_feature_density = (
    pm1_draw_feature_density
    / _draw_feature_norm[:, None]
)

number_feature_draws = pm1_draw_feature_density.shape[0]


# ------------------------------------------------------------
# Vectorized smoothing and derivatives for all posterior draws.
# ------------------------------------------------------------

feature_draw_curves = {}

for _smooth_scale in tqdm(
    FEATURE_SMOOTH_SCALES_LNM,
    desc="Vectorized 1D feature curves",
    unit="scale",
):
    _sigma_grid = _smooth_scale / dlog_m_grid

    _p_smooth = gaussian_filter1d(
        pm1_draw_feature_density,
        sigma=_sigma_grid,
        axis=1,
        mode="nearest",
    )

    _p_smooth = np.where(
        np.isfinite(_p_smooth)
        & (_p_smooth > 0.0),
        _p_smooth,
        0.0,
    )

    _smooth_norm = np.einsum(
        "si,i->s",
        _p_smooth,
        MASS_MEASURE_WIDTHS,
        optimize=True,
    )

    if np.any(
        (~np.isfinite(_smooth_norm))
        | (_smooth_norm <= 0.0)
    ):
        raise RuntimeError(
            f"Invalid smoothed normalization at scale {_smooth_scale:g}."
        )

    _p_smooth = _p_smooth / _smooth_norm[:, None]

    _valid = (
        np.isfinite(_p_smooth)
        & (
            _p_smooth
            >= FEATURE_DERIVATIVE_DENSITY_FLOOR
        )
    )

    _log_p = np.full_like(
        _p_smooth,
        np.nan,
    )

    np.log(
        _p_smooth,
        out=_log_p,
        where=_valid,
    )

    _alpha = np.gradient(
        _log_p,
        log_m_grid,
        axis=1,
        edge_order=2,
    )

    _kappa = np.gradient(
        _alpha,
        log_m_grid,
        axis=1,
        edge_order=2,
    )

    feature_draw_curves[_smooth_scale] = {
        "p": _p_smooth,
        "alpha": _alpha,
        "kappa": _kappa,
        "valid": _valid,
    }


# ------------------------------------------------------------
# Draw-level detection helpers.
# ------------------------------------------------------------

def _safe_local_maxima(values, valid_mask):
    """The reference finder, returning no candidates if the domain is short."""

    valid_mask = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(values)
    )

    if np.count_nonzero(valid_mask) < 3:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
        )

    try:
        return _find_local_maxima(
            np.asarray(values, dtype=float),
            valid_mask,
        )
    except RuntimeError:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
        )


def _persistent_draw_records(
    indices_by_scale,
    prominence_by_scale,
):
    """Apply the same cross-scale persistence rule to one draw."""

    reference_indices = np.asarray(
        indices_by_scale[FEATURE_REFERENCE_SMOOTH_LNM],
        dtype=int,
    )

    reference_prominence = np.asarray(
        prominence_by_scale[FEATURE_REFERENCE_SMOOTH_LNM],
        dtype=float,
    )

    records = []

    for reference_index, prominence in zip(
        reference_indices,
        reference_prominence,
    ):
        reference_log_mass = log_m_grid[reference_index]
        persistence = 0

        for smooth_scale in FEATURE_SMOOTH_SCALES_LNM:
            candidates = np.asarray(
                indices_by_scale[smooth_scale],
                dtype=int,
            )

            if candidates.size == 0:
                continue

            nearest_distance = float(
                np.min(
                    np.abs(
                        log_m_grid[candidates]
                        - reference_log_mass
                    )
                )
            )

            if nearest_distance <= FEATURE_MATCH_TOLERANCE_LNM:
                persistence += 1

        if persistence >= FEATURE_MIN_PERSISTENCE:
            records.append(
                {
                    "index": int(reference_index),
                    "prominence": float(prominence),
                    "persistence": int(persistence),
                }
            )

    return records


def _draw_shoulder_transition_band(
    onset_index,
    alpha,
    kappa,
    valid_mask,
    persistent_up_knees,
    persistent_down_knees,
):
    """Apply the reference K+--K---kappa-zero definition to one draw."""

    up_indices = np.sort(
        np.asarray(
            [record["index"] for record in persistent_up_knees],
            dtype=int,
        )
    )

    down_indices = np.sort(
        np.asarray(
            [record["index"] for record in persistent_down_knees],
            dtype=int,
        )
    )

    preceding_up = up_indices[up_indices < onset_index]
    following_down = down_indices[down_indices > onset_index]

    if preceding_up.size == 0 or following_down.size == 0:
        raise RuntimeError("Incomplete shoulder-transition landmarks.")

    up_index = int(preceding_up[-1])
    down_index = int(following_down[0])

    if not (
        up_index
        < onset_index
        < down_index
    ):
        raise RuntimeError("Invalid draw-level shoulder ordering.")

    right_crossing = _first_negative_to_positive_kappa_crossing(
        kappa,
        start_index=down_index,
        valid_mask=(
            np.asarray(valid_mask, dtype=bool)
            & np.isfinite(kappa)
        ),
    )

    alpha_prominence_band = _prominence_band(
        values=alpha,
        center_index=onset_index,
        valid_mask=(
            np.asarray(valid_mask, dtype=bool)
            & np.isfinite(alpha)
        ),
        invert=False,
        rel_height=FEATURE_BAND_REL_HEIGHT,
    )

    left_mass = float(m_grid[up_index])
    center_mass = float(m_grid[down_index])
    right_mass = float(right_crossing["mass"])

    if not (
        left_mass
        < center_mass
        < right_mass
    ):
        raise RuntimeError("Invalid draw-level shoulder extent.")

    return {
        "left": left_mass,
        "center": center_mass,
        "right": right_mass,
        "width_mass": right_mass - left_mass,
        "width_lnm": float(np.log(right_mass / left_mass)),
        "width_relative": float(
            (right_mass - left_mass) / center_mass
        ),
        "prominence": float(alpha_prominence_band["prominence"]),
        "detected_type": "shoulder",
        "onset_mass": float(m_grid[onset_index]),
    }


def _draw_shoulder_location_candidate(
    onset_index,
    persistent_down_knees,
    prominence,
):
    """
    Locate a shoulder without requiring its outer boundaries.

    The reported shoulder location is the first persistent maximum-
    steepening landmark K- after the persistent shoulder onset.  Failure to
    find that landmark means that the shoulder is not location-supported;
    failure of the later kappa=0 crossing affects only bounded support.
    """

    down_indices = np.sort(
        np.asarray(
            [record["index"] for record in persistent_down_knees],
            dtype=int,
        )
    )

    following_down = down_indices[down_indices > onset_index]

    if following_down.size == 0:
        raise RuntimeError("No persistent K- follows the shoulder onset.")

    down_index = int(following_down[0])

    return {
        "center": float(m_grid[down_index]),
        "center_index": down_index,
        "onset_mass": float(m_grid[onset_index]),
        "onset_index": int(onset_index),
        "prominence": float(prominence),
        "detected_type": "shoulder",
    }


def _draw_extremum_location_candidate(record, detected_type):
    """Locate a persistent peak or dip before constructing its extent."""

    center_index = int(record["index"])

    return {
        "center": float(m_grid[center_index]),
        "center_index": center_index,
        "prominence": float(record["prominence"]),
        "detected_type": detected_type,
    }


def _draw_peak_or_dip_band(
    values,
    valid_mask,
    center_index,
    detected_type,
):
    """Construct a draw-level half-prominence peak or dip extent."""

    band = _prominence_band(
        values=values,
        center_index=center_index,
        valid_mask=valid_mask,
        invert=(detected_type == "dip"),
        rel_height=FEATURE_BAND_REL_HEIGHT,
    )

    band["width_mass"] = band["right"] - band["left"]
    band["width_relative"] = (
        band["width_mass"] / band["center"]
    )
    band["detected_type"] = detected_type
    return band


def _log_curve_at_mass(
    density,
    mass,
):
    """Interpolate ln p at one mass in ln(m)."""

    density = np.asarray(
        density,
        dtype=float,
    )

    valid = (
        np.isfinite(density)
        & (density > 0.0)
    )

    if np.count_nonzero(valid) < 2:
        return np.nan

    log_mass = float(
        np.log(mass)
    )

    valid_log_m = log_m_grid[valid]

    if not (
        valid_log_m[0]
        <= log_mass
        <= valid_log_m[-1]
    ):
        return np.nan

    return float(
        np.interp(
            log_mass,
            valid_log_m,
            np.log(density[valid]),
        )
    )


def _draw_feature_contrast(
    density,
    candidate,
):
    """
    Morphology-specific strength from one draw's own feature geometry.

    Peaks and dips use the topological saddle level implied by their
    draw-specific prominence.  If Pi is the prominence, then
    p_sad = p(m_C) - Pi for a peak and p_sad = p(m_C) + Pi for a dip.
    The half-prominence crossings define the width only, so the resulting
    contrast is not capped at ln(2) and does not depend on the potentially
    distant left and right prominence-base locations.
    Shoulders use the change between the pre- and post-transition mean
    logarithmic slopes across their draw-specific transition landmarks.
    """

    feature_type = candidate["detected_type"]

    if feature_type in (
        "peak",
        "dip",
    ):
        log_p_center = _log_curve_at_mass(
            density,
            candidate["center"],
        )

        prominence = float(
            candidate["prominence"]
        )

        if not (
            np.isfinite(log_p_center)
            and np.isfinite(prominence)
            and prominence > 0.0
        ):
            return np.nan

        p_center = float(
            np.exp(log_p_center)
        )

        if feature_type == "peak":
            p_saddle = p_center - prominence

            if not (
                np.isfinite(p_saddle)
                and p_saddle > 0.0
            ):
                return np.nan

            return float(np.log(p_center / p_saddle))

        p_saddle = p_center + prominence

        if not (
            np.isfinite(p_saddle)
            and p_saddle > 0.0
        ):
            return np.nan

        return float(np.log(p_saddle / p_center))

    if feature_type == "shoulder":
        log_p_left = _log_curve_at_mass(
            density,
            candidate["left"],
        )

        log_p_center = _log_curve_at_mass(
            density,
            candidate["center"],
        )

        log_p_right = _log_curve_at_mass(
            density,
            candidate["right"],
        )

        if not np.all(
            np.isfinite(
                [
                    log_p_left,
                    log_p_center,
                    log_p_right,
                ]
            )
        ):
            return np.nan

        onset_mass = candidate.get(
            "onset_mass",
            np.nan,
        )

        if not (
            np.isfinite(onset_mass)
            and candidate["left"]
            < onset_mass
            < candidate["center"]
            < candidate["right"]
        ):
            return np.nan

        log_p_onset = _log_curve_at_mass(
            density,
            onset_mass,
        )

        if not np.isfinite(log_p_onset):
            return np.nan

        slope_pre = (
            (log_p_onset - log_p_left)
            / np.log(
                onset_mass
                / candidate["left"]
            )
        )

        slope_post = (
            (log_p_right - log_p_center)
            / np.log(
                candidate["right"]
                / candidate["center"]
            )
        )

        return float(
            slope_pre
            - slope_post
        )

    raise RuntimeError(
        f"Unsupported feature type: {feature_type}"
    )


def _deduplicate_peak_shoulder_candidates(candidates):
    """A resolved density peak supersedes a co-located shoulder."""

    peak_centers = np.asarray(
        [
            candidate["center"]
            for candidate in candidates
            if candidate["detected_type"] == "peak"
        ],
        dtype=float,
    )

    if peak_centers.size == 0:
        return candidates

    retained = []

    for candidate in candidates:
        if candidate["detected_type"] == "peak":
            retained.append(candidate)
            continue

        nearest_peak = float(
            np.min(
                np.abs(
                    np.log(peak_centers)
                    - np.log(candidate["center"])
                )
            )
        )

        if nearest_peak > FEATURE_PEAK_SHOULDER_DEDUP_LNM:
            retained.append(candidate)

    return retained


def _reference_identity_intervals_lnm(
    reference_ids,
    reference_lookup,
):
    """
    Finite identity intervals separated by geometric midpoints.

    Internal boundaries are halfway between adjacent reference locations in
    ln(m).  The two outer intervals extend by half of the nearest-reference
    spacing.  The construction uses only the posterior-median catalogue and
    introduces no additional hand-selected matching scale.
    """

    if not reference_ids:
        return {}

    ordered_ids = sorted(
        reference_ids,
        key=lambda feature_id: reference_lookup[feature_id]["location"],
    )

    reference_log_locations = np.log(
        np.asarray(
            [
                reference_lookup[feature_id]["location"]
                for feature_id in ordered_ids
            ],
            dtype=float,
        )
    )

    if reference_log_locations.size == 1:
        return {
            ordered_ids[0]: (
                float(log_m_grid[0]),
                float(log_m_grid[-1]),
            )
        }

    boundaries = np.empty(
        reference_log_locations.size + 1,
        dtype=float,
    )

    boundaries[1:-1] = 0.5 * (
        reference_log_locations[:-1]
        + reference_log_locations[1:]
    )

    boundaries[0] = (
        reference_log_locations[0]
        - 0.5
        * (
            reference_log_locations[1]
            - reference_log_locations[0]
        )
    )

    boundaries[-1] = (
        reference_log_locations[-1]
        + 0.5
        * (
            reference_log_locations[-1]
            - reference_log_locations[-2]
        )
    )

    boundaries[0] = max(
        boundaries[0],
        float(log_m_grid[0]),
    )

    boundaries[-1] = min(
        boundaries[-1],
        float(log_m_grid[-1]),
    )

    return {
        feature_id: (
            float(boundaries[index]),
            float(boundaries[index + 1]),
        )
        for index, feature_id in enumerate(ordered_ids)
    }


def _nearest_reference_location_match(
    reference_ids,
    candidates,
    reference_lookup,
    identity_intervals_lnm,
):
    """
    Match candidates to their nearest compatible median feature.

    A candidate is never reassigned to its second-nearest feature merely
    because its nearest feature already has a better match.  This prevents a
    missing high-mass feature from causing a cascade of lower-mass candidates
    through the remaining reference labels.
    """

    if not reference_ids or not candidates:
        return []

    reference_locations = np.asarray(
        [reference_lookup[feature_id]["location"] for feature_id in reference_ids],
        dtype=float,
    )

    reference_log_locations = np.log(reference_locations)
    proposals = {
        feature_id: []
        for feature_id in reference_ids
    }

    for candidate in candidates:
        candidate_log_location = float(
            np.log(candidate["center"])
        )

        distances = np.abs(
            reference_log_locations
            - candidate_log_location
        )

        nearest_index = int(
            np.argmin(distances)
        )

        feature_id = reference_ids[nearest_index]
        interval_left, interval_right = identity_intervals_lnm[feature_id]

        if not (
            interval_left
            <= candidate_log_location
            <= interval_right
        ):
            continue

        proposals[feature_id].append(
            (
                candidate,
                float(distances[nearest_index]),
            )
        )

    matches = []

    for feature_id in reference_ids:
        feature_proposals = proposals[feature_id]

        if not feature_proposals:
            continue

        candidate, distance = min(
            feature_proposals,
            key=lambda proposal: (
                proposal[1],
                -float(proposal[0]["prominence"]),
            ),
        )

        matches.append(
            (
                feature_id,
                candidate,
                distance,
            )
        )

    return matches


# ------------------------------------------------------------
# Reference feature families and aligned output arrays.
# ------------------------------------------------------------

feature_reference_lookup = {}

for _feature_type, _prefix in (
    ("peak", "P"),
    ("dip", "D"),
    ("shoulder", "S"),
):
    for _band in feature_bands[_feature_type]:
        _feature_id = f"{_prefix}{int(_band['number'])}"
        feature_reference_lookup[_feature_id] = {
            "id": _feature_id,
            "reference_type": _feature_type,
            "family": (
                "suppression"
                if _feature_type == "dip"
                else "positive"
            ),
            "location": float(_band["center"]),
            "band": _band,
        }

feature_posterior_order = sorted(
    feature_reference_lookup,
    key=lambda feature_id: feature_reference_lookup[feature_id]["location"],
)

positive_reference_ids = [
    feature_id
    for feature_id in feature_posterior_order
    if feature_reference_lookup[feature_id]["family"] == "positive"
]

suppression_reference_ids = [
    feature_id
    for feature_id in feature_posterior_order
    if feature_reference_lookup[feature_id]["family"] == "suppression"
]

positive_identity_intervals_lnm = _reference_identity_intervals_lnm(
    positive_reference_ids,
    feature_reference_lookup,
)

suppression_identity_intervals_lnm = _reference_identity_intervals_lnm(
    suppression_reference_ids,
    feature_reference_lookup,
)

feature_identity_intervals_lnm = {
    **positive_identity_intervals_lnm,
    **suppression_identity_intervals_lnm,
}

print(
    "\nDraw-level feature identity intervals "
    "(geometric midpoints between compatible median features):"
)

for _feature_id in feature_posterior_order:
    _identity_left_lnm, _identity_right_lnm = (
        feature_identity_intervals_lnm[_feature_id]
    )
    print(
        f"  {_feature_id:>3s}: "
        f"[{np.exp(_identity_left_lnm):.3f}, "
        f"{np.exp(_identity_right_lnm):.3f}]"
        f"{COORDINATE_UNIT_SUFFIX}"
    )

feature_draw_posteriors = {}

for _feature_id in feature_posterior_order:
    feature_draw_posteriors[_feature_id] = {
        "located": np.zeros(number_feature_draws, dtype=bool),
        "bounded": np.zeros(number_feature_draws, dtype=bool),
        # Backward-compatible alias: "present" means fully bounded.
        "present": np.zeros(number_feature_draws, dtype=bool),
        "realized_type": np.full(
            number_feature_draws,
            "absent",
            dtype=object,
        ),
        "left": np.full(number_feature_draws, np.nan),
        "center": np.full(number_feature_draws, np.nan),
        "right": np.full(number_feature_draws, np.nan),
        "width_mass": np.full(number_feature_draws, np.nan),
        "width_lnm": np.full(number_feature_draws, np.nan),
        "width_relative": np.full(number_feature_draws, np.nan),
        "prominence": np.full(number_feature_draws, np.nan),
        "onset_mass": np.full(number_feature_draws, np.nan),
        "background_left_mass": np.full(number_feature_draws, np.nan),
        "background_right_mass": np.full(number_feature_draws, np.nan),
         "contrast": np.full(number_feature_draws, np.nan),
        # For peaks and dips:
        # A = 1 - exp(-Delta), bounded between zero and one.
        # Undefined for shoulders, whose contrast measures a slope change.
        "relative_prominence": np.full(number_feature_draws, np.nan),
        "match_distance_lnm": np.full(number_feature_draws, np.nan),
        "band_probability": np.full(number_feature_draws, np.nan),
        "mu_hat": np.full(number_feature_draws, np.nan),
        "mu_deficit": np.full(number_feature_draws, np.nan),
    }

feature_extent_failure_count = {
    "peak": 0,
    "dip": 0,
    "shoulder": 0,
}

feature_location_failure_count = {
    "peak": 0,
    "dip": 0,
    "shoulder": 0,
}

feature_unmatched_candidate_count = {
    "positive": 0,
    "suppression": 0,
}


# ------------------------------------------------------------
# Detect, construct extents, and match every posterior draw.
# ------------------------------------------------------------

for _draw_index in tqdm(
    range(number_feature_draws),
    desc="Posterior 1D feature inference",
    unit="draw",
):
    _indices_by_type = {
        feature_type: {}
        for feature_type in (
            "peak",
            "dip",
            "shoulder",
            "up_knee",
            "down_knee",
        )
    }

    _prominence_by_type = {
        feature_type: {}
        for feature_type in _indices_by_type
    }

    for _smooth_scale in FEATURE_SMOOTH_SCALES_LNM:
        _curves = feature_draw_curves[_smooth_scale]
        _p = _curves["p"][_draw_index]
        _alpha = _curves["alpha"][_draw_index]
        _kappa = _curves["kappa"][_draw_index]
        _valid = _curves["valid"][_draw_index]

        _peaks, _peak_prominence = _safe_local_maxima(
            _p,
            _valid,
        )

        _dips, _dip_prominence = _safe_local_maxima(
            -_p,
            _valid,
        )

        _shoulder_all, _shoulder_prominence_all = _safe_local_maxima(
            _alpha,
            _valid & np.isfinite(_alpha),
        )

        # Apply the same coordinate-consistent physical criterion used for
        # the posterior-median catalogue: alpha[dP/dm1] < 0. In log-density
        # mode this is alpha[dP/dln(m1)] < 1.
        _shoulder_keep = (
            _alpha[_shoulder_all]
            < SHOULDER_ALPHA_MAX_SELECTED
        )
        _shoulders = _shoulder_all[_shoulder_keep]
        _shoulder_prominence = _shoulder_prominence_all[_shoulder_keep]

        _down_knees, _down_prominence = _safe_local_maxima(
            -_kappa,
            _valid & np.isfinite(_kappa),
        )

        _up_knees, _up_prominence = _safe_local_maxima(
            _kappa,
            _valid & np.isfinite(_kappa),
        )

        for _feature_type, _indices, _prominence in (
            ("peak", _peaks, _peak_prominence),
            ("dip", _dips, _dip_prominence),
            ("shoulder", _shoulders, _shoulder_prominence),
            ("down_knee", _down_knees, _down_prominence),
            ("up_knee", _up_knees, _up_prominence),
        ):
            _indices_by_type[_feature_type][_smooth_scale] = _indices
            _prominence_by_type[_feature_type][_smooth_scale] = _prominence

    _persistent = {
        feature_type: _persistent_draw_records(
            _indices_by_type[feature_type],
            _prominence_by_type[feature_type],
        )
        for feature_type in _indices_by_type
    }

    _reference_curves_draw = feature_draw_curves[
        FEATURE_REFERENCE_SMOOTH_LNM
    ]

    _p_reference_draw = _reference_curves_draw["p"][_draw_index]
    _alpha_reference_draw = _reference_curves_draw["alpha"][_draw_index]
    _kappa_reference_draw = _reference_curves_draw["kappa"][_draw_index]
    _valid_reference_draw = _reference_curves_draw["valid"][_draw_index]

    _positive_candidates = []
    _suppression_candidates = []

    for _record in _persistent["peak"]:
        _positive_candidates.append(
            _draw_extremum_location_candidate(_record, "peak")
        )

    for _record in _persistent["dip"]:
        _suppression_candidates.append(
            _draw_extremum_location_candidate(_record, "dip")
        )

    for _record in _persistent["shoulder"]:
        try:
            _candidate = _draw_shoulder_location_candidate(
                onset_index=_record["index"],
                persistent_down_knees=_persistent["down_knee"],
                prominence=_record["prominence"],
            )
            _positive_candidates.append(_candidate)
        except (RuntimeError, ValueError, FloatingPointError):
            feature_location_failure_count["shoulder"] += 1

    _positive_candidates = _deduplicate_peak_shoulder_candidates(
        _positive_candidates
    )

    _positive_matches = _nearest_reference_location_match(
        positive_reference_ids,
        _positive_candidates,
        feature_reference_lookup,
        positive_identity_intervals_lnm,
    )

    _suppression_matches = _nearest_reference_location_match(
        suppression_reference_ids,
        _suppression_candidates,
        feature_reference_lookup,
        suppression_identity_intervals_lnm,
    )

    feature_unmatched_candidate_count["positive"] += max(
        0,
        len(_positive_candidates) - len(_positive_matches),
    )

    feature_unmatched_candidate_count["suppression"] += max(
        0,
        len(_suppression_candidates) - len(_suppression_matches),
    )

    for _feature_id, _candidate, _distance in (
        _positive_matches + _suppression_matches
    ):
        _output = feature_draw_posteriors[_feature_id]
        _output["located"][_draw_index] = True
        _output["realized_type"][_draw_index] = _candidate["detected_type"]
        _output["center"][_draw_index] = float(_candidate["center"])
        _output["prominence"][_draw_index] = float(
            _candidate["prominence"]
        )
        _output["match_distance_lnm"][_draw_index] = _distance

        if _candidate["detected_type"] == "shoulder":
            _output["onset_mass"][_draw_index] = float(
                _candidate["onset_mass"]
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if _candidate["detected_type"] in ("peak", "dip"):
                    _bounded_candidate = _draw_peak_or_dip_band(
                        _p_reference_draw,
                        _valid_reference_draw,
                        _candidate["center_index"],
                        _candidate["detected_type"],
                    )
                else:
                    _bounded_candidate = _draw_shoulder_transition_band(
                        onset_index=_candidate["onset_index"],
                        alpha=_alpha_reference_draw,
                        kappa=_kappa_reference_draw,
                        valid_mask=_valid_reference_draw,
                        persistent_up_knees=_persistent["up_knee"],
                        persistent_down_knees=_persistent["down_knee"],
                    )

        except (RuntimeError, ValueError, FloatingPointError):
            feature_extent_failure_count[
                _candidate["detected_type"]
            ] += 1
            continue

        _output["bounded"][_draw_index] = True
        _output["present"][_draw_index] = True

        for _field in (
            "left",
            "center",
            "right",
            "width_mass",
            "width_lnm",
            "width_relative",
            "prominence",
        ):
            _output[_field][_draw_index] = float(
                _bounded_candidate[_field]
            )

        if _candidate["detected_type"] in ("peak", "dip"):
            _output["background_left_mass"][_draw_index] = float(
                m_grid[int(_bounded_candidate["left_base_index"])]
            )
            _output["background_right_mass"][_draw_index] = float(
                m_grid[int(_bounded_candidate["right_base_index"])]
            )

        _draw_contrast = _draw_feature_contrast(
            _p_reference_draw,
            _bounded_candidate,
        )

        _output["contrast"][_draw_index] = _draw_contrast

        if (
            _candidate["detected_type"] in ("peak", "dip")
            and np.isfinite(_draw_contrast)
        ):
            # Stable evaluation of
            # A = 1 - exp(-Delta).
            #
            # For a peak:
            #   A_P = (p_C - p_sad) / p_C.
            #
            # For a dip:
            #   A_D = (p_sad - p_C) / p_sad.
            _output["relative_prominence"][_draw_index] = float(
                np.clip(
                    -np.expm1(-_draw_contrast),
                    0.0,
                    1.0,
                )
            )


# ------------------------------------------------------------
# Moving-band probability, generic mass scale, and dip-deficit centroid.
# All integrations use the measure selected in Cell 1.
# ------------------------------------------------------------

for _feature_id in feature_posterior_order:
    _output = feature_draw_posteriors[_feature_id]
    _bounded_indices = np.flatnonzero(_output["bounded"])

    if _bounded_indices.size == 0:
        continue

    _left = _output["left"][_bounded_indices]
    _right = _output["right"][_bounded_indices]

    if MASS_DENSITY_MEASURE == "linear":
        _left_coordinate = _left
        _right_coordinate = _right
    else:
        _left_coordinate = np.log(_left)
        _right_coordinate = np.log(_right)

    _overlap_coordinate_width = np.maximum(
        0.0,
        np.minimum(
            MASS_MEASURE_CELL_EDGES[1:][None, :],
            _right_coordinate[:, None],
        )
        - np.maximum(
            MASS_MEASURE_CELL_EDGES[:-1][None, :],
            _left_coordinate[:, None],
        ),
    )

    # Allocate each full quadrature weight in proportion to the fraction
    # of its selected-coordinate cell covered by the moving interval.
    _overlap = (
        MASS_MEASURE_WIDTHS[None, :]
        * _overlap_coordinate_width
        / MASS_MEASURE_CELL_WIDTHS[None, :]
    )

    _density = pm1_draw_feature_density[_bounded_indices]
    _probability = np.einsum(
        "si,si->s",
        _density,
        _overlap,
        optimize=True,
    )

    _first_mass_moment = np.einsum(
        "si,si,i->s",
        _density,
        _overlap,
        m_grid,
        optimize=True,
    )

    _mu_hat = np.full(_bounded_indices.size, np.nan)
    _valid_probability = (
        np.isfinite(_probability)
        & (_probability > 0.0)
    )
    _mu_hat[_valid_probability] = (
        _first_mass_moment[_valid_probability]
        / _probability[_valid_probability]
    )

    _output["band_probability"][_bounded_indices] = _probability
    _output["mu_hat"][_bounded_indices] = _mu_hat

    # For dips, locate the suppression itself rather than weighting the
    # relatively dense edges.  The background is the log-linear curve
    # joining the adjacent draw-level maxima (the prominence bases), and the
    # centroid is evaluated from the positive deficit inside [m_L, m_R].
    if feature_reference_lookup[_feature_id]["family"] == "suppression":
        _background_left = _output[
            "background_left_mass"
        ][_bounded_indices]
        _background_right = _output[
            "background_right_mass"
        ][_bounded_indices]

        _left_base_index = np.searchsorted(
            m_grid,
            _background_left,
        )
        _right_base_index = np.searchsorted(
            m_grid,
            _background_right,
        )

        _left_base_index = np.clip(
            _left_base_index,
            0,
            m_grid.size - 1,
        )
        _right_base_index = np.clip(
            _right_base_index,
            0,
            m_grid.size - 1,
        )

        _smoothed_density = feature_draw_curves[
            FEATURE_REFERENCE_SMOOTH_LNM
        ]["p"][_bounded_indices]

        _row_index = np.arange(_bounded_indices.size)
        _p_background_left = _smoothed_density[
            _row_index,
            _left_base_index,
        ]
        _p_background_right = _smoothed_density[
            _row_index,
            _right_base_index,
        ]

        _valid_background = (
            np.isfinite(_p_background_left)
            & np.isfinite(_p_background_right)
            & (_p_background_left > 0.0)
            & (_p_background_right > 0.0)
            & (_background_left < _background_right)
        )

        _mu_deficit = np.full(_bounded_indices.size, np.nan)

        if np.any(_valid_background):
            _log_background_left_mass = np.log(
                _background_left[_valid_background]
            )
            _log_background_right_mass = np.log(
                _background_right[_valid_background]
            )

            _background_fraction = (
                (
                    log_m_grid[None, :]
                    - _log_background_left_mass[:, None]
                )
                / (
                    _log_background_right_mass
                    - _log_background_left_mass
                )[:, None]
            )

            _log_background_density = (
                (
                    1.0
                    - _background_fraction
                )
                * np.log(
                    _p_background_left[_valid_background]
                )[:, None]
                + _background_fraction
                * np.log(
                    _p_background_right[_valid_background]
                )[:, None]
            )

            _background_density = np.exp(
                _log_background_density
            )

            _deficit = np.maximum(
                _background_density
                - _smoothed_density[_valid_background],
                0.0,
            )

            _deficit_weight = (
                _deficit
                * _overlap[_valid_background]
            )

            _deficit_probability = np.sum(
                _deficit_weight,
                axis=1,
            )

            _deficit_first_moment = np.einsum(
                "si,i->s",
                _deficit_weight,
                m_grid,
                optimize=True,
            )

            _valid_deficit = (
                np.isfinite(_deficit_probability)
                & (_deficit_probability > 0.0)
            )

            _local_mu_deficit = np.full(
                np.count_nonzero(_valid_background),
                np.nan,
            )
            _local_mu_deficit[_valid_deficit] = (
                _deficit_first_moment[_valid_deficit]
                / _deficit_probability[_valid_deficit]
            )

            _mu_deficit[_valid_background] = _local_mu_deficit

        _output["mu_deficit"][_bounded_indices] = _mu_deficit


# ------------------------------------------------------------
# Posterior summaries.
# ------------------------------------------------------------

def _conditional_quantiles(values, present):
    usable = (
        np.asarray(present, dtype=bool)
        & np.isfinite(values)
    )
    if not np.any(usable):
        return np.asarray([np.nan, np.nan, np.nan])
    return np.quantile(
        np.asarray(values, dtype=float)[usable],
        FEATURE_POSTERIOR_QUANTILES,
    )


feature_posterior_summary_rows = []
feature_posterior_summary_lookup = {}

for _feature_id in feature_posterior_order:
    _output = feature_draw_posteriors[_feature_id]
    _located = _output["located"]
    _bounded = _output["bounded"]
    _realized = _output["realized_type"]
    _number_located = int(np.count_nonzero(_located))
    _number_bounded = int(np.count_nonzero(_bounded))
    _p_located = float(np.mean(_located))
    _p_bounded = float(np.mean(_bounded))
    _p_bounded_given_located = (
        float(_number_bounded / _number_located)
        if _number_located
        else np.nan
    )

    _unconditional = {
        feature_type: float(np.mean(_realized == feature_type))
        for feature_type in ("peak", "dip", "shoulder")
    }

    if _number_located:
        _conditional = {
            feature_type: float(
                np.mean(_realized[_located] == feature_type)
            )
            for feature_type in ("peak", "dip", "shoulder")
        }
    else:
        _conditional = {
            feature_type: np.nan
            for feature_type in ("peak", "dip", "shoulder")
        }

    _quantile_fields = {}

    # A location does not require complete outer boundaries.
    for _field in (
        "center",
        "match_distance_lnm",
    ):
        _q05, _q50, _q95 = _conditional_quantiles(
            _output[_field],
            _located,
        )
        _quantile_fields[f"{_field}_q05"] = float(_q05)
        _quantile_fields[f"{_field}_q50"] = float(_q50)
        _quantile_fields[f"{_field}_q95"] = float(_q95)

    # Extents, integrated summaries, and contrasts require valid boundaries.
    for _field in (
        "left",
        "right",
        "width_mass",
        "width_lnm",
        "width_relative",
        "band_probability",
        "mu_hat",
        "mu_deficit",
    ):
        _q05, _q50, _q95 = _conditional_quantiles(
            _output[_field],
            _bounded,
        )
        _quantile_fields[f"{_field}_q05"] = float(_q05)
        _quantile_fields[f"{_field}_q50"] = float(_q50)
        _quantile_fields[f"{_field}_q95"] = float(_q95)

    _contrast_fields = {}

    for _feature_type in (
        "peak",
        "dip",
        "shoulder",
    ):
        _class_mask = (
            _bounded
            & (_realized == _feature_type)
        )

        _class_count = int(
            np.count_nonzero(
                _class_mask
                & np.isfinite(_output["contrast"])
            )
        )

        _contrast_q05, _contrast_q50, _contrast_q95 = (
            _conditional_quantiles(
                _output["contrast"],
                _class_mask,
            )
        )

        _contrast_fields[
            f"number_{_feature_type}_contrast"
        ] = _class_count

        _contrast_fields[
            f"contrast_{_feature_type}_q05"
        ] = float(_contrast_q05)

        _contrast_fields[
            f"contrast_{_feature_type}_q50"
        ] = float(_contrast_q50)

        _contrast_fields[
            f"contrast_{_feature_type}_q95"
        ] = float(_contrast_q95)

        if _feature_type in ("peak", "dip"):
            (
                _relative_prominence_q05,
                _relative_prominence_q50,
                _relative_prominence_q95,
            ) = _conditional_quantiles(
                _output["relative_prominence"],
                _class_mask,
            )
        else:
            (
                _relative_prominence_q05,
                _relative_prominence_q50,
                _relative_prominence_q95,
            ) = (np.nan, np.nan, np.nan)

        _contrast_fields[
            f"relative_prominence_{_feature_type}_q05"
        ] = float(_relative_prominence_q05)

        _contrast_fields[
            f"relative_prominence_{_feature_type}_q50"
        ] = float(_relative_prominence_q50)

        _contrast_fields[
            f"relative_prominence_{_feature_type}_q95"
        ] = float(_relative_prominence_q95)

    _row = {
        "feature": _feature_id,
        "number_draws": number_feature_draws,
        "number_located": _number_located,
        "number_bounded": _number_bounded,
        "probability_location": _p_located,
        "probability_bounded": _p_bounded,
        "probability_bounded_given_location": _p_bounded_given_located,
        # Backward-compatible aliases for downstream notebook cells.
        "number_present": _number_bounded,
        "probability_present": _p_bounded,
        "probability_absent": 1.0 - _p_bounded,
        "probability_peak": _unconditional["peak"],
        "probability_dip": _unconditional["dip"],
        "probability_shoulder": _unconditional["shoulder"],
        "probability_peak_given_present": _conditional["peak"],
        "probability_dip_given_present": _conditional["dip"],
        "probability_shoulder_given_present": _conditional["shoulder"],
        "probability_peak_given_location": _conditional["peak"],
        "probability_dip_given_location": _conditional["dip"],
        "probability_shoulder_given_location": _conditional["shoulder"],
        **_quantile_fields,
        **_contrast_fields,
    }

    feature_posterior_summary_rows.append(_row)
    feature_posterior_summary_lookup[_feature_id] = _row

feature_posterior_summary = pd.DataFrame(
    feature_posterior_summary_rows
)


# ------------------------------------------------------------
# Clear numerical report for the paper.
# ------------------------------------------------------------

def _format_interval(q05, q50, q95, digits=2):
    if not np.all(np.isfinite([q05, q50, q95])):
        return "---"
    return (
        f"{q50:.{digits}f} "
        f"[{q05:.{digits}f}, {q95:.{digits}f}]"
    )


print("\nPosterior feature locations, bounded extents, classes, and strengths")
print(
    "  analyzed density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL} "
    f"(output tag: {MASS_DENSITY_OUTPUT_TAG})"
)
print("  Location intervals are conditional on location support.")
_scale_interval_label = (
    "mass-scale"
    if COORDINATE_NAME in ("m1", "m2")
    else "location"
)
print(
    "  Edge, width, band, and "
    f"{_scale_interval_label} intervals are conditional on bounded support."
)
print("  Morphology fractions are conditional on location support.")
print(
    "  Strength intervals are conditional on bounded support and the "
    "realized morphology."
)

for _row in feature_posterior_summary_rows:
    _feature_id = _row["feature"]
    print(f"\n{_feature_id}")
    print(
        f"  P_loc = {_row['probability_location']:.4f}; "
        f"P_bnd = {_row['probability_bounded']:.4f}; "
        f"P_bnd|loc = {_row['probability_bounded_given_location']:.4f}"
    )
    print(
        "  morphology | location: "
        f"peak={_row['probability_peak_given_location']:.4f}, "
        f"dip={_row['probability_dip_given_location']:.4f}, "
        f"shoulder={_row['probability_shoulder_given_location']:.4f}"
    )

    for _feature_type, _short_label in (
        ("peak", "peak"),
        ("dip", "dip"),
        ("shoulder", "shoulder"),
    ):
        _class_probability = _row[
            f"probability_{_feature_type}_given_location"
        ]

        _class_count = _row[
            f"number_{_feature_type}_contrast"
        ]

        if (
            not np.isfinite(_class_probability)
            or _class_count == 0
        ):
            continue

        # Internal morphology-specific contrast. This is retained
        # because it is also used by the H0-dependence analysis.
        print(
            f"  Delta_{_short_label} | bounded,{_short_label:8s}: "
            + _format_interval(
                _row[f"contrast_{_feature_type}_q05"],
                _row[f"contrast_{_feature_type}_q50"],
                _row[f"contrast_{_feature_type}_q95"],
                digits=3,
            )
            + f"  (N={_class_count})"
        )

        # Paper-facing bounded relative prominence for peaks and dips.
        # Shoulders instead retain Delta_S because their contrast
        # measures a change in logarithmic slope.
        if _feature_type in ("peak", "dip"):
            print(
                f"  A_{_short_label} | bounded,{_short_label:8s}: "
                + _format_interval(
                    _row[
                        f"relative_prominence_{_feature_type}_q05"
                    ],
                    _row[
                        f"relative_prominence_{_feature_type}_q50"
                    ],
                    _row[
                        f"relative_prominence_{_feature_type}_q95"
                    ],
                    digits=3,
                )
            )

            
    _coordinate_bracket_unit = (
        f" [{COORDINATE_PLAIN_UNIT}]" if COORDINATE_PLAIN_UNIT else ""
    )
    _relative_width_label = (
        "relative width (mR-mL)/mC"
        if COORDINATE_NAME in ("m1", "m2")
        else "relative width"
    )
    _moving_location_label = (
        "moving-band mu_hat"
        if COORDINATE_NAME in ("m1", "m2")
        else "moving-band location"
    )
    for _field, _label, _digits in (
        ("center", f"location{_coordinate_bracket_unit}", 2),
        ("left", f"left edge{_coordinate_bracket_unit}", 2),
        ("right", f"right edge{_coordinate_bracket_unit}", 2),
        ("width_mass", f"width{_coordinate_bracket_unit}", 2),
        ("width_lnm", f"width in ln({COORDINATE_LOG_ARGUMENT})", 3),
        ("width_relative", _relative_width_label, 3),
        ("mu_hat", f"{_moving_location_label}{_coordinate_bracket_unit}", 2),
        (
            "mu_deficit",
            f"dip deficit centroid{_coordinate_bracket_unit}",
            2,
        ),
        (
            "match_distance_lnm",
            f"match distance in ln({COORDINATE_LOG_ARGUMENT})",
            3,
        ),
    ):
        print(
            f"  {_label:28s}: "
            + _format_interval(
                _row[f"{_field}_q05"],
                _row[f"{_field}_q50"],
                _row[f"{_field}_q95"],
                digits=_digits,
            )
        )

print("\nLocation-construction failures before matching:")
for _feature_type, _count in feature_location_failure_count.items():
    print(f"  {_feature_type:10s}: {_count}")

print("Extent-construction failures after location matching:")
for _feature_type, _count in feature_extent_failure_count.items():
    print(f"  {_feature_type:10s}: {_count}")

print("Unmatched persistent draw-level candidates:")
for _family, _count in feature_unmatched_candidate_count.items():
    print(f"  {_family:12s}: {_count}")

print("\nCompact summary")
display(
    feature_posterior_summary[
        [
            "feature",
            "probability_location",
            "probability_bounded",
            "probability_bounded_given_location",
            "probability_peak_given_location",
            "probability_dip_given_location",
            "probability_shoulder_given_location",
            "center_q05",
            "center_q50",
            "center_q95",
            "left_q50",
            "right_q50",
            "width_mass_q50",
            "width_lnm_q50",
            "width_relative_q50",
            "mu_hat_q50",
            "mu_deficit_q50",
            "contrast_peak_q50",
            "contrast_dip_q50",
            "contrast_shoulder_q50",
            "relative_prominence_peak_q50",
            "relative_prominence_dip_q50",
        ]
    ]
)


# ------------------------------------------------------------
# Save the machine-readable and paper-ready tables.
# ------------------------------------------------------------

feature_posterior_summary.to_csv(
    FEATURE_POSTERIOR_CSV,
    index=False,
)


def _tex_mass_interval(row, field, digits=2):
    q05 = row[f"{field}_q05"]
    q50 = row[f"{field}_q50"]
    q95 = row[f"{field}_q95"]
    if not np.all(np.isfinite([q05, q50, q95])):
        return r"---"
    return (
        rf"${q50:.{digits}f}"
        rf"^{{+{q95 - q50:.{digits}f}}}"
        rf"_{{-{q50 - q05:.{digits}f}}}$"
    )


def _tex_strength_cell(row):
    """
    Class-conditional feature strength.

    Peaks and dips are reported through the bounded relative
    prominence A. Shoulders retain the logarithmic-slope contrast
    Delta_S.
    """

    entries = []

    for feature_type, type_label in (
        ("peak", "P"),
        ("dip", "D"),
        ("shoulder", "S"),
    ):
        probability = row[
            f"probability_{feature_type}_given_location"
        ]

        # Suppress morphologies whose probability rounds to zero
        # in the adjacent table column.
        if (
            not np.isfinite(probability)
            or round(probability, 2) == 0.0
        ):
            continue

        if feature_type in ("peak", "dip"):
            q05 = row[
                f"relative_prominence_{feature_type}_q05"
            ]
            q50 = row[
                f"relative_prominence_{feature_type}_q50"
            ]
            q95 = row[
                f"relative_prominence_{feature_type}_q95"
            ]

            if not np.all(np.isfinite([q05, q50, q95])):
                continue

            entries.append(
                rf"$A_{{\rm {type_label}}}="
                rf"{q50:.2f}^{{+{q95 - q50:.2f}}}"
                rf"_{{-{q50 - q05:.2f}}}$"
            )

        else:
            q05 = row["contrast_shoulder_q05"]
            q50 = row["contrast_shoulder_q50"]
            q95 = row["contrast_shoulder_q95"]

            if not np.all(np.isfinite([q05, q50, q95])):
                continue

            entries.append(
                rf"$\Delta_{{\rm S}}="
                rf"{q50:.2f}^{{+{q95 - q50:.2f}}}"
                rf"_{{-{q50 - q05:.2f}}}$"
            )

    return r"; ".join(entries) if entries else r"---"

_tex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Posterior characterization of the automatically identified "
    rf"one-dimensional primary-mass features in {MASS_DENSITY_TEX_LABEL}.  "
    r"$P_{\rm loc}$ is the "
    r"posterior probability that a compatible feature location is found, "
    r"whereas $P_{\rm bnd}$ additionally requires complete left and right "
    r"boundaries.  The morphology column gives the peak (P), dip (D), and "
    r"shoulder (S) probabilities conditional on location support.  For "
    r"peaks and dips, the strength column reports the bounded relative "
    r"prominence $A_{\rm P}=1-p_{\rm sad}/p_C$ or "
    r"$A_{\rm D}=1-p_C/p_{\rm sad}$, conditional on bounded support and "
    r"the stated morphology.  Here $p_{\rm sad}$ is the draw-dependent "
    r"topological saddle density.  Shoulder strength is instead reported "
    r"through the logarithmic-slope change $\Delta_{\rm S}$.  Locations "
    r"are conditional on $P_{\rm loc}$; edges and relative widths are "
    r"conditional on $P_{\rm bnd}$.  Peaks and dips use half-prominence "
    r"edges, whereas shoulder realizations use the preceding $K_+$ and "
    r"following $\kappa=0$ boundaries.  The relative width is "
    r"$w_f=(m_R-m_L)/m_C$.  The final row reports the global upper-tail "
    r"scale.}",
    r"\label{tab:primary_mass_features}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2.8pt}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Feature & $P_{\rm loc}/P_{\rm bnd}$ & Morphology $\mid$ loc. "
    r"& Strength $\mid$ class & $m_C\,[M_\odot]$ "
    r"& $m_L,m_R\,[M_\odot]$ & $w_f$ \\",
    r"\midrule",
]

for _row in feature_posterior_summary_rows:
    _feature_id = _row["feature"]
    _feature_tex = rf"${_feature_id[0]}_{{{_feature_id[1:]}}}$"

    if feature_reference_lookup[_feature_id]["family"] == "suppression":
        _morphology_tex = (
            rf"D: {_row['probability_dip_given_location']:.2f}"
        )
    else:
        _morphology_tex = (
            rf"P: {_row['probability_peak_given_location']:.2f}; "
            rf"S: {_row['probability_shoulder_given_location']:.2f}"
        )

    _edge_cell = (
        _tex_mass_interval(_row, "left")
        + r",\," 
        + _tex_mass_interval(_row, "right")
    )

    _tex_lines.append(
        " & ".join(
            [
                _feature_tex,
                (
                    f"${_row['probability_location']:.3f}"
                    f"/{_row['probability_bounded']:.3f}$"
                ),
                _morphology_tex,
                _tex_strength_cell(_row),
                _tex_mass_interval(_row, "center"),
                _edge_cell,
                _tex_mass_interval(_row, "width_relative", digits=2),
            ]
        )
        + r" \\"
    )

_tex_lines.extend(
    [
        r"\midrule",
        (
            rf"$m_{{{percentile_label}}}$ & --- & upper-tail scale & --- & "
            + _tex_mass_interval(
                {
                    "tail_q05": m_high_q05,
                    "tail_q50": m_high_q50,
                    "tail_q95": m_high_q95,
                },
                "tail",
                digits=1,
            )
            + r" & --- & --- \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
)

Path(FEATURE_POSTERIOR_TEX).write_text(
    "\n".join(_tex_lines) + "\n",
    encoding="utf-8",
)

print("\nSaved feature posterior tables:")
print(f"  {FEATURE_POSTERIOR_CSV}")
print(f"  {FEATURE_POSTERIOR_TEX}")
