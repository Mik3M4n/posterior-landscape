# ============================================================
# Diagnose feature absence and adjacent-feature dependence.
#
# This cell separates:
#   1. a persistent compatible morphology being found;
#   2. successful construction of m_L, m_C, and m_R.
#
# It uses exactly the smoothing, persistence, feature-family, and
# identity-interval rules of the draw-level feature workflow.
# Run it after the posterior feature-inference cell.
# ============================================================

import warnings
import numpy as np
import pandas as pd

from tqdm.auto import tqdm


# ------------------------------------------------------------
# Configuration and validation.
# ------------------------------------------------------------

# A candidate is called "shifted" only if it lies immediately outside a
# feature's identity interval, within the existing cross-scale matching
# tolerance.  This is a diagnostic category, not part of feature matching.
FEATURE_SHIFT_MARGIN_LNM = FEATURE_MATCH_TOLERANCE_LNM

FEATURE_DEPENDENCE_PAIRS = (
    ("D1", "P2"),
    ("D2", "P3"),
    ("P3", "S1"),
    ("S1", "S2"),
)

_required_diagnostic_products = (
    "feature_draw_curves",
    "feature_draw_posteriors",
    "feature_reference_lookup",
    "feature_posterior_order",
    "positive_reference_ids",
    "suppression_reference_ids",
    "positive_identity_intervals_lnm",
    "suppression_identity_intervals_lnm",
    "feature_identity_intervals_lnm",
    "number_feature_draws",
    "FEATURE_SMOOTH_SCALES_LNM",
    "FEATURE_REFERENCE_SMOOTH_LNM",
    "FEATURE_MATCH_TOLERANCE_LNM",
    "FEATURE_PEAK_SHOULDER_DEDUP_LNM",
    "_safe_local_maxima",
    "_persistent_draw_records",
    "_draw_peak_or_dip_band",
    "_draw_shoulder_transition_band",
    "MASS_DENSITY_PLAIN_LABEL",
    "MASS_DENSITY_OUTPUT_TAG",
)

_missing_diagnostic_products = [
    name
    for name in _required_diagnostic_products
    if name not in globals()
]

if _missing_diagnostic_products:
    raise RuntimeError(
        "Run the draw-level feature-inference cell first. Missing: "
        + ", ".join(_missing_diagnostic_products)
    )

for _feature_id_a, _feature_id_b in FEATURE_DEPENDENCE_PAIRS:
    for _feature_id in (_feature_id_a, _feature_id_b):
        if _feature_id not in feature_reference_lookup:
            raise RuntimeError(
                f"Unknown feature in FEATURE_DEPENDENCE_PAIRS: {_feature_id}"
            )


# ------------------------------------------------------------
# Helpers.
# ------------------------------------------------------------

def _diagnostic_assign_raw_candidates(
    reference_ids,
    candidates,
    identity_intervals_lnm,
):
    """Assign at most one pre-extent candidate to each reference feature."""

    proposals = {
        feature_id: []
        for feature_id in reference_ids
    }

    if not candidates:
        return {}

    reference_log_locations = np.log(
        np.asarray(
            [
                feature_reference_lookup[feature_id]["location"]
                for feature_id in reference_ids
            ],
            dtype=float,
        )
    )

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
        interval_left, interval_right = (
            identity_intervals_lnm[feature_id]
        )

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

    assigned = {}

    for feature_id, feature_proposals in proposals.items():
        if not feature_proposals:
            continue

        candidate, distance = min(
            feature_proposals,
            key=lambda proposal: (
                proposal[1],
                -float(proposal[0]["prominence"]),
            ),
        )

        assigned[feature_id] = {
            "candidate": candidate,
            "distance_lnm": distance,
        }

    return assigned


def _diagnostic_deduplicate_raw_positive(candidates):
    """Apply the peak-over-shoulder hierarchy before edge construction."""

    peak_centres = np.asarray(
        [
            candidate["center"]
            for candidate in candidates
            if candidate["detected_type"] == "peak"
        ],
        dtype=float,
    )

    if peak_centres.size == 0:
        return candidates

    retained = []

    for candidate in candidates:
        if candidate["detected_type"] == "peak":
            retained.append(candidate)
            continue

        nearest_peak_distance = float(
            np.min(
                np.abs(
                    np.log(peak_centres)
                    - np.log(candidate["center"])
                )
            )
        )

        if nearest_peak_distance > FEATURE_PEAK_SHOULDER_DEDUP_LNM:
            retained.append(candidate)

    return retained


def _diagnostic_shifted_candidate(
    feature_id,
    family_candidates,
    family_reference_ids,
):
    """Find a same-family candidate immediately outside an identity range."""

    if not family_candidates:
        return None

    interval_left, interval_right = (
        feature_identity_intervals_lnm[feature_id]
    )

    candidate_log_locations = np.asarray(
        [
            np.log(candidate["center"])
            for candidate in family_candidates
        ],
        dtype=float,
    )

    distances_to_interval = np.maximum(
        np.maximum(
            interval_left - candidate_log_locations,
            candidate_log_locations - interval_right,
        ),
        0.0,
    )

    outside = distances_to_interval > 0.0

    if not np.any(outside):
        return None

    outside_indices = np.flatnonzero(outside)
    nearest_outside_index = int(
        outside_indices[
            np.argmin(
                distances_to_interval[outside_indices]
            )
        ]
    )

    boundary_distance = float(
        distances_to_interval[nearest_outside_index]
    )

    if boundary_distance > FEATURE_SHIFT_MARGIN_LNM:
        return None

    candidate = family_candidates[nearest_outside_index]
    candidate_log_location = float(
        candidate_log_locations[nearest_outside_index]
    )

    reference_log_locations = np.log(
        np.asarray(
            [
                feature_reference_lookup[reference_id]["location"]
                for reference_id in family_reference_ids
            ],
            dtype=float,
        )
    )

    nearest_reference_index = int(
        np.argmin(
            np.abs(
                reference_log_locations
                - candidate_log_location
            )
        )
    )

    return {
        "candidate": candidate,
        "assigned_neighbor": family_reference_ids[
            nearest_reference_index
        ],
        "distance_from_identity_boundary_lnm": boundary_distance,
    }


def _safe_conditional_probability(numerator_mask, denominator_mask):
    denominator_count = int(
        np.count_nonzero(denominator_mask)
    )

    if denominator_count == 0:
        return np.nan

    return float(
        np.count_nonzero(
            numerator_mask
            & denominator_mask
        )
        / denominator_count
    )


# ------------------------------------------------------------
# Output arrays.
# ------------------------------------------------------------

feature_morphology_present = {
    feature_id: np.zeros(number_feature_draws, dtype=bool)
    for feature_id in feature_posterior_order
}

feature_raw_morphology = {
    feature_id: np.full(
        number_feature_draws,
        "absent",
        dtype=object,
    )
    for feature_id in feature_posterior_order
}

feature_pre_extent_success = {
    feature_id: np.zeros(number_feature_draws, dtype=bool)
    for feature_id in feature_posterior_order
}

feature_absence_status = {
    feature_id: np.full(
        number_feature_draws,
        "not_found",
        dtype=object,
    )
    for feature_id in feature_posterior_order
}

feature_shifted_to = {
    feature_id: np.full(
        number_feature_draws,
        "",
        dtype=object,
    )
    for feature_id in feature_posterior_order
}


# ------------------------------------------------------------
# Rerun detection while retaining candidates before edge construction.
# ------------------------------------------------------------

for _draw_index in tqdm(
    range(number_feature_draws),
    desc="Feature presence/extent diagnostic",
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

        _shoulder_keep = _alpha[_shoulder_all] < 0.0
        _shoulders = _shoulder_all[_shoulder_keep]
        _shoulder_prominence = _shoulder_prominence_all[
            _shoulder_keep
        ]

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

    _reference_curves = feature_draw_curves[
        FEATURE_REFERENCE_SMOOTH_LNM
    ]

    _p_reference = _reference_curves["p"][_draw_index]
    _alpha_reference = _reference_curves["alpha"][_draw_index]
    _kappa_reference = _reference_curves["kappa"][_draw_index]
    _valid_reference = _reference_curves["valid"][_draw_index]

    _raw_positive_candidates = []
    _raw_suppression_candidates = []

    for _record in _persistent["peak"]:
        _extent_success = True

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _draw_peak_or_dip_band(
                    _p_reference,
                    _valid_reference,
                    _record["index"],
                    "peak",
                )
        except (RuntimeError, ValueError, FloatingPointError):
            _extent_success = False

        _raw_positive_candidates.append(
            {
                "center": float(m_grid[_record["index"]]),
                "prominence": float(_record["prominence"]),
                "detected_type": "peak",
                "extent_success": _extent_success,
            }
        )

    for _record in _persistent["dip"]:
        _extent_success = True

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _draw_peak_or_dip_band(
                    _p_reference,
                    _valid_reference,
                    _record["index"],
                    "dip",
                )
        except (RuntimeError, ValueError, FloatingPointError):
            _extent_success = False

        _raw_suppression_candidates.append(
            {
                "center": float(m_grid[_record["index"]]),
                "prominence": float(_record["prominence"]),
                "detected_type": "dip",
                "extent_success": _extent_success,
            }
        )

    _persistent_down_indices = np.sort(
        np.asarray(
            [
                record["index"]
                for record in _persistent["down_knee"]
            ],
            dtype=int,
        )
    )

    for _record in _persistent["shoulder"]:
        _onset_index = int(_record["index"])
        _following_down = _persistent_down_indices[
            _persistent_down_indices > _onset_index
        ]

        # Use the maximum-steepening point when it exists; otherwise the
        # shoulder onset is still sufficient to identify the mass region.
        _proxy_index = int(
            _following_down[0]
            if _following_down.size
            else _onset_index
        )

        _extent_success = True

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _draw_shoulder_transition_band(
                    onset_index=_onset_index,
                    alpha=_alpha_reference,
                    kappa=_kappa_reference,
                    valid_mask=_valid_reference,
                    persistent_up_knees=_persistent["up_knee"],
                    persistent_down_knees=_persistent["down_knee"],
                )
        except (RuntimeError, ValueError, FloatingPointError):
            _extent_success = False

        _raw_positive_candidates.append(
            {
                "center": float(m_grid[_proxy_index]),
                "onset": float(m_grid[_onset_index]),
                "prominence": float(_record["prominence"]),
                "detected_type": "shoulder",
                "extent_success": _extent_success,
            }
        )

    _raw_positive_candidates = (
        _diagnostic_deduplicate_raw_positive(
            _raw_positive_candidates
        )
    )

    _raw_positive_matches = _diagnostic_assign_raw_candidates(
        positive_reference_ids,
        _raw_positive_candidates,
        positive_identity_intervals_lnm,
    )

    _raw_suppression_matches = _diagnostic_assign_raw_candidates(
        suppression_reference_ids,
        _raw_suppression_candidates,
        suppression_identity_intervals_lnm,
    )

    _raw_matches = {
        **_raw_positive_matches,
        **_raw_suppression_matches,
    }

    for _feature_id, _match in _raw_matches.items():
        _candidate = _match["candidate"]
        feature_morphology_present[_feature_id][_draw_index] = True
        feature_raw_morphology[_feature_id][_draw_index] = (
            _candidate["detected_type"]
        )
        feature_pre_extent_success[_feature_id][_draw_index] = bool(
            _candidate["extent_success"]
        )

    # A final characterized match is definitive even if the raw proxy
    # location used above happened to select a different candidate.
    for _feature_id in feature_posterior_order:
        _characterized = bool(
            feature_draw_posteriors[_feature_id]["present"][
                _draw_index
            ]
        )

        if _characterized:
            feature_morphology_present[_feature_id][_draw_index] = True
            feature_raw_morphology[_feature_id][_draw_index] = (
                feature_draw_posteriors[_feature_id]["realized_type"][
                    _draw_index
                ]
            )
            feature_pre_extent_success[_feature_id][_draw_index] = True
            feature_absence_status[_feature_id][_draw_index] = (
                "characterized"
            )
            continue

        if feature_morphology_present[_feature_id][_draw_index]:
            if feature_pre_extent_success[_feature_id][_draw_index]:
                feature_absence_status[_feature_id][_draw_index] = (
                    "unselected_after_extent"
                )
            else:
                feature_absence_status[_feature_id][_draw_index] = (
                    "extent_failed"
                )
            continue

        _family = feature_reference_lookup[_feature_id]["family"]

        if _family == "positive":
            _family_candidates = _raw_positive_candidates
            _family_reference_ids = positive_reference_ids
        else:
            _family_candidates = _raw_suppression_candidates
            _family_reference_ids = suppression_reference_ids

        _shifted = _diagnostic_shifted_candidate(
            _feature_id,
            _family_candidates,
            _family_reference_ids,
        )

        if _shifted is not None:
            feature_absence_status[_feature_id][_draw_index] = "shifted"
            feature_shifted_to[_feature_id][_draw_index] = (
                _shifted["assigned_neighbor"]
            )


# ------------------------------------------------------------
# Feature-level absence decomposition.
# ------------------------------------------------------------

_status_order = (
    "characterized",
    "extent_failed",
    "unselected_after_extent",
    "shifted",
    "not_found",
)

feature_presence_status_rows = []

for _feature_id in feature_posterior_order:
    _status = feature_absence_status[_feature_id]
    _morphology_present = feature_morphology_present[_feature_id]
    _characterized = np.asarray(
        feature_draw_posteriors[_feature_id]["present"],
        dtype=bool,
    )

    _p_morphology = float(
        np.mean(_morphology_present)
    )

    _p_characterized = float(
        np.mean(_characterized)
    )

    _p_characterized_given_morphology = (
        _p_characterized / _p_morphology
        if _p_morphology > 0.0
        else np.nan
    )

    _row = {
        "feature": _feature_id,
        "P_morphology": _p_morphology,
        "P_characterized": _p_characterized,
        "P_characterized_given_morphology": (
            _p_characterized_given_morphology
        ),
    }

    for _status_name in _status_order:
        _row[f"P_{_status_name}"] = float(
            np.mean(_status == _status_name)
        )

    feature_presence_status_rows.append(_row)

feature_presence_status_diagnostic = pd.DataFrame(
    feature_presence_status_rows
)

print(
    "\nFeature morphology and extent-construction diagnostic"
)

print(
    "  analyzed density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL} "
    f"(output tag: {MASS_DENSITY_OUTPUT_TAG})"
)

print(
    "  P_morphology: a persistent compatible candidate is found "
    "before edge construction."
)

print(
    "  P_characterized: the candidate is finally matched with valid "
    "m_L, m_C, and m_R.\n"
)

with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    220,
    "display.precision",
    4,
):
    print(
        feature_presence_status_diagnostic.to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Detailed S1 absence decomposition.
# ------------------------------------------------------------

if "S1" in feature_reference_lookup:
    _s1_status = feature_absence_status["S1"]
    _s1_shifted_to = feature_shifted_to["S1"]

    print(
        "\nS1 status counts:"
    )

    for _status_name in _status_order:
        _count = int(
            np.count_nonzero(
                _s1_status == _status_name
            )
        )
        print(
            f"  {_status_name:24s}: "
            f"{_count:5d} "
            f"({_count / number_feature_draws:.4f})"
        )

    _s1_shifted_mask = _s1_status == "shifted"

    if np.any(_s1_shifted_mask):
        _neighbors, _neighbor_counts = np.unique(
            _s1_shifted_to[_s1_shifted_mask],
            return_counts=True,
        )

        print(
            "  shifted candidates assigned nearest to:"
        )

        for _neighbor, _count in zip(
            _neighbors,
            _neighbor_counts,
        ):
            print(
                f"    {_neighbor:>3s}: {_count:d}"
            )


# ------------------------------------------------------------
# Joint occurrence of neighboring features.
# ------------------------------------------------------------

feature_joint_occurrence_rows = []

for _feature_a, _feature_b in FEATURE_DEPENDENCE_PAIRS:
    _a = feature_morphology_present[_feature_a]
    _b = feature_morphology_present[_feature_b]

    _p_a = float(np.mean(_a))
    _p_b = float(np.mean(_b))
    _p_both = float(np.mean(_a & _b))
    _p_a_only = float(np.mean(_a & (~_b)))
    _p_b_only = float(np.mean((~_a) & _b))
    _p_neither = float(np.mean((~_a) & (~_b)))

    _p_a_given_b = _safe_conditional_probability(
        _a,
        _b,
    )

    _p_b_given_a = _safe_conditional_probability(
        _b,
        _a,
    )

    _excess_cooccurrence = (
        _p_both
        - _p_a * _p_b
    )

    _variance_product = (
        _p_a
        * (1.0 - _p_a)
        * _p_b
        * (1.0 - _p_b)
    )

    _phi = (
        _excess_cooccurrence
        / np.sqrt(_variance_product)
        if _variance_product > 0.0
        else np.nan
    )

    feature_joint_occurrence_rows.append(
        {
            "pair": f"{_feature_a}-{_feature_b}",
            "P_A": _p_a,
            "P_B": _p_b,
            "P_both": _p_both,
            "P_A_only": _p_a_only,
            "P_B_only": _p_b_only,
            "P_neither": _p_neither,
            "P_A_given_B": _p_a_given_b,
            "P_B_given_A": _p_b_given_a,
            "excess_cooccurrence": _excess_cooccurrence,
            "phi_binary": _phi,
        }
    )

feature_joint_occurrence_diagnostic = pd.DataFrame(
    feature_joint_occurrence_rows
)

print(
    "\nJoint occurrence of neighboring feature morphologies"
)

print(
    "  excess_cooccurrence = P(A and B) - P(A)P(B)."
)

print(
    "  Positive values indicate preferential coexistence; negative "
    "values indicate substitution.\n"
)

with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    220,
    "display.precision",
    4,
):
    print(
        feature_joint_occurrence_diagnostic.to_string(
            index=False
        )
    )


# These objects remain available for subsequent interpretation.
#   feature_morphology_present
#   feature_absence_status
#   feature_shifted_to
#   feature_presence_status_diagnostic
#   feature_joint_occurrence_diagnostic


