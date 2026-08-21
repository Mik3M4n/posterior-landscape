# ============================================================
# Shoulder morphology and extent-failure diagnostic.
#
# Run after feature_presence_and_cooccurrence_diagnostic_cell.py.
# This cell answers:
#   1. What are the peak/shoulder fractions before edge construction?
#   2. Is successful characterization morphology-dependent?
#   3. Which exact shoulder-boundary step fails for S1 and S2?
# ============================================================

import warnings
import numpy as np
import pandas as pd

from tqdm.auto import tqdm


# ------------------------------------------------------------
# Validate products from the preceding cells.
# ------------------------------------------------------------

_required_shoulder_diagnostic_products = (
    "feature_morphology_present",
    "feature_raw_morphology",
    "feature_draw_posteriors",
    "feature_draw_curves",
    "feature_reference_lookup",
    "feature_posterior_order",
    "positive_reference_ids",
    "positive_identity_intervals_lnm",
    "number_feature_draws",
    "FEATURE_SMOOTH_SCALES_LNM",
    "FEATURE_REFERENCE_SMOOTH_LNM",
    "FEATURE_PEAK_SHOULDER_DEDUP_LNM",
    "FEATURE_BAND_REL_HEIGHT",
    "_safe_local_maxima",
    "_persistent_draw_records",
    "_diagnostic_deduplicate_raw_positive",
    "_diagnostic_assign_raw_candidates",
    "_first_negative_to_positive_kappa_crossing",
    "_prominence_band",
    "MASS_DENSITY_PLAIN_LABEL",
    "MASS_DENSITY_OUTPUT_TAG",
)

_missing_shoulder_diagnostic_products = [
    name
    for name in _required_shoulder_diagnostic_products
    if name not in globals()
]

if _missing_shoulder_diagnostic_products:
    raise RuntimeError(
        "Run the draw-level feature workflow and the feature-presence "
        "diagnostic first. Missing: "
        + ", ".join(_missing_shoulder_diagnostic_products)
    )


# ------------------------------------------------------------
# Peak/shoulder fractions before and after extent construction.
# ------------------------------------------------------------

feature_raw_morphology_rows = []

for _feature_id in feature_posterior_order:
    if feature_reference_lookup[_feature_id]["family"] != "positive":
        continue

    _morphology_present = np.asarray(
        feature_morphology_present[_feature_id],
        dtype=bool,
    )

    _raw_type = np.asarray(
        feature_raw_morphology[_feature_id],
        dtype=object,
    )

    _characterized = np.asarray(
        feature_draw_posteriors[_feature_id]["present"],
        dtype=bool,
    )

    _characterized_type = np.asarray(
        feature_draw_posteriors[_feature_id]["realized_type"],
        dtype=object,
    )

    _number_morphology = int(
        np.count_nonzero(_morphology_present)
    )

    _row = {
        "feature": _feature_id,
        "number_draws": number_feature_draws,
        "number_morphology": _number_morphology,
        "P_morphology": float(np.mean(_morphology_present)),
        "P_characterized": float(np.mean(_characterized)),
    }

    for _feature_type in ("peak", "shoulder"):
        _raw_type_mask = (
            _morphology_present
            & (_raw_type == _feature_type)
        )

        _characterized_type_mask = (
            _characterized
            & (_characterized_type == _feature_type)
        )

        _number_raw_type = int(
            np.count_nonzero(_raw_type_mask)
        )

        _row[
            f"P_{_feature_type}_given_morphology"
        ] = (
            _number_raw_type / _number_morphology
            if _number_morphology
            else np.nan
        )

        _row[
            f"P_characterized_given_raw_{_feature_type}"
        ] = (
            np.count_nonzero(
                _characterized_type_mask
                & _raw_type_mask
            )
            / _number_raw_type
            if _number_raw_type
            else np.nan
        )

        _row[f"number_raw_{_feature_type}"] = _number_raw_type

    feature_raw_morphology_rows.append(_row)

feature_raw_morphology_diagnostic = pd.DataFrame(
    feature_raw_morphology_rows
)

print(
    "\nPositive-feature morphology before extent construction"
)

print(
    "  analyzed density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL} "
    f"(output tag: {MASS_DENSITY_OUTPUT_TAG})"
)

print(
    "  Morphology fractions are conditional on a persistent compatible "
    "candidate being found."
)

print(
    "  Characterization rates give the probability of obtaining valid "
    "edges conditional on that raw morphology.\n"
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
        feature_raw_morphology_diagnostic.to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Instrument the shoulder transition construction.
# ------------------------------------------------------------

def _diagnose_shoulder_transition_extent(
    onset_index,
    alpha,
    kappa,
    valid_mask,
    persistent_up_knees,
    persistent_down_knees,
):
    """Mirror the production definition while returning a failure reason."""

    up_indices = np.sort(
        np.asarray(
            [
                record["index"]
                for record in persistent_up_knees
            ],
            dtype=int,
        )
    )

    down_indices = np.sort(
        np.asarray(
            [
                record["index"]
                for record in persistent_down_knees
            ],
            dtype=int,
        )
    )

    preceding_up = up_indices[up_indices < onset_index]
    following_down = down_indices[down_indices > onset_index]

    if preceding_up.size == 0:
        return {
            "success": False,
            "reason": "missing_preceding_Kplus",
            "left_mass": np.nan,
            "center_mass": (
                float(m_grid[following_down[0]])
                if following_down.size
                else float(m_grid[onset_index])
            ),
            "right_mass": np.nan,
        }

    if following_down.size == 0:
        return {
            "success": False,
            "reason": "missing_following_Kminus",
            "left_mass": float(m_grid[preceding_up[-1]]),
            "center_mass": float(m_grid[onset_index]),
            "right_mass": np.nan,
        }

    up_index = int(preceding_up[-1])
    down_index = int(following_down[0])

    if not up_index < onset_index < down_index:
        return {
            "success": False,
            "reason": "invalid_landmark_order",
            "left_mass": float(m_grid[up_index]),
            "center_mass": float(m_grid[down_index]),
            "right_mass": np.nan,
        }

    try:
        right_crossing = _first_negative_to_positive_kappa_crossing(
            kappa,
            start_index=down_index,
            valid_mask=(
                np.asarray(valid_mask, dtype=bool)
                & np.isfinite(kappa)
            ),
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return {
            "success": False,
            "reason": "missing_following_kappa_zero",
            "left_mass": float(m_grid[up_index]),
            "center_mass": float(m_grid[down_index]),
            "right_mass": np.nan,
        }

    try:
        _prominence_band(
            values=alpha,
            center_index=onset_index,
            valid_mask=(
                np.asarray(valid_mask, dtype=bool)
                & np.isfinite(alpha)
            ),
            invert=False,
            rel_height=FEATURE_BAND_REL_HEIGHT,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return {
            "success": False,
            "reason": "alpha_prominence_failure",
            "left_mass": float(m_grid[up_index]),
            "center_mass": float(m_grid[down_index]),
            "right_mass": float(right_crossing["mass"]),
        }

    left_mass = float(m_grid[up_index])
    center_mass = float(m_grid[down_index])
    right_mass = float(right_crossing["mass"])

    if not left_mass < center_mass < right_mass:
        return {
            "success": False,
            "reason": "invalid_extent_order",
            "left_mass": left_mass,
            "center_mass": center_mass,
            "right_mass": right_mass,
        }

    return {
        "success": True,
        "reason": "success",
        "left_mass": left_mass,
        "center_mass": center_mass,
        "right_mass": right_mass,
    }


shoulder_extent_records = []

for _draw_index in tqdm(
    range(number_feature_draws),
    desc="Shoulder failure reasons",
    unit="draw",
):
    _indices_by_type = {
        feature_type: {}
        for feature_type in (
            "peak",
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

    _alpha_reference = _reference_curves["alpha"][_draw_index]
    _kappa_reference = _reference_curves["kappa"][_draw_index]
    _valid_reference = _reference_curves["valid"][_draw_index]

    _persistent_down_indices = np.sort(
        np.asarray(
            [
                record["index"]
                for record in _persistent["down_knee"]
            ],
            dtype=int,
        )
    )

    _raw_positive_candidates = [
        {
            "center": float(m_grid[record["index"]]),
            "prominence": float(record["prominence"]),
            "detected_type": "peak",
            "extent_success": True,
            "extent_result": None,
        }
        for record in _persistent["peak"]
    ]

    for _record in _persistent["shoulder"]:
        _onset_index = int(_record["index"])
        _following_down = _persistent_down_indices[
            _persistent_down_indices > _onset_index
        ]

        _proxy_index = int(
            _following_down[0]
            if _following_down.size
            else _onset_index
        )

        _extent_result = _diagnose_shoulder_transition_extent(
            onset_index=_onset_index,
            alpha=_alpha_reference,
            kappa=_kappa_reference,
            valid_mask=_valid_reference,
            persistent_up_knees=_persistent["up_knee"],
            persistent_down_knees=_persistent["down_knee"],
        )

        _raw_positive_candidates.append(
            {
                "center": float(m_grid[_proxy_index]),
                "onset": float(m_grid[_onset_index]),
                "prominence": float(_record["prominence"]),
                "detected_type": "shoulder",
                "extent_success": bool(_extent_result["success"]),
                "extent_result": _extent_result,
            }
        )

    _raw_positive_candidates = (
        _diagnostic_deduplicate_raw_positive(
            _raw_positive_candidates
        )
    )

    _assigned = _diagnostic_assign_raw_candidates(
        positive_reference_ids,
        _raw_positive_candidates,
        positive_identity_intervals_lnm,
    )

    for _feature_id in ("S1", "S2"):
        if _feature_id not in _assigned:
            continue

        _candidate = _assigned[_feature_id]["candidate"]

        if _candidate["detected_type"] != "shoulder":
            continue

        _extent_result = _candidate["extent_result"]

        _valid_indices = np.flatnonzero(_valid_reference)
        _max_valid_mass = (
            float(m_grid[_valid_indices[-1]])
            if _valid_indices.size
            else np.nan
        )

        shoulder_extent_records.append(
            {
                "draw": _draw_index,
                "feature": _feature_id,
                "success": bool(_extent_result["success"]),
                "reason": _extent_result["reason"],
                "onset_mass": float(_candidate["onset"]),
                "center_mass": float(_candidate["center"]),
                "left_mass": float(_extent_result["left_mass"]),
                "right_mass": float(_extent_result["right_mass"]),
                "max_valid_mass": _max_valid_mass,
            }
        )

shoulder_extent_failure_records = pd.DataFrame(
    shoulder_extent_records
)

if shoulder_extent_failure_records.empty:
    print(
        "\nNo S1 or S2 shoulder candidates were assigned."
    )
else:
    shoulder_extent_failure_summary = (
        shoulder_extent_failure_records
        .groupby(
            ["feature", "reason"],
            as_index=False,
        )
        .agg(
            number=("draw", "size"),
            onset_median=("onset_mass", "median"),
            center_median=("center_mass", "median"),
            max_valid_mass_median=("max_valid_mass", "median"),
        )
    )

    shoulder_totals = (
        shoulder_extent_failure_records
        .groupby("feature")["draw"]
        .size()
        .to_dict()
    )

    shoulder_extent_failure_summary[
        "fraction_of_assigned_shoulders"
    ] = [
        row.number / shoulder_totals[row.feature]
        for row in shoulder_extent_failure_summary.itertuples()
    ]

    print(
        "\nS1/S2 shoulder extent-construction outcomes"
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.precision",
        4,
    ):
        print(
            shoulder_extent_failure_summary.to_string(
                index=False
            )
        )

    print(
        "\nInterpretation guide:"
    )
    print(
        "  missing_following_kappa_zero: the decline does not return to "
        "kappa=0 before the valid tail ends."
    )
    print(
        "  missing_following_Kminus: no persistent maximum-steepening "
        "landmark follows the shoulder onset."
    )
    print(
        "  missing_preceding_Kplus: no persistent left transition "
        "landmark precedes the onset."
    )


# Retained outputs:
#   feature_raw_morphology_diagnostic
#   shoulder_extent_failure_records
#   shoulder_extent_failure_summary  (when records are non-empty)


