# ============================================================
# Prominence-defined bands for persistent 1D features.
#
# Feature centres are those already identified by the discovery
# cell. Band edges are the interpolated half-prominence crossings:
#
#   peaks:     half prominence of the selected density
#   dips:      half prominence of minus the selected density
#   shoulders: complete transition in
#              alpha = d ln(selected density) / d ln m1
#
# Knees are not retained as separate feature classes.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import peak_prominences, peak_widths
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ------------------------------------------------------------
# Configuration.
# ------------------------------------------------------------

FEATURE_BAND_REL_HEIGHT = 0.50

PEAK_BAND_ALPHA = 0.12
DIP_BAND_ALPHA = 0.10
SHOULDER_BAND_ALPHA = 0.12
HIGH_MASS_BAND_ALPHA = 0.08

FEATURE_BAND_LINEWIDTH = 1.5

# Optional manual replacements after visual inspection.
# Keys are ("peak", number), ("dip", number), or
# ("shoulder", number), with numbering starting from one.
#
# Example:
# FEATURE_BAND_MANUAL_OVERRIDES = {
#     ("peak", 2): (8.0, 12.5),
# }
FEATURE_BAND_MANUAL_OVERRIDES = {}


# ------------------------------------------------------------
# Validate products from the discovery cell.
# ------------------------------------------------------------

_required_feature_products = (
    "persistent_features",
    "feature_curves",
    "FEATURE_REFERENCE_SMOOTH_LNM",
    "m_high_q05",
    "m_high_q50",
    "m_high_q95",
    "percentile_label",
    "MASS_DENSITY_MEASURE",
    "MASS_DENSITY_PLAIN_LABEL",
    "MASS_DENSITY_OUTPUT_TAG",
)

_missing_feature_products = [
    name
    for name in _required_feature_products
    if name not in globals()
]

if _missing_feature_products:
    raise RuntimeError(
        "Run the one-dimensional feature-discovery cell first. "
        "Missing: "
        + ", ".join(_missing_feature_products)
    )

reference_curves = feature_curves[
    FEATURE_REFERENCE_SMOOTH_LNM
]

reference_p = np.asarray(
    reference_curves["p"],
    dtype=float,
)

# Backward-compatible name used by the remaining notebook cells.  It denotes
# dP/dm1 in linear mode and dP/dln(m1) in log mode.
reference_density = reference_p

reference_alpha = np.asarray(
    reference_curves["alpha"],
    dtype=float,
)

reference_kappa = np.asarray(
    reference_curves["kappa"],
    dtype=float,
)

reference_valid = np.asarray(
    reference_curves["valid"],
    dtype=bool,
)


# ------------------------------------------------------------
# Half-prominence band helper.
# ------------------------------------------------------------

def _prominence_band(
    values,
    center_index,
    valid_mask,
    invert=False,
    rel_height=0.5,
):
    """
    Return the half-prominence band around an already selected maximum.

    If invert=True, the maximum is found in -values, so the result
    describes a dip in the original curve.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    valid_mask = (
        np.asarray(
            valid_mask,
            dtype=bool,
        )
        & np.isfinite(values)
    )

    valid_indices = np.flatnonzero(
        valid_mask
    )

    if valid_indices.size < 3:
        raise RuntimeError(
            "Too few valid grid points to define a prominence band."
        )

    first_index = int(
        valid_indices[0]
    )

    last_index = int(
        valid_indices[-1]
    )

    if not (
        first_index
        <= center_index
        <= last_index
    ):
        raise RuntimeError(
            f"Feature index {center_index} lies outside the valid domain."
        )

    local_values = values[
        first_index:last_index + 1
    ]

    if invert:
        local_values = -local_values

    local_center = int(
        center_index
        - first_index
    )

    prominence, left_base, right_base = peak_prominences(
        local_values,
        np.asarray(
            [local_center],
            dtype=int,
        ),
    )

    if (
        prominence.size != 1
        or not np.isfinite(prominence[0])
        or prominence[0] <= 0.0
    ):
        raise RuntimeError(
            "Could not determine a positive prominence for the "
            f"feature at m={m_grid[center_index]:.6g}."
        )

    widths, evaluation_height, left_crossing, right_crossing = (
        peak_widths(
            local_values,
            np.asarray(
                [local_center],
                dtype=int,
            ),
            rel_height=rel_height,
            prominence_data=(
                prominence,
                left_base,
                right_base,
            ),
        )
    )

    # peak_widths returns fractional positions in array-index space.
    # Interpolate in ln(m), since m_grid is geometrically spaced.
    left_grid_position = float(
        left_crossing[0]
        + first_index
    )

    right_grid_position = float(
        right_crossing[0]
        + first_index
    )

    grid_positions = np.arange(
        m_grid.size,
        dtype=float,
    )

    left_mass = float(
        np.exp(
            np.interp(
                left_grid_position,
                grid_positions,
                log_m_grid,
            )
        )
    )

    right_mass = float(
        np.exp(
            np.interp(
                right_grid_position,
                grid_positions,
                log_m_grid,
            )
        )
    )

    evaluated_level = float(
        evaluation_height[0]
    )

    if invert:
        evaluated_level = -evaluated_level

    return {
        "left": left_mass,
        "center": float(
            m_grid[center_index]
        ),
        "right": right_mass,
        "level": evaluated_level,
        "prominence": float(
            prominence[0]
        ),
        "left_base_index": int(
            left_base[0]
            + first_index
        ),
        "right_base_index": int(
            right_base[0]
            + first_index
        ),
        "width_lnm": float(
            np.log(
                right_mass / left_mass
            )
        ),
    }



def _first_negative_to_positive_kappa_crossing(
    kappa,
    start_index,
    valid_mask,
):
    """
    Find the first kappa=0 crossing after start_index for which
    kappa changes from negative to non-negative.

    The crossing is interpolated in ln(m), consistently with the
    geometrically spaced mass grid.
    """

    kappa = np.asarray(
        kappa,
        dtype=float,
    )

    valid_mask = (
        np.asarray(
            valid_mask,
            dtype=bool,
        )
        & np.isfinite(kappa)
    )

    for left_index in range(
        int(start_index),
        kappa.size - 1,
    ):
        right_index = (
            left_index + 1
        )

        if not (
            valid_mask[left_index]
            and valid_mask[right_index]
        ):
            continue

        kappa_left = float(
            kappa[left_index]
        )

        kappa_right = float(
            kappa[right_index]
        )

        if (
            kappa_left <= 0.0
            and kappa_right >= 0.0
            and kappa_right > kappa_left
        ):
            if kappa_left == 0.0:
                fraction = 0.0
            elif kappa_right == 0.0:
                fraction = 1.0
            else:
                fraction = (
                    -kappa_left
                    / (
                        kappa_right
                        - kappa_left
                    )
                )

            crossing_log_mass = (
                log_m_grid[left_index]
                + fraction
                * (
                    log_m_grid[right_index]
                    - log_m_grid[left_index]
                )
            )

            crossing_position = (
                left_index
                + fraction
            )

            return {
                "mass": float(
                    np.exp(
                        crossing_log_mass
                    )
                ),
                "grid_position": float(
                    crossing_position
                ),
                "left_index": int(
                    left_index
                ),
                "right_index": int(
                    right_index
                ),
            }

    raise RuntimeError(
        "Could not find a negative-to-positive kappa crossing "
        f"after m={m_grid[start_index]:.6g}."
    )


def _shoulder_transition_band(
    center_index,
):
    """
    Construct the complete transition band associated with a shoulder.

    The shoulder centre is the persistent local maximum of alpha.

    The left boundary is the immediately preceding persistent
    positive-curvature maximum K+.

    The associated maximum-steepening point is the immediately
    following persistent negative-curvature minimum K-.

    The right boundary is the first kappa=0 crossing after K-,
    where alpha reaches its following local minimum and the
    steepening phase ends.
    """

    up_knee_indices = np.sort(
        np.asarray(
            [
                record["index"]
                for record in persistent_features[
                    "up_knee"
                ]
            ],
            dtype=int,
        )
    )

    down_knee_indices = np.sort(
        np.asarray(
            [
                record["index"]
                for record in persistent_features[
                    "down_knee"
                ]
            ],
            dtype=int,
        )
    )

    preceding_up_knees = (
        up_knee_indices[
            up_knee_indices
            < center_index
        ]
    )

    following_down_knees = (
        down_knee_indices[
            down_knee_indices
            > center_index
        ]
    )

    if preceding_up_knees.size == 0:
        raise RuntimeError(
            "No persistent K+ was found before the shoulder at "
            f"m={m_grid[center_index]:.6g}."
        )

    if following_down_knees.size == 0:
        raise RuntimeError(
            "No persistent K- was found after the shoulder at "
            f"m={m_grid[center_index]:.6g}."
        )

    up_knee_index = int(
        preceding_up_knees[-1]
    )

    down_knee_index = int(
        following_down_knees[0]
    )

    if not (
        up_knee_index
        < center_index
        < down_knee_index
    ):
        raise RuntimeError(
            "Invalid K+ / shoulder / K- ordering at "
            f"m={m_grid[center_index]:.6g}."
        )

    right_crossing = (
        _first_negative_to_positive_kappa_crossing(
            reference_kappa,
            start_index=down_knee_index,
            valid_mask=(
                reference_valid
                & np.isfinite(
                    reference_kappa
                )
            ),
        )
    )

    # Retain the alpha prominence as a diagnostic, but do not use
    # its half-prominence crossings as the shoulder boundaries.
    alpha_prominence_band = _prominence_band(
        values=reference_alpha,
        center_index=center_index,
        valid_mask=(
            reference_valid
            & np.isfinite(
                reference_alpha
            )
        ),
        invert=False,
        rel_height=FEATURE_BAND_REL_HEIGHT,
    )

    left_mass = float(
        m_grid[
            up_knee_index
        ]
    )

    right_mass = float(
        right_crossing[
            "mass"
        ]
    )

    if not (
        left_mass
        < m_grid[center_index]
        < m_grid[down_knee_index]
        < right_mass
    ):
        raise RuntimeError(
            "Invalid shoulder-transition ordering: "
            f"K+={left_mass:.6g}, "
            f"S={m_grid[center_index]:.6g}, "
            f"K-={m_grid[down_knee_index]:.6g}, "
            f"kappa-zero={right_mass:.6g}."
        )

    return {
        "left": left_mass,

        # Characteristic shoulder location: maximum steepening.
        "center": float(
            m_grid[
                down_knee_index
            ]
        ),
        "index": int(
            down_knee_index
        ),

        # Retain the original alpha maximum as the shoulder onset.
        "onset_mass": float(
            m_grid[
                center_index
            ]
        ),
        "onset_index": int(
            center_index
        ),

        "right": right_mass,
        "width_lnm": float(
            np.log(
                right_mass
                / left_mass
            )
        ),
        "prominence": float(
            alpha_prominence_band[
                "prominence"
            ]
        ),
        "level": float(
            alpha_prominence_band[
                "level"
            ]
        ),
        "left_base_index": int(
            alpha_prominence_band[
                "left_base_index"
            ]
        ),
        "right_base_index": int(
            alpha_prominence_band[
                "right_base_index"
            ]
        ),
        "old_half_prominence_left": float(
            alpha_prominence_band[
                "left"
            ]
        ),
        "old_half_prominence_right": float(
            alpha_prominence_band[
                "right"
            ]
        ),
        "up_knee_index": up_knee_index,
        "up_knee_mass": left_mass,
        "down_knee_index": down_knee_index,
        "down_knee_mass": float(
            m_grid[
                down_knee_index
            ]
        ),
        "right_zero_mass": right_mass,
        "right_zero_grid_position": float(
            right_crossing[
                "grid_position"
            ]
        ),
        "definition": "curvature_transition",
    }

# ------------------------------------------------------------
# Construct bands for the retained feature classes.
# ------------------------------------------------------------

feature_bands = {
    "peak": [],
    "dip": [],
    "shoulder": [],
}

feature_definitions = {
    "peak": {
        "curve": reference_p,
        "valid": (
            reference_valid
            & np.isfinite(reference_p)
        ),
        "invert": False,
    },
    "dip": {
        "curve": reference_p,
        "valid": (
            reference_valid
            & np.isfinite(reference_p)
        ),
        "invert": True,
    },
    "shoulder": {
        "curve": reference_alpha,
        "valid": (
            reference_valid
            & np.isfinite(reference_alpha)
        ),
        "invert": False,
    },
}

skipped_unbounded_reference_shoulders = []

for feature_type in (
    "peak",
    "dip",
    "shoulder",
):
    definition = feature_definitions[
        feature_type
    ]

    for record in persistent_features[feature_type]:
        if feature_type == "shoulder":
            try:
                band = _shoulder_transition_band(
                    center_index=record["index"],
                )
            except RuntimeError as error:
                message = str(error)
                expected_unbounded_failure = (
                    message.startswith("No persistent K+ was found")
                    or message.startswith("No persistent K- was found")
                    or message.startswith(
                        "Could not find a negative-to-positive kappa crossing"
                    )
                )

                if not expected_unbounded_failure:
                    raise

                skipped_unbounded_reference_shoulders.append(
                    {
                        "mass": float(m_grid[record["index"]]),
                        "reason": message,
                    }
                )
                continue
        else:
            band = _prominence_band(
                values=definition["curve"],
                center_index=record["index"],
                valid_mask=definition["valid"],
                invert=definition["invert"],
                rel_height=FEATURE_BAND_REL_HEIGHT,
            )

        # Assign catalogue numbers only after a complete band has been
        # constructed, so rejected boundary candidates create no gaps.
        feature_number = len(feature_bands[feature_type]) + 1

        override_key = (
            feature_type,
            feature_number,
        )

        if override_key in FEATURE_BAND_MANUAL_OVERRIDES:
            manual_left, manual_right = (
                FEATURE_BAND_MANUAL_OVERRIDES[
                    override_key
                ]
            )

            if not (
                np.isfinite(manual_left)
                and np.isfinite(manual_right)
                and manual_left
                < band["center"]
                < manual_right
            ):
                raise RuntimeError(
                    "Invalid manual band override for "
                    f"{override_key}: "
                    f"{FEATURE_BAND_MANUAL_OVERRIDES[override_key]}"
                )

            band["automatic_left"] = band["left"]
            band["automatic_right"] = band["right"]
            band["left"] = float(manual_left)
            band["right"] = float(manual_right)
            band["manual_override"] = True

        else:
            band["manual_override"] = False

        band["number"] = feature_number

        if feature_type == "shoulder":
            # The discovery index is the alpha maximum; band["index"]
            # is the associated K- and defines the plotted location.
            band["discovery_index"] = int(
                record["index"]
            )
        else:
            band["index"] = int(
                record["index"]
            )

        band["persistence"] = int(
            record["persistence"]
        )

        band["width_lnm"] = float(
            np.log(
                band["right"]
                / band["left"]
            )
        )

        feature_bands[
            feature_type
        ].append(
            band
        )


if skipped_unbounded_reference_shoulders:
    print(
        "Skipped persistent shoulder candidates without a complete "
        "reference transition:"
    )
    for skipped in skipped_unbounded_reference_shoulders:
        print(
            f"  {COORDINATE_NAME}={skipped['mass']:.3f}"
            f"{COORDINATE_UNIT_SUFFIX}: "
            f"{skipped['reason']}"
        )


# ------------------------------------------------------------
# Print the resulting feature definitions.
# ------------------------------------------------------------

print(
    "Prominence-defined one-dimensional feature bands"
)

print(
    "  analyzed density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL} "
    f"(output tag: {MASS_DENSITY_OUTPUT_TAG})"
)

print(
    "One-dimensional feature bands"
)

print(
    "  peaks and dips: "
    f"{100.0 * FEATURE_BAND_REL_HEIGHT:.0f}% "
    "prominence crossings"
)

print(
    "  shoulders: preceding persistent K+ to the first "
    "kappa=0 crossing after the following persistent K-"
)

print(
    "\n"
    f"{'ID':>5s}"
    f"  {'left':>10s}"
    f"  {'location':>10s}"
    f"  {'right':>10s}"
    f"  {f'width ln({COORDINATE_LOG_ARGUMENT})':>12s}"
    f"  {'prominence':>12s}"
)

print(
    "-" * 70
)

feature_prefix = {
    "peak": "P",
    "dip": "D",
    "shoulder": "S",
}

for feature_type in (
    "peak",
    "dip",
    "shoulder",
):
    for band in feature_bands[
        feature_type
    ]:
        feature_id = (
            f"{feature_prefix[feature_type]}"
            f"{band['number']}"
        )

        override_marker = (
            "*"
            if band["manual_override"]
            else ""
        )

        print(
            f"{feature_id:>5s}"
            f"  {band['left']:10.3f}"
            f"  {band['center']:10.3f}"
            f"  {band['right']:10.3f}"
            f"  {band['width_lnm']:12.4f}"
            f"  {band['prominence']:12.5g}"
            f"{override_marker}"
        )


print(
    "\nShoulder transition landmarks"
)

print(
    f"{'ID':>5s}"
    f"  {'left K+':>10s}"
    f"  {'onset':>10s}"
    f"  {'location K-':>12s}"
    f"  {'right kappa=0':>14s}"
)

print(
    "-" * 61
)

for band in feature_bands[
    "shoulder"
]:
    feature_id = (
        f"S{band['number']}"
    )

    print(
        f"{feature_id:>5s}"
        f"  {band['up_knee_mass']:10.3f}"
        f"  {band['onset_mass']:10.3f}"
        f"  {band['center']:12.3f}"
        f"  {band['right_zero_mass']:14.3f}"
    )

    
if FEATURE_BAND_MANUAL_OVERRIDES:
    print(
        "\n  * manually overridden band boundaries"
    )

print(
    "\nUpper-tail scale:"
)

print(
    f"  {'m' if COORDINATE_NAME in ('m1', 'm2') else COORDINATE_NAME}_"
    f"{percentile_label} = "
    f"{m_high_q50:.3f} "
    f"[{m_high_q05:.3f}, {m_high_q95:.3f}]"
    f"{COORDINATE_UNIT_SUFFIX}"
)

