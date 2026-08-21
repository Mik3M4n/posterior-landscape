# ============================================================
# One-dimensional feature discovery in a selectable probability measure:
# peaks, dips, shoulders, knees, and m99.
#
# MASS_DENSITY_MEASURE = "linear" analyzes dP/dm1.
# MASS_DENSITY_MEASURE = "log" analyzes dP/dln(m1) = m1 dP/dm1.
#
# No extrema or numerical derivatives are evaluated separately
# for individual posterior draws.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences


# ------------------------------------------------------------
# Configuration.
# ------------------------------------------------------------

# Select the probability-density measure for the complete 1D workflow.
# Change only this line to reproduce the linear-density analysis.
MASS_DENSITY_MEASURE = "log"  # allowed values: "linear", "log"

if MASS_DENSITY_MEASURE not in ("linear", "log"):
    raise ValueError(
        "MASS_DENSITY_MEASURE must be either 'linear' or 'log'."
    )

if MASS_DENSITY_MEASURE == "linear":
    if COORDINATE_NAME in ("m1", "m2"):
        MASS_DENSITY_OUTPUT_TAG = "dP_dm1"
        MASS_DENSITY_PLAIN_LABEL = "dP/dm1"
        MASS_DENSITY_TEX_LABEL = r"$dP/dm_1$"
        MASS_DENSITY_YLABEL = r"$dP/dm_1\,[M_\odot^{-1}]$"
    else:
        MASS_DENSITY_OUTPUT_TAG = f"dP_d{COORDINATE_NAME}"
        MASS_DENSITY_PLAIN_LABEL = f"dP/d{COORDINATE_NAME}"
        MASS_DENSITY_TEX_LABEL = f"dP/d{COORDINATE_NAME}"
        _coordinate_inverse_unit = (
            f" [{COORDINATE_UNIT}^-1]" if COORDINATE_UNIT else ""
        )
        MASS_DENSITY_YLABEL = (
            f"dP/d{COORDINATE_NAME}{_coordinate_inverse_unit}"
        )
    MASS_DENSITY_LOG_JACOBIAN_POWER = 0.0
else:
    if COORDINATE_NAME in ("m1", "m2"):
        MASS_DENSITY_OUTPUT_TAG = "dP_dlnm1"
        MASS_DENSITY_PLAIN_LABEL = "dP/dln(m1)"
        MASS_DENSITY_TEX_LABEL = r"$dP/d\ln m_1$"
        MASS_DENSITY_YLABEL = r"$dP/d\ln m_1$"
    else:
        MASS_DENSITY_OUTPUT_TAG = f"dP_dlog{COORDINATE_NAME}"
        MASS_DENSITY_PLAIN_LABEL = f"dP/dlog({COORDINATE_NAME})"
        MASS_DENSITY_TEX_LABEL = f"dP/dlog({COORDINATE_NAME})"
        MASS_DENSITY_YLABEL = f"dP/dlog({COORDINATE_LABEL})"
    MASS_DENSITY_LOG_JACOBIAN_POWER = 1.0

# Shoulder morphology is defined by a local flattening while the underlying
# linear-mass density p(m1) is decreasing. If the selected density is
# q=m^beta p, then alpha_q=alpha_p+beta, so alpha_p<0 is equivalent to
# alpha_q<beta. This keeps the physical shoulder criterion unchanged when
# switching between dP/dm1 and dP/dln(m1).
SHOULDER_ALPHA_MAX_SELECTED = MASS_DENSITY_LOG_JACOBIAN_POWER

# All smoothing scales are at or above the adopted resolution floor.
FEATURE_SMOOTH_SCALES_LNM = (
    0.050,
    0.075,
    0.100,
)

# The finest scale is used for plotted candidate locations.
FEATURE_REFERENCE_SMOOTH_LNM = (
    FEATURE_SMOOTH_SCALES_LNM[0]
)

# Extrema must be separated by at least one resolution scale.
FEATURE_MIN_SEPARATION_LNM = 0.050

# Candidates at different smoothing scales are identified if their
# locations agree within this logarithmic-mass distance.
FEATURE_MATCH_TOLERANCE_LNM = 0.100

# Require detection at two or more smoothing scales.
FEATURE_MIN_PERSISTENCE = 2

# Reuse the plotting density floor to avoid differentiating numerical tails.
FEATURE_DERIVATIVE_DENSITY_FLOOR = MARGINAL_FLOOR

# Plot styling.
FEATURE_COLOR = marginal_color
FEATURE_FILL = marginal_fill

SMOOTH_COLORS = {
    0.050: "#D95F02",
    0.075: "#E6A04B",
    0.100: "0.45",
}

PEAK_COLOR = "#D95F02"
DIP_COLOR = "#3B6FB6"
SHOULDER_COLOR = "#2A9D8F"
DOWN_KNEE_COLOR = "#B2182B"
UP_KNEE_COLOR = "#2166AC"
M99_COLOR = "0.20"

FEATURE_AXIS_LABEL_FS = AXIS_LABEL_FS
FEATURE_TICK_LABEL_FS = TICK_LABEL_FS
FEATURE_LEGEND_FS = 10

#M99_CONDITION_MIN = 0.0
HIGH_MASS_PERCENTILE = 0.9999
MASS_PERCENTILE_CONDITION_MIN = 0.0


# ------------------------------------------------------------
# Validate the linear-density input and construct the selected measure.
# ------------------------------------------------------------

pm1_linear_samples = np.asarray(
    pm1_samples,
    dtype=float,
)

if (
    pm1_linear_samples.ndim != 2
    or pm1_linear_samples.shape[1] != m_grid.size
):
    raise RuntimeError(
        "pm1_samples must have shape (number of draws, len(m_grid))."
    )

if not np.all(
    np.diff(m_grid) > 0.0
):
    raise RuntimeError(
        "m_grid must be strictly increasing."
    )

if not np.all(m_grid > 0.0):
    raise RuntimeError(
        "m_grid must be positive for a logarithmic-mass analysis."
    )

log_m_grid = np.log(
    m_grid
)

dlog_m_grid = float(
    np.median(
        np.diff(log_m_grid)
    )
)

if not np.allclose(
    np.diff(log_m_grid),
    dlog_m_grid,
    rtol=1.0e-5,
    atol=1.0e-12,
):
    raise RuntimeError(
        "This diagnostic assumes a geometrically spaced m_grid."
    )

# Midpoint cells locate partial-band overlaps. Their widths need not equal
# the notebook's quadrature weights `dm` (for example, `dm` may use
# trapezoidal endpoint weights).
linear_mass_cell_edges = np.empty(m_grid.size + 1, dtype=float)
linear_mass_cell_edges[1:-1] = 0.5 * (m_grid[:-1] + m_grid[1:])
linear_mass_cell_edges[0] = m_grid[0] - 0.5 * (m_grid[1] - m_grid[0])
linear_mass_cell_edges[-1] = m_grid[-1] + 0.5 * (
    m_grid[-1] - m_grid[-2]
)
linear_mass_cell_widths = np.diff(linear_mass_cell_edges)

linear_mass_quadrature_weights = np.asarray(
    dm,
    dtype=float,
).reshape(-1)

if linear_mass_quadrature_weights.size != m_grid.size:
    raise RuntimeError(
        "dm must contain one linear-mass quadrature weight per m_grid point."
    )

if not np.all(
    np.isfinite(linear_mass_quadrature_weights)
    & (linear_mass_quadrature_weights >= 0.0)
) or not np.any(linear_mass_quadrature_weights > 0.0):
    raise RuntimeError(
        "dm contains invalid linear-mass quadrature weights."
    )

# Midpoint cells in logarithmic mass.
log_mass_cell_edges = np.empty(m_grid.size + 1, dtype=float)
log_mass_cell_edges[1:-1] = 0.5 * (
    log_m_grid[:-1] + log_m_grid[1:]
)
log_mass_cell_edges[0] = log_m_grid[0] - 0.5 * (
    log_m_grid[1] - log_m_grid[0]
)
log_mass_cell_edges[-1] = log_m_grid[-1] + 0.5 * (
    log_m_grid[-1] - log_m_grid[-2]
)
log_mass_cell_widths = np.diff(log_mass_cell_edges)

# Preserve the original discrete probability weights exactly under
# q(ln m) = m p(m): [m p(m)] [dm / m] = p(m) dm.
log_mass_quadrature_weights = (
    linear_mass_quadrature_weights
    / m_grid
)

pm1_linear_samples = np.where(
    np.isfinite(pm1_linear_samples)
    & (pm1_linear_samples >= 0.0),
    pm1_linear_samples,
    0.0,
)

linear_draw_norm = np.einsum(
    "si,i->s",
    pm1_linear_samples,
    linear_mass_quadrature_weights,
    optimize=True,
)

if not np.all(
    np.isfinite(linear_draw_norm)
    & (linear_draw_norm > 0.0)
):
    raise RuntimeError(
        "pm1_samples contains draws with invalid linear-mass normalization."
    )

pm1_linear_samples = (
    pm1_linear_samples
    / linear_draw_norm[:, None]
)

if MASS_DENSITY_MEASURE == "linear":
    MASS_MEASURE_CELL_EDGES = linear_mass_cell_edges
    MASS_MEASURE_CELL_WIDTHS = linear_mass_cell_widths
    MASS_MEASURE_WIDTHS = linear_mass_quadrature_weights
    pm1_feature_samples = pm1_linear_samples.copy()
else:
    MASS_MEASURE_CELL_EDGES = log_mass_cell_edges
    MASS_MEASURE_CELL_WIDTHS = log_mass_cell_widths
    MASS_MEASURE_WIDTHS = log_mass_quadrature_weights
    pm1_feature_samples = (
        pm1_linear_samples
        * m_grid[None, :]
    )

# Renormalize in the selected measure.  For a sufficiently fine grid this
# correction is very close to unity, but making it explicit prevents a
# measure mismatch from propagating into feature probabilities.
selected_draw_norm_before = np.einsum(
    "si,i->s",
    pm1_feature_samples,
    MASS_MEASURE_WIDTHS,
    optimize=True,
)

if not np.all(
    np.isfinite(selected_draw_norm_before)
    & (selected_draw_norm_before > 0.0)
):
    raise RuntimeError(
        "Invalid normalization in the selected mass-density measure."
    )

pm1_feature_samples = (
    pm1_feature_samples
    / selected_draw_norm_before[:, None]
)

selected_draw_norm_after = np.einsum(
    "si,i->s",
    pm1_feature_samples,
    MASS_MEASURE_WIDTHS,
    optimize=True,
)

if not np.allclose(
    selected_draw_norm_after,
    1.0,
    rtol=1.0e-10,
    atol=1.0e-12,
):
    raise RuntimeError(
        "Selected density draws do not integrate to unity."
    )

minimum_extrema_distance = max(
    1,
    int(
        np.ceil(
            FEATURE_MIN_SEPARATION_LNM
            / dlog_m_grid
        )
    ),
)

pm1_q05_feature, pm1_q50_feature, pm1_q95_feature = np.quantile(
    pm1_feature_samples,
    [
        0.05,
        0.50,
        0.95,
    ],
    axis=0,
)


# ------------------------------------------------------------
# Helper functions.
# ------------------------------------------------------------

def _normalized_smoothed_density(
    density,
    smooth_scale_lnm,
):
    """Smooth on the log-mass grid and normalize in the selected measure."""

    sigma_grid = (
        smooth_scale_lnm
        / dlog_m_grid
    )

    smoothed = gaussian_filter1d(
        np.asarray(
            density,
            dtype=float,
        ),
        sigma=sigma_grid,
        mode="nearest",
    )

    smoothed = np.where(
        np.isfinite(smoothed)
        & (smoothed > 0.0),
        smoothed,
        0.0,
    )

    norm = np.sum(
        smoothed * MASS_MEASURE_WIDTHS
    )

    if (
        not np.isfinite(norm)
        or norm <= 0.0
    ):
        raise RuntimeError(
            "Invalid normalization after smoothing the selected density."
        )

    return (
        smoothed / norm
    )


def _valid_contiguous_slice(
    valid_mask,
):
    """Return the first and last valid indices of a contiguous domain."""

    valid_indices = np.flatnonzero(
        valid_mask
    )

    if valid_indices.size < 3:
        raise RuntimeError(
            "Too few valid grid points for derivative feature discovery."
        )

    return (
        int(valid_indices[0]),
        int(valid_indices[-1]),
    )


def _find_local_maxima(
    values,
    valid_mask,
):
    """Find local maxima and their prominences inside the valid domain."""

    first_index, last_index = _valid_contiguous_slice(
        valid_mask
    )

    local_values = np.asarray(
        values[
            first_index:last_index + 1
        ],
        dtype=float,
    )

    local_indices, _ = find_peaks(
        local_values,
        distance=minimum_extrema_distance,
    )

    global_indices = (
        local_indices
        + first_index
    )

    if global_indices.size:
        prominence, _, _ = peak_prominences(
            local_values,
            local_indices,
        )
    else:
        prominence = np.asarray(
            [],
            dtype=float,
        )

    return (
        global_indices.astype(int),
        np.asarray(
            prominence,
            dtype=float,
        ),
    )


def _match_persistence(
    reference_indices,
    candidates_by_scale,
):
    """
    Count the number of smoothing scales containing a candidate
    within FEATURE_MATCH_TOLERANCE_LNM of each reference candidate.
    """

    records = []

    for reference_index in reference_indices:
        reference_log_mass = (
            log_m_grid[
                reference_index
            ]
        )

        matched_scales = []

        for smooth_scale in FEATURE_SMOOTH_SCALES_LNM:
            candidate_indices = np.asarray(
                candidates_by_scale[
                    smooth_scale
                ],
                dtype=int,
            )

            if candidate_indices.size == 0:
                continue

            distance = np.abs(
                log_m_grid[
                    candidate_indices
                ]
                - reference_log_mass
            )

            if np.nanmin(
                distance
            ) <= FEATURE_MATCH_TOLERANCE_LNM:
                matched_scales.append(
                    smooth_scale
                )

        records.append(
            {
                "index": int(
                    reference_index
                ),
                "mass": float(
                    m_grid[
                        reference_index
                    ]
                ),
                "persistence": len(
                    matched_scales
                ),
                "matched_scales": tuple(
                    matched_scales
                ),
            }
        )

    return records


def _retain_persistent(
    records,
):
    """Keep candidates detected at the required number of scales."""

    return [
        record
        for record in records
        if record["persistence"]
        >= FEATURE_MIN_PERSISTENCE
    ]


def _posterior_mass_percentile(
    density_samples,
    percentile,
    lower_mass=None,
):
    """
    Compute a mass percentile for every density draw.

    If lower_mass is supplied, compute the percentile of the
    conditional distribution p(m1 | m1 > lower_mass).
    """

    integration_widths = np.asarray(
        MASS_MEASURE_WIDTHS,
        dtype=float,
    ).copy()

    if (
        lower_mass is not None
        and lower_mass > m_grid[0]
    ):
        if MASS_DENSITY_MEASURE == "linear":
            lower_coordinate = float(lower_mass)
        else:
            lower_coordinate = float(np.log(lower_mass))

        # Retain the corresponding fraction of each quadrature weight.
        overlap_widths = np.maximum(
            0.0,
            MASS_MEASURE_CELL_EDGES[1:]
            - np.maximum(
                MASS_MEASURE_CELL_EDGES[:-1],
                lower_coordinate,
            ),
        )

        integration_widths = (
            MASS_MEASURE_WIDTHS
            * overlap_widths
            / MASS_MEASURE_CELL_WIDTHS
        )

    weighted_cells = (
        density_samples
        * integration_widths[None, :]
    )

    draw_norm = np.sum(
        weighted_cells,
        axis=1,
    )

    valid_draws = (
        np.isfinite(draw_norm)
        & (draw_norm > 0.0)
    )

    cdf = np.full_like(
        weighted_cells,
        np.nan,
    )

    cdf[valid_draws] = np.cumsum(
        weighted_cells[
            valid_draws
        ],
        axis=1,
    ) / draw_norm[
        valid_draws,
        None,
    ]

    percentile_samples = np.full(
        density_samples.shape[0],
        np.nan,
    )

    for sample_index in np.flatnonzero(
        valid_draws
    ):
        cdf_i = cdf[
            sample_index
        ]

        usable = (
            np.isfinite(cdf_i)
            & (
                integration_widths
                > 0.0
            )
        )

        if np.count_nonzero(
            usable
        ) < 2:
            continue

        cdf_unique, unique_indices = np.unique(
            cdf_i[
                usable
            ],
            return_index=True,
        )

        mass_unique = m_grid[
            usable
        ][
            unique_indices
        ]

        if (
            cdf_unique[0]
            <= percentile
            <= cdf_unique[-1]
        ):
            percentile_samples[
                sample_index
            ] = np.interp(
                percentile,
                cdf_unique,
                mass_unique,
            )

    return percentile_samples

# ------------------------------------------------------------
# Construct smoothed densities, slopes, and curvatures.
# ------------------------------------------------------------

feature_curves = {}

for smooth_scale in FEATURE_SMOOTH_SCALES_LNM:
    p_smooth = _normalized_smoothed_density(
        pm1_q50_feature,
        smooth_scale,
    )

    valid = (
        np.isfinite(p_smooth)
        & (
            p_smooth
            >= FEATURE_DERIVATIVE_DENSITY_FLOOR
        )
    )

    log_p_smooth = np.full_like(
        p_smooth,
        np.nan,
    )

    log_p_smooth[valid] = np.log(
        p_smooth[
            valid
        ]
    )

    first_valid, last_valid = _valid_contiguous_slice(
        valid
    )

    # Derivatives are evaluated only on the contiguous valid interval.
    alpha_valid = np.gradient(
        log_p_smooth[
            first_valid:last_valid + 1
        ],
        log_m_grid[
            first_valid:last_valid + 1
        ],
        edge_order=2,
    )

    kappa_valid = np.gradient(
        alpha_valid,
        log_m_grid[
            first_valid:last_valid + 1
        ],
        edge_order=2,
    )

    alpha = np.full_like(
        p_smooth,
        np.nan,
    )

    kappa = np.full_like(
        p_smooth,
        np.nan,
    )

    alpha[
        first_valid:last_valid + 1
    ] = alpha_valid

    kappa[
        first_valid:last_valid + 1
    ] = kappa_valid

    feature_curves[
        smooth_scale
    ] = {
        "p": p_smooth,
        "alpha": alpha,
        "kappa": kappa,
        "valid": valid,
    }


# ------------------------------------------------------------
# Identify candidate types at every smoothing scale.
# ------------------------------------------------------------

candidate_indices = {
    "peak": {},
    "dip": {},
    "shoulder": {},
    "down_knee": {},
    "up_knee": {},
}

candidate_prominence = {
    "peak": {},
    "dip": {},
    "shoulder": {},
    "down_knee": {},
    "up_knee": {},
}

for smooth_scale in FEATURE_SMOOTH_SCALES_LNM:
    curves = feature_curves[
        smooth_scale
    ]

    p_smooth = curves[
        "p"
    ]

    alpha = curves[
        "alpha"
    ]

    kappa = curves[
        "kappa"
    ]

    valid = curves[
        "valid"
    ]

    # Peaks: local maxima of the selected density.
    peaks, peak_prominence = _find_local_maxima(
        p_smooth,
        valid,
    )

    # Dips: local minima of the selected density.
    dips, dip_prominence = _find_local_maxima(
        -p_smooth,
        valid,
    )

    # Shoulders: local maxima of the selected-density slope for which the
    # underlying linear-mass density remains decreasing.
    shoulder_all, shoulder_prominence_all = _find_local_maxima(
        alpha,
        valid & np.isfinite(alpha),
    )

    shoulder_keep = (
        alpha[
            shoulder_all
        ]
        < SHOULDER_ALPHA_MAX_SELECTED
    )

    shoulders = shoulder_all[
        shoulder_keep
    ]

    shoulder_prominence = shoulder_prominence_all[
        shoulder_keep
    ]

    # Downward knees: local minima of kappa.
    down_knees, down_knee_prominence = _find_local_maxima(
        -kappa,
        valid & np.isfinite(kappa),
    )

    # Upward knees: local maxima of kappa.
    up_knees, up_knee_prominence = _find_local_maxima(
        kappa,
        valid & np.isfinite(kappa),
    )

    candidate_indices[
        "peak"
    ][smooth_scale] = peaks

    candidate_indices[
        "dip"
    ][smooth_scale] = dips

    candidate_indices[
        "shoulder"
    ][smooth_scale] = shoulders

    candidate_indices[
        "down_knee"
    ][smooth_scale] = down_knees

    candidate_indices[
        "up_knee"
    ][smooth_scale] = up_knees

    candidate_prominence[
        "peak"
    ][smooth_scale] = peak_prominence

    candidate_prominence[
        "dip"
    ][smooth_scale] = dip_prominence

    candidate_prominence[
        "shoulder"
    ][smooth_scale] = shoulder_prominence

    candidate_prominence[
        "down_knee"
    ][smooth_scale] = down_knee_prominence

    candidate_prominence[
        "up_knee"
    ][smooth_scale] = up_knee_prominence


# ------------------------------------------------------------
# Require persistence across smoothing scales.
# ------------------------------------------------------------

persistent_features = {}

for feature_type in candidate_indices:
    reference_candidates = candidate_indices[
        feature_type
    ][
        FEATURE_REFERENCE_SMOOTH_LNM
    ]

    records = _match_persistence(
        reference_candidates,
        candidate_indices[
            feature_type
        ],
    )

    persistent_features[
        feature_type
    ] = _retain_persistent(
        records
    )


# ------------------------------------------------------------
# Compute unconditional posterior m99.
# ------------------------------------------------------------

m_high_percentile_samples = _posterior_mass_percentile(
    pm1_feature_samples,
    percentile=HIGH_MASS_PERCENTILE,
    lower_mass=MASS_PERCENTILE_CONDITION_MIN,
)

m_high_q05, m_high_q50, m_high_q95 = np.nanquantile(
    m_high_percentile_samples,
    [
        0.05,
        0.50,
        0.95,
    ],
)


# ------------------------------------------------------------
# Print candidate tables.
# ------------------------------------------------------------

feature_titles = {
    "peak": "Peaks",
    "dip": "Dips",
    "shoulder": "Shoulders",
    "down_knee": "Downward knees",
    "up_knee": "Upward knees",
}

print(
    "Persistent one-dimensional feature candidates"
)

print(
    "  analyzed density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL} "
    f"(output tag: {MASS_DENSITY_OUTPUT_TAG})"
)

print(
    "  reference smoothing scale: "
    f"{FEATURE_REFERENCE_SMOOTH_LNM:.3f} in "
    f"ln({COORDINATE_LOG_ARGUMENT})"
)

print(
    "  smoothing scales tested: "
    + ", ".join(
        f"{scale:.3f}"
        for scale in FEATURE_SMOOTH_SCALES_LNM
    )
)

print(
    "  persistence requirement: "
    f"{FEATURE_MIN_PERSISTENCE} of "
    f"{len(FEATURE_SMOOTH_SCALES_LNM)} scales"
)

print(
    "  matching tolerance: "
    f"{FEATURE_MATCH_TOLERANCE_LNM:.3f} in "
    f"ln({COORDINATE_LOG_ARGUMENT})"
)

print(
    "  shoulder slope requirement in selected measure: "
    f"alpha < {SHOULDER_ALPHA_MAX_SELECTED:g} "
    f"(equivalent to alpha[dP/d{COORDINATE_NAME}] < 0)"
)

for feature_type in (
    "peak",
    "dip",
    "shoulder",
    "down_knee",
    "up_knee",
):
    print()
    print(
        feature_titles[
            feature_type
        ]
    )

    records = persistent_features[
        feature_type
    ]

    if not records:
        print(
            "  none"
        )
        continue

    reference_indices = candidate_indices[
        feature_type
    ][
        FEATURE_REFERENCE_SMOOTH_LNM
    ]

    reference_prominence = candidate_prominence[
        feature_type
    ][
        FEATURE_REFERENCE_SMOOTH_LNM
    ]

    prominence_lookup = {
        int(index): float(prominence)
        for index, prominence in zip(
            reference_indices,
            reference_prominence,
        )
    }

    if feature_type == "shoulder":
        value_curve = feature_curves[
            FEATURE_REFERENCE_SMOOTH_LNM
        ][
            "alpha"
        ]

        value_name = "alpha"
    elif feature_type in (
        "down_knee",
        "up_knee",
    ):
        value_curve = feature_curves[
            FEATURE_REFERENCE_SMOOTH_LNM
        ][
            "kappa"
        ]

        value_name = "kappa"
    else:
        value_curve = feature_curves[
            FEATURE_REFERENCE_SMOOTH_LNM
        ][
            "p"
        ]

        value_name = MASS_DENSITY_PLAIN_LABEL

    print(
        f"  {COORDINATE_VALUE_HEADING:>10s}"
        f"  {value_name:>14s}"
        f"  {'prominence':>14s}"
        f"  {'persistence':>12s}"
    )

    for record in records:
        index = record[
            "index"
        ]

        print(
            f"  {record['mass']:10.3f}"
            f"  {value_curve[index]:14.6g}"
            f"  {prominence_lookup[index]:14.6g}"
            f"  {record['persistence']:5d}/"
            f"{len(FEATURE_SMOOTH_SCALES_LNM):<6d}"
        )

print()

percentile_label = (
    f"{100.0 * HIGH_MASS_PERCENTILE:g}"
)

print(
    f"Posterior conditional "
    f"{'m' if COORDINATE_NAME in ('m1', 'm2') else COORDINATE_NAME}_"
    f"{percentile_label} for {COORDINATE_NAME} > "
    f"{MASS_PERCENTILE_CONDITION_MIN:g}{COORDINATE_UNIT_SUFFIX}:"
)

print(
    f"  {m_high_q50:.3f} "
    f"[{m_high_q05:.3f}, {m_high_q95:.3f}]"
    f"{COORDINATE_UNIT_SUFFIX} "
    f"(median and 90% credible interval)"
)

