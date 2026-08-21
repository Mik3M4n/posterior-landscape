# ============================================================
# CELL 1: aligned summaries, dependence metrics, and tables.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import rankdata, spearmanr

from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


# ------------------------------------------------------------
# Configuration.
# ------------------------------------------------------------

H0_FEATURE_ORDER = (
    "P1",
    "D1",
    "P2",
    "D2",
    "P3",
    "S1",
    "S2",
    "m99.99",
)

H0_FEATURE_LABELS = {
    "P1": r"$P_1$",
    "D1": r"$D_1$",
    "P2": r"$P_2$",
    "D2": r"$D_2$",
    "P3": r"$P_3$",
    "S1": r"$S_1$",
    "S2": r"$S_2$",
    "m99.99": r"$m_{99.99}$",
}

H0_FEATURE_COLORS = {
    "P1": PEAK_COLOR,
    "P2": PEAK_COLOR,
    "P3": PEAK_COLOR,
    "D1": DIP_COLOR,
    "D2": DIP_COLOR,
    "S1": SHOULDER_COLOR,
    "S2": SHOULDER_COLOR,
    "m99.99": "0.30",
}

# Continuous summaries evaluated only where they are defined.
H0_CHANNEL_ORDER = (
    "mass scale",
    "band probability",
    "width",
    "peak contrast",
    "dip contrast",
    "shoulder contrast",
)

H0_CHANNEL_LABELS = {
    "mass scale": r"Feature mass scale",
    "band probability": r"Band probability $P_f$",
    "width": r"Relative width $w_f$",
    "peak contrast": r"Peak strength $\Delta_{\rm P}$",
    "dip contrast": r"Dip strength $\Delta_{\rm D}$",
    "shoulder contrast": r"Shoulder strength $\Delta_{\rm S}$",
}

H0_CHANNEL_SHORT_LABELS = {
    "mass scale": r"mass scale",
    "band probability": r"$P_f$",
    "width": r"$w_f$",
    "peak contrast": r"$\Delta_{\rm P}$",
    "dip contrast": r"$\Delta_{\rm D}$",
    "shoulder contrast": r"$\Delta_{\rm S}$",
}

H0_CHANNEL_COLORS = {
    "mass scale": "#0072B2",
    "band probability": "#D55E00",
    "width": "#CC79A7",
    "peak contrast": "#009E73",
    "dip contrast": "#7A9E00",
    "shoulder contrast": "#8C564B",
}

# Mutual-information estimator and null calibration.
MI_N_NEIGHBORS = 5
MI_N_PERMUTATIONS = 100
MI_RANDOM_SEED = RANDOM_SEED
MI_MIN_CHANNEL_SAMPLES = 200

# A binary association is not reported unless both groups contain this
# many draws. This prevents nearly universal features from receiving an
# unstable present-versus-absent statistic.
BINARY_MIN_GROUP_SAMPLES = 20

# Plot configuration shared by both production figures.
H0_PLOT_NBINS = 70
H0_PLOT_SMOOTH_SIGMA = 1.15
H0_PLOT_DISPLAY_QUANTILE = 0.995

H0_AXIS_LABEL_FS = 12
H0_TICK_LABEL_FS = 9.5
H0_TITLE_FS = 12
H0_ANNOTATION_FS = 8.5


# ------------------------------------------------------------
# Validate exact draw alignment and required feature products.
# ------------------------------------------------------------

_required_h0_products = (
    "params",
    "trace",
    "feature_draw_posteriors",
    "m_high_percentile_samples",
    "MASS_DENSITY_OUTPUT_TAG",
    "MASS_DENSITY_PLAIN_LABEL",
    "MASS_DENSITY_TEX_LABEL",
)

_missing_h0_products = [
    name
    for name in _required_h0_products
    if name not in globals()
]

if _missing_h0_products:
    raise RuntimeError(
        "Run the revised draw-level feature cell first. Missing: "
        + ", ".join(_missing_h0_products)
    )

print(
    "H0 feature analysis uses the 1D density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL}"
)
print(f"Output tag: {MASS_DENSITY_OUTPUT_TAG}")

for _feature_id in H0_FEATURE_ORDER[:-1]:
    if _feature_id not in feature_draw_posteriors:
        raise RuntimeError(
            f"Missing draw-level posterior feature: {_feature_id}"
        )

H0_samples = np.asarray(
    params["H0"],
    dtype=float,
).reshape(-1)

if not np.all(np.isfinite(H0_samples)):
    raise RuntimeError("H0 contains non-finite samples.")

number_posterior_draws = H0_samples.size

number_chains = int(trace.posterior.sizes["chain"])
number_draws_per_chain = int(trace.posterior.sizes["draw"])

if number_chains * number_draws_per_chain != number_posterior_draws:
    raise RuntimeError(
        "H0 does not contain the complete chain-major posterior."
    )

m9999_samples = np.asarray(
    m_high_percentile_samples,
    dtype=float,
).reshape(-1)

if m9999_samples.size != number_posterior_draws:
    raise RuntimeError(
        "m_high_percentile_samples is not aligned with H0."
    )

for _feature_id in H0_FEATURE_ORDER[:-1]:
    _output = feature_draw_posteriors[_feature_id]

    for _field in (
        "located",
        "bounded",
        "present",
        "realized_type",
        "center",
        "left",
        "right",
        "width_relative",
        "band_probability",
        "mu_hat",
        "mu_deficit",
        "contrast",
    ):
        if _field not in _output:
            raise RuntimeError(
                f"{_feature_id} is missing draw-level field '{_field}'."
            )

        if np.asarray(_output[_field]).reshape(-1).size != number_posterior_draws:
            raise RuntimeError(
                f"{_feature_id}:{_field} is not aligned with H0."
            )

print("Aligned posterior samples:")
print(f"  chains             = {number_chains}")
print(f"  draws per chain    = {number_draws_per_chain}")
print(f"  total draws        = {number_posterior_draws}")


# ------------------------------------------------------------
# Circular-shift null permutations.
#
# H0 is shifted independently within each chain. This preserves its
# marginal distribution and within-chain autocorrelation while removing
# draw-by-draw association with the feature summaries.
# ------------------------------------------------------------

_permutation_rng = np.random.default_rng(MI_RANDOM_SEED)

_H0_by_chain = H0_samples.reshape(
    number_chains,
    number_draws_per_chain,
)

H0_null_permutations = np.empty(
    (MI_N_PERMUTATIONS, number_posterior_draws),
    dtype=float,
)

for _permutation_index in range(MI_N_PERMUTATIONS):
    _shifted_chains = []

    for _chain_index in range(number_chains):
        _shift = int(
            _permutation_rng.integers(
                1,
                number_draws_per_chain,
            )
        )

        _shifted_chains.append(
            np.roll(
                _H0_by_chain[_chain_index],
                _shift,
            )
        )

    H0_null_permutations[_permutation_index] = np.concatenate(
        _shifted_chains
    )


# ------------------------------------------------------------
# Dependence estimators.
# ------------------------------------------------------------

def _h0new_standardize(values):
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))

    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Cannot standardize a constant variable.")

    return (values - np.mean(values)) / scale


def _h0new_mutual_information_bits(
    x,
    y,
    random_state,
):
    """KSG-1 continuous-continuous mutual information, in bits."""

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    number_samples = x.size

    if number_samples <= MI_N_NEIGHBORS + 2:
        return np.nan

    x = _h0new_standardize(x)
    y = _h0new_standardize(y)

    rng = np.random.default_rng(random_state)
    jitter_scale = 1.0e-10
    x = x + jitter_scale * rng.normal(size=number_samples)
    y = y + jitter_scale * rng.normal(size=number_samples)

    joint = np.column_stack((x, y))
    joint_tree = cKDTree(joint)

    joint_distances, _ = joint_tree.query(
        joint,
        k=MI_N_NEIGHBORS + 1,
        p=np.inf,
    )

    epsilon = joint_distances[:, MI_N_NEIGHBORS]

    if not np.all(np.isfinite(epsilon)):
        raise RuntimeError("Invalid nearest-neighbor distances.")

    epsilon_strict = np.nextafter(epsilon, 0.0)

    x_tree = cKDTree(x[:, None])
    y_tree = cKDTree(y[:, None])

    number_x = (
        x_tree.query_ball_point(
            x[:, None],
            r=epsilon_strict,
            p=np.inf,
            return_length=True,
        )
        - 1
    )

    number_y = (
        y_tree.query_ball_point(
            y[:, None],
            r=epsilon_strict,
            p=np.inf,
            return_length=True,
        )
        - 1
    )

    information_nats = (
        digamma(MI_N_NEIGHBORS)
        + digamma(number_samples)
        - np.mean(
            digamma(number_x + 1)
            + digamma(number_y + 1)
        )
    )

    return float(information_nats / np.log(2.0))


def _h0new_continuous_dependence(
    values,
    statistic_name,
):
    """Spearman and permutation-calibrated MI for one summary."""

    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size != number_posterior_draws:
        raise RuntimeError(
            f"{statistic_name} is not aligned with H0."
        )

    valid = np.isfinite(values) & np.isfinite(H0_samples)
    number_valid = int(np.count_nonzero(valid))

    if number_valid < MI_MIN_CHANNEL_SAMPLES:
        return {
            "rho_s": np.nan,
            "mi_raw_bits": np.nan,
            "mi_null_median_bits": np.nan,
            "mi_calibrated_bits": np.nan,
            "mi_permutation_probability": np.nan,
            "number_samples": number_valid,
        }

    rho_s = float(
        spearmanr(
            values[valid],
            H0_samples[valid],
        ).statistic
    )

    information_raw = _h0new_mutual_information_bits(
        values[valid],
        H0_samples[valid],
        random_state=MI_RANDOM_SEED,
    )

    information_null = np.empty(MI_N_PERMUTATIONS, dtype=float)

    for permutation_index in range(MI_N_PERMUTATIONS):
        information_null[permutation_index] = (
            _h0new_mutual_information_bits(
                values[valid],
                H0_null_permutations[permutation_index, valid],
                random_state=(
                    MI_RANDOM_SEED
                    + permutation_index
                    + 1
                ),
            )
        )

    null_median = float(np.nanmedian(information_null))
    calibrated_information = float(information_raw - null_median)

    permutation_probability = float(
        (
            1
            + np.count_nonzero(
                information_null >= information_raw
            )
        )
        / (MI_N_PERMUTATIONS + 1)
    )

    return {
        "rho_s": rho_s,
        "mi_raw_bits": information_raw,
        "mi_null_median_bits": null_median,
        "mi_calibrated_bits": calibrated_information,
        "mi_permutation_probability": permutation_probability,
        "number_samples": number_valid,
    }


def _h0new_rank_biserial(
    continuous,
    indicator,
):
    """
    Rank-biserial association between H0 and a binary indicator.

    Positive values mean larger H0 when indicator==1.
    """

    continuous = np.asarray(continuous, dtype=float).reshape(-1)
    indicator = np.asarray(indicator, dtype=bool).reshape(-1)

    group_one = continuous[indicator]
    group_zero = continuous[~indicator]

    number_one = group_one.size
    number_zero = group_zero.size

    if (
        number_one < BINARY_MIN_GROUP_SAMPLES
        or number_zero < BINARY_MIN_GROUP_SAMPLES
    ):
        return np.nan

    ranks = rankdata(
        np.concatenate((group_one, group_zero)),
        method="average",
    )

    rank_sum_one = float(np.sum(ranks[:number_one]))

    mann_whitney_u = (
        rank_sum_one
        - number_one * (number_one + 1) / 2.0
    )

    auc = mann_whitney_u / (number_one * number_zero)
    return float(2.0 * auc - 1.0)


def _h0new_binary_dependence(
    indicator,
    eligible=None,
):
    """H0 dependence of a binary presence or boundedness indicator."""

    indicator = np.asarray(indicator, dtype=bool).reshape(-1)

    if indicator.size != number_posterior_draws:
        raise RuntimeError("Binary feature indicator is not aligned.")

    if eligible is None:
        eligible = np.ones(number_posterior_draws, dtype=bool)
    else:
        eligible = np.asarray(eligible, dtype=bool).reshape(-1)

    valid = eligible & np.isfinite(H0_samples)
    indicator_valid = indicator[valid]
    h0_valid = H0_samples[valid]

    number_one = int(np.count_nonzero(indicator_valid))
    number_zero = int(indicator_valid.size - number_one)

    if number_one:
        median_one = float(np.median(h0_valid[indicator_valid]))
    else:
        median_one = np.nan

    if number_zero:
        median_zero = float(np.median(h0_valid[~indicator_valid]))
    else:
        median_zero = np.nan

    delta_median = (
        median_one - median_zero
        if np.isfinite(median_one) and np.isfinite(median_zero)
        else np.nan
    )

    rank_biserial = _h0new_rank_biserial(
        h0_valid,
        indicator_valid,
    )

    if np.isfinite(rank_biserial):
        null_values = np.empty(MI_N_PERMUTATIONS, dtype=float)

        for permutation_index in range(MI_N_PERMUTATIONS):
            null_values[permutation_index] = _h0new_rank_biserial(
                H0_null_permutations[permutation_index, valid],
                indicator_valid,
            )

        permutation_probability = float(
            (
                1
                + np.count_nonzero(
                    np.abs(null_values)
                    >= abs(rank_biserial)
                )
            )
            / (MI_N_PERMUTATIONS + 1)
        )
    else:
        permutation_probability = np.nan

    return {
        "number_eligible": int(np.count_nonzero(valid)),
        "number_true": number_one,
        "number_false": number_zero,
        "probability_true": (
            number_one / indicator_valid.size
            if indicator_valid.size
            else np.nan
        ),
        "median_H0_true": median_one,
        "median_H0_false": median_zero,
        "delta_median_H0_true_minus_false": delta_median,
        "rank_biserial": rank_biserial,
        "permutation_probability": permutation_probability,
    }


# ------------------------------------------------------------
# Draw-level moving-band summaries.
# ------------------------------------------------------------

h0_feature_statistics = {}

for _feature_id in H0_FEATURE_ORDER:
    if _feature_id == "m99.99":
        _median_mass = float(np.nanmedian(m9999_samples))
        _delta_mass = np.full_like(m9999_samples, np.nan)
        _valid_mass = np.isfinite(m9999_samples) & (m9999_samples > 0.0)
        _delta_mass[_valid_mass] = np.log(
            m9999_samples[_valid_mass] / _median_mass
        )

        h0_feature_statistics[_feature_id] = {
            "located": np.ones(
                number_posterior_draws,
                dtype=bool,
            ),
            "bounded": np.ones(
                number_posterior_draws,
                dtype=bool,
            ),
            "mass_scale": m9999_samples,
            "delta_mass_scale": _delta_mass,
            "band_probability": None,
            "width": None,
            "contrasts": {},
        }
        continue

    _output = feature_draw_posteriors[_feature_id]

    _located = np.asarray(
        _output["located"],
        dtype=bool,
    ).reshape(-1)

    _bounded = np.asarray(
        _output["bounded"],
        dtype=bool,
    ).reshape(-1)

    _realized_type = np.asarray(
        _output["realized_type"],
        dtype=object,
    ).reshape(-1)

    _mass_scale_field = (
        "mu_deficit"
        if feature_reference_lookup[_feature_id]["family"] == "suppression"
        else "mu_hat"
    )

    _mass_scale = np.asarray(
        _output[_mass_scale_field],
        dtype=float,
    ).reshape(-1)

    _band_probability = np.asarray(
        _output["band_probability"],
        dtype=float,
    ).reshape(-1)

    _width = np.asarray(
        _output["width_relative"],
        dtype=float,
    ).reshape(-1)

    _contrast = np.asarray(
        _output["contrast"],
        dtype=float,
    ).reshape(-1)

    for _values in (_mass_scale, _band_probability, _width, _contrast):
        if _values.size != number_posterior_draws:
            raise RuntimeError(
                f"A moving-band summary for {_feature_id} is misaligned."
            )

    _median_mass = float(
        np.nanmedian(_mass_scale[_bounded])
    )

    _delta_mass = np.full(number_posterior_draws, np.nan)
    _valid_mass = (
        _bounded
        & np.isfinite(_mass_scale)
        & (_mass_scale > 0.0)
    )
    _delta_mass[_valid_mass] = np.log(
        _mass_scale[_valid_mass] / _median_mass
    )

    _contrasts = {}

    for _feature_type in ("peak", "dip", "shoulder"):
        _typed_contrast = np.full(number_posterior_draws, np.nan)
        _typed_mask = (
            _bounded
            & (_realized_type == _feature_type)
            & np.isfinite(_contrast)
        )
        _typed_contrast[_typed_mask] = _contrast[_typed_mask]

        if np.count_nonzero(_typed_mask):
            _contrasts[_feature_type] = _typed_contrast

    h0_feature_statistics[_feature_id] = {
        "located": _located,
        "bounded": _bounded,
        "realized_type": _realized_type,
        "mass_scale_definition": _mass_scale_field,
        "mass_scale": _mass_scale,
        "delta_mass_scale": _delta_mass,
        "band_probability": _band_probability,
        "width": _width,
        "contrasts": _contrasts,
    }


# ------------------------------------------------------------
# Binary location and boundedness dependence.
# ------------------------------------------------------------

_presence_rows = []

for _feature_id in H0_FEATURE_ORDER[:-1]:
    _statistics = h0_feature_statistics[_feature_id]
    _located = _statistics["located"]
    _bounded = _statistics["bounded"]

    _location_metrics = _h0new_binary_dependence(
        _located
    )

    _boundedness_metrics = _h0new_binary_dependence(
        _bounded,
        eligible=_located,
    )

    _presence_rows.append(
        {
            "feature": _feature_id,
            "P_location": float(np.mean(_located)),
            "P_bounded": float(np.mean(_bounded)),
            "P_bounded_given_location": float(
                np.count_nonzero(_bounded)
                / np.count_nonzero(_located)
            ),
            "location_rank_biserial": _location_metrics[
                "rank_biserial"
            ],
            "location_permutation_probability": _location_metrics[
                "permutation_probability"
            ],
            "location_delta_median_H0": _location_metrics[
                "delta_median_H0_true_minus_false"
            ],
            "number_located": _location_metrics["number_true"],
            "number_not_located": _location_metrics["number_false"],
            "boundedness_rank_biserial_given_location": (
                _boundedness_metrics["rank_biserial"]
            ),
            "boundedness_permutation_probability": (
                _boundedness_metrics["permutation_probability"]
            ),
            "boundedness_delta_median_H0": (
                _boundedness_metrics[
                    "delta_median_H0_true_minus_false"
                ]
            ),
            "number_bounded_given_location": (
                _boundedness_metrics["number_true"]
            ),
            "number_unbounded_given_location": (
                _boundedness_metrics["number_false"]
            ),
        }
    )

h0_feature_presence_dependence = pd.DataFrame(_presence_rows)

print("\nH0 dependence of feature location support and boundedness:")
with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    240,
    "display.precision",
    4,
):
    display(h0_feature_presence_dependence)


# ------------------------------------------------------------
# Continuous conditional summaries and their dependence on H0.
# ------------------------------------------------------------

h0_feature_channel_samples = {}

for _feature_id in H0_FEATURE_ORDER:
    _statistics = h0_feature_statistics[_feature_id]

    _channels = {
        "mass scale": _statistics["mass_scale"],
    }

    if _feature_id != "m99.99":
        _channels["band probability"] = _statistics["band_probability"]
        _channels["width"] = _statistics["width"]

        for _feature_type, _values in _statistics["contrasts"].items():
            _channels[f"{_feature_type} contrast"] = _values

    h0_feature_channel_samples[_feature_id] = _channels

_continuous_tasks = [
    (_feature_id, _channel_name, _values)
    for _feature_id in H0_FEATURE_ORDER
    for _channel_name, _values in h0_feature_channel_samples[
        _feature_id
    ].items()
]

_continuous_rows = []

for _feature_id, _channel_name, _values in tqdm(
    _continuous_tasks,
    desc="H0 conditional-summary dependence",
    unit="channel",
):
    _metrics = _h0new_continuous_dependence(
        _values,
        statistic_name=f"{_feature_id}: {_channel_name}",
    )

    _continuous_rows.append(
        {
            "feature": _feature_id,
            "channel": _channel_name,
            **_metrics,
        }
    )

h0_feature_continuous_dependence = pd.DataFrame(_continuous_rows)

print("\nConditional dependence of moving-band summaries on H0:")
with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    220,
    "display.precision",
    4,
):
    display(h0_feature_continuous_dependence)


# ------------------------------------------------------------
# Minimal robustness check for mixed peak/shoulder families.
# ------------------------------------------------------------

H0_MIXED_MORPHOLOGY_FEATURES = (
    "P3",
    "S1",
    "S2",
)

_class_indicator_rows = []
_class_mass_scale_rows = []

for _feature_id in H0_MIXED_MORPHOLOGY_FEATURES:
    _statistics = h0_feature_statistics[_feature_id]
    _located = _statistics["located"]
    _bounded = _statistics["bounded"]
    _realized_type = _statistics["realized_type"]
    _mass_scale = _statistics["mass_scale"]

    _positive_class = np.isin(
        _realized_type,
        ("peak", "shoulder"),
    )
    _class_eligible = _located & _positive_class
    _is_shoulder = _realized_type == "shoulder"

    _class_metrics = _h0new_binary_dependence(
        _is_shoulder,
        eligible=_class_eligible,
    )

    _number_classified = int(np.count_nonzero(_class_eligible))
    _number_peak = int(
        np.count_nonzero(
            _class_eligible & (_realized_type == "peak")
        )
    )
    _number_shoulder = int(
        np.count_nonzero(
            _class_eligible & (_realized_type == "shoulder")
        )
    )

    _class_indicator_rows.append(
        {
            "feature": _feature_id,
            "number_location_supported": _number_classified,
            "P_peak_given_location": (
                _number_peak / _number_classified
                if _number_classified
                else np.nan
            ),
            "P_shoulder_given_location": (
                _number_shoulder / _number_classified
                if _number_classified
                else np.nan
            ),
            "shoulder_vs_peak_rank_biserial": _class_metrics[
                "rank_biserial"
            ],
            "class_permutation_probability": _class_metrics[
                "permutation_probability"
            ],
            "median_H0_shoulder_minus_peak": _class_metrics[
                "delta_median_H0_true_minus_false"
            ],
        }
    )

    for _feature_type in ("peak", "shoulder"):
        _typed_mass_scale = np.full(
            number_posterior_draws,
            np.nan,
        )
        _typed_mask = (
            _bounded
            & (_realized_type == _feature_type)
            & np.isfinite(_mass_scale)
        )
        _typed_mass_scale[_typed_mask] = _mass_scale[_typed_mask]

        _typed_metrics = _h0new_continuous_dependence(
            _typed_mass_scale,
            statistic_name=(
                f"{_feature_id}: {_feature_type}-conditional mass scale"
            ),
        )

        _class_mass_scale_rows.append(
            {
                "feature": _feature_id,
                "realized_type": _feature_type,
                **_typed_metrics,
            }
        )

h0_feature_class_indicator_dependence = pd.DataFrame(
    _class_indicator_rows
)

h0_feature_class_mass_scale_dependence = pd.DataFrame(
    _class_mass_scale_rows
)

print("\nH0 dependence of peak-versus-shoulder classification:")
with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    220,
    "display.precision",
    4,
):
    display(h0_feature_class_indicator_dependence)

print(
    "\nClass-conditional feature-"
    f"{FEATURE_SCALE_CHANNEL_LABEL.replace(' ', '-') } dependence on H0:"
)
with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    220,
    "display.precision",
    4,
):
    display(h0_feature_class_mass_scale_dependence)


# ------------------------------------------------------------
# Dominant conditional channel and compact paper-facing summary.
# ------------------------------------------------------------

_dependence_lookup = h0_feature_continuous_dependence.set_index(
    ["feature", "channel"]
)

_dominant_rows = []

for _feature_id in H0_FEATURE_ORDER:
    _feature_rows = h0_feature_continuous_dependence[
        h0_feature_continuous_dependence["feature"] == _feature_id
    ]

    _finite_rows = _feature_rows[
        np.isfinite(_feature_rows["mi_calibrated_bits"])
    ]

    if _finite_rows.empty:
        _dominant_rows.append(
            {
                "feature": _feature_id,
                "dominant_channel": "---",
                "rho_s": np.nan,
                "mi_calibrated_bits": np.nan,
                "number_samples": 0,
            }
        )
        continue

    _best_row = _finite_rows.loc[
        _finite_rows["mi_calibrated_bits"].idxmax()
    ]

    _dominant_rows.append(
        {
            "feature": _feature_id,
            "dominant_channel": _best_row["channel"],
            "rho_s": float(_best_row["rho_s"]),
            "mi_calibrated_bits": float(
                _best_row["mi_calibrated_bits"]
            ),
            "number_samples": int(_best_row["number_samples"]),
        }
    )

h0_feature_dominant_channels = pd.DataFrame(_dominant_rows)

_compact_rows = []

for _feature_id in H0_FEATURE_ORDER:
    _row = {"feature": _feature_id}

    if _feature_id == "m99.99":
        _row.update(
            {
                "P_location": np.nan,
                "P_bounded": np.nan,
                "location_rank_biserial": np.nan,
                "location_permutation_probability": np.nan,
            }
        )
    else:
        _presence_row = h0_feature_presence_dependence[
            h0_feature_presence_dependence["feature"] == _feature_id
        ].iloc[0]

        _row.update(
            {
                "P_location": float(_presence_row["P_location"]),
                "P_bounded": float(_presence_row["P_bounded"]),
                "location_rank_biserial": float(
                    _presence_row["location_rank_biserial"]
                ),
                "location_permutation_probability": float(
                    _presence_row["location_permutation_probability"]
                ),
            }
        )

    for _channel_name in (
        "mass scale",
        "band probability",
        "width",
    ):
        _key = (_feature_id, _channel_name)

        if _key in _dependence_lookup.index:
            _channel_row = _dependence_lookup.loc[_key]
            _row[f"{_channel_name}: rho_s"] = float(
                _channel_row["rho_s"]
            )
            _row[f"{_channel_name}: MI [bits]"] = float(
                _channel_row["mi_calibrated_bits"]
            )
            _row[f"{_channel_name}: N"] = int(
                _channel_row["number_samples"]
            )
        else:
            _row[f"{_channel_name}: rho_s"] = np.nan
            _row[f"{_channel_name}: MI [bits]"] = np.nan
            _row[f"{_channel_name}: N"] = 0

    _contrast_rows = h0_feature_continuous_dependence[
        (h0_feature_continuous_dependence["feature"] == _feature_id)
        & h0_feature_continuous_dependence["channel"].str.endswith(
            "contrast"
        )
        & np.isfinite(
            h0_feature_continuous_dependence["mi_calibrated_bits"]
        )
    ]

    if _contrast_rows.empty:
        _row.update(
            {
                "strongest contrast": "---",
                "strongest contrast: rho_s": np.nan,
                "strongest contrast: MI [bits]": np.nan,
                "strongest contrast: N": 0,
            }
        )
    else:
        _best_contrast = _contrast_rows.loc[
            _contrast_rows["mi_calibrated_bits"].idxmax()
        ]
        _row.update(
            {
                "strongest contrast": _best_contrast["channel"],
                "strongest contrast: rho_s": float(
                    _best_contrast["rho_s"]
                ),
                "strongest contrast: MI [bits]": float(
                    _best_contrast["mi_calibrated_bits"]
                ),
                "strongest contrast: N": int(
                    _best_contrast["number_samples"]
                ),
            }
        )

    _dominant_row = h0_feature_dominant_channels[
        h0_feature_dominant_channels["feature"] == _feature_id
    ].iloc[0]

    _row["dominant conditional channel"] = _dominant_row[
        "dominant_channel"
    ]
    _row["dominant conditional MI [bits]"] = float(
        _dominant_row["mi_calibrated_bits"]
    )

    _compact_rows.append(_row)

h0_feature_summary_table = pd.DataFrame(_compact_rows)

print("\nCompact H0-feature summary:")
with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    260,
    "display.precision",
    4,
):
    display(h0_feature_summary_table)


# ------------------------------------------------------------
# Save numerical products.
# ------------------------------------------------------------

h0_feature_presence_dependence.to_csv(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_presence_dependence_revised.csv",
    ),
    index=False,
)

h0_feature_continuous_dependence.to_csv(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_continuous_dependence_revised.csv",
    ),
    index=False,
)

h0_feature_summary_table.to_csv(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_summary_table_revised.csv",
    ),
    index=False,
)

h0_feature_class_indicator_dependence.to_csv(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_class_indicator_dependence.csv",
    ),
    index=False,
)

h0_feature_class_mass_scale_dependence.to_csv(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_class_mass_scale_dependence.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# Generate a compact LaTeX table.
#
# Presence uses rank-biserial association / permutation probability.
# Continuous cells use Spearman rho / calibrated MI in bits.
# ------------------------------------------------------------

def _h0new_continuous_table_cell(
    feature_id,
    channel_name,
    dominant_channel,
):
    key = (feature_id, channel_name)

    if key not in _dependence_lookup.index:
        return r"---"

    row = _dependence_lookup.loc[key]
    rho = float(row["rho_s"])
    mi = float(row["mi_calibrated_bits"])

    if not np.isfinite(rho) or not np.isfinite(mi):
        return r"---"

    mi_text = (
        rf"\mathbf{{{mi:.3f}}}"
        if channel_name == dominant_channel
        else f"{mi:.3f}"
    )

    return rf"${rho:+.2f}\,/\,{mi_text}$"


_latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Dependence on $H_0$ of the automatically identified "
    rf"{FEATURE_COLLECTION_LABEL} in {MASS_DENSITY_TEX_LABEL}. The location column gives "
    r"the rank-biserial association "
    r"between $H_0$ and the location-support indicator followed by its "
    r"circular-shift permutation probability, $r_{\rm rb}/p_\pi$. A dash "
    r"is shown when either the present or absent group is too small. The "
    r"continuous-summary columns give the Spearman coefficient and "
    r"permutation-calibrated mutual information in bits, $\rho_s/I$. All "
    r"continuous quantities use draw-dependent feature intervals and are "
    r"conditional on a complete bounded characterization; contrasts are "
    r"additionally conditional on the stated morphology. Bold values mark "
    r"the largest estimated conditional mutual information in each row.}",
    r"\label{tab:H0_feature_summary_dependence}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3.2pt}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Feature & Location $r_{\rm rb}/p_\pi$ "
    rf"& {FEATURE_SCALE_CHANNEL_LABEL}: $\rho_s/I$ "
    r"& $P_f$: $\rho_s/I$ "
    r"& $w_f$: $\rho_s/I$ "
    r"& Strongest contrast: $\rho_s/I$ "
    r"& Dominant conditional summary \\",
    r"\midrule",
]

for _feature_id in H0_FEATURE_ORDER:
    _dominant_channel = h0_feature_dominant_channels.loc[
        h0_feature_dominant_channels["feature"] == _feature_id,
        "dominant_channel",
    ].iloc[0]

    if _feature_id == "m99.99":
        _location_cell = r"---"
    else:
        _presence_row = h0_feature_presence_dependence[
            h0_feature_presence_dependence["feature"] == _feature_id
        ].iloc[0]

        _r_rb = float(_presence_row["location_rank_biserial"])
        _p_perm = float(
            _presence_row["location_permutation_probability"]
        )

        if np.isfinite(_r_rb) and np.isfinite(_p_perm):
            _location_cell = rf"${_r_rb:+.2f}\,/\,{_p_perm:.3f}$"
        else:
            _location_cell = r"---"

    _mass_cell = _h0new_continuous_table_cell(
        _feature_id,
        "mass scale",
        _dominant_channel,
    )

    _probability_cell = _h0new_continuous_table_cell(
        _feature_id,
        "band probability",
        _dominant_channel,
    )

    _width_cell = _h0new_continuous_table_cell(
        _feature_id,
        "width",
        _dominant_channel,
    )

    _contrast_rows = h0_feature_continuous_dependence[
        (h0_feature_continuous_dependence["feature"] == _feature_id)
        & h0_feature_continuous_dependence["channel"].str.endswith(
            "contrast"
        )
        & np.isfinite(
            h0_feature_continuous_dependence["mi_calibrated_bits"]
        )
    ]

    if _contrast_rows.empty:
        _contrast_cell = r"---"
    else:
        _best_contrast = _contrast_rows.loc[
            _contrast_rows["mi_calibrated_bits"].idxmax()
        ]
        _contrast_channel = _best_contrast["channel"]
        _contrast_label = H0_CHANNEL_SHORT_LABELS[_contrast_channel]
        _contrast_value = _h0new_continuous_table_cell(
            _feature_id,
            _contrast_channel,
            _dominant_channel,
        )
        _contrast_cell = _contrast_label + " " + _contrast_value

    _dominant_label = (
        H0_CHANNEL_SHORT_LABELS[_dominant_channel]
        if _dominant_channel in H0_CHANNEL_SHORT_LABELS
        else r"---"
    )

    _latex_lines.append(
        " & ".join(
            [
                H0_FEATURE_LABELS[_feature_id],
                _location_cell,
                _mass_cell,
                _probability_cell,
                _width_cell,
                _contrast_cell,
                _dominant_label,
            ]
        )
        + r" \\"
    )

_latex_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
)

h0_feature_summary_latex = "\n".join(_latex_lines)

print("\nLaTeX table:\n")
print(h0_feature_summary_latex)

with open(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_summary_table_revised.tex",
    ),
    "w",
) as _table_file:
    _table_file.write(h0_feature_summary_latex)

print("\nSaved numerical outputs:")
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_presence_dependence_revised.csv"
)
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_continuous_dependence_revised.csv"
)
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_summary_table_revised.csv"
)
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_summary_table_revised.tex"
)
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_class_indicator_dependence.csv"
)
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_class_mass_scale_dependence.csv"
)


# ------------------------------------------------------------
# Plot helpers used by Cells 2 and 3.
# ------------------------------------------------------------

def _h0new_smoothed_histogram_hpd(
    x,
    y,
    x_range,
    y_range,
    bins=H0_PLOT_NBINS,
    smooth_sigma=H0_PLOT_SMOOTH_SIGMA,
    probabilities=(0.50, 0.90),
):
    valid = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(valid) < 20:
        raise RuntimeError("Too few samples for a two-dimensional HPD.")

    histogram, x_edges, y_edges = np.histogram2d(
        np.asarray(x)[valid],
        np.asarray(y)[valid],
        bins=bins,
        range=(x_range, y_range),
    )

    histogram = gaussian_filter(
        histogram,
        sigma=smooth_sigma,
        mode="nearest",
    )

    histogram = np.where(
        np.isfinite(histogram) & (histogram > 0.0),
        histogram,
        0.0,
    )

    if np.sum(histogram) <= 0.0:
        raise RuntimeError("Invalid two-dimensional histogram.")

    sorted_density = np.sort(histogram.ravel())[::-1]
    cumulative = np.cumsum(sorted_density)
    cumulative /= cumulative[-1]

    thresholds = []

    for probability in probabilities:
        threshold_index = int(
            np.searchsorted(cumulative, probability, side="left")
        )
        threshold_index = min(
            threshold_index,
            sorted_density.size - 1,
        )
        thresholds.append(float(sorted_density[threshold_index]))

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    return histogram, x_centers, y_centers, thresholds


def _h0new_normal_score(values):
    """Rank-normalize finite posterior values; preserve NaNs."""

    values = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(values)
    result = np.full(values.size, np.nan)
    number_valid = int(np.count_nonzero(valid))

    if number_valid < 3:
        return result

    ranks = rankdata(values[valid], method="average")
    cumulative_probability = (ranks - 0.5) / number_valid
    result[valid] = np.asarray(
        np.sqrt(2.0)
        * _h0new_erfinv(2.0 * cumulative_probability - 1.0)
    )
    return result


# scipy.stats.norm.ppf without introducing another plotting-only dependency.
from scipy.special import erfinv as _h0new_erfinv


def _h0new_draw_hpd(
    ax,
    x,
    y,
    x_range,
    y_range,
    color,
    fill_alpha=0.20,
    linewidth_50=1.25,
    linewidth_90=0.90,
    alpha=1.0,
    fill_50=True,
    zorder=3,
):
    histogram, x_centers, y_centers, thresholds = (
        _h0new_smoothed_histogram_hpd(
            x,
            y,
            x_range=x_range,
            y_range=y_range,
        )
    )

    threshold_50, threshold_90 = thresholds
    histogram_plot = histogram.T
    maximum_density = float(np.nanmax(histogram_plot))

    X, Y = np.meshgrid(
        x_centers,
        y_centers,
        indexing="xy",
    )

    if fill_50 and maximum_density > threshold_50:
        ax.contourf(
            X,
            Y,
            histogram_plot,
            levels=[threshold_50, maximum_density * (1.0 + 1.0e-6)],
            colors=[to_rgba(color, fill_alpha)],
            antialiased=True,
            zorder=zorder,
        )

    if threshold_90 < threshold_50:
        ax.contour(
            X,
            Y,
            histogram_plot,
            levels=[threshold_90, threshold_50],
            colors=[color, color],
            linestyles=["--", "-"],
            linewidths=[linewidth_90, linewidth_50],
            alpha=alpha,
            zorder=zorder + 1,
        )

