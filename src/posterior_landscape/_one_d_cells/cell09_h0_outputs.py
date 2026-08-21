#============================================================
# Production H0--feature outputs.
#
# This cell replaces the two previous plotting cells. It produces:
#   1. H0 versus the physical moving-band mass scale in Msun;
#   2. the main numerical/LaTeX dependence table.
#
# It deliberately does NOT produce the multi-summary contour figure.
# Run it after CELL 1 of the revised H0-dependence analysis.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


# ------------------------------------------------------------
# Validate products made by the preceding analysis cell.
# ------------------------------------------------------------

_required_main_h0_products = (
    "H0_samples",
    "H0_FEATURE_ORDER",
    "H0_FEATURE_LABELS",
    "H0_FEATURE_COLORS",
    "H0_CHANNEL_SHORT_LABELS",
    "h0_feature_statistics",
    "h0_feature_presence_dependence",
    "h0_feature_continuous_dependence",
    "_h0new_draw_hpd",
    "fin",
    "_savefig",
    "MASS_DENSITY_OUTPUT_TAG",
    "MASS_DENSITY_PLAIN_LABEL",
    "MASS_DENSITY_TEX_LABEL",
    "EXTERNAL_PARAMETER_AXIS_LABEL",
    "FEATURE_SCALE_CHANNEL_LABEL",
    "FEATURE_SCALE_AXIS_LABEL",
    "FEATURE_COLLECTION_LABEL",
)

_missing_main_h0_products = [
    name
    for name in _required_main_h0_products
    if name not in globals()
]

if _missing_main_h0_products:
    raise RuntimeError(
        "Run CELL 1 of the revised H0-feature analysis first. Missing: "
        + ", ".join(_missing_main_h0_products)
    )

print(
    "Production H0 outputs use the 1D density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL}"
)
print(f"Output tag: {MASS_DENSITY_OUTPUT_TAG}")


# ------------------------------------------------------------
# Helpers for selecting and formatting dependence estimates.
# ------------------------------------------------------------

def _h0main_continuous_row(feature_id, channel_name):
    rows = h0_feature_continuous_dependence[
        (h0_feature_continuous_dependence["feature"] == feature_id)
        & (h0_feature_continuous_dependence["channel"] == channel_name)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def _h0main_display_mi(value):
    """MI estimates below zero are finite-sample noise around zero."""

    value = float(value)

    if not np.isfinite(value):
        return np.nan

    return max(0.0, value)


def _h0main_dominant_channel(feature_id):
    rows = h0_feature_continuous_dependence[
        (h0_feature_continuous_dependence["feature"] == feature_id)
        & np.isfinite(
            h0_feature_continuous_dependence["mi_calibrated_bits"]
        )
        & (
            h0_feature_continuous_dependence["number_samples"]
            >= MI_MIN_CHANNEL_SAMPLES
        )
    ].copy()

    if rows.empty:
        return None

    rows["mi_for_display"] = np.maximum(
        0.0,
        rows["mi_calibrated_bits"].to_numpy(dtype=float),
    )

    return str(
        rows.loc[
            rows["mi_for_display"].idxmax(),
            "channel",
        ]
    )


def _h0main_strongest_contrast(feature_id):
    rows = h0_feature_continuous_dependence[
        (h0_feature_continuous_dependence["feature"] == feature_id)
        & h0_feature_continuous_dependence["channel"].str.endswith(
            "contrast"
        )
        & np.isfinite(
            h0_feature_continuous_dependence["mi_calibrated_bits"]
        )
        & (
            h0_feature_continuous_dependence["number_samples"]
            >= MI_MIN_CHANNEL_SAMPLES
        )
    ].copy()

    if rows.empty:
        return None

    rows["mi_for_display"] = np.maximum(
        0.0,
        rows["mi_calibrated_bits"].to_numpy(dtype=float),
    )

    return rows.loc[
        rows["mi_for_display"].idxmax()
    ]


def _h0main_latex_continuous_cell(
    feature_id,
    channel_name,
    dominant_channel,
):
    row = _h0main_continuous_row(
        feature_id,
        channel_name,
    )

    if row is None:
        return r"---"

    rho = float(row["rho_s"])
    mi = _h0main_display_mi(row["mi_calibrated_bits"])

    if not np.isfinite(rho) or not np.isfinite(mi):
        return r"---"

    mi_text = f"{mi:.3f}"

    if channel_name == dominant_channel:
        mi_text = rf"\mathbf{{{mi_text}}}"

    return rf"${rho:+.2f}\,/\,{mi_text}$"


def _h0main_latex_binary_cell(
    row,
    association_column,
    probability_column,
):
    association = float(row[association_column])
    probability = float(row[probability_column])

    if not (
        np.isfinite(association)
        and np.isfinite(probability)
    ):
        return r"---"

    return rf"${association:+.2f}\,/\,{probability:.3f}$"


# ------------------------------------------------------------
# Main figure: H0 versus the physical mass scale in Msun.
# ------------------------------------------------------------

_h0_plot_limits = np.quantile(
    H0_samples,
    [
        1.0 - H0_PLOT_DISPLAY_QUANTILE,
        H0_PLOT_DISPLAY_QUANTILE,
    ],
)

_h0_median = float(
    np.median(H0_samples)
)

_h0_number_panels = len(H0_FEATURE_ORDER)
_h0_number_columns = min(4, max(1, _h0_number_panels))
_h0_number_rows = int(
    np.ceil(_h0_number_panels / _h0_number_columns)
)

fig, axes = plt.subplots(
    _h0_number_rows,
    _h0_number_columns,
    figsize=(11.0, 3.05 * _h0_number_rows + 0.25),
    sharex=True,
    sharey=False,
    squeeze=False,
)

axes = axes.ravel()

for panel_index, feature_id in enumerate(H0_FEATURE_ORDER):
    ax = axes[panel_index]

    mass_scale = np.asarray(
        h0_feature_statistics[feature_id]["mass_scale"],
        dtype=float,
    )

    valid = (
        np.isfinite(H0_samples)
        & np.isfinite(mass_scale)
        & (mass_scale > 0.0)
    )

    if np.count_nonzero(valid) < 20:
        raise RuntimeError(
            f"Too few {FEATURE_SCALE_CHANNEL_LABEL} samples for {feature_id}."
        )

    mass_lower, mass_upper = np.quantile(
        mass_scale[valid],
        [
            1.0 - H0_PLOT_DISPLAY_QUANTILE,
            H0_PLOT_DISPLAY_QUANTILE,
        ],
    )

    mass_span = float(
        mass_upper - mass_lower
    )

    if not (
        np.isfinite(mass_span)
        and mass_span > 0.0
    ):
        raise RuntimeError(
            f"Invalid {FEATURE_SCALE_CHANNEL_LABEL} plotting range for "
            f"{feature_id}."
        )

    mass_padding = 0.06 * mass_span
    mass_plot_limits = (
        max(0.0, mass_lower - mass_padding),
        mass_upper + mass_padding,
    )

    _h0new_draw_hpd(
        ax,
        H0_samples[valid],
        mass_scale[valid],
        x_range=_h0_plot_limits,
        y_range=mass_plot_limits,
        color=H0_FEATURE_COLORS[feature_id],
        fill_alpha=0.18,
        fill_50=True,
    )

    median_mass_scale = float(
        np.median(mass_scale[valid])
    )

    ax.axhline(
        median_mass_scale,
        color="0.58",
        lw=0.7,
        ls=":",
        zorder=0,
    )

    ax.axvline(
        _h0_median,
        color="0.58",
        lw=0.7,
        ls=":",
        zorder=0,
    )

    mass_row = _h0main_continuous_row(
        feature_id,
        "mass scale",
    )

    if mass_row is None:
        raise RuntimeError(
            f"Missing H0 {FEATURE_SCALE_CHANNEL_LABEL} dependence for "
            f"{feature_id}."
        )

    mass_mi = _h0main_display_mi(
        mass_row["mi_calibrated_bits"]
    )

    ax.text(
        0.04,
        0.96,
        (
            rf"$\rho_s={float(mass_row['rho_s']):.2f}$"
            "\n"
            rf"$I={mass_mi:.3f}$ bits"
            "\n"
            rf"$N={int(mass_row['number_samples'])}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=H0_ANNOTATION_FS,
        color="0.20",
    )

    ax.set_title(
        H0_FEATURE_LABELS[feature_id],
        fontsize=H0_TITLE_FS,
        color=H0_FEATURE_COLORS[feature_id],
    )

    ax.set_xlim(
        _h0_plot_limits
    )

    ax.set_ylim(
        mass_plot_limits
    )

    ax.tick_params(
        which="both",
        direction="out",
        labelsize=H0_TICK_LABEL_FS,
        length=3.5,
        width=0.8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

for ax in axes[_h0_number_panels:]:
    ax.set_visible(False)

_h0_last_row_start = (
    (_h0_number_rows - 1) * _h0_number_columns
)
for ax in axes[_h0_last_row_start:_h0_number_panels]:
    ax.set_xlabel(
        EXTERNAL_PARAMETER_AXIS_LABEL,
        fontsize=H0_AXIS_LABEL_FS,
    )

fig.supylabel(
    FEATURE_SCALE_AXIS_LABEL,
    fontsize=H0_AXIS_LABEL_FS,
    x=0.008,
)

fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color="0.30",
            lw=1.25,
            ls="-",
            label=r"50\% HPD",
        ),
        Line2D(
            [0],
            [0],
            color="0.30",
            lw=0.9,
            ls="--",
            label=r"90\% HPD",
        ),
    ],
    frameon=False,
    fontsize=10,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
)

fig.tight_layout(
    rect=(0.025, 0.0, 1.0, 0.96)
)

_savefig(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_mass_scale_dependence_prod.pdf",
    )
)

plt.show()


# ------------------------------------------------------------
# Main paper table.
#
# P_loc is the probability that a compatible location is found.
# P_bnd is the probability that it also receives complete moving bounds.
# Binary cells contain rank-biserial association / permutation probability.
# Continuous cells contain Spearman rho / calibrated MI in bits.
# ------------------------------------------------------------

_main_table_rows = []

for feature_id in H0_FEATURE_ORDER:
    dominant_channel = _h0main_dominant_channel(
        feature_id
    )

    if feature_id == "m99.99":
        presence_row = None
        probability_location = np.nan
        probability_bounded = np.nan
        location_association = np.nan
        location_probability = np.nan
        bounded_association = np.nan
        bounded_probability = np.nan
    else:
        presence_row = h0_feature_presence_dependence[
            h0_feature_presence_dependence["feature"] == feature_id
        ].iloc[0]

        probability_location = float(
            presence_row["P_location"]
        )

        probability_bounded = float(
            presence_row["P_bounded"]
        )

        location_association = float(
            presence_row["location_rank_biserial"]
        )

        location_probability = float(
            presence_row["location_permutation_probability"]
        )

        bounded_association = float(
            presence_row[
                "boundedness_rank_biserial_given_location"
            ]
        )

        bounded_probability = float(
            presence_row[
                "boundedness_permutation_probability"
            ]
        )

    mass_row = _h0main_continuous_row(
        feature_id,
        "mass scale",
    )

    probability_row = _h0main_continuous_row(
        feature_id,
        "band probability",
    )

    width_row = _h0main_continuous_row(
        feature_id,
        "width",
    )

    contrast_row = _h0main_strongest_contrast(
        feature_id
    )

    _main_table_rows.append(
        {
            "feature": feature_id,
            "P_location": probability_location,
            "P_bounded": probability_bounded,
            "location r_rb": location_association,
            "location p_perm": location_probability,
            "boundedness r_rb | location": bounded_association,
            "boundedness p_perm | location": bounded_probability,
            "mass scale rho_s": (
                float(mass_row["rho_s"])
                if mass_row is not None
                else np.nan
            ),
            "mass scale MI [bits]": (
                _h0main_display_mi(
                    mass_row["mi_calibrated_bits"]
                )
                if mass_row is not None
                else np.nan
            ),
            "band probability rho_s": (
                float(probability_row["rho_s"])
                if probability_row is not None
                else np.nan
            ),
            "band probability MI [bits]": (
                _h0main_display_mi(
                    probability_row["mi_calibrated_bits"]
                )
                if probability_row is not None
                else np.nan
            ),
            "width rho_s": (
                float(width_row["rho_s"])
                if width_row is not None
                else np.nan
            ),
            "width MI [bits]": (
                _h0main_display_mi(
                    width_row["mi_calibrated_bits"]
                )
                if width_row is not None
                else np.nan
            ),
            "contrast morphology": (
                str(contrast_row["channel"])
                if contrast_row is not None
                else "---"
            ),
            "contrast rho_s": (
                float(contrast_row["rho_s"])
                if contrast_row is not None
                else np.nan
            ),
            "contrast MI [bits]": (
                _h0main_display_mi(
                    contrast_row["mi_calibrated_bits"]
                )
                if contrast_row is not None
                else np.nan
            ),
            "dominant conditional channel": (
                dominant_channel
                if dominant_channel is not None
                else "---"
            ),
        }
    )

h0_feature_main_table = pd.DataFrame(
    _main_table_rows
)

print("\nMain H0-feature table:")

with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    260,
    "display.precision",
    4,
):
    display(h0_feature_main_table)

h0_feature_main_table.to_csv(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_dependence_main.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# LaTeX version of the main table.
# ------------------------------------------------------------

_main_latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Dependence on $H_0$ of the automatically identified "
    rf"{FEATURE_COLLECTION_LABEL} in {MASS_DENSITY_TEX_LABEL}. The second "
    r"column gives the probability that a "
    r"compatible feature location is found and the probability that it also "
    r"admits complete moving boundaries, $P_{\rm loc}/P_{\rm bnd}$. "
    r"The next two columns report the rank-biserial association with "
    r"location support and, conditional on location, with successful "
    r"boundary construction, followed by the corresponding circular-shift "
    r"permutation probability, $r_{\rm rb}/p_\pi$. Dashes indicate that "
    r"one binary group is too small for a stable comparison. Continuous "
    r"columns give the Spearman coefficient and permutation-calibrated "
    r"mutual information in bits, $\rho_s/I$, for the draw-dependent "
    rf"feature {FEATURE_SCALE_CHANNEL_LABEL}, band probability, relative width, and "
    r"strongest morphology-specific contrast. They are conditional on a "
    r"complete bounded characterization; contrasts are additionally "
    r"conditional on the stated morphology. Small negative calibrated "
    r"mutual-information estimates, which are numerical fluctuations "
    r"around zero, are displayed as zero. Bold values mark the largest "
    r"estimated conditional mutual information in each row.}",
    r"\label{tab:H0_feature_summary_dependence}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2.7pt}",
    r"\begin{tabular}{lccccccc}",
    r"\toprule",
    r"Feature "
    r"& $P_{\rm loc}/P_{\rm bnd}$ "
    r"& Location $r_{\rm rb}/p_\pi$ "
    r"& Bounds $r_{\rm rb}/p_\pi$ "
    rf"& {FEATURE_SCALE_CHANNEL_LABEL.capitalize()}: $\rho_s/I$ "
    r"& $P_f$: $\rho_s/I$ "
    r"& $w_f$: $\rho_s/I$ "
    r"& Contrast: $\rho_s/I$ \\",
    r"\midrule",
]

for feature_id in H0_FEATURE_ORDER:
    dominant_channel = _h0main_dominant_channel(
        feature_id
    )

    if feature_id == "m99.99":
        location_probability_cell = r"---"
        location_cell = r"---"
        bounds_cell = r"---"
    else:
        presence_row = h0_feature_presence_dependence[
            h0_feature_presence_dependence["feature"] == feature_id
        ].iloc[0]

        location_probability_cell = (
            rf"${float(presence_row['P_location']):.3f}"
            rf"\,/\,{float(presence_row['P_bounded']):.3f}$"
        )

        location_cell = _h0main_latex_binary_cell(
            presence_row,
            "location_rank_biserial",
            "location_permutation_probability",
        )

        bounds_cell = _h0main_latex_binary_cell(
            presence_row,
            "boundedness_rank_biserial_given_location",
            "boundedness_permutation_probability",
        )

    mass_cell = _h0main_latex_continuous_cell(
        feature_id,
        "mass scale",
        dominant_channel,
    )

    probability_cell = _h0main_latex_continuous_cell(
        feature_id,
        "band probability",
        dominant_channel,
    )

    width_cell = _h0main_latex_continuous_cell(
        feature_id,
        "width",
        dominant_channel,
    )

    contrast_row = _h0main_strongest_contrast(
        feature_id
    )

    if contrast_row is None:
        contrast_cell = r"---"
    else:
        contrast_channel = str(
            contrast_row["channel"]
        )

        contrast_cell = (
            H0_CHANNEL_SHORT_LABELS[contrast_channel]
            + " "
            + _h0main_latex_continuous_cell(
                feature_id,
                contrast_channel,
                dominant_channel,
            )
        )

    _main_latex_lines.append(
        " & ".join(
            [
                H0_FEATURE_LABELS[feature_id],
                location_probability_cell,
                location_cell,
                bounds_cell,
                mass_cell,
                probability_cell,
                width_cell,
                contrast_cell,
            ]
        )
        + r" \\"
    )

_main_latex_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
)

h0_feature_main_table_latex = "\n".join(
    _main_latex_lines
)

print("\nMain LaTeX table:\n")
print(h0_feature_main_table_latex)

with open(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
        "H0_feature_dependence_main.tex",
    ),
    "w",
) as table_file:
    table_file.write(
        h0_feature_main_table_latex
    )

print("\nSaved production outputs:")
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_mass_scale_dependence_prod.pdf"
)
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_dependence_main.csv"
)
print(
    f"  fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "H0_feature_dependence_main.tex"
)
