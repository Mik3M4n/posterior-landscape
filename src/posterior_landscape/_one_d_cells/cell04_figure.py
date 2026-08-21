# ============================================================
# Compact production figure: posterior selected density, feature bands,
# conditional feature-location intervals, and upper-tail scale.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ------------------------------------------------------------
# Validate required products.
# ------------------------------------------------------------

_required_compact_figure_products = (
    "feature_bands",
    "feature_posterior_summary_lookup",
    "pm1_q05_feature",
    "pm1_q50_feature",
    "pm1_q95_feature",
    "reference_p",
    "m_high_q05",
    "m_high_q50",
    "m_high_q95",
    "MASS_DENSITY_OUTPUT_TAG",
    "MASS_DENSITY_PLAIN_LABEL",
    "MASS_DENSITY_YLABEL",
    "COORDINATE_XLABEL",
    "COORDINATE_OUTPUT_TAG",
    "TAIL_TEX_LABEL",
)

_missing_compact_figure_products = [
    name
    for name in _required_compact_figure_products
    if name not in globals()
]

if _missing_compact_figure_products:
    raise RuntimeError(
        "Run the feature-band and draw-level posterior cells first. Missing: "
        + ", ".join(_missing_compact_figure_products)
    )


# Require the complete shoulder-transition definition.
for _shoulder_band in feature_bands["shoulder"]:
    _required_shoulder_fields = (
        "left",
        "onset_mass",
        "center",
        "right",
        "down_knee_mass",
    )

    _missing_shoulder_fields = [
        field
        for field in _required_shoulder_fields
        if field not in _shoulder_band
    ]

    if _missing_shoulder_fields:
        raise RuntimeError(
            "The shoulder bands use the old definition. Missing fields for "
            f"S{_shoulder_band.get('number', '?')}: "
            + ", ".join(_missing_shoulder_fields)
        )

    if not (
        _shoulder_band["left"]
        < _shoulder_band["onset_mass"]
        < _shoulder_band["center"]
        < _shoulder_band["right"]
    ):
        raise RuntimeError(
            "Invalid shoulder-transition ordering for "
            f"S{_shoulder_band.get('number', '?')}."
        )


# ------------------------------------------------------------
# Interpolate the smoothed posterior-median density at a mass.
# ------------------------------------------------------------

def _compact_feature_density_at_mass(mass):
    usable = (
        np.isfinite(reference_p)
        & (reference_p > 0.0)
    )

    if np.count_nonzero(usable) < 2:
        return np.nan

    return float(
        np.exp(
            np.interp(
                np.log(mass),
                log_m_grid[usable],
                np.log(reference_p[usable]),
            )
        )
    )


# ------------------------------------------------------------
# Figure and reconstructed spectrum.
# ------------------------------------------------------------

fig, mass_ax = plt.subplots(
    figsize=(8.0, 2.5),
)

fig.subplots_adjust(
    left=0.13,
    right=0.98,
    bottom=0.14,
    top=0.97,
)

mass_ax.fill_between(
    m_grid,
    pm1_q05_feature,
    pm1_q95_feature,
    color=FEATURE_FILL,
    alpha=MARGINAL_FILL_ALPHA,
    linewidth=0,
    zorder=1,
)

mass_ax.plot(
    m_grid,
    pm1_q50_feature,
    color=FEATURE_COLOR,
    lw=MARGINAL_LINEWIDTH,
    zorder=4,
)


# ------------------------------------------------------------
# Peaks: median-spectrum bands and conditional location intervals.
# ------------------------------------------------------------

for _band in feature_bands["peak"]:
    _feature_id = f"P{_band['number']}"
    _summary = feature_posterior_summary_lookup[_feature_id]

    _q05 = _summary["center_q05"]
    _q50 = _summary["center_q50"]
    _q95 = _summary["center_q95"]
    _location_y = _compact_feature_density_at_mass(_q50)

    mass_ax.axvspan(
        _band["left"],
        _band["right"],
        color=PEAK_COLOR,
        alpha=PEAK_BAND_ALPHA,
        linewidth=0,
        zorder=0,
    )

    mass_ax.errorbar(
        _q50,
        _location_y,
        xerr=np.asarray(
            [[_q50 - _q05], [_q95 - _q50]]
        ),
        fmt="o",
        ms=7,
        color=PEAK_COLOR,
        markerfacecolor=PEAK_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
        elinewidth=1.5,
        capsize=3,
        zorder=8,
    )

    mass_ax.annotate(
        rf"$P_{{{_band['number']}}}$",
        xy=(_q50, _location_y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color=PEAK_COLOR,
        zorder=9,
    )


# ------------------------------------------------------------
# Dips.
# ------------------------------------------------------------

for _band in feature_bands["dip"]:
    _feature_id = f"D{_band['number']}"
    _summary = feature_posterior_summary_lookup[_feature_id]

    _q05 = _summary["center_q05"]
    _q50 = _summary["center_q50"]
    _q95 = _summary["center_q95"]
    _location_y = _compact_feature_density_at_mass(_q50)

    mass_ax.axvspan(
        _band["left"],
        _band["right"],
        color=DIP_COLOR,
        alpha=DIP_BAND_ALPHA,
        linewidth=0,
        zorder=0,
    )

    mass_ax.errorbar(
        _q50,
        _location_y,
        xerr=np.asarray(
            [[_q50 - _q05], [_q95 - _q50]]
        ),
        fmt="v",
        ms=7,
        color=DIP_COLOR,
        markerfacecolor="white",
        markeredgecolor=DIP_COLOR,
        markeredgewidth=1.2,
        elinewidth=1.5,
        capsize=3,
        zorder=8,
    )

    mass_ax.annotate(
        rf"$D_{{{_band['number']}}}$",
        xy=(_q50, _location_y),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        color=DIP_COLOR,
        zorder=9,
    )


# ------------------------------------------------------------
# Shoulder transitions.
# ------------------------------------------------------------

for _band in feature_bands["shoulder"]:
    _feature_id = f"S{_band['number']}"
    _summary = feature_posterior_summary_lookup[_feature_id]

    _q05 = _summary["center_q05"]
    _q50 = _summary["center_q50"]
    _q95 = _summary["center_q95"]
    _location_y = _compact_feature_density_at_mass(_q50)

    mass_ax.axvspan(
        _band["left"],
        _band["right"],
        color=SHOULDER_COLOR,
        alpha=SHOULDER_BAND_ALPHA,
        linewidth=0,
        zorder=0,
    )

    mass_ax.errorbar(
        _q50,
        _location_y,
        xerr=np.asarray(
            [[_q50 - _q05], [_q95 - _q50]]
        ),
        fmt="s",
        ms=7,
        color=SHOULDER_COLOR,
        markerfacecolor=SHOULDER_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
        elinewidth=1.5,
        capsize=3,
        zorder=8,
    )

    mass_ax.annotate(
        rf"$S_{{{_band['number']}}}$",
        xy=(_q50, _location_y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color=SHOULDER_COLOR,
        zorder=9,
    )


# ------------------------------------------------------------
# Upper-tail scale.
# ------------------------------------------------------------

mass_ax.axvspan(
    m_high_q05,
    m_high_q95,
    color="0.25",
    alpha=HIGH_MASS_BAND_ALPHA,
    linewidth=0,
    zorder=0,
)

mass_ax.axvline(
    m_high_q50,
    color=M99_COLOR,
    lw=1.5,
    ls="-.",
    zorder=6,
)


# ------------------------------------------------------------
# Axes and legend.
# ------------------------------------------------------------

mass_ax.set_xscale("log")
mass_ax.set_yscale("log")

mass_ax.set_xlim(
    M_GRID_MIN,
    M_GRID_MAX,
)

mass_ax.set_ylim(
    MARGINAL_FLOOR,
    1.50 * np.nanmax(pm1_q95_feature),
)

mass_ax.set_xlabel(
    COORDINATE_XLABEL,
    fontsize=FEATURE_AXIS_LABEL_FS,
)

mass_ax.set_ylabel(
    MASS_DENSITY_YLABEL,
    fontsize=FEATURE_AXIS_LABEL_FS,
)

mass_ax.tick_params(
    which="both",
    direction="out",
    labelsize=FEATURE_TICK_LABEL_FS,
    length=4.0,
    width=0.8,
)

for _spine in mass_ax.spines.values():
    _spine.set_linewidth(0.8)

_legend_handles = [
    Line2D(
        [0],
        [0],
        color=FEATURE_COLOR,
        lw=MARGINAL_LINEWIDTH,
        label="Posterior median",
    ),
    Patch(
        facecolor=FEATURE_FILL,
        alpha=MARGINAL_FILL_ALPHA,
        edgecolor="none",
        label=r"90\% credible interval",
    ),
    Patch(
        facecolor=PEAK_COLOR,
        alpha=0.30,
        edgecolor=PEAK_COLOR,
        label="Peak",
    ),
    Patch(
        facecolor=DIP_COLOR,
        alpha=0.25,
        edgecolor=DIP_COLOR,
        label="Dip",
    ),
    Patch(
        facecolor=SHOULDER_COLOR,
        alpha=0.30,
        edgecolor=SHOULDER_COLOR,
        label="Shoulder transition",
    ),
    Line2D(
        [0],
        [0],
        color=M99_COLOR,
        lw=1.5,
        ls="-.",
        label=TAIL_TEX_LABEL,
    ),
]

mass_ax.legend(
    handles=_legend_handles,
    frameon=False,
    fontsize=FEATURE_LEGEND_FS,
    loc="upper right",
    bbox_to_anchor=(0.985, 0.985),
    borderaxespad=0,
    ncol=1,
    handlelength=2.0,
    columnspacing=1.0,
    handletextpad=0.6,
)


# ------------------------------------------------------------
# Save.
# ------------------------------------------------------------

_savefig(
    os.path.join(
        fin,
        f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_{COORDINATE_OUTPUT_TAG}_features_and_support_prod.pdf",
    )
)

plt.show()
