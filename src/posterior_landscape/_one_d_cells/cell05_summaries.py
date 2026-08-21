# ============================================================
# Paper-summary exports for the selected one-dimensional measure.
#
# Run after the draw-level feature-inference cell.  This cell does not
# rerun the finder; it only extracts compact moving-band probability and
# morphology-specific strength summaries from feature_draw_posteriors.
# ============================================================

import os
import numpy as np
import pandas as pd


FEATURE_SUMMARY_ORDER = (
    "P1",
    "D1",
    "P2",
    "D2",
    "P3",
    "S1",
    "S2",
)

_required_paper_summary_products = (
    "feature_draw_posteriors",
    "MASS_DENSITY_OUTPUT_TAG",
    "MASS_DENSITY_PLAIN_LABEL",
    "fin",
)

_missing_paper_summary_products = [
    name
    for name in _required_paper_summary_products
    if name not in globals()
]

if _missing_paper_summary_products:
    raise RuntimeError(
        "Run the draw-level feature-inference cell first. Missing: "
        + ", ".join(_missing_paper_summary_products)
    )

print(
    "Paper summaries use the 1D density measure: "
    f"{MASS_DENSITY_PLAIN_LABEL}"
)
print(f"Output tag: {MASS_DENSITY_OUTPUT_TAG}")


def _paper_summary_quantiles(values, mask):
    values = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    usable = mask & np.isfinite(values)

    if not np.any(usable):
        return 0, np.nan, np.nan, np.nan

    q05, q50, q95 = np.quantile(
        values[usable],
        [0.05, 0.50, 0.95],
    )

    return int(np.count_nonzero(usable)), q05, q50, q95


_probability_rows = []
_strength_rows = []

for _feature_id in FEATURE_SUMMARY_ORDER:
    if _feature_id not in feature_draw_posteriors:
        raise RuntimeError(
            f"Missing draw-level feature posterior for {_feature_id}."
        )

    _output = feature_draw_posteriors[_feature_id]

    _required_fields = (
        "bounded",
        "band_probability",
        "realized_type",
        "relative_prominence",
        "contrast",
    )

    _missing_fields = [
        field
        for field in _required_fields
        if field not in _output
    ]

    if _missing_fields:
        raise RuntimeError(
            f"{_feature_id} is missing: "
            + ", ".join(_missing_fields)
        )

    _bounded = np.asarray(
        _output["bounded"],
        dtype=bool,
    ).reshape(-1)

    _realized_type = np.asarray(
        _output["realized_type"],
        dtype=object,
    ).reshape(-1)

    (
        _number_probability,
        _probability_q05,
        _probability_q50,
        _probability_q95,
    ) = _paper_summary_quantiles(
        _output["band_probability"],
        _bounded,
    )

    if _number_probability == 0:
        raise RuntimeError(
            f"No bounded moving-band probabilities for {_feature_id}."
        )

    _probability_rows.append(
        {
            "feature": _feature_id,
            "number_bounded_draws": _number_probability,
            "P_f_q05": _probability_q05,
            "P_f_q50": _probability_q50,
            "P_f_q95": _probability_q95,
        }
    )

    for _morphology in ("peak", "dip", "shoulder"):
        _class_mask = (
            _bounded
            & (_realized_type == _morphology)
        )

        if _morphology in ("peak", "dip"):
            _statistic_name = (
                "A_peak"
                if _morphology == "peak"
                else "A_dip"
            )
            _statistic_values = _output["relative_prominence"]
        else:
            _statistic_name = "Delta_S"
            _statistic_values = _output["contrast"]

        (
            _number_strength,
            _strength_q05,
            _strength_q50,
            _strength_q95,
        ) = _paper_summary_quantiles(
            _statistic_values,
            _class_mask,
        )

        if _number_strength == 0:
            continue

        _strength_rows.append(
            {
                "feature": _feature_id,
                "morphology": _morphology,
                "statistic": _statistic_name,
                "number_draws": _number_strength,
                "strength_q05": _strength_q05,
                "strength_q50": _strength_q50,
                "strength_q95": _strength_q95,
            }
        )


feature_band_probability_summary = pd.DataFrame(
    _probability_rows
)

feature_relative_strength_summary = pd.DataFrame(
    _strength_rows
)

print(
    "\nMoving-band population probabilities "
    "(conditional on complete boundaries)"
)

for _row in _probability_rows:
    print(
        f"  {_row['feature']:>3s}  "
        f"{_row['P_f_q50']:.4f} "
        f"[{_row['P_f_q05']:.4f}, {_row['P_f_q95']:.4f}]  "
        f"N={_row['number_bounded_draws']}"
    )

print(
    "\nMorphology-specific strengths "
    "(conditional on complete boundaries and realized morphology)"
)

for _row in _strength_rows:
    print(
        f"  {_row['feature']:>3s}  {_row['statistic']:>7s}: "
        f"{_row['strength_q50']:.3f} "
        f"[{_row['strength_q05']:.3f}, {_row['strength_q95']:.3f}]  "
        f"N={_row['number_draws']}"
    )

_probability_summary_filename = (
    f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "pm1_feature_band_probability_summary.csv"
)

_strength_summary_filename = (
    f"fullpop_{MASS_DENSITY_OUTPUT_TAG}_"
    "pm1_feature_relative_strength_summary.csv"
)

feature_band_probability_summary.to_csv(
    os.path.join(fin, _probability_summary_filename),
    index=False,
)

feature_relative_strength_summary.to_csv(
    os.path.join(fin, _strength_summary_filename),
    index=False,
)

print("\nSaved:")
print(f"  {_probability_summary_filename}")
print(f"  {_strength_summary_filename}")

