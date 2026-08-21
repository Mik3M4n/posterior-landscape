# posterior-landscape

`posterior-landscape` identifies morphological features in a two-dimensional
probability density. It can analyze one density or an ensemble of posterior
density draws, match features across draws, propagate their uncertainty, and
optionally associate feature summaries with any aligned scalar parameter.

The two-dimensional catalogue includes peaks, pits, ridges, valleys, and
optional slope-change shoulders. Ordered-tail summaries and independent
one-dimensional marginal analyses are separate opt-in components.

## Install and run

Python 3.11 or newer is required.

```bash
python -m pip install -e .
posterior-landscape validate settings.ini
posterior-landscape run settings.ini
```

The old command remains valid:

```bash
posterior-landscape settings.ini
```

For source-tree development, `python run.py settings.ini` is equivalent.

Start from one of the supplied files:

- `settings.example.ini`: a generic full-square synthetic analysis;
- `settings.gw.example.ini`: an explicit rendering of the ordered-mass
  analysis.

All paths in an INI file are resolved relative to that file.

## Minimal generic configuration

```ini
[input]
file = density_draws.h5
density_dataset = p
coordinate1_dataset = x1
coordinate2_dataset = x2
domain = full
density_measure = linear
coordinate1_name = x1
coordinate2_name = x2
coordinate1_label = $x_1$
coordinate2_label = $x_2$

[analysis]
run = 2d
geometry = linear
feature_measure = linear
persistence_threshold = auto
persistence_gap_min_log = 1.0
detect_shoulders = false
ordered_tails = false

[one_dimensional]
enabled = true

[association]
enabled = auto
dataset = lambda
chain_id_dataset = chain_id
parameter_name = lambda
parameter_label = $\lambda$

[output]
directory = output/synthetic
profile = essential
```

Here the 2D analysis and the requested independent 1D marginal analyses are
both run. Set `one_dimensional.enabled = false` for 2D only. Use
`analysis.run = 1d` together with `one_dimensional.enabled = true` for 1D only.

For `persistence_threshold = auto`, `persistence_gap_min_log` is the minimum
natural-log separation between adjacent eligible persistence levels. The
default `1.0` requires a ratio of at least `e`; the global extrema and the
terminal gap that would isolate one finite feature do not calibrate the cut.
If no eligible gap qualifies, the conservative v0.8 fallback is used and
recorded explicitly. A numeric persistence threshold bypasses this selection.

## Input contract

HDF5 is recommended for large ensembles; NPZ is useful for small examples.
Dataset names are configurable. With the names above, the input contains:

| Dataset | Shape | Required | Meaning |
|---|---:|:---:|---|
| `p` | `(B, N1, N2)` | yes | one non-negative density grid per draw |
| `x1` | `(N1,)` | yes | strictly increasing first coordinate |
| `x2` | `(N2,)` | yes | strictly increasing second coordinate |
| `mask` | `(N1, N2)` | for `domain=mask` | valid cells |
| `weights` | `(B,)` | no | draw weights; equal by default |
| `lambda` | `(B,)` | no | aligned external-parameter samples |
| `chain_id` | `(B,)` | no | aligned chain labels |

The array convention is `p[draw, coordinate1, coordinate2]`. The coordinate
grids may have different lengths. Every valid cell must be finite and
non-negative; values outside the mask are ignored.

`input.density_measure` states the measure of the supplied density:

- `linear`: density per `dx1 dx2`;
- `log`: density per `dlog_base(x1) dlog_base(x2)`.

`analysis.feature_measure` independently chooses the measure in which
morphology and probability summaries are computed. The package applies the
Jacobian internally and normalizes every draw in its declared input measure.
Log input, log geometry, log feature measures, 1D analysis, and directional
shoulders require positive coordinates.

The supported domains are:

- `full`: the complete rectangular grid;
- `mask`: exactly the supplied mask;
- `ordered`: every valid cell must satisfy `coordinate2 < coordinate1`; an
  optional supplied mask may further restrict that triangle;
- `legacy`: using `mask` when present and otherwise using
  `m2 < m1`.

The helper `posterior_landscape.write_hdf5(...)` writes the expected HDF5
layout. Dataset names and density measure can be supplied explicitly.

## Optional analyses

### One-dimensional marginals

The independent 1D finder is disabled unless it is explicitly requested, or
an older configuration uses `analysis.run = 1d` or `both`. It analyzes both
marginals and can compare their features with projections of the 2D feature
regions. Its current smoothing and matching coordinates are logarithmic, so
both coordinate grids must be positive. It also retains the requirement
of equal-weight posterior draws.

### Ordered tails

Set `analysis.ordered_tails = true` only for a domain entirely contained in
`coordinate2 < coordinate1`. The package then computes the configured global
upper-tail scales and the ordered `P_any`, `P_both`, and `P_straddle`
decomposition. This option is rejected on a full square or a non-ordered mask.

### Directional shoulders

Set `analysis.detect_shoulders = true` to run the existing common-rescaling
slope-change detector. It uses logarithms of both coordinates and therefore
requires positive grids. Generic configurations default to shoulders off.

### External parameter

The external scalar is no longer hard-coded to `H0`. Configure its dataset,
name, label, and optional unit in `[association]`. Samples must be finite and
aligned draw by draw with `p`; negative and zero values are allowed. When
`chain_id` is present, calibrations preserve chains. Association analysis
currently requires equal draw weights. With `enabled = auto`, absence of the
configured dataset produces a warning and the density analysis continues.

## Outputs

`output.profile = essential` keeps the compact paper-facing result:

- `features.csv`, the consolidated feature catalogue;
- `landscape.pdf` and requested 1D feature/support PDFs;
- `external_parameter_associations.csv` when applicable;
- selected 1D/2D comparison CSV files when applicable;
- `results.h5` (or the NPZ fallback), `manifest.json`, a settings copy, and
  the run log.

`output.profile = full` writes the complete diagnostic result under
`<output.directory>/<output.full_subdirectory>/` and, from that same completed
run, materializes an `essential/` sibling at `<output.directory>/essential/`.
The latter is the compact paper-facing view; no topology, posterior, or
association calculation is repeated. Draw-level numerical archives are kept
in the `full` directory and are removed only from the compact `essential`
view. Publication figures are written only as PDF and human-readable tables
only as CSV. Numerical archives and JSON metadata are retained in their
appropriate formats. The obsolete `write_png` setting is accepted so old INI
files still parse, but PNG output is not produced.

Runs are resumable. Use a new output directory when changing scientific
settings; use `overwrite = true` only when replacing generated products is
intended.

## Previous compatibility

See `COMPATIBILITY.md` for the frozen regression values and production release
gate.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Numba is optional acceleration. Install it with `pip install -e '.[speed]'`.
