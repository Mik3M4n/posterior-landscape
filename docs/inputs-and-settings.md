# Inputs and analysis settings

[README](../README.md) · [Output reference](outputs.md)

This guide describes the public interface in version 0.8.8. The runnable
[demo](../examples/demo.ini) and the three root-level example INIs provide
complete starting configurations. Dataset names, coordinate labels, and paths
can be adapted to the application.

## Input arrays

HDF5 (`.h5`, `.hdf5`) and NPZ (`.npz`) files contain a density array with shape
`(B, N1, N2)` and two increasing coordinate arrays. The first spatial array
dimension belongs to coordinate 1. Even a single field needs a draw dimension:
`field[None, :, :]`. The coordinate arrays must be finite, strictly increasing,
and have at least two entries; they need not have equal lengths.

Every active density value must be finite and non-negative, and each draw must
have positive integrated mass. Values outside the selected mask are ignored.
The package normalizes the draws; raw-integral diagnostics reveal discrepancies
with the declared conventions rather than serving as a hard acceptance cut.

Dataset names are configured in `[input]`:

```ini
[input]
file = density_draws.h5
density_dataset = p
coordinate1_dataset = x1
coordinate2_dataset = x2
mask_dataset = mask
weights_dataset = weights
coordinate1_name = x1
coordinate2_name = x2
coordinate1_label = $x_1$
coordinate2_label = $x_2$
domain = full
density_measure = linear
grid_location = nodes
grid_measure = linear
```

`coordinate1_name` and `coordinate2_name` label the analysis; the corresponding
`*_dataset` options select arrays in the file. Labels and optional units control
presentation. They do not rescale the input values.

### Domains and masks

| Domain | Rule |
| --- | --- |
| `full` | The entire rectangle is active. A supplied mask must be all true. |
| `mask` | Use the supplied nonempty Boolean mask. |
| `ordered` | Require active centres to satisfy `x2 <= x1`, up to floating-point roundoff. Without a mask, construct that inclusive triangle; a supplied mask may restrict it further. |
| `legacy` | Use a supplied mask, or construct the historical strict `x2 < x1` triangle. |

The historical default is `legacy`. New analyses should choose an explicit
domain. In finite-volume mode the mask identifies which whole bins are present;
it is not a prescription to intersect bins with a continuous ordering boundary.
All active diagonal bins receive full nominal weight.

### Posterior weights and aligned parameters

Optional draw weights have shape `(B,)`, are finite and non-negative, and have
positive sum. The 2D workflow normalizes them. The independent 1D workflow and
association estimators require equal weights: omit the dataset, or supply
normalized equal weights `1/B`. An all-ones array is not the safest choice for
the current 1D input path, which checks weights before the 2D normalization step.

An optional external scalar has shape `(B,)`, is finite, and is aligned with the
density draws; zero and negative parameter values are accepted. Chain labels may
accompany it. Without labels, the samples form one chain. Chain labels without
the corresponding external parameter are invalid. The 1D external-parameter
calibration currently requires chains of equal length.

## Measures, geometry, and grid placement

These are independent choices:

- `input.density_measure = linear` means density per `dx1 dx2`; `log` means
  density per `dlog_base(x1) dlog_base(x2)`.
- `analysis.feature_measure` chooses the measure of the field in which features
  are extracted and measured.
- `analysis.geometry = linear` or `log` sets the coordinates used for smoothing
  and spatial distances.
- `analysis.log_base` sets the logarithm base; use `e` for natural logarithms.
- `input.grid_location` specifies `nodes` or `cell_centres`.
- `input.grid_measure` specifies how finite cells were constructed. It has no
  numerical effect in nodal mode, although it is recorded in run signatures.

The historical default logarithm base is 10. Set it explicitly when using natural
logarithms. If the input includes `density_measure` or `log_base` attributes,
they must agree with the declared input conventions. Logarithmic input or feature
measures, log geometry, the current 1D finder, and shoulders require positive
coordinates. Finite-cell conversions involving logarithmic widths additionally
require positive physical cell edges.

### Nodes

`grid_location = nodes` is the default. Spatial integration uses the existing
separable trapezoidal weights in the selected measure. Density conversion uses
the existing pointwise Jacobian. Choosing `grid_measure = log` does not activate
finite-volume integration for nodes.

### Finite-volume cells

For `grid_location = cell_centres`, `grid_measure` is `linear` or `log`.
Interior edges are midpoints between adjacent centres in that native coordinate;
the two exterior edges extrapolate half the nearest spacing. For log-native
cells, infer the logarithmic edges first and transform those edges to obtain
physical widths. The package does not infer another set of bins by midpointing
transformed centres.

Conversion between measures `a` and `b` uses the product of axis-width ratios:

```text
F_b[i,j] = F_a[i,j] * (w1_a[i] * w2_a[j]) / (w1_b[i] * w2_b[j])
```

This preserves each cell's probability before the final numerical-safety
normalization. For natural-log native bins, log-to-linear conversion uses
`Delta ln(x) / Delta x`, rather than `1/x`. Explicit edge datasets and fractional
cell areas are not part of the current public interface.

The two PixelPop placement settings are:

```ini
[input]
grid_location = cell_centres
grid_measure = log
```

Keep the original inclusive mask, `domain = ordered`, and the density's actual
declared measure. [The PixelPop example](../settings.pixelpop.example.ini) shows
the complete configuration.

## Feature analysis

`analysis.run` accepts `2d`, `1d`, or `both`. The explicit
`one_dimensional.enabled` option controls whether the independent marginal
workflow runs. When omitted, it defaults to enabled for `1d` and `both` and
disabled for `2d`; an explicit value takes precedence. For a new 1D-only INI,
specify both `analysis.run = 1d` and `one_dimensional.enabled = true`.

| Setting | Role |
| --- | --- |
| `scales` | Non-negative smoothing scales in the selected analysis geometry; the first requested scale supplies the reference point/branch identities and measurements. |
| `persistence_threshold = auto` | Calibrate a persistence threshold separately at each scale from eligible gaps in the reference persistence spectrum. |
| Numerical `persistence_threshold` | Apply the same absolute cutoff at all requested scales. |
| `persistence_gap_min_log` | Minimum eligible natural-log gap for automatic selection; default `1` corresponds to a ratio of `e`. |
| `minimum_support` | Minimum posterior recovery for retaining reference families in the final catalogue. |
| `credible_mass` | Central interval mass for the 2D posterior summaries. |
| `support_mass` | Probability-support selection used by relevant morphology operations. |
| `matching_policy` | `coherent` by default; match extrema and saddle identities before their connections. |
| `ridge_region_policy` | `mask_supported` by default; supported boundaries can close ridge sections. |

Automatic persistence selection excludes the terminal gap and records its
selection or fallback diagnostics. Numerical thresholds bypass the automatic
rule. Final multiscale catalogues are consequently not guaranteed to be nested
when replacing automatic calibration with a numerical cutoff.

Nonzero Gaussian scales assume approximately uniform spacing in the selected
geometry. The package warns when spacings vary substantially; the convolution
scale is then approximate. This is separate from the chosen quadrature.

### Boundary behavior and historical comparisons

Peaks and ridges on the ordered diagonal are valid features. With
`mask_supported` ridge measurements, a supported boundary may close one side of
the region; widths are the supported extent, not a doubled estimate. Regions do
not extend into unsupported cells or continue across mask holes. Array limits
and the physical ordered diagonal have distinct diagnostic labels.

To compare the complete pre-0.8.8 matching and ridge-region behavior, use:

```ini
[analysis]
matching_policy = legacy
ridge_region_policy = legacy_two_sided
```

The policies are independently selectable. These compatibility controls differ
from the strictly opt-in finite-cell convention introduced in 0.8.7. Read
[COMPATIBILITY.md](../COMPATIBILITY.md) and the migration notes before comparing
historical products.

## Optional analyses

### Independent marginal finder

`one_dimensional.enabled = true` analyzes both marginals and, with a 2D run,
enables projection comparisons. Its own measure, smoothing scales, separation,
and matching tolerance are configured in `[one_dimensional]`. It smooths and
matches in logarithmic coordinates, including with a linear density measure,
and requires positive axes and equal-weight draws.

The native `q05/q50/q95` location and strength exports use central 90% intervals,
independently of the configurable 2D interval mass. Finite-cell marginalization,
conservative resampling when required, interval overlaps, and cell-edge CDFs are
activated only by `grid_location = cell_centres`.

### External scalar associations

Use `[association] enabled = auto`, `on`, or `off`, and specify `dataset`,
`chain_id_dataset`, `parameter_name`, `parameter_label`, and optionally
`parameter_unit`. Both `auto` and `on` currently warn and continue without
associations if the dataset is absent. `off` disables these calculations.
`knn`, `permutations`, `uncertainty_resamples`, and `random_seed` control the
estimators and calibration. Optional-module requirements still apply even after
input validation succeeds.

### Tails and slope-change shoulders

`ordered_tails = true` requires a support entirely contained in the ordered
domain. It supplies global tail scales and exceedance decompositions, not extra
localized feature families. `detect_shoulders = true` enables the
common-rescaling logarithmic-slope detector; its threshold is
`shoulder_alpha_threshold`.

With an explicit generic domain these modules default to off. Legacy-domain
defaults preserve older behavior. Set the options explicitly for reproducibility.

## Writing input from Python

`posterior_landscape.write_hdf5` writes the expected HDF5 layout:

```python
from posterior_landscape import write_hdf5

# p.shape == (B, len(x1), len(x2))
write_hdf5(
    "density_draws.h5", p, x1, x2,
    coordinate1_dataset="x1",
    coordinate2_dataset="x2",
    density_measure="linear",
    log_base=10.0,  # Must agree with analysis.log_base, including for linear input.
)
```

Use `mask=`, `weights=`, `external_parameter=`, and `chain_id=` when needed;
their dataset names can also be configured. The helper writes the declared
measure and log-base metadata. It does not encode all analysis choices; those
belong in the INI. The [demo generator](../examples/make_demo.py) is a complete
example.

## Execution, output, and resume

`posterior-landscape validate settings.ini --all-draws --json` performs a
read-only audit of every input draw. Without `--all-draws`, up to 16 draws are
sampled. Normalization discrepancies are reported; they are not automatically
rejected as malformed input.

The `[compute]` section controls workers, batch size, and resume. Use a fresh
output directory for changed scientific settings. Fingerprints and schema
checks prevent incompatible quadrature, matching, or region results from being
reused. `overwrite = true` explicitly allows replacing generated products.

`profile = essential` writes directly to `output.directory`. `profile = full`
writes to its `full_subdirectory` and creates an `essential/` sibling. The
essential profile retains the HDF5 draw-level arrays. See the
[output reference](outputs.md) for exact contents and workflow availability.

## Batch workflow

The repository script `run_posterior_landscape_batch.py` manages a specific
cached-density layout. It is not a general replacement for the configurable
single-run input interface. Its caches use HDF5 with fixed `p`, `m1`, `m2`, and
`mask` datasets, plus paired `h0`/`chain_id` when association samples are present.
Selection uses `requested_draws_per_chain` metadata or the supported filename
pattern. The comparison workflow assumes positive coordinates.

```bash
python run_posterior_landscape_batch.py batch_settings.ini
python run_posterior_landscape_batch.py batch_settings.ini --run-only
python run_posterior_landscape_batch.py batch_settings.ini --aggregate-only
```

`--aggregate-only` regenerates comparisons from completed results without
feature extraction. For a dry run, set `[batch] dry_run = true`; there is no
`--dry-run` flag. Required settings include `parent_directory`,
`draws_per_chain`, `template_settings`, and `output_subdirectory`; the supplied
[aggregation INI](../batch_settings.ensemble.aggregate.ini) documents that mode
and needs its paths edited. The template supplies the per-run scientific
configuration. The baseline selector for comparisons must identify one completed
run unambiguously.
