<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/logo_exports/posterior_landscape_wordmark.svg">
  <img src="assets/brand/logo_exports/posterior_landscape_wordmark_on_light.svg" width="640" alt="posterior-landscape">
</picture>

**Morphological features and uncertainty in posterior density landscapes.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Version 0.8.8](https://img.shields.io/badge/version-0.8.8-60dfce?style=flat-square&labelColor=111326)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-f1b06c?style=flat-square&labelColor=111326)](LICENSE)

[Quickstart](#quickstart) · [Inputs](#inputs) · [Outputs](#outputs) · [Output reference](docs/outputs.md) · [Citation](#citation)

</div>

---

`posterior-landscape` turns a two-dimensional density, or an ensemble of posterior
density draws, into a catalogue of **peaks, pits, ridges, valleys, plateaus, and
slope-change shoulders**, with the classes selected by the analysis settings.
It identifies corresponding features across draws and reports their recovery,
locations, finite measurement regions, and uncertainty.

The interface supports generic coordinates and ordered component-mass
distributions. Independent marginal analysis, joint-to-marginal comparisons,
ordered-tail summaries, and associations with an aligned external parameter are
available as optional analyses.

## What you get

| Question | Package result |
| --- | --- |
| Where is the structure? | A reference catalogue of extrema, connecting branches, and optional slope-change fronts. |
| How consistently is it recovered? | Draw-weighted recovery probabilities, location intervals, and matching diagnostics. |
| What is its extent? | Automatically measured regions, probabilities, centroids, covariance, widths, and contrast, with validity flags. |
| How does it appear in a marginal? | Independent 1D catalogues compared with projections of the measured 2D regions. |
| Is it associated with another parameter? | Conditional rank correlations and mutual-information estimates for aligned posterior samples. |
| Can I inspect the calculation? | CSV tables, PDF figures, draw-level numerical arrays, saved settings, and a diagnostic manifest. |

The input is an already reconstructed density ensemble. Its declared measure,
grid placement, domain, and analysis settings determine the meaning of the
catalogue and its measurements.

## Quickstart

### Install from the repository

Requires **Python 3.11 or newer**. From a checkout of this repository:

```bash
python -m pip install -e .
posterior-landscape --version
```

NumPy, SciPy, h5py, Matplotlib, and pandas are installed as dependencies. Optional
Numba acceleration is available with `python -m pip install -e '.[speed]'`.

### Run the included example

The example generates 64 illustrative density draws on a small rectangular grid,
then runs the 2D and independent marginal analyses. No external data are needed.

```bash
python examples/make_demo.py
posterior-landscape validate examples/demo.ini --all-draws
posterior-landscape run examples/demo.ini
```

Open `output/demo/essential/landscape.pdf` and
`output/demo/essential/features.csv` first. Complete reports are in
`output/demo/full/`. The perturbed Gaussian mixtures illustrate the interface;
they are not posterior samples from a scientific inference.

### Run your own ensemble

Copy an example INI, set the input file and output directory, and run:

```bash
posterior-landscape validate settings.ini --all-draws --json
posterior-landscape run settings.ini
```

Validation checks the input contract and reports normalization diagnostics
without feature extraction. `--all-draws` audits every draw; without it, a sample
is checked. Inspect the raw integrals even when validation succeeds: the analysis
normalizes each draw, so validation does not require that the original values
integrated to one.

The shorter `posterior-landscape settings.ini` command remains supported.
`python run.py settings.ini` and `python -m posterior_landscape run settings.ini`
are alternatives. Paths in an analysis INI are relative to that INI file.

| Starting point | Use |
| --- | --- |
| [examples/demo.ini](examples/demo.ini) | A complete, runnable synthetic example. |
| [settings.example.ini](settings.example.ini) | Generic positive coordinates on a rectangular nodal grid. |
| [settings.gw.example.ini](settings.gw.example.ini) | An ordered component-mass analysis on nodes. |
| [settings.pixelpop.example.ini](settings.pixelpop.example.ini) | An ordered finite-volume field with logarithmic cell centres. |

## Inputs

The package reads **HDF5** (`.h5`, `.hdf5`) or **NumPy NPZ** (`.npz`). HDF5 is
recommended for large ensembles. Dataset names are configurable; the coordinate
names below are examples.

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `p` | `(B, N1, N2)` | Non-negative densities; array order is draw, coordinate 1, coordinate 2. |
| `x1`, `x2` | `(N1,)`, `(N2,)` | Finite, strictly increasing coordinates, each with at least two entries. |
| `mask` | `(N1, N2)` | Boolean support; required for `domain = mask`, optional for `ordered`. |
| `weights` | `(B,)` | Optional posterior-draw weights for 2D analysis, never spatial integration weights. |
| External parameter | `(B,)` | Optional finite scalar samples aligned with the density draws. |
| `chain_id` | `(B,)` | Optional chain labels accompanying the external parameter. |

A single density is represented by `p = field[None, :, :]`; raw 2D input is not
accepted. Grid axes may have different lengths. Active values must be finite
and non-negative, with positive integrated mass in every draw. Values outside
the mask are ignored.

The 2D workflow supports nonuniform non-negative draw weights. For independent
1D analysis and parameter associations, use equal-weight draws: omit `weights`
or provide normalized equal weights. Positive coordinates are required for
logarithmic measures or geometry, the independent 1D finder, and shoulders.

### Specify the scientific conventions

| Setting | Specifies |
| --- | --- |
| `input.density_measure` | Density per `dx1 dx2` or per `dlog_base(x1) dlog_base(x2)`. |
| `analysis.feature_measure` | The density measure used to identify and measure features. |
| `analysis.geometry` | Linear or logarithmic coordinates for spatial geometry and smoothing. |
| `input.grid_location` | Coordinates supplied as `nodes` or finite-volume `cell_centres`. |
| `input.grid_measure` | Linear or logarithmic native coordinates for constructing finite cells; no numerical effect for nodes. |
| `analysis.log_base` | The logarithm base; use `e` for natural logarithms. |

**Nodes are the default.** They use separable trapezoidal quadrature and the
existing pointwise Jacobian for density-measure conversion. Explicit
`cell_centres` mode infers native edges, uses full cell widths, and converts
measures through width ratios that preserve each cell's probability. Linear
and logarithmic widths describe the same physical bins.

For a PixelPop-style field supplied at logarithmic cell centres:

```ini
[input]
grid_location = cell_centres
grid_measure = log
density_measure = log
domain = ordered

[analysis]
log_base = e
```

In `ordered` mode, active centres satisfy `x2 <= x1`, including the diagonal.
Every active finite-volume cell receives its full nominal weight, without
automatic half-weighting or clipping against the ordering line.

Use `domain = full` for a complete rectangle, `mask` for supplied support, and
`ordered` for the inclusive ordered domain with an optional restrictive mask.
Historical `legacy` mode uses a supplied mask or falls back to strict `x2 < x1`.
Prefer an explicit domain in new configurations.

See [the input and settings guide](docs/inputs-and-settings.md) for the complete
contract, measure conversion, optional analyses, and compatibility choices.

## Outputs

### Start with three files

For a completed 2D analysis:

1. **`landscape.pdf`** shows the reference density and detected structure.
2. **`features.csv`** combines the retained feature families, their recovery,
   posterior intervals, and class-specific measurements.
3. **`manifest.json`** records completion, resolved conventions, the selected
   threshold, feature counts, normalization diagnostics, and available files.

Then open **`results.h5`** for the draw-level measurements and masks needed for
custom plots or post-processing. Keep the original input: the result archive
does not reproduce the complete density-draw file.

### Choose a report profile

```ini
[output]
directory = output/my_analysis
profile = full
full_subdirectory = full
```

| Profile | Location | Contents |
| --- | --- | --- |
| `essential` | Directly in `output.directory` | Main feature and marginal tables, selected comparisons, overview PDFs, metadata, and the numerical archive. |
| `full` | `output.directory/full/` by default, plus `output.directory/essential/` | All diagnostic reports, with the essential view copied from the same completed analysis. |

**`essential` selects fewer reports; it retains the draw-level arrays in
`results.h5`.** Additional NPZ exports and diagnostic tables/figures are kept
only in `full`. Creating the essential view does not repeat the analysis.

Figures are PDF and human-readable tables are CSV. JSON stores metadata; HDF5,
NPZ, and state files store numerical products. PNG generation is not part of the
analysis output, even though older INIs may still contain `write_png`.

### What is in each output family?

| Output | Contents and use |
| --- | --- |
| `features.csv` | Combined catalogue: reference identities, recovery, posterior location and measurement intervals, and finite-region summaries. Columns are class-dependent. |
| `catalogue.csv`, `posterior_intervals.csv` | Full-profile source tables separating the point/branch catalogue from its posterior intervals. |
| `curve_event_catalogue.csv` | Complete ridge or valley events formed from two arms sharing a saddle. Use for the recovery of an entire connection. |
| `shoulder_catalogue.csv` | Optional slope-change fronts with region and slope measurements. |
| `feature_location_configurations.csv` | Separately resolved configurations of a feature's conditional centroid distribution, when available. |
| `results.h5` | Coordinates and quadrature, reference maps, posterior normalization, draw-level features, presence and validity masks, regions, curves, projections, and optional tails/associations. |
| `*feature_posterior_summary.csv` | Independent marginal families, location intervals, and recovery/classification probabilities. |
| `*feature_relative_strength_summary.csv` | Full-profile marginal strengths and their measurement availability. |
| `one_two_dimensional*comparison.csv`, `one_two_dimensional_feature_families.csv` | Relations between independent marginal features and projected 2D measurements, including overlaps and capture fractions. |
| `external_parameter_associations.csv` | Optional feature-specific correlations and mutual information, with sample and calibration diagnostics. |
| `two_dimensional_feature_cooccurrence.csv`, `one_dimensional_*_feature_cooccurrence.csv` | Full-profile co-recovery probabilities and dependence summaries. |
| `global_tail_scales.csv`, `global_tail_probability_curves.csv` | Optional ordered upper-tail scales and probabilities; global summaries rather than localized features. |
| `settings.ini`, `manifest.json`, `run.log` | Saved configuration, metadata, file inventory, and execution record for the 2D run. |

Only requested analyses produce their corresponding files. A 1D-only run has
its own marginal manifests and state files; it does not create the 2D
`results.h5`, `manifest.json`, or `landscape.pdf`. Marginal filenames retain
historical prefixes such as `fullpop_` even for generic inputs; these do not
identify the population model used.

### Read the probabilities correctly

| Quantity | Interpretation |
| --- | --- |
| **Feature recovery** | The posterior draw-weight fraction in which the corresponding reference feature is recovered. |
| **Measurement support** | The draw-weight fraction in which that feature is recovered and the requested measurement is valid. |
| **Region or feature probability** | Population probability integrated over a region within a density draw; distinct from feature recovery. |

Point/arm tables use fields such as `location_support`, `geometry_support`, and
`measurement_support`; event and shoulder rows also use `P_morph`, `P_loc`, and
`P_reg`. Their masks and aliases are explained in the
[output reference](docs/outputs.md#recovery-validity-and-conditioning).

For points, `Pmed(region)` describes the topological component or basin, while
`Pmed(feature)` describes the finite half-prominence/depth window. A pit or
valley's integrated **deficit** also differs from the population probability
inside its window. Inspect the quantity and feature class before comparing
numbers.

Location intervals are conditional on recovery. Widths, strengths, centroids,
and other finite measurements use their applicable validity masks. The 2D
interval mass follows `analysis.credible_mass`; native marginal `q05/q50/q95`
columns use central 90% intervals. Missing or inapplicable values are not zero.

Catalogue coordinates (`x1`, `x2`) summarize matched point locations or arm
saddles; posterior curve-endpoint intervals and finite-region centroids
(`mu1`, `mu2`) describe other location summaries. IDs are
local to a run: compare physical structure and connectivity across settings,
rather than relying on IDs alone. A paired ridge and its two arms are related
descriptions of one connection, not three independent structures.

### Inspect a result in Python

```python
from pathlib import Path
import h5py
import pandas as pd

run_dir = Path("output/demo/full")
catalogue = pd.read_csv(run_dir / "features.csv")
print(catalogue.filter(items=[
    "ID", "type", "x1", "x2",
    "location_support", "measurement_support", "P_loc", "P_reg",
]))

with h5py.File(run_dir / "results.h5", "r") as result:
    print("Feature groups:", list(result["features"]))
    print("Paired events:", list(result["events"]))
    print("Grid:", result["coordinates/x1"].shape,
          result["coordinates/x2"].shape)
```

For a full file map, HDF5 groups, valid-sample examples, figure contents,
projection definitions, and legacy field names, see
**[Understanding the outputs](docs/outputs.md)**.

## Optional analyses

### Independent marginals and projections

```ini
[one_dimensional]
enabled = true
measure = log
scales = 0.05, 0.075, 0.10
```

This analyzes both marginals independently and, with a 2D run, compares them
with projections of measured joint regions. Enable it explicitly with
`analysis.run = 2d`, or use `analysis.run = both`. For 1D only, set
`analysis.run = 1d` and `one_dimensional.enabled = true`.

The current finder smooths and matches in logarithmic coordinates even when
its density measure is linear, so both axes must be positive. It requires
equal-weight draws. Finite-cell marginalization, conservative resampling, and
cell-edge cumulative probabilities are activated only by explicit
`grid_location = cell_centres`.

### Association with an external parameter

```ini
[association]
enabled = auto
dataset = lambda
chain_id_dataset = chain_id
parameter_name = lambda
parameter_label = $\lambda$
```

Samples must align with the density draws; they may be positive, zero, or
negative. Without chain IDs they are treated as one chain. Associations require
equal draw weights; the 1D chain-based calibration also requires equal-length
chains. An absent parameter dataset produces a warning and leaves density
analysis available.

Correlations and information describe **posterior association** in each
measurement's valid sample. Information is in bits; values from different
features are not additive contributions, independent constraints, or effects of
removing those features from an inference.

### Ordered tails and slope changes

`analysis.ordered_tails = true` enables global upper-tail summaries on a support
contained in the ordered domain. `analysis.detect_shoulders = true` enables
common-rescaling logarithmic-slope changes. Both modules are optional. See
[the settings guide](docs/inputs-and-settings.md#optional-analyses).

## Reproducibility and compatibility

Keep the input, INI, and numerical outputs together. Metadata records the
package and schema versions, threshold selection, quadrature and normalization,
matching and region policies, and file inventory. Compatible runs can resume;
use a fresh output directory when changing scientific settings.
`overwrite = true` deliberately replaces generated products.

Version 0.8.8 defaults to `matching_policy = coherent` and
`ridge_region_policy = mask_supported`: connections preserve endpoint identities
and ridge regions can close against the supported domain, including the ordered
diagonal. Nodal placement alone does not restore pre-0.8.8 matching and region
behavior; use the explicit legacy policies for that comparison.

- [Changelog](CHANGELOG.md)
- [Compatibility and regression notes](COMPATIBILITY.md)
- [Finite-cell migration: 0.8.7](MIGRATION-0.8.7.md)
- [Matching and boundary-region migration: 0.8.8](MIGRATION-0.8.8.md)
- [Read-only reference audit](scripts/audit_boundary_reference.py)

Nonzero Gaussian smoothing scales are approximate on nonuniform grids; inspect
the spacing warnings. The independent marginal finder has its own resampling
convention. Automatic thresholds are calibrated separately at each scale;
numerical thresholds apply a common cutoff. Their multiscale catalogues need
not be nested.

## Development and batch workflows

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

A synthetic robustness runner and a batch driver for cached density ensembles
are included. Start with `python run_posterior_landscape_batch.py --help` and
[batch workflow notes](docs/inputs-and-settings.md#batch-workflow). The supplied
[aggregation example](batch_settings.ensemble.aggregate.ini) needs its paths
edited for your machine.

## Citation

Software metadata is in [CITATION.cff](CITATION.cff). Please cite the accompanying
methods paper when using the package in research, and report the package version
and scientific configuration alongside the results.

## Contributing

Bug reports and focused contributions are welcome. For numerical issues,
include the version, a minimal configuration, relevant manifest/log entries,
and a small reproducible input when it can be shared. Run the tests before
submitting changes to scientific calculations.

## License and branding

The code is released under the [MIT license](LICENSE). The supplied logo,
wordmark, and their reproducible notebook are in
[assets/brand](assets/brand/README.md).
