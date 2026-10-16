# Reading the outputs

Start with `features.csv` and `landscape.pdf` for a 2D run, and the
`*feature_posterior_summary.csv` tables and `*features_and_support*.pdf`
figures for a 1D run. Use `results.h5` for draw-level 2D measurements and
spatial maps. These products answer different questions: feature support
describes how often a feature is recovered across posterior draws;
population probability describes how much density lies inside its window
within a draw.

This reference describes version 0.8.8. Optional products depend on the
selected workflow, detected features, plotting settings, ordered-tail
analysis, and availability of an aligned external parameter. Retain the
original input file: the numerical result is a derived archive and does
not contain the original ensemble of density fields.

## Output directories and profiles

With `output.profile = essential`, products are written directly to
`output.directory`. With `profile = full`, the full result is written to
`<output.directory>/<output.full_subdirectory>/`; the same completed run
also produces `<output.directory>/essential/` by copying and pruning its
products. Scientific calculations are not repeated for that view.

Both profiles deliver figures as PDF and human-readable tables as CSV;
top-level PNG, TXT, and TEX products are removed. JSON metadata and
numerical archives retain their own formats.

The essential view retains:

- The consolidated 2D `features.csv`, `landscape.pdf`, and applicable
  `external_parameter_associations.csv`.
- 1D `*feature_posterior_summary.csv` tables and
  `*features_and_support*.pdf` figures; applicable 1D external-parameter
  summary and main-dependence CSVs.
- The three main combined-workflow comparison CSVs listed below.
- `results.h5` or its reduced NPZ fallback, available manifests, 1D state
  caches, and the settings copy and log produced by the 2D workflow.

**Essential retains the full `results.h5`, including its draw-level
feature arrays.** It removes diagnostic CSVs/PDFs and selected standalone
draw archives: `global_tail_scales_draws.npz`,
`one_two_dimensional_full_external_parameter_draws.npz`, and
`one_two_dimensional_feature_family_draws.npz`. A full run retains those
archives. A 1D-only run uses its coordinate-specific state and manifests;
it does not produce the 2D HDF5 archive or consolidated 2D table.

## File map

Here `*` denotes a filename pattern, not a literal filename. The retained
files in a particular run are the authoritative inventory.

| Product | Contents | Availability |
| --- | --- | --- |
| `features.csv` | Consolidated points, individual curve arms, paired events, plateaus/floors, and shoulders, with applicable posterior intervals | 2D; both profiles |
| `landscape.pdf` | Reference density, detected structures and posterior support overview | 2D; both profiles when plotting |
| `results.h5` | 2D draw measurements, maps, quadrature and scientific metadata | 2D; both profiles |
| `manifest.json`, `settings.ini`, `run.log` | Run identity, conventions, configuration and execution record | 2D; both profiles |
| `catalogue.csv`, `posterior_intervals.csv` | Source catalogue and conditional interval columns used by the consolidated table | Full 2D |
| `ridge_valley_diagnostics.csv` | Individual arms: topology and retained endpoints, widths, extents, matching and truncation | Full 2D |
| `curve_event_catalogue.csv`, `shoulder_catalogue.csv` | Paired events and directional shoulders, with separate support levels and scalar intervals | Full 2D |
| `topology_frequencies.csv` | Feature-count signatures, weighted posterior probabilities and raw draw counts | Full 2D |
| `feature_geometry.pdf`, `shoulder_geometry.pdf` | Spatial geometry, location uncertainty and support summaries | Full 2D when applicable |
| `feature_projections.pdf`, `feature_projections_m2.pdf` | Adaptive-window contributions to each coordinate marginal | Full 2D when applicable |
| `feature_location_configurations.csv`, `feature_location_configurations.pdf` | Alternative posterior locations of a fixed family | Full 2D; figure when applicable |
| `two_dimensional_feature_cooccurrence.csv` | Coexistence of distinct families' location indicators | Full 2D |
| `external_parameter_associations.csv`, `external_parameter_corner.pdf` | Occurrence and continuous associations with the configured scalar | CSV in both profiles; figure in full |
| `global_tail_scales.csv`, `global_tail_probability_curves.csv`, `global_tail_scales_draws.npz`, `global_tail_probabilities.pdf` | Global percentile scales, tail curves and draw samples | Full, when ordered tails enabled |
| `fullpop_*_feature_posterior_summary.csv`, `fullpop_*_features_and_support_prod.pdf` | Separate 1D marginal analyses for both coordinates | Both profiles when 1D enabled |
| `one_dimensional_manifest.json`, `one_dimensional_m2_manifest.json`, `one_dimensional_state.pkl.gz`, `one_dimensional_m2_state.pkl.gz` | 1D provenance and reusable state for the two coordinates | Both profiles when 1D enabled |
| `one_dimensional_m1_feature_cooccurrence.csv`, `one_dimensional_m2_feature_cooccurrence.csv` | 1D location-indicator coexistence | Full 1D |
| `one_two_dimensional_feature_comparison.csv`, `one_two_dimensional_m2_feature_comparison.csv`, `one_two_dimensional_feature_families.csv` | Projected relationships and shared support between 1D and 2D results | Combined workflow; both profiles |
| `one_two_dimensional_manifest.json` | Combined-workflow metadata | Combined workflow |

For example, with coordinate names `m1`, `m2` and a linear 1D measure,
representative filenames include
`fullpop_dP_dm1_pm1_feature_posterior_summary.csv` and
`fullpop_dP_dm2_pm2_features_and_support_prod.pdf`. Logarithmic measures
and generic coordinate names change these tags. Additional full 1D
tables include `*feature_band_probability_summary.csv`,
`*feature_relative_strength_summary.csv`, and
`*external_parameter_feature_*dependence*.csv`.

List the exact products, including their profile subdirectories, with:

```python
from pathlib import Path

root = Path("results")  # Replace with your configured output.directory.
for path in sorted(root.rglob("*")):
    if path.is_file():
        print(path.relative_to(root))
```

## Recovery, validity and conditioning

For posterior weights `w`, the following are weighted probabilities over
density draws, without conditioning on successful matches:

| Field | Interpretation |
| --- | --- |
| `support`, `P_morph` | Successful match to the fixed catalogue morphology |
| `geometry_support` | Match and valid finite geometric measurement |
| Event/shoulder `P_loc` | Match and available retained curve/front |
| Event/shoulder `P_reg` | Match and valid adaptive region |
| `measurement_support` | Probability of the valid measurement selection used for the relevant feature |

These support levels need not coincide. A topological valley or boundary
pit can be recovered without a finite localized population measurement.
A boundary-supported ridge window may have measurable width while its
contrast remains unavailable. Inspect the validity flags and ridge
`strength_valid`/`strength_reference` metadata before using that strength.

`Pmed(region)` for a point is the median population mass of its flow basin.
`Pmed(feature)` is the median density mass inside its finite
half-prominence/half-depth window. Individual arm `Pmed(region)` is
undefined; its `Pmed(feature)` refers to its transverse window.
Plateau/floor entries use their region mass for both fields. These masses
are distinct from support and need not form a partition: adaptive windows
can overlap.

In event/shoulder tables, `region_probability` is the actual population
mass inside the adaptive window. In association and location-configuration
tables, the point statistic named `region_probability` uses the finite
point **feature** mass, not its larger flow-basin mass.
`deficit_probability` instead integrates positive missing density relative
to a defined background; it is neither occurrence support nor the actual
population mass of the window.

The consolidated table merges the source catalogues and posterior
intervals by `ID`, preserving source columns when names coincide. Its
headers replace `m1`/`m2` by `x1`/`x2`; source tables and numerical datasets
retain compatibility names. It does not append 1D or location-configuration
rows. Empty or NaN fields can be inapplicable or unmeasured, rather than
zero. `persist.` denotes retention across discovery smoothing scales,
not posterior support.

Catalogue `x1`/`x2` describe matched point/plateau positions or an arm's
**saddle**. `mu1`/`mu2` describe a finite window's weighted centroid. The
arm location interval columns inherited from `posterior_intervals.csv`
describe the retained curve endpoint; use `saddle_x1`/`saddle_x2` and
`topology_end_x1`/`topology_end_x2` for topology endpoints. These are
different location summaries.

2D `lower`, `median`, `upper` are weighted quantiles at
`(1-credible_mass)/2`, `0.5`, `(1+credible_mass)/2`, with weights
renormalized inside the selected draws:

- Point location, basin mass and prominence: matched draws. Point widths,
  feature mass and log contrast: matched draws with valid geometry.
  Centroids, covariance, projected bounds and deficit: matched draws with
  valid measurements.
- Arms: retained endpoints and curve extent require an available curve;
  widths and finite-window mass require valid geometry. Full topology
  extent and topology endpoints require a match.
- Paired events: scalar intervals require a valid adaptive region and a
  finite value. Shoulders use that rule except extent, `alpha_max`, and
  match distance/margin, which require an available front.

Missing selected draws produce NaN. `width_along_p10`/`width_along_p90`
describe variation along a curve, not posterior credible limits.
Boundary and truncation probabilities are unconditional joint
probabilities with presence. Catalogue ambiguity is likewise
`P(present and ambiguous)`; configuration-table ambiguity is conditional
on the selected configuration.

Native coordinate values and units follow the input metadata. Legacy
names containing `mass`, `m1`, `m2`, or `h0` do not impose a physical unit.
Arrays ending in `_geometry`, and curve extents, use the configured
analysis geometry, which can be logarithmic; centroids, projected bounds
and transverse widths use native coordinate values. Relative widths are
dimensionless. External-parameter samples are used as supplied without
unit conversion or a log transform. Covariance units are products of the
corresponding native units.

## Arms, paired events and projections

An individual ridge/valley arm connects a saddle to an extremum. A paired
event joins the two compatible arms at their common saddle and measures
their combined window. The `arms` field records that relationship.
`features.csv` includes both levels; counting an event and its arms as
independent discoveries double-counts the structure. Associations use
paired events rather than individual arms.

Peaks, ridge events and shoulders use density weights for centroids,
covariance and projections. Finite pits use the positive saddle deficit
`max(base_level - density, 0)`; valley events use a positive deficit under
a log-interpolated transverse side-shoulder background.

`projection_m1` integrates the selected density or deficit over coordinate
2; `projection_m2` integrates over coordinate 1. These are unnormalized
marginal contributions. Integrating against the remaining coordinate's
feature-measure quadrature gives the window population mass or deficit
integral, not one. Projection figures compare their conditional posterior
bands with the full marginal. Read `projection_weight` to identify
`density` versus `deficit`.

Point `width_major`/`width_minor` are equivalent widths from the uniform
physical-area covariance of the half-prominence/depth region. They are
not derived from the density/deficit-weighted `spatial_covariance_mass`.
Relative projected widths equal the native projected span divided by
the corresponding centroid.

## Alternative locations and coexistence

`feature_location_configurations.csv` distinguishes robust alternative
locations of one family. `A` is the most probable configuration;
subsequent letters are alternatives. Configurations are mutually
exclusive draw-level realizations. `P_configuration` is unconditional;
`P_configuration_given_region` divides it by `P_family_region`.
Continuous intervals are conditional on that configuration.

The detector checks disconnected 90% HPD components of whitened log
centroids across several bandwidths. That 90% mode-detection convention
is fixed, independently of `credible_mass`. Unimodal, unstable and
insufficient-support families remain unsplit. HDF5 labels are `-1`
outside finite positive-centroid selected draws, with configuration
indices starting at zero.

Cooccurrence tables ask whether distinct families' location indicators
are true in the same draw. They report weighted joint and conditional
probabilities, covariance, Bernoulli correlation `phi`, and
`lift = P(A and B)/(P(A)P(B))`. They do not merge catalogue labels.

## External-parameter associations

`external_parameter_associations.csv` is optional. `parameter_name`
identifies the configured scalar. `record_scope`, `parent_ID` and
`configuration` distinguish ordinary features, multimodal parent
families and individual configurations. A multimodal parent suppresses
pooled continuous tests; its configurations supply separate tests.

| Columns | Interpretation |
| --- | --- |
| `morph_rrb`, `morph_p_perm` | Rank-biserial scalar effect for morphology occurrence over all draws, and calibrated two-sided null probability |
| `loc_given_morph_*`, `reg_given_loc_*` | Location availability within morphology, and valid-region availability within location |
| `config_*` | Configuration membership within the family's valid-region draws |
| `*_n_yes`, `*_n_no` | Raw group counts used by each binary test |
| `<statistic>_rho`, `<statistic>_mi_bits`, `<statistic>_n` | Spearman correlation, calibrated KSG mutual information, and finite selected draw count |
| `mu_vector_mi_bits` | Information between the scalar and the paired centroid vector |
| `mu2_given_mu1_cmi_bits` | Conditional information between scalar and centroid 2 given centroid 1 |
| `cov_parameter_mu1`, related `cov_*`/`rho_*` | Native covariance and Spearman matrix entries for scalar and centroids |
| `mu1_coverage50_*`, `mu1_coverage80_*` | Centroid-1 tests restricted to bounded fractions of at least 0.5 or 0.8 |

A positive rank-biserial effect means higher scalar ranks in the yes
group. Null calibration circularly shifts the scalar within each chain.
MI/CMI are reported in **bits**, subtract the median null estimate and
are clipped at zero. They measure dependence and are not probabilities.
Information values from overlapping features or separately calibrated
estimators must not be added as independent contributions.

Continuous tests condition on a valid adaptive region or configuration
and finite statistic. Binary tests require at least 20 draws per group;
continuous estimators have minimum-sample and variation requirements.
NaN can therefore mean insufficient information. CMI `_lower`/`_upper`
are 5th/95th percentiles over repeated 80% subsets, not posterior credible
intervals. Association currently requires equal draw weights; the 1D
notebook calibration additionally requires equal-length chains.

## HDF5 numerical archive

`results.h5` has root `schema_version = "2.1"`. Root attributes record
package/run identity, matching and ridge-region policies, measures,
geometry, credible/support masses, threshold selection, quadrature
signature and normalization diagnostics. Inspect group attributes as
well as datasets.

| Group | Main contents |
| --- | --- |
| `/coordinates` | `x1`, `x2`; compatibility `m1`, `m2`, `log_m1`, `log_m2`; `mask`; feature, input and linear quadrature weights; inferred/native edges for cell-centred grids. Attributes hold names, labels and units. |
| `/posterior` | Draw `weights`, input/feature normalization factors, marginal quantiles and their probabilities; optional `external_parameter`, compatibility `h0`, and `chain_id`. |
| `/maps` | `reference_density`, `ridge_probability`, `valley_probability`. The curve maps give posterior probability of proximity within the configured geometry-space radius, not density or integrated window mass. |
| `/features/<ID>` | Points, individual arms and plateau/floor records: applicable draw arrays, validity flags, matching, reference geometry and spatial maps. |
| `/events/<ID>` | Paired ridge/valley records: arm IDs, support levels, validity, adaptive summaries, projections and inclusion maps. |
| `/shoulders/<ID>` | Directional fronts, thresholds, support levels, finite-region summaries and projections. |
| `/h0_associations/<ID>` | Optional association records stored as attributes under a compatibility group name. Internal field names retain `h0`. |
| `/global_tail` | Optional percentile scales, tail-curve quantiles, composition fractions, boundary flags and event-convention attributes. |

For `D` draws and grid shape `(n1, n2)`, draw scalars have shape `(D,)`,
centroids `(D, 2)`, covariance summaries `(D, 3)` ordered as
`(var1, cov12, var2)`, and projected bounds `(D, 4)` ordered as
`(lower1, upper1, lower2, upper2)`. Projections have shapes `(D, n1)` and
`(D, n2)`. Curves have a draw axis followed by sampled points and two
geometry coordinates. Maps have shape `(n1, n2)`. Feature types have
different available datasets; enumerate groups before selecting fields.

Conditional inclusion maps describe membership given the relevant valid
geometry or adaptive region;
unconditional inclusion maps include its posterior support. Location
probability/density and HPD maps summarize location uncertainty, not
population probability. Ridge boundary metadata distinguishes domain
closure from density crossings and independently records strength
validity.

The emergency `results.npz` fallback is a reduced flat archive of
coordinates, quadrature, posterior/marginal/tail summaries, maps and
metadata. It does not reproduce the HDF5 feature/event/shoulder groups.

This example discovers IDs and calculates a point centroid average only
over valid measurement draws, renormalizing posterior weights:

```python
import h5py
import numpy as np

with h5py.File("results/full/results.h5", "r") as result:
    for group_name in ("features", "events", "shoulders"):
        print(group_name, list(result[group_name]))

    points = [
        identifier for identifier, group in result["features"].items()
        if group.attrs["type"] in {"peak", "pit"}
    ]
    if points:
        identifier = points[0]  # Choose a listed point ID for your analysis.
        group = result["features"][identifier]
        centroid = group["mass_centroid"][:]
        valid = (
            group["present"][:].astype(bool)
            & group["measurement_valid"][:].astype(bool)
            & np.all(np.isfinite(centroid), axis=1)
        )
        weights = result["posterior/weights"][:]
        selected_mass = weights[valid].sum()
        print(identifier, "valid support:", selected_mass)
        if selected_mass > 0:
            selected_weights = weights[valid] / selected_mass
            print("conditional mean:", np.average(
                centroid[valid], axis=0, weights=selected_weights
            ))
```

## 1D, combined comparisons and global tails

1D summaries use fixed `q05`, `q50`, `q95` intervals. Location statistics
condition on `located`; moving endpoints, widths, band probability and
integrated scales condition on complete boundaries (`bounded`);
morphology-specific strengths additionally select the realized type.
The legacy `probability_present`/`number_present` aliases mean bounded.
The legacy `probability_peak_given_present` and analogous dip/shoulder
fields instead alias the corresponding `*_given_location` fields.
Prefer the explicit location/bounded column names.

Combined CSVs describe projected interval overlap, reference/draw
Jaccard measures, directional coverage, projection capture, common
valid-draw support and coordinate-scale relationships. One-to-many and
many-to-one relationships are allowed. The family comparison relates
both marginal coordinates to a common 2D family; optional scalar
information compares their joint and conditional dependence. Projected
overlap does not establish independent or identical physical features.

Global tail scales are marginal percentiles, not detected local
features. Read `event_convention` and definitions in `/global_tail` and
the manifest. With `full_cell_overlap`, `P_any` means either coordinate
exceeds the threshold, `P_both` means both do, and `P_straddle` means
exactly one does; cell overlap is evaluated explicitly. The legacy
`ordered_marginal` convention uses the ordered-component marginals.
Composition fractions at the first-coordinate scale are conditional on
that coordinate exceeding its draw-specific threshold. Boundary-limited
flags identify scales constrained by the represented domain.
