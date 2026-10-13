# LVK compatibility and release gate

## 0.8.8 connection and ridge-region compatibility

The 0.8.8 defaults intentionally correct two scientific behaviors:

```ini
[analysis]
matching_policy = coherent
ridge_region_policy = mask_supported
```

These policies apply to **both nodes and cell centres**. The finite-volume
opt-in contract below is unchanged, but grid placement does not control
connection matching or ridge-region geometry. To reproduce the complete
pre-0.8.8 scientific result, retain all original settings and explicitly set:

```ini
[analysis]
matching_policy = legacy
ridge_region_policy = legacy_two_sided
```

Corrected matching preserves endpoint identities and original event arms.
Boundary membership alone cannot reject a peak or ridge: a stable diagonal
feature remains eligible, as does a feature moving a small distance onto or
off that boundary. Array-limit features are retained with separate boundary
diagnostics because an endpoint peak may reflect a genuine model cutoff,
a physical excess, or truncation of the sampled range. Coordinate tolerances,
one-to-one assignments, saddle roles, and ambiguity diagnostics remain.

Corrected ridge measurements use the region supported by the input mask.
A support boundary can terminate a section, including a one-sided diagonal
ridge. Widths describe supported extent, without doubling or exterior-density
extrapolation. Nodal regions retain nodal spatial weights; finite-cell regions
retain complete active native bins, including full ordered diagonal weights.
The continuous ordering boundary never clips an active finite cell.

The density grids, normalization, quadrature, measure conversion, smoothing,
population-density HPDs, automatic-threshold selection formula, independent
1D numerical results, and tail calculations remain unchanged for identical
input settings. Matching changes can alter morphology recovery, conditional
feature-location contours, conditional geometry, projections, and association
outputs. Newly available diagonal-region measurements need not reproduce
previous unavailable values. Existing valid interior measurement definitions
are retained. The legacy corner-pit exclusion and observed-shoulder valley
deficit policy are unchanged; this release does not redefine those policies.

Resolved policy names and separately versioned matching and ridge-region
schemas participate in signatures. Checkpoints and completed outputs missing
the required policy metadata cannot be reused as corrected results. This
also applies when requesting only downstream associations from a completed
catalogue. A shared release-family version is not a compatibility guarantee.
Use fresh output directories and inspect regenerated feature identities before
updating tables or plots; restored branches can renumber catalogue IDs.

Reference-extraction acceptance checks restore four PixelPop ridge arms forming
two paired events, and retain FullPop's four peaks, six arms, and three
paired events. The FullPop corrected reference has six valid arm regions and
three valid paired regions, compared with two and one in the legacy path.
`scripts/audit_boundary_reference.py` performs these saved-field checks and
verifies unchanged source fields and quadrature. Exported float32 projections
are checked at storage precision; float64 marginalization is checked
independently. Full posterior recovery probabilities and conditional summaries
must be established by rerunning the original density draws. Saved summaries
or median fields alone cannot certify those quantities. See
`MIGRATION-0.8.8.md` for baseline rerun instructions.

## 0.8.7 grid-placement compatibility

The 0.8.7 finite-volume change remains strictly opt-in through
`input.grid_location = cell_centres`. If that setting is absent or is
`nodes`, the 0.8.6 spatial integration and density conversion are retained.
With the two legacy policies above, the full pre-0.8.8 scientific numerical
behavior must be reproduced within the existing regression tolerances. This
includes the 2D catalogue,
posterior support, finite-region measurements, independent notebook-based
1D analyses, projections, tails, and external-parameter association outputs.
The centre-evaluated density Jacobian and nodal trapezoidal weights remain
unchanged. `input.grid_measure` has no numerical effect in `nodes` mode;
metadata, version strings, fingerprints, logs, and reuse decisions may change.

Analytic densities sampled at nodes, the spectral reconstruction, and the
synthetic validation suite therefore retain their existing default
quadrature. The generic and GW example settings explicitly use `nodes`.
Only the PixelPop finite-volume example enables `cell_centres`.

In finite-cell mode, native edges are inferred once, then transformed to
physical/logarithmic coordinates. Measure conversion preserves each cell's
probability exactly by multiplying density by the ratio of source to target
axis widths. All active mask cells have full area, including ordered diagonal
cells; no clipping, half-weighting, strict-mask substitution, or HDF5 change
is part of the conversion. Posterior-draw `weights` retain their existing
meaning.

Full nominal ordered diagonal cells can contain points on both sides of the
continuous ordering line. Finite-cell either/both/exactly-one exceedance tails
therefore use joint axis interval-overlap products, without diagonal clipping,
rather than marginal ordering identities. The primary-tail composition is
still conditional on primary-coordinate exceedance and is distinct from the
overall exactly-one tail. Nodal tail summaries retain their existing rules.

Explicit `domain = ordered` accepts `coordinate2 <= coordinate1` with a small
floating-point boundary tolerance. The historical missing-mask strict
triangle of `domain = legacy` remains unchanged.

Quadrature settings and construction metadata participate in run and cache
signatures. Incompatible completed outputs, topology checkpoints, independent
1D caches, projection comparisons, and batch skip decisions cannot be reused
after a convention change. Historical products may be recognized only as
nodal products under the appropriate compatibility checks, never as finite
cells. A new PixelPop output directory is required for migration; see
`MIGRATION-0.8.7.md`.

## Frozen settings

The supplied v0.7 GW configuration, with the two explicit legacy policies,
resolves in v0.8.8 to the same historical scientific choices:

- `p`, `m1`, `m2`, optional `mask`, `weights`, `h0`, and `chain_id`;
- legacy mask-first ordered domain;
- logarithmic input density, feature measure, and geometry with natural logs;
- scales `0`, `0.025`, and `0.05`;
- directional shoulders and ordered tails enabled;
- both independent 1D marginals enabled, with upper percentile `0.9999`;
- H0 associations with `k=5`, 100 permutations, 100 uncertainty resamples,
  and seed 1729;
- `matching_policy = legacy` and `ridge_region_policy = legacy_two_sided`.

The distributed GW example uses the corrected defaults. Select the legacy
pair when comparing it with the frozen historical catalogue below.

The intentional presentation change is PDF-only figures and the compact
output profile unless `profile=full` is selected explicitly. A full run now
also materializes the same compact view in an `essential/` sibling directory;
the topology and posterior calculations are performed only once.

## Numerical regression

The test suite runs the legacy-policy path on the deterministic ordered
posterior fixture used to freeze v0.7 behavior. It requires the same catalogue
counts and checks
representative peak, valley, and paired-ridge values, including:

- 2 peaks, 3 pits, 2 ridges, 4 valleys, 1 ridge event, and 2 valley events;
- `P2D1 = (4.1246263829, 1.7012542799)` with prominence
  `4.8267602869`;
- `V2D4 = (100, 7.0170382867)` with status
  `outer_drainage_low_density`;
- `R2DE1` median centroid `(17.8209823674, 4.3739524799)`.

The same suite covers a non-square full rectangle, a linear input and feature
measure, a signed external parameter, explicit 1D marginals, ordered submasks,
and both output profiles.

## Production release gate

The original production HDF5 files referenced by the supplied settings are
external to the distribution. Saved baseline exports support reference checks
but do not replace a complete original-draw rerun. Before tagging a production
release:

1. Run `posterior-landscape validate settings.ini` in an environment with
   `h5py` installed.
2. Check the legacy-policy run against the frozen archive in its own fresh
   output directory.
3. Run each corrected PixelPop and FullPop baseline into another fresh
   directory. For PixelPop keep logarithmic finite-cell placement, the full
   inclusive mask, and automatic persistence selection.
4. Confirm the reference-extraction targets above and compare draw-level
   recovery, conditional locations, region support, widths, probabilities,
   moments, projections, and associations. Attribute changes to matching or
   region policy using their independently selectable legacy choices.
5. Verify unchanged densities, population-density HPDs, independent marginals,
   and tails. Inspect the baseline before deriving any fixed-threshold runs.

This final real-data check is the release gate; the representative golden
regression is not a substitute for it.
