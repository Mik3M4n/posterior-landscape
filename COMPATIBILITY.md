# LVK compatibility and release gate

## 0.8.7 grid-placement compatibility

Finite-volume behavior is strictly opt-in through
`input.grid_location = cell_centres`. If that setting is absent or is
`nodes`, the 0.8.6 scientific numerical behavior must be reproduced within
the existing regression tolerances. This includes the 2D catalogue,
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

The supplied v0.7 GW configuration resolves in v0.8 to the same scientific
choices:

- `p`, `m1`, `m2`, optional `mask`, `weights`, `h0`, and `chain_id`;
- legacy mask-first ordered domain;
- logarithmic input density, feature measure, and geometry with natural logs;
- scales `0`, `0.025`, and `0.05`;
- directional shoulders and ordered tails enabled;
- both independent 1D marginals enabled, with upper percentile `0.9999`;
- H0 associations with `k=5`, 100 permutations, 100 uncertainty resamples,
  and seed 1729.

The intentional presentation change is PDF-only figures and the compact
output profile unless `profile=full` is selected explicitly. A full run now
also materializes the same compact view in an `essential/` sibling directory;
the topology and posterior calculations are performed only once.

## Numerical regression

The test suite runs v0.8 on the deterministic ordered posterior fixture used
to freeze v0.7 behavior. It requires the same catalogue counts and checks
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

The GWTC-5 HDF5 file referenced by the supplied settings is external and was
not available in the patch environment. Before tagging a production release:

1. Run `posterior-landscape validate settings.ini` in an environment with
   `h5py` installed.
2. Run v0.8 into a new output directory; do not overwrite the v0.7 result.
3. Compare the numerical feature catalogue and draw-level summaries with the
   v0.7 archive before comparing figures or filenames.
4. Confirm that only the intended output-layout changes remain.

This final real-data check is the release gate; the representative golden
regression is not a substitute for it.
