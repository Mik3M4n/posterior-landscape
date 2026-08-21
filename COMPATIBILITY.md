# LVK compatibility and release gate

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
