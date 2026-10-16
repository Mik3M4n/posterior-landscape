The two `nodal_086_*_science.npz` files are numerical golden results from the
unmodified `posterior-landscape 6.zip` attachment, package version 0.8.6.
The ordered relative paths plus bytes of all original package Python source
files have SHA-256 `f77cf5d18501b34a5e2ed0039aeb6cef5e347343c04b20cc39ecf55203f97833`.
They were generated with `tests/test_nodal_opt_in.py::_write_fixture()` and
the original package CLI, once with logarithmic and once with linear 2D/1D
feature measures. Each run used a fresh output directory. The inputs are 40
deterministic density draws on an inclusive, ordered 41 by 41 geometric grid,
with aligned H0 values and two chains. The original default nodal quadrature
normalizes the input. The fixtures intentionally include nonzero smoothing,
point/curve finite geometry, independently fitted marginals, tail summaries,
projections and parameter associations.

`_snapshot()` records every numerical HDF5 dataset and attribute, numeric
nested notebook state, exported NPZ array, and scientific CSV field. Metadata
such as version, source path, run signatures and logs is not included. New
metadata can be added without regenerating these scientific expectations.
The snapshot also retains CSV identifiers and classifications, so a changed
catalogue cannot pass by matching only numerical arrays.

The golden should only be regenerated against the original 0.8.6 source,
never against the implementation under test. New tests compare omitted grid
placement, explicit `nodes` with `grid_measure = linear`, and explicit `nodes`
with `grid_measure = log`. All scientific values must match at `rtol=1e-12`,
`atol=1e-14` (including NaN locations).
