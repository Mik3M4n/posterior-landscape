# Migrating connection matching and boundary-supported ridges

Version 0.8.8 corrects connection matching and ridge-region measurements.
Unlike the 0.8.7 finite-volume change, these corrections apply to **both
nodal and cell-centred inputs**. They do not change input normalization,
quadrature, density-measure conversion, smoothing, independent 1D finding,
or the automatic persistence-threshold formula.

## Corrected baseline

Copy the original baseline INI, keep its scientific settings, and make the
new analysis policies explicit:

```ini
[analysis]
matching_policy = coherent
ridge_region_policy = mask_supported
persistence_threshold = auto
```

Choose a fresh output directory, for example:

```ini
[output]
directory = output/fullpop_base_v088
overwrite = false
```

For PixelPop use a different directory such as
`output/pixelpop_base_v088`. Retain these existing finite-volume settings:

```ini
[input]
grid_location = cell_centres
grid_measure = log
```

Keep `density_measure = log`, `domain = ordered`, the inclusive original
mask, and the posterior-draw weights. All 5050 PixelPop active cells,
including the 100 diagonal cells, retain their full nominal weights. Do not
modify the original HDF5. FullPop's nodal input remains `grid_location =
nodes`; `grid_measure` has no numerical effect in that mode.

For example, with the original baseline INI next to the copied file:

```bash
cp fullpop_base.ini fullpop_base_v088.ini
# Edit the copied INI to set the policies and fresh output directory above.
posterior-landscape validate fullpop_base_v088.ini --all-draws --json
posterior-landscape run fullpop_base_v088.ini
```

INI paths are resolved relative to the INI file. Keep the copy in the same
directory or adjust its relative input paths. Run these commands where the
original density draws and `h5py` are available; saved median fields and
catalogues do not contain enough information for a posterior rerun.

## What to inspect

Coherent matching identifies the extrema and saddle of an arm before matching
the connection. Paired events retain their two original arms. Boundary labels
remain diagnostic and do not automatically reject a feature or match.
The PixelPop reference target is four ridge arms forming two paired events;
the FullPop target is retention of four peaks, six arms, and three paired
events. These targets concern reference extraction, not fixed posterior
recovery percentages.

You can audit saved baseline exports before the original-draw rerun:

```bash
python scripts/audit_boundary_reference.py pixelpop_base/full/results.h5 \
  --expected-peaks 3 --expected-arms 4 --expected-events 2 \
  --output pixelpop_reference_audit.json
python scripts/audit_boundary_reference.py fullpop_base/full/results.h5 \
  --expected-peaks 4 --expected-arms 6 --expected-events 3 \
  --output fullpop_reference_audit.json
```

Replace the example paths with the old baseline export paths. The script
reads their adjacent `settings.ini`; use `--settings` for a different location.
It only re-extracts reference catalogues from the saved median density and
compares legacy, matching-only, and corrected policies. It leaves the exports
unchanged. Input-normalization diagnostics in its report are stored values,
not a new all-draw integration. Exported float32 projections are checked at
storage precision, separately from float64 marginal integration checks.

The tested exports satisfy the reference targets above. PixelPop changes
from two retained arms and no paired event to four arms and two paired events.
FullPop retains six arms and three events; its valid reference arm regions
increase from two to six and valid paired regions from one to three. These
counts do not determine posterior morphology or region-recovery probabilities,
which still require the original draws.

The supported-region policy can make previously unavailable diagonal-ridge
measurements valid. A side may end at the domain support instead of a density
crossing. Widths describe the actual supported extent, are not doubled, and
no exterior density or contrast is invented. Finite cells remain the original
full active bins. Nodal integration remains nodal. The legacy corner-pit
exclusion and valley deficit's observed-shoulder requirement are unchanged.

Inspect morphology and region recovery separately, together with conditional
locations, widths, probabilities, moments, projected weights, and any
association outputs. Matching changes may increase or decrease recovery and
change the matched draw set. Unchanged source fields should reproduce their
normalization, population-density HPDs, independent 1D results, and tails.
Some feature IDs can renumber when missing arms return: compare endpoint
identities and coordinates before editing paper tables or plotting labels.

First rerun **only the baseline** and inspect its automatically selected
`T0`. The selection formula is unchanged, but record its measured value for
the new run. Do not automatically reuse PixelPop's old `0.050442` or the old
fixed values `0.025221` and `0.100884`. Derive `0.5 T0` and `2 T0` variants
only after reviewing the baseline, each in a fresh output directory.

## Historical reproduction and controlled comparisons

For complete pre-0.8.8 scientific reproduction retain all original settings
and use:

```ini
[analysis]
matching_policy = legacy
ridge_region_policy = legacy_two_sided
```

The two policies are independently selectable. Compare legacy matching with
corrected regions, and corrected matching with legacy regions, in separate
directories if you need to isolate their effects. A nodal placement by itself
does not restore historical matching or region semantics.

The policy names and separate schema versions are stored in signatures and
metadata: `matching_schema_version = 2.0` and
`ridge_region_schema_version = 2.0` in this release. Draw-level ridge metadata
include `boundary_bounded_fraction`, `strength_valid`, and
`strength_reference`, plus arm `width_reference`. Their reference attributes
use the `reference_` prefix. Checkpoints and completed outputs with
incompatible or missing corrected-policy metadata cannot supply a corrected run, including an
association-only rerun. Do not direct resume at an old output directory; keep
historical products intact and generate new baseline products.
