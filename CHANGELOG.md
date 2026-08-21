# Changelog

## 0.8.6 — 2026-08-20

- Preserve draw-level comparison and global-tail products in the full output
  profile.
- Materialize an `essential/` compact view alongside `full/` from the same
  completed run, without repeating topology or posterior calculations.

## 0.8.5 — 2026-08-20

- Excluded pits touching both the outer domain and support/mask boundaries
  from the physical reference catalogue, together with their incident valley
  branches and paired events. Single-boundary pits and their valleys remain
  available as physical transition structures.

## 0.8.4 — 2026-08-20

- Enforced the configured two-of-three multiscale requirement before the
  posterior-median reference catalogue is identified and numbered.
- Preserved each retained curve event's original hierarchy arms when
  multiscale filtering removes an unstable endpoint; surviving arms are not
  paired again.
- Prevented branch matching from substituting a boundary saddle or extremum
  for an interior hierarchy endpoint.

## 0.8.3 — 2026-08-19

- Made the automatic 2D persistence threshold insensitive to a lone dominant
  finite feature by excluding the terminal persistence gap from calibration.
- Excluded the join- and split-tree global extrema from automatic threshold
  calibration while retaining them in the catalogue.
- Added the configurable natural-log gap requirement
  `persistence_gap_min_log` (default `1.0`) and recorded the selected gap or
  conservative fallback in the existing numerical outputs, manifest, and log.
- Left explicit thresholds, 1D analysis, shoulders, plateaus, geometry,
  projections, and external-parameter associations unchanged.

## 0.8.1 — 2026-08-18

- Fixed full 1D association outputs for configured scalar parameters other
  than H0 while preserving the private legacy implementation identifiers.
- Applied configured parameter and coordinate labels to public association
  figures, tables, and combined 1D--2D diagnostics.
- Made final external-parameter filenames consistent with resume checks.
- Added a production-output regression with enough draws to exercise the
  previously skipped output cell; the v0.8.0 1D cache remains reusable.

## 0.8.0 — 2026-08-14

- Added explicit full, ordered, mask, and v0.7-compatible input domains.
- Added configurable coordinate datasets, names, labels, and units.
- Added linear or logarithmic input-density measures with internal Jacobian
  conversion to the selected feature measure.
- Generalized the aligned scalar association from hard-coded H0 to a
  configurable external parameter.
- Made independent 1D marginal analysis explicit and optional.
- Made ordered-tail summaries explicit, optional, and valid only on ordered
  domains.
- Made directional shoulders optional while retaining the v0.7 LVK algorithm.
- Added compact and full output profiles; figures are PDF and tables are CSV.
- Added a read-only `validate` command and a small public Python API.
- Added generic full-domain coverage and a numerical v0.7 regression fixture.

The production LVK density file is external to the distribution and must be
used for the final real-data release certification.
