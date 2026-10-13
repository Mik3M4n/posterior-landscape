# Changelog

## 0.8.8 — 2026-10-13

- Add `analysis.matching_policy = coherent` as the corrected default. Match
  arms through corresponding saddle and extremum identities and preserve
  both original arms of paired events. Distinguish join and split saddles
  even when they occupy the same grid index, and align event orientation
  before recording endpoint summaries.
- Remove boundary membership as an independent matching veto. Peaks and
  ridges on the ordered diagonal, at array limits, and at their intersections
  remain eligible subject to coordinate limits and consistent identities.
- Add `analysis.ridge_region_policy = mask_supported` as the corrected
  default. Allow one-sided ridge regions bounded by support, record boundary
  termination, and measure only their supported extent without doubling a
  width or inventing exterior density or contrast.
- Preserve full native active bins and diagonal weights for finite-cell
  regions. Leave nodal quadrature, conservative cell-measure conversion,
  smoothing, threshold selection, and independent 1D analysis unchanged.
- Retain explicit `matching_policy = legacy` and
  `ridge_region_policy = legacy_two_sided` for historical scientific
  reproduction. The new matching and ridge policies are independent of
  grid placement and apply to nodal and finite-cell analyses alike.
- Record matching and region policies with separately versioned schemas;
  reject incompatible topology checkpoints and completed-output reuse,
  including downstream association-only reuse.
- Keep valley shoulder/deficit definitions and the legacy corner-pit
  exclusion outside this correction. Add focused matching, boundary-region,
  finite-cell, and resume regression coverage and migration instructions.
- Require fresh PixelPop and FullPop baseline reruns to establish posterior
  recovery probabilities and conditional region summaries. Do not carry
  previous PixelPop numerical persistence thresholds into new variations.

## 0.8.7 — 2026-10-12

- Add explicit finite-volume input support with `input.grid_location =
  cell_centres` and native `input.grid_measure = linear` or `log`. Cell edges
  are inferred once in the native coordinate; all integration widths refer
  to the same physical bins.
- Preserve each finite cell's probability during density-measure conversion
  using axis-width ratios. Retain the existing centre Jacobian and
  trapezoidal quadrature in default `nodes` mode; `grid_measure` has no
  numerical effect in that mode.
- Propagate the selected quadrature through normalization, reference fields,
  support regions, finite-region moments, physical areas, marginals, the
  independent 1D workflow, projections, interval capture, and tail summaries.
- Compute finite-cell joint either/both/exactly-one tail probabilities from
  axis interval-overlap products, retaining full diagonal bins. Keep the
  primary-tail conditional composition separate from the overall
  exactly-one curve; retain the existing nodal tail calculation.
- Accept inclusive ordered masks, including floating-point boundary
  tolerance. Every active finite cell, including a diagonal cell, receives
  its full nominal area; no geometric clipping is inferred.
- Record the quadrature convention, inferred edges and weights, active and
  diagonal cell counts, and normalization diagnostics in numerical outputs
  and metadata. Add all-draw validation with
  `posterior-landscape validate settings.ini --all-draws --json`.
- Include quadrature semantics in fingerprints and resume/cache guards,
  including independent 1D, projection comparison, and batch reuse.
- Add a PixelPop baseline example and migration instructions. PixelPop must
  be rerun in a fresh output directory with automatic persistence selection;
  previous numerical thresholds are not carried forward.

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
