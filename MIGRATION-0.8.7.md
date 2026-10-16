# Migrating PixelPop to finite-volume quadrature

Existing nodal configurations need no numerical migration. Omitting
`grid_location`, or specifying `grid_location = nodes`, keeps the 0.8.6
scientific numerical behavior. `grid_measure` has no numerical effect in
that mode. Leave the spectral and synthetic nodal inputs unchanged.

For PixelPop, add exactly these two lines to `[input]`:

```ini
grid_location = cell_centres
grid_measure = log
```

Keep `density_measure = log`, `domain = ordered`, `mask_dataset = mask`,
`weights_dataset = weights`, and `analysis.log_base = e`. The mask selects
full finite-volume cells: all 5050 active cells, including all 100 diagonal
cells, retain their full nominal logarithmic area. Do not modify the HDF5,
change the domain to `full`, delete the diagonal, or clip diagonal areas.
`weights_dataset` still contains posterior-draw weights.

1. Install 0.8.7 in an environment with `h5py`, and copy the baseline INI
   (or use `settings.pixelpop.example.ini`). Select a **fresh** output
   directory, for example
   `output/posterior_landscape_pixelpop_reference_v1_base_cell_centres`.
   Do not point `resume = true` at previous PixelPop nodal results.
2. Keep `persistence_threshold = auto`. Run the read-only all-draw audit:

   ```bash
   posterior-landscape validate settings.pixelpop.example.ini --all-draws --json
   ```

   Check the reported number of draws, `5050` active cells, `100` active
   diagonal cells, raw-integral range/quantiles, maximum raw normalization
   error, post-normalization error, and both marginal-normalization errors.
   The diagnostic reports actual file values; the expected small input-level
   error is not hard-coded. Native edges, selected widths, and full diagonal
   weighting must also appear in the convention metadata.
3. Run **only the baseline**:

   ```bash
   posterior-landscape run settings.pixelpop.example.ini
   ```

   Inspect its normalization diagnostics, reference field, principal feature
   catalogue, and newly selected automatic persistence threshold `T0`.
4. After inspecting that baseline, derive new fixed-threshold INIs at
   `0.5 T0` and `2 T0`, each with its own fresh output directory. Do not
   reuse the old `0.050442`, `0.025221`, or `0.100884` values automatically.

Finite-cell measure conversion now uses source/target width ratios to
preserve every cell's probability before normalization. It uses the same
physical bins for input, feature, physical-area, 1D, projection, and tail
integrations. Any final conversion-normalization factor is a numerical
safeguard and should be near unity for normalized inputs. Cache signatures
identify the convention and reject incompatible reuse.

The production PixelPop HDF5 is external to the package. Its actual audit
results and the patched baseline's new `T0` must be established on the
machine where that file is available before interpreting changes to the
previous PixelPop catalogue or preparing new persistence variations.
