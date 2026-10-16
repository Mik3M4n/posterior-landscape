# Brand assets

The supplied icon and wordmark are available in SVG, PNG, and PDF in
[`logo_exports/`](logo_exports/). The README uses the two wordmark variants
selected through a standard HTML `picture` element.

| Asset stem | Intended use |
| --- | --- |
| `posterior_landscape_wordmark_on_light` | Dark lettering on a transparent canvas for light pages |
| `posterior_landscape_wordmark` | Light lettering on a dark background for dark pages |
| `posterior_landscape_logo` | Icon on its rounded dark tile |
| `posterior_landscape_logo_transparent` | Icon without the tile |

Use SVG for the repository and other scalable displays, PDF for print, and
PNG when a raster image is needed.

## Regenerate the assets

Open [`posterior_landscape_logo.ipynb`](posterior_landscape_logo.ipynb) in
Jupyter from this directory and run all cells. The notebook requires
Matplotlib and IPython; it reads no result files. Colours and export resolution
are editable in its settings cell. It writes the twelve named exports to
`logo_exports/`.

The original vector geometry and the supplied exports are preserved.
The wordmark uses outlined DejaVu Sans lettering, so viewers do not need
the font installed.
