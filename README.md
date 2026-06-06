# Batch Isometric Asset Fixer

A small Streamlit tool for batch-processing isometric game assets.

It can:

- remove white / single-color / corner-detected background
- export transparent PNG
- crop transparent borders
- remove white matte / defringe semi-transparent edges
- bleed visible edge colors into transparent pixels to prevent GPU sampling halos
- resize RGBA with premultiplied-alpha math
- apply global `scale_x` and `scale_y`
- place processed assets on a fixed transparent canvas
- align center or bottom-center
- batch export with CSV log
- recursively process a whole raw asset folder while preserving subfolders
- preview before / after
- optionally attempt OpenCV base-angle analysis
- measure output silhouette left/right isometric Y/X ratio against the target 0.5
- optionally auto-scale each figure's Y axis to align the measured average ratio to 0.5
- estimate a virtual bottom corner for base plates with chopped or flattened bottom corners

## Download

Desktop builds are published on GitHub Releases:

- macOS: `BatchIsoFixer-macOS.zip`
- Windows: `BatchIsoFixer-Windows.zip`

The packaged app starts a local Streamlit server and opens the interface in your
browser. It creates a working folder at:

```text
~/BatchIsoFixer/
```

Put source files into `~/BatchIsoFixer/input/` or
`~/BatchIsoFixer/input/raw/`, then export to `~/BatchIsoFixer/output/`.

## Quick start

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Desktop packaging

Install PyInstaller and build the app locally:

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --clean --noconfirm build.spec
```

The GitHub release workflow builds both platforms from a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

That workflow uploads the macOS and Windows zip files to the GitHub Release.
The website in `docs/` links to the latest release assets.

## Folder workflow

Put source images into:

```text
input/
```

Run the app, tune settings, then click **Process all images**.

Output will be written to:

```text
output/fixed/
```

Each image becomes:

```text
original_name_fixed.png
```

A log will be generated:

```text
output/fixed/process_log.csv
```

The log includes `iso_right_abs_ratio` and `iso_left_abs_ratio`. For a 1:2
isometric base, both values should be close to `0.5`.

If **Align figure to 1:2 isometric scale** is enabled, the log also includes
`iso_pre_avg_ratio`, `iso_auto_scale_y`, `iso_avg_ratio`, and
`iso_align_skipped_reason`. Files are skipped when the measured left/right sides
differ more than the configured maximum.

For bases with a chopped or flattened bottom corner, keep **Use virtual bottom
corner for chopped bases** enabled. The app fits the two lower base edges and
uses their intersection as the measurement corner without changing the visible
shape of the asset. The summary column `iso_measurement_method` shows whether
the row used `physical-bottom` or `virtual-bottom`.

Settings are saved to:

```text
output/fixed/settings.json
```

## Recursive raw folder workflow

For a full model folder with nested categories, put the source tree under:

```text
input/raw/
```

Then use **Recursive Raw Folder Export** and click **Process raw folder tree**.
Processed PNGs will be written under:

```text
output/modified/
```

The subfolder structure is preserved. For example:

```text
input/raw/government/4x4/university4-01.png
-> output/modified/government/4x4/university4-01.png
```

Non-PNG inputs are also exported as `.png` because the processing pipeline writes
transparent RGBA assets.

## Suggested settings for Norton's isometric assets

For AI-generated white background assets:

```text
Background mode: auto_corner or white
Tolerance: 20–35
Feather alpha edges: on
Remove white matte / defringe: on
RGB bleed pixels: 2–4
Crop transparent border: on
Crop padding: 20
Scale X: 1.00
Scale Y: 0.90–1.10 depending on batch
Canvas: 1024 x 1024
Alignment: bottom-center
Bottom margin: 40
```

## Avoiding white halos in game

White jagged edges usually come from RGB data around transparent pixels, not
only from the alpha channel. If a PNG has semi-transparent white pixels on the
building outline, or fully transparent pixels whose hidden RGB is white, game
scaling/filtering can sample that white and show a halo.

This tool now processes assets in this order:

```text
source image
-> remove background / alpha mask
-> remove white matte from semi-transparent edges
-> bleed visible edge RGB into transparent pixels
-> crop
-> premultiplied-alpha resize
-> bleed RGB again
-> place on final transparent canvas
-> bleed RGB again
-> PNG
```

Use the **Halo debug backgrounds** preview to check the output on dark, green,
and gray backgrounds before exporting a full batch. If the edge still looks too
bright, try `RGB bleed pixels: 3` or `4`; if tiny details start to smear, lower
it back to `2`.

## Notes

The OpenCV angle detection is experimental. For reliable batch work, start with manual `scale_x` / `scale_y`, preview several images, then process all.
