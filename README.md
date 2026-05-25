# Batch Isometric Asset Fixer

A small Streamlit tool for batch-processing isometric game assets.

It can:

- remove white / single-color / corner-detected background
- export transparent PNG
- crop transparent borders
- apply global `scale_x` and `scale_y`
- place processed assets on a fixed transparent canvas
- align center or bottom-center
- batch export with CSV log
- preview before / after
- optionally attempt OpenCV base-angle analysis
- measure output silhouette left/right isometric Y/X ratio against the target 0.5
- optionally auto-scale each figure's Y axis to align the measured average ratio to 0.5
- estimate a virtual bottom corner for base plates with chopped or flattened bottom corners

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

## Suggested settings for Norton's isometric assets

For AI-generated white background assets:

```text
Background mode: auto_corner or white
Tolerance: 20–35
Feather alpha edges: on
Crop transparent border: on
Crop padding: 20
Scale X: 1.00
Scale Y: 0.90–1.10 depending on batch
Canvas: 1024 x 1024
Alignment: bottom-center
Bottom margin: 40
```

## Notes

The OpenCV angle detection is experimental. For reliable batch work, start with manual `scale_x` / `scale_y`, preview several images, then process all.
