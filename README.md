# Measure It: camera calibration

## Assessment status and repository

This is the existing xis.ai **Measure It** Computer Vision assessment project, now
versioned on branch **`dev-test01`**. Work was completed locally before Git setup; the
repository imports that work without inventing earlier development history.

| Stage | Actual status |
| --- | --- |
| Original-image camera calibration | Completed: 39 images, all detected |
| Object dataset | 77 valid original images |
| Reproducible split | 54 train / 15 validation / 8 test, seed 42 |
| Duplicate audit | Completed; 0 exact duplicates and 0 flagged cross-split pairs |
| Object undistortion | Pending portrait-orientation mapping and preprocessing |
| Manual annotation | Not completed |
| Model training and final evaluation | Not started |

The object is a **green-yellow sponge**. The target is the **green top surface only**,
with user-measured dimensions **93 mm x 74 mm**. Yellow side faces and background are
outside that target. The camera is the same itel S24 used for calibration.

See [PLAN.md](PLAN.md), [OBJECTIVES.md](OBJECTIVES.md), and
[DEFINITION_OF_DONE.md](DEFINITION_OF_DONE.md) for completed work and remaining steps.
See [dataset/README.md](dataset/README.md) for the audit and manual annotation guidance.
The saved [split manifest](dataset/split_manifest.csv) is the assignment authority.

### Repository contents and local-only data

Git includes the Python scripts, dependency versions, documentation, split CSV, audit
reports, configuration, and both small `camera_params.npz` calibration files. Original
calibration/object photos, generated images, undistorted derivatives, model weights,
virtual environments, caches, temporary files, and secrets are excluded by `.gitignore`.
Ignored files remain on the local machine. Images referenced by reports are not available
from a clone; supply the original folders locally to reproduce image-processing runs.
Generated reports retain paths and hashes from the actual `E:\Computer_Vision` run.

To reproduce dataset preparation with the approved 77-image split, after installing
`requirements.txt` and restoring the local image folders:

```powershell
python prepare_dataset.py --train 54 --validation 15 --test 8 --seed 42
```

This command audits images and writes manifests/reports; it does not label or train.
Do not rerun it casually after training begins, because the recorded split must remain fixed.
The 37 portrait object images require a validated rotation into calibration coordinates
before subsequent undistortion. The configuration pins the current calibration by hash.

## Current calibration: original camera images

The active `calibration/camera_params.npz` now comes from the 39 original JPEGs in
`Callibration_Img/Callibrate`, all **3968 x 2976 pixels**. Inspection confirmed **7 x 9
inner corners**, and calibration uses **24 mm** squares with subpixel refinement.
All **39/39** images succeeded; **0** failed and none were excluded.

| Metric | Original-image result |
| --- | ---: |
| RMS reprojection error | 2.957892646 px |
| Mean Euclidean error | 2.309072 px |
| Per-image RMS range | 0.828736867 to 5.720435759 px |
| RMS expressed at the old 1600 x 1200 scale | 1.192698648 px |
| Previous WhatsApp RMS | 1.199105136 px |
| Resolution-normalized change | 0.534272% lower |

The originals have 2.48 times the previous width and height. The normalized fit is very
similar; the larger native-pixel RMS does not by itself indicate worse calibration.
This comparison is a change of pixel units, not independent physical-accuracy validation.
Visible paper creases may contribute to residuals. Original JPEG files were read only;
their hashes were checked unchanged after calibration.

Camera matrix (pixels):

```text
[[2810.549551515,    0.000000000, 1980.389397169],
 [   0.000000000, 2811.223867885, 1480.131259476],
 [   0.000000000,    0.000000000,    1.000000000]]
```

Distortion `(k1, k2, p1, p2, k3)`:
`[0.131131359, -0.609566886, 0.001128378, -0.001109389, 0.708897693]`.

The entire previous output folder is preserved in `calibration_previous_whatsapp`.
Current per-image errors are in `calibration/per_image_errors.csv`; full reports and
comparison are in `calibration/calibration_report.json` and
`calibration/comparison_with_previous.txt`. Two before/after examples are saved as
`calibration/undistortion_comparison.jpg` and `calibration/undistortion_comparison_2.jpg`,
with full-resolution PNG pairs alongside them. They use `alpha=1` without cropping.

To reproduce the current calibration from PowerShell (dependencies below):

```powershell
Set-Location E:\Computer_Vision
python inspect_images.py --images Callibration_Img/Callibrate --output calibration_originals
python calibrate.py --images Callibration_Img/Callibrate --pattern 7 9 --square-size-mm 24
python compare_calibrations.py
```

These commands overwrite only current generated outputs. The preserved WhatsApp archive
is not overwritten. The initial archival run additionally used
`--archive-existing calibration_previous_whatsapp`; do not repeat that option with an
existing archive. `calibrate.py` and `inspect_images.py` now default to the original-image
folder. The existing `undistort.py` loads the new parameters automatically and expects
3968 x 2976 input unless an explicit compatible resize is allowed.

## Previous WhatsApp run documentation (historical)

The remaining results and example filenames below describe the previous run. Its output
paths now reside under `calibration_previous_whatsapp/` instead of `calibration/`. Use the
commands above for the active original-image run.

Python/OpenCV camera calibration only. Inputs in `calibration_images` are never modified.

## Inspected checkerboard

The photographs were inspected before choosing a pattern size. The first photograph
(`WhatsApp Image 2026-11-06 at 9.46.11 PM.jpeg`) shows **8 squares across and 10 down**,
so there are **7 x 9 inner corners**, or **63 intersections**. The last photograph shows
the same board rotated in-plane. The contact sheet covers all 39 photos; the annotated
detections confirm the complete 63-point grid in every image. Pattern dimensions refer
to board axes, so rotating the board within an image does not change the configured grid.

The **24 mm square side** is the user's measured input; it cannot be inferred in physical
units from the images alone. Object points are `(column * 24, row * 24, 0)` in millimeters.
All 39 files decode at **1600 x 1200 pixels**. EXIF auto-rotation is disabled consistently
for calibration and undistortion, preserving encoded pixel coordinates.

## Reproduce

Tested with Python 3.12.10, OpenCV 4.12.0, and NumPy 2.2.6. From PowerShell:

```powershell
Set-Location E:\Computer_Vision
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe inspect_images.py
.\.venv\Scripts\python.exe calibrate.py --pattern 7 9 --square-size-mm 24
```

An existing environment with these dependencies can use `python` directly. No downloads
of image data are needed. File enumeration is sorted, all detections use native image
resolution, OpenCV uses one thread and a fixed RNG seed, and the report records versions,
solver settings, and each input's SHA-256 hash. Small numerical differences across OpenCV
builds/platforms are possible. Re-running replaces generated artifacts at the output path;
use `--output calibration_trial` for a separate run.

## Method

1. Detect the full grid with `findChessboardCorners`; retry failures using the sector-based
   `findChessboardCornersSB` detector.
2. Explicitly refine each successful detection using `cornerSubPix` with an 11 x 11
   half-window, up to 100 iterations and 0.001 pixel stopping tolerance.
3. Use all successful views with `calibrateCamera`, a planar grid in mm, and the standard
   five-coefficient pinhole distortion model `(k1, k2, p1, p2, k3)`. No fixed intrinsics,
   rational/fisheye model, or outlier rejection is used. Mixed image dimensions are rejected.
4. Reproject each object point using its estimated pose, independently calculate errors,
   and verify agreement with OpenCV's RMS.
5. Reload the saved NPZ and generate an undistortion example from the median per-view-RMS
   image, using `getOptimalNewCameraMatrix` and `undistort` with `alpha=1`, without cropping.

Reference: [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).

## Actual results

| Measurement | Result |
| --- | ---: |
| Input images | 39 |
| Successful full-board detections | 39 |
| Failed detections | 0 |
| Unreadable images | 0 |
| Images used in calibration | 39 |
| Corners per image | 63 |
| Total observed corners | 2457 |
| RMS reprojection error | 1.199105 px |
| Mean Euclidean reprojection error | 0.938503 px |
| Per-image RMS range | 0.350064–2.301904 px |

The RMS is `sqrt(sum(dx**2 + dy**2) / number_of_points)`. Mean Euclidean error is
`mean(sqrt(dx**2 + dy**2))`. These are distinct metrics, and both use original image pixels.

Camera matrix in pixels (rounded here; full precision is in the NPZ):

```text
[[1135.74586,    0.00000, 798.579868],
 [   0.00000, 1135.98106, 596.883874],
 [   0.00000,    0.00000,   1.000000]]
```

Distortion coefficients, in `(k1, k2, p1, p2, k3)` order:

```text
[0.13964769, -0.65094055, 0.00119158, -0.00108110, 0.76100769]
```

This is an in-sample fit, not a held-out measurement accuracy result, and a pixel error
cannot directly be read as a millimeter error. The board's paper has visible creases;
any nonplanarity violates the calibration's flat-board assumption and may contribute to
residual error. The filenames indicate WhatsApp handling and the supplied resolution is
1600 x 1200; their compression/resizing history cannot be confirmed from filenames alone.
These parameters describe the supplied image geometry, not automatically the phone's
original full-resolution sensor output. For more accurate physical measurements, use a
rigid flat target, original photos, and consistent lens/zoom/focus settings, then validate
against independent known measurements. No accuracy threshold was supplied, so this run
does not label the calibration as passing a measurement tolerance.

## Undistort a new image

```powershell
python E:\Computer_Vision\undistort.py "E:\Computer_Vision\new_image.jpg" --output "E:\Computer_Vision\undistorted\new_image.png"
```

The default parameter path is relative to the script, independent of the current working
directory. `--params` selects another NPZ. The output must differ from the input image
and parameter file. Default `--alpha 1` retains the field of view and may introduce black
borders. `--alpha 0` minimizes black borders; add `--crop` to crop to the valid-pixel ROI.
A JSON sidecar stores the output camera matrix (adjusted for any crop), output size, and
ROI. Undistorted pixels use this new camera matrix and zero distortion, not the original
distorted-image matrix.

New images must use the same camera lens, zoom, focus regime, crop, processing, and encoded
orientation. Different dimensions are rejected by default. `--allow-resize` explicitly
allows uniform rescaling with the same aspect ratio and scales the intrinsic matrix;
it cannot compensate for cropping, rotation, different camera modes, or a different lens.

## Files

- `inspect_images.py`: inventory and contact sheet.
- `calibration_utils.py`: checkerboard geometry, detection/refinement, reprojection metrics.
- `calibrate.py`: calibration CLI and report generation.
- `undistort.py`: standalone undistortion CLI and parameter loading.
- `calibration/camera_params.npz`: `camera_matrix`, `dist_coeffs`, `image_size` (width,
  height), `pattern_size`, `square_size_mm`, `rms_reprojection_error_px`, `rvecs`, `tvecs`
  (mm), `object_points`, `image_points`, `image_names`, and `per_view_rms_px`.
- `calibration/calibration_report.json`: full precision results, per-image errors, input
  hashes, detector choice, settings, example metadata, and limitations.
- `calibration/detection_report.json`: detection statuses, including any failed filenames.
- `calibration/results.txt`: readable numeric results.
- `calibration/image_inventory.json` and `contact_sheet.jpg`: inspection evidence.
- `calibration/detections/`: 39 images with refined checkerboard corners drawn.
- `calibration/example_before.png`, `example_after.png`: full-resolution example images.
- `calibration/undistortion_comparison.jpg`: labeled side-by-side preview.

The example source is `WhatsApp Image 2026-11-06 at 9.46.36 PM (1).jpeg`. Undistortion
corrects lens distortion; it does not flatten the perspective of the photographed board.
