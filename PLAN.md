# Measure It implementation plan

This plan records the actual state of the xis.ai Computer Vision assessment. Git history
begins when this existing work is imported; it does not reconstruct earlier development.

## Completed: camera calibration

- Inspected 39 original itel S24 camera JPEGs in `Callibration_Img/Callibrate`.
- Verified 3968 x 2976 resolution and a checkerboard with 7 x 9 inner corners.
- Used the measured 24 mm square size, full-board detections, and subpixel refinement.
- Calibrated using all 39 successful views, with no failed detections or outlier removal.
- Saved camera parameters, per-image errors, reports, and local undistortion examples.
- Achieved 2.957892646 px RMS at original resolution, equivalent to 1.192698648 px
  at the previous 1600 x 1200 scale. Previous WhatsApp RMS was 1.199105136 px.
- Preserved previous results in `calibration_previous_whatsapp/`.

## Completed: dataset preparation

- Audited 77 valid original sponge images in `Object/`; none were unreadable.
- Recorded 40 landscape images (3968 x 2976) and 37 portrait images (2976 x 3968).
- Saved `dataset/split_manifest.csv` before annotation or training.
- Used the user-approved 54 train / 15 validation / 8 test counts with fixed seed 42.
- Grouped suspected related images before assigning the splits.
- Checked all 2,926 pairs: 0 exact duplicates, 63 visual candidate pairs, and 20
  close-capture pairs, comprising 75 distinct flagged pairs. None cross the saved splits.
- Recorded the current calibration path and hash for subsequent undistortion.

## Next: resolve preprocessing and annotate manually

1. Review candidate pairs and difficult images at full resolution. Keep the manifest fixed.
2. Establish the portrait-to-calibration rotation mapping; EXIF values do not determine it.
3. Generate undistorted derivatives outside the original folders, using the current
   original-resolution calibration, and save the resulting coordinate transforms/matrices.
4. Define the manual polygon and corner annotation convention for the green top surface.
   Exclude yellow side faces and background; record uncertain or occluded boundaries.
5. Annotate and review each eligible image in its final undistorted coordinate space.
6. Preserve the measured 93 mm x 74 mm top dimensions as ground truth. Specify a metric
   plane/pose/scale method before implementing measurement; calibration alone is insufficient.

## Later: training and evaluation (not started)

- Select the segmentation/model approach after annotations are ready and training is authorized.
- Apply augmentation to training data only; use validation data for tuning.
- Define evaluation metrics and acceptance tolerances before final test evaluation.
- Report segmentation and dimensional measurement errors on the held-out test images.
- Avoid using known target dimensions to force outputs and then calling them independent
  measurement accuracy. Document the limitations of a small, single-sponge capture session.

## Repository scope

Version source code, configuration, compact calibration parameters, CSV manifests,
machine-readable reports, and documentation. Keep original photos, generated media,
model weights, environments, temporary files, and secrets local and ignored.
