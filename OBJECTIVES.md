# Assessment objectives

Build a reproducible computer-vision workflow for the xis.ai "Measure It" assessment,
using an itel S24 phone and a green-yellow sponge.

## Target and ground truth

- Object: green-yellow sponge.
- Segmentation and measurement target: **green top surface only**.
- Measured top dimensions: **93 mm x 74 mm**, supplied by the user.
- Yellow side faces, background, and cast shadows are outside the top-surface target.
- Shadows falling on the green top do not change which surface is being annotated.

## Completed objectives

1. Determine camera intrinsics and distortion from original checkerboard photos using
   Python/OpenCV, measured 24 mm squares, and verified 7 x 9 inner corners.
2. Record transparent calibration quality metrics and retain the previous calibration.
3. Inspect the source dataset, preserve originals, and freeze a reproducible split manifest.
4. Audit duplicate and near-duplicate risk and prevent flagged groups crossing splits.

## Remaining objectives

1. Resolve portrait orientation and produce consistent undistorted image coordinates.
2. Manually annotate and review the green top surface and any required corner landmarks.
3. Establish a defensible conversion from image geometry to metric dimensions.
4. Train and evaluate a model only after annotation and explicit authorization to train.
5. Report measured segmentation and dimensional errors without claiming unsupported accuracy.

## Reproducibility constraints

- Dataset: 77 valid images, fixed seed 42, 54 train / 15 validation / 8 test.
- `dataset/split_manifest.csv` is the assignment authority; derived images inherit it.
- `dataset/dataset_config.json` identifies the current calibration and its SHA-256 hash.
- Preserve originals and existing results. Keep raw photos and secrets out of Git.
- Do not claim independent-session or unseen-object generalization from this dataset.

Annotation, model training, and final assessment evaluation are **not completed**.
