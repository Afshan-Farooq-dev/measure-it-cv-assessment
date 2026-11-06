# Definition of done

Checked items are supported by existing project artifacts. Unchecked items remain work
to be completed; this document does not declare the full assessment finished.

## Camera calibration

- [x] Inspect all 39 original calibration images and confirm 3968 x 2976 resolution.
- [x] Verify the 7 x 9 inner-corner pattern and use 24 mm squares.
- [x] Detect and refine corners; report 39 successes and 0 failures.
- [x] Save intrinsic matrix and distortion coefficients in `calibration/camera_params.npz`.
- [x] Report RMS and per-image errors and compare against the previous calibration.
- [x] Provide an undistortion script and generate local before/after examples.
- [x] Preserve previous results in `calibration_previous_whatsapp/`.

## Dataset preparation

- [x] Decode and inventory all 77 object images without modifying originals.
- [x] Record landscape/portrait resolutions and identify the orientation issue.
- [x] Save the 54/15/8 split with seed 42 before any model training.
- [x] Record exact and near-duplicate checks and cross-split leakage flags.
- [x] Keep all 75 identified candidate pairs within their respective splits.
- [x] Record original-file hashes and the original-resolution calibration reference.

## Annotation and preprocessing

- [ ] Resolve and validate the portrait rotation mapping into calibration coordinates.
- [ ] Generate and inspect undistorted object-image derivatives.
- [ ] Define and apply a consistent green-top polygon/corner annotation convention.
- [ ] Review annotation quality, occlusion/truncation, and coordinate consistency.
- [ ] Preserve split assignments and record 93 mm x 74 mm ground truth separately.

## Model, measurement, and evaluation

- [ ] Define the physical scale/pose/plane method and evaluation metrics/tolerances.
- [ ] Obtain authorization to train and complete a reproducible training run.
- [ ] Tune only on training/validation data; keep final test evaluation held out.
- [ ] Report actual segmentation and physical-dimension errors, including failure cases.
- [ ] Document inference commands, model provenance, and limitations.

There is no supplied numerical acceptance tolerance for final measurement accuracy.
Do not invent one or infer millimeter accuracy directly from calibration pixel RMS.
