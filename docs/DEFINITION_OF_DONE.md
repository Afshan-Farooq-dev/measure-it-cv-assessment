# Definition of Done

- [x] Save calibration parameters and a reproducible calibration report for 39 original images (`calibration/camera_params.npz`, `calibration/calibration_report.json`).
- [x] Record the 77-image source inventory and fixed split manifest (`dataset/inventory.json`, `dataset/split_manifest.csv`).
- [x] Keep source photos out of Git and retain hashes in the reports.
- [ ] Verify the checkerboard was rigid and flat enough for the final calibration; the recorded board shows creases.
- [ ] Undistort the object images and create checked masks for the green top surface.
- [ ] Use a final split that groups related images without using test images for training, tuning, or checkpoint selection.
- [ ] Train a permitted segmentation model and report held-out test IoU, mAP50, mAP50:95, precision, and recall.
- [ ] Provide one-image inference that finds a same-plane reference, segments the sponge, overlays dimensions and confidence, and handles missing objects or references.
- [ ] Measure at least 10 new examples against manual ground truth; report mean, median, and worst error, plus raw versus undistorted results.
- [ ] Test one required robustness condition and document the refusal threshold for unreliable results.
- [ ] Record reproducible commands, dependency versions, hardware, and timing from five runs with median and spread.
- [ ] Meet repository submission requirements: public repository, a `dev-test01` to `main` pull request, and a Loom video. Public visibility and a `main` branch were not verifiable from this checkout.
- [ ] Meet the required plan-and-split-before-code commit order. The existing history does not meet this requirement and will not be rewritten.
