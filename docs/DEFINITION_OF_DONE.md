# Definition of Done

- [x] Save calibration parameters and a reproducible calibration report for 39 original images (`calibration/camera_params.npz`, `calibration/calibration_report.json`).
- [x] Record the 77-image source inventory and fixed split manifest (`dataset/inventory.json`, `dataset/split_manifest.csv`).
- [x] Keep source photos out of Git and retain hashes in the reports.
- [ ] Verify the checkerboard was rigid and flat enough for the final calibration; the recorded board shows creases.
- [x] Undistort the object images and create draft masks for the green top surface; remaining independent annotation QA is incomplete.
- [ ] Use a final split that groups related images without using test images for training, tuning, or checkpoint selection.
- [x] Train a permitted U-Net and report held-out split metrics in `results/evaluation/evaluation_report.json`; draft-mask labels limit the claim.
- [x] Provide checkerboard-reference inference with annotated width, height, confidence, and explicit rejection results.
- [ ] Measure at least 10 successful examples against manual ground truth. The 14-photo run has 12 successes; MAE is 5.059 mm and does not meet the 3 mm objective.
- [ ] Test a documented stress condition. Two photos were refused as not reliably green, but lighting/tilt severity was not measured.
- [x] Record five CPU timing runs: median 17.888 s, range 13.724-22.656 s; Python 3.12.10, OpenCV 4.12.0, PyTorch 2.9.1+cpu, Windows 10.
- [ ] Meet repository submission requirements: public repository, a `dev-test01` to `main` pull request, and a Loom video. Public visibility and a `main` branch were not verifiable from this checkout.
- [ ] Meet the required plan-and-split-before-code commit order. The existing history does not meet this requirement and will not be rewritten.
