# Measure It CV assessment

This repository contains calibration, draft segmentation, U-Net evaluation, and checkerboard-reference measurement work for the xis.ai Measure It assessment.

## Current evidence

- Camera: itel S24; EXIF records model `itel S667LN`.
- Calibration: 39 original images, 3968 x 2976, 7 x 9 inner corners, 24 mm squares, all detected. RMS reprojection error: 2.957893 px.
- Object: green-yellow sponge. Target: green top surface only. Ground truth: 93 mm x 74 mm.
- Dataset: 77 valid source images (40 landscape, 37 portrait). Current seed-42 split: 54 train / 15 validation / 8 test.
- The previous WhatsApp-image calibration is retained for comparison; its recorded RMS is 1.199105 px. Lower RMS alone does not make it the preferred calibration because those images were compressed.
- Fixed 8-image segmentation test: IoU 0.9692, precision 0.9782, recall 0.9908, mAP50 1.0000, mAP50:95 0.9551, evaluated against draft masks.
- New 14-photo checkerboard measurement run: 12 successes, 2 refusals; undistorted MAE 5.059 mm (median 4.874 mm, worst 8.406 mm). Raw MAE was 4.679 mm for these same scenes; this small result does not establish a general undistortion benefit.

Source photos and generated images are ignored by Git. Calibration and dataset reports contain their hashes. `calibration/calibration_report.json` and `dataset/inventory.json` are the authoritative inventories; similarly named legacy inventories are retained as historical evidence.

## Setup

Tested with Python 3.12.10, OpenCV 4.12.0, NumPy 2.2.6, and Pillow 12.0.0.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use a new output directory for every run. The scripts refuse to replace an existing report.

```powershell
# Inspect an image directory (the directory may be outside this repository)
python inspect_images.py --images E:\Computer_Vision\Object --output runs\object-inspection

# Recalibrate without changing the saved evidence
python calibrate.py --images E:\Computer_Vision\Callibration_Img\Callibrate --pattern 7 9 --square-size-mm 24 --output runs\calibration-run

# Compare saved reports only; no source photos are required
python compare_calibrations.py

# Create a new split/audit run; this does not replace dataset/ evidence
python prepare_dataset.py --images E:\Computer_Vision\Object --params calibration\camera_params.npz --train 54 --validation 15 --test 8 --seed 42 --output runs\dataset-run

# Produce, but do not apply, a group-aware split proposal
python propose_split.py --output runs\split-proposal --seed 42

# Measure a new photo only when the 24 mm checkerboard is fully visible and coplanar
python measure_sponge.py "E:\path\to\photo.jpg" --checkpoint best_model.pth --params calibration\camera_params.npz --output outputs\measurement.png
```

## Repository layout

- `calibration/`: saved camera parameters and calibration reports.
- `dataset/`: committed source inventory, split manifest, and duplicate audit.
- `docs/`: plan, objectives, and evidence-based definition of done.
- `calibrate.py`, `inspect_images.py`, `prepare_dataset.py`: reproducible preparation scripts.
- `undistort.py`: applies saved calibration parameters to a new image.

## Remaining assessment work

The project is not yet assessment-complete. The 3 mm measurement target was missed; only 12 successful manual checks exist, labels are draft masks, stress severity is not quantified, the checkpoint (`0b354b017fe03d97310f60d6ff0a63fcb0a34af312775c9cb56a3582319081bb`) is local-only with no reproducible download link, and PR/Loom submission items remain. The assessment's required plan-and-split-before-code commit order was not met by the existing history and is recorded in [docs/DEFINITION_OF_DONE.md](docs/DEFINITION_OF_DONE.md).

## Hardware

The preparation work was run on Windows 10 Pro (10.0.19045), Intel Core i5-6300U at 2.40 GHz, 8 GiB RAM, and Microsoft Basic Display Adapter. Record the actual training hardware and GPU, if any, with the training results.
