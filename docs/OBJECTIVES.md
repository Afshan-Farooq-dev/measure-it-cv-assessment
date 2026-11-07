# Objectives

## MO-1: measurement accuracy

Measure the green top surface on at least 10 new photos with a rigid, known-size reference in the same plane. Compare predicted width and height with ruler or calliper ground truth. Target mean absolute error: **3 mm or less**. Report mean, median, and worst error in millimetres.

Status: incomplete. On 14 new checkerboard-reference photos, 12 were measured and 2 were refused. Undistorted MAE was 5.059 mm (median 4.874 mm; worst 8.406 mm), so the 3 mm target was not met. See `results/evaluation/measurement_summary_14.json`.

## MO-2: segmentation quality

Evaluate the chosen permitted segmentation model on the held-out test masks only. Target mask IoU: **0.85 or higher**. Also report the required mAP50, mAP50:95, precision, and recall.

Status: measured on the fixed 8-image split: IoU 0.9692, Dice 0.9842, precision 0.9782, recall 0.9908, mAP50 1.0000, and mAP50:95 0.9551. These compare against draft masks, not independently hand-labelled masks.

## Current evidence

- Camera calibration used 39 original 3968 x 2976 images. The saved report records 7 x 9 inner corners, 24 mm squares, and RMS reprojection error of 2.957893 px.
- The object dataset inventory contains 77 valid images: 40 at 3968 x 2976 and 37 at 2976 x 3968.
- The current committed manifest uses seed 42 and a 54/15/8 train/validation/test split. It is not used for training yet.
