# Objectives

## MO-1: measurement accuracy

Measure the green top surface on at least 10 new photos with a rigid, known-size reference in the same plane. Compare predicted width and height with ruler or calliper ground truth. Target mean absolute error: **3 mm or less**. Report mean, median, and worst error in millimetres.

Status: pending. No measurement experiment has been run.

## MO-2: segmentation quality

Evaluate the chosen permitted segmentation model on the held-out test masks only. Target mask IoU: **0.85 or higher**. Also report the required mAP50, mAP50:95, precision, and recall.

Status: pending. No masks or model exist yet.

## Current evidence

- Camera calibration used 39 original 3968 x 2976 images. The saved report records 7 x 9 inner corners, 24 mm squares, and RMS reprojection error of 2.957893 px.
- The object dataset inventory contains 77 valid images: 40 at 3968 x 2976 and 37 at 2976 x 3968.
- The current committed manifest uses seed 42 and a 54/15/8 train/validation/test split. It is not used for training yet.
