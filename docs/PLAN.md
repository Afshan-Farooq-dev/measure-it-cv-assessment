# Plan

This plan was written after the initial calibration and dataset-preparation work. It records the intended assessment workflow and current remaining work; it was not committed before the implementation.

## Eight-hour allocation

| Work | Planned time |
| --- | ---: |
| Confirm requirements and calibrate the camera | 1.0 h |
| Inspect images and make the split | 1.0 h |
| Annotate masks and check them | 1.0 h |
| Train and evaluate a permitted segmentation model | 2.0 h |
| Build reference-based measurement inference | 1.25 h |
| Accuracy, robustness, and raw/undistorted comparisons | 1.25 h |
| README, PR, and Loom recording | 0.5 h |
| **Total** | **8.0 h** |

## Decisions

- Use OpenCV chessboard calibration with 7 x 9 inner corners and 24 mm squares.
- Keep the 77 source photos unchanged. The current whole-image split is 54 train, 15 validation, and 8 test with seed 42.
- Use a fixed, committed manifest before any training. Test images are never used for training, tuning, or checkpoint selection.
- Measure only the green top surface of the sponge (93 mm x 74 mm ground truth). The final method must use a rigid, known-size reference in the same plane as the sponge.
- Do not use Ultralytics YOLO or a Roboflow segmentation model. Select a permitted model only after masks exist.

## Current limitations and changes

- Calibration is complete, but the checkerboard report notes board creases; flatness has not been independently verified.
- There are 77, not 78, valid object photos. The 54/15/8 split is the closest whole-image approximation to 70/20/10, not an exact ratio.
- The current split keeps related candidates apart across splits, but its validation and test sets contain only singleton groups. A proposal that gives both evaluation sets related groups is ready for review and has not been applied.
- Draft masks, U-Net training, checkerboard-reference inference, and a 14-photo accuracy run are complete. The error target was not met and only 12 of 14 photos produced measurements.
- The existing Git history began with completed work rather than a plan and split manifest. That ordering cannot be corrected without rewriting history, so it remains a documented assessment risk.
