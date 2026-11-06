"""Undistort a new photo using a saved calibration, without modifying its source."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from inspect_images import ROOT, read_image, write_image


def load_parameters(path: Path):
    with np.load(path, allow_pickle=False) as data:
        matrix = data["camera_matrix"].astype(np.float64)
        distortion = data["dist_coeffs"].astype(np.float64)
        image_size = tuple(int(v) for v in data["image_size"])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Invalid calibration parameters.")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or min(image_size) <= 0:
        raise ValueError("Invalid focal lengths or image dimensions.")
    return matrix, distortion, image_size


def matrix_for_size(matrix, calibration_size, target_size, allow_resize=False):
    if tuple(target_size) == tuple(calibration_size):
        return matrix.copy()
    if not allow_resize:
        raise ValueError(
            f"Image size {target_size} differs from calibration {calibration_size}. "
            "Use --allow-resize only for a uniform resize of the same camera view."
        )
    cw, ch = calibration_size
    w, h = target_size
    if w * ch != h * cw:
        raise ValueError("Aspect ratio differs: cropped or rotated images require a matching calibration.")
    scaled = matrix.copy()
    scaled[0, :] *= w / cw
    scaled[1, :] *= h / ch
    return scaled


def undistort_image(image, matrix, distortion, calibration_size, alpha=1.0, allow_resize=False):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    size = (image.shape[1], image.shape[0])
    adjusted_matrix = matrix_for_size(matrix, calibration_size, size, allow_resize)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(adjusted_matrix, distortion, size, alpha, size)
    result = cv2.undistort(image, adjusted_matrix, distortion, None, new_matrix)
    return result, new_matrix, tuple(int(v) for v in roi)


def comparison_image(before, after):
    """Equal-scale, labeled preview; full-resolution outputs are saved separately."""
    panels = []
    for image, label in [(before, "Before: original"), (after, "After: undistorted (alpha=1)")]:
        h, w = image.shape[:2]
        preview = cv2.resize(image, (800, round(h * 800 / w)), interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(preview, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(panel, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--params", type=Path, default=ROOT / "calibration/camera_params.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0, help="0 minimizes black borders; 1 retains field of view")
    parser.add_argument("--crop", action="store_true", help="Crop to valid-pixel ROI; metadata adjusts principal point")
    parser.add_argument("--allow-resize", action="store_true", help="Explicitly allow uniform resizing of the same camera view")
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    sources = {args.image.resolve(), args.params.resolve()}
    if args.output.resolve() in sources or metadata_path.resolve() in sources:
        parser.error("Output must not overwrite the input image or calibration parameters.")
    try:
        matrix, distortion, size = load_parameters(args.params)
        result, new_matrix, roi = undistort_image(
            read_image(args.image), matrix, distortion, size, args.alpha, args.allow_resize
        )
        if args.crop:
            x, y, w, h = roi
            if w <= 0 or h <= 0:
                raise ValueError("No valid crop ROI for these parameters.")
            result = result[y:y + h, x:x + w]
            new_matrix[0, 2] -= x
            new_matrix[1, 2] -= y
        write_image(args.output, result)
        metadata_path.write_text(json.dumps({
            "source": str(args.image.resolve()), "parameters": str(args.params.resolve()),
            "alpha": args.alpha, "cropped": args.crop, "valid_roi_before_crop": roi,
            "output_size": [result.shape[1], result.shape[0]],
            "output_camera_matrix": new_matrix.tolist(),
            "output_dist_coeffs": [0.0] * 5,
        }, indent=2), encoding="utf-8")
        print(f"Saved {args.output} ({result.shape[1]} x {result.shape[0]})")
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        parser.exit(1, f"Undistortion failed: {exc}\n")


if __name__ == "__main__":
    main()
