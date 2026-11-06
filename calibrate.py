"""Calibrate a mobile camera from full checkerboard detections at native resolution."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np

from calibration_utils import detect_corners, object_grid, reprojection_metrics
from inspect_images import ROOT, image_paths, read_image, write_image
from undistort import comparison_image, load_parameters, undistort_image
from output_utils import new_output_directory


def collect_detections(paths, pattern, output):
    records, points, successful_paths = [], [], []
    sizes = set()
    for index, path in enumerate(paths, start=1):
        record = {"index": index, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            image = read_image(path)
        except ValueError as exc:
            record.update(status="unreadable", reason=str(exc))
            records.append(record)
            print(f"[{index:02d}/{len(paths)}] UNREADABLE {path.name}", flush=True)
            continue
        size = (image.shape[1], image.shape[0])
        sizes.add(size)
        record["image_size"] = list(size)
        corners, method = detect_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern)
        if corners is None:
            record.update(status="corners_not_found", corners_detected=0)
            cv2.putText(image, "FULL BOARD NOT DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            record.update(status="detected", corners_detected=len(corners), detector=method)
            points.append(corners)
            successful_paths.append(path)
            cv2.drawChessboardCorners(image, pattern, corners, True)
        preview = output / "detections" / f"{index:02d}_{path.stem}.jpg"
        write_image(preview, image)
        record["annotation"] = str(preview.relative_to(output))
        records.append(record)
        print(f"[{index:02d}/{len(paths)}] {record['status']} {path.name}", flush=True)
    (output / "detection_report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if len(sizes) != 1:
        raise ValueError(f"Expected one image resolution; found {sorted(sizes)}. No calibration was fitted.")
    return records, points, successful_paths, sizes.pop()


def run_calibration(images, output, pattern, square_size_mm):
    with new_output_directory(output, [images]) as staging:
        return _calibrate(images, staging, pattern, square_size_mm, output.resolve())


def _calibrate(images, output, pattern, square_size_mm, published_output):
    grid = object_grid(pattern, square_size_mm)
    paths = image_paths(images)
    if not paths:
        raise ValueError(f"No supported images found in {images}")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    records, image_points, successful_paths, size = collect_detections(paths, pattern, output)
    if len(image_points) < 3:
        raise ValueError("At least three successful views are required; capture 10 or more varied views.")
    objects = [grid.copy() for _ in image_points]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        objects, image_points, size, None, None, flags=0, criteria=criteria
    )
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Calibration returned non-finite parameters.")
    metrics = reprojection_metrics(objects, image_points, rvecs, tvecs, matrix, distortion)
    if not np.isclose(rms, metrics["rms_px"], atol=1e-4, rtol=1e-4):
        raise ValueError("Independent reprojection RMS disagrees with OpenCV.")
    for path, error in zip(successful_paths, metrics["per_view"]):
        error["file"] = path.name
    with (output / "per_image_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "rms_px", "mean_euclidean_px", "max_px"])
        writer.writeheader()
        writer.writerows(metrics["per_view"])
    params_path = output / "camera_params.npz"
    np.savez_compressed(
        params_path, camera_matrix=matrix, dist_coeffs=distortion,
        image_size=np.array(size), pattern_size=np.array(pattern), square_size_mm=square_size_mm,
        rms_reprojection_error_px=rms, rvecs=np.array(rvecs), tvecs=np.array(tvecs),
        object_points=grid, image_points=np.array(image_points),
        image_names=np.array([p.name for p in successful_paths]),
        per_view_rms_px=np.array([v["rms_px"] for v in metrics["per_view"]]),
    )
    # Demonstrate that the serialized parameters can actually be loaded and used.
    saved_matrix, saved_distortion, saved_size = load_parameters(params_path)
    example_index = int(np.argsort([v["rms_px"] for v in metrics["per_view"]])[len(image_points) // 2])
    example_path = successful_paths[example_index]
    before = read_image(example_path)
    after, new_matrix, roi = undistort_image(before, saved_matrix, saved_distortion, saved_size)
    write_image(output / "example_before.png", before)
    write_image(output / "example_after.png", after)
    write_image(output / "undistortion_comparison.jpg", comparison_image(before, after))
    warnings = [
        "Reprojection error is an in-sample pixel fit, not an independent measurement-accuracy test.",
        "Reuse only with the same lens, zoom, focus regime, crop, and image processing/orientation.",
        "The supplied photos show creases in the paper; nonplanarity violates the planar-board assumption.",
        f"Parameters apply to the supplied {size[0]}x{size[1]} image geometry; pixel errors depend on resolution.",
    ]
    if len(image_points) < 10:
        warnings.append("Fewer than 10 successful views; collect more varied board poses.")
    report = {
        "image_directory": str(images.resolve()),
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "pattern_size_inner_corners": list(pattern), "square_size_mm": square_size_mm,
        "image_size": list(size), "orientation_policy": "encoded pixels; EXIF auto-rotation disabled",
        "total_images": len(paths), "successful_detections": len(image_points),
        "failed_detections": sum(r["status"] == "corners_not_found" for r in records),
        "unreadable_images": sum(r["status"] == "unreadable" for r in records),
        "images_used": len(image_points), "outlier_images_removed": 0,
        "model": "OpenCV pinhole with k1,k2,p1,p2,k3; calibration flags=0",
        "corner_refinement": {"window": [11, 11], "max_iterations": 100, "epsilon": 0.001},
        "solver": {"max_iterations": 100, "epsilon": 1e-9, "threads": 1, "rng_seed": 0},
        "opencv_rms_px": float(rms), "reprojection": metrics,
        "camera_matrix": matrix.tolist(), "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "dist_coeffs": distortion.ravel().tolist(),
        "example": {"source": example_path.name, "selection": "median per-view RMS",
                    "alpha": 1.0, "cropped": False, "valid_roi": roi,
                    "output_camera_matrix": new_matrix.tolist()},
        "limitations": warnings, "images": records,
    }
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = (
        f"Inner corners: {pattern[0]} x {pattern[1]} ({len(grid)} points/view)\n"
        f"Square size: {square_size_mm:g} mm; image size: {size[0]} x {size[1]}\n"
        f"Detected: {len(image_points)}/{len(paths)}; not detected: {report['failed_detections']}; "
        f"unreadable: {report['unreadable_images']}\n"
        f"All {len(image_points)} detections used; no outlier removal.\n"
        f"OpenCV RMS reprojection error: {rms:.6f} px\n"
        f"Independently calculated RMS: {metrics['rms_px']:.6f} px\n"
        f"Mean Euclidean reprojection error: {metrics['mean_euclidean_px']:.6f} px\n"
        f"Per-image RMS range: {min(v['rms_px'] for v in metrics['per_view']):.6f} to "
        f"{max(v['rms_px'] for v in metrics['per_view']):.6f} px\n"
        f"Camera matrix (pixels):\n{np.array2string(matrix, precision=8)}\n"
        f"Distortion [k1, k2, p1, p2, k3]:\n{np.array2string(distortion.ravel(), precision=8)}\n"
        f"Example source: {example_path.name}\n"
        f"Parameters: {published_output / 'camera_params.npz'}\n\nLimitations:\n" + "\n".join(f"- {w}" for w in warnings) + "\n"
    )
    (output / "results.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "Callibration_Img/Callibrate")
    parser.add_argument("--output", type=Path, required=True, help="A new run directory; existing calibration is never replaced")
    # Deliberately require the visually inspected pattern; do not guess a default.
    parser.add_argument("--pattern", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"))
    parser.add_argument("--square-size-mm", type=float, required=True)
    args = parser.parse_args()
    try:
        run_calibration(args.images, args.output, tuple(args.pattern), args.square_size_mm)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"Calibration failed: {exc}\n")


if __name__ == "__main__":
    main()
