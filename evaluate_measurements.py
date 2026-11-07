"""Batch-evaluate 14 same-plane checkerboard sponge photos without changing inputs."""

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from measure_sponge import (annotate, calibration_geometry, checkerboard_homography,
                            measure_surface, run_unet)
from generate_object_masks import undistort_preserving_orientation
from inspect_images import read_image, write_image
from undistort import load_parameters


GROUND_TRUTH_MM = (93.0, 74.0)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def numeric_key(path):
    digits = "".join(re.findall(r"\d+", path.stem))
    return (int(digits) if digits else -1, path.name.lower())


def errors(width, height):
    long_mm, short_mm = sorted((float(width), float(height)), reverse=True)
    long_error, short_error = abs(long_mm - 93), abs(short_mm - 74)
    return {"long_side_mm": long_mm, "short_side_mm": short_mm, "long_abs_error_mm": long_error,
            "short_abs_error_mm": short_error, "long_percentage_error": 100 * long_error / 93,
            "short_percentage_error": 100 * short_error / 74, "mae_mm": (long_error + short_error) / 2}


def measure(path, raw, checkpoint, matrix, distortion, calibration_size, pattern, square_size, device):
    source = read_image(path)
    try:
        if raw:
            if (source.shape[1], source.shape[0]) not in (calibration_size, calibration_size[::-1]):
                raise ValueError("raw_image_size_not_supported")
            image, orientation = source, "not_applied_raw_evaluation"
        else:
            image, orientation = undistort_preserving_orientation(source, matrix, distortion, calibration_size)
        homography, corners, detector, residual = checkerboard_homography(image, pattern, square_size)
        probability, mask = run_unet(image, checkpoint, device)
        contour, rectangle, width, height, confidence = measure_surface(image, mask, probability, homography, corners)
        result = {"status": "measured", "filename": path.name, "width_mm": width, "height_mm": height,
                  "segmentation_confidence": confidence, "reference_status": "detected", "reference_detector": detector,
                  "homography_residual_mm": residual, "orientation_handling": orientation}
        result.update(errors(width, height)); annotated = annotate(image, corners, contour, rectangle, width, height, confidence)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        result = {"status": "refused", "filename": path.name, "reason": str(exc), "reference_status": "not_usable"}
        annotated = annotate(source, reason=str(exc))
    return result, annotated


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Output already exists; choose a new directory.")
    paths = sorted((p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=numeric_key)
    if len(paths) != 14: parser.error(f"Expected 14 images, found {len(paths)}.")
    matrix, distortion, calibration_size = load_parameters(args.params)
    pattern, square_size = calibration_geometry(args.params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True); rows = []
    for path in paths:
        corrected, corrected_image = measure(path, False, args.checkpoint, matrix, distortion, calibration_size, pattern, square_size, device)
        raw, raw_image = measure(path, True, args.checkpoint, matrix, distortion, calibration_size, pattern, square_size, device)
        write_image(args.output / "undistorted" / f"{path.stem}.png", corrected_image)
        write_image(args.output / "raw" / f"{path.stem}.png", raw_image)
        write_json(args.output / "json" / f"{path.stem}.json", {"undistorted": corrected, "raw": raw})
        row = {"filename": path.name, "undistorted_status": corrected["status"], "undistorted_reason": corrected.get("reason", ""),
               "raw_status": raw["status"], "raw_reason": raw.get("reason", ""),
               "undistorted_width_mm": corrected.get("width_mm", ""), "undistorted_height_mm": corrected.get("height_mm", ""), "undistorted_mae_mm": corrected.get("mae_mm", ""),
               "raw_width_mm": raw.get("width_mm", ""), "raw_height_mm": raw.get("height_mm", ""), "raw_mae_mm": raw.get("mae_mm", "")}
        row["mae_raw_minus_undistorted_mm"] = raw.get("mae_mm", 0) - corrected.get("mae_mm", 0) if raw["status"] == corrected["status"] == "measured" else ""
        rows.append(row); print(f"{path.name}: undistorted={corrected['status']}; raw={raw['status']}", flush=True)
    with (args.output / "measurement_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    def summarize(name):
        values = [float(r[f"{name}_mae_mm"]) for r in rows if r[f"{name}_status"] == "measured"]
        return {"successful": len(values), "refused": len(rows) - len(values), "mae_mm": float(np.mean(values)) if values else None,
                "median_mae_mm": float(np.median(values)) if values else None, "worst_mae_mm": float(max(values)) if values else None}
    corrected_summary, raw_summary = summarize("undistorted"), summarize("raw")
    report = {"images_discovered": [p.name for p in paths], "ground_truth_mm": {"long": 93, "short": 74}, "undistorted": corrected_summary,
              "raw": raw_summary, "raw_vs_undistorted_mae_difference_mm": None if None in (corrected_summary["mae_mm"], raw_summary["mae_mm"]) else raw_summary["mae_mm"] - corrected_summary["mae_mm"],
              "hardware": {"device": str(device)}, "note": "All rows, including refusals, are retained in measurement_results.csv."}
    write_json(args.output / "measurement_summary.json", report); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
