"""Undistort sponge photos and create reviewable OpenCV draft masks for the green top."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from inspect_images import ROOT, image_paths, read_image, write_image
from output_utils import new_output_directory
from undistort import load_parameters, undistort_image


GREEN_LOWER = np.array((35, 35, 15), dtype=np.uint8)
GREEN_UPPER = np.array((105, 255, 180), dtype=np.uint8)
CALIBRATION_SIZE = (3968, 2976)


def undistort_preserving_orientation(image, matrix, distortion, calibration_size):
    """Use calibration coordinates for a swapped portrait raster, then restore layout."""
    size = (image.shape[1], image.shape[0])
    if size == calibration_size:
        result, _, _ = undistort_image(image, matrix, distortion, calibration_size)
        return result, "landscape_direct"
    if size == calibration_size[::-1]:
        # EXIF orientation is 0 in this collection. Rotate into the calibrated pixel
        # axes instead of resizing; restore the original display layout afterwards.
        landscape = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        corrected, _, _ = undistort_image(landscape, matrix, distortion, calibration_size)
        return cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE), "portrait_cw_then_restore"
    raise ValueError(f"Unsupported image size {size}; expected {calibration_size} or its swapped orientation.")


def draft_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(threshold.shape, np.uint8), {"status": "manual_correction", "reason": "no_green_contour"}

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    image_area = image.shape[0] * image.shape[1]
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 0.0
    component_share = area / max(1, cv2.countNonZero(threshold))
    mask = np.zeros(threshold.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    area_fraction = area / image_area
    reasons = []
    if not 0.005 <= area_fraction <= 0.45:
        reasons.append("implausible_area")
    if solidity < 0.65:
        reasons.append("low_solidity")
    if component_share < 0.70:
        reasons.append("multiple_green_regions")
    return mask, {
        "status": "manual_correction" if reasons else "generated",
        "reason": ";".join(reasons),
        "area_fraction": area_fraction,
        "solidity": solidity,
        "component_share": component_share,
    }


def overlay(image, mask, status):
    result = image.copy()
    result[mask > 0] = (0.35 * result[mask > 0] + 0.65 * np.array((0, 255, 0))).astype(np.uint8)
    color = (0, 160, 0) if status == "generated" else (0, 0, 220)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 5)
    return result


def contact_sheet(records, output):
    pages = []
    for start in range(0, len(records), 12):
        tiles = []
        for record in records[start:start + 12]:
            image = cv2.imread(str(output / "overlays" / record["output_file"]), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (300, round(image.shape[0] * 300 / image.shape[1])), interpolation=cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
            label = f"{record['output_file']} | {record['status']}"
            cv2.putText(image, label[:45], (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 0, 180) if record["status"] != "generated" else (0, 100, 0), 1, cv2.LINE_AA)
            tiles.append(image)
        height = max(tile.shape[0] for tile in tiles)
        tiles.extend([np.full((height, 300, 3), 245, np.uint8)] * ((4 - len(tiles) % 4) % 4))
        tiles = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for tile in tiles]
        page = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        filename = f"contact_sheet_{start // 12 + 1:02d}.jpg"
        write_image(output / filename, page)
        pages.append(filename)
    return pages


def save_csv(path, rows):
    fields = ["source_file", "output_file", "orientation_handling", "status", "reason", "area_fraction", "solidity", "component_share"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(images, params, output):
    matrix, distortion, calibration_size = load_parameters(params)
    if calibration_size != CALIBRATION_SIZE:
        raise ValueError(f"Expected original-resolution calibration {CALIBRATION_SIZE}, got {calibration_size}.")
    paths = image_paths(images)
    if len(paths) != 77:
        raise ValueError(f"Expected 77 object images, found {len(paths)}.")
    records = []
    for path in paths:
        source = read_image(path)
        corrected, handling = undistort_preserving_orientation(source, matrix, distortion, calibration_size)
        mask, result = draft_mask(corrected)
        output_file = f"{path.stem}.png"
        write_image(output / "undistorted" / output_file, corrected)
        write_image(output / "masks" / output_file, mask)
        write_image(output / "overlays" / output_file, overlay(corrected, mask, result["status"]))
        records.append({"source_file": path.name, "output_file": output_file, "orientation_handling": handling, **result})
        print(f"{path.name}: {result['status']}", flush=True)
    sheets = contact_sheet(records, output)
    save_csv(output / "mask_report.csv", records)
    manual = [record for record in records if record["status"] != "generated"]
    (output / "manual_correction.csv").write_text(
        "source_file,reason\n" + "".join(f"{r['source_file']},{r['reason']}\n" for r in manual), encoding="utf-8"
    )
    summary = {
        "source_directory": str(images.resolve()), "parameter_file": str(params.resolve()),
        "total_images": len(records), "generated_masks": len(records) - len(manual),
        "manual_correction_count": len(manual), "manual_correction_files": [r["source_file"] for r in manual],
        "orientation_handling": dict(Counter(r["orientation_handling"] for r in records)),
        "mask_method": {"color_space": "HSV", "green_lower": GREEN_LOWER.tolist(), "green_upper": GREEN_UPPER.tolist(),
                        "selection": "largest external green contour", "postprocessing": "11px close, 5px open"},
        "contact_sheets": sheets,
        "note": "These are automatic draft masks, not checked ground-truth annotations.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "Object")
    parser.add_argument("--params", type=Path, default=ROOT / "calibration/camera_params.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations")
    args = parser.parse_args()
    try:
        with new_output_directory(args.output, [args.images, args.params]) as staging:
            run(args.images, args.params, staging)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Annotation generation failed: {exc}\n")


if __name__ == "__main__":
    main()
