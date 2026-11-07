"""Refine only previously flagged sponge masks; never replace accepted draft masks."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from generate_object_masks import overlay, undistort_preserving_orientation
from inspect_images import ROOT, read_image, write_image
from output_utils import new_output_directory
from undistort import load_parameters


def adaptive_green_mask(image):
    """Use the central object region to adapt HSV limits under different lighting."""
    h, w = image.shape[:2]
    normalized = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    normalized[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized[:, :, 0])
    normalized = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    center = hsv[int(h * .3):int(h * .7), int(w * .3):int(w * .7)]
    likely_green = center[(center[:, :, 0] >= 65) & (center[:, :, 0] <= 110) & (center[:, :, 1] >= 35)]
    if len(likely_green) < 100:
        return np.zeros((h, w), np.uint8), {"status": "manual_correction", "reason": "insufficient_green_seed"}
    hue = likely_green[:, 0]
    saturation = likely_green[:, 1]
    value = likely_green[:, 2]
    lower_h = max(55, int(np.percentile(hue, 5)) - 10)
    upper_h = min(115, int(np.percentile(hue, 95)) + 10)
    lower_s = max(35, int(np.percentile(saturation, 5)) - 15)
    lower_v = max(8, int(np.percentile(value, 2)) - 10)
    raw = cv2.inRange(hsv, np.array((lower_h, lower_s, lower_v), np.uint8), np.array((upper_h, 255, 255), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return raw, {"status": "manual_correction", "reason": "no_contour", "hsv_lower": [lower_h, lower_s, lower_v]}
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    area = cv2.contourArea(contour)
    area_fraction = area / (h * w)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 0.0
    component_share = area / max(1, cv2.countNonZero(raw))
    reasons = []
    # A close-up top-down photo can legitimately fill almost half of the frame.
    if not .01 <= area_fraction <= .55:
        reasons.append("implausible_area")
    if solidity < .75:
        reasons.append("low_solidity")
    if component_share < .82:
        reasons.append("multiple_green_regions")
    return mask, {"status": "generated" if not reasons else "manual_correction", "reason": ";".join(reasons),
                  "area_fraction": area_fraction, "solidity": solidity, "component_share": component_share,
                  "hsv_lower": [lower_h, lower_s, lower_v], "hsv_upper": [upper_h, 255, 255]}


def triptych(source, mask, status):
    detected = source.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(detected, contours, -1, (0, 0, 255) if status != "generated" else (0, 255, 0), 6)
    panels = [(source, "Undistorted source"), (overlay(source, mask, status), "Mask overlay"), (detected, "Detected contour")]
    output = []
    for image, label in panels:
        preview = cv2.resize(image, (420, round(image.shape[0] * 420 / image.shape[1])), interpolation=cv2.INTER_AREA)
        preview = cv2.copyMakeBorder(preview, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(preview, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        output.append(preview)
    return np.hstack(output)


def run(images, params, previous_report, output):
    flagged = list(csv.DictReader(previous_report.open(encoding="utf-8", newline="")))
    matrix, distortion, size = load_parameters(params)
    rows = []
    for previous in flagged:
        path = images / previous["source_file"]
        source, handling = undistort_preserving_orientation(read_image(path), matrix, distortion, size)
        mask, result = adaptive_green_mask(source)
        name = f"{path.stem}.png"
        write_image(output / "masks" / name, mask)
        write_image(output / "reviews" / name, triptych(source, mask, result["status"]))
        rows.append({"source_file": path.name, "output_file": name, "orientation_handling": handling, **result})
    fields = ["source_file", "output_file", "orientation_handling", "status", "reason", "area_fraction", "solidity", "component_share", "hsv_lower", "hsv_upper"]
    with (output / "refinement_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    sheets = []
    for index, row in enumerate(rows):
        sheet = cv2.imread(str(output / "reviews" / row["output_file"]), cv2.IMREAD_COLOR)
        filename = f"contact_sheet_{index + 1:02d}.jpg"
        write_image(output / filename, sheet); sheets.append(filename)
    manual = [r["source_file"] for r in rows if r["status"] != "generated"]
    summary = {"previous_masks_preserved": True, "attempted": len(rows), "refined_generated": len(rows) - len(manual),
               "still_manual_correction": manual, "contact_sheets": sheets,
               "method": "CLAHE lighting normalization plus center-seeded adaptive HSV and largest-contour filtering"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "Object")
    parser.add_argument("--params", type=Path, default=ROOT / "calibration/camera_params.npz")
    parser.add_argument("--previous-report", type=Path, default=ROOT / "annotations/manual_correction.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations_refined")
    args = parser.parse_args()
    try:
        with new_output_directory(args.output, [args.images, args.params, args.previous_report]) as staging:
            run(args.images, args.params, args.previous_report, staging)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Mask refinement failed: {exc}\n")


if __name__ == "__main__":
    main()
