"""Compare saved original/WhatsApp calibrations and save a second undistortion example."""

import hashlib
import json
import shutil
from collections import Counter

import numpy as np

from inspect_images import ROOT, read_image, write_image
from undistort import comparison_image, load_parameters, undistort_image


def main():
    output = ROOT / "calibration"
    previous = ROOT / "calibration_previous_whatsapp"
    old = json.loads((previous / "calibration_report.json").read_text(encoding="utf-8"))
    new = json.loads((output / "calibration_report.json").read_text(encoding="utf-8"))
    scale_x = new["image_size"][0] / old["image_size"][0]
    scale_y = new["image_size"][1] / old["image_size"][1]
    if not np.isclose(scale_x, scale_y):
        raise ValueError("Cannot compare scalar pixel errors across different aspect ratios.")
    normalized_rms = new["opencv_rms_px"] / scale_x
    change_percent = 100 * (normalized_rms / old["opencv_rms_px"] - 1)
    for record in new["images"]:
        source = ROOT / "Callibration_Img/Callibrate" / record["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source file changed after detection: {source}")
    matrix, distortion, size = load_parameters(output / "camera_params.npz")
    np.testing.assert_array_equal(matrix, np.asarray(new["camera_matrix"]))
    np.testing.assert_array_equal(distortion.ravel(), np.asarray(new["dist_coeffs"]))

    # First original is a different, near-frontal view from the median-error example.
    first = next(r for r in new["images"] if r["status"] == "detected")
    source = ROOT / "Callibration_Img/Callibrate" / first["file"]
    before = read_image(source)
    after, new_matrix, roi = undistort_image(before, matrix, distortion, size)
    write_image(output / "example_2_before.png", before)
    write_image(output / "example_2_after.png", after)
    write_image(output / "undistortion_comparison_2.jpg", comparison_image(before, after))
    second_example = {
        "source": first["file"], "alpha": 1.0, "cropped": False,
        "output_camera_matrix": new_matrix.tolist(), "valid_roi": roi,
    }
    (output / "example_2_metadata.json").write_text(json.dumps(second_example, indent=2), encoding="utf-8")
    for filename in ("contact_sheet.jpg", "image_inventory.json"):
        shutil.copy2(ROOT / "calibration_originals" / filename, output / filename)

    comparison = {
        "previous_report": str(previous / "calibration_report.json"),
        "current_report": str(output / "calibration_report.json"),
        "previous_image_size": old["image_size"], "current_image_size": new["image_size"],
        "linear_resolution_scale": scale_x,
        "previous_rms_px": old["opencv_rms_px"], "current_rms_px": new["opencv_rms_px"],
        "current_rms_in_previous_resolution_pixels": normalized_rms,
        "previous_rms_in_current_resolution_pixels": old["opencv_rms_px"] * scale_x,
        "resolution_normalized_change_percent": change_percent,
        "previous_camera_matrix": old["camera_matrix"], "current_camera_matrix": new["camera_matrix"],
        "current_camera_matrix_scaled_to_previous_resolution":
            (np.diag([1 / scale_x, 1 / scale_y, 1]) @ matrix).tolist(),
        "previous_dist_coeffs": old["dist_coeffs"], "current_dist_coeffs": new["dist_coeffs"],
        "original_source_hashes_unchanged": True,
        "note": "Resolution normalization is a change of pixel units, not a new calibration or independent accuracy test.",
    }
    (output / "comparison_with_previous.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    per_view = new["reprojection"]["per_view"]
    lowest = min(per_view, key=lambda view: view["rms_px"])
    highest = max(per_view, key=lambda view: view["rms_px"])
    text = (
        "Original-image recalibration versus preserved WhatsApp calibration\n\n"
        f"Previous: {old['image_size']}, {old['successful_detections']}/{old['total_images']} detected, "
        f"RMS {old['opencv_rms_px']:.9f} px\n"
        f"Current: {new['image_size']}, {new['successful_detections']}/{new['total_images']} detected, "
        f"RMS {new['opencv_rms_px']:.9f} px\n"
        f"Linear resolution scale: {scale_x:g}\n"
        f"Current RMS expressed at previous resolution: {normalized_rms:.9f} px\n"
        f"Previous RMS expressed at current resolution: {old['opencv_rms_px'] * scale_x:.9f} px\n"
        f"Resolution-normalized RMS change: {change_percent:+.6f}% (negative = lower)\n"
        "The normalized fit is very similar; this is not proof of improved physical measurement accuracy.\n\n"
        f"Lowest per-image RMS: {lowest['file']} = {lowest['rms_px']:.9f} px\n"
        f"Highest per-image RMS: {highest['file']} = {highest['rms_px']:.9f} px\n"
        f"Detectors: {dict(Counter(r.get('detector') for r in new['images']))}\n"
        f"Camera matrix:\n{np.array2string(matrix, precision=9, suppress_small=True)}\n"
        f"Distortion [k1, k2, p1, p2, k3]: {np.array2string(distortion.ravel(), precision=9)}\n\n"
        "All 39 original source hashes remain unchanged after calibration.\n"
        "Full per-image errors: per_image_errors.csv\n"
        "Original input files are camera JPEGs, not lossless/raw files; no new input recompression was performed.\n"
        "The visible creases may contribute to residual errors in a planar-board calibration.\n"
    )
    (output / "comparison_with_previous.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
