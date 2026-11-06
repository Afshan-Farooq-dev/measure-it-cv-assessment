"""Measure a U-Net segmented sponge top against a coplanar calibration checkerboard."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from calibration_utils import detect_corners, object_grid
from colab.train_unet_colab import UNet
from generate_object_masks import undistort_preserving_orientation
from inspect_images import ROOT, read_image, write_image
from undistort import load_parameters


def calibration_geometry(parameters):
    with np.load(parameters, allow_pickle=False) as data:
        pattern = tuple(int(value) for value in data["pattern_size"])
        square_size_mm = float(data["square_size_mm"])
    if pattern != (7, 9) or square_size_mm != 24.0:
        raise ValueError("Expected the saved 7x9-inner-corner, 24 mm checkerboard calibration.")
    return pattern, square_size_mm


def letterbox_rgb(image):
    height, width = image.shape[:2]
    scale = min(256 / width, 256 / height)
    resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    top, left = (256 - resized.shape[0]) // 2, (256 - resized.shape[1]) // 2
    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas, (left, top, resized.shape[1], resized.shape[0])


def run_unet(image, checkpoint, device):
    model = UNet().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    model.eval()
    prepared, (left, top, scaled_width, scaled_height) = letterbox_rgb(image)
    tensor = torch.from_numpy(prepared.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    cropped = probability[top:top + scaled_height, left:left + scaled_width]
    probability = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return probability, (probability >= .5).astype(np.uint8) * 255


def checkerboard_homography(image, pattern, square_size_mm):
    corners, detector = detect_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern)
    if corners is None:
        raise ValueError("missing_checkerboard_reference")
    target = object_grid(pattern, square_size_mm)[:, :2]
    homography, _ = cv2.findHomography(corners.reshape(-1, 2), target, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("invalid_checkerboard_geometry")
    projected = cv2.perspectiveTransform(corners.reshape(1, -1, 2), homography)[0]
    residual_mm = float(np.sqrt(np.mean(np.sum((projected - target) ** 2, axis=1))))
    if residual_mm > 3.0:
        raise ValueError("unreliable_checkerboard_geometry")
    return homography, corners, detector, residual_mm


def measure_surface(image, mask, probability, homography, checkerboard_corners):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("missing_sponge_segmentation")
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    image_area = mask.shape[0] * mask.shape[1]
    x, y, width, height = cv2.boundingRect(contour)
    if x <= 5 or y <= 5 or x + width >= mask.shape[1] - 5 or y + height >= mask.shape[0] - 5:
        raise ValueError("segmentation_touches_frame_edge")
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 0.0
    if not .005 <= area / image_area <= .60 or solidity < .55:
        raise ValueError("unreliable_sponge_segmentation")
    selected = np.zeros_like(mask); cv2.drawContours(selected, [contour], -1, 255, cv2.FILLED)
    board = np.zeros_like(mask)
    cv2.fillConvexPoly(board, cv2.convexHull(checkerboard_corners.reshape(-1, 2)).astype(np.int32), 255)
    overlap = cv2.countNonZero(cv2.bitwise_and(selected, board)) / max(1, cv2.countNonZero(selected))
    if overlap > .05:
        raise ValueError("sponge_segmentation_overlaps_checkerboard_reference")
    confidence = float(probability[selected > 0].mean())
    if confidence < .60:
        raise ValueError("low_segmentation_confidence")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_fraction = float(np.mean(((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 105) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 15))[selected > 0]))
    if green_fraction < .55:
        raise ValueError("segmented_region_is_not_reliably_green")
    rectangle = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    plane_rectangle = cv2.perspectiveTransform(rectangle[None], homography)[0]
    edges = np.linalg.norm(plane_rectangle - np.roll(plane_rectangle, -1, axis=0), axis=1)
    width_mm, height_mm = float((edges[0] + edges[2]) / 2), float((edges[1] + edges[3]) / 2)
    if min(width_mm, height_mm) <= 1 or max(width_mm, height_mm) > 1000:
        raise ValueError("invalid_measurement_geometry")
    return contour, rectangle, width_mm, height_mm, confidence


def annotate(image, corners=None, contour=None, rectangle=None, width_mm=None, height_mm=None, confidence=None, reason=None):
    result = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(result, (7, 9), corners, True)
        cv2.putText(result, "Checkerboard: 7x9 inner corners, 24 mm squares", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2, cv2.LINE_AA)
    if contour is not None:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 4)
        cv2.polylines(result, [rectangle.astype(np.int32)], True, (0, 220, 255), 3)
        cv2.putText(result, f"width: {width_mm:.1f} mm | height: {height_mm:.1f} mm | confidence: {confidence:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 100, 0), 2, cv2.LINE_AA)
    elif reason:
        cv2.putText(result, f"MEASUREMENT REJECTED: {reason}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 220), 2, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Photo containing both the sponge top and coplanar checkerboard")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "best_model.pth")
    parser.add_argument("--params", type=Path, default=ROOT / "calibration/camera_params.npz")
    parser.add_argument("--output", type=Path, required=True, help="New annotated PNG; a JSON sidecar is also created")
    args = parser.parse_args()
    metadata = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or metadata.exists(): parser.error("Output already exists; choose a new filename.")
    if not args.image.is_file() or not args.checkpoint.is_file() or not args.params.is_file(): parser.error("Input image, checkpoint, and calibration parameters must exist.")
    source = read_image(args.image)
    try:
        matrix, distortion, calibration_size = load_parameters(args.params)
        pattern, square_size_mm = calibration_geometry(args.params)
        undistorted, orientation = undistort_preserving_orientation(source, matrix, distortion, calibration_size)
        homography, corners, detector, reference_residual_mm = checkerboard_homography(undistorted, pattern, square_size_mm)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probability, mask = run_unet(undistorted, args.checkpoint, device)
        contour, rectangle, width_mm, height_mm, confidence = measure_surface(undistorted, mask, probability, homography, corners)
        outcome = {"status": "measured", "width_mm": width_mm, "height_mm": height_mm, "segmentation_confidence": confidence,
                   "reference_status": "detected", "reference": {"pattern_inner_corners": list(pattern), "square_size_mm": square_size_mm, "detector": detector, "homography_residual_mm": reference_residual_mm},
                   "orientation_handling": orientation, "device": str(device), "method": "checkerboard-plane homography"}
        output = annotate(undistorted, corners, contour, rectangle, width_mm, height_mm, confidence)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        outcome = {"status": "rejected", "reason": str(exc), "measurement": None,
                   "reference_status": "not_usable", "reference_requirement": "The 7x9-inner-corner checkerboard with 24 mm squares must be fully visible and coplanar with the green top surface."}
        output = annotate(source, reason=str(exc))
    write_image(args.output, output)
    metadata.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__": main()
