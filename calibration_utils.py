"""Shared checkerboard geometry, detection, and calibration metrics."""

import cv2
import numpy as np


def object_grid(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    """OpenCV pattern size is (columns, rows) of INNER corners."""
    if min(pattern_size) < 3 or not np.isfinite(square_size_mm) or square_size_mm <= 0:
        raise ValueError("Use at least 3 x 3 inner corners and a positive finite square size.")
    points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    points[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)
    points *= square_size_mm
    return points


def detect_corners(gray: np.ndarray, pattern_size: tuple[int, int]):
    """Detect the full board, then explicitly refine at native resolution."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    method = "findChessboardCorners"
    if not found:
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        method = "findChessboardCornersSB"
    if not found:
        return None, None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
    if len(corners) != pattern_size[0] * pattern_size[1] or not np.isfinite(corners).all():
        return None, None
    return corners, method


def reprojection_metrics(object_points, image_points, rvecs, tvecs, camera_matrix, distortion):
    """RMS = sqrt(mean(dx**2 + dy**2)), in image pixels, NOT norm/N."""
    distances = []
    per_view = []
    for obj, observed, rotation, translation in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rotation, translation, camera_matrix, distortion)
        delta = observed.reshape(-1, 2).astype(np.float64) - projected.reshape(-1, 2)
        errors = np.linalg.norm(delta, axis=1)
        distances.append(errors)
        per_view.append({
            "rms_px": float(np.sqrt(np.mean(errors ** 2))),
            "mean_euclidean_px": float(errors.mean()),
            "max_px": float(errors.max()),
        })
    errors = np.concatenate(distances)
    return {
        "rms_px": float(np.sqrt(np.mean(errors ** 2))),
        "mean_euclidean_px": float(errors.mean()),
        "max_px": float(errors.max()),
        "point_count": int(errors.size),
        "per_view": per_view,
    }
