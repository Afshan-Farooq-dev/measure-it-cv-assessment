"""Evaluate the saved U-Net checkpoint on the fixed test split and saved measurement records."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from colab.train_unet_colab import UNet
from measure_sponge import letterbox_rgb, run_unet
from output_utils import new_output_directory


GROUND_TRUTH_MM = (93.0, 74.0)


def test_metrics(dataset, checkpoint):
    manifest = list(csv.DictReader((dataset / "manifest.csv").open(encoding="utf-8", newline="")))
    test = [row for row in manifest if row["split"] == "test"]
    if len(test) != 8:
        raise ValueError(f"Expected 8 test images, found {len(test)}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"]); model.eval()
    rows = []
    for row in test:
        name = row["filename"]
        image = cv2.imread(str(dataset / "test/images" / name), cv2.IMREAD_COLOR)
        target = cv2.imread(str(dataset / "test/masks" / name), cv2.IMREAD_GRAYSCALE) >= 128
        prepared, (left, top, width, height) = letterbox_rgb(image)
        tensor = torch.from_numpy(prepared.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
        with torch.no_grad(): probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        probability = cv2.resize(probability[top:top + height, left:left + width], (256, 256), interpolation=cv2.INTER_LINEAR)
        predicted = probability >= .5
        intersection = int(np.logical_and(predicted, target).sum()); union = int(np.logical_or(predicted, target).sum())
        precision = intersection / max(1, int(predicted.sum())); recall = intersection / max(1, int(target.sum()))
        rows.append({"filename": name, "iou": intersection / max(1, union), "precision": precision, "recall": recall,
                     "dice": 2 * intersection / max(1, int(predicted.sum()) + int(target.sum())), "confidence": float(probability[predicted].mean()) if predicted.any() else 0.0})
    def ap_at(threshold):
        # One prediction and one ground-truth instance per image. Sort by mask confidence,
        # then calculate the ranked average precision at this IoU threshold.
        ordered = sorted(rows, key=lambda item: item["confidence"], reverse=True); hits = 0; precision_sum = 0.
        for rank, item in enumerate(ordered, 1):
            if item["iou"] >= threshold:
                hits += 1; precision_sum += hits / rank
        return precision_sum / len(ordered)
    thresholds = np.arange(.5, .96, .05)
    return {"test_images": len(rows), "device": str(device), "threshold": .5,
            "iou": float(np.mean([r["iou"] for r in rows])), "precision": float(np.mean([r["precision"] for r in rows])),
            "recall": float(np.mean([r["recall"] for r in rows])), "dice": float(np.mean([r["dice"] for r in rows])),
            "mAP_50": ap_at(.5), "mAP_50_95": float(np.mean([ap_at(t) for t in thresholds])), "per_image": rows,
            "mAP_note": "Ranked AP over one predicted mask and one target mask per image; confidence is mean probability inside the predicted mask."}


def error_summary(records):
    errors = []
    for record in records:
        values = (float(record["width_mm"]), float(record["height_mm"]))
        possibilities = [(abs(values[0] - GROUND_TRUTH_MM[0]), abs(values[1] - GROUND_TRUTH_MM[1])),
                         (abs(values[1] - GROUND_TRUTH_MM[0]), abs(values[0] - GROUND_TRUTH_MM[1]))]
        chosen = min(possibilities, key=sum)
        errors.extend(chosen)
    return {"instances": len(records), "ground_truth_mm": list(GROUND_TRUTH_MM), "absolute_errors_mm": errors,
            "mae_mm": float(np.mean(errors)), "median_absolute_error_mm": float(np.median(errors)), "worst_absolute_error_mm": float(max(errors)),
            "mean_percentage_error": float(np.mean([errors[i] / GROUND_TRUTH_MM[i % 2] * 100 for i in range(len(errors))]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--undistorted-results", type=Path, nargs=2, required=True)
    parser.add_argument("--raw-results", type=Path, nargs=2, required=True)
    parser.add_argument("--timings", type=float, nargs=5, required=True, help="Five end-to-end CLI durations in seconds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with new_output_directory(args.output) as staging:
            report = {"test_segmentation": test_metrics(args.dataset, args.checkpoint),
                      "measurement_undistorted": error_summary([json.loads(p.read_text(encoding="utf-8")) for p in args.undistorted_results]),
                      "measurement_raw": error_summary([json.loads(p.read_text(encoding="utf-8")) for p in args.raw_results]),
                      "timing_seconds": {"runs": args.timings, "median": statistics.median(args.timings), "spread_min_max": [min(args.timings), max(args.timings)]},
                      "limitations": ["Only two measurement instances are available; this does not meet the required 10+ manual checks.", "Test masks are draft masks, not independently hand-labelled ground truth."]}
            report["raw_vs_undistorted"] = {"mae_difference_mm_raw_minus_undistorted": report["measurement_raw"]["mae_mm"] - report["measurement_undistorted"]["mae_mm"],
                                               "note": "Comparison uses the same two images and only assesses these scenes."}
            (staging / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__": main()
