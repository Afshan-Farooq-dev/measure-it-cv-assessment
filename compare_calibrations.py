"""Compare saved calibration reports without loading photos or changing evidence."""

import argparse
import json
import math
from pathlib import Path

from inspect_images import ROOT
from output_utils import new_output_directory


def compare_reports(previous, current):
    old_w, old_h = previous["image_size"]
    new_w, new_h = current["image_size"]
    if min(old_w, old_h, new_w, new_h) <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = new_w / old_w
    if not math.isclose(scale, new_h / old_h):
        raise ValueError("Cannot normalize RMS across different aspect ratios.")
    old_rms, new_rms = previous["opencv_rms_px"], current["opencv_rms_px"]
    if not all(math.isfinite(v) and v >= 0 for v in (old_rms, new_rms)):
        raise ValueError("RMS values must be finite and nonnegative.")
    normalized = new_rms / scale
    return {
        "previous_image_size": previous["image_size"],
        "current_image_size": current["image_size"],
        "linear_resolution_scale": scale,
        "previous_rms_px": old_rms,
        "current_rms_px": new_rms,
        "current_rms_in_previous_resolution_pixels": normalized,
        "resolution_normalized_change_percent": 100 * (normalized / old_rms - 1) if old_rms else None,
        "note": "Pixel-unit normalization is not independent measurement-accuracy validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=ROOT / "calibration_previous_whatsapp/calibration_report.json")
    parser.add_argument("--current", type=Path, default=ROOT / "calibration/calibration_report.json")
    parser.add_argument("--output", type=Path, help="Optional new directory; default prints only")
    args = parser.parse_args()
    try:
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
        current = json.loads(args.current.read_text(encoding="utf-8"))
        result = compare_reports(previous, current)
        text = json.dumps(result, indent=2)
        if args.output:
            with new_output_directory(args.output) as staging:
                (staging / "comparison.json").write_text(text + "\n", encoding="utf-8")
        print(text)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
