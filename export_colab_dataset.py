"""Create a 256x256 letterboxed Colab dataset from the fixed split and final draft masks."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from inspect_images import ROOT, write_image
from output_utils import new_output_directory


def letterbox(image, interpolation):
    """Resize within 256x256 without changing aspect ratio; pad with black."""
    height, width = image.shape[:2]
    scale = min(256 / width, 256 / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=interpolation)
    top, left = (256 - resized.shape[0]) // 2, (256 - resized.shape[1]) // 2
    canvas = np.zeros((256, 256) if image.ndim == 2 else (256, 256, image.shape[2]), dtype=image.dtype)
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas, {"source_size": [width, height], "scale": scale, "padding_ltrb": [left, top, 256 - left - resized.shape[1], 256 - top - resized.shape[0]]}


def run(manifest, images, base_masks, refined_masks, output):
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 54, "validation": 15, "test": 8}:
        raise ValueError(f"Expected fixed 54/15/8 split, found {dict(counts)}.")
    exported = []
    for row in rows:
        stem = Path(row["filename"]).stem
        source = images / f"{stem}.png"
        base_mask = base_masks / f"{stem}.png"
        refined_mask = refined_masks / f"{stem}.png"
        selected_mask = refined_mask if refined_mask.is_file() else base_mask
        if not source.is_file() or not selected_mask.is_file():
            raise ValueError(f"Missing generated image or mask for {row['filename']}.")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(selected_mask), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Invalid image/mask pair for {row['filename']}.")
        small_image, transform = letterbox(image, cv2.INTER_AREA)
        small_mask, _ = letterbox(mask, cv2.INTER_NEAREST)
        filename = f"{stem}.png"
        write_image(output / row["split"] / "images" / filename, small_image)
        write_image(output / row["split"] / "masks" / filename, small_mask)
        exported.append({"filename": filename, "split": row["split"], "mask_source": "refined" if selected_mask == refined_mask else "base", **transform})
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader(); writer.writerows(exported)
    summary = {"images": len(exported), "split_counts": dict(counts), "image_size": [256, 256],
               "resize": "aspect-ratio-preserving letterbox", "refined_masks_used": sum(r["mask_source"] == "refined" for r in exported),
               "test_policy": "test files are exported but are not used by train_unet_colab.py"}
    (output / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/split_manifest.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "annotations/undistorted")
    parser.add_argument("--base-masks", type=Path, default=ROOT / "annotations/masks")
    parser.add_argument("--refined-masks", type=Path, default=ROOT / "annotations_refined_v2/masks")
    parser.add_argument("--output", type=Path, default=ROOT / "colab_dataset")
    args = parser.parse_args()
    try:
        with new_output_directory(args.output, [args.manifest.parent, args.images, args.base_masks, args.refined_masks]) as staging:
            run(args.manifest, args.images, args.base_masks, args.refined_masks, staging)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Colab export failed: {exc}\n")


if __name__ == "__main__":
    main()
