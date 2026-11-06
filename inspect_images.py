"""Inventory calibration images and create a numbered contact sheet."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from output_utils import new_output_directory

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_paths(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path: Path) -> np.ndarray:
    """Use encoded pixel orientation consistently; do not auto-rotate via EXIF."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def inspect_images(directory: Path, output: Path) -> None:
    with new_output_directory(output, [directory]) as staging:
        _inspect_images(directory, staging)


def _inspect_images(directory: Path, output: Path) -> None:
    paths = image_paths(directory)
    if not paths:
        raise SystemExit("No calibration images found.")
    tiles, inventory = [], []
    for index, path in enumerate(paths, start=1):
        tile = np.full((260, 320, 3), 245, np.uint8)
        try:
            image = read_image(path)
            h, w = image.shape[:2]
            scale = min(320 / w, 230 / h)
            preview = cv2.resize(image, (round(w * scale), round(h * scale)))
            ph, pw = preview.shape[:2]
            tile[:ph, :pw] = preview
            record = {"index": index, "file": path.name, "width": w, "height": h}
            label = f"{index:02d} | {w} x {h}"
        except ValueError as exc:
            record = {"index": index, "file": path.name, "error": str(exc)}
            label = f"{index:02d} | UNREADABLE"
        cv2.putText(tile, label, (8, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        inventory.append(record)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 245))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    write_image(output / "contact_sheet.jpg", sheet)
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    sizes = Counter((r.get("width"), r.get("height")) for r in inventory)
    print(f"Images: {len(paths)}; sizes (width, height): {dict(sizes)}")
    print(f"Contact sheet: {output / 'contact_sheet.jpg'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "Callibration_Img/Callibrate")
    parser.add_argument("--output", type=Path, required=True, help="A new inspection directory")
    args = parser.parse_args()
    try:
        inspect_images(args.images, args.output)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Inspection failed: {exc}\n")


if __name__ == "__main__":
    main()
