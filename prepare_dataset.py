"""Audit source photos and save a deterministic split; never label or train."""

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image

from inspect_images import ROOT, IMAGE_EXTENSIONS, read_image, write_image
from undistort import load_parameters
from output_utils import new_output_directory


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bits_to_int(bits):
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hashes(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    ac = cv2.dct(small)[:8, :8].ravel()[1:]
    phash = bits_to_int(ac > np.median(ac))  # 63 AC bits; DC excluded.
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = bits_to_int(small[:, 1:] > small[:, :-1])
    return phash, dhash


def ssim(left, right):
    """Mean local grayscale SSIM on 128 x 128 audit thumbnails, range approximately [-1, 1]."""
    left, right = left.astype(np.float32), right.astype(np.float32)
    mu1 = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(right, (11, 11), 1.5)
    var1 = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu1 * mu1
    var2 = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu2 * mu2
    cov = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu1 * mu2
    score = ((2 * mu1 * mu2 + 6.5025) * (2 * cov + 58.5225)) / (
        (mu1 * mu1 + mu2 * mu2 + 6.5025) * (var1 + var2 + 58.5225)
    )
    return float(np.mean(score[5:-5, 5:-5]))


def capture_time(name):
    match = re.fullmatch(r"IMG_(\d{8})_(\d{6})_(\d{3})\.[^.]+", name)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S%f")


def preview_tile(image, label, width=320, height=280):
    tile = np.full((height, width, 3), 245, np.uint8)
    scale = min(width / image.shape[1], (height - 40) / image.shape[0])
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y, x = (height - 40 - small.shape[0]) // 2, (width - small.shape[1]) // 2
    tile[y:y + small.shape[0], x:x + small.shape[1]] = small
    for index, line in enumerate(label.split("\n")):
        cv2.putText(tile, line, (6, height - 24 + index * 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (15, 15, 15), 1, cv2.LINE_AA)
    return tile


def contact_pages(records, previews, output):
    for start in range(0, len(records), 20):
        tiles = [preview_tile(previews[i], f"{i + 1:02d}: {records[i]['filename']}\n"
                              f"{records[i]['width']} x {records[i]['height']}")
                 for i in range(start, min(start + 20, len(records)))]
        while len(tiles) % 5:
            tiles.append(np.full_like(tiles[0], 245))
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        write_image(output / f"contact_sheet_{start // 20 + 1:02d}.jpg", sheet)


def read_metadata(path):
    """Keep usable pixels even if EXIF is absent or damaged; record the problem."""
    metadata = {"exif_orientation": "unknown", "camera_make": "unknown", "camera_model": "unknown"}
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            with Image.open(path) as image:
                exif = image.getexif()
                metadata.update(
                    exif_orientation=exif.get(274, "absent"),
                    camera_make=str(exif.get(271, "absent")).strip("\x00 "),
                    camera_model=str(exif.get(272, "absent")).strip("\x00 "),
                )
        problems = [str(notice.message) for notice in notices]
        if metadata["exif_orientation"] not in ("absent", 1, 2, 3, 4, 5, 6, 7, 8):
            problems.append(f"Invalid EXIF orientation: {metadata['exif_orientation']}")
        metadata["metadata_status"] = "warning" if problems else "ok"
        metadata["metadata_error"] = "; ".join(problems)
    except (OSError, ValueError, SyntaxError, TypeError) as exc:
        metadata.update(metadata_status="error", metadata_error=f"{type(exc).__name__}: {exc}")
    return metadata


def inspect_sources(source, output):
    records, invalid, previews, features = [], [], [], []
    files = sorted((p for p in source.rglob("*") if p.is_file()), key=lambda p: p.relative_to(source).as_posix())
    unsupported = [p.relative_to(source).as_posix() for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]
    for path in files:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(source).as_posix()
        try:
            image = read_image(path)
        except ValueError as exc:
            invalid.append({"filename": relative, "reason": str(exc)})
            continue
        h, w = image.shape[:2]
        record = {
            "filename": relative, "source_relative_path": relative,
            "width": w, "height": h, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "decoded_sha256": hashlib.sha256(f"{image.shape}|".encode() + image.tobytes()).hexdigest(),
            "exif_orientation": "unknown", "orientation": "landscape" if w >= h else "portrait",
        }
        record.update(read_metadata(path))
        if record["metadata_status"] != "ok":
            print(f"Metadata {record['metadata_status']}: {relative}: {record['metadata_error']}", flush=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        rotations = [np.ascontiguousarray(np.rot90(thumb, k)) for k in range(4)]
        features.append({"rotations": rotations, "hashes": [hashes(g) for g in rotations]})
        previews.append(cv2.resize(image, (round(w * 400 / max(w, h)), round(h * 400 / max(w, h)))))
        records.append(record)
        print(f"Inspected {len(records):02d}: {relative} ({w} x {h})", flush=True)
    contact_pages(records, previews, output)
    return records, invalid, unsupported, previews, features


def audit_pairs(records, features):
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            p1, d1 = features[i]["hashes"][0]
            candidates = []
            for k, (p2, d2) in enumerate(features[j]["hashes"]):
                candidates.append(((p1 ^ p2).bit_count(), (d1 ^ d2).bit_count(), k))
            # Compare all quarter-turns for hashes, and SSIM for the closest hash orientation.
            pd, dd, rotation = min(candidates, key=lambda v: (v[0] + v[1], v[0], v[2]))
            similarity = ssim(features[i]["rotations"][0], features[j]["rotations"][rotation])
            byte_exact = a["sha256"] == b["sha256"]
            pixel_exact = a["decoded_sha256"] == b["decoded_sha256"]
            visual_candidate = pd <= 12 or (dd <= 10 and similarity >= 0.65) or similarity >= 0.92
            ta, tb = capture_time(Path(a["filename"]).name), capture_time(Path(b["filename"]).name)
            time_gap = abs((ta - tb).total_seconds()) if ta is not None and tb is not None else None
            burst_candidate = time_gap is not None and time_gap <= 2.0
            reasons = []
            if byte_exact:
                reasons.append("identical_file")
            if pixel_exact:
                reasons.append("identical_decoded_pixels")
            if visual_candidate and not pixel_exact:
                reasons.append("visual_near_duplicate_candidate")
            if burst_candidate:
                reasons.append("capture_gap_at_most_2_seconds")
            pairs.append({
                "index_a": i, "index_b": j, "filename_a": a["filename"], "filename_b": b["filename"],
                "exact_file": byte_exact, "exact_pixels": pixel_exact,
                "phash_distance_63bit": pd, "dhash_distance_64bit": dd,
                "thumbnail_ssim": similarity, "rotation_b_ccw_degrees": rotation * 90,
                "capture_gap_seconds": time_gap, "visual_candidate": visual_candidate,
                "burst_candidate": burst_candidate, "flagged": bool(reasons), "reasons": ";".join(reasons),
            })
    return pairs


def connected_groups(count, pairs, exact_only=False):
    parents = list(range(count))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for pair in pairs:
        selected = pair["exact_file"] or pair["exact_pixels"] if exact_only else pair["flagged"]
        if selected:
            parents[root(pair["index_b"])] = root(pair["index_a"])
    groups = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def assign_groups(groups, train, validation, test, seed):
    """Meet exact counts, with whole groups and related views in both evaluation sets."""
    if min(train, validation, test) < 0 or sum(map(len, groups)) != train + validation + test:
        raise ValueError("Split counts must be nonnegative and sum to the number of images.")
    ordered = list(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    require_related = any(len(group) > 1 for group in groups)
    states = {(0, 0, False, False): []}
    assigned = 0
    for group in ordered:
        size = len(group)
        next_states = {}
        split_order = ["train", "validation", "test"]
        rng.shuffle(split_order)
        for (v, t, related_val, related_test), choices in states.items():
            for split in split_order:
                nv, nt = v + (size if split == "validation" else 0), t + (size if split == "test" else 0)
                if nv <= validation and nt <= test and assigned + size - nv - nt <= train:
                    key = (nv, nt, related_val or (split == "validation" and size > 1),
                           related_test or (split == "test" and size > 1))
                    next_states.setdefault(key, choices + [split])
        states = next_states
        assigned += size
    goal = (validation, test, require_related and validation > 0, require_related and test > 0)
    if goal not in states:
        return None
    return {i: (split, f"group_{n + 1:03d}")
            for n, (group, split) in enumerate(zip(ordered, states[goal])) for i in group}


def prepare_dataset(args, output):
    source = args.images.resolve()
    params = args.params.resolve()
    matrix, distortion, calibration_size = load_parameters(params)
    if calibration_size != (3968, 2976):
        raise ValueError("Expected the original-resolution calibration (3968 x 2976).")
    cv2.setNumThreads(1)
    records, invalid, unsupported, previews, features = inspect_sources(source, output)
    (output / "inventory.json").write_text(json.dumps({"valid": records, "invalid": invalid,
        "unsupported": unsupported}, indent=2), encoding="utf-8")
    if len(records) != args.train + args.validation + args.test:
        raise ValueError(f"{len(records)} valid images do not match requested split counts.")
    pairs = audit_pairs(records, features)
    groups = connected_groups(len(records), pairs)
    assignments = assign_groups(groups, args.train, args.validation, args.test, args.seed)
    grouping = "All exact, visual-candidate, and <=2-second capture groups kept together."
    if assignments is None:
        raise ValueError("No whole-group split fits the counts with a related-image group in each evaluation set.")
    calibration_hash = hashlib.sha256(params.read_bytes()).hexdigest()
    for i, row in enumerate(records):
        row["split"], row["group_id"] = assignments[i]
        row["seed"] = args.seed
        row["calibration_sha256"] = calibration_hash
        row["undistortion_status"] = ("pending_same_resolution" if (row["width"], row["height"]) == calibration_size
                                      else "pending_orientation_mapping_review")
        if hashlib.sha256((source / row["filename"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source changed during audit: {row['filename']}")
    for pair in pairs:
        pair["split_a"] = records[pair["index_a"]]["split"]
        pair["split_b"] = records[pair["index_b"]]["split"]
        pair["cross_split"] = pair["split_a"] != pair["split_b"]
        pair["possible_leakage"] = pair["flagged"] and pair["cross_split"]
    manifest_fields = ["filename", "source_relative_path", "split", "group_id", "seed", "width", "height",
                       "sha256", "decoded_sha256", "exif_orientation", "metadata_status", "metadata_error",
                       "undistortion_status", "calibration_sha256"]
    save_csv(output / "split_manifest.csv", records, manifest_fields)
    pair_fields = [k for k in pairs[0] if k not in ("index_a", "index_b")] if pairs else [
        "filename_a", "filename_b", "exact_file", "exact_pixels", "phash_distance_63bit",
        "dhash_distance_64bit", "thumbnail_ssim", "rotation_b_ccw_degrees", "capture_gap_seconds",
        "visual_candidate", "burst_candidate", "flagged", "reasons", "split_a", "split_b",
        "cross_split", "possible_leakage",
    ]
    save_csv(output / "all_pair_similarities.csv", pairs, pair_fields)
    flagged = [p for p in pairs if p["flagged"]]
    leaks = [p for p in pairs if p["possible_leakage"]]
    save_csv(output / "duplicate_candidates.csv", flagged, pair_fields)
    save_csv(output / "possible_leakage.csv", leaks, pair_fields)
    review_pairs = sorted(flagged, key=lambda p: (not p["possible_leakage"], p["phash_distance_63bit"]))
    for start in range(0, len(review_pairs), 8):
        rows = []
        for pair in review_pairs[start:start + 8]:
            a, b = pair["index_a"], pair["index_b"]
            tiles = [preview_tile(previews[k], f"{k + 1:02d}: {records[k]['filename']}\n{records[k]['split']}", 400, 270)
                     for k in (a, b)]
            row = np.hstack(tiles)
            row = cv2.copyMakeBorder(row, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
            label = f"pHash={pair['phash_distance_63bit']} SSIM={pair['thumbnail_ssim']:.3f} | {pair['reasons']}"
            cv2.putText(row, label, (5, row.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
            rows.append(row)
        write_image(output / "pair_review" / f"pairs_{start // 8 + 1:02d}.jpg", np.vstack(rows))
    summary = {
        "source": str(source), "valid_images": len(records), "invalid_images": invalid,
        "unsupported_files": unsupported, "resolutions": dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
        "split_counts": dict(Counter(r["split"] for r in records)), "seed": args.seed,
        "grouping": grouping, "group_sizes": sorted([len(g) for g in groups], reverse=True),
        "exact_file_duplicate_pairs": sum(p["exact_file"] for p in pairs),
        "exact_decoded_duplicate_pairs": sum(p["exact_pixels"] for p in pairs),
        "visual_near_duplicate_candidate_pairs": sum(p["visual_candidate"] and not p["exact_pixels"] for p in pairs),
        "burst_candidate_pairs": sum(p["burst_candidate"] for p in pairs),
        "total_flagged_pairs": len(flagged), "possible_cross_split_leakage_pairs": len(leaks),
        "pairs_compared": len(pairs), "original_hashes_unchanged": True,
        "similarity_rules": {"phash_63bit_max": 12, "dhash_64bit_max_with_ssim": 10,
                             "dhash_min_ssim": 0.65, "ssim_alone_min": 0.92, "burst_gap_seconds_max": 2.0,
                             "quarter_turns_checked": True, "thumbnail_size": [128, 128]},
        "limitations": ["Visual candidates are heuristic flags, not automatic duplicate confirmations.",
                        "Different views of the same sponge and capture setting remain correlated across splits.",
                        "No session-independent or unseen-object generalization claim follows from this split."],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__},
        "metadata_status_counts": dict(Counter(r["metadata_status"] for r in records)),
        "exif_orientation_counts": dict(Counter(str(r["exif_orientation"]) for r in records)),
        "camera_models": dict(Counter(str(r.get("camera_model", "unknown")) for r in records)),
    }
    config = {
        "schema_version": 2, "manifest_path_base": "source_directory",
        "source_directory": str(source), "manifest": str(args.output.resolve() / "split_manifest.csv"),
        "calibration_file": str(params), "calibration_sha256": calibration_hash,
        "calibration_image_size": list(calibration_size), "camera_matrix": matrix.tolist(),
        "dist_coeffs": distortion.ravel().tolist(), "encoded_pixel_orientation": True,
        "future_undistortion": {"script": str(ROOT / "undistort.py"), "alpha": 1.0, "crop": False,
                                "portrait_policy": "Resolve rotation direction/camera coordinates before applying calibration; do not use --allow-resize to force portrait inputs.",
                                "output_directory": str(args.output.resolve() / "undistorted"),
                                "annotation_coordinate_space": "undistorted pixels; preserve original split and record the output camera matrix"},
        "green_top_surface_mm": {"length": 93, "width": 74, "source": "user-measured"},
        "images_undistorted": False, "annotations_created": False, "training_started": False,
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "Object")
    parser.add_argument("--output", type=Path, required=True, help="A new run directory; existing evidence is never replaced")
    parser.add_argument("--params", type=Path, default=ROOT / "calibration/camera_params.npz")
    parser.add_argument("--train", type=int, default=54)
    parser.add_argument("--validation", type=int, default=15)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if min(args.train, args.validation, args.test) < 0 or args.train + args.validation + args.test < 2:
            raise ValueError("Provide nonnegative counts for at least two images.")
        with new_output_directory(args.output, [args.images]) as staging:
            prepare_dataset(args, staging)
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        parser.exit(1, f"Dataset preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
