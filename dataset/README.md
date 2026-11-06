# Sponge dataset preparation

Prepared from `E:\Computer_Vision\Object`. There are **77 valid images**, not 78.
The user approved **54 train / 15 validation / 8 test** for the available images.
No originals were modified, no labels were created, no undistortion batch was run, and
no model training was started. File SHA-256 hashes were checked unchanged after processing.

## Inventory and split

| Item | Result |
| --- | ---: |
| Valid, decodable images | 77 |
| Invalid images | 0 |
| Unsupported files | 0 |
| Landscape, 3968 x 2976 | 40 |
| Portrait, 2976 x 3968 | 37 |
| Train | 54 (70.13%) |
| Validation | 15 (19.48%) |
| Test | 8 (10.39%) |
| Random seed | 42 |

`split_manifest.csv` is the saved split authority. It lists each source filename exactly
once, its split, related-image group, resolution, hashes, and pending undistortion status.
No image copies in train/validation/test folders are necessary; future processing should
read this manifest. `inventory.json` records metadata for every image. The four numbered
contact sheets cover all 77 images in filename order and were visually inspected.

Reproduce from the project directory:

```powershell
python E:\Computer_Vision\prepare_dataset.py --train 54 --validation 15 --test 8 --seed 42
```

The script sorts relative paths, computes duplicate/capture groups, shuffles the groups
with Python's `random.Random(42)`, stably orders them by size, and assigns whole groups
using exact-capacity dynamic programming. This is a seeded group-constrained split,
not an independent shuffle of individual images. It preserves the requested counts while
avoiding identified related-image pairs crossing splits. The script stops if the valid
image count changes; do not silently add new images to the saved split. It regenerates
the manifest and reports when rerun, so version/freeze the manifest before later training.

## Duplicate and leakage audit

All **2,926 image pairs** were checked using file and decoded-pixel SHA-256, 63-bit DCT
perceptual hashes, 64-bit difference hashes, and thumbnail SSIM. Quarter-turn variants
were compared to accommodate both orientations. EXIF orientation is not automatically
applied. The audit uses grayscale 128 x 128 thumbnails only for similarity checks; these
thumbnails are not model inputs and do not modify the originals.

| Check | Flagged pairs |
| --- | ---: |
| Identical file bytes | 0 |
| Identical decoded pixels | 0 |
| Visual near-duplicate candidates | 63 |
| Capture times at most 2 seconds apart | 20 |
| Distinct candidate pairs across both criteria | 75 |
| Candidate pairs crossing splits | 0 |

Eight pairs satisfy both visual and capture-time criteria. Candidate links form 37
connected groups, including singletons; the largest group contains 23 images. Every
flagged pair stays in the same split. These are conservative similarity flags, not 63
confirmed duplicate photographs. Background, pose, and framing can produce false positives.
Examples of especially similar views include `IMG_20261106_225312_733.jpg` with
`IMG_20261106_225316_247.jpg`, and `IMG_20261106_230337_954.jpg` with
`IMG_20261106_230338_421.jpg`; both pairs are in training.

The exact thresholds and software versions are in `audit_report.json`. A pair is flagged
for visual review if pHash distance <= 12, or dHash distance <= 10 with SSIM >= 0.65, or
SSIM >= 0.92. Capture timestamps are parsed from filenames, not assumed to be EXIF times.
SSIM is evaluated at the rotation with the lowest combined hash distance.

Files:

- `duplicate_candidates.csv`: all 75 candidate pairs, reasons, scores, and split assignments.
- `possible_leakage.csv`: header only, because none of those candidates cross splits.
- `all_pair_similarities.csv`: all 2,926 pair comparisons for review.
- `pair_review/`: side-by-side candidate previews; previews do not constitute annotations.

Zero flagged cross-split pairs is not proof that all leakage has been eliminated. These
photos show the same sponge, repeated backgrounds, and one capture session, so different
views across splits remain correlated. This split does not measure performance on an
unseen sponge or independent capture session. No images were removed automatically.

## Calibration for subsequent undistortion

`dataset_config.json` pins `E:\Computer_Vision\calibration\camera_params.npz` by SHA-256
and records its 3968 x 2976 camera matrix and distortion coefficients. This is the new
original-resolution calibration (2.957893 px RMS), not the archived WhatsApp calibration.
Future processing must verify that hash before using it.

The 40 landscape images match its encoded dimensions. The 37 portrait images require
the rotation from sensor/calibration coordinates to their encoded pixels to be established
before applying that matrix. EXIF orientation is 0 in 45 files and 1 in 32 files; it does
not supply an unambiguous portrait rotation direction. Do not force portrait images through
`--allow-resize`, stretch them, or merely swap focal lengths and assume calibration is valid.
Resolve the capture orientation using reliable camera metadata/workflow or a matching
portrait checkerboard validation, then apply the corresponding image/coordinate transform.

Later, save undistorted derivatives under `dataset/undistorted/`, keeping source names and
split assignments. Use the existing `undistort.py`, retain the output camera matrix and
any orientation transform, and annotate that final coordinate space. The planned policy
is `alpha=1` without cropping. No derivative sponge images have been generated yet.

## Next annotation steps

1. Review the candidate pairs and image quality at full resolution. The contact sheets
   show strong shadows, reflective backgrounds, distant views, and some blur. In particular,
   review `IMG_20261106_230539_201.jpg` for blur and `IMG_20261106_225240_985.jpg` /
   `IMG_20261106_225247_740.jpg` for top-face visibility near the frame boundary. Valid decoding
   does not mean every image is suitable for precise corner measurement.
2. Resolve portrait orientation and generate undistorted copies using the pinned calibration.
   Inspect those outputs before labeling; maintain the saved split manifest.
3. Manually trace the **green top surface** with a polygon. Exclude yellow side faces,
   background, fingers, and cast shadows. Shadows on the top surface still belong to that
   surface. Use a consistent class name such as `sponge_top`.
4. If the measurement stage needs corners, manually record the four top-face corners in a
   consistent clockwise order where visible. Mark uncertainty, occlusion, or truncation
   instead of guessing hidden boundaries. Store polygons/corners in undistorted-image pixels.
5. Record **93 mm x 74 mm** as user-measured ground truth for the green top only. Calibration
   alone does not provide metric scale for arbitrary object poses. Define the later
   plane/pose/scale method separately; do not use these same dimensions to force a result
   and then report that result as independent measurement accuracy.
6. Review annotations against image overlays, preserve the train/validation/test assignments,
   and apply any later augmentation to training images only. Reserve test images for final
   evaluation, without using their performance to tune the model.

These are instructions for subsequent manual work; no annotation files or training runs
were created during dataset preparation.
