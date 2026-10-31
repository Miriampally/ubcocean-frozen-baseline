"""
tile_and_label_ubc_ocean.py

Turns UBC-OCEAN whole-slide thumbnails into labeled 224x224 patches,
ready to feed into multimodal_preprocessing_agent.py (Stage 1) and then
Stage 2 (EfficientNetB0 feature extraction).

What it does
------------
1. Reads train.csv (image_id, label, image_width, image_height, is_tma)
2. For each thumbnail (e.g. 4_thumbnail.png), tiles it into a grid of
   224x224 patches, skipping mostly-background (white/black) tiles
3. Writes each patch out to disk, organized by label:
       output_dir/HGSC/4_tile_0003.png
       output_dir/EC/66_tile_0012.png
       ...
4. Writes a manifest.csv mapping every patch -> its source image_id and label

Why organize by label subfolder?
---------------------------------
The multimodal_preprocessing_agent.py script auto-detects "modality" from
folder names. Sorting these into per-CLASS folders (not modality folders)
means when you're ready for Stage 4/5 (SMOTE + LightGBM), you can point
straight at output_dir and read off the label from each patch's parent
folder -- no separate label file needed downstream, though manifest.csv
is also written for convenience / auditability.

Usage
-----
    python tile_and_label_ubc_ocean.py \\
        --thumbnails_dir "D:\\2026\\Alliance university\\UBC-OCEAN\\train_thumbnails" \\
        --labels_csv "D:\\2026\\Alliance university\\UBC-OCEAN\\train.csv" \\
        --output_dir "D:\\2026\\Alliance university\\UBC-OCEAN\\patches" \\
        --tile_size 224 \\
        --max_tiles_per_image 50

Dependencies
------------
    pip install pillow numpy --break-system-packages
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # thumbnails can still be large; disable PIL's decompression-bomb guard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tile_ubc_ocean")


def load_labels(labels_csv: Path) -> dict[str, str]:
    """Return {image_id: label} from train.csv."""
    mapping = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["image_id"].strip()] = row["label"].strip()
    log.info("Loaded %d labels from %s", len(mapping), labels_csv)
    return mapping


def find_thumbnail_files(thumbnails_dir: Path) -> dict[str, Path]:
    """Map image_id -> file path, based on filenames like '4_thumbnail.png'."""
    mapping = {}
    for p in thumbnails_dir.glob("*_thumbnail.png"):
        image_id = p.stem.replace("_thumbnail", "")
        mapping[image_id] = p
    log.info("Found %d thumbnail files in %s", len(mapping), thumbnails_dir)
    return mapping


def is_mostly_background(tile: np.ndarray, white_thresh: int = 220, black_thresh: int = 20,
                          max_background_fraction: float = 0.85) -> bool:
    """
    Reject tiles that are mostly white slide-background or mostly black
    (outside the scanned region) -- these carry no diagnostic signal and
    would just dilute training with noise.
    """
    gray = tile.mean(axis=-1) if tile.ndim == 3 else tile
    white_frac = (gray > white_thresh).mean()
    black_frac = (gray < black_thresh).mean()
    return (white_frac + black_frac) > max_background_fraction


def tile_image(
    img_path: Path,
    tile_size: int,
    max_tiles: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Slice a thumbnail into non-overlapping tile_size x tile_size tiles, filtering background."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    tiles = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = arr[y:y + tile_size, x:x + tile_size]
            if not is_mostly_background(tile):
                tiles.append(tile)

    if not tiles:
        # fall back: image may be smaller than tile_size, or entirely background.
        # Resize whole image to tile_size as a last resort rather than dropping it.
        resized = np.array(img.resize((tile_size, tile_size), Image.LANCZOS))
        return [resized]

    random.Random(seed).shuffle(tiles)
    return tiles[:max_tiles]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tile UBC-OCEAN thumbnails into labeled 224x224 patches.")
    p.add_argument("--thumbnails_dir", required=True, help="Folder containing *_thumbnail.png files")
    p.add_argument("--labels_csv", required=True, help="Path to train.csv")
    p.add_argument("--output_dir", required=True, help="Where to write patches, organized by label")
    p.add_argument("--tile_size", type=int, default=224)
    p.add_argument("--max_tiles_per_image", type=int, default=50,
                    help="Cap tiles per source image to keep classes from being dominated by a few huge slides")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()

    thumbnails_dir = Path(args.thumbnails_dir)
    labels_csv = Path(args.labels_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_csv)
    thumbnails = find_thumbnail_files(thumbnails_dir)

    matched_ids = set(labels) & set(thumbnails)
    missing_labels = set(thumbnails) - set(labels)
    missing_images = set(labels) - set(thumbnails)
    if missing_labels:
        log.warning("%d thumbnails have no matching label in train.csv (skipped)", len(missing_labels))
    if missing_images:
        log.warning("%d labeled image_ids have no thumbnail file on disk (skipped)", len(missing_images))
    log.info("Processing %d matched image/label pairs", len(matched_ids))

    manifest_rows = []
    total_patches = 0

    for image_id in sorted(matched_ids, key=lambda x: int(x) if x.isdigit() else x):
        label = labels[image_id]
        img_path = thumbnails[image_id]

        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        try:
            tiles = tile_image(img_path, args.tile_size, args.max_tiles_per_image, args.seed)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to tile %s (image_id=%s): %s", img_path, image_id, exc)
            continue

        for i, tile in enumerate(tiles):
            out_name = f"{image_id}_tile_{i:04d}.png"
            out_path = label_dir / out_name
            Image.fromarray(tile).save(out_path)
            manifest_rows.append({
                "patch_path": str(out_path),
                "source_image_id": image_id,
                "label": label,
            })
            total_patches += 1

        log.info("image_id=%s (%s): wrote %d patches", image_id, label, len(tiles))

    manifest_path = output_dir / "patch_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_path", "source_image_id", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    log.info("Done. %d total patches written across %d source images.", total_patches, len(matched_ids))
    log.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
