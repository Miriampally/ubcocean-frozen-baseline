"""
02_feature_extraction_agent.py

Stage 2 (Deep Feature Extraction) for:
"A Novel Hybrid Framework for Accurate and Balanced Ovarian Cancer
Severity Level Detection from Medical Imaging" -- RAC-3 proposal, Section 6.2, Eq. (1).

Reads the preprocessed .npy patches produced by 01_preprocessing_agent.py
(organized as preprocessed_dir/<LABEL>/<patch>.npy, e.g. preprocessed/HGSC/xxx.npy),
runs each through ImageNet-pretrained EfficientNetB0 with the top removed and
global-average-pooling on the output, and writes a single features.npz containing:

    X : float32 array, shape (n_samples, 1280)   -- one embedding per patch
    y : int array, shape (n_samples,)             -- integer label id
    label_names : array of str                    -- label_names[y[i]] = original class name
    patch_paths : array of str                    -- source .npy path for traceability

This features.npz is the direct input to 03_train_evaluate_agent.py (Stages 3-6:
PCA -> SMOTE/class-weighting -> LightGBM -> evaluation & explanation).

Setup
-----
    pip install tensorflow numpy --break-system-packages

(CPU-only tensorflow is fine; GPU tensorflow will just run faster. On Windows,
`pip install tensorflow` installs a CPU+GPU build automatically if you have a
compatible NVIDIA GPU + CUDA/cuDNN set up -- otherwise it runs on CPU, which is
fine for feature extraction, it's just slower than training would be.)

Usage
-----
    python 02_feature_extraction_agent.py ^
        --preprocessed_dir "D:\\2026\\Alliance university\\Data Set\\UBC Cancer subtype\\preprocessed" ^
        --output_file "D:\\2026\\Alliance university\\Data Set\\UBC Cancer subtype\\features.npz" ^
        --batch_size 32

Notes
-----
- Patch-level features, not slide-level: UBC-OCEAN labels are per whole-slide-image
  (image_id), and tile_and_label_ubc_ocean.py assigns every tile from a slide the
  slide's label. That's fine for training/testing at the *patch* level (what this
  script and 03_train_evaluate_agent.py do), but if you later want slide-level
  accuracy, aggregate per-patch predictions back to image_id (majority vote or mean
  probability) -- patch_paths in the output .npz retain the image_id prefix
  (e.g. "44962_tile_0003") so that aggregation is a simple groupby.
- This uses ImageNet-pretrained weights with no fine-tuning (matches proposal §6.2:
  "ImageNet-pretrained weights are used to start the network"). Optional fine-tuning
  of the top conv blocks (also mentioned in §6.2, lr=1e-5) is NOT implemented here --
  add it as a follow-up once the frozen-feature baseline is validated end-to-end.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_extraction_agent")


class EfficientNetB0FeatureExtractor:
    """
    Proposal §6.2, Eq. (1): F = f_EfficientNetB0(X) in R^1280.

    Loads ImageNet-pretrained EfficientNetB0 with include_top=False and
    pooling='avg', so a forward pass on a (224,224,3) image yields a single
    1280-D vector directly (the global-average-pooled output of the last
    conv block) -- no manual pooling needed.
    """

    def __init__(self, batch_size: int = 32):
        # Imported lazily so the rest of this module (and 03_..._agent.py) can be
        # imported/tested without requiring tensorflow to be installed.
        import tensorflow as tf
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input

        self.tf = tf
        self.preprocess_input = preprocess_input
        self.batch_size = batch_size

        log.info("Loading ImageNet-pretrained EfficientNetB0 (include_top=False, pooling=avg)...")
        self.model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        log.info("Model loaded. Output dim: %d", self.model.output_shape[-1])

    def extract_batch(self, images_0_1: np.ndarray) -> np.ndarray:
        """
        images_0_1: float32 array (N, 224, 224, 3) in [0, 1], as written by
        01_preprocessing_agent.py. EfficientNet's preprocess_input expects
        [0, 255]-range RGB, so we rescale before handing off.
        """
        images_0_255 = images_0_1 * 255.0
        batch = self.preprocess_input(images_0_255)
        return self.model.predict(batch, batch_size=self.batch_size, verbose=0)


def discover_patches(preprocessed_dir: Path) -> tuple[list[Path], list[str]]:
    """Walk preprocessed_dir/<LABEL>/*.npy and return (paths, label_names)."""
    label_dirs = sorted([d for d in preprocessed_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        raise FileNotFoundError(
            f"No label subfolders found under {preprocessed_dir}. Expected "
            f"{preprocessed_dir}/<LABEL>/<patch>.npy (the output layout of "
            "01_preprocessing_agent.py with preserve_subfolders=True)."
        )

    paths, labels = [], []
    for d in label_dirs:
        npy_files = sorted(d.glob("*.npy"))
        if not npy_files:
            log.warning("No .npy files found in label folder %s -- skipping", d)
            continue
        log.info("  %-8s : %d patches", d.name, len(npy_files))
        for p in npy_files:
            paths.append(p)
            labels.append(d.name)
    return paths, labels


def run(preprocessed_dir: str, output_file: str, batch_size: int = 32,
        extraction_batch: int = 64):
    preprocessed_dir = Path(preprocessed_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("Scanning %s for label-organized patches...", preprocessed_dir)
    paths, label_names_raw = discover_patches(preprocessed_dir)
    n = len(paths)
    if n == 0:
        raise RuntimeError("No patches found -- nothing to extract features from.")
    log.info("Found %d total patches across %d classes.", n, len(set(label_names_raw)))

    classes = sorted(set(label_names_raw))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in label_names_raw], dtype=np.int64)

    extractor = EfficientNetB0FeatureExtractor(batch_size=batch_size)

    X = np.zeros((n, 1280), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, extraction_batch):
        end = min(start + extraction_batch, n)
        batch_imgs = np.stack([np.load(p) for p in paths[start:end]], axis=0).astype(np.float32)
        X[start:end] = extractor.extract_batch(batch_imgs)
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        log.info("  extracted %d/%d patches (%.1f patches/sec)", end, n, rate)

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        label_names=np.array(classes),
        patch_paths=np.array([str(p) for p in paths]),
    )
    log.info("Done. Wrote %s  (X: %s, y: %s, classes: %s)",
              output_file, X.shape, y.shape, classes)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 2: extract EfficientNetB0 features from preprocessed patches."
    )
    p.add_argument("--preprocessed_dir", required=True,
                    help="Output dir from 01_preprocessing_agent.py "
                         "(must contain <LABEL>/<patch>.npy subfolders)")
    p.add_argument("--output_file", required=True, help="Path to write features.npz")
    p.add_argument("--batch_size", type=int, default=32, help="EfficientNetB0 inference batch size")
    p.add_argument("--extraction_batch", type=int, default=64,
                    help="How many .npy files to load from disk into memory per chunk")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args.preprocessed_dir, args.output_file, args.batch_size, args.extraction_batch)
