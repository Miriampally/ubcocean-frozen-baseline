"""
multimodal_preprocessing_agent.py

Stage 1 (Data Preprocessing) agent for:
"A Novel Hybrid Framework for Accurate and Balanced Ovarian Cancer
Severity Level Detection from Medical Imaging"

Implements the preprocessing pipeline described in Section 6.1 of the proposal:
  1. Modality-aware loading      (CT / MRI DICOM, Ultrasound, Histopathology)
  2. Normalisation               (min-max or z-score)
  3. Noise removal               (Gaussian / median filtering)
  4. Contrast enhancement        (CLAHE)
  5. Data augmentation           (flip ±15°, zoom ±10%, brightness jitter) - train split only
  6. Resizing to 224x224         (EfficientNetB0 input requirement)

Output: preprocessed image tensors ready to feed into Stage 2
(EfficientNetB0 deep feature extraction), plus a manifest CSV recording
what was done to every file for reproducibility/auditability.

Design goals
------------
- Single entry point class `MultimodalPreprocessingAgent` so it can be dropped
  straight into the rest of the pipeline (Stage 2 onward) as an object.
- Modality is auto-detected from file extension/folder, but can be overridden.
- Fails loudly per-file (logs + skips) rather than crashing a whole batch run,
  since clinical datasets are messy (corrupt DICOM headers, mixed bit depths, etc).
- No hidden state: every run produces a manifest so preprocessing decisions are
  auditable, which matters for the clinical trust / interpretability angle
  in Sections 2.3 and 6.6 of the proposal.

Dependencies
------------
    pip install opencv-python-headless numpy pillow pydicom --break-system-packages

pydicom is optional -- only required if you actually feed in CT/MRI .dcm files.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_agent")

VALID_MODALITIES = {"ct", "mri", "ultrasound", "histopathology"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ModalityDetectionError(Exception):
    """Raised when modality can't be confidently detected and lenient mode is off."""


@dataclass
class PreprocessConfig:
    """All knobs for Stage 1, matching Section 6.1 of the proposal."""
    target_size: tuple[int, int] = (224, 224)          # EfficientNetB0 input
    normalization: str = "minmax"                       # "minmax" or "zscore"
    denoise_method: str = "gaussian"                     # "gaussian", "median", or "none"
    gaussian_ksize: int = 3
    median_ksize: int = 3
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    augment: bool = False                                # only True for training split
    flip_prob: float = 0.5
    max_rotation_deg: float = 15.0
    zoom_range: tuple[float, float] = (0.9, 1.1)
    brightness_jitter: float = 0.15                      # +/- fraction
    seed: int = 42


@dataclass
class ProcessedSample:
    source_path: str
    modality: str
    original_shape: tuple[int, ...]
    final_shape: tuple[int, ...]
    augmented: bool
    array: np.ndarray = field(repr=False)


class MultimodalPreprocessingAgent:
    """
    Stage 1 preprocessing agent. Auto-detects modality, applies the
    normalisation -> denoise -> CLAHE -> resize (-> augment) pipeline,
    and returns/saves preprocessed arrays for handoff to EfficientNetB0.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.cfg = config or PreprocessConfig()
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_grid,
        )

    # ------------------------------------------------------------------ #
    # Modality detection & loading
    # ------------------------------------------------------------------ #
    def detect_modality(self, path: Path, hint: Optional[str] = None,
                         lenient: bool = False) -> str:
        """
        Best-effort modality detection: explicit hint > parent folder name > extension.

        IMPORTANT: if neither a hint nor a confident signal is found, this raises
        ModalityDetectionError instead of silently guessing. A dataset organized by
        class label (e.g. .../patches/EC/xxx.png, as tile_and_label_ubc_ocean.py
        produces) will NOT contain a modality name in the path, so guessing here is
        actively harmful -- e.g. guessing "ultrasound" for histopathology tiles forces
        grayscale conversion in load_image() and silently destroys the H&E stain color
        channels that carry diagnostic signal. Pass --modality explicitly for
        label-organized datasets. Set lenient=True to restore the old
        default-to-ultrasound behaviour (not recommended).
        """
        if hint and hint.lower() in VALID_MODALITIES:
            return hint.lower()

        parent_name = path.parent.name.lower()
        for m in VALID_MODALITIES:
            if m in parent_name:
                return m

        if path.suffix.lower() in DICOM_EXTENSIONS:
            # DICOM is CT or MRI; default to CT unless folder says otherwise
            return "ct"
        if "histo" in path.stem.lower() or "slide" in path.stem.lower():
            return "histopathology"
        if "us" in path.stem.lower() or "ultrasound" in path.stem.lower():
            return "ultrasound"

        if lenient:
            log.warning("Could not confidently detect modality for %s; defaulting to 'ultrasound'", path)
            return "ultrasound"

        raise ModalityDetectionError(
            f"Could not confidently detect modality for {path} "
            f"(parent folder: '{path.parent.name}'). "
            "This usually means the dataset is organized by class label, not modality "
            "(e.g. patches/EC/, patches/HGSC/...). Pass --modality explicitly "
            f"(choices: {sorted(VALID_MODALITIES)}), or rerun with --lenient to fall "
            "back to the old (unsafe) auto-guess behaviour."
        )

    def load_image(self, path: Path, modality: str) -> np.ndarray:
        """Load a file into a float32 single/multi-channel numpy array, grayscale where relevant."""
        if path.suffix.lower() in DICOM_EXTENSIONS:
            if not _HAS_PYDICOM:
                raise RuntimeError(
                    f"{path} is a DICOM file but pydicom is not installed. "
                    "Run: pip install pydicom --break-system-packages"
                )
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            # Apply rescale slope/intercept if present (standard for CT Hounsfield units)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            if arr.ndim == 3:  # multi-frame DICOM: take middle slice
                arr = arr[arr.shape[0] // 2]
            return arr

        # Standard image formats (US, histopathology, exported MRI/CT slices)
        img = Image.open(path)
        if modality == "histopathology":
            img = img.convert("RGB")   # histology stains carry diagnostic color info
        else:
            img = img.convert("L")     # US/CT/MRI: grayscale is standard
        return np.array(img).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Pipeline stages (each independently testable / callable)
    # ------------------------------------------------------------------ #
    def normalize(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.normalization == "minmax":
            lo, hi = img.min(), img.max()
            if hi - lo < 1e-8:
                return np.zeros_like(img, dtype=np.float32)
            return ((img - lo) / (hi - lo)).astype(np.float32)
        elif self.cfg.normalization == "zscore":
            mu, sigma = img.mean(), img.std()
            sigma = sigma if sigma > 1e-8 else 1.0
            z = (img - mu) / sigma
            # rescale z-scored data into [0,1] for downstream visualisation/consistency
            z = (z - z.min()) / (z.max() - z.min() + 1e-8)
            return z.astype(np.float32)
        raise ValueError(f"Unknown normalization method: {self.cfg.normalization}")

    def denoise(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.denoise_method == "none":
            return img
        img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if self.cfg.denoise_method == "gaussian":
            k = self.cfg.gaussian_ksize | 1  # ensure odd
            out = cv2.GaussianBlur(img_u8, (k, k), 0)
        elif self.cfg.denoise_method == "median":
            k = self.cfg.median_ksize | 1
            out = cv2.medianBlur(img_u8, k)
        else:
            raise ValueError(f"Unknown denoise method: {self.cfg.denoise_method}")
        return (out.astype(np.float32) / 255.0)

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement (Section 6.1) — grayscale or per-channel for RGB."""
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img_u8.ndim == 2:
            enhanced = self._clahe.apply(img_u8)
        else:
            # apply CLAHE on the L channel in LAB space to preserve color fidelity (histopathology)
            lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0

    def resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.cfg.target_size, interpolation=cv2.INTER_AREA)

    def augment(self, img: np.ndarray) -> np.ndarray:
        """Random flip, rotation-as-proxy-for-zoom-crop, and brightness jitter. Train split only."""
        out = img.copy()

        if random.random() < self.cfg.flip_prob:
            out = np.fliplr(out)

        angle = random.uniform(-self.cfg.max_rotation_deg, self.cfg.max_rotation_deg)
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        zoom = random.uniform(*self.cfg.zoom_range)
        zh, zw = int(h / zoom), int(w / zoom)
        y0 = max((h - zh) // 2, 0)
        x0 = max((w - zw) // 2, 0)
        cropped = out[y0:y0 + zh, x0:x0 + zw]
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

        jitter = 1.0 + random.uniform(-self.cfg.brightness_jitter, self.cfg.brightness_jitter)
        out = np.clip(out * jitter, 0, 1)

        return out.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def process_one(self, path: Path, modality_hint: Optional[str] = None,
                     lenient: bool = False) -> ProcessedSample:
        modality = self.detect_modality(path, modality_hint, lenient=lenient)
        raw = self.load_image(path, modality)
        original_shape = raw.shape

        img = self.normalize(raw)
        img = self.denoise(img)
        img = self.enhance_contrast(img)
        img = self.resize(img)

        augmented = False
        if self.cfg.augment:
            img = self.augment(img)
            augmented = True

        # EfficientNetB0 expects 3 channels; replicate grayscale if needed
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        return ProcessedSample(
            source_path=str(path),
            modality=modality,
            original_shape=original_shape,
            final_shape=img.shape,
            augmented=augmented,
            array=img,
        )

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        modality_hint: Optional[str] = None,
        save_npy: bool = True,
        save_preview_png: bool = True,
        lenient: bool = False,
        preserve_subfolders: bool = True,
        overwrite: bool = False,
    ) -> list[ProcessedSample]:
        """
        preserve_subfolders=True (default): mirrors each file's path relative to
        input_dir into output_dir, so a label-organized input
        (patches/EC/xxx.png, patches/HGSC/yyy.png, ...) produces a matching
        label-organized output (preprocessed/EC/xxx.npy, preprocessed/HGSC/yyy.npy, ...).
        This matters because Stage 2 (feature extraction) reads the label straight
        off the folder name -- flattening everything into one output folder would
        silently discard the label association for every patch.

        overwrite=False (default): skips any file whose output .npy already exists.
        This makes the whole stage safely resumable -- stop with Ctrl+C at any point
        and rerun the exact same command; already-done files are skipped instead of
        reprocessed, so you only pay for the files that weren't finished yet.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = [
            p for p in in_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS | DICOM_EXTENSIONS
        ]
        if not files:
            log.warning("No supported image files found under %s", in_path)
            return []

        log.info("Found %d candidate files under %s", len(files), in_path)

        results: list[ProcessedSample] = []
        manifest_rows = []
        skipped_errors = 0
        skipped_done = 0

        for i, f in enumerate(files, 1):
            rel_dir = f.parent.relative_to(in_path) if preserve_subfolders else Path(".")
            dest_dir = out_path / rel_dir
            dest_npy = dest_dir / f"{f.stem}.npy"

            if not overwrite and dest_npy.exists():
                skipped_done += 1
                continue

            try:
                sample = self.process_one(f, modality_hint, lenient=lenient)
            except ModalityDetectionError as exc:
                # Fail loudly and stop the whole batch: this is a configuration
                # problem (missing --modality), not a per-file data problem, and
                # will be wrong for every remaining file too.
                log.error(str(exc))
                raise
            except Exception as exc:  # noqa: BLE001 - per-file data problems: skip and continue
                log.error("Skipping %s: %s", f, exc)
                skipped_errors += 1
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)

            if save_npy:
                np.save(dest_npy, sample.array)
            if save_preview_png:
                preview = (np.clip(sample.array, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(preview).save(dest_dir / f"{f.stem}_preview.png")

            manifest_rows.append({
                "source_path": sample.source_path,
                "output_path": str(dest_npy),
                "label": rel_dir.parts[0] if preserve_subfolders and rel_dir.parts else "",
                "modality": sample.modality,
                "original_shape": sample.original_shape,
                "final_shape": sample.final_shape,
                "augmented": sample.augmented,
            })
            results.append(sample)

            if i % 200 == 0 or i == len(files):
                log.info("Progress: %d/%d files handled (%d newly processed, %d already done, %d errors)",
                          i, len(files), len(results), skipped_done, skipped_errors)

        if skipped_done:
            log.info("Skipped %d file(s) already processed (resume mode). "
                      "Use --overwrite to force reprocessing.", skipped_done)
        if skipped_errors:
            log.warning("%d file(s) skipped due to per-file errors (see log above).", skipped_errors)

        manifest_path = out_path / "preprocessing_manifest.csv"
        if manifest_rows:
            file_exists = manifest_path.exists()
            mode = "a" if file_exists else "w"
            with open(manifest_path, mode, newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=manifest_rows[0].keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerows(manifest_rows)

        log.info("Processed %d new file(s) this run (%d already done, %d errors). Manifest: %s",
                  len(results), skipped_done, skipped_errors, manifest_path)
        return results


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1 preprocessing agent for the ovarian cancer hybrid framework."
    )
    p.add_argument("--input_dir", required=True, help="Directory of raw images (CT/MRI/US/histopathology)")
    p.add_argument("--output_dir", required=True, help="Where to write preprocessed .npy + previews")
    p.add_argument("--modality", choices=sorted(VALID_MODALITIES), default=None,
                    help="Force modality instead of auto-detecting")
    p.add_argument("--normalization", choices=["minmax", "zscore"], default="minmax")
    p.add_argument("--denoise", choices=["gaussian", "median", "none"], default="gaussian")
    p.add_argument("--augment", action="store_true", help="Enable augmentation (use for TRAIN split only)")
    p.add_argument("--size", type=int, default=224, help="Output square size (default 224 for EfficientNetB0)")
    p.add_argument("--lenient", action="store_true",
                    help="Fall back to guessing 'ultrasound' when modality can't be detected, "
                         "instead of raising an error. Not recommended for label-organized datasets.")
    p.add_argument("--no_preview", action="store_true",
                    help="Skip writing _preview.png files (only .npy is needed by Stage 2). "
                         "Roughly halves disk writes -- recommended on HDDs for large runs. "
                         "Spot-check a small subset WITH previews first, then rerun the full "
                         "batch with --no_preview once you've confirmed the output looks right.")
    p.add_argument("--overwrite", action="store_true",
                    help="Reprocess files even if their output .npy already exists. "
                         "Without this flag, already-done files are skipped -- so stopping "
                         "with Ctrl+C and rerunning the same command safely resumes.")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = PreprocessConfig(
        target_size=(args.size, args.size),
        normalization=args.normalization,
        denoise_method=args.denoise,
        augment=args.augment,
    )
    agent = MultimodalPreprocessingAgent(cfg)
    agent.process_directory(args.input_dir, args.output_dir, modality_hint=args.modality,
                             lenient=args.lenient, save_preview_png=not args.no_preview,
                             overwrite=args.overwrite)


if __name__ == "__main__":
    main()
