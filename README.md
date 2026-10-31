# Frozen-feature baseline for ovarian carcinoma subtype classification on UBC-OCEAN

Code accompanying the manuscript *"A Frozen-Feature Baseline for Ovarian Cancer Histological Subtype
Classification on UBC-OCEAN: Patch-Level Performance, Slide-Level Aggregation and the Effect of Data Leakage"*
(under review).

The repository contains the preprocessing, feature-extraction, slide-grouped evaluation and slide-level
classification code. **No data are included.** The UBC-OCEAN data are available from the Kaggle competition
"UBC Ovarian Cancer Subtype Classification and Outlier Detection" (https://www.kaggle.com/competitions/UBC-OCEAN).

## Pipeline (run in this order)

| Step | Script | What it does |
|---|---|---|
| 1 | `tile_and_label_ubc_ocean.py` | Cuts the competition's slide *thumbnails* into non-overlapping 224x224 tiles, discards tiles with more than 85% near-white/near-black pixels, keeps up to 50 random tiles per slide, and writes `patches/<LABEL>/<image_id>_tile_NNNN.png`. Arguments: `--thumbnails_dir`, `--labels_csv` (`train.csv`), `--output_dir`, `--tile_size`, `--max_tiles_per_image`, `--seed`. |
| 2 | `01_preprocessing_agent.py` | Min-max normalisation, 3x3 Gaussian filter, CLAHE (clip 2.0, 8x8 tiles, LAB lightness), resize to 224x224 and random augmentation (horizontal flip, rotation up to 15 degrees, centre-crop zoom up to 10%, brightness jitter 15%). |
| 3 | `02_feature_extraction_agent.py` | Frozen ImageNet EfficientNetB0 (no top, global average pooling) -> 1,280-d embedding per patch -> `features.npz` (`X`, `y`, `label_names`, `patch_paths`). |
| 4 | `grouped_rerun.py` | Slide-grouped evaluation of the patch-level pipeline (PCA, SMOTE, LightGBM), the ablation variants and slide-level aggregation of patch probabilities. Also contains `diagnose`, which reproduces the original random patch-level split to measure leakage. |
| 5 | `slide_level.py` | Slide-level classifiers on mean- or mean+max-pooled embeddings, repeated stratified 5-fold CV over slides. |

Helper scripts: `verify_pipeline.py` (checks the edge-artefact filter counts, the TMA/whole-slide composition and
that an environment reproduces `features.npz`) and `check_preproc.py` (identifies which preprocessing options
produced a set of stored `.npy` arrays by replaying the first manifest rows).

## Example commands

Tiling (thumbnails of the whole-slide images; `train.csv` from the Kaggle data):

    python tile_and_label_ubc_ocean.py --thumbnails_dir train_thumbnails --labels_csv train.csv --output_dir patches --tile_size 224 --max_tiles_per_image 50

Preprocessing (histopathology tiles organised by label folder; augmentation on, as in the manuscript):

    python 01_preprocessing_agent.py --input_dir patches --output_dir preprocessed --modality histopathology --augment

Feature extraction:

    python 02_feature_extraction_agent.py --preprocessed_dir preprocessed --output_file features.npz --batch_size 32

Leak diagnostic and slide-grouped evaluation (five splits, seeds 0-4; about 20 minutes per seed on a laptop CPU):

    python grouped_rerun.py diagnose --npz features.npz
    python grouped_rerun.py rerun --npz features.npz --out results_5seeds --seeds 0 1 2 3 4

Slide-level classification (run after the previous step; both use all CPU cores):

    python slide_level.py --npz features.npz --out results_slide --seeds 0 1 2 3 4

## Notes

* The slide (`image_id`) is the unit of independence. UBC-OCEAN provides no patient identifiers.
* Only whole-slide images are used (the 25 TMA cores of the 538 labelled training images are not tiled).
* `grouped_rerun.py diagnose` reproduces the original development split (`train_test_split(test_size=0.2,
  stratify=y, random_state=42)`) and reports how many test patches come from slides that also contribute training
  patches. All other analyses use slide-grouped splits (`StratifiedGroupKFold`), with PCA and SMOTE fitted on
  training data only.
* Augmentation in step 2 is applied to every patch before any train/test partition (see the manuscript's limitations).

## Environment

Python 3.13.7, NumPy 2.2.6, scikit-learn 1.7.2, LightGBM 4.6.0, imbalanced-learn 0.14.0 for the analysis scripts;
TensorFlow 2.21.0 for feature extraction (see `requirements.txt`).

## Citation

Citation details will be added after publication.

## Disclaimer

This repository is intended for research and educational use. It is not a clinically validated diagnostic system and must not be used to make clinical decisions.
