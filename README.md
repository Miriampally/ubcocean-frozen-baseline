# Frozen-Feature Baseline for Ovarian Carcinoma Subtype Classification on UBC-OCEAN

## Overview

This repository contains starter code and helper scripts for a frozen-feature baseline workflow for ovarian carcinoma subtype classification using the UBC-OCEAN histopathology dataset.

The repository currently includes preprocessing checks, grouped evaluation utilities, slide-level evaluation code, and pipeline verification utilities.

## Repository Contents

| File | Purpose |
|---|---|
| `check_preproc.py` | Checks preprocessing outputs and stored arrays. |
| `grouped_rerun.py` | Runs grouped evaluation and related diagnostic experiments. |
| `slide_level.py` | Performs slide-level classification and evaluation. |
| `verify_pipeline.py` | Checks pipeline artefacts and feature-file reproducibility. |
| `requirements.txt` | Lists Python dependencies. |
| `.gitignore` | Intended to exclude unnecessary or sensitive files; add this file if not yet uploaded. |

## Dataset

The project is intended for research using the UBC-OCEAN ovarian cancer histopathology dataset.

Users must obtain the dataset through its authorised source and comply with its applicable terms and conditions. Dataset files are not included in this repository.

## Reproducibility

Before running experiments:

1. Review and install the dependencies listed in `requirements.txt`.
2. Inspect each script's command-line arguments and input-file requirements.
3. Configure the dataset and output paths for your environment.
4. Verify that patient or slide groups remain separated during evaluation to prevent data leakage.
5. Record the random seeds, software versions, data splits, and evaluation metrics.

The complete workflow must be verified against the available scripts and required input artefacts before claiming end-to-end reproducibility.

## Current Status

The repository contains the initial set of helper scripts and documentation. Additional pipeline components may be required to reproduce the complete workflow described in the original project.

## Disclaimer

This repository is intended for research and educational use. It is not a clinically validated diagnostic system and must not be used to make clinical decisions.
